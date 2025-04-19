import torch
import torch.nn as nn
import unittest
from einops import rearrange, repeat
from transformers.cache_utils import DynamicCache
from typing import Any, Callable, Dict, List, Optional, Tuple
import triton.language as tl
import triton

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from netfit import NonLinearOp
from neurons import CIFNeuron, IFNeuron, BurstNeuron

torch.set_printoptions(precision=6)

def detailed_allclose_check(ref: torch.Tensor, test: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6):
    # 检查形状是否一致
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: ref.shape = {ref.shape}, test.shape = {test.shape}")
    
    # 检查NaN，并将双方都是NaN的情况认为是相等
    both_nan_mask = torch.isnan(ref) & torch.isnan(test)
    
    # 替换NaN为零以避免对allclose的影响
    ref_no_nan = torch.where(both_nan_mask, torch.zeros_like(ref), ref)
    test_no_nan = torch.where(both_nan_mask, torch.zeros_like(test), test)
    
    # 使用allclose检查
    close_mask = torch.isclose(ref_no_nan, test_no_nan, atol=atol, rtol=rtol)
    close_mask |= both_nan_mask  # NaN对齐视为相等

    # 如果所有位置都符合预期，返回成功信息
    if close_mask.all():
        #print("All values are within the tolerance.")
        return True
    
    # 找到第一个不符合的索引
    mismatched_indices = torch.nonzero(~close_mask, as_tuple=True)
    first_mismatch_idx = tuple(idx[0].item() for idx in mismatched_indices)
    
    # 输出详细错误信息
    ref_val = ref[first_mismatch_idx].item()
    test_val = test[first_mismatch_idx].item()
    raise AssertionError(
        f"Mismatch found at index {first_mismatch_idx}:\n"
        f"  Reference value: {ref_val}\n"
        f"  Test value: {test_val}\n"
        f"  Allowed tolerance: atol={atol}, rtol={rtol}\n"
        f"  Difference: {abs(ref_val - test_val)}, {abs(ref_val - test_val) / abs(ref_val)}\n"
        f"  Reference tensor contains NaN: {torch.isnan(ref).any().item()}\n"
        f"  Test tensor contains NaN: {torch.isnan(test).any().item()}"
    )

class SpikeQuantizer(nn.Module):
    '''
    do quantization on data shape of [B, L, d]
    input shape = [(T*B), L, d]
    '''
    def __init__(self, T):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)
        self.if_neuron = IFNeuron()
        self.fc2 = nn.Linear(2, 1)

        self.fc1.weight.data[0, 0] = 1.
        self.fc1.weight.data[1, 0] = -1.
        self.fc2.weight.data[0, 0] = 1.
        self.fc2.weight.data[0, 1] = -1.

        self.fc1.bias.data[0] = 0.
        self.fc1.bias.data[1] = 0.
        self.fc2.bias.data[0] = 0.

        self.T = T
    
    @classmethod
    def from_pretrained(cls, quantizer, T):
        instance = cls(T)
        instance.if_neuron.threshold = quantizer.scale * quantizer.qmax
        instance.if_neuron.threshold.unsqueeze_(-1) # [c, 1, 1]
        instance.group_size = quantizer.group_size
        return instance
    
    def forward(self, x):
        x = rearrange(x, '(T B) L (c g) -> T B L c g', T=self.T, g=self.group_size)
        x = x.unsqueeze(-1)
        x = self.fc1(x)
        out = torch.zeros_like(x)
        self.if_neuron.reset()
        # self.if_neuron.init_max_spike(self.T)
        for i in range(self.T):
            out[i] = self.if_neuron(x[i])
        out = self.fc2(out)
        out = out * self.if_neuron.threshold.to(out)
        out = out.squeeze(-1)
        out = rearrange(out, 'T B L c g -> (T B) L (c g)', T=self.T, g=self.group_size)
        # print(self.if_neuron.threshold)
        # print(out)
        # print(rearrange(out, '(T B) ... -> T B ...', T=self.T).mean(0))
        return out


def spike_matmul(X: torch.Tensor, Y: torch.Tensor, T: int):
    '''
    X.shape = ((T, *), d1, d2), typically = ((T, B, H), L, d)
    Y.shape = ((T, *), d2, d3), typically = ((T, B, H), d, L)
    output expected to be (1/T * sum(X_t)) @ (1/T * sum(Y_t))
    '''
    assert X.shape[-1] == Y.shape[-2], 'dim of X and Y must match for matmul.'
    _, d1, d2 = X.shape
    _, d2, d3 = Y.shape
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    Xsum = torch.zeros_like(Xs[0])
    Ysum = torch.zeros_like(Ys[0])
    out = torch.zeros(*Xs.shape[:-2], d1, d3).to(Xs)
    Phi = torch.zeros_like(out[0]).float()
    correction = torch.zeros_like(out[0])
    for t in range(T):
        Xsum += Xs[t]
        phi_t = torch.bmm(Xsum, Ys[t]) + torch.bmm(Xs[t], Ysum)
        Ysum += Ys[t]
        Phi += phi_t.float()
        # t_psi_t = (t+1) * Phi / (t+1)**2
        t_psi_t = Phi / (t+1)
        output = t_psi_t - correction
        correction = t_psi_t
        out[t] = output
    return out.flatten(0, 1)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=[]
)
@triton.jit
def spike_matmul_kernel(
    X_ptr, Y_ptr, Xsum_ptr, Ysum_ptr, out_ptr,
    T: tl.constexpr, BD1: tl.constexpr, BD2: tl.constexpr, ND2: tl.constexpr, BD3: tl.constexpr,
    d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr,
    B: tl.constexpr,  # 新增批量维度参数
):
    '''
    X_ptr, Xsum_ptr: [T, B, d1, d2]
    Y_ptr, Ysum_ptr: [T, B, d3, d2]
    out_ptr: [T, B, d1, d3]

    parallel grid: [B, ND1, ND3]
    '''
    b_id = tl.program_id(0)
    d1_block = tl.program_id(1)
    d3_block = tl.program_id(2)

    # 后续代码与原实现相同...
    d1_offsets = d1_block * BD1 + tl.arange(0, BD1)
    d3_offsets = d3_block * BD3 + tl.arange(0, BD3)

    x_offsets = d1_offsets[:, None] * d2 + tl.arange(0, BD2)
    y_offsets = d3_offsets[:, None] * d2 + tl.arange(0, BD2)
    out_offsets = d1_offsets[:, None] * d3 + d3_offsets


    x_ptr_t = X_ptr + b_id * d1 * d2
    y_ptr_t = Y_ptr + b_id * d3 * d2
    xsum_ptr_t = Xsum_ptr + b_id * d1 * d2
    ysum_ptr_t = Ysum_ptr + b_id * d3 * d2
    out_ptr_t = out_ptr + b_id * d1 * d3

    Phi = tl.zeros((BD1, BD3), dtype=tl.float32)
    correction = tl.zeros((BD1, BD3), dtype=tl.float32)
    for t in range(T):

        k_offset = 0
        for j in range(ND2):
            x_mask = (d1_offsets[:, None] < d1) & (k_offset + tl.arange(0, BD2) < d2)
            y_mask = (d3_offsets[:, None] < d3) & (k_offset + tl.arange(0, BD2) < d2)
            x = tl.load(x_ptr_t + x_offsets + k_offset, mask=x_mask)
            y = tl.load(y_ptr_t + y_offsets + k_offset, mask=y_mask)
            xsum = tl.load(xsum_ptr_t + x_offsets + k_offset, mask=x_mask)
            ysum = tl.load(ysum_ptr_t + y_offsets + k_offset, mask=y_mask)

            Phi = tl.dot(xsum, y.trans(), Phi) 
            Phi = tl.dot(x, ysum.trans(), Phi)

            k_offset += BD2

        t_psi_t = Phi / (t + 1.0)
        output = t_psi_t - correction
        correction = t_psi_t

        out_mask = (d1_offsets[:, None] < d1) & (d3_offsets[None, :] < d3)
        tl.store(out_ptr_t + out_offsets, output, mask=out_mask)

        x_ptr_t += B * d1 * d2
        y_ptr_t += B * d3 * d2
        xsum_ptr_t += B * d1 * d2
        ysum_ptr_t += B * d3 * d2
        out_ptr_t += B * d1 * d3

def spike_matmul_triton(X: torch.Tensor, Y: torch.Tensor, T: int):
    assert X.shape[-1] == Y.shape[-2], 'dim of X and Y must match for matmul.'
    _, d1, d2 = X.shape
    _, d2, d3 = Y.shape
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    
    # Precompute cumulative sums with correct shifts
    Xsum = torch.cumsum(Xs, dim=0)
    Ysum = torch.cumsum(Ys, dim=0)
    Ysum = torch.roll(Ysum, shifts=1, dims=0)
    Ysum[0] = 0.0

    # Reshape for kernel
    Xshape = Xs.shape

    Xs = Xs.view(T, -1, d1, d2).contiguous()
    Ys = Ys.view(T, -1, d2, d3).transpose(-2, -1).contiguous()

    Xsum = Xsum.view(T, -1, d1, d2).contiguous()
    Ysum = Ysum.view(T, -1, d2, d3).transpose(-2, -1).contiguous()

    # Create output tensor
    B = Xs.shape[1]
    out = torch.empty((T, B, d1, d3), device=X.device, dtype=X.dtype)

    # Configure kernel parameters
    BD1, BD2, BD3 = 16, 16, 16
    ND2 = triton.cdiv(d2, BD2)
    grid = (B, triton.cdiv(d1, BD1), triton.cdiv(d3, BD3))
    
    # 启动参数调整
    spike_matmul_kernel[grid](
        Xs, Ys, Xsum, Ysum, out,
        T, BD1, BD2, ND2, BD3,
        d1, d2, d3,
        B,  # 新增批量维度参数
    )

    out = out.view(T, *Xshape[1:-2], d1, d3)
    return out.flatten(0, 1)

def spike_matmul_mean(X: torch.Tensor, Y: torch.Tensor, T: int):
    '''
    X.shape = ((T, *), d1, d2), typically = ((T, B, H), L, d)
    Y.shape = ((T, *), d2, d3), typically = ((T, B, H), d, L)
    output expected to be (1/T * sum(X_t)) @ (1/T * sum(Y_t))
    '''
    assert X.shape[-1] == Y.shape[-2], 'dim of X and Y must match for matmul.'
    _, d1, d2 = X.shape
    _, d2, d3 = Y.shape
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    Xsum = torch.zeros_like(Xs[0])
    Ysum = torch.zeros_like(Ys[0])
    out = torch.zeros(*Xs.shape[:-2], d1, d3).to(Xs)
    Phi = torch.zeros_like(out[0]).float()
    for t in range(T):
        Xsum += Xs[t]
        phi_t = torch.bmm(Xsum, Ys[t]) + torch.bmm(Xs[t], Ysum)
        Ysum += Ys[t]
        Phi += phi_t.float()
        out[t] = Phi / (t+1)**2
    return out.flatten(0, 1)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=[]
)
@triton.jit
def spike_matmul_mean_kernel(
    X_ptr, Y_ptr, Xsum_ptr, Ysum_ptr, out_ptr,
    T: tl.constexpr, BD1: tl.constexpr, BD2: tl.constexpr, ND2: tl.constexpr, BD3: tl.constexpr,
    d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr,
    B: tl.constexpr,  # 新增批量维度参数
):
    '''
    X_ptr, Xsum_ptr: [T, B, d1, d2]
    Y_ptr, Ysum_ptr: [T, B, d3, d2]
    out_ptr: [T, B, d1, d3]

    parallel grid: [B, ND1, ND3]
    '''
    b_id = tl.program_id(0)
    d1_block = tl.program_id(1)
    d3_block = tl.program_id(2)

    # 后续代码与原实现相同...
    d1_offsets = d1_block * BD1 + tl.arange(0, BD1)
    d3_offsets = d3_block * BD3 + tl.arange(0, BD3)

    x_offsets = d1_offsets[:, None] * d2 + tl.arange(0, BD2)
    y_offsets = d3_offsets[:, None] * d2 + tl.arange(0, BD2)
    out_offsets = d1_offsets[:, None] * d3 + d3_offsets


    x_ptr_t = X_ptr + b_id * d1 * d2
    y_ptr_t = Y_ptr + b_id * d3 * d2
    xsum_ptr_t = Xsum_ptr + b_id * d1 * d2
    ysum_ptr_t = Ysum_ptr + b_id * d3 * d2
    out_ptr_t = out_ptr + b_id * d1 * d3

    Phi = tl.zeros((BD1, BD3), dtype=tl.float32)
    for t in range(T):

        k_offset = 0
        for j in range(ND2):
            x_mask = (d1_offsets[:, None] < d1) & (k_offset + tl.arange(0, BD2) < d2)
            y_mask = (d3_offsets[:, None] < d3) & (k_offset + tl.arange(0, BD2) < d2)
            x = tl.load(x_ptr_t + x_offsets + k_offset, mask=x_mask)
            y = tl.load(y_ptr_t + y_offsets + k_offset, mask=y_mask)
            xsum = tl.load(xsum_ptr_t + x_offsets + k_offset, mask=x_mask)
            ysum = tl.load(ysum_ptr_t + y_offsets + k_offset, mask=y_mask)

            Phi = tl.dot(xsum, y.trans(), Phi) 
            Phi = tl.dot(x, ysum.trans(), Phi)

            k_offset += BD2

        output = Phi / ((t + 1.0) * (t + 1.0))

        out_mask = (d1_offsets[:, None] < d1) & (d3_offsets[None, :] < d3)
        tl.store(out_ptr_t + out_offsets, output, mask=out_mask)

        x_ptr_t += B * d1 * d2
        y_ptr_t += B * d3 * d2
        xsum_ptr_t += B * d1 * d2
        ysum_ptr_t += B * d3 * d2
        out_ptr_t += B * d1 * d3

def spike_matmul_mean_triton(X: torch.Tensor, Y: torch.Tensor, T: int):
    assert X.shape[-1] == Y.shape[-2], 'dim of X and Y must match for matmul.'
    _, d1, d2 = X.shape
    _, d2, d3 = Y.shape
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    
    # Precompute cumulative sums with correct shifts
    Xsum = torch.cumsum(Xs, dim=0)
    Ysum = torch.cumsum(Ys, dim=0)
    Ysum = torch.roll(Ysum, shifts=1, dims=0)
    Ysum[0] = 0.0

    # Reshape for kernel
    Xshape = Xs.shape

    Xs = Xs.view(T, -1, d1, d2).contiguous()
    Ys = Ys.view(T, -1, d2, d3).transpose(-2, -1).contiguous()

    Xsum = Xsum.view(T, -1, d1, d2).contiguous()
    Ysum = Ysum.view(T, -1, d2, d3).transpose(-2, -1).contiguous()

    # Create output tensor
    B = Xs.shape[1]
    out = torch.empty((T, B, d1, d3), device=X.device, dtype=X.dtype)

    # Configure kernel parameters
    BD1, BD2, BD3 = 16, 16, 16
    ND2 = triton.cdiv(d2, BD2)
    grid = (B, triton.cdiv(d1, BD1), triton.cdiv(d3, BD3))
    
    # 启动参数调整
    spike_matmul_mean_kernel[grid](
        Xs, Ys, Xsum, Ysum, out,
        T, BD1, BD2, ND2, BD3,
        d1, d2, d3,
        B,  # 新增批量维度参数
    )

    out = out.view(T, *Xshape[1:-2], d1, d3)
    return out.flatten(0, 1)


def spike_elementwise_dot(X: torch.Tensor, Y: torch.Tensor, T: int):
    '''
    X.shape = ((T, *), d1, d2), typically = ((T, B, H), L, d)
    Y.shape = ((T, *), d1, d2), typically = ((T, B, H), L, d)
    output expected to be (1/T * sum(X_t)) * (1/T * sum(Y_t))
    '''
    assert X.shape[-1] == Y.shape[-1] and X.shape[-2] == Y.shape[-2], \
           'dim of X and Y should be the same for elementwise.'
    d1 = X.shape[-2]
    d2 = X.shape[-1]
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    Xsum = torch.zeros_like(Xs[0])
    Ysum = torch.zeros_like(Ys[0])
    out = torch.zeros_like(Xs)
    Phi = torch.zeros_like(out[0])
    correction = torch.zeros_like(out[0])
    for t in range(T):
        Xsum += Xs[t]
        phi_t = Xsum * Ys[t] + Xs[t] * Ysum
        Ysum += Ys[t]
        Phi += phi_t
        t_psi_t = (t+1) * Phi / (t+1)**2
        output = t_psi_t - correction
        correction = t_psi_t
        out[t] = output
    return out.flatten(0, 1)

@triton.jit
def spike_elementwise_dot_kernel(
    X_ptr, Y_ptr, Xsum_ptr, Ysum_ptr, out_ptr,
    T: tl.constexpr, BD1: tl.constexpr, ND1: tl.constexpr, BD2: tl.constexpr, ND2: tl.constexpr,
    d1: tl.constexpr, d2: tl.constexpr,
    B: tl.constexpr,  # 新增批量维度参数
):
    '''
    X_ptr, Xsum_ptr: [T, B, d1, d2]
    Y_ptr, Ysum_ptr: [T, B, d1, d2]
    out_ptr: [T, B, d1, d2]

    parallel grid: [B, ND1, ND2]
    '''
    b_id = tl.program_id(0)
    d1_block = tl.program_id(1)
    d2_block = tl.program_id(2)

    # 后续代码与原实现相同...
    d1_offsets = d1_block * BD1 + tl.arange(0, BD1)
    d2_offsets = d2_block * BD2 + tl.arange(0, BD2)

    offsets = d1_offsets[:, None] * d2 + tl.arange(0, BD2)

    x_ptr_t = X_ptr + b_id * d1 * d2 + offsets
    y_ptr_t = Y_ptr + b_id * d1 * d2 + offsets
    xsum_ptr_t = Xsum_ptr + b_id * d1 * d2 + offsets
    ysum_ptr_t = Ysum_ptr + b_id * d1 * d2 + offsets
    out_ptr_t = out_ptr + b_id * d1 * d2 + offsets
    mask = (d1_offsets[:, None] < d1) & (d2_offsets < d2)

    Phi = tl.zeros((BD1, BD2), dtype=tl.float32)
    correction = tl.zeros((BD1, BD2), dtype=tl.float32)
    for t in range(T):

        x = tl.load(x_ptr_t, mask=mask)
        y = tl.load(y_ptr_t, mask=mask)
        xsum = tl.load(xsum_ptr_t, mask=mask)
        ysum = tl.load(ysum_ptr_t, mask=mask)
        Phi += xsum * y + x * ysum

        t_psi_t = Phi / (t + 1.0)
        output = t_psi_t - correction
        correction = t_psi_t

        tl.store(out_ptr_t, output, mask=mask)

        x_ptr_t += B * d1 * d2
        y_ptr_t += B * d1 * d2
        xsum_ptr_t += B * d1 * d2
        ysum_ptr_t += B * d1 * d2
        out_ptr_t += B * d1 * d2

def spike_elementwise_dot_triton(X: torch.Tensor, Y: torch.Tensor, T: int):
    assert X.shape[-1] == Y.shape[-1] and X.shape[-2] == Y.shape[-2], \
           'dim of X and Y should be the same for elementwise.'
    d1 = X.shape[-2]
    d2 = X.shape[-1]
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    
    # Precompute cumulative sums with correct shifts
    Xsum = torch.cumsum(Xs, dim=0)
    Ysum = torch.cumsum(Ys, dim=0)
    Ysum = torch.roll(Ysum, shifts=1, dims=0)
    Ysum[0] = 0.0

    # Reshape for kernel
    Xshape = Xs.shape

    Xs = Xs.view(T, -1, d1, d2).contiguous()
    Ys = Ys.view(T, -1, d1, d2).contiguous()

    Xsum = Xsum.view(T, -1, d1, d2).contiguous()
    Ysum = Ysum.view(T, -1, d1, d2).contiguous()

    # Create output tensor
    B = Xs.shape[1]
    out = torch.empty((T, B, d1, d2), device=X.device, dtype=X.dtype)

    # Configure kernel parameters
    BD1, BD2 = 16, 16
    ND1 = triton.cdiv(d1, BD1)
    ND2 = triton.cdiv(d2, BD2)
    grid = (B, triton.cdiv(d1, BD1), triton.cdiv(d2, BD2))
    
    # 启动参数调整
    spike_elementwise_dot_kernel[grid](
        Xs, Ys, Xsum, Ysum, out,
        T, BD1, ND1, BD2, ND2,
        d1, d2,
        B,  # 新增批量维度参数
    )

    out = out.view(T, *Xshape[1:-2], d1, d2)
    return out.flatten(0, 1)

class SpikeSoftmax(nn.Module):
    def __init__(self, exp_weight_path: str, inv_weight_path: str, T: int):
        super().__init__()
        self.expop = NonLinearOp.from_pretrained(exp_weight_path)
        self.invop = NonLinearOp.from_pretrained(inv_weight_path)
        self.T = T
    
    def forward(self, x):
        x = x - x.max(dim=-1, keepdim=True)[0]
        Xs = x
        exps = self.expop(Xs)
        norms = exps.sum(dim=-1)
        invs = self.invop(norms)
        # print('[debug]', invs.mean(dim=0))
        invs = invs.unsqueeze(-1).broadcast_to(exps.shape)
        return spike_elementwise_dot(exps, invs, self.T)

class SpikeRMSNorm(nn.Module):
    def __init__(self, variance_epsilon: float, alpha: float, rsqrt_weight_path: str, T: int):
        super().__init__()
        self.variance_epilon = variance_epsilon * alpha**2
        self.alpha = alpha
        self.rsqrtop = NonLinearOp.from_pretrained(rsqrt_weight_path)
        self.T = T
    
    def forward(self, x):
        x = x * self.alpha
        x2 = spike_elementwise_dot(x, x, self.T)
        x2 = x2.mean(dim=-1)
        x2_rsqrt = self.rsqrtop(x2 + self.variance_epilon)
        x2_rsqrt = x2_rsqrt.unsqueeze(-1).broadcast_to(x.shape)
        return spike_elementwise_dot(x, x2_rsqrt, self.T)

class SpikeLlamaMLP(nn.Module):
    def __init__(self, mlp: nn.Module, silu_weight_path: str, T: int):
        super().__init__()
        self.gate_proj = nn.Linear(mlp.hidden_size, mlp.intermediate_size, bias=False)
        self.up_proj = nn.Linear(mlp.hidden_size, mlp.intermediate_size, bias=False)
        self.down_proj = nn.Linear(mlp.intermediate_size, mlp.hidden_size, bias=False)
        self.silu_op = NonLinearOp.from_pretrained(silu_weight_path)
        with torch.no_grad():
            self.gate_proj.weight.data.copy_(mlp.gate_proj.weight.data)
            self.up_proj.weight.data.copy_(mlp.up_proj.weight.data)
            self.down_proj.weight.data.copy_(mlp.down_proj.weight.data)
        self.T = T
    
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = self.silu_op(gate)
        down = spike_elementwise_dot(gate, up, self.T)
        down = self.down_proj(down)
        return down


def softmax_func(x: torch.Tensor):
    x = x - x.max(dim=-1, keepdim=True)[0]
    exps = x.exp() * 0.1
    norms = exps.sum(dim=-1, keepdim=True)
    return exps / norms

import pathlib
import sys
path = pathlib.Path(__file__).parent.parent
sys.path.append(str(path))
from quantize.quantizer import UniformAffineQuantizer

class TestSpikeOps(unittest.TestCase):
    def test_spike_matmul(self):
        T = 5
        B = 2
        X = torch.randn(T * B, 2, 3).to(torch.float)
        Y = torch.randn(T * B, 3, 4).to(torch.float)
        out_pred = spike_matmul(X, Y, T)
        out_pred = out_pred.reshape(T, B, 2, 4).mean(dim=0)
        X_mean = X.reshape(T, B, 2, 3).mean(dim=0)
        Y_mean = Y.reshape(T, B, 3, 4).mean(dim=0)
        out_true = torch.bmm(X_mean, Y_mean)
        self.assertTrue(torch.allclose(out_pred, out_true, atol=1e-3))

    def test_spike_elementwise_dot(self):
        T = 5
        B = 2
        X = torch.randint(0, 2, (T * B, 2, 3)).to(torch.float)
        Y = torch.randint(0, 2, (T * B, 2, 3)).to(torch.float)
        out_pred = spike_elementwise_dot(X, Y, T)
        out_pred = out_pred.reshape(T, B, 2, 3).mean(dim=0)
        X_mean = X.reshape(T, B, 2, 3).mean(dim=0)
        Y_mean = Y.reshape(T, B, 2, 3).mean(dim=0)
        out_true = X_mean * Y_mean
        self.assertTrue(torch.allclose(out_pred, out_true, atol=1e-3))

    def test_spike_matmul_mean_kernel(self):
        T = 4
        B = 1
        X = torch.randn((T * B, 2, 3)).to(torch.float).cuda() / 10
        Y = torch.randn((T * B, 3, 4)).to(torch.float).cuda() / 10
        out_true = spike_matmul_mean(X, Y, T)
        out_pred = spike_matmul_mean_triton(X, Y, T)
        print(out_true.view(4, 1, 2, 4))
        print(out_pred.view(4, 1, 2, 4))
        self.assertTrue(detailed_allclose_check(out_pred, out_true, atol=0, rtol=1e-5))

    def test_spike_matmul_kernel(self):
        T = 4
        B = 1
        torch.manual_seed(0)
        X = torch.randn((T * B, 2, 3)).to(torch.float).cuda()
        Y = torch.randn((T * B, 3, 4)).to(torch.float).cuda()
        out_true = spike_matmul(X, Y, T)
        out_pred = spike_matmul_triton(X, Y, T)
        print(out_true.view(T, B, 2, 4))
        print(out_pred.view(T, B, 2, 4))
        self.assertTrue(detailed_allclose_check(out_pred, out_true, atol=0, rtol=1e-5))

    def test_spike_elementwise_dot_kernel(self):
        T = 4
        B = 1
        torch.manual_seed(0)
        X = torch.randn((T * B, 2, 3)).to(torch.float).cuda()
        Y = torch.randn((T * B, 2, 3)).to(torch.float).cuda()
        out_true = spike_elementwise_dot(X, Y, T)
        out_pred = spike_elementwise_dot_triton(X, Y, T)
        print(out_true.view(T, B, 2, 3))
        print(out_pred.view(T, B, 2, 3))
        self.assertTrue(detailed_allclose_check(out_pred, out_true, atol=0, rtol=1e-5))

    def test_spike_softmax(self):
        T = 5
        B = 2
        X = torch.randn((B, 10, 10)) * 10
        X = X.unsqueeze(0).repeat(T, 1, 1, 1).flatten(0, 1)
        softmax = SpikeSoftmax('exp.pth', 'inv.pth', T)
        out_pred = softmax(X)
        out_pred = out_pred.reshape(T, B, 10, 10).mean(dim=0)
        X_mean = X.reshape(T, B, 10, 10).mean(dim=0)
        # out_pred = softmax_func(X_mean)
        out_true = torch.softmax(X_mean, dim=-1)
        print(out_pred[0])
        print(out_true[0])
        self.assertTrue(torch.allclose(out_pred, out_true, atol=1e-3))

    def test_spike_quantizer(self):
        x = torch.randn((1, 3, 4))
        quantizer = UniformAffineQuantizer(
            n_bits=4,
            quantized_shape=(2, 3, 4),
            asym=False,
            group_size=-1,
            quant_type='activation',
            mode='static',
            minmax_init=False
        )
        quantizer.scale.data[0, 0] = 0.2
        # quantizer.scale.data[1, 0] = 0.1
        out_true = quantizer(x)
        spike_quantizer = SpikeQuantizer.from_pretrained(quantizer, 16)
        print('input: ')
        print(x)
        x = x.unsqueeze(0).repeat(16, 1, 1, 1).flatten(0, 1)
        out_pred = spike_quantizer(x)
        out_pred = out_pred.reshape(16, 1, 3, 4).mean(0)
        print('spike quantize: ')
        print(out_pred)
        print('real quantize: ')
        print(out_true)

if __name__ == '__main__':
    test = TestSpikeOps()
    test.test_spike_quantizer()
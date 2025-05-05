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
    
    # Reshape inputs
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    
    # Compute cumulative sums with correct timing
    X_cumsum = torch.cumsum(Xs, dim=0)  # Xsum includes current timestep
    Y_cumsum = torch.cat([torch.zeros_like(Ys[:1]), torch.cumsum(Ys[:-1], dim=0)], dim=0)  # Ysum excludes current
    
    # Compute phi terms
    term1 = torch.einsum('t...ij,t...jk->t...ik', X_cumsum, Ys)
    # term2 = torch.einsum('t...ij,t...jk->t...ik', Xs, Y_cumsum)
    # Phi = torch.cumsum(term1 + term2, dim=0).float()
    Phi = term1.float()
    
    # Compute t_psi_t terms
    times = torch.arange(1, T+1, device=X.device).view(T, *([1]*(Phi.dim()-1)))
    times = (times + 1) / 2
    t_psi_t = Phi / times
    
    # Compute corrections - this part cannot be fully parallelized
    # But we can use cumsum trick for the correction terms
    # Even more elegant with torch.diff
    out = torch.diff(t_psi_t, dim=0, prepend=torch.zeros_like(t_psi_t[:1]))
    
    out = out.to(X.dtype)
    return out.flatten(0, 1)

def spike_matmul_mean(X: torch.Tensor, Y: torch.Tensor, T: int):
    '''
    X.shape = ((T, *), d1, d2), typically = ((T, B, H), L, d)
    Y.shape = ((T, *), d2, d3), typically = ((T, B, H), d, L)
    output expected to be (1/T * sum(X_t)) @ (1/T * sum(Y_t))
    '''
    assert X.shape[-1] == Y.shape[-2], 'dim of X and Y must match for matmul.'
    
    # Reshape to (T, B..., d1, d2) and (T, B..., d2, d3)
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    
    # Xsum is standard cumsum (includes current timestep)
    X_cumsum = torch.cumsum(Xs, dim=0)
    
    # Ysum is cumsum but shifted right by 1 (excludes current timestep)
    Y_cumsum = torch.cumsum(Ys, dim=0)
    Y_cumsum = torch.roll(Y_cumsum, shifts=1, dims=0)
    Y_cumsum[0] = 0  # First element should be zero
    
    # Compute the two terms
    term1 = torch.einsum('t...ij,t...jk->t...ik', X_cumsum, Ys)  # Xsum @ Ys[t]
    term2 = torch.einsum('t...ij,t...jk->t...ik', Xs, Y_cumsum)   # Xs[t] @ Ysum
    
    # Accumulate and average
    Phi = torch.cumsum(term1 + term2, dim=0).float()
    times = torch.arange(1, T+1, device=X.device).view(T, *([1]*(Phi.dim()-1)))
    out = Phi / (times ** 2)
    out = out.to(X.dtype)
    return out.flatten(0, 1)

def spike_elementwise_dot(X: torch.Tensor, Y: torch.Tensor, T: int):
    '''
    X.shape = ((T, *), d1, d2), typically = ((T, B, H), L, d)
    Y.shape = ((T, *), d1, d2), typically = ((T, B, H), L, d)
    output expected to be (1/T * sum(X_t)) * (1/T * sum(Y_t))
    '''
    assert X.shape[-1] == Y.shape[-1] and X.shape[-2] == Y.shape[-2], \
           'dim of X and Y should be the same for elementwise.'
    # Reshape inputs
    Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
    Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
    
    # Compute cumulative sums with correct timing
    X_cumsum = torch.cumsum(Xs, dim=0)  # Xsum includes current timestep
    Y_cumsum = torch.cat([torch.zeros_like(Ys[:1]), torch.cumsum(Ys[:-1], dim=0)], dim=0)  # Ysum excludes current
    
    # Compute phi terms
    term1 = X_cumsum * Ys
    term2 = Xs * Y_cumsum
    Phi = torch.cumsum(term1 + term2, dim=0).float()
    
    # Compute t_psi_t terms
    times = torch.arange(1, T+1, device=X.device).view(T, *([1]*(Phi.dim()-1)))
    t_psi_t = Phi / times
    
    # Compute corrections - this part cannot be fully parallelized
    # But we can use cumsum trick for the correction terms
    # Even more elegant with torch.diff
    out = torch.diff(t_psi_t, dim=0, prepend=torch.zeros_like(t_psi_t[:1]))
    
    out = out.to(X.dtype)
    return out.flatten(0, 1)


class SpikeSoftmax(nn.Module):
    def __init__(self, exp_weight_path: str, inv_weight_path: str, T: int):
        super().__init__()
        self.expop = NonLinearOp.from_pretrained(exp_weight_path, tag='exp')
        self.invop = NonLinearOp.from_pretrained(inv_weight_path, tag='inv')
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
        self.rsqrtop = NonLinearOp.from_pretrained(rsqrt_weight_path, tag='rinv')
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
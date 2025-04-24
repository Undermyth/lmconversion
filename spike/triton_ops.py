import triton
import torch
import unittest
import triton.language as tl
from einops import rearrange
from ops import spike_matmul_mean, spike_matmul, spike_elementwise_dot, detailed_allclose_check

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

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=[]
)
@triton.jit
def spike_matmul_mean_fwd_kernel(
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
    spike_matmul_mean_fwd_kernel[grid](
        Xs, Ys, Xsum, Ysum, out,
        T, BD1, BD2, ND2, BD3,
        d1, d2, d3,
        B,  # 新增批量维度参数
    )

    out = out.view(T, *Xshape[1:-2], d1, d3)
    return out.flatten(0, 1)

def spike_matmul_mean_bwd_pytorch(dphi: torch.Tensor, Xsum: torch.Tensor, Ysum: torch.Tensor, T: int):
    '''
    dphi: [(T B) ... d1 d3]
    Xsum: [T B' d1 d2]
    Ysum: [T B' d3 d2]
    return dX: [(T B) ... d1 d2], dY: [(T B) ... d2 d3]
    '''
    B = Xsum.shape[1]
    T, B, d1, d2 = Xsum.shape
    _, _, d3, _ = Ysum.shape

    dphi = rearrange(dphi, '(T B) ... -> T B ...', T=T)
    mid_shape = dphi.shape[1:-2]
    dphi = dphi.view(T, B, d1, d3)

    dXs = []
    dYs = []
    dX = torch.zeros_like(Xsum[0])
    dY = torch.zeros_like(Ysum[0])

    for t in range(T, -1, -1):
        dX += torch.bmm(dphi[t], Ysum[t].trans())
        dY += torch.bmm(Xsum[t].trans(), dphi[t])

        dXs.append(dX)
        dYs.append(dY)
    
    dX = torch.stack(dXs, dim=0)
    dY = torch.stack(dYs, dim=0)
    dX = dX.view(T, *mid_shape, d1, d2)
    dY = dY.view(T, *mid_shape, d3, d2).transpose(-2, -1)

    return dX.flatten(0, 1), dY.flatten(0, 1)

class SpikeMatmulMeanPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y, T):
        _, d1, d2 = X.shape
        _, d2, d3 = Y.shape
        Xs = rearrange(X, '(T B) ... -> T B ...', T=T)
        Ys = rearrange(Y, '(T B) ... -> T B ...', T=T)
        
        # Precompute cumulative sums with correct shifts
        Xsum = torch.cumsum(Xs, dim=0)
        Ysum = torch.cumsum(Ys, dim=0)

        Xsum = Xsum.view(T, -1, d1, d2).contiguous()
        Ysum = Ysum.view(T, -1, d2, d3).transpose(-2, -1).contiguous()

        ctx.save_for_backward(Xsum, Ysum)
        ctx.T = T

        return spike_matmul_mean(X, Y, T)

    @staticmethod
    def backward(ctx, dphi):
        T = ctx.T
        Xsum, Ysum = ctx.saved_tensors
        dX, dY = spike_matmul_mean_bwd_pytorch(dphi, Xsum, Ysum, T)
        return dX, dY, None
    
spike_matmul_mean_pytorch = SpikeMatmulMeanPytorch.apply

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=[]
)
@triton.jit
def spike_matmul_mean_bwd_dx_kernel(
    X_ptr, Y_ptr, Xsum_ptr, Ysum_ptr,
    dphi_ptr, dX_ptr,
    T: tl.constexpr, BD1: tl.constexpr, BD2: tl.constexpr, 
    BD3: tl.constexpr, ND3: tl.constexpr,
    d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr,
    B: tl.constexpr,  # 新增批量维度参数
):
    '''
    X_ptr, Xsum_ptr: [T, B, d1, d2]
    Y_ptr, Ysum_ptr: [T, B, d2, d3], different from fwd kernel
    dphi_ptr: [T, B, d1, d3]
    out_ptr: [T, B, d1, d3]

    parallel grid: [B, ND1, ND2]
    '''
    b_id = tl.program_id(0)
    d1_block_id = tl.program_id(1)
    d2_block_id = tl.program_id(2)

    dphi_ptr_t = dphi_ptr + (T - 1) * B * d1 * d3 + b_id * d1 * d3
    ysum_ptr_t = Ysum_ptr + (T - 1) * B * d2 * d3 + b_id * d2 * d3
    dx_ptr_t = dX_ptr + (T - 1) * B * d1 * d2 + b_id * d1 * d2

    d1_offsets = d1_block_id * BD1 + tl.arange(0, BD1)
    d2_offsets = d2_block_id * BD2 + tl.arange(0, BD2)

    dphi_offsets = d1_offsets[:, None] * d3 + tl.arange(0, BD3)
    ysum_offsets = d2_offsets[:, None] * d3 + tl.arange(0, BD3)
    dx_offsets = d1_offsets[:, None] * d2 + tl.arange(0, BD2)

    dx_mask = (d1_offsets[:, None] < d1) & (d2_offsets < d2)

    dx = tl.zeros((BD1, BD2), dtype=tl.float32)
    for t in range(T-1, -1, -1):

        d3_offset = 0
        for j in range(ND3):
            dphi_mask = (d1_offsets[:, None] < d1) & (d3_offset + tl.arange(0, BD3) < d3)
            ysum_mask = (d2_offsets[:, None] < d2) & (d3_offset + tl.arange(0, BD3) < d3)
            dphi = tl.load(dphi_ptr_t + dphi_offsets + d3_offset, mask=dphi_mask)
            ysum = tl.load(ysum_ptr_t + ysum_offsets + d3_offset, mask=ysum_mask)
            dx = tl.dot(dphi, ysum.trans(), dx)
            d3_offset += BD3
        
        tl.store(dx_ptr_t, dx, mask=dx_mask)
        # dphi_ptr_t -= B * 

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

class TestTritonOps(unittest.TestCase):
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

    def test_spike_matmul_mean_kernel(self):
        T = 4
        B = 1
        X = torch.randn((T * B, 2, 3), requires_grad=True).to(torch.float).cuda() / 10
        Y = torch.randn((T * B, 3, 4), requires_grad=True).to(torch.float).cuda() / 10
        X2 = X.clone()
        Y2 = Y.clone()
        out_true = spike_matmul_mean(X, Y, T)
        out_pred = spike_matmul_mean_pytorch(X2, Y2, T)
        self.assertTrue(detailed_allclose_check(out_pred, out_true, atol=0, rtol=1e-5))
        print('fwd check passed')
        out_true.mean().backward()
        out_pred.mean().backward()
        self.assertTrue(detailed_allclose_check(X.grad, X2.grad, atol=0, rtol=1e-5))
        self.assertTrue(detailed_allclose_check(Y.grad, Y2.grad, atol=0, rtol=1e-5))
        print('bwd check passed')
    
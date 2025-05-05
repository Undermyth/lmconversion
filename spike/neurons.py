import torch.nn as nn
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import unittest

class Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        alpha = 4.0
        alpha_x = F.sigmoid(alpha * x)
        grad_input  = grad_output * alpha * alpha_x * (1 - alpha_x)
        # print('neuron backward passed')
        return grad_input
    
surrogate = Surrogate.apply

@triton.jit
def dsurrogate(x):
    alpha = 4.0
    alpha_x = tl.sigmoid(alpha * x)
    return alpha * alpha_x * (1 - alpha_x)

class IFNeuron(nn.Module):

    def __init__(self):
        super().__init__()
        self.mem = 0
        self.threshold = None
        self.spike_mode = True
        self.t = 0

    def forward(self, x):
        if self.spike_mode:
            if self.t == 0:
                self.mem = torch.zeros_like(x).to(x.device)
                self.threshold = self.threshold.to(x.device)
                self.mem += self.threshold / 2
            self.t += 1
            self.mem = self.mem + x
            # spike = (self.mem >= self.threshold).float()
            spike = surrogate(self.mem - self.threshold)
            self.mem = self.mem - spike * self.threshold
            return spike
            # return clip_floor(F.relu(x), 16, self.threshold)
        else:
            return F.relu(x)
        
    def reset(self):
        self.t = 0

class CIFNeuron(nn.Module):

    def __init__(self):
        super().__init__()
        self.mem = 0
        self.threshold = None
        self.spike_mode = False
        self.t = 0
        self.burst = True

    def forward(self, x):
        if self.spike_mode:
            torch.cuda.empty_cache()
            if self.t == 0:
                self.mem = torch.zeros_like(x).to(x.device)
                self.threshold = self.threshold.to(x.device)
                self.mem += self.threshold / 2
                self.total_mem = self.mem.clone()
                self.spike_count = torch.zeros_like(self.mem)
            self.t += 1
            print(f'------------------- step {self.t} -------------------')
            print(f'membrane before update: ')
            print(self.mem)
            print('input: ')
            print(x)
            self.mem += x

            if self.burst:
                spike = torch.zeros_like(self.mem)
                while (self.mem > self.threshold).any():
                    mask = self.mem > self.threshold
                    self.mem[mask] = (self.mem - self.threshold)[mask]
                    self.spike_count[mask] = (self.spike_count + self.threshold)[mask]
                    spike[mask] = (spike + self.threshold)[mask]
                while torch.logical_and(self.mem < -self.threshold, self.spike_count > 0).any():
                    mask = torch.logical_and(self.mem < -self.threshold, self.spike_count > 0)
                    self.mem[mask] = (self.mem + self.threshold)[mask]
                    self.spike_count[mask] = (self.spike_count - self.threshold)[mask]
                    spike[mask] = (spike - self.threshold)[mask]
            
            else:
                spike = (self.mem >= self.threshold).float() * self.threshold
                neg_spikes = torch.logical_and(self.mem <= -self.threshold, self.spike_count > 0) * -self.threshold 
                spike += neg_spikes
                self.mem = (self.mem - spike).detach()
                self.spike_count = (self.spike_count + spike).detach()

            print('spikes: ')
            print(spike)
            print('membrane after update & reset: ')
            print(self.mem)
            self.total_mem = (self.total_mem + x).detach()
            return spike
        else:
            return torch.clip(x, 0, 1)
        
    def reset(self):
        self.t = 0

class BurstNeuron(nn.Module):

    def __init__(self):
        super().__init__()
        self.mem = 0
        self.threshold = None
        self.spike_mode = False
        self.t = 0
        self.max_spike = None

    def init_max_spike(self, T):
        assert self.threshold is not None
        self.max_spike = T * self.threshold

    def forward(self, x, burst = False):
        if self.spike_mode:
            if self.t == 0:
                self.mem = torch.zeros_like(x)
                self.threshold = self.threshold.to(x)
                self.mem += self.threshold / 2
                self.total_mem = self.mem.clone()
                self.spike_count = torch.zeros_like(self.mem)
                if self.max_spike is not None:
                    self.max_spike = self.max_spike.to(x)
            self.t += 1
            self.mem += x

            remain_spikes = None
            if self.max_spike is not None:
                remain_spikes = self.max_spike - self.spike_count

            if burst:
                spike = torch.zeros_like(self.mem)
                # print('enter')
                # print(self.mem.shape)
                while torch.logical_and(self.mem > self.threshold, spike <= self.max_spike).any():
                    mask = self.mem > self.threshold
                    # print((self.mem - self.threshold)[mask], self.threshold)
                    self.mem[mask] = (self.mem - self.threshold)[mask]
                    self.spike_count[mask] = (self.spike_count + self.threshold)[mask]
                    spike[mask] = (spike + self.threshold)[mask]
                while torch.logical_and(torch.logical_and(self.mem < -self.threshold, self.spike_count > 0), spike <= self.max_spike).any():
                    mask = torch.logical_and(self.mem < -self.threshold, self.spike_count > 0)
                    self.mem[mask] = (self.mem + self.threshold)[mask]
                    self.spike_count[mask] = (self.spike_count - self.threshold)[mask]
                    spike[mask] = (spike - self.threshold)[mask]
                # print('exit')
                del mask
            
            else:
                spike = ((self.mem >= self.threshold).float() * self.threshold).to(x)
                neg_spikes = torch.logical_and(self.mem <= -self.threshold, self.spike_count > 0) * -self.threshold 
                spike += neg_spikes
                self.mem -= spike
                self.spike_count += spike
                del neg_spikes

            
            if remain_spikes is not None:
                spike = torch.min(spike, remain_spikes)

            # print('spikes: ')
            # print(spike)
            # print('membrane after update & reset: ')
            # print(self.mem)
            self.total_mem += x

            # after = torch.cuda.memory_allocated(device=x.device) / (1024**3)
            # print(f'GPU Memory change (GB): {after - pre}')
            # print(f'GPU Memory usage at end (GB): {after}')
            return spike
        else:
            return torch.clip(x, 0, 1)
        
    def reset(self):
        self.t = 0

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=[]
)
@triton.jit
def multistep_neuron_update_fwd_kernel(x_ptr, thr_ptr, spk_out_ptr, 
                                       T: tl.constexpr, B: tl.constexpr, BD: tl.constexpr, d: tl.constexpr):
    '''
    x.shape = [T, B, d]
    spk_out.shape = [T, B, d]
    thr.shape = [d]

    parallel grid: [B, ND]
    '''
    b_id = tl.program_id(0)
    d_block_id = tl.program_id(1)
    d_offset = d_block_id * BD + tl.arange(0, BD)
    mask = d_offset < d

    # base addresses
    x_base_ptr = x_ptr + b_id * d + d_offset
    spk_out_base_ptr = spk_out_ptr + b_id * d + d_offset
    thr_base_ptr = thr_ptr + d_offset

    # initialization
    thr = tl.load(thr_base_ptr, mask=mask)
    mem = tl.zeros((BD, ), dtype=tl.float32)
    mem = mem + thr / 2
    spike_count = tl.zeros((BD, ), dtype=tl.float32)

    # iteration
    for i in range(T):
        x = tl.load(x_base_ptr, mask=mask)
        mem += x
        spike = (mem > thr) * thr
        neg_spike = ((mem <= -thr) & (spike_count > 0)) * (-thr)
        spike += neg_spike
        mem -= spike
        spike_count += spike
        tl.store(spk_out_base_ptr, spike, mask=mask)

        x_base_ptr += B * d
        spk_out_base_ptr += B * d

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=[]
)
@triton.jit
def multistep_neuron_update_bwd_kernel(dspk_ptr, m_ptr, c_ptr, thr_ptr,
                                       dx_ptr, 
                                       T: tl.constexpr, B: tl.constexpr, BD: tl.constexpr, d: tl.constexpr):
    '''
    dspk.shape = [T, B, d]
    m.shape = [T, B, d]
    c.shape = [T, B, d]
    thr.shape = [d]

    parallel grid: [B, ND]
    '''
    b_id = tl.program_id(0)
    d_block_id = tl.program_id(1)
    d_offset = d_block_id * BD + tl.arange(0, BD)
    mask = d_offset < d

    # base addresses
    dspk_base_ptr = dspk_ptr + (T - 1) * B * d + b_id * d + d_offset
    m_base_ptr = m_ptr + (T - 1) * B * d + b_id * d + d_offset
    c_base_ptr = c_ptr + (T - 1) * B * d + b_id * d + d_offset
    dx_base_ptr = dx_ptr + (T - 1) * B * d + b_id * d + d_offset
    thr_base_ptr = thr_ptr + d_offset

    # initialization
    thr = tl.load(thr_base_ptr, mask=mask)
    
    c = tl.load(c_base_ptr, mask=mask)
    m = tl.zeros_like(c)
    dc = tl.zeros_like(c)
    ds = tl.zeros_like(c)
    dm = tl.zeros_like(c)
    for t in range(T - 1, 0, -1):

        dspk = tl.load(dspk_base_ptr, mask=mask)

        dc -= tl.cast(ds * thr * dsurrogate(c) * (m <= -thr), tl.float16)
        ds = dspk - dm + dc

        m = tl.load(m_base_ptr, mask=mask)
        m_base_ptr -= B * d
        c_base_ptr -= B * d
        c = tl.load(c_base_ptr, mask=mask)

        dm += tl.cast(ds * thr * (dsurrogate(m - thr) - dsurrogate(m + thr) * (c > 0)), tl.float16)

        tl.store(dx_base_ptr, dm, mask=mask)
        dx_base_ptr -= B * d


def multistep_neuron_update_bwd_triton(dspk, m, c, thr):
    origin_shape = dspk.shape
    T = origin_shape[0]
    d = origin_shape[-1]
    dspk = dspk.view(T, -1, d).contiguous()

    BD = 128
    ND = triton.cdiv(d, BD)
    B = dspk.shape[1]
    dx = torch.empty_like(dspk)
    multistep_neuron_update_bwd_kernel[(B, ND)](dspk, thr, dx, T, B, BD, d)
    dx = dx.view(*origin_shape)
    # print('signed neuron bwd passed')
    return dx

def multistep_neuron_update_fwd_triton(x, thr):
    '''
    x: [T, *, d]
    thr: [d]
    '''
    origin_shape = x.shape
    T = origin_shape[0]
    d = origin_shape[-1]
    x = x.view(T, -1, d).contiguous()
    BD = 128
    ND = triton.cdiv(d, BD)
    B = x.shape[1]
    spk_out = torch.empty_like(x)
    multistep_neuron_update_fwd_kernel[(B, ND)](x, thr, spk_out, T, B, BD, d)
    spk_out = spk_out.view(*origin_shape)
    return spk_out
 
class MultistepNeuronUpdateTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, thr):
        spk_out = multistep_neuron_update_fwd_triton(x, thr)
        ctx.save_for_backward(thr)
        return spk_out

    @staticmethod
    def backward(ctx, dspk):
        return None, None
    
multistep_neuron_update_triton = MultistepNeuronUpdateTriton.apply

class TestNeuron(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.neuron = BurstNeuron().cuda()
        self.neuron.spike_mode = True
        self.neuron.threshold = torch.randn(8).cuda()
    
    def test_fwd_numeric(self):
        x = torch.randn(4, 1, 8).cuda()

        print(self.neuron.threshold)
        output = []
        for i in range(4):
            output.append(self.neuron(x[i], burst=False).detach())
        out_ref = torch.stack(output)

        out_test = multistep_neuron_update_triton(x, self.neuron.threshold)

        print(out_ref[:, 0])
        print(out_test[:, 0])

        self.assertTrue(torch.allclose(out_ref, out_test, rtol=1e-5, atol=1e-5))

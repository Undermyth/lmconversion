import torch
import torch.nn as nn
import unittest
from einops import rearrange, repeat
from transformers.cache_utils import DynamicCache
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from netfit import NonLinearOp
from relu_to_if import CIFNeuron, IFNeuron, BurstNeuron

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
        assert not torch.isnan(Xs[t]).any(), f'X {t}'
        assert not torch.isnan(Ys[t]).any(), f'Y {t}'
        Xsum += Xs[t]
        phi_t = torch.bmm(Xsum, Ys[t]) + torch.bmm(Xs[t], Ysum)
        assert not torch.isinf(phi_t).any()
        Ysum += Ys[t]
        Phi += phi_t.float()
        assert not torch.isinf(Phi).any()
        # t_psi_t = (t+1) * Phi / (t+1)**2
        t_psi_t = Phi / (t+1)
        output = t_psi_t - correction
        assert not torch.isinf(t_psi_t).any()
        assert not torch.isinf(correction).any()
        assert not torch.isnan(output).any()
        correction = t_psi_t
        out[t] = output
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
        assert not torch.isnan(Xs[t]).any(), f'X {t}'
        assert not torch.isnan(Ys[t]).any(), f'Y {t}'
        Xsum += Xs[t]
        phi_t = torch.bmm(Xsum, Ys[t]) + torch.bmm(Xs[t], Ysum)
        assert not torch.isinf(phi_t).any()
        Ysum += Ys[t]
        Phi += phi_t.float()
        out[t] = Phi / (t+1)**2
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
    def __init__(self, rsqrt_weight_path: str, T: int):
        self.rsqrtop = NonLinearOp.from_pretrained(rsqrt_weight_path)
        self.T = T
    
    def forward(self, x):
        x2 = spike_elementwise_dot(x, x, self.T)
        x2 = x2.mean(dim=-1)
        x2_rsqrt = self.rsqrtop(x2)
        x2_rsqrt = x2_rsqrt.unsqueeze(-1).broadcast_to(x.shape)
        return spike_elementwise_dot(x, x2_rsqrt, self.T)


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
        self.assertTrue(torch.allclose(out_pred, out_true, atol=1e-2))

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
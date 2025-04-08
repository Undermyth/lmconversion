import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple
from functools import partial
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from relu_to_if import BurstNeuron, ReLUtoIFHook
from einops import rearrange
import copy

class Approximator(nn.Module):
    def __init__(self, n_neurons: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.fc1 = nn.Linear(1, n_neurons)
        self.clip = BurstNeuron()
        self.fc2 = nn.Linear(n_neurons, 1)
        self.clip.spike_mode = False
    
    def forward(self, x: torch.Tensor, T = None) -> torch.Tensor:
        if self.clip.spike_mode:
            assert T is not None
            self.reset()
            self.clip.init_max_spike(T)
            x = self.fc1(x)
            x = rearrange(x, '(T B) ... -> T B ...', T=T)
            output = []
            for i in range(T):
                output.append(self.clip(x[i], burst=False).detach())
                torch.cuda.empty_cache()
            output = torch.stack(output)
            output = rearrange(output, 'T B ... -> (T B) ...')
            x = self.fc2(output)
        else:
            x = self.fc1(x)
            x = self.clip(x)
            x = self.fc2(x)
        return x

    def set_snn_threshold(self, x_range: Tuple[int, int], T: int):
        handler = ReLUtoIFHook(T)
        hook = self.clip.register_forward_hook(handler)
        x = torch.linspace(x_range[0], x_range[1], 1000).unsqueeze(1).cuda()
        self(x)
        hook.remove()
        self.clip.spike_mode = True
        print(self.clip.threshold)

    def reset(self):
        self.clip.reset()

    def spike_on(self):
        self.clip.spike_mode = True
    
    def spike_off(self):
        self.clip.spike_mode = False

class NonLinearOp(nn.Module):
    '''
    A small network that approximates a non-linear function.
    when applied, all elements are passed through the net separately.
    if `self.spike=False`: 
        it works as a normal non-linear function, 
        the inputs/outputs are as the same shape in the original network.
        only fitting loss is considered in this case.
    if `self.spike=True`:
        it works as a spiking non-linear function in multistep mode, 
        the inputs/outputs are shaped as ((T*B), d), and will be reshaped
        into (T, B, d) inside the forward pass. 
        fitting loss, quantization loss and other possible losses are considered.
    '''
    def __init__(self, approximator: Approximator, T: int, spike: bool = False):
        super().__init__()
        self.approximator = approximator
        self.spike = spike
        self.T = T

    @classmethod
    def from_pretrained(cls, weight_path: str):
        checkpoint = torch.load(weight_path)
        n_neurons = checkpoint['n_neurons']
        T = checkpoint['T']
        state_dict = checkpoint['state_dict']
        state_dict['approximator.fc2.bias'] = state_dict['approximator.fc2.bias'].unsqueeze(0)
        approximator = Approximator(n_neurons)
        approximator.clip.threshold = checkpoint['threshold']
        instance = cls(approximator, T, spike=True)
        instance.eval()
        instance.load_state_dict(checkpoint['state_dict'])
        return instance
    
    def save_to_pretrained(self, weight_path: str):
        checkpoint = {
            'n_neurons': self.approximator.n_neurons,
            'T': self.T,
            'threshold': self.approximator.clip.threshold,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, weight_path)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        if self.spike:
            self.approximator.spike_on()
            x = self.approximator(x, self.T)
        else:
            self.approximator.spike_off()
            x = self.approximator(x)
        x = x.squeeze(-1)
        return x
    
def direct_train(n_neurons, T, xrange: Tuple[float, float], func):
    net = Approximator(n_neurons).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    best_net = None
    best_loss = 1e10
    for i in range(50000):
        x = torch.rand(128, 1).cuda() * (xrange[1] - xrange[0]) + xrange[0]
        y = func(x)
        y_hat = net(x)
        loss = torch.mean((y - y_hat)**2) + torch.mean(net.fc2.weight.abs() / (12 * T**2 * net.fc1.weight.view(1, n_neurons)**2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 100 == 0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_net = copy.deepcopy(net)
            print(i, loss.item())
    return best_net, losses
        

class FakeNonLinearOp(Approximator):    # actually a quantizer. Only for visualization
    def __init__(self, n_neurons: int, T: int):
        super().__init__(n_neurons)
        self.T = T
        self.step_mode = 'm'
        self.clip.spike_mode = True
    
    @classmethod
    def from_approx(cls, approximator, T):
        device = approximator.fc1.weight.data.device
        instance = cls(approximator.n_neurons, T)
        instance.fc1.weight.data = approximator.fc1.weight.data.clone()
        instance.fc1.bias.data = approximator.fc1.bias.data.clone()
        instance.fc2.weight.data = approximator.fc2.weight.data.clone()
        instance.fc2.bias.data = approximator.fc2.bias.data.clone()
        instance.clip.threshold = approximator.clip.threshold.clone().to(device)
        return instance
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        if self.step_mode == 'm':
            assert self.clip.spike_mode == True
            self.reset()
            x = self.fc1(x)
            output = 0
            for i in range(self.T):
                output += self.clip(x)
            x = output / self.T
            x = self.fc2(x)
        else:
            x = self.fc1(x)
            x = self.clip(x)
            x = self.fc2(x)
        return x.squeeze(-1)


def simpson_integrate(f: Callable, x0: torch.Tensor, x1: torch.Tensor, n_secs = 10):
    """
    Approximates the integral of a function over the interval [x0, x1] using Simpson's rule.

    Simpson's rule is a numerical integration technique that approximates the integral of a function
    by fitting parabolic arcs between pairs of points. This method provides a more accurate approximation
    than the trapezoidal rule for smooth functions.

    Args:
        f (Callable): The function to integrate. This function should take three arguments: x0, x1, and lambdas,
                      where x0 and x1 are the start and end points of the interval, and lambdas is a tensor of
                      values between 0 and 1 representing the relative positions within the interval.
        x0 (torch.Tensor): The start point of the integration interval.
        x1 (torch.Tensor): The end point of the integration interval.
        n_secs (int, optional): The number of sections to divide the interval into for the approximation.
                                Defaults to 10.

    Returns:
        torch.Tensor: The approximate value of the integral of the function over the interval [x0, x1].
    """
    lambdas = torch.linspace(0, 1, n_secs + 1)
    delta = 1. / n_secs
    half_lambdas = lambdas[1:] - 0.5 * delta
    f_halfs = 4 * f(x0, x1, half_lambdas).sum()
    fs = 2 * f(x0, x1, lambdas[1:-1]).sum()
    integral = 1/6 * (x1 - x0) * (f(x0, x1, lambdas[0]) + f_halfs + fs + f(x0, x1, lambdas[-1]))
    return integral


def get_relative_weight(bin_size, weights: torch.Tensor, x_secs: torch.Tensor):
    '''
    Given a finer histgram of weight distribution (defined by `bin_size` and `weights`), and a segmentation
    of the whole interval (defined by `x_secs`), calculate the summed weight of each subinterval.

    Args:
        bin_size (float): bin size of weights.
        weights: (torch.Tensor): weight distribution depicted as hist data. The start and end point should 
                                 match that of `x_secs`.
        x_secs: (torch.Tensor): section points of the segmentation.
    '''
    x_starts = x_secs[:-1]
    x_ends = x_secs[1:]
    start_indexes = torch.floor((x_starts - x_starts[0]) / bin_size).to(torch.long)
    end_indexes = torch.floor((x_ends - x_starts[0]) / bin_size).to(torch.long)
    start_indexes[0] = -1
    end_indexes[-1] = weights.shape[0]
    res = []
    for start_index, end_index in zip(start_indexes, end_indexes):
        if start_index + 1 == end_index:
            res.append(0)
        else:
            res.append(weights[start_index + 1: end_index].sum().item())
    return torch.Tensor(res)

def loss_func(func: Callable, x_secs: torch.Tensor, y_secs: torch.Tensor, T: int, weights = None, bin_size = None, n_internal_secs = 10):
    """
    Computes the loss function for a given set of section points and their corresponding function values.

    The loss function is calculated as the mean squared error (MSE) between the true function values and the 
    linearly interpolated values over each section. The integration over each section is performed using 
    Simpson's rule to approximate the integral of the squared error.

    Args:
        func (Callable): The true function to compare against. This function should take a tensor of x-values 
                         and return the corresponding y-values.
        x_secs (torch.Tensor): A tensor of x-values representing the start and end points of each section.
        y_secs (torch.Tensor): A tensor of y-values corresponding to the x-values in `x_secs`.
        T (int): Quantization levels. Used for quantization loss estimation.
        n_internal_secs (int, optional): The number of internal sections to use for Simpson's integration 
                                         within each section. Defaults to 10.

    Returns:
        torch.Tensor: The computed loss value, which is the sum of the integrated squared errors over all sections.
    """
    n_secs = x_secs.shape[0] - 1
    x_starts = x_secs[:-1]
    x_ends = x_secs[1:]
    y_starts = y_secs[:-1]
    y_ends = y_secs[1:]

    if weights is not None and bin_size is not None:
        balance = get_relative_weight(bin_size, weights, x_starts, x_ends).detach()
    else:
        balance = torch.ones(n_secs).detach()

    def mse(x0, x1, lambdas, y0, y1):
        x = (1 - lambdas) * x0 + lambdas * x1
        y = (1 - lambdas) * y0 + lambdas * y1
        return (func(x) - y) ** 2
    
    loss = 0
    for i in range(n_secs):
        mse_func = partial(mse, y0=y_starts[i], y1=y_ends[i])
        smooth_loss = simpson_integrate(mse_func, x_starts[i], x_ends[i], n_internal_secs)
        quant_loss = (x_ends[i] - x_starts[i]) * (y_ends[i] - y_starts[i]) ** 2 / (12 * T**2)
        loss += balance[i] * (smooth_loss + quant_loss)
    loss = loss / (x_secs[-1] - x_secs[0])  # TODO: is this correct?
    return loss

def seg_fit(func: Callable, x_range: Tuple[int, int], n_neurons: int, T: int, weights = None, bin_size = None):
    """
    Fits a segmented approximation of a given function `func` over a specified range `x_range`.
    
    Args:
        func (Callable): The target function to approximate. It should be able to autodiffed.
        x_range (Tuple[int, int]): A tuple indicating the start and end points of the interval over which to fit the function.
        n_neurons (int): Number of segments (or neurons) to divide the interval into.
        T (int): Simulation step of spiking neuron. Used for quantization error estimation.
        weights (optional): Histgram weight of `x_range`. should match the range of `x_range` exactly.
        bin_size (optional): Bin size of `weights`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: optimal segmentation points with linear interpolation.
    """
    # initialization
    x_inits = torch.linspace(x_range[0], x_range[1], n_neurons + 1)
    x_inits.requires_grad_(False)
    x_learnable = x_inits[1:-1]
    x_learnable.requires_grad_(True)
    y_secs = func(x_inits)
    y_secs.requires_grad_(True)
    optimizer = torch.optim.Adam([x_learnable, y_secs], lr=1e-2)
    losses = []
    for i in range(1500):
        optimizer.zero_grad()
        x_secs = torch.concat([x_inits[0].unsqueeze(0), x_learnable, x_inits[-1].unsqueeze(0)], dim=0)
        loss = loss_func(func, x_secs, y_secs, T, weights=weights, bin_size=bin_size)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % 100 == 0:
            print(f"loss: {loss.item()}")
    return x_secs, y_secs

def fixed_seg_fit(func: Callable, x_range: Tuple[int, int], n_neurons: int, T: int, start_y: float, end_y: float, weights = None, bin_size = None):
    """
    Fits a segmented approximation of a given function `func` over a specified range `x_range`.
    
    Args:
        func (Callable): The target function to approximate. It should be able to autodiffed.
        x_range (Tuple[int, int]): A tuple indicating the start and end points of the interval over which to fit the function.
        n_neurons (int): Number of segments (or neurons) to divide the interval into.
        T (int): Simulation step of spiking neuron. Used for quantization error estimation.
        weights (optional): Histgram weight of `x_range`. should match the range of `x_range` exactly.
        bin_size (optional): Bin size of `weights`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: optimal segmentation points with linear interpolation.
    """
    x_start = torch.Tensor([x_range[0]])
    x_end = torch.Tensor([x_range[1]])
    y_start = torch.Tensor([start_y])
    y_end = torch.Tensor([end_y])

    x_learnable = torch.linspace(-1, 1, n_neurons - 1)
    x_learnable.requires_grad_(True)

    x_secs = x_start + torch.sigmoid(x_learnable) * (x_end - x_start)
    y_learnable = func(x_secs.detach())
    y_learnable.requires_grad_(True)

    optimizer = torch.optim.Adam([x_learnable, y_learnable], lr=1e-2)
    losses = []
    best_loss = 1e10
    best_x_secs = None
    best_y_secs = None
    for i in range(8000):
        optimizer.zero_grad()
        x_secs = x_start + torch.sigmoid(x_learnable) * (x_end - x_start)
        x_secs = torch.concat([x_start, x_secs, x_end], dim=0)
        y_secs = torch.concat([y_start, y_learnable, y_end], dim=0)
        x_secs, idx = torch.sort(x_secs)
        y_secs = y_secs[idx]
        loss = loss_func(func, x_secs, y_secs, T, weights=weights, bin_size=bin_size)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x_secs = x_secs
            best_y_secs = y_secs
        if i % 100 == 0:
            print(f"loss: {loss.item()}")
    return best_x_secs, best_y_secs

def segs_to_net_param(x_secs: torch.Tensor, y_secs: torch.Tensor, net: Approximator):

    # calculate params
    x_starts = x_secs[:-1]
    x_ends = x_secs[1:]
    y_starts = y_secs[:-1]
    y_ends = y_secs[1:]
    w1 = 1. / (x_ends - x_starts)
    b1 = -x_starts / (x_ends - x_starts)
    w2 = y_ends - y_starts
    b2 = y_starts[0]

    # substitute into net
    net.fc1.weight.data = w1.clone().unsqueeze(1).to(net.fc1.weight.device)
    net.fc1.bias.data = b1.clone().to(net.fc1.bias.device)
    net.fc2.weight.data = w2.clone().unsqueeze(1).transpose(0, 1).to(net.fc2.weight.device)
    net.fc2.bias.data = b2.clone().to(net.fc2.bias.device)

    return net


if __name__ == '__main__':
    x_range = (-5., 40.)
    n_neurons = 8
    T = 32

    x_secs, y_secs = seg_fit(torch.nn.functional.silu, x_range, n_neurons, T)

    net = Approximator(n_neurons=n_neurons).cuda()

    segs_to_net_param(x_secs, y_secs, net)

    net.to_snn(x_range, T)
    net.step_mode = 'm'

    net = FakeNonLinearOp.from_approx(net, T)
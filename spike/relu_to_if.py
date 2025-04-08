import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

def clip_floor(relu_output, T, Vth):
    clipped = torch.clamp(relu_output * T / Vth, 0, T)
    floored = torch.floor(clipped)
    return floored * Vth / T

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
            self.mem += x
            spike = (self.mem >= self.threshold).float()
            self.mem -= spike * self.threshold
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
            # print(f'------------------- step {self.t} -------------------')
            # pre = torch.cuda.memory_allocated(device=x.device) / (1024**3)
            # print(f'GPU Memory usage at start (GB): {pre}')
            if self.t == 0:
                self.mem = torch.zeros_like(x)
                self.threshold = self.threshold.to(x)
                self.mem += self.threshold / 2
                self.total_mem = self.mem.clone()
                self.spike_count = torch.zeros_like(self.mem)
                self.max_spike = self.max_spike.to(x)
            self.t += 1
            # print(f'membrane before update: ')
            # print(self.mem)
            # print('input: ')
            # print(x)
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
    
class ReLUtoIFHook:
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.momemtum = 0.9

    def __call__(self, module, input, output):
        if module.threshold is None:
            module.threshold = self.search_mse_threshold(output)
        else:
            cur_thres = self.search_mse_threshold(output)
            module.threshold = self.momemtum * module.threshold + (1 - self.momemtum) * cur_thres
        
    def search_mse_threshold(self, relu_output):
        num_channels = relu_output.shape[-1]
        reshaped_output = relu_output.reshape(-1, num_channels)
        max_acts = reshaped_output.max(0)[0]
        best_scores = torch.ones_like(max_acts) * float('inf')
        best_thresholds = max_acts.clone()
        for i in range(95):
            threshold = max_acts * (1 - i * 0.01)
            mse_score = (clip_floor(reshaped_output, self.T, threshold) - reshaped_output) ** 2
            mse_score = mse_score.sum(dim=0)
            mask = mse_score < best_scores
            best_scores[mask] = mse_score[mask]
            best_thresholds[mask] = threshold[mask]
        return best_thresholds
            
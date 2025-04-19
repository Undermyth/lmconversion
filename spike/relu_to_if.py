import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import triton
import triton.language as tl

def clip_floor(relu_output, T, Vth):
    clipped = torch.clamp(relu_output * T / Vth, 0, T)
    floored = torch.floor(clipped)
    return floored * Vth / T
    
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
            
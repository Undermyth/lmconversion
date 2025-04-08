import torch
import torch.nn as nn
from einops import repeat, rearrange

def firing_prehook(module, input, T):
    new_inputs = []
    for input_tensor in input:
        new_inputs.append(repeat(input_tensor, '... -> T ...', T=T).flatten(0, 1))
    return tuple(new_inputs)
 
class FiringModule(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
    def forward(self, x):
        return repeat(x, '... -> T ...', T=self.T).flatten(0, 1)
    
def firing_after_hook(module, input, output, T):
    return repeat(output, '... -> T ...', T=T).flatten(0, 1)

def avg_after_tuple_hook(module, input, output, T):
    new_output = []
    new_output.append(rearrange(output[0], '(T B) ... -> T B ...', T=T).mean(0))
    for i in range(1, len(output)):
        new_output.append(output[i])
    return tuple(new_output)

def avg_after_hook(module, input, output, T):
    return rearrange(output, '(T B) ... -> T B ...', T=T).mean(0)
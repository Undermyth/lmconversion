import pathlib
import sys
path = pathlib.Path(__file__).parent.parent
sys.path.append(str(path))

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from utils import quant_utils, data_utils, model_utils
from netfit import fixed_seg_fit, Approximator, NonLinearOp, segs_to_net_param

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="model path")
parser.add_argument("--neurons", type=int, help="number of neurons")
parser.add_argument("--T", type=int, help="simulation step")
parser.add_argument("--save_path", type=str, help="save path")
args = parser.parse_args()

# model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cpu', torch_dtype=torch.float16, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
# loader, _ = data_utils.get_loaders("wikitext2", tokenizer, train_size=64, val_size=0, seed=0, seqlen=512)

# # get activation range
# model = model.cuda()
# layers = model_utils.get_layers(model)
# for layer in layers:
#     quant_utils.apply_naive_sdpa(layer.self_attn)

# x_range = [None, None]
# def activation_hook(module, input, output):
#     if isinstance(input, tuple):
#         input = input[0]
#     input = input - input.max(dim=-1, keepdim=True)[0]
#     input = input.exp()
#     input = input.sum(dim=-1)
#     minimum = input.min()
#     maximum = input.max()
#     x_range[0] = minimum if x_range[0] is None else min(x_range[0], minimum)
#     x_range[1] = maximum if x_range[1] is None else max(x_range[1], maximum)

# hooks = []
# for layer in layers:
#     hooks.append(layer.self_attn.softmax.register_forward_hook(activation_hook))

# for i in tqdm(range(len(loader)), desc='obtain activation stat'):
#     data = loader[i][0].cuda()
#     model(data)

# for hook in hooks:
#     hook.remove()

def inv_func(x: torch.Tensor):
    return 1. / x

x_range = (0.1, 40.)
x_secs, y_secs = fixed_seg_fit(inv_func, x_range, args.neurons, args.T, start_y=1/x_range[0], end_y=0.)
net = Approximator(n_neurons=args.neurons).cuda()
segs_to_net_param(x_secs, y_secs, net)
net.set_snn_threshold(x_range, args.T)
net = NonLinearOp(net, args.T, spike=True)

net.save_to_pretrained(args.save_path)

x = torch.linspace(x_range[0], x_range[1], steps=100).view(100, 1).cuda()
y_true = inv_func(x).squeeze(-1)
x = x.repeat(args.T, 1, 1).flatten(0, 1)
y = net(x)
y = y.reshape(args.T, 100, -1)
y_pred = y.mean(dim=0).squeeze(-1)
import matplotlib.pyplot as plt
x = torch.linspace(x_range[0], x_range[1], steps=100)
plt.plot(x.cpu().detach(), y_true.cpu().detach(), label='y_true')
plt.plot(x.cpu().detach(), y_pred.cpu().detach(), label='y_pred')
plt.legend()
plt.savefig('output_plot.png')
plt.close()
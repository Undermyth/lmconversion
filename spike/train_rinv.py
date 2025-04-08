import pathlib
import sys
path = pathlib.Path(__file__).parent.parent
sys.path.append(str(path))

import argparse
import torch
from netfit import fixed_seg_fit, Approximator, NonLinearOp, segs_to_net_param
import math

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="model path")
parser.add_argument("--neurons", type=int, help="number of neurons")
parser.add_argument("--T", type=int, help="simulation step")
parser.add_argument("--save_path", type=str, help="save path")
args = parser.parse_args()

def rinv_func(x: torch.Tensor):
    return 1. / torch.sqrt(x)

x_range = (0.001, 0.13)
x_secs, y_secs = fixed_seg_fit(rinv_func, x_range, args.neurons, args.T, start_y=1./math.sqrt(x_range[0]), end_y=1./math.sqrt(x_range[1]))
net = Approximator(n_neurons=args.neurons).cuda()
segs_to_net_param(x_secs, y_secs, net)
net.set_snn_threshold(x_range, args.T)
net = NonLinearOp(net, args.T, spike=True)

net.save_to_pretrained(args.save_path)

x = torch.linspace(x_range[0], x_range[1], steps=100).view(100, 1).cuda()
y_true = rinv_func(x).squeeze(-1)
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
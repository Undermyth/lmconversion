import pathlib
import sys
path = pathlib.Path(__file__).parent.parent
sys.path.append(str(path))

import argparse
import torch
from netfit import fixed_seg_fit, Approximator, NonLinearOp, segs_to_net_param

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="model path")
parser.add_argument("--neurons", type=int, help="number of neurons")
parser.add_argument("--T", type=int, help="simulation step")
parser.add_argument("--save_path", type=str, help="save path")
parser.add_argument("--thres", type=float, default=-5., help="clipping threshold of softmax input")
parser.add_argument("--scale", type=float, default=1., help="fit scale*exp of softmax input")
args = parser.parse_args()


def exp_func(x: torch.Tensor):
    return args.scale * torch.exp(x)

x_range = (args.thres, 1.)
x_secs, y_secs = fixed_seg_fit(exp_func, x_range, args.neurons, args.T, start_y=0., end_y=exp_func(torch.Tensor([x_range[1]])))
net = Approximator(n_neurons=args.neurons).cuda()
segs_to_net_param(x_secs, y_secs, net)
net.set_snn_threshold(x_range, args.T)
net = NonLinearOp(net, args.T, spike=True)

net.save_to_pretrained(args.save_path)


# validation
# 原始x
x = torch.linspace(x_range[0], x_range[1], steps=1000).view(1000, 1).cuda()
y_true = exp_func(x).squeeze(-1)
repeat_x = x.repeat(args.T, 1, 1).flatten(0, 1)
net.approximator.clip.spike_mode = True


y = net(repeat_x)
y = y.reshape(args.T, 1000, -1)
y_pred = y.mean(dim=0).squeeze(-1)
import matplotlib.pyplot as plt
x = torch.linspace(x_range[0], x_range[1], steps=1000)
plt.plot(x.cpu().detach(), y_true.cpu().detach(), label='y_true')
plt.plot(x.cpu().detach(), y_pred.cpu().detach(), label='y_pred')
plt.legend()
plt.savefig('output_plot.png')
plt.close()
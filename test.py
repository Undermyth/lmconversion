import torch

Q = torch.load('Q.pt').float()
weight = torch.load('mlp_weight.pt').float()
bias = torch.load('mlp_bias.pt').float()
weight_ref = torch.load('mlp_weight_ref.pt').float()
bias_ref = torch.load('mlp_bias_ref.pt').float()
print(weight_ref)
print(torch.matmul(Q.T, weight_ref))
print(weight)
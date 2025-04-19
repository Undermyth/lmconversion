import triton.language as tl
import triton
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

X = torch.randint(0, 10, (2, 2)).to(torch.float).cuda() / 10
Y = torch.randint(0, 10, (2, 2)).to(torch.float).cuda() / 10
Z = torch.empty((2, 2), device='cuda')

@triton.jit
def kernel(x_ptr, y_ptr, z_ptr, d: tl.constexpr, B: tl.constexpr):
    offset = tl.arange(0, B)[:, None] * d + tl.arange(0, B)
    mask = (tl.arange(0, B)[:, None] < d) & (tl.arange(0, B) < d)
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    z = tl.dot(x, y, allow_tf32=False)
    tl.store(z_ptr + offset, z, mask=mask)

kernel[(1,)](X, Y, Z, 2, 16)

print(torch.matmul(X, Y))
print(Z)
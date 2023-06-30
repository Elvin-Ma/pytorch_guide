import torch
from torch._dynamo import optimize
import torch._inductor.config


torch._inductor.config.debug = True


def fn(x):
    a = torch.sin(x).cuda()
    b = torch.sin(a).cuda()
    return b

new_fn = optimize("inductor")(fn)
input_tensor = torch.randn(10000).to(device="cuda:0")
a = new_fn(input_tensor)
print("run dynamo_hell.py successfully !!!")

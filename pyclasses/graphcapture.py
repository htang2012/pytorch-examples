# Graph capture: Computational graph representation for your models
# and functions. PyTorch technologies: TorchDynamo, Torch FX, FX IR

import torch
import math
import os
import matplotlib.pyplot as plt
import torch._dynamo
from torchvision import models
from torch.fx.passes.graph_drawer import FxGraphDrawer
from IPython.display import Markdown as md

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def f(x):
    return torch.sin(x)**2 + torch.cos(x)**2 

gm = torch.fx.symbolic_trace(f)
gm.graph.print_tabular()
print(gm.graph)
gm.print_readable()
print(gm.code)

################################################

def inspect_backend(gm, sample_inputs):
    code = gm.print_readable()
    with open("forward.svg", "wb") as file:
        file.write(FxGraphDrawer(gm,'f').get_dot_graph().create_svg())
    return gm.forward

torch._dynamo.reset()
compiled_f = torch.compile(f, backend=inspect_backend)

x = torch.rand(1000, requires_grad=True).to(device)
out = compiled_f(x)
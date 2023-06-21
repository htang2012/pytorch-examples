# (Optional topic) Decomposition to Core Aten IR and Prims IR

import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified
from torch._decomp import core_aten_decompositions




def f_loss(x, y):
    f_x = torch.sin(x)**2 + torch.cos(x)**2
    return torch.nn.functional.mse_loss(f_x, y)

decompositions = core_aten_decompositions() # Use decomposition to Core Aten IR
#decompositions = {} # Don't use decomposition to Core Aten IR

def inspect_backend(gm, sample_inputs): 
    def fw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, 'fn')
        with open("forward_decomp.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)
    
    def bw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, 'fn')
        with open("backward_decomp.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=fw,
        bw_compiler=bw,
        decompositions=decompositions
    )
    

device = torch.device("cuda")
torch.manual_seed(0)
x = torch.rand(1000, requires_grad=True).to(device)
y = torch.ones_like(x)

torch._dynamo.reset()
compiled_f = torch.compile(f_loss, backend=inspect_backend)
out = compiled_f(x,y).backward()

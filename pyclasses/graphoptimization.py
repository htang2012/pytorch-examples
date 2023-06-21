
import torch
import torch._dynamo



def f(x):
    return torch.sin(x)**2 + torch.cos(x)**2 

torch._dynamo.reset()
compiled_f = torch.compile(f, backend='inductor',
                            options={'trace.enabled':True,
                                    'trace.graph_diagram':True})


# device = 'cpu'
device = torch.device('cpu')
#device = torch.device('cuda')

torch.manual_seed(0)
x = torch.rand(1000, requires_grad=True).to(device)
y = torch.ones_like(x)

out = torch.nn.functional.mse_loss(compiled_f(x),y).backward()
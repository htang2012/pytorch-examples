# to find out what code cause graph break, using dynamo.explain

import torch
import torch._dynamo as dynamo

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("woo")
    if b.sum() < 0:
        b = b * -1
    return x * b

explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose,= dynamo.explain(
    toy_example, torch.randn(10), torch.randn(10))

print(explanation)

print(out_guards)

print(graphs)

print(ops_per_graph)

print(break_reasons)

print(explanation_verbose)
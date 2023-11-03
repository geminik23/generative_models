import torch
import math

def create_betas(steps:int, start, end, type="linear"):
    betas = None
    if type == "linear":
        betas = torch.linspace(start, end, steps)
    elif type == "power":
        betas = torch.linspace(math.sqrt(start), math.sqrt(end), steps)**2
    elif type == "sigmoid":
        betas = torch.sigmoid(torch.linspace(-10, 8, steps))*(end-start) + start
    return betas
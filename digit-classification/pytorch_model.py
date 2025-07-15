# imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PyTorchMLP(nn.Module):
    def __init__(self, nin, nouts):
        super().__init__()
        self.lay1 = nn.Linear(nin, nouts[0])
        self.lay2 = nn.Linear(nouts[0], nouts[1])
        self.out = nn.Linear(nouts[1], nouts[2])
    
    def forward(self, x):
        x = torch.tanh(self.lay1(x))
        x = torch.tanh(self.lay2(x))
        return self.out(x)
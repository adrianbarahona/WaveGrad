import torch
import numpy as np

class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    def nparams(self):
        # pars = 0
        # for p in self.parameters():
        #     if p.requires_grad:
        #         pars = pars + np.sum(p.numel())
        return 0

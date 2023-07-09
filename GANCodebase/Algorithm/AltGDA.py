from torch.optim import Optimizer 
import torch
import math


class AltGDA(Optimizer):
    def __init__(self,params,lr=1e-3):
        if lr < 0.0:
            raise ValueError
        defaults = dict(lr=lr)
        super(AltGDA,self).__init__(params,defaults)

    def step(self, dLoss, gLoss, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dGroup = self.param_groups[0]
        gGroup = self.param_groups[1]

        dLoss.backward()
        for p in dGroup['params']:
            # boilerplate
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            # Algo
            p.data = p.data - dGroup['lr']*grad

        gLoss.backward()
        for p in gGroup['params']:
            # boilerplate
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            # Algorithm
            p.data = p.data - gGroup['lr']*grad

        return loss

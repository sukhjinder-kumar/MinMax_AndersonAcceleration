from torch.optim import Optimizer 
import torch
import math


class SimGDA(Optimizer):
    def __init__(self,params,lr=1e-3):
        if lr < 0.0:
            raise ValueError
        defaults = dict(lr=lr)
        super(SimGDA,self).__init__(params,defaults)

    def step(self, dLoss, gLoss, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dGroup = self.param_groups[0]
        gGroup = self.param_groups[1]

        dLoss.backward()
        dGrads = []
        for p in dGroup['params']:
            # boilerplate
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            dGrads.append(grad)

        gLoss.backward()
        gGrads = []
        for p in gGroup['params']:
            # boilerplate
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            gGrads.append(grad)

        for i,p in enumerate(dGroup['params']):
            # Algo
            p.data = p.data - dGroup['lr']*dGrads[i]
        for i,p in enumerate(gGroup['params']):
            # Algo
            p.data = p.data - gGroup['lr']*gGrads[i]

        return loss

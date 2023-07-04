from torch.optim import Optimizer 
import torch
import math

class AdamW(Optimizer):
    '''
    Implement Adam algorithm with weight decay fix
    Source: https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
    '''

    def __init__(self,params, lr=1e-3, betas=(0.9,0.99), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr<0.0 or not 0.0<=betas[0]<1.0 or not 0.0<=betas[0]<1.0 or eps<0:
            raise ValueError
        defaults = dict(lr=lr,betas=betas,eps=eps,
                        weight_decay=weight_decay,correct_bias=correct_bias)
        super(AdamW,self).__init__(params,defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # boilerplate
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                # Access current optimizer state
                state = self.state[p]

                # Initialization (check if state empty dict. len(dictInstance) gives number of keys)
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Read the last updated values of stored variables
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Algorithm
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Store the changes
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss


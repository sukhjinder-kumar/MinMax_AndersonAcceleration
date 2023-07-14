'''
Use case:
G = Generator()
D = Discriminator()
opt = ConOpt(G, D, gamma=0.1, lr=lr, device='cpu')
ConOpt.step(dLoss, GLoss)
'''

import torch
import torch.autograd as autograd


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


def Hvp_vec(grad_vec, params, vec, retain_graph=False):
    '''
    Parameters:
        - grad_vec: Tensor of which the Hessian vector product will be computed
        - params: list of params, w.r.t which the Hessian will be computed
        - vec: The "vector" in Hessian vector product
    return: Hessian vector product
    '''
    if torch.isnan(grad_vec).any():
        raise ValueError('Gradvec nan')
    if torch.isnan(vec).any():
        raise ValueError('vector nan')
        # zero padding for None
    grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec,
                              retain_graph=retain_graph, allow_unused=True)
    grad_list = []
    for i, p in enumerate(params):
        if grad_grad[i] is None:
            grad_list.append(torch.zeros_like(p).view(-1))
        else:
            grad_list.append(grad_grad[i].contiguous().view(-1))
    hvp = torch.cat(grad_list)
    if torch.isnan(hvp).any():
        raise ValueError('hvp Nan')
    return hvp


class ConOpt(object):
    def __init__(self, gen, disc, gamma=0.1, lr=1e-3, device='cpu'):
        self.gen = gen
        self.disc = disc
        self.max_params_list = list(self.disc.parameters())
        self.min_params_list = list(self.gen.parameters())
        self.max_params = self.disc.parameters()  # max_params
        self.min_params = self.gen.parameters()  # min_params
        # print(type(max_params))
        # print(type(min_params))

        self.state = {}
        # self.state['max_params'] = list(max_params)
        # self.state['min_params'] = list(min_params)
        self.state['lr'] = lr
        self.state['device'] = device
        self.state['gamma'] = gamma
        self.t = 0
        self.beta = 0.9
        self.eps = 1e-8
        self.moment_x = 0
        self.moment_y = 0
        self.moment = 0
        self.opt_disc = torch.optim.RMSprop(self.disc.parameters(), lr=lr)
        self.opt_gen = torch.optim.RMSprop(self.gen.parameters(), lr=lr)

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def step(self, d_loss, g_loss):
        # Fetching parameters.
        gamma = self.state['gamma']
        d_param = [d for d in self.disc.parameters()]
        g_param = [g for g in self.gen.parameters()]

        # Get grad()
        grad_d = torch.autograd.grad(d_loss, self.disc.parameters(),
                                     retain_graph=True, create_graph=True)
        grad_g = torch.autograd.grad(g_loss, self.gen.parameters(),
                                     retain_graph=True, create_graph=True)
        grad = grad_d + grad_g
        param = d_param + g_param
        reg_d = 0.5*sum(torch.sum(g**2)
                        for g in self.disc.parameters())  # grad_d)
        reg_g = 0.5*sum(torch.sum(g**2)
                        for g in self.gen.parameters())  # grad_g)
        # reg = 0.5*sum(torch.sum(g**2) for g in grad)
        # reg = 0.5*sum(torch.sum(torch.square(g)) for g in grad)

        Jgrad_d = torch.autograd.grad(reg_d, d_param)
        Jgrad_g = torch.autograd.grad(reg_g, g_param)
        # Jgrad = torch.autograd.grad(reg, param)
        # Jgrad_d = Jgrad[0:len(grad_d)]
        # Jgrad_g = Jgrad[len(grad_d):]
        # del Jgrad

        final_grad_d = [g + gamma * j for j, g in zip(Jgrad_d, grad_d)]
        final_grad_g = [g + gamma * j for j, g in zip(Jgrad_g, grad_g)]

        for p, g in zip(self.disc.parameters(), final_grad_d):
            p.grad = g.detach()

        for p, g in zip(self.gen.parameters(), final_grad_g):
            p.grad = g.detach()

        self.opt_disc.step()
        self.opt_gen.step()

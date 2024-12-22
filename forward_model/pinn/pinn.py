import torch
from   torch import nn

import math

class PINN(nn.Module):
    def __init__(self, fc_list, lambda1, lambda2):
        super().__init__()

        self.activation = nn.Tanh() # for some reason Tanh is the only one that works

        self.lambda1 = lambda1
        self.lambda2 = lambda2 # viscosity coefficient

        self.loss_function = nn.MSELoss()

        self.fc_list = fc_list # first, last = input_size, output_size
        self.network = nn.ModuleList([nn.Linear(fc_list[i], fc_list[i + 1]) for i in range(len(fc_list) - 1)])

        # works better than nothing or nn.init.xavier_normal_
        for i in range(len(self.fc_list) - 1):
            nn.init.kaiming_normal_(self.network[i].weight.data)

    def forward(self, x):
        output = x

        for i in range(len(self.fc_list) - 2):
            output = self.network[i](output)
            output = self.activation(output)

        # even though result is between -1 and 1 is looks like it's better not to apply Tanh
        return self.network[-1](output)

    # t = 0 -> F(x, t) = sin(-pi * x)
    def error_initial(self, state):
        return self.loss_function(self.forward(state), (-math.pi * state[:, 0].clone().detach()).sin().unsqueeze(1))

    # x = ±1 -> F(x, t) = 0
    # also F(0, t) = 0 but it's more fair not to include x=0 here
    def error_border(self, state):
        return self.loss_function(self.forward(state), torch.zeros(state.size(0), 1))

    # ∂t*u + λ1*u*∂x*u - λ2*∂xx*u tells how far is model from truth
    def error_pde(self, state):
        # it's better not to touch anything under here because it always fails to work normally

        state_ = state.clone()
        state_.requires_grad = True

        u = self.forward(state_)

        dx_dt = torch.autograd.grad(
            u, state_,
            torch.ones([state_.shape[0],1]),
            retain_graph=True, create_graph=True
        )[0]
        dxx_dtt = torch.autograd.grad(
            dx_dt, state_,
            torch.ones(state_.shape),
            create_graph=True
        )[0]

        u_t = dx_dt[:,[1]]
        u_x = dx_dt[:,[0]]
        u_xx = dxx_dtt[:,[0]]

        residue = u_t + self.lambda1 * u * u_x - self.lambda2 * u_xx
        self.last_pde_error = self.loss_function(residue, torch.zeros_like(residue))
        return self.loss_function(residue, torch.zeros_like(residue))

    def error_total(self, state_beginning, state_border, state_general):
        return 0.35 * self.error_initial(state_beginning) + 0.35 * self.error_border(state_border) + self.error_pde(state_general)

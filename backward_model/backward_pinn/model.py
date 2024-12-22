import torch
from torch import nn
import math

class PINN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.activation = nn.Tanh()

        self.lambda1 = nn.Parameter(torch.Tensor([math.log(1.1)]), requires_grad=True)
        self.lambda2 = nn.Parameter(torch.Tensor([math.log(0.01)]), requires_grad=True)

        self.loss_function = nn.MSELoss()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, *input_data):
        return self.network(torch.stack(input_data, dim=1)).flatten()

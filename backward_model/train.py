import torch
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import random_split

import math
import numpy as np

from utils.loadmat import loadmat

from pathlib import Path

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-l', '--load-model')
parser.add_argument('-o', '--output', default='./model.pt', help='where to save the model')
parser.add_argument('-t', '--optimizer', default='lbfgs', choices=['adam', 'lbfgs'], help='optimizer to use: Adam or LBFGS')

args = parser.parse_args()

LOAD_PATH = Path(args.load_model) if args.load_model is not None else None
SAVE_PATH = Path(args.output)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

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

def pde_loss(model, t, x, u):
    t.requires_grad = True
    x.requires_grad = True

    u_pred = model(t, x)

    u_t = torch.autograd.grad(
        u_pred, t,
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u_pred, x,
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]

    residue = u_t + torch.exp(model.lambda1) * u_pred * u_x - torch.exp(model.lambda2) * u_xx
    pde_loss = residue.pow(2).mean()
    u_loss = (u_pred - u).pow(2).mean()
    return u_loss + pde_loss, u_loss, pde_loss

TRAINING_FRACTION = 0.95

class CustomDataset(Dataset):
    def __init__(self, t, x, u):
        super(CustomDataset).__init__()
        self.t = t
        self.x = x
        self.u = u

    def __len__(self):
        return len(self.t)

    def __getitem__(self, item):
        return self.t[item], self.x[item], self.u[item]

x_vector, t_vector, u, _ = loadmat()
x, t = np.meshgrid(x_vector, t_vector)
x, t, u = (torch.Tensor(v.flatten()) for v in (x, t, u))

dataset = CustomDataset(t, x, u)

training_dataset, validation_dataset = random_split(dataset,
                                                    (int(len(dataset) * TRAINING_FRACTION),
                                                     len(dataset) - int(len(dataset) * TRAINING_FRACTION)),
                                                    generator=torch.Generator().manual_seed(238)
                                                    )

# %%
EPOCHS = 10000
LEARNING_RATE = 0.015
SCHEDULER_RATE = 20
GAMMA = 0.9999
BATCH_SIZE = 1024

def train_batch(model, t, x, u, optimizer):
    loss, pde_error, u_error = pde_loss(model, t, x, u)
    optimizer.zero_grad()
    # maybe retain_graph=False
    loss.backward()
    optimizer.step()

    return loss.item(), pde_error, u_error

def validate_batch(model, t, x, u):
    return pde_loss(model, t, x, u)


def train_adam(model, training_loader, validation_loader, optimizer, scheduler):
    training_loss_history, validation_loss_history = [], []

    for epoch in range(EPOCHS):
        training_loss = 0
        validation_loss = 0
        training_cnt = 0
        validation_cnt = 0

        for i, (t, x, u) in enumerate(training_loader):
            loss, _, _ = train_batch(model, t, x, u, optimizer)
            training_loss += loss
            training_cnt += 1

        for i, (t, x, u) in enumerate(validation_loader):
            loss, pde_error, u_error = validate_batch(model, t, x, u)
            validation_loss += loss
            validation_cnt += 1

        training_loss_history.append(training_loss / training_cnt)
        validation_loss_history.append(validation_loss / validation_cnt)

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, PDE error: {pde_error.item()}, U Error: {u_error.item()}")
            print(
                f"Epoch {epoch}, lambda1: {torch.exp(model.lambda1).item()}, lambda2: {torch.exp(model.lambda2).item()}")
            # plot_t()

        if epoch % SCHEDULER_RATE == 0:
            scheduler.step()

epoch = 0

def train_LBFGS(model, dataset: CustomDataset):
    optim = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=50000,
        max_eval=50000,
        history_size=50,
        tolerance_grad=1e-6,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe"
    )
    t = dataset.t.to(device)
    x = dataset.x.to(device)
    u = dataset.u.to(device)

    def closure():
        global epoch
        epoch += 1
        optim.zero_grad()
        loss, pde_error, u_error = pde_loss(model, t, x, u)
        loss.backward()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, PDE error: {pde_error.item()}, U Error: {u_error.item()}")
            print(
                f"Epoch {epoch}, lambda1: {torch.exp(model.lambda1).item()}, lambda2: {torch.exp(model.lambda2).item()}")
        return loss

    optim.step(closure)

if LOAD_PATH is not None:
    model = torch.load(LOAD_PATH)
else:
    model = PINN(2, 1, 20)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f'training with optimizer: {args.optimizer}')

if args.optimizer == 'adam':
    train_adam(model, training_loader, validation_loader, optimizer, scheduler)
else:
    train_LBFGS(model, dataset)

print('\n=== training finished ===\n')
print(f'saving to: {SAVE_PATH}')

torch.save(model.cpu(), SAVE_PATH)

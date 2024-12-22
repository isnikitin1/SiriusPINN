import torch

import numpy as np

from argparse import ArgumentParser

from backward_pinn.model import PINN
from backward_pinn.dataset import load_dataset

from torch.utils.data import random_split, DataLoader

parser = ArgumentParser()
parser.add_argument('-o', '--output', default='./model.pt', help='where to save the model')
parser.add_argument('-t', '--optimizer', default='lbfgs', choices=['adam', 'lbfgs'], help='optimizer to use: Adam or LBFGS')
parser.add_argument('-d', '--device', default="cuda:0" if torch.cuda.is_available() else "cpu", help="device to train on (cuda or cpu)")
parser.add_argument('-e', '--epochs', default=10000, type=int)
parser.add_argument('--initial-lr', default=None, type=float) # Defaults to 0.015 for Adam and 1.0 with LBFGS 
parser.add_argument('--batch-size', default=1024, type=int)
parser.add_argument('--scheduler-gamma', default=0.999975, type=float)


args = parser.parse_args()

device = torch.device(args.device)

TRAINING_FRACTION = 0.95


# %%
EPOCHS = args.epochs
GAMMA = args.scheduler_gamma
BATCH_SIZE = args.batch_size

epoch = 0

dataset = load_dataset()
training_dataset, validation_dataset = random_split(dataset,
                                                    (int(len(dataset) * TRAINING_FRACTION),
                                                     len(dataset) - int(len(dataset) * TRAINING_FRACTION)),
                                                    generator=torch.Generator().manual_seed(238)
                                                    )

training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

model = PINN(2, 1, 20).to(device)

print(f'training with optimizer: {args.optimizer}')

if args.optimizer == 'adam':
    from backward_pinn.train.adam import train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr or 0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    train(model, EPOCHS, training_loader, validation_loader, optimizer, scheduler)
else:
    from backward_pinn.train.lbfgs import train
    optim = torch.optim.LBFGS(
        model.parameters(),
        lr=args.initial_lr or 1.0,
        max_iter=50000,
        max_eval=50000,
        history_size=50,
        tolerance_grad=1e-6,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe"
    )
    train(model, optim, dataset, device)

print('\n=== training finished ===\n')
print(f'saving to: {args.output}')

torch.save(model.cpu(), args.output)

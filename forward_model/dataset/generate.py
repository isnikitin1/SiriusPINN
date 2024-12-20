import torch

from   torch.utils.data import Dataset
from   torch.utils.data import DataLoader
from   torch.utils.data import random_split

class CustomDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()

        self.num_samples = num_samples

        num_beginning = int(num_samples)
        num_border = int(num_samples)
        num_random = int(num_samples)
        num_origin = int(num_samples)

        self.origin_x = (torch.rand((num_origin, 1)) * 2 - 1) / 11
        self.origin_t = torch.sqrt(torch.rand((num_origin, 1)))
        self.origin_all = torch.cat([self.origin_t_x, self.origin_t], dim=1)

        self.random_x = torch.rand((num_random, 1)) * 2 - 1
        self.random_t = torch.rand((num_random, 1))
        self.random_all = torch.cat([self.random_x, self.random_t], dim=1)

        self.border_x = torch.randint(0, 2, (num_border, 1), dtype=torch.float32) * 2 - 1
        self.border_t = torch.rand((num_border, 1))
        self.border_all = torch.cat([self.border_x, self.border_t], dim=1)

        self.beginning_x = torch.rand((num_beginning, 1)) * 2 - 1
        self.beginning_t = torch.zeros(num_beginning, 1)
        self.beginning_all = torch.cat([self.beginning_x, self.beginning_t], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return self.beginning_all[item], self.border_all[item], self.random_all[item], self.origin_all[item]

def generate(dataset_size, training_fraction, batch_size):
    dataset = CustomDataset(dataset_size)

    training_dataset, validation_dataset = random_split(dataset,
        (int(len(dataset) * training_fraction), len(dataset) -  int(len(dataset) * training_fraction)),
        generator=torch.Generator().manual_seed(238)
    )

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return training_loader, validation_loader

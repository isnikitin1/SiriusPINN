import torch

class BurgersDataset:
    def __init__(self, num_samples):
        super().__init__()

        self.num_samples = num_samples

        num_beginning = int(num_samples * 0.1)
        num_border = int(num_samples * 0.1)
        num_random = int(num_samples * 0.8)

        self.beginning_x = torch.rand((num_beginning, 1)) * 2 - 1
        self.beginning_t = torch.zeros(num_beginning, 1)
        self.beginning_all = torch.cat([self.beginning_x, self.beginning_t], dim=1)

        self.border_x = torch.randint(0, 2, (num_border, 1), dtype=torch.float32) * 2 - 1
        self.border_t = torch.rand((num_border, 1))
        self.border_all = torch.cat([self.border_x, self.border_t], dim=1)

        self.random_x = torch.rand((num_random, 1)) * 2 - 1
        self.random_t = torch.rand((num_random, 1))
        self.random_all = torch.cat([self.random_x, self.random_t], dim=1)

        p_beginning = torch.randperm(num_beginning)
        self.beginning_all = self.beginning_all[p_beginning]

        p_border = torch.randperm(num_border)
        self.border_all = self.border_all[p_border]

        p_random = torch.randperm(num_random)
        self.random_all = self.random_all[p_random]

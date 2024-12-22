import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.loadmat import loadmat

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

def load_dataset():
    x_vector, t_vector, u, _ = loadmat()
    x, t = np.meshgrid(x_vector, t_vector)
    x, t, u = (torch.Tensor(v.flatten()) for v in (x, t, u))

    return CustomDataset(t, x, u)
 

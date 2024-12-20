import torch

import numpy as np
from utils import loadmat
from utils import plot_t
from utils import diff_u
from utils import diff_logu

FILENAME = "./trained_burgers"

def main():
    x, t, exact, X = loadmat.loadmat()

    solver = torch.load(FILENAME)

    U = [[0]*256 for _ in range(100)]
    for i in range(100):
        for j in range(256):
            U[i][j] = solver.forward(torch.Tensor([x[j], t[i]]).view([1,2])).item()

    plot_t.plot_T(U, X)
    diff_u.diff_U(U, exact)
    diff_logu.diff_logU(U, exact)

if __name__ == "__main__":
    main()

import math
import torch
from   pinn      import pinn
from   dataset   import generate
from   optimizer import optimizer

EPOCHS = 150000

FILENAME = "trained_burgers"

LEARNING_RATE = 0.001
SCHEDULER_RATE = 100
GAMMA = 0.99

DATASET_SIZE = 20000
TRAINING_FRACTION = 0.98

TRAINING_LIMIT = 2e-5

FC_LAYERS = [2, 16, 64, 32, 64, 16, 1]

LAMBDA1 = 1
LAMBDA2 = 0.01 / math.pi

def main():
    solver = pinn.PINN(FC_LAYERS, LAMBDA1, LAMBDA2)

    training_loader = generate.BurgersDataset(int(DATASET_SIZE * TRAINING_FRACTION))
    validation_loader = generate.BurgersDataset(DATASET_SIZE - int(DATASET_SIZE * TRAINING_FRACTION))

    if optimizer.optimize(solver, training_loader, validation_loader, LEARNING_RATE, EPOCHS, SCHEDULER_RATE, GAMMA, TRAINING_LIMIT):
        print("Target reached")
    else:
        print("Unsuccessful training")

    torch.save(solver, FILENAME)

    print("Model saved as {0}".format(FILENAME))

if __name__ == "__main__":
    main()

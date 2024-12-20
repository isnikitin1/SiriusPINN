import math
import torch
import optimizer.optimizer
from   pinn      import pinn
from   dataset   import generate
from   optimizer import optimizer

EPOCHS = 15000

FILENAME = "trained_burgers"

LEARNING_RATE = 0.0015
SCHEDULER_RATE = 25
GAMMA = 0.9975

DATASET_SIZE = 5000
TRAINING_FRACTION = 0.975
BATCH_SIZE = 1000

TRAINING_LIMIT = 0.0001

FC_LAYERS = [2, 128, 64, 32, 16, 8, 4, 2, 1]

LAMBDA1 = 1
LAMBDA2 = 0.01 / math.pi

def main():
    solver = pinn.PINN(FC_LAYERS, LAMBDA1, LAMBDA2)

    training_loader, validation_loader = generate.generate(DATASET_SIZE, TRAINING_FRACTION, BATCH_SIZE)

    if optimizer.optimize(solver, training_loader, validation_loader, LEARNING_RATE, EPOCHS, SCHEDULER_RATE, GAMMA, TRAINING_LIMIT):
        print("Target reached")
    else:
        print("Unsuccessful training")

    torch.save(solver, FILENAME)

    print("Model saved as {0}".format(FILENAME))

if __name__ == "__main__":
    main()

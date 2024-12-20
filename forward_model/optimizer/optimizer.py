from   torch     import optim
from   optimizer import trainer

def optimize(model, training_loader, validation_loader, learning_rate, epochs, scheduler_rate, gamma, training_limit):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return trainer.train(model, training_loader, validation_loader, optimizer, scheduler, epochs, scheduler_rate, training_limit)

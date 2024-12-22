from optimizer import batch

def train(model, training_loader, validation_loader, optimizer, scheduler, epochs, scheduler_rate, training_limit):
    training_loss_history, validation_loss_history = [], []

    for epoch in range(epochs):
        training_loss = batch.train_batch(model, training_loader.beginning_all, training_loader.border_all, training_loader.random_all, optimizer)
        validation_loss = batch.validate_batch(model, validation_loader.beginning_all, validation_loader.border_all, validation_loader.random_all)

        training_loss_history.append(training_loss)
        validation_loss_history.append(validation_loss)

        if training_limit > model.last_pde_error.item():
            return True

        if epoch % 10 == 0:
            print("Epoch {0:6d}, PDE MSE: {1}".format(epoch, model.last_pde_error.item()))

        if epoch % scheduler_rate == 0:
            scheduler.step()

    return False

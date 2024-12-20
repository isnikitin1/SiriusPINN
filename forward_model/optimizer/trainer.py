from optimizer import batch

def train(model, training_loader, validation_loader, optimizer, scheduler, epochs, scheduler_rate, training_limit):
    training_loss_history, validation_loss_history = [], []

    for epoch in range(epochs):
        training_loss = 0
        validation_loss = 0
        training_cnt = 0
        validation_cnt = 0

        for i, (batch_beginning, batch_border, batch_general, batch_origin) in enumerate(training_loader):
            training_loss += batch.train_batch(model, batch_beginning, batch_border, batch_general, batch_origin, optimizer)
            training_cnt += 1

        for i, (batch_beginning, batch_border, batch_general, batch_origin) in enumerate(validation_loader):
            validation_loss += batch.validate_batch(model, batch_beginning, batch_border, batch_origin, batch_general)
            validation_cnt += 1

        training_loss_history.append(training_loss / training_cnt)
        validation_loss_history.append(validation_loss / validation_cnt)

        if training_limit > model.last_pde_error.item():
            return True

        if epoch % 10 == 0:
            print("Epoch {0:6d}, PDE MSE Error: {1}".format(epoch, model.last_pde_error.item()))

        if epoch % scheduler_rate == 0:
            scheduler.step()

    return False

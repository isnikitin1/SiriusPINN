from ..loss import pde_loss
from ..report import report_epoch

def train_batch(model, t, x, u, optimizer):
    loss, pde_error, u_error = pde_loss(model, t, x, u)
    optimizer.zero_grad()
    # maybe retain_graph=False
    loss.backward()
    optimizer.step()

    return loss.item(), pde_error, u_error

def validate_batch(model, t, x, u):
    return pde_loss(model, t, x, u)

def train(model, epochs, training_loader, validation_loader, optimizer, scheduler):
    training_loss_history, validation_loss_history = [], []

    for epoch in range(epochs):
        training_loss = 0
        validation_loss = 0
        training_cnt = 0
        validation_cnt = 0

        for i, (t, x, u) in enumerate(training_loader):
            loss, _, _ = train_batch(model, t, x, u, optimizer)
            training_loss += loss
            training_cnt += 1

        for i, (t, x, u) in enumerate(validation_loader):
            loss, pde_error, u_error = validate_batch(model, t, x, u)
            validation_loss += loss
            validation_cnt += 1

        training_loss_history.append(training_loss / training_cnt)
        validation_loss_history.append(validation_loss / validation_cnt)

        report_epoch(epoch, pde_error.item(), u_error.item(), model)

        scheduler.step()

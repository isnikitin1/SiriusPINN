from ..loss import pde_loss
from ..report import report_epoch

def train(model, optim, dataset, device):

    t = dataset.t.to(device)
    x = dataset.x.to(device)
    u = dataset.u.to(device)

    epoch = 0

    def closure():
        nonlocal epoch
        epoch += 1
        optim.zero_grad()
        loss, pde_error, u_error = pde_loss(model, t, x, u)
        loss.backward()
        if epoch % 100 == 0:
            report_epoch(epoch, pde_error.item(), u_error.item(), model)
        return loss

    optim.step(closure)


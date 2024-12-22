import torch

def pde_loss(model, t, x, u):
    t.requires_grad = True
    x.requires_grad = True

    u_pred = model(t, x)

    u_t = torch.autograd.grad(
        u_pred, t,
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u_pred, x,
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]

    residue = u_t + torch.exp(model.lambda1) * u_pred * u_x - torch.exp(model.lambda2) * u_xx
    pde_loss = residue.pow(2).mean()
    u_loss = (u_pred - u).pow(2).mean()
    return u_loss + pde_loss, u_loss, pde_loss

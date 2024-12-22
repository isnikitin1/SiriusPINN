import torch

def report_epoch(epoch, pde_loss, u_loss, model):
        print(f"Epoch {epoch:6d}, PDE error: {pde_loss:e}, U Error: {u_loss:e}")
        print(
            f"              λ₁ = {torch.exp(model.lambda1).item()}, λ₂ = {torch.exp(model.lambda2).item()}")

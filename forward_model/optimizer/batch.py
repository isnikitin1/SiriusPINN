def train_batch(model, batch_beginning, batch_border, batch_general, optimizer):
    loss = model.error_total(batch_beginning, batch_border, batch_general)
    optimizer.zero_grad()
    loss.backward(retain_graph=False) # maybe retain_graph=False
    optimizer.step()

    return loss.item()

def validate_batch(model, batch_beginning, batch_border, batch_general):
    loss = model.error_total(batch_beginning, batch_border, batch_general)

    return loss.item()

import torch
import torch.nn as nn

# Example target and prediction tensors
ytgt = torch.Tensor([0])
ypred = torch.Tensor([-10.3969])  # Raw logits (before applying sigmoid)

# Loss function (Binary Cross Entropy with logits)
loss_fn = nn.BCEWithLogitsLoss()

# Calculate the loss
loss = loss_fn(ypred, ytgt)

print(f'Loss: {loss.item()}')
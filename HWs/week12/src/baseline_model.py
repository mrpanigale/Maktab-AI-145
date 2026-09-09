"""
In this file, a baseline model is to be tested, and the loss is to be calculated manually.
"""

# ===========imports==========
import torch
import torch.nn as nn
import numpy as np

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========variables==========

g = torch.Generator().manual_seed(0)
synthetic_data = torch.randn((1, 28, 28), dtype=torch.float32, generator=g).to(DEVICE)
synthetic_label = torch.zeros(size=(1,), dtype=torch.long).to(DEVICE)
# ===========Base Model==========
simple_layer = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
simple_layer.to(DEVICE)

# ===========logit==========
logits = simple_layer(synthetic_data)

# ===========probability==========
proba = nn.Softmax(dim=1)(logits)

# ===========loss==========
loss_fn = nn.CrossEntropyLoss()
loss = -(torch.log(proba[0, synthetic_label]))
loss_torch = loss_fn(logits, synthetic_label)

print(f"Manual Loss: {loss.item()}\n\
1/10 log(worst loss for 10 class): {-(np.log(0.1))}\n\
 PyTorch loss: {loss_torch.item()}")

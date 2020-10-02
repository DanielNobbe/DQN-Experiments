import torch
from torch import nn

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = torch.Tensor(x) # Seems like this does nothing, even when numpy gets passed into Q
        layer_1 = self.l1(x)
        hidden = nn.functional.relu(layer_1) # Apply activation function as function rather than later
        output = self.l2(hidden)
        return output



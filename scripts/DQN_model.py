import torch
from torch import nn

# Code structure based on lab4 from Reinforcement Learning course at University of Amsterdam


class QNetwork(nn.Module):
    
    def __init__(self, in_size, out_size, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(in_size, num_hidden)
        self.l2 = nn.Linear(num_hidden, out_size)

    def forward(self, x):
        if len(x.shape) == 0:
            x = x.unsqueeze(dim=0)
        x = torch.tensor(x) # Seems like this does nothing, even when numpy gets passed into Q
        layer_1 = self.l1(x)
        hidden = nn.functional.relu(layer_1) # Apply activation function as function rather than later
        output = self.l2(hidden)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        return output



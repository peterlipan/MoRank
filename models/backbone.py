import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, n_layers=2, dropout=0.0, activation='relu'):
        super(MLP, self).__init__()
        
        # Define activation function
        self.activation = self.get_activation_fn(activation)
        
        # Create layers
        layers = []
        for i in range(n_layers):
            in_dim = d_in if i == 0 else d_hid
            out_dim = d_hid if i < n_layers - 1 else d_out
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:  # Apply activation and dropout only to hidden layers
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout))
        
        # Combine layers into a sequential module
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    @staticmethod
    def get_activation_fn(name):
        if name == "relu":
            return nn.ReLU()
        elif name == "elu":
            return nn.ELU()
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

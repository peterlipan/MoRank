import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, n_layers=2, dropout=0.0, activation='relu'):
        super(MLP, self).__init__()

        self.activation = self.get_activation_fn(activation)

        layers = []
        for i in range(n_layers):
            in_dim = d_in if i == 0 else d_hid
            out_dim = d_hid if i < n_layers - 1 else d_out
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        elif name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.1)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

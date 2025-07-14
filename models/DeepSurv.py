import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import MLP
from .utils import ModelOutputs
from pycox.models.loss import CoxPHLoss


class CoxSurvLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.cph = CoxPHLoss()
        self.eps = eps
    def forward(self, outputs, data):
        # directly predict the risk factors
        risk_clamped = outputs.risk.clamp(min=self.eps) # for numerical stability
        return self.cph(risk_clamped.log(), data['duration'], data['event'])


class DeepSurv(nn.Module):
    def __init__(self, args):
        super(DeepSurv, self).__init__()
        
        self.encoder = MLP(
            d_in=args.n_features,
            d_hid=args.d_hid,
            d_out=args.d_hid,
            n_layers=args.n_layers, # Number of encoder layers
            dropout=args.dropout,
            activation=args.activation
        )
        self.n_classes = args.n_classes
        self.head = nn.Linear(args.d_hid, 1)
        self.criterion = CoxSurvLoss()
    
    def forward(self, data):
        features = self.encoder(data['data'])
        logits = self.head(features)
        risk = torch.sigmoid(logits).view(-1) 
        return ModelOutputs(features=features, logits=logits, risk=risk)

    def compuite_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )


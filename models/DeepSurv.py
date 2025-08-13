import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs
from pycox.models.loss import CoxPHLoss
import torchtuples as tt
from pycox.models import CoxPH


class CoxSurvLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.cph = CoxPHLoss()
        self.eps = eps
    def forward(self, outputs, data):
        # directly predict the risk factors
        # risk_clamped = outputs.risk.clamp(min=self.eps) # for numerical stability
        return self.cph(outputs.risk, data['duration'], data['event'])


class DeepSurv(nn.Module):
    def __init__(self, args):
        super(DeepSurv, self).__init__()
        
        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, 1)
        self.criterion = CoxSurvLoss()
    
    def forward(self, data):
        features = self.encoder(data['data'])
        logits = self.head(features)
        risk = logits.view(-1)
        return ModelOutputs(features=features, logits=logits, risk=risk)        



    def compute_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )


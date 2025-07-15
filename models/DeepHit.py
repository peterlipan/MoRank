import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import MLP
from .utils import ModelOutputs
from pycox.models.loss import DeepHitSingleLoss


class DeepHitsurvLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dhl = DeepHitSingleLoss(alpha=0.5, sigma=0.5)
    
    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)

        rank_mat = ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i
        return rank_mat

    def forward(self, outputs, data):
        # directly predict the risk factors
        rank_mat = self.pair_rank_mat(data['label'], data['event'])
        return self.dhl(outputs.logits, data['label'], data['event'], rank_mat)


class DeepHit(nn.Module):
    def __init__(self, args):
        super(DeepHit, self).__init__()
        
        self.encoder = MLP(
            d_in=args.n_features,
            d_hid=args.d_hid,
            d_out=args.d_hid,
            n_layers=args.n_layers, # Number of encoder layers
            dropout=args.dropout,
            activation=args.activation
        )
        self.head = nn.Linear(args.d_hid, args.n_classes)
        self.n_classes = args.n_classes
        self.criterion = DeepHitsurvLoss()
    
    def forward(self, data):
        features = self.encoder(data['data'])
        logits = self.head(features)
        pmf = F.softmax(logits, dim=-1)
        fht = torch.argmax(pmf, dim=1)
        prob_at_fht = torch.gather(pmf, 1, fht.unsqueeze(1)).squeeze(1)  # probability at first hitting time
        risk = 1 / (fht + 1).float()  # risk is normalized by the first hitting time
        return ModelOutputs(features=features, logits=logits, pmf=pmf, risk=risk)

    def compuite_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )

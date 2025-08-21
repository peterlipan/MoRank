import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
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

    def forward(self, logits, event, duration, label):
        # directly predict the risk factors
        rank_mat = self.pair_rank_mat(duration, event)
        return self.dhl(logits, label, event, rank_mat)


class DeepHit(nn.Module):
    def __init__(self, args):
        super(DeepHit, self).__init__()
        
        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, args.n_classes)
        self.n_classes = args.n_classes
        self.criterion = DeepHitsurvLoss()

    def forward(self, x):
        # the full forward pass
        features = self.encoder(x)
        logits = self.head(features)
        pmf = F.softmax(logits, dim=1)  # probability mass function
        fht = torch.argmax(pmf, dim=1)  # first hitting time
        prob_at_fht = torch.gather(pmf, 1, fht.unsqueeze(1)).squeeze(1)  # probability at first hitting time
        risk = -fht
        cdf = torch.cumsum(pmf, dim=1)  # cumulative distribution function
        cdf = cdf.clamp(min=0, max=1)  # ensure
        surv = 1. - cdf
        return ModelOutputs(features=features, logits=logits, pmf=pmf, risk=risk, cdf=cdf, surv=surv, fht=fht, prob_at_fht=prob_at_fht)
    
    def get_risk_logits(self, features):
        logits = self.head(features)
        pmf = F.softmax(logits, dim=1)  # probability mass function
        fht = torch.argmax(pmf, dim=1)  # first hitting time
        risk = -fht
        return ModelOutputs(logits=logits, risk=risk)

    def compute_loss(self, logits, event, duration, label):
        return self.criterion(
            logits,
            event,
            duration,
            label
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, logits, event, duration, label, bs):
        # directly predict the risk factors
        rank_mat = self.pair_rank_mat(duration, event)
        return self.dhl(logits, label, event, rank_mat)


class DeepHit(nn.Module):
    def __init__(self, d_hid, n_classes):
        super(DeepHit, self).__init__()

        self.d_hid = d_hid
        self.n_classes = n_classes
        self.head = nn.Linear(self.d_hid, self.n_classes)
        self.criterion = DeepHitsurvLoss()

    def forward(self, features):
        logits = self.head(features)
        pmf = F.softmax(logits, dim=1)
        fht = torch.argmax(pmf, dim=1) 
        prob_at_fht = torch.gather(pmf, 1, fht.unsqueeze(1)).squeeze(1)
        risk = -fht
        cdf = torch.cumsum(pmf, dim=1)
        cdf = cdf.clamp(min=0, max=1)
        surv = 1. - cdf
        return ModelOutputs(features=features, logits=logits, pmf=pmf, risk=risk, cdf=cdf, surv=surv, fht=fht, prob_at_fht=prob_at_fht)

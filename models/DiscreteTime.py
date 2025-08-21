import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs
from pycox.models.loss import DeepHitSingleLoss


class NllSurvLoss(nn.Module):
    def __init__(self, sigma=0.5, lambda_rank=0.1):
        super().__init__()
        self.sigma = sigma
        self.lambda_rank = lambda_rank

    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)
        return ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i

    def rank_loss_on_risk(self, risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        diff_risk = risk.view(-1, 1) - risk.view(1, -1)
        rank_mat = self.pair_rank_mat(durations, events)
        loss = rank_mat * torch.exp(-diff_risk / self.sigma)
        return loss.sum() / (rank_mat.sum() + 1e-6)
    
    @staticmethod
    def nll_loss(logits, label, event, alpha=0.4, eps=1e-7):
        batch_size = len(label)
        idx_duration = label.view(batch_size, 1) # ground truth bin, 1,2,...,k
        event = event.view(batch_size, 1).float()

        hazards = F.sigmoid(logits)
        surv_prob = torch.cumprod(1 - hazards, dim=1)  # survival probability

        # without padding, S(0) = S[0], h(0) = h[0]
        surv_prob_padded = torch.cat([torch.ones_like(event), surv_prob], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
        # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
        #h[y] = h(1)
        #S[1] = S(1)
        uncensored_loss = -event * (torch.log(torch.gather(surv_prob_padded, 1, idx_duration).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, idx_duration).clamp(min=eps)))
        censored_loss = - (1 - event) * torch.log(torch.gather(surv_prob_padded, 1, idx_duration+1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1-alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss

    def forward(self, logits, event, duration, label):
        hazards = F.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        risk = - torch.sum(surv, dim=1)
        rank_loss = self.rank_loss_on_risk(risk, duration, event)
        surv_loss = self.nll_loss(logits, label, event)
        return self.lambda_rank * rank_loss + surv_loss


class DiscreteTime(nn.Module):
    def __init__(self, args):
        super(DiscreteTime, self).__init__()
        
        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, self.n_classes)
        self.criterion = NllSurvLoss()

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        hazards = F.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        risk = - torch.sum(surv, dim=1)  # risk is the negative cumulative survival
        return ModelOutputs(features=features, logits=logits, hazards=hazards, surv=surv, risk=risk)
    
    def get_risk_logits(self, features):
        logits = self.head(features)
        hazards = F.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        risk = - torch.sum(surv, dim=1)
        return ModelOutputs(logits=logits, risk=risk)

    def compute_loss(self, logits, event, duration, label):
        return self.criterion(
            logits,
            event,
            duration,
            label
        )

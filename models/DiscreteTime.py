import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ModelOutputs


class NllSurvLoss(nn.Module):
    def __init__(self, sigma=0.5, lambda_rank=0.1):
        super().__init__()
        self.sigma = sigma
        self.lambda_rank = lambda_rank
    
    def rank_loss_on_cdf(self, logits, durations, events):
        hazards = torch.sigmoid(logits)             # [N, T]
        surv = torch.cumprod(1 - hazards, dim=1)    # survival
        cdf = 1 - surv                              # failure CDF
        
        # CDF at observed times
        N, T = cdf.shape
        y = torch.zeros_like(cdf).scatter(1, durations.view(-1,1), 1.)
        F_mat = cdf.matmul(y.T)                     # [N,N]
        diag_F = F_mat.diag().view(1, -1)           # F_j(T_j)
        R = F_mat - diag_F                          # R_ij
        
        # Ranking loss
        rank_mat = self.pair_rank_mat(durations, events)
        loss = rank_mat * torch.exp(-R / self.sigma)
        return loss.sum() / (rank_mat.sum() + 1e-6)


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
        surv_prob_padded = torch.cat([torch.ones_like(event), surv_prob], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
        uncensored_loss = -event * (torch.log(torch.gather(surv_prob_padded, 1, idx_duration).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, idx_duration).clamp(min=eps)))
        censored_loss = - (1 - event) * torch.log(torch.gather(surv_prob_padded, 1, idx_duration+1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1-alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss

    def forward(self, logits, event, duration, label, bs):
        hazards = F.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        risk = - torch.sum(surv, dim=1)
        # rank_loss = self.rank_loss_on_risk(risk, duration, event)
        rank_loss = self.rank_loss_on_cdf(logits, label, event)
        # if bs not specified, calculate the surv loss only on the current batch
        if bs is not None:
            logits = logits[:bs]
            event = event[:bs]
            label = label[:bs]
        surv_loss = self.nll_loss(logits, label, event)
        return self.lambda_rank * rank_loss + surv_loss


class DiscreteTime(nn.Module):
    def __init__(self, d_hid, n_classes):
        super(DiscreteTime, self).__init__()

        self.head = nn.Linear(d_hid, n_classes)
        self.criterion = NllSurvLoss()

    def forward(self, features):
        logits = self.head(features)
        hazards = F.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        risk = - torch.sum(surv, dim=1)  # risk is the negative cumulative survival
        return ModelOutputs(features=features, logits=logits, hazards=hazards, surv=surv, risk=risk)

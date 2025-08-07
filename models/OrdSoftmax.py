import math
import torch
import torch.nn as nn
from scipy.special import binom
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
    
    def ordinal_cox_cumsum_loss(self, logits: torch.Tensor, labels: torch.Tensor, direction: str = 'both', eps: float = 1e-8) -> torch.Tensor:
        """
        Class-space Cox-like loss using cumulative softmax-like structure.

        - For each sample with true class y:
            - If direction == 'left': risk set is [0..y]
            - If direction == 'right': risk set is [y..C-1]
            - If direction == 'both': both left and right cumulative risk

        This mimics the cumulative sum approximation in CoxPH.

        Args:
            logits: (B, C) cosine similarity logits
            labels: (B,) class indices
            direction: 'left', 'right', or 'both'
        """
        B, C = logits.shape
        log_probs = []

        if direction in ['left', 'both']:
            # Flip logits so cumsum simulates leftward accumulation: [0, ..., y]
            flipped = torch.flip(logits, dims=[1])  # (B, C)
            exp_flipped = torch.exp(flipped)
            cumsum_exp_flipped = torch.cumsum(exp_flipped, dim=1)
            log_cumsum_left = torch.log(cumsum_exp_flipped + eps)
            log_cumsum_left = torch.flip(log_cumsum_left, dims=[1])  # flip back
            log_h = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            log_risk = log_cumsum_left.gather(1, labels.unsqueeze(1)).squeeze(1)
            log_probs.append(log_risk - log_h)

        if direction in ['right', 'both']:
            exp_logits = torch.exp(logits)
            cumsum_exp = torch.cumsum(exp_logits, dim=1)
            log_cumsum_right = torch.log(cumsum_exp + eps)
            log_h = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            log_risk = log_cumsum_right.gather(1, labels.unsqueeze(1)).squeeze(1)
            log_probs.append(log_risk - log_h)

        loss = sum(log_probs)
        return loss.mean()


    @staticmethod
    def cosine_monotonicity_loss(weights: torch.Tensor, anchor_idx: int = 0) -> torch.Tensor:
        """
        Encourages monotonic increase of cosine similarity between w_anchor and other class weights.

        Only adjacent comparisons are used to keep it lightweight and promote monotonicity.

        Args:
            weights: (C, D) normalized class weight matrix
            anchor_idx: index of anchor weight (default: 0)

        Returns:
            Scalar monotonicity loss
        """
        weights = F.normalize(weights, p=2, dim=1)  # (C, D)
        anchor = weights[anchor_idx]                # (D,)
        cos_sims = weights @ anchor                 # (C,)

        # Remove anchor from sequence
        idx = torch.arange(len(cos_sims))
        mask = idx != anchor_idx
        cos_sims_no_anchor = cos_sims[mask]

        # Adjacent differences: cos_sim[i+1] - cos_sim[i] should be > 0
        diffs = cos_sims_no_anchor[1:] - cos_sims_no_anchor[:-1]
        loss = F.relu(-diffs).mean()  # Penalize non-increasing behavior

        return loss


    def forward(self, outputs, data):
        # directly predict the risk factors
        rank_mat = self.pair_rank_mat(data['label'], data['event'])
        return self.dhl(outputs.logits, data['label'], data['event'], rank_mat)


class OrdSoftmax(nn.Module):
    def __init__(self, args):
        super(OrdSoftmax, self).__init__()

        self.encoder = MLP(
            d_in=args.n_features,
            d_hid=args.d_hid,
            d_out=args.d_hid,
            n_layers=args.n_layers, # Number of encoder layers
            dropout=args.dropout,
            activation=args.activation
        )
        self.head = nn.Linear(args.d_hid, args.n_classes, bias=False)
        self.n_classes = args.n_classes
        self.criterion = DeepHitsurvLoss()

    def forward(self, data):
        features = self.encoder(data['data'])
        feature_norm = F.normalize(features, p=2, dim=1)
        w_norm = F.normalize(self.head.weight, p=2, dim=1)
        cos_sim = feature_norm @ w_norm.t()  # Ordinal softmax logits
        logits = cos_sim
        pmf = F.softmax(logits, dim=1)  # probability mass function
        fht = torch.argmax(pmf, dim=1)  # first hitting time
        prob_at_fht = torch.gather(pmf, 1, fht.unsqueeze(1)).squeeze(1)  # probability at first hitting time
        risk = -fht
        cdf = torch.cumsum(pmf, dim=1)  # cumulative distribution function
        cdf = cdf.clamp(min=0, max=1)  # ensure
        surv = 1. - cdf
        return ModelOutputs(features=features, logits=logits, pmf=pmf, 
                            risk=risk, cdf=cdf, surv=surv, fht=fht, 
                            prob_at_fht=prob_at_fht, cos_sim=cos_sim,
                            weights=self.head.weight)

    def compute_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )

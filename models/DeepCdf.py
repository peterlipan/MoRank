import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import MLP
from .utils import ModelOutputs


class CDFLoss(nn.Module):
    def __init__(self, monotonic_weight: float = 0.1, sigma: float = 0.5, margin= 0.1, beta=0.1):
        """
        Args:
            num_bins: total number of time bins (T)
            monotonic_weight: weight for monotonicity regularization
        """
        super().__init__()
        self.sigma = sigma  # for rank loss
        self.monotonic_weight = monotonic_weight
        self.margin = margin
        self.beta = beta

    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)

        rank_mat = ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i
        return rank_mat

    def rank_loss_on_risk(self, risk: torch.Tensor, idx_durations: torch.Tensor, events: torch.Tensor, sigma: float) -> torch.Tensor:
        n = risk.shape[0]
        
        # Pairwise differences in risk scores
        diff_risk = risk.view(-1, 1) - risk.view(1, -1)  # [batch, batch]
        
        rank_mat = self.pair_rank_mat(idx_durations, events)
        assert rank_mat.sum() > 0, "Rank matrix should not be empty"
        
        # Exponential penalty for incorrect ordering
        loss = rank_mat * torch.exp(-diff_risk / sigma)

        loss = loss.sum() / rank_mat.sum()

        return loss

    def forward(self, outputs, data):
        """
        Args:
            F_pred: [B, T] predicted cumulative incidence function F(t)
            event: [B] binary (1 = uncensored, 0 = censored)
            label: [B] integer bin index (0-based) of event/censoring time

        Returns:
            scalar loss
        """
        F_pred = outputs.cdf  # [B, T] predicted CDF
        risk = outputs.risk  # [B] risk factors (not used in this loss)
        label = data['duration']  # [B] integer bin index (0-based)
        event = data['event']  # [B] binary (1 = uncensored,
        B, T = F_pred.shape
        device = F_pred.device

        # --- Create target CDF ---
        target = torch.zeros_like(F_pred)  # [B, T]
        mask = torch.zeros_like(F_pred, dtype=torch.bool)  # [B, T]

        for i in range(B):
            t_idx = label[i].item()
            if event[i] == 1:
                target[i, t_idx:] = 1
                mask[i, :] = True
            else: 
                mask[i, :t_idx + 1] = True

        # --- Binary Cross-Entropy Loss ---
        bce = F.binary_cross_entropy(F_pred[mask], target[mask], reduction='mean')

        # --- Monotonicity Penalty: ReLU(F(t) - F(t+1)) ---
        # encourage F(t+1) > F(t)
        monotonic_penalty = F.relu(F_pred[:, :-1] - F_pred[:, 1:] + self.margin).mean()
        bce += self.monotonic_weight * monotonic_penalty
        # bce += self.rank_loss_on_risk(risk, data['duration'], data['event'], self.sigma)
        # regularization on the biases
        bce += self.beta * torch.mean(outputs.biases ** 2)

        return bce


class DeepCdf(nn.Module):
    def __init__(self, args):
        super(DeepCdf, self).__init__()
        
        self.encoder = MLP(
            d_in=args.n_features,
            d_hid=args.d_hid,
            d_out=args.d_hid,
            n_layers=args.n_layers, # Number of encoder layers
            dropout=args.dropout,
            activation=args.activation
        )
        self.n_classes = args.n_classes
        self.head = nn.Linear(args.d_hid, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.n_classes)) 
        self.criterion = CDFLoss()
    
    def forward(self, data):
        features = self.encoder(data['data'])
        proj = self.head(features)
        logits = proj + self.biases.view(1, -1)  # add biases for each time point
        cdf = torch.sigmoid(logits)  # cumulative distribution function
        risk = proj.view(-1)  # risk is the same as logits
        return ModelOutputs(features=features, logits=logits, cdf=cdf, risk=risk, biases=self.biases)

    def compuite_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )


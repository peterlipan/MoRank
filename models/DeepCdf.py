import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import MLP
from .utils import ModelOutputs


class CDFLoss(nn.Module):
    def __init__(self, 
                 cdf_weight: float = 1.0,
                 monotonic_weight: float = 0.1,
                 sigma: float = 0.5,
                 margin: float = 1.,
                 beta: float = 0.,
                 rank_weight: float = 0.1,
                 gamma: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.cdf_weight = cdf_weight
        self.monotonic_weight = monotonic_weight
        self.margin = margin
        self.beta = beta
        self.rank_weight = rank_weight
        self.gamma = gamma

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

    def forward(self, outputs, data):
        F_pred = outputs.cdf
        logits = outputs.logits
        risk = outputs.risk
        label = data['label']
        event = data['event']
        durations = data.get('duration', label)

        B, T = F_pred.shape
        device = F_pred.device

        time_idx = torch.arange(T, device=device).view(1, -1)
        label_exp = label.view(-1, 1)
        target = (time_idx >= label_exp).float()
        distance = (time_idx - label_exp).abs().float()
        decay_weight = torch.exp(-self.gamma * distance)

        mask = torch.where(event.view(-1, 1).bool(),
                           torch.ones_like(target).bool(),
                           (time_idx <= label_exp))

        decay_weight = decay_weight * mask.float()
        cdf_loss = ((F_pred - target) ** 2 * decay_weight)[mask].sum() / (decay_weight[mask].sum() + 1e-6)

        monotonic_penalty = F.relu(F_pred[:, :-1] - F_pred[:, 1:] + self.margin).mean()

        # Apply weight norm regularization (on projection weights)
        projection_weight = outputs.projection_weight
        weight_reg = projection_weight.norm(2)

        rank_loss = self.rank_loss_on_risk(risk, label, event)

        total_loss = (
            self.cdf_weight * cdf_loss +
            self.monotonic_weight * monotonic_penalty +
            self.rank_weight * rank_loss +
            self.beta * weight_reg
        )

        return total_loss


class DeepCdf(nn.Module):
    def __init__(self, args):
        super(DeepCdf, self).__init__()
        
        self.encoder = MLP(
            d_in=args.n_features,
            d_hid=args.d_hid,
            d_out=args.d_hid,
            n_layers=args.n_layers,
            dropout=args.dropout,
            activation=args.activation
        )
        self.n_classes = args.n_classes
        self.head = nn.Linear(args.d_hid, 1, bias=False)

        # Linearly spaced fixed biases (learnable can be tested separately)
        time_idx = torch.linspace(-1, 1, self.n_classes)
        self.biases = nn.Parameter(time_idx, requires_grad=False)

        self.criterion = CDFLoss()
        self.temp = 1.5

    def forward(self, data):
        features = self.encoder(data['data'])
        proj = self.head(features)  # [B, 1]
        logits = proj + self.biases.view(1, -1)
        cdf = torch.sigmoid(logits / self.temp).clamp(0, 1)
        risk = proj.view(-1)
        surv = 1. - cdf

        return ModelOutputs(features=features,
                            logits=logits,
                            cdf=cdf,
                            risk=risk,
                            surv=surv,
                            biases=self.biases,
                            projection_weight=self.head.weight.view(-1))

    def compuite_loss(self, outputs, data):
        return self.criterion(outputs, data)

    def project_2d(self, data):
        features = self.encoder(data['data'])
        proj = self.head(features)

        w = self.head.weight.squeeze(0)
        w = w / w.norm()
        v = torch.randn_like(w)
        v = v - (v @ w) * w
        v = v / v.norm()

        proj_x = features @ w
        proj_y = features @ v

        return ModelOutputs(x=proj_x, y=proj_y)

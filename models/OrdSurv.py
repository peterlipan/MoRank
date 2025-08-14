import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class CDFLoss(nn.Module):
    def __init__(self, 
                 cdf_weight: float = 1.,
                 monotonic_weight: float = 0.1,
                 sigma: float = 0.5,
                 margin: float = 0.,
                 rank_weight: float = 0.1,
                 gamma: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.cdf_weight = cdf_weight
        self.monotonic_weight = monotonic_weight
        self.margin = margin
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
    
    def calibration_loss(self, F_pred: torch.Tensor, label: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Compare predicted marginal CDF against empirical event histogram.
        Only use uncensored samples (event == 1).
        """
        B, T = F_pred.shape
        device = F_pred.device

        with torch.no_grad():
            hist = torch.zeros(T, device=device)
            mask = (event == 1)
            times = label[mask]
            hist.scatter_add_(0, times, torch.ones_like(times, dtype=torch.float32))

        hist = hist / (hist.sum() + 1e-6)  # Normalize empirical histogram
        pred_mass = F_pred[event == 1]     # Only use uncensored predictions
        marginal_pred = pred_mass.sum(dim=0)
        marginal_pred = marginal_pred / (marginal_pred.sum() + 1e-6)

        return F.mse_loss(marginal_pred, hist)

    def forward(self, outputs, data):
        F_pred = outputs.cdf
        risk = outputs.risk
        # cos_sim = outputs.cos_sim
        label = data['label']
        event = data['event']

        _, T = F_pred.shape
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
        # bce_loss = F.binary_cross_entropy(F_pred, target, reduction='none')
        # cdf_loss = (bce_loss * decay_weight)[mask].sum() / (decay_weight[mask].sum() + 1e-6)

        monotonic_penalty = F.relu(F_pred[:, :-1] - F_pred[:, 1:] + self.margin).mean()

        rank_loss = self.rank_loss_on_risk(risk, label, event)
        # calibration = self.calibration_loss(F_pred, label, event)

        total_loss = (
            self.cdf_weight * cdf_loss +
            self.monotonic_weight * monotonic_penalty +
            self.rank_weight * rank_loss
        )

        return total_loss


class OrdSurv(nn.Module):
    def __init__(self, args):
        super(OrdSurv, self).__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, 1, bias=False)

        self.biases = nn.Parameter(torch.linspace(-1, 1, self.n_classes), requires_grad=True)
        # self.biases = nn.Parameter(torch.randn(self.n_classes), requires_grad=True)

        self.criterion = CDFLoss()
        self.scaler = nn.Parameter(1. * torch.ones(1))  # Scale for logits

    def forward(self, data):

        features = self.encoder(data['data'])
        proj = self.head(features)
        # multiply the biases by the magnitude of features and weights
        biases = self.biases.view(1, -1) * (features.norm(dim=1, keepdim=True) * self.head.weight.norm())
        logits = proj + biases # if self.training else proj + self.biases.view(1, -1)
        cdf = torch.sigmoid(logits * self.scaler)
        risk = proj.view(-1)  # [B * T]
        surv = 1. - cdf  # [B, T]

        # features = self.encoder(data['data'])
        # w = self.head.weight.squeeze(0)
        # w = F.normalize(w, dim=0, p=2)  # Normalize the weight vector
        # features = F.normalize(features, dim=1, p=2)  # Normalize the features
        # proj = features @ w  # [B, 1]
        # proj = proj.view(-1, 1)  # Reshape to [B, 1]
        # logits = proj + self.biases.view(1, -1)  # [B, T]
        # cdf = torch.sigmoid(logits* self.scaler)
        # risk = proj.view(-1)  # [B * T]
        # surv = 1. - cdf  # [B, T]

        # features = self.encoder(data['data'])
        # proj = self.head(features)  # [B, 1]
        # logits = proj + self.biases.view(1, -1)  # [B, T]
        # cdf = torch.sigmoid(logits * self.scaler)  # [B, T]
        # risk = proj.view(-1)  # [B * T]
        # surv = 1. - cdf

        # features_norm = F.normalize(features, dim=1, p=2)
        # weight_norm = F.normalize(self.head.weight.squeeze(0), dim=0, p=2)
        # cos_sim = features_norm @ weight_norm  # [B, 1]
        # cos_sim = cos_sim.view(-1)
        # risk = cos_sim


        return ModelOutputs(features=features,
                            logits=logits,
                            cdf=cdf,
                            risk=risk,
                            # cos_sim=cos_sim,
                            surv=surv,
                            biases=self.biases,
                            projection_weight=self.head.weight.view(-1))

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)

    def project_2d(self, data):
        features = self.encoder(data['data'])
        features = F.normalize(features, dim=1, p=2)
        proj = self.head(features)

        w = self.head.weight.squeeze(0)
        w = F.normalize(w, dim=0, p=2)
        v = torch.randn_like(w)
        v = v - (v @ w) * w
        v = F.normalize(v, dim=0, p=2)

        proj_x = features @ w
        proj_y = features @ v

        return ModelOutputs(x=proj_x, y=proj_y)

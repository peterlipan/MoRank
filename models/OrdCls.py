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
                 margin: float = 0.1,
                 rank_weight: float = 0.1,
                 gamma: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.cdf_weight = cdf_weight
        self.monotonic_weight = monotonic_weight
        self.margin = margin
        self.rank_weight = rank_weight
        self.gamma = gamma

    def rank_loss_on_score(self, score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label_i = label.view(-1, 1)
        label_j = label.view(1, -1)
        rank_mat = (label_i < label_j).float()
        diff_score = score.view(-1, 1) - score.view(1, -1)
        loss = rank_mat * torch.exp(-diff_score / self.sigma)
        return loss.sum() / (rank_mat.sum() + 1e-6)

    def forward(self, outputs, data):
        F_pred = outputs.cdf
        score = outputs.score

        label = data['label']

        _, C = F_pred.shape # C: number of classes
        device = F_pred.device

        label_idx = torch.arange(C, device=device).view(1, -1)
        label_exp = label.view(-1, 1)
        target = (label_idx >= label_exp).float()
        distance = (label_idx - label_exp).abs().float()
        decay_weight = torch.exp(-self.gamma * distance)

        cdf_loss = ((F_pred - target) ** 2 * decay_weight).sum() / (decay_weight.sum() + 1e-6)

        monotonic_penalty = F.relu(F_pred[:, :-1] - F_pred[:, 1:] + self.margin).mean()

        rank_loss = self.rank_loss_on_score(score, label)

        total_loss = (
            self.cdf_weight * cdf_loss +
            self.monotonic_weight * monotonic_penalty +
            self.rank_weight * rank_loss
        )

        return total_loss


class OrdCls(nn.Module):
    def __init__(self, args):
        super(OrdCls, self).__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, 1, bias=False)

        self.biases = nn.Parameter(torch.linspace(-1, 1, self.n_classes), requires_grad=True)

        self.criterion = CDFLoss()
        self.scaler = nn.Parameter(1. * torch.ones(1))  # Scale for logits
        self.temp = nn.Parameter(torch.tensor(1.5))

    def forward(self, data):

        features = self.encoder(data['data'])
        w = self.head.weight.squeeze(0)
        w = F.normalize(w, dim=0, p=2)  # Normalize the weight vector
        features = F.normalize(features, dim=1, p=2)  # Normalize the features
        proj = features @ w  # [B, 1]
        proj = proj.view(-1, 1) 
        # proj = self.head(features)

        logits = proj + self.biases.view(1, -1)  # [B, T]
        # logits = logits * self.scaler  # Scale the logits
        cdf = torch.sigmoid(logits * self.scaler)
        score = proj.view(-1)  # countinuous prediction
        # y_pred is the first bin that has a cdf > 0.5
        y_pred = (logits >= 0.).float().argmax(dim=1).clamp(min=1, max=self.n_classes - 3)  # [B]
        # print(f"y_pred: {y_pred}, cdf: {cdf}")


        return ModelOutputs(features=features,
                            logits=logits,
                            cdf=cdf,
                            score=score,
                            y_pred=y_pred,
                            biases=self.biases)

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

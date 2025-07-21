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
                 margin: float = 0.,
                 beta: float = 0.,
                 pmf_weight: float = 0.,
                 rank_weight: float = 0.1,
                 gamma: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.cdf_weight = cdf_weight
        self.monotonic_weight = monotonic_weight
        self.margin = margin
        self.beta = beta
        self.pmf_weight = pmf_weight
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
        """
        Args:
            outputs: ModelOutputs object with cdf/logits/risk/biases
            data: dict with 'label', 'event', 'duration'
        Returns:
            scalar loss
        """
        F_pred = outputs.cdf        # [B, T]
        logits = outputs.logits     # [B, T]
        risk = outputs.risk         # [B]
        label = data['label']       # [B]
        event = data['event']       # [B]
        durations = data.get('duration', label)  # fallback if no separate durations

        B, T = F_pred.shape
        device = F_pred.device

        # --- Vectorized CDF target & mask ---
        time_idx = torch.arange(T, device=device).view(1, -1)  # [1, T]
        label_exp = label.view(-1, 1)                          # [B, 1]
        target = (time_idx >= label_exp).float()              # [B, T]
        distance = (time_idx - label_exp).abs().float()  # |t - label|
        decay_weight = torch.exp(-self.gamma * distance)
        mask = torch.where(event.view(-1, 1).bool(),
                           torch.ones_like(target).bool(),
                           (time_idx <= label_exp))           # [B, T]

        # --- Binary CDF Loss (MSE on masked entries) ---
        decay_weight = decay_weight * mask.float()
        cdf_loss = ((F_pred - target) ** 2 * decay_weight)[mask].sum() / (decay_weight[mask].sum() + 1e-6)

        # cdf_loss = F.binary_cross_entropy(F_pred[mask], target[mask])


        # --- Monotonicity Penalty: ReLU(F(t) - F(t+1) + margin) ---
        monotonic_penalty = F.relu(F_pred[:, :-1] - F_pred[:, 1:] + self.margin).mean()

        # --- Bias Regularization ---
        bias_reg = self.beta * outputs.biases.square().mean()

        # # --- PMF Supervision via CrossEntropyLoss ---
        pmf_logits = logits.clone()
        pmf_logits[:, 1:] = logits[:, 1:] - logits[:, :-1]
        pmf_logits[:, 0] = logits[:, 0]  # first bin stays the same
        pmf_loss = F.cross_entropy(pmf_logits[event == 1], label[event == 1]) if (event == 1).any() else 0.0


        rank_loss = self.rank_loss_on_risk(risk, label, event)


        # --- Final Loss ---
        total_loss = (
            self.cdf_weight * cdf_loss +
            self.monotonic_weight * monotonic_penalty +
            self.pmf_weight * pmf_loss +
            self.rank_weight * rank_loss +
            self.beta * bias_reg
        )

        return total_loss



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
        # self.biases = nn.Parameter(torch.zeros(self.n_classes))
        time_idx = torch.linspace(-1, 1, self.n_classes)
        self.biases = nn.Parameter(time_idx, requires_grad=True)  # initialize biases to linearly spaced values
        self.criterion = CDFLoss()
        self.temp = 1.5
    
    def forward(self, data):
        # features = self.encoder(data['data'])
        # logits = self.head(features)
        # # apply sigmoid then add biases for stability
        # proj = F.sigmoid(logits / self.temp)  # [B, 1]
        # cdf = proj + self.biases.view(1, -1)  # add biases for each time point
        # cdf = cdf.clamp(min=0, max=1)  # ensure CDF is in [0, 1]
        # risk = logits.view(-1)  # risk is the same as logits

        features = self.encoder(data['data'])
        proj = self.head(features)  # [B, 1]
        logits = proj + self.biases.view(1, -1)  # add biases for each time point
        cdf = F.sigmoid(logits / self.temp)  # apply sigmoid then add biases for stability
        cdf = cdf.clamp(min=0, max=1)
        risk = proj.view(-1)  # risk is the same as logits
        return ModelOutputs(features=features, logits=logits, cdf=cdf, risk=risk, biases=self.biases)

    def compuite_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )

    def project_2d(self, data):
        features = self.encoder(data['data'])  # [B, D]
        proj = self.head(features)  # [B, 1]

        # Get weight vector (projection direction)
        w = self.head.weight.squeeze(0)  # [D]
        w = w / w.norm()  # normalize

        # Get a perpendicular vector to w (in 2D: rotate 90°)
        # In D > 2, take arbitrary orthogonal direction (e.g., Gram-Schmidt or fixed)
        # For visualization, we just need any consistent orthogonal direction
        # Let's pick a stable perpendicular vector using Gram-Schmidt
        v = torch.randn_like(w)
        v = v - (v @ w) * w  # remove projection on w
        v = v / v.norm()

        # Project features onto w (scalar) and v (perpendicular)
        proj_x = (features @ w)  # [B]
        proj_y = (features @ v)  # [B]

        return ModelOutputs(x=proj_x, y=proj_y)


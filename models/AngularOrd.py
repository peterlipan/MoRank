import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .backbone import MLP
from .utils import ModelOutputs
from pycox.models.loss import DeepHitSingleLoss



class ChebOrdinalWeightGenerator(nn.Module):
    def __init__(self, n_classes, d_out, n_cheb=5):
        """
        n_classes: number of ordinal categories
        d_out: dimension of each weight vector
        n_cheb: number of Chebyshev basis functions
        """
        super().__init__()
        self.n_classes = n_classes
        self.d_out = d_out
        self.n_cheb = n_cheb

        # Raw angle parameters (to be transformed into ordered phi in [0, pi])
        self.raw_phi = nn.Parameter(torch.randn(n_classes))  # unconstrained
        # Chebyshev basis vectors a_n in R^d
        self.basis = nn.Parameter(torch.randn(n_cheb, d_out))  # [n_cheb, d_out]

        # Optional per-class scale (can be removed or fixed)
        self.scales = nn.Parameter(torch.ones(n_classes))

    def _get_ordered_phi(self):
        # Ensure 0 < phi_1 < ... < phi_K < pi
        softplus_phi = F.softplus(self.raw_phi)
        phi = torch.cumsum(softplus_phi, dim=0)
        phi = phi / phi[-1] * np.pi
        return phi  # [K]

    def _get_chebyshev_embeddings(self, phi):
        """
        Compute Chebyshev embedding matrix of shape [K, n_cheb]
        """
        # Outer product: [K, N] where each row is [cos(0*phi_k), cos(1*phi_k), ..., cos((N-1)*phi_k)]
        n = torch.arange(self.n_cheb, device=phi.device).view(1, -1)
        phi = phi.view(-1, 1)  # [K, 1]
        emb = torch.cos(n * phi)  # [K, n_cheb]
        return emb

    def forward(self):
        phi = self._get_ordered_phi()                           # [K]
        cheb_emb = self._get_chebyshev_embeddings(phi)          # [K, n_cheb]
        W = torch.matmul(cheb_emb, self.basis)                  # [K, d_out]
        W = self.scales.view(-1, 1) * W                         # Optional: scale per class
        return W  # class weight matrix [K, D]


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

    def forward(self, outputs, data):
        # directly predict the risk factors
        rank_mat = self.pair_rank_mat(data['label'], data['event'])
        return self.dhl(outputs.logits, data['label'], data['event'], rank_mat)


class AngularOrdRep(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = MLP(
            d_in=args.n_features,
            d_hid=args.d_hid,
            d_out=args.d_hid,
            n_layers=args.n_layers,
            dropout=args.dropout,
            activation=args.activation
        )
        self.n_classes = args.n_classes
        self.criterion = DeepHitsurvLoss()

        self.weight_generator = ChebOrdinalWeightGenerator(
            n_classes=args.n_classes,
            d_out=args.d_hid,
            n_cheb=5  # You can increase this for more expressive capacity
        )

    def _get_unit_orthogonal(self, base, v):
        base = base / (base.norm() + 1e-8)
        proj = (v @ base) * base
        ortho = v - proj
        return ortho / (ortho.norm() + 1e-8)

    def _get_class_weights(self):
        base = self.w0 / (self.w0.norm() + 1e-8)
        ortho = self._get_unit_orthogonal(base, self.v)
        W = torch.stack([
            self.scales[i] * (torch.cos(theta) * base + torch.sin(theta) * ortho)
            for i, theta in enumerate(self.angles)
        ], dim=0)  # [T, D]
        return W

    def forward(self, data):
        features = self.encoder(data['data'])  # [B, D]
        W = self._get_class_weights()          # [T, D]
        logits = features @ W.T    # [B, T]
        pmf = F.softmax(logits, dim=1)
        fht = torch.argmax(pmf, dim=1)
        prob_at_fht = torch.gather(pmf, 1, fht.unsqueeze(1)).squeeze(1)
        risk = -fht.float()
        cdf = torch.cumsum(pmf, dim=1).clamp(0, 1)
        surv = 1. - cdf

        return ModelOutputs(
            features=features, logits=logits, pmf=pmf, risk=risk,
            cdf=cdf, surv=surv, fht=fht, prob_at_fht=prob_at_fht
        )

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)


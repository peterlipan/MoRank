import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import MLP
from .utils import ModelOutputs
from pycox.models.loss import DeepHitSingleLoss


class DeepHitsurvLoss(nn.Module):
    def __init__(self, alpha=0.5, sigma=0.5):
        super().__init__()
        self.dhl = DeepHitSingleLoss(alpha=alpha, sigma=sigma)

    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)
        return ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i

    def forward(self, outputs, data):
        rank_mat = self.pair_rank_mat(data['label'], data['event'])
        return self.dhl(outputs.logits, data['label'], data['event'], rank_mat)


class OrdSoftmax(nn.Module):
    """
    Ordinal softmax - minibatch sample based class-wise ranking + inter-class separation (JS).
    Integrates with DeepHit survival loss (keeps your earlier pipeline).
    """
    def __init__(self, args, lambda_rank=0.5, lambda_js=0.5, m0=0.2, eps=1e-9):
        """
        lambda_rank: weight for class-wise ranking loss
        lambda_js: weight for inter-class separation (we subtract lambda_js * mean_JS)
        m0: base margin per ordinal step (margin = m0 * |i-j|)
        """
        super().__init__()
        self.encoder = MLP(
            d_in=args.n_features,
            d_hid=args.d_hid,
            d_out=args.d_hid,
            n_layers=args.n_layers,
            dropout=args.dropout,
            activation=args.activation
        )
        self.head = nn.Linear(args.d_hid, args.n_classes, bias=False)  # you used bias=False earlier
        self.n_classes = args.n_classes
        self.criterion = DeepHitsurvLoss()
        self.lambda_rank = lambda_rank
        self.lambda_js = lambda_js
        self.m0 = float(m0)
        self.eps = eps

    def forward(self, data):
        features = self.encoder(data['data'])
        feature_norm = F.normalize(features, p=2, dim=1)
        w_norm = F.normalize(self.head.weight, p=2, dim=1)
        logits = feature_norm @ w_norm.t()  # (N, C) (cosine logits)
        pmf = F.softmax(logits, dim=1)
        fht = torch.argmax(pmf, dim=1)
        prob_at_fht = torch.gather(pmf, 1, fht.unsqueeze(1)).squeeze(1)
        cdf = torch.cumsum(pmf, dim=1).clamp(min=0., max=1.)
        surv = 1. - cdf
        risk = -fht
        return ModelOutputs(features=features, logits=logits, pmf=pmf,
                            risk=risk, cdf=cdf, surv=surv, fht=fht,
                            prob_at_fht=prob_at_fht, cos_sim=logits,
                            weights=self.head.weight)

    # ---------- Core: minibatch-based class statistics ----------
    @staticmethod
    def _per_class_stats(logits: torch.Tensor, pmf: torch.Tensor, labels: torch.Tensor, C: int, eps: float):
        """
        Vectorized per-class sums and counts and derived means.
        Returns:
            counts: (C,) float
            mean_logits: (C, C) where mean_logits[i, j] = mean over samples of class i of logit_j
            mean_pmf: (C, C) where mean_pmf[i, j] = mean over samples of class i of pmf_j
            present_mask: (C,) bool indicating classes present in batch
        """
        device = logits.device
        N = logits.shape[0]
        labels = labels.long().to(device)
        onehot = F.one_hot(labels, num_classes=C).float().to(device)  # (N, C)
        counts = onehot.sum(dim=0)  # (C,)
        counts_safe = counts.clamp_min(1.0)  # avoid div0

        # sum of logits per (true class i) over predicted class j -> (C, C)
        # sums[i, j] = sum_{n: y_n = i} logits[n, j]
        sums_logits = onehot.t() @ logits  # (C, C)
        mean_logits = sums_logits / counts_safe.view(C, 1)  # (C, C)

        # per-class average pmf distributions
        sums_pmf = onehot.t() @ pmf  # (C, C)
        mean_pmf = sums_pmf / counts_safe.view(C, 1)  # (C, C)

        present_mask = (counts > 0)
        # optionally, for absent classes mean rows contain arbitrary numbers - we'll skip them later
        return counts, mean_logits, mean_pmf, present_mask

    # ---------- Loss components ----------
    def _classwise_ranking_loss(self, mean_logits: torch.Tensor, counts: torch.Tensor, present_mask: torch.Tensor):
        """
        mean_logits: (C, C) matrix M where M[i,j] = E_{x in class i}[ logit_j(x) ]
        For present classes only, enforce: M[i,i] - M[i,j] >= m0 * |i-j|
        Use smooth softplus for hinge to keep gradients everywhere.
        """
        device = mean_logits.device
        C = mean_logits.shape[0]
        idx = torch.arange(C, device=device)
        dist = (idx.view(-1,1) - idx.view(1,-1)).abs().float()  # (C,C)
        margin = self.m0 * dist  # (C,C)

        # restrict to present classes
        present_idx = torch.nonzero(present_mask, as_tuple=False).squeeze(1)
        if present_idx.numel() == 0:
            return torch.tensor(0., device=device)

        Mp = mean_logits[present_idx][:, present_idx]  # (m, m)
        margin_p = margin[present_idx][:, present_idx]  # (m, m)
        # diag of Mp: (m,)
        diag = torch.diagonal(Mp, 0).view(-1, 1)  # (m,1)
        diff = diag - Mp  # (m, m) where row i col j = M[i,i] - M[i,j]
        # soft hinge
        loss_mat = F.softplus(margin_p - diff)  # (m,m)
        # zero diagonal (no self-comparison)
        eye = torch.eye(loss_mat.size(0), device=device)
        loss_mat = loss_mat * (1.0 - eye)

        # weight rows by inverse counts to prevent tiny classes dominating
        counts_present = counts[present_idx].clamp_min(1.0)
        row_weight = (1.0 / counts_present).view(-1, 1)  # (m,1)
        weighted = loss_mat * row_weight

        # normalize by number of active pairs
        denom = (loss_mat.numel() - loss_mat.shape[0])  # m*m - m
        rank_loss_val = weighted.sum() / (denom + self.eps)
        return rank_loss_val

    def _pairwise_js_among_mean_pmfs(self, mean_pmf: torch.Tensor, present_mask: torch.Tensor):
        """
        Compute pairwise JS among per-class mean pmfs for present classes only (vectorized).
        mean_pmf: (C, C), rows sum ~1 (where present)
        Returns mean_JS (scalar) averaged across ordered pairs (i<j)
        """
        device = mean_pmf.device
        C = mean_pmf.shape[0]
        present_idx = torch.nonzero(present_mask, as_tuple=False).squeeze(1)
        m = present_idx.numel()
        if m <= 1:
            return torch.tensor(0., device=device)

        P = mean_pmf[present_idx]  # (m, C)
        # clamp for stability
        P = P.clamp(min=1e-9)
        # compute pairwise JS: use broadcasting
        p = P.unsqueeze(1)   # (m,1,C)
        q = P.unsqueeze(0)   # (1,m,C)
        m_mix = 0.5 * (p + q)  # (m,m,C)
        # KL(p||m)
        kl_p_m = (p * (p.log() - m_mix.log())).sum(dim=-1)  # (m,m)
        kl_q_m = (q * (q.log() - m_mix.log())).sum(dim=-1)  # (m,m)
        js = 0.5 * (kl_p_m + kl_q_m)  # (m,m)
        # zero diagonal
        eye = torch.eye(m, device=device)
        js = js * (1.0 - eye)

        # weight by ordinal distance among present classes
        idx_map = present_idx  # these are original class indices
        dist = (idx_map.view(-1,1) - idx_map.view(1,-1)).abs().float()  # (m,m)
        w = dist
        w = w * (1.0 - eye)
        wsum = w.sum()
        if wsum.item() == 0:
            # fallback to unweighted mean
            mean_js = js.sum() / (m * (m - 1) + self.eps)
        else:
            mean_js = (w * js).sum() / (wsum + self.eps)
        return mean_js

    # ---------- Public compute_loss ----------
    def compute_loss(self, outputs: ModelOutputs, data: dict):
        """
        outputs: ModelOutputs from forward()
        data: dict with 'label' (N,) and 'event' etc. - we pass through DeepHit loss
        Returns: total loss tensor (ready for backward)
        """
        labels = data['label'].long().to(outputs.features.device)
        base_loss = self.criterion(outputs, data)  # e.g., DeepHit loss

        logits = outputs.logits  # (N, C)
        pmf = outputs.pmf      # (N, C)
        C = self.n_classes

        counts, mean_logits, mean_pmf, present_mask = self._per_class_stats(logits, pmf, labels, C, self.eps)

        rank_loss = self._classwise_ranking_loss(mean_logits, counts, present_mask)
        js_mean = self._pairwise_js_among_mean_pmfs(mean_pmf, present_mask)

        # We want to *maximize* JS, so subtract lambda_js * js_mean
        total = base_loss + self.lambda_rank * rank_loss - self.lambda_js * js_mean

        # stats for logging (detach)
        # return total (tensor); log bookkeeping externally if needed
        return total

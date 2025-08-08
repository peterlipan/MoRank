import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import MLP
from .utils import ModelOutputs

# -------------------------
# Numerical helpers (vectorized)
# -------------------------
def log_C_d_approx(kappa: torch.Tensor, d: int) -> torch.Tensor:
    v = torch.tensor(d / 2 - 1, dtype=kappa.dtype, device=kappa.device)
    half_d = torch.tensor(d / 2, dtype=kappa.dtype, device=kappa.device)

    logI = torch.zeros_like(kappa)

    small_mask = kappa < 1e-3
    large_mask = ~small_mask

    # Small kappa
    logI[small_mask] = (
        v * torch.log(kappa[small_mask] / 2.0 + 1e-12) - torch.lgamma(v + 1.0)
    )

    # Large kappa
    logI[large_mask] = (
        kappa[large_mask]
        - 0.5 * torch.log(2 * torch.pi * kappa[large_mask] + 1e-12)
    )

    return -v * torch.log(kappa + 1e-12) - half_d * torch.log(torch.tensor(2 * torch.pi, dtype=kappa.dtype, device=kappa.device)) - logI



def approx_A_d(kappa, d):
    """
    Approximate A_d(kappa) = I_{v+1}(kappa) / I_v(kappa)
    Use simple approximations:
      - for large kappa: A ~ 1 - (d-1)/(2*kappa)
      - for small kappa: A ~ kappa / d
    Vectorized.
    """
    kappa = kappa.clamp_min(1e-8)
    mask = (kappa > 50.0)
    A = torch.empty_like(kappa)
    A[mask] = 1.0 - (d - 1.0) / (2.0 * kappa[mask])
    A[~mask] = kappa[~mask] / float(d)
    return A


def vmf_logpdf_matrix(features, mus, kappas):
    """
    Vectorized vmf logpdf evaluator.
    features: (N, d) unit vectors
    mus: (K, d) unit prototypes
    kappas: (K,) positive
    returns L (N, K) with log p(x_n | mu_k, kappa_k) = log C_d(kappa_k) + kappa_k * (x_n dot mu_k)
    """
    # dot: (N, K)
    dot = features @ mus.t()
    # logC per class: (K,)
    logC = log_C_d_approx(kappas, features.shape[1]).view(1, -1)  # broadcast to (1,K)
    kappa_row = kappas.view(1, -1)  # (1, K)
    return logC + kappa_row * dot  # (N, K)


def kl_vmf_matrix(mus, kappas):
    """
    Vectorized symmetric KL proxy: compute KL(p_i || p_j) for all i,j using analytic approx.
    Returns KL_mat (K, K) where KL_mat[i,j] = KL(vMF_i || vMF_j)
    Uses formula:
      KL(p||q) = log C_d(kappa_p) - log C_d(kappa_q) + A_d(kappa_p) * (kappa_p - kappa_q * mu_q^T mu_p)
    """
    K, d = mus.shape
    # pairwise dot mu_q^T mu_p -> (K, K) where entry (i,j) = mu_i dot mu_j
    mu_dot = mus @ mus.t()  # (K, K)
    # logC vectors
    logC = log_C_d_approx(kappas, d)  # (K,)
    logC_p = logC.view(-1, 1)  # (K,1)
    logC_q = logC.view(1, -1)  # (1,K)
    # A_d(kappa_p) vector
    A = approx_A_d(kappas, d).view(-1, 1)  # (K,1)
    kappa_p = kappas.view(-1, 1)  # (K,1)
    kappa_q = kappas.view(1, -1)  # (1,K)
    kl = (logC_p - logC_q) + A * (kappa_p - kappa_q * mu_dot)
    return kl  # (K, K) KL(p_i || p_j)


# -------------------------
# Main Loss Module (vectorized)
# -------------------------
class VmfOrdinalLossVectorized(nn.Module):
    """
    Vectorized loss combining:
      - class-wise ranking (prototype-level, empirical/minibatch)
      - JS overlap estimated via minibatch samples (vectorized over class pairs)
    If a class has zero samples, JS for pairs involving that class falls back to symmetric KL (analytic).
    """
    def __init__(self, n_classes, feat_dim,
                 lambda_rank=1.0, lambda_js=1.0, m0=0.5,
                 js_mc_cap=128, eps=1e-9, device='cpu'):
        super().__init__()
        self.K = n_classes
        self.d = feat_dim
        self.lambda_rank = lambda_rank
        self.lambda_js = lambda_js
        self.m0 = m0
        self.js_mc_cap = js_mc_cap
        self.eps = eps
        self.device = device

    def forward(self, features, labels, mus_param, rho_kappa):
        """
        features: (N, d) unit-normalized
        labels: (N,) int in [0..K-1]
        mus_param: (K, d) prototype params (nn.Parameter)
        rho_kappa: (K,) raw params -> softplus -> kappa
        returns: total_loss (scalar), diagnostics dict
        """
        N = features.shape[0]
        K = self.K
        d = self.d
        device = features.device

        # Normalize and derive kappas
        features = F.normalize(features, dim=1)
        mus = F.normalize(mus_param, dim=1)
        kappas = F.softplus(rho_kappa).clamp_min(1e-6)  # (K,)

        # ---- Per-class empirical means and counts (vectorized) ----
        # one-hot labels (N, K)
        labels_onehot = F.one_hot(labels.long(), num_classes=K).float().to(device)  # (N, K)
        counts = labels_onehot.sum(dim=0).clamp_min(0.0)  # (K,)
        # sums: (K, d) = labels_onehot.T @ features
        sums = labels_onehot.t() @ features  # (K, d)
        # means: if count==0, replace with prototype mus to avoid NaNs (we'll mark these)
        counts_unsq = counts.unsqueeze(1).clamp_min(1.0)  # avoid div0
        means = sums / counts_unsq  # (K, d) - for zero counts will be zero; we'll fix below
        zero_mask = (counts == 0.0)
        if zero_mask.any():
            zero_mask_unsq = zero_mask.unsqueeze(1)  # (K,1)
            # where selects entries from 'mus' when zero_mask True, else from 'means'
            means = torch.where(zero_mask_unsq, mus, means)

        # ---- Class-wise ranking loss (vectorized) ----
        # We want for each true class i and any class j:
        #   margin_{i,j} = m0 * |i-j|
        #   enforce: E[s_i] - E[s_j] >= margin  -> hinge on margin - diff
        # where E[s_k on class i samples] ~= kappa_k * mu_k^T mean_i
        # Compute matrix dot_mu_mean: (K, K) where entry (i,j) = mu_j dot mean_i
        mu_dot_mean = means @ mus.t()  # (K, K): row i col j = mean_i · mu_j
        # E_si: vector length K: kappa_i * (mu_i · mean_i) -> use diag of mu_dot_mean
        mu_diag = torch.diagonal(mu_dot_mean, 0)  # (K,)
        E_si_vec = kappas * mu_diag  # (K,)
        # E_sj_on_i: matrix (K, K): for row i, col j = kappa_j * (mu_j · mean_i) = kappas.view(1,K) * mu_dot_mean
        E_sj_on_i = (kappas.view(1, K) * mu_dot_mean)  # (K, K)
        # diff matrix (K, K): row i col j = E_si - E_sj
        diff = E_si_vec.view(K, 1) - E_sj_on_i  # (K, K)
        # margin matrix M_ij = m0 * |i-j|
        idx = torch.arange(K, device=device)
        dist_mat = (idx.view(-1, 1) - idx.view(1, -1)).abs().float()  # (K, K)
        margin_mat = self.m0 * dist_mat
        # hinge on margin - diff, ignoring diagonal
        hinge = F.relu(margin_mat - diff)
        # hinge.fill_diagonal_(0.0)

        # Optionally weight by class frequency -> give less weight to missing classes
        # weight per row = 1 / max(1, count_i) to avoid tiny classes dominating
        eye = torch.eye(K, device=device, dtype=hinge.dtype)  # (K,K)

        # out-of-place zeroing of diagonal
        hinge = hinge * (1.0 - eye)   # zero diagonal safely
        # row weights (unchanged)
        count_weight = 1.0 / counts.clamp_min(1.0)  # (K,)
        row_weights = count_weight.view(K, 1)


        
        rank_loss = (hinge * row_weights).sum() / (K * K + self.eps)  # normalize

        # ---- JS overlap loss (vectorized MC estimate using minibatch samples) ----
        # Compute logpdf matrix L (N, K)
        L = vmf_logpdf_matrix(features, mus, kappas)  # (N, K)
        # Expand to Li and Lj for pairwise (N, K, K): Li[n,i,j] = L[n,i], Lj[n,i,j]=L[n,j]
        Li = L.unsqueeze(2)  # (N, K, 1)
        Lj = L.unsqueeze(1)  # (N, 1, K)
        Li = Li.expand(-1, K, K)  # (N,K,K)
        Lj = Lj.expand(-1, K, K)  # (N,K,K)

        # log m = log(0.5*exp(Li) + 0.5*exp(Lj)) = logsumexp(Li+ln0.5, Lj+ln0.5)
        ln_half = math.log(0.5)
        stacked = torch.stack([Li + ln_half, Lj + ln_half], dim=0)  # (2, N, K, K)
        # compute logm shape (N,K,K)
        logm = torch.logsumexp(stacked, dim=0)  # (N,K,K)

        # For KL(p_i || m), average over samples with label==i of (Li - logm)
        # Build mask M_i: (N, K) where M_i[n,i] = 1 if label[n]==i
        M = labels_onehot  # (N, K)
        # Expand for pair dims -> (N, K, K) where mask_for_row_i is 1 for samples that belong to row index
        mask_row = M.unsqueeze(2).expand(-1, K, K)  # (N,K,K) ; mask_row[n,i,j]=1 if sample n belongs to class i
        # sum_n (Li - logm) * mask_row over n, divide by count_i
        numer_pi = ((Li - logm) * mask_row).sum(dim=0)  # (K,K) sum over n
        # counts per class (K,) with clamp to 1 for zero to avoid div0 (we handle zeros later)
        counts_safe = counts.clamp_min(1.0)
        kl_pi_m = numer_pi / counts_safe.view(-1, 1)  # (K, K) per pair i,j (may be invalid if count_i was 0)
        # Similarly KL(p_j || m): average over samples with label==j -> mask_col
        mask_col = M.unsqueeze(1).expand(-1, K, K)  # (N,K,K) ; mask_col[n,i,j]=1 if sample n belongs to class j
        numer_pj = ((Lj - logm) * mask_col).sum(dim=0)  # (K,K)
        kl_pj_m = numer_pj / counts_safe.view(1, -1)  # (K,K)

        # JS estimate = 0.5 * (kl_pi_m + kl_pj_m)
        js_est = 0.5 * (kl_pi_m + kl_pj_m)  # (K,K) but entries where a class had zero samples are inaccurate
        # Detect pairs where either count_i == 0 or count_j == 0 -> fallback to sym KL analytic
        zero_row = (counts == 0.0)  # (K,)
        zero_col = zero_row
        zero_pair_mask = (zero_row.view(-1, 1) | zero_col.view(1, -1))  # (K,K) True when pair has missing samples

        if zero_pair_mask.any():
            # compute analytic KL matrix (K,K)
            kl_mat = kl_vmf_matrix(mus, kappas)  # KL(p_i || p_j)
            kl_sym = 0.5 * (kl_mat + kl_mat.t())  # symmetrized KL
            # Use kl_sym as fallback for js_est where zero_pair_mask is True
            js_est = torch.where(zero_pair_mask.to(js_est.device), kl_sym, js_est)

        # We want to penalize overlap, so larger JS (or KL fallback) should reduce loss.
        # Use weighted sum: weight by ordinal distance (|i-j|) to emphasize distant classes more
        weight = dist_mat.clone()  # if dist_mat is used elsewhere, clone to be safe
        weight = weight * (1.0 - eye)  # zero diagonal, out-of-place
        weight_sum = weight.sum() + self.eps
        js_loss = (weight * js_est).sum() / weight_sum

        # Final combined loss
        total_loss = self.lambda_rank * rank_loss + self.lambda_js * js_loss

        stats = {'rank_loss': rank_loss.detach(), 'js_loss': js_loss.detach(), 'total_loss': total_loss.detach()}
        return total_loss, stats


# -------------------------
# Example model (MLP backbone + vMF prototypes) - fits your earlier style
# -------------------------
class VmfOrdinalModel(nn.Module):
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
        # prototype parameters
        self.mu = nn.Parameter(torch.randn(args.n_classes, args.d_hid))
        self.rho_kappa = nn.Parameter(torch.ones(args.n_classes) * 0.1)
        self.n_classes = args.n_classes
        # loss module (instantiate later with device)
        self.criterion = VmfOrdinalLossVectorized(n_classes=args.n_classes,
                                                  feat_dim=args.d_hid,
                                                  lambda_rank=getattr(args, 'lambda_rank', 1.0),
                                                  lambda_js=getattr(args, 'lambda_js', 1.0),
                                                  m0=getattr(args, 'm0', 0.5),
                                                  js_mc_cap=getattr(args, 'js_mc_cap', 128),
                                                  device=getattr(args, 'device', 'gpu'))

    def forward(self, data):
        x = data['data']
        feats = self.encoder(x)
        feats = F.normalize(feats, dim=1)
        mus = F.normalize(self.mu, dim=1)
        kappas = F.softplus(self.rho_kappa).clamp_min(1e-6)
        logits = (kappas.unsqueeze(0) * (feats @ mus.t()))
        pmf = F.softmax(logits, dim=1)
        fht = torch.argmax(pmf, dim=1)
        prob_at_fht = torch.gather(pmf, 1, fht.unsqueeze(1)).squeeze(1)
        cdf = torch.cumsum(pmf, dim=1).clamp(0.0, 1.0)
        surv = 1.0 - cdf
        risk = -fht
        return ModelOutputs(features=feats, logits=logits, pmf=pmf, risk=risk, cdf=cdf, surv=surv,
                            fht=fht, prob_at_fht=prob_at_fht)

    def compute_loss(self, outputs, data):
        labels = data['label'].long().to(outputs.features.device)
        # vMF negative log-likelihood per sample can be approximated by cross entropy over logits
        # For strict correctness include log C_d(kappa) offsets; here we use CE over logits for simplicity
        nll = F.cross_entropy(outputs.logits, labels)

        ordinal_loss, stats = self.criterion(outputs.features, labels, self.mu, self.rho_kappa)
        # small kappa regularization to avoid runaway kappas
        reg_kappa = 1e-4 * (F.softplus(self.rho_kappa)).pow(2).mean()
        total = nll + ordinal_loss + reg_kappa
        return total


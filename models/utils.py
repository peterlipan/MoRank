import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelOutputs:
    def __init__(self, features=None, logits=None, **kwargs):
        self.dict = {'features': features, 'logits': logits}
        self.dict.update(kwargs)
    
    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
    
    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return str(self.dict)

    def __getattr__(self, key):
        try:
            return self.dict[key]
        except KeyError as e:
            raise AttributeError(f"'ModelOutputs' object has no attribute '{key}'") from e

    def __contains__(self, key):
        return key in self.dict


def get_model(args):
    if args.method.lower() == 'deephit':
        from .DeepHit import DeepHit
        return DeepHit(args)
    elif args.method.lower() == 'deepsurv':
        from .DeepSurv import DeepSurv
        return DeepSurv(args)
    elif args.method.lower() == 'discrete':
        from .DiscreteTime import DiscreteTime
        return DiscreteTime(args)
    else:
        raise ValueError(f"Unknown method: {args.method}.")


class CreateModel(nn.Module):
    def __init__(self, args, freeze=False):
        super(CreateModel, self).__init__()
        self.model = get_model(args)
        self.d_hid = self.model.d_hid
        self.compute_loss = self.model.compute_loss
        self.get_risk_logits = self.model.get_risk_logits

        # freeze the params for teacher model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.ema_decay = args.ema_decay
    
    def ema_update(self, student, step):
        alpha = min(1 - 1 / (step + 1), self.ema_decay)
        for param_self, param_stu in zip(self.parameters(), student.parameters()):
            param_self.data.mul_(alpha).add_(param_stu.data, alpha=1 - alpha)
        
    def forward(self, x):
        return self.model(x)
    
    def get_features(self, x):
        return self.model.encoder(x)


class SurvivalQueue(nn.Module):
    def __init__(self, dim, K, bin_func, expand_ratio=2.0, alpha=0.4, eps=1e-6):
        super().__init__()
        self.K = K
        self.dim = dim
        # self.num_bins = num_bins
        self.bin_func = bin_func
        self.expand_ratio = expand_ratio
        self.alpha = alpha
        self.eps = eps

        # Buffers for queue
        self.register_buffer("z", torch.zeros(K, dim))                # features
        self.register_buffer("e", torch.zeros(K))                     # events
        self.register_buffer("t", torch.zeros(K))                     # durations
        self.register_buffer("b", torch.zeros(K, dtype=torch.long))   # time bins
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("size", torch.zeros(1, dtype=torch.long))

        # Gaussian stats per bin (only event=1)
        # self.register_buffer("mu", torch.zeros(num_bins, dim))
        # self.register_buffer("var", torch.ones(num_bins, dim))
        # self.register_buffer("count", torch.zeros(num_bins))

    @staticmethod
    def pair_rank_mat(durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """Ranking matrix using bin order + censoring."""
        dur_i = durations.view(-1, 1)
        dur_j = durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)
        return ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i
    
    @torch.no_grad()
    def find_hard_samples(self, risk, durations, events):
        rank_mat = self.pair_rank_mat(durations, events)
        risk_i = risk.view(-1, 1)
        risk_j = risk.view(1, -1)
        risk_mat = (risk_i > risk_j).float()
        viol_mat = rank_mat * (1 - risk_mat)
        hard_mask = (viol_mat.sum(dim=1) + viol_mat.sum(dim=0)) > 0
        return hard_mask

    @torch.no_grad()
    def _update_gaussians(self, z, b, e):
        """Update Gaussian stats from event=1 samples."""
        for bin_idx in b[e == 1].unique():
            bin_mask = (b == bin_idx) & (e == 1)
            if bin_mask.sum() == 0:
                continue
            z_bin = z[bin_mask]
            Nb = z_bin.size(0)

            mu_old = self.mu[bin_idx].clone()
            cnt_old = self.count[bin_idx].item()
            cnt_new = cnt_old + Nb
            w_old = cnt_old / cnt_new
            w_new = Nb / cnt_new

            mu_new = w_old * mu_old + w_new * z_bin.mean(dim=0)
            var_new = w_old * self.var[bin_idx] + w_new * z_bin.var(dim=0, unbiased=False)

            self.mu[bin_idx] = mu_new
            self.var[bin_idx] = var_new + self.eps
            self.count[bin_idx] = cnt_new

    @torch.no_grad()
    def _sample_virtuals(self, N):
        """Generate virtual samples from Gaussian estimates."""
        virt_z, virt_e, virt_b = [], [], []
        for _ in range(N):
            bin_idx = torch.randint(0, self.num_bins, (1,)).item()
            if self.count[bin_idx] < 2:  # not enough stats
                continue
            mu = self.mu[bin_idx]
            var = self.var[bin_idx]
            sample = mu + torch.randn_like(var) * var.sqrt()
            virt_z.append(sample.unsqueeze(0))
            virt_e.append(torch.tensor([1.0], device=mu.device))   # always event=1
            virt_b.append(torch.tensor([bin_idx], device=mu.device))
        if len(virt_z) == 0:
            return None
        return (torch.cat(virt_z, dim=0),
                torch.cat(virt_e, dim=0),
                torch.cat(virt_b, dim=0))

    @torch.no_grad()
    def interpolate_virtuals(self, z, t, e, b, num_virtual):
        """Generate virtual samples by mixup on hard samples"""
        if z.size(0) < 2 or num_virtual == 0:
            return None

        idx1 = torch.randint(0, z.size(0), (num_virtual,), device=z.device)
        idx2 = torch.randint(0, z.size(0), (num_virtual,), device=z.device)

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((num_virtual,)).to(z.device)
        lam = lam.view(-1, 1)

        z_new = lam * z[idx1] + (1 - lam) * z[idx2]
        t_new = lam.squeeze() * t[idx1] + (1 - lam.squeeze()) * t[idx2]
        e_new = e[idx1]  # both shall have e = 1
        b_new = torch.from_numpy(self.bin_func(t_new.cpu().numpy())).to(z_new.device)

        return z_new, t_new, e_new, b_new

    @torch.no_grad()
    def enqueue(self, risk, z_new, e_new, t_new, b_new):
        # 1) Find hard samples
        mask = self.find_hard_samples(risk, t_new, e_new)
        # keep only the uncensored samples as real hard ones
        mask = mask & (e_new == 1)
        hard_z = z_new[mask].detach()
        hard_e = e_new[mask].detach()
        hard_t = t_new[mask].detach()
        hard_b = b_new[mask].detach()

        # 2) Interpolation
        B = hard_z.size(0)
        virt_size = int(B * self.expand_ratio) if B > 0 else 0
        if virt_size > 0:
            virt_z, virt_t, virt_e, virt_b = self.interpolate_virtuals(hard_z, hard_b, hard_e, hard_b, virt_size)
            if virt_z is not None:
                hard_z = torch.cat([hard_z, virt_z], dim=0)
                hard_t = torch.cat([hard_t, virt_t], dim=0)
                hard_e = torch.cat([hard_e, virt_e], dim=0)
                hard_b = torch.cat([hard_b, virt_b], dim=0)

        # 2) Update Gaussian stats (from all event=1)
        # self._update_gaussians(z_new, b_new, e_new)

        # 3) Add virtuals
        # Nh = hard_z.size(0)
        # Nv = int(Nh * (self.virt_ratio / (1 - self.virt_ratio))) if Nh > 0 and self.virt_ratio > 0 else 0
        # virt = self._sample_virtuals(Nv) if Nv > 0 else None
        # if virt is not None:
        #     vz, ve, vb = virt
        #     hard_z = torch.cat([hard_z, vz], dim=0)
        #     hard_e = torch.cat([hard_e, ve], dim=0)
        #     hard_b = torch.cat([hard_b, vb], dim=0)        

        # 4) Place into queue
        B = hard_z.size(0)

        if B < 1:
            return
        p = int(self.ptr.item())
        end = p + B

        def place(buf, src):
            if end <= self.K:
                buf[p:end] = src
            else:
                first = self.K - p
                buf[p:] = src[:first]
                buf[:end % self.K] = src[first:]

        place(self.z, hard_z)
        place(self.e, hard_e)
        place(self.t, hard_t)
        place(self.b, hard_b)

        self.ptr[0] = end % self.K
        self.size[0] = min(self.size + B, self.K)

    def get(self):
        N = int(self.size.item())
        return (self.z[:N], self.e[:N], self.t[:N], self.b[:N])
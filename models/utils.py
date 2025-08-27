import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder


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


class GatedAttMIL(nn.Module):
    def __init__(self, d_in, d_att=128):
        super().__init__()
        self.V = nn.Linear(d_in, d_att, bias=True)
        self.U = nn.Linear(d_in, d_att, bias=True)
        self.w = nn.Linear(d_att, 1, bias=False)

    def forward(self, feats, index, num_groups):
        """
        feats: (N, D)
        index: (N,) long in [0, num_groups-1]
        returns: (num_groups, D)
        """
        N, D = feats.shape
        A = torch.tanh(self.V(feats)) * torch.sigmoid(self.U(feats))  # (N, d_att)
        scores = self.w(A).squeeze(1)                                  # (N,)
        # group-wise softmax over tiles per patient
        # we compute softmax by subtracting max per group for stability
        max_per_group = torch.full((num_groups,), -1e9, device=feats.device)
        max_per_group = max_per_group.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
        scores_stab = scores - max_per_group.index_select(0, index)
        exp_scores = scores_stab.exp()
        denom = torch.zeros(num_groups, device=feats.device).scatter_add_(0, index, exp_scores)
        weights = exp_scores / (denom.index_select(0, index) + 1e-8)   # (N,)
        # weighted sum
        out = torch.zeros(num_groups, D, device=feats.device).index_add_(0, index, feats * weights.unsqueeze(1))
        return out

def _aggregate_by_index(feats, index, num_groups, mode="mean", att_module: nn.Module = None):
    if mode == "att":
        assert att_module is not None, "Attention module is None but mode=='att'."
        return att_module(feats, index, num_groups)

    D = feats.size(1)
    if mode == "mean":
        summed = torch.zeros(num_groups, D, device=feats.device).index_add_(0, index, feats)
        counts = torch.zeros(num_groups, device=feats.device).scatter_add_(0, index, torch.ones_like(index, dtype=feats.dtype))
        return summed / (counts.clamp_min(1e-8).unsqueeze(1))
    elif mode == "max":
        # scatter_reduce requires PyTorch>=1.12. Fallback: do segmentwise max manually
        out = torch.full((num_groups, D), -1e9, device=feats.device)
        out = out.scatter_reduce(0, index.view(-1,1).expand(-1, D), feats, reduce="amax", include_self=True)
        return out
    elif mode == "min":
        out = torch.full((num_groups, D), 1e9, device=feats.device)
        out = out.scatter_reduce(
            0, index.view(-1, 1).expand(-1, D), feats,
            reduce="amin", include_self=True
        )
        return out
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


class CreateModel(nn.Module):
    """
    Patient-level wrapper:
      - Encodes tiles with self.encoder
      - Aggregates per patient using batch_patient_index
      - Feeds patient embeddings to survival head
    """
    def __init__(self, args, freeze=False, aggregator="mean", att_dim=128):
        super().__init__()
        self.encoder = get_encoder(args)
        self.d_hid = getattr(args, "d_hid", getattr(self.encoder, "d_hid", None))
        if self.d_hid is None:
            raise ValueError("d_hid not found: set args.d_hid or ensure encoder exposes .d_hid")

        # survival heads take (D, n_classes) or (D, 1) depending on method
        if args.method.lower() == 'deephit':
            from .DeepHit import DeepHit
            self.surv_model = DeepHit(self.d_hid, args.n_classes)
        elif args.method.lower() == 'deepsurv':
            from .DeepSurv import DeepSurv
            self.surv_model = DeepSurv(self.d_hid, args.n_classes)
        elif args.method.lower() == 'discrete':
            from .DiscreteTime import DiscreteTime
            self.surv_model = DiscreteTime(self.d_hid, args.n_classes)
        else:
            raise ValueError(f"Unknown method: {args.method}.")

        # Aggregator selection
        self.agg_mode = None
        self.att_pool = None
        if aggregator == 'mean':
            self.agg_mode = "mean"
        elif aggregator == 'max':
            self.agg_mode = "max"
        elif aggregator == 'min':
            self.agg_mode = "min"
        elif aggregator == 'att':
            self.agg_mode = "att"
            self.att_pool = GatedAttMIL(self.d_hid, d_att=att_dim)
        else:
            self.agg_mode = None
            print("No valid aggregator specified: running at image level...")


        # Optional freeze (for teacher)
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.surv_model.parameters():
                p.requires_grad = False

        self.ema_decay = getattr(args, "ema_decay", 0.999)

    @torch.no_grad()
    def ema_update(self, student, step):
        alpha = min(1 - 1 / (step + 1), self.ema_decay)
        for p_self, p_stu in zip(self.parameters(), student.parameters()):
            p_self.data.mul_(alpha).add_(p_stu.data, alpha=1 - alpha)

    def _select_input_tensor(self, data):
        if "x" in data:            # eval path
            return data["x"]
        elif self.freeze:          # teacher uses weak
            return data["xw"]
        else:                      # student uses strong
            return data["xs"]

    def encode_tiles(self, data):
        x = self._select_input_tensor(data)     # (Ntot, C, H, W)
        feats = self.encoder(x)                 # Expect (Ntot, D)
        return feats

    def aggregate_patients(self, feats, data):
        if self.agg_mode is None:
            return feats, None  # tile-level, no grouping

        if "batch_patient_index" not in data:
            raise RuntimeError("batch_patient_index missing in data for patient-level aggregation.")

        index = data["batch_patient_index"].to(feats.device).long()  # (Ntot,)
        num_groups = int(index.max().item()) + 1 if index.numel() > 0 else 0

        patient_feats = _aggregate_by_index(
            feats, index, num_groups, mode=self.agg_mode, att_module=self.att_pool
        )
        return patient_feats, index
    
    def aggregate_labels(self, data):
        if hasattr(data, 'batch_patient_index'):
            patient_idx = data.batch_patient_index       # (N_images,)
            durations = data['duration']                 # (N_images,)
            events = data['event']                       # (N_images,)
            labels = data['label']                       # (N_images,)

            num_patients = patient_idx.max().item() + 1

            # Use scatter_reduce to compute patient-level aggregation
            agg_duration = torch.full((num_patients,), -1e9, device=durations.device)
            agg_duration = agg_duration.scatter_reduce(0, patient_idx, durations, reduce='amax', include_self=True)

            agg_event = torch.zeros(num_patients, device=events.device)
            agg_event = agg_event.scatter_reduce(0, patient_idx, events, reduce='amax', include_self=True)

            agg_label = torch.zeros(num_patients, device=labels.device, dtype=labels.dtype)
            agg_label = agg_label.scatter_reduce(0, patient_idx, labels, reduce='amax', include_self=True)

            return agg_duration, agg_event, agg_label

        else:
            return data['duration'], data['event'], data['label']


    def get_features(self, data):
        feats = self.encode_tiles(data)
        patient_feats, _ = self.aggregate_patients(feats, data)
        return patient_feats

    def get_surv_stats(self, feats):
        return self.surv_model(feats)

    def forward(self, data):
        feats = self.get_features(data)
        return self.get_surv_stats(feats)

    def compute_loss(self, logits, event, duration, label, bs=None):
        return self.surv_model.criterion(logits, event, duration, label, bs)
    
    @torch.no_grad()
    def prepare_for_validation(self, train_loader, bin_times, device=None):
        if device is None:
            device = next(self.parameters()).device

        durations_list, events_list, lp_list = [], [], []
        for batch in train_loader:
            batch = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in batch.items()}
            out = self.forward(batch)
            lp_list.append(out.risk.detach().cpu())
            durations_list.append(batch['duration'].detach().cpu())
            events_list.append(batch['event'].detach().cpu())

        durations = torch.cat(durations_list).float()
        events = torch.cat(events_list).int()
        risk = torch.cat(lp_list).float()
        
        self.surv_model.prepare_for_validation(risk, durations, events, bin_times, device=device)


class SurvivalQueue(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.K = K
        self.dim = dim

        # Buffers for queue
        self.register_buffer("z", torch.zeros(K, dim))                # features
        self.register_buffer("e", torch.zeros(K))                     # events
        self.register_buffer("t", torch.zeros(K))                     # durations
        self.register_buffer("b", torch.zeros(K, dtype=torch.long))   # time bins
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("size", torch.zeros(1, dtype=torch.long))

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
        risk_mat = (risk_i < risk_j).float()
        viol_mat = rank_mat * risk_mat
        hard_mask = (viol_mat.sum(dim=1) + viol_mat.sum(dim=0)) > 0
        return hard_mask

    @torch.no_grad()
    def enqueue(self, risk, z_new, e_new, t_new, b_new):
        # 1) Find hard samples
        mask = self.find_hard_samples(risk, t_new, e_new)
        # mask = torch.ones_like(mask, dtype=torch.bool)
        hard_z = z_new[mask].detach()
        hard_e = e_new[mask].detach()
        hard_t = t_new[mask].detach()
        hard_b = b_new[mask].detach()   

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
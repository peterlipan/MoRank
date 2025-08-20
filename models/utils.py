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

    def get_logits(self, features):
        return self.model.head(features)


class SurvivalQueue(nn.Module):
    def __init__(self, dim, K):
        """
        dim: feature dimension
        K:   queue size (capacity)
        """
        super().__init__()
        self.K = K
        self.dim = dim

        # Buffers for features and survival info
        self.register_buffer("z", torch.zeros(K, dim))      # features
        self.register_buffer("t", torch.zeros(K))           # durations
        self.register_buffer("e", torch.zeros(K))           # events
        self.register_buffer("b", torch.zeros(K, dtype=torch.long))  # time bins

        # Pointers & size tracker
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("size", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, z_new, t_new, e_new, b_new):
        """
        Add a minibatch into the queue (FIFO).
        z_new: [B, dim]
        t_new: [B]
        e_new: [B]
        b_new: [B]
        """
        B = z_new.size(0)
        p = int(self.ptr.item())
        end = p + B

        # detach
        z_new = z_new.detach()
        t_new = t_new.detach()
        e_new = e_new.detach()
        b_new = b_new.detach()

        def place(buf, src):
            if end <= self.K:
                buf[p:end] = src
            else:
                first = self.K - p
                buf[p:] = src[:first]
                buf[:end % self.K] = src[first:]

        place(self.z, z_new)
        place(self.t, t_new)
        place(self.e, e_new)
        place(self.b, b_new)

        # update pointer & current size
        self.ptr[0] = end % self.K
        self.size[0] = min(self.size + B, self.K)

    def get(self):
        """
        Get the current contents of the queue.
        Returns only the filled portion if not full yet.
        """
        N = int(self.size.item())
        return (self.z[:N], self.t[:N], self.e[:N], self.b[:N])
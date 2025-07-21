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
        return self.dict[key]


def CreateModel(args):
    if args.method.lower() == 'deephit':
        from .DeepHit import DeepHit
        return DeepHit(args)
    elif args.method.lower() == 'deepcdf':
        from .DeepCdf import DeepCdf
        return DeepCdf(args)
    elif args.method.lower() == 'deepsurv':
        from .DeepSurv import DeepSurv
        return DeepSurv(args)
    elif args.method.lower() == 'discrete':
        from .DiscreteTime import DiscreteTime
        return DiscreteTime(args)
    else:
        raise ValueError(f"Unknown method: {args.method}. Supported methods are: DeepHit, DeepCdf, DeepSurv.")
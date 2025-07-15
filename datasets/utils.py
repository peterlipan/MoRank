def CreateDataset(args):
    if args.dataset.lower() == 'metabric':
        from .metabric import METABRICData
        return METABRICData(feature_file=args.feature_file, label_file=args.label_file, 
                            n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed)
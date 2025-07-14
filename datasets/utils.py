def CreateDataset(args):
    if args.dataset.lower() == 'metabric':
        from .metabric import METABRICData
        return METABRICData(feature_file=args.feature_file, label_file=args.label_file)
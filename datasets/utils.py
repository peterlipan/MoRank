def CreateDataset(args):
    if args.dataset.lower() == 'gbmlgg':
        from .tcga_gbmlgg import TcgaGbmLggData
        return TcgaGbmLggData(data_root=args.data_root, pickle_path=args.pickle_path, backbone=args.backbone,
                              n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed, patient_level=args.patient_level)
    elif args.dataset.lower() == 'collagen':
        from .collagen import CollagenData
        return CollagenData(data_root=args.data_root, xlsx_path=args.xlsx_path, backbone=args.backbone,
                            n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed, patient_level=args.patient_level)
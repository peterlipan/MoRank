def CreateDataset(args):
    if args.dataset.lower() == 'metabric':
        from .metabric_deephit import METABRICData
        return METABRICData(feature_file=args.feature_file, label_file=args.label_file, 
                            n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed)
    elif args.dataset.lower() == 'support':
        from .pycox_datasets import SupportDataset
        return SupportDataset(n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed, normalize=args.normalize)
    elif args.dataset.lower() == 'gbsg':
        from .pycox_datasets import GBSGDataset
        return GBSGDataset(n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed, normalize=args.normalize)
    elif args.dataset.lower() == 'flchain':
        from .pycox_datasets import FlchainDataset
        return FlchainDataset(n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed, normalize=args.normalize)
    elif args.dataset.lower() == 'nwtco':
        from .pycox_datasets import NWTCODataSet
        return NWTCODataSet(n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed, normalize=args.normalize)
    elif args.dataset.lower() == 'gbmlgg':
        from .tcga_gbmlgg import TcgaGbmLggData
        return TcgaGbmLggData(data_root=args.data_root, pickle_path=args.pickle_path, backbone=args.backbone, task=args.task,
                              n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed)
    elif args.dataset.lower() == 'eyepacs':
        from .eyepacs import EyepacsData
        return EyepacsData(root=args.data_root, backbone=args.backbone)
    elif args.dataset.lower() == 'knee_osteoarthritis':
        from .knee_osteoarthritis import KneeOsteoarthritisData
        return KneeOsteoarthritisData(root=args.data_root, backbone=args.backbone)
    elif args.dataset.lower() == 'collagen':
        from .collagen import CollagenData
        return CollagenData(data_root=args.data_root, xlsx_path=args.xlsx_path, backbone=args.backbone,
                            n_bins=args.n_bins, stratify=args.stratify, kfold=args.kfold, seed=args.seed, patient_level=args.patient_level)
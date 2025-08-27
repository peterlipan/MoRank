import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from .augmentations import Transforms


def aggregate_data(root, dst):
    # aggregate the samples from existing KFold to one pickle file
    pkl_path = os.path.join(root, "splits", "gbmlgg15cv_all_st_0_0_0.pkl")
    data_cv = pickle.load(open(pkl_path, 'rb'))

    train_path = data_cv['cv_splits'][1]['train']['x_path']
    test_path = data_cv['cv_splits'][1]['test']['x_path']
    train_grade = data_cv['cv_splits'][1]['train']['g']
    test_grade = data_cv['cv_splits'][1]['test']['g']
    train_omic = data_cv['cv_splits'][1]['train']['x_omic']
    test_omic = data_cv['cv_splits'][1]['test']['x_omic']
    train_patname = data_cv['cv_splits'][1]['train']['x_patname']
    test_patname = data_cv['cv_splits'][1]['test']['x_patname']
    train_event = data_cv['cv_splits'][1]['train']['e']
    test_event = data_cv['cv_splits'][1]['test']['e']
    train_duration = data_cv['cv_splits'][1]['train']['t']
    test_duration = data_cv['cv_splits'][1]['test']['t']

    path = np.concatenate([train_path, test_path])
    omic = np.concatenate([train_omic, test_omic])
    grade = np.concatenate([train_grade, test_grade])
    patname = np.concatenate([train_patname, test_patname])
    event = np.concatenate([train_event, test_event]).astype(np.int64)
    duration = np.concatenate([train_duration, test_duration]).astype(np.float32)

    path = [p.replace("./data/TCGA_GBMLGG/", "") for p in path] # keep relative path
    path = [os.path.join(root, p) for p in path]  # absolute path
    path = np.array(path)

    # merge to a pickle file
    data = {
        'patient': patname,
        'path': path,
        'omic': omic,
        'grade': grade,
        'event': event,
        'duration': duration
    }
    with open(dst, 'wb') as f:
        pickle.dump(data, f)


class TcgaGbmLggData:
    def __init__(self, data_root, pickle_path, backbone, task, n_bins=-1, stratify=False, kfold=5, seed=42):
        self.data_root = data_root
        self.pickle_path = pickle_path
        self.n_bins = n_bins
        self.stratify = stratify
        self.kfold = kfold
        self.seed = seed
        self.backbone = backbone
        self.n_features = 224 if backbone.startswith(('resnet', 'densenet', 'efficient')) else 1024  # CNNs are fixed to 224. Use original size for Vision Transformers
        transform = Transforms(self.n_features)

        self.strong_transform = transform.strong_transform
        self.weak_transform = transform.weak_transform
        self.test_transform = transform.test_transform

        if not os.path.exists(self.pickle_path):
            os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
            aggregate_data(self.data_root, self.pickle_path)

        # Load the aggregated data
        with open(self.pickle_path, 'rb') as f:
            data = pickle.load(f)

        self.omic = data['omic']
        self.duration = data['duration']
        self.event = data['event']
        self.patient = data['patient']
        self.path = data['path']
        self.grade = data['grade']

        if task == 'classification':
            # some g = -1. mask those samples out
            mask = self.grade != -1
            self.omic = self.omic[mask]
            self.duration = self.duration[mask]
            self.event = self.event[mask]
            self.patient = self.patient[mask]
            self.path = self.path[mask]
            self.grade = self.grade[mask]
            # label = grade in this task
            self.label = self.grade.astype(np.int64) + 1 # dummy class for the head of CDF
            self.n_classes = len(np.unique(self.label)) + 2  # +2 for edge bins

        elif task == 'survival':
            if n_bins > 0:
                self.bin_durations(n_bins)
                self.n_classes = n_bins  # for edge bins
            else:
                self.n_classes = int(np.max(self.duration) * 1.2)  # default DeepHit-style horizon
                self.label = self.duration.astype(np.int64)  # convert to int64 for compatibility
        else:
            raise ValueError(f"Unsupported task: {task}")
        
    def bin_durations(self, n_bins):
        # Bin duration using quantiles (equal number of samples per bin)
        self.bin_edges = np.quantile(self.duration, q=np.linspace(0, 1, n_bins + 1))
        self.label = np.digitize(self.duration, self.bin_edges[1:-1]).astype(np.int64)

    def _duration_to_label(self, duration):
        if self.n_bins > 0:
            return np.digitize(duration, self.bin_edges[1:-1]).astype(np.int64)
        else:
            return duration.astype(np.int64)
    
    def get_kfold_datasets(self):
        # kfold on the patient level
        pid = np.unique(self.patient)
        patient_event = np.array([self.event[self.patient == p][0] for p in pid])
        if self.stratify:
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(pid, patient_event):
                train_idx = np.where(np.isin(self.patient, pid[train_idx]))[0]
                test_idx = np.where(np.isin(self.patient, pid[test_idx]))[0]
                yield TcgaGbmLggImageDataset(self, train_idx, training=True), TcgaGbmLggImageDataset(self, test_idx, training=False)
        else:
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(pid):
                train_idx = np.where(np.isin(self.patient, pid[train_idx]))[0]
                test_idx = np.where(np.isin(self.patient, pid[test_idx]))[0]
                yield TcgaGbmLggImageDataset(self, train_idx, training=True), TcgaGbmLggImageDataset(self, test_idx, training=False)


class TcgaGbmLggImageDataset(Dataset):
    def __init__(self, tcga_data, indices, training=False):
        super().__init__()
        self.duration = tcga_data.duration[indices]
        self.event = tcga_data.event[indices]
        self.label = tcga_data.label[indices]
        self.patient = tcga_data.patient[indices]
        self.path = [tcga_data.path[i] for i in indices]
        self.n_features = tcga_data.n_features
        self.n_classes = tcga_data.n_classes
        self.n_events = np.unique(self.event).size
        self._duration_to_label = tcga_data._duration_to_label
        self.training = training
        self.strong_transform = tcga_data.strong_transform
        self.weak_transform = tcga_data.weak_transform
        self.test_transform = tcga_data.test_transform

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.path[idx])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        if self.training:
            xs = self.strong_transform(image=image)['image']
            xw = self.weak_transform(image=image)['image']
            return {
                'xs': xs,
                'xw': xw,
                'duration': self.duration[idx],
                'event': self.event[idx],
                'label': self.label[idx],
                'patient_id': self.patient[idx]
            }
        else:
            x = self.test_transform(image=image)['image']
            return {
                'x': x,
                'duration': self.duration[idx],
                'event': self.event[idx],
                'label': self.label[idx],
                'patient_id': self.patient[idx]
            }

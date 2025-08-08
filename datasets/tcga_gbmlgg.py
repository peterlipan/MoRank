import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


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

    path = [p.replace("./data/TCGA_GBMLGG", "") for p in path] # keep relative path

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
    def __init__(self, data_root, pickle_path, n_bins=-1, stratify=False, kfold=5, seed=42):
        self.data_root = data_root
        self.pickle_path = pickle_path
        self.n_bins = n_bins
        self.stratify = stratify
        self.kfold = kfold
        self.seed = seed
        self.imgae_size = 1024

        if not os.path.exists(self.pickle_path):
            aggregate_data(self.data_root, self.pickle_path)

        # Load the aggregated data
        with open(self.pickle_path, 'rb') as f:
            data = pickle.load(f)

        self.omic = data['omic']
        self.duration = data['duration']
        self.event = data['event']
        self.patient = data['patient']
        self.path = data['path']

        if n_bins > 0:
            self.bin_durations(n_bins)
            self.n_classes = n_bins + 2  # for edge bins
        else:
            self.n_classes = int(np.max(self.duration) * 1.2)  # default DeepHit-style horizon
            self.label = self.duration.astype(np.int64)  # convert to int64 for compatibility
        
    def bin_durations(self, n_bins):
        # Bin duration using quantiles (equal number of samples per bin)
        self.bin_edges = np.quantile(self.duration, q=np.linspace(0, 1, n_bins + 1))
        self.label = np.digitize(self.duration, self.bin_edges[1:-1]).astype(np.int64) + 1

    def _duration_to_label(self, duration):
        if self.n_bins > 0:
            return np.digitize(duration, self.bin_edges[1:-1]).astype(np.int64) + 1
        else:
            return duration.astype(np.int64)
    
    def get_kfold_datasets(self):
        if self.stratify:
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(self.omic, self.event):
                yield TcgaGbmLggDataset(self, train_idx), TcgaGbmLggDataset(self, test_idx)
        else:
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(self.omic):
                yield TcgaGbmLggDataset(self, train_idx), TcgaGbmLggDataset(self, test_idx)
                
class TcgaGbmLggDataset:
    def __init__(self, tcga_data, indices):
        self.omic = tcga_data.omic[indices]
        self.duration = tcga_data.duration[indices]
        self.event = tcga_data.event[indices]
        self.label = tcga_data.label[indices] if hasattr(tcga_data, 'label') else self.duration.astype(np.int64)
        self.patient = tcga_data.patient[indices]
        self.path = [tcga_data.path[i] for i in indices]
        self.n_features = tcga_data.omic.shape[1]
        self.n_classes = tcga_data.n_classes
        self.n_events = np.unique(self.event).size





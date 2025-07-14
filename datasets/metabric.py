import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold


class METABRICData:
    def __init__(self, feature_file, label_file):
        self.data, self.duration, self.event = self._load_and_normalize(feature_file, label_file)
        self.n_features = self.data.shape[1]
        self.n_events = int(len(np.unique(self.event)) - 1) # ignore censoring
        self.n_classes = int(np.max(self.duration) * 1.2) # following DeepHit, to have enough time-horizon

    def _load_and_normalize(self, feature_file, label_file):
        df_data = pd.read_csv(feature_file)
        df_label = pd.read_csv(label_file)

        data = df_data.values.astype(np.float32)
        duration = df_label['event_time'].values.astype(np.float32)
        event = df_label['label'].values.astype(np.float32)

        data = self._normalize(data)
        return data, duration, event

    def _normalize(self, X):
        # followign DeepHit to normalize along samples
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        X = (X - mean) / std
        return X

    def get_kfold_datasets(self, k=5, seed=42):
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(self.data):
            yield METABRICDataset(self, train_idx), METABRICDataset(self, test_idx)


class METABRICDataset(Dataset):
    def __init__(self, metabric_data, indices):
        self.data = metabric_data.data[indices].astype(np.float32)
        self.duration = metabric_data.duration[indices].astype(np.int64)
        self.event = metabric_data.event[indices].astype(np.int64)
        self.n_features = metabric_data.n_features
        self.n_classes = metabric_data.n_classes
        self.n_events = metabric_data.n_events

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'duration': self.duration[idx],
            'event': self.event[idx]
        }

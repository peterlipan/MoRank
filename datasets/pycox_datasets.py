import torch
import pycox
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset
from pycox.datasets import metabric, support, gbsg, flchain, nwtco, sac3, rr_nl_nhp, sac_admin5


class PycoxDataset:
    def __init__(self, pycox_dataloader, n_bins=-1, stratify=False, kfold=5, seed=42, normalize=False):
        self.pycox_dataloader = pycox_dataloader
        self.df = self._load_df()
        self.data, self.duration, self.event = self._preprocess_data(normalize=normalize)
        self.n_features = self.data.shape[1]
        self.n_events = int(len(np.unique(self.event)) - 1)  # ignore censoring
        self.stratify = stratify
        self.kfold = kfold
        self.seed = seed
        self.n_bins = n_bins

        if n_bins > 0:
            self.bin_durations(n_bins)
            self.n_classes = n_bins + 10  # for edge bins
        else:
            self.n_classes = int(np.max(self.duration) * 1.2)  # default DeepHit-style horizon
            self.label = self.duration.astype(np.int64)

    def _load_df(self):
        return self.pycox_dataloader.read_df()
    
    def _preprocess_data(self, normalize=False):
        duration = self.df['duration'].values.astype(float)
        event = self.df['event'].values.astype(int)
        data = self.df.iloc[:, :-2].values.astype(float) # Exclude 'duration' and 'event' columns
        if normalize:
            data = self._normalize(data)
        return data, duration, event

    def _normalize(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        return (X - mean) / std
    
    def bin_durations(self, n_bins):
        # Bin duration using quantiles (equal number of samples per bin)
        self.bin_edges = np.quantile(self.duration, q=np.linspace(0, 1, n_bins + 1))
        self.label = np.digitize(self.duration, self.bin_edges[1:-1]).astype(np.int64) + 5

    def _duration_to_label(self, duration):
        if self.n_bins > 0:
            return np.digitize(duration, self.bin_edges[1:-1]).astype(np.int64) + 5
        else:
            return duration.astype(np.int64)
    
    def get_kfold_datasets(self):
        if self.stratify:
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(self.data, self.event):
                yield PytorchDataset(self, train_idx), PytorchDataset(self, test_idx)
        else:
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(self.data):
                yield PytorchDataset(self, train_idx), PytorchDataset(self, test_idx)


class PytorchDataset(Dataset):
    def __init__(self, pycox_data, indices):
        self.duration = pycox_data.duration[indices]
        self.event = pycox_data.event[indices]
        self.data = pycox_data.data[indices]
        self.label = pycox_data.label[indices]
        self.n_features = pycox_data.n_features
        self.n_classes = pycox_data.n_classes
        self.n_events = pycox_data.n_events
        self._duration_to_label = pycox_data._duration_to_label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.label[idx], dtype=torch.long),
            'event': torch.tensor(self.event[idx], dtype=torch.long),
            'duration': torch.tensor(self.duration[idx], dtype=torch.float32)
        }

class MetabricDataset(PycoxDataset):
    def __init__(self, n_bins=-1, stratify=False, kfold=5, seed=42, normalize=False):
        super().__init__(metabric, n_bins=n_bins, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize)

class SupportDataset(PycoxDataset):
    def __init__(self, n_bins=-1, stratify=False, kfold=5, seed=42, normalize=False):
        super().__init__(support, n_bins=n_bins, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize)

class GBSGDataset(PycoxDataset):
    def __init__(self, n_bins=-1, stratify=False, kfold=5, seed=42, normalize=False):
        super().__init__(gbsg, n_bins=n_bins, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize)

class FlchainDataset(PycoxDataset):
    def __init__(self, n_bins=-1, stratify=False, kfold=5, seed=42, normalize=False):
        super().__init__(flchain, n_bins=n_bins, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize)
    
    def _load_df(self):
        df = self.pycox_dataloader.read_df(processed=False)
        # drop the categorical columns, sample.yr and flc.grp
        df = df.drop(['chapter', 'sample.yr', 'flc.grp'], axis=1).loc[lambda x: x['creatinine'].isna() == False].reset_index(drop=True).assign(sex=lambda x: (x['sex'] == 'M'))
        df = df.rename(columns={'futime': 'duration', 'death': 'event'})
        return df

class NWTCODataSet(PycoxDataset):
    def __init__(self, n_bins=-1, stratify=False, kfold=5, seed=42, normalize=False):
        super().__init__(nwtco, n_bins=n_bins, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize)

    def _load_df(self):
        df = self.pycox_dataloader.read_df(processed=False)
        df = df.assign(instit_2=df['instit'] - 1, histol_2=df['histol'] - 1, study_4=df['study'] - 3,
                       stage=df['stage'].astype('category')).drop(['seqno', 'instit', 'histol', 'study', 'in.subcohort', 'rownames'], axis=1)
        df = df.rename(columns={'edrel': 'duration', 'rel': 'event'})
        # rearrange columns
        cols = ['duration', 'event']
        df = df[list(df.columns.drop(cols)) + cols]
        return df
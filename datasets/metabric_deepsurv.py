import h5py
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from typing import Tuple, Optional


class METABRICData:
    def __init__(self, file_path: str, n_bins: int = -1):
        """
        Load METABRIC dataset and bin durations if needed.

        Args:
            file_path (str): Path to the HDF5 file.
            n_bins (int): Number of quantile-based bins. If -1, use raw durations.
        """
        self.n_bins = n_bins
        self.train_raw, self.test_raw = self._load_data(file_path)

        self.n_features = self.train_raw['x'].shape[1]
        self.n_events = int(np.unique(self.train_raw['e']).max())  # assumes 0 is censoring

        if n_bins > 0:
            self.bin_edges, self.train_raw['label'] = self._fit_bins(self.train_raw['t'], n_bins)
            self.test_raw['label'] = self._duration_to_label(self.test_raw['t'])
            self.n_classes = n_bins + 2  # one for each bin + edge cases
        else:
            self.train_raw['label'] = self.train_raw['t'].astype(np.int64)
            self.test_raw['label'] = self.test_raw['t'].astype(np.int64)
            self.n_classes = int(self.train_raw['t'].max() * 1.2)

    def _load_data(self, file_path: str) -> Tuple[dict, dict]:
        """Load train/test splits from HDF5."""
        data = defaultdict(dict)
        with h5py.File(file_path, 'r') as f:
            for split in f:
                for key in f[split]:
                    data[split][key] = f[split][key][:]
        return data['train'], data['test']

    def _fit_bins(self, durations: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit quantile-based bins to training durations.

        Args:
            durations (np.ndarray): Duration values from training set.
            n_bins (int): Number of bins to split into.

        Returns:
            Tuple of bin edges and corresponding bin labels for durations.
        """
        bin_edges = np.quantile(durations, q=np.linspace(0, 1, n_bins + 1), interpolation='linear')
        labels = np.digitize(durations, bins=bin_edges[1:-1]) + 1
        return bin_edges, labels.astype(np.int64)

    def _duration_to_label(self, durations: np.ndarray) -> np.ndarray:
        """
        Convert durations into pre-fitted bin labels or raw labels.

        Args:
            durations (np.ndarray): Duration array.

        Returns:
            np.ndarray: Bin labels.
        """
        if self.n_bins > 0:
            return (np.digitize(durations, bins=self.bin_edges[1:-1]) + 1).astype(np.int64)
        return durations.astype(np.int64)

    def get_official_train_test(self) -> Tuple["METABRICDataset", "METABRICDataset"]:
        """Return Dataset objects for train and test."""
        return (
            METABRICDataset(self, self.train_raw),
            METABRICDataset(self, self.test_raw)
        )


class METABRICDataset(Dataset):
    def __init__(self, meta: METABRICData, data: dict):
        """
        Torch Dataset wrapper for METABRIC data.

        Args:
            meta (METABRICData): Reference to the main data class for meta info.
            data (dict): Dictionary with 'x', 't', 'e', and 'label'.
        """
        self.x = data['x'].astype(np.float32)
        self.duration = data['t'].astype(np.int64)
        self.event = data['e'].astype(np.int64)
        self.label = data['label'].astype(np.int64)

        self.n_features = meta.n_features
        self.n_classes = meta.n_classes
        self.n_events = meta.n_events
        self._duration_to_label = meta._duration_to_label

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict:
        return {
            'data': self.x[idx],
            'label': self.label[idx],
            'duration': self.duration[idx],
            'event': self.event[idx]
        }

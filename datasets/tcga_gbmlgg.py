import os
import torch
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
    def __init__(self, data_root, pickle_path, backbone, n_bins=-1, stratify=False, kfold=5, seed=42, patient_level=False):
        self.data_root = data_root
        self.pickle_path = pickle_path
        self.n_bins = n_bins
        self.stratify = stratify
        self.kfold = kfold
        self.seed = seed
        self.backbone = backbone
        self.patient_level = patient_level
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

        if n_bins > 0:
            self.bin_durations(n_bins)
            self.n_classes = n_bins  # for edge bins
        else:
            self.n_classes = int(np.max(self.duration) * 1.2)  # default DeepHit-style horizon
            self.label = self.duration.astype(np.int64)  # convert to int64 for compatibility

        
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
        _dataset = TcgaGbmLggPatientDataset if self.patient_level else TcgaGbmLggImageDataset
        pid = np.unique(self.patient)
        patient_event = np.array([self.event[self.patient == p][0] for p in pid])
        if self.stratify:
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(pid, patient_event):
                train_idx = np.where(np.isin(self.patient, pid[train_idx]))[0]
                test_idx = np.where(np.isin(self.patient, pid[test_idx]))[0]
                yield _dataset(self, train_idx, training=True), _dataset(self, test_idx, training=False)
        else:
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(pid):
                train_idx = np.where(np.isin(self.patient, pid[train_idx]))[0]
                test_idx = np.where(np.isin(self.patient, pid[test_idx]))[0]
                yield _dataset(self, train_idx, training=True), _dataset(self, test_idx, training=False)


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
        self._duration_to_label = tcga_data._duration_to_label
        self.training = training
        self.strong_transform = tcga_data.strong_transform
        self.weak_transform = tcga_data.weak_transform
        self.test_transform = tcga_data.test_transform
        self.n_samples = len(self.path)
        self.n_patients = len(np.unique(self.patient))

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


class TcgaGbmLggPatientDataset(Dataset):
    def __init__(self, tcga_data, indices, training=False):
        super().__init__()
        self.training = training
        self.strong_transform = tcga_data.strong_transform
        self.weak_transform = tcga_data.weak_transform
        self.test_transform = tcga_data.test_transform
        self._duration_to_label = tcga_data._duration_to_label
        self.n_features = tcga_data.n_features
        self.n_classes = tcga_data.n_classes

        self.n_samples = len(indices)
        self.n_patients = len(np.unique(tcga_data.patient[indices]))

        # subset by indices
        duration = tcga_data.duration[indices]
        event = tcga_data.event[indices]
        label = tcga_data.label[indices]
        patient = tcga_data.patient[indices]
        path = [tcga_data.path[i] for i in indices]

        # Group by patient
        self.patients = []
        for pid in np.unique(patient):
            mask = patient == pid
            record = {
                "patient_id": pid,
                "images": [path[i] for i in np.where(mask)[0]],
                "duration": duration[mask][0],
                "event": event[mask][0],
                "label": label[mask][0],
            }
            self.patients.append(record)

        self.n_patients = len(self.patients)

    def __len__(self):
        return self.n_patients

    def __getitem__(self, idx):
        record = self.patients[idx]
        x, xs, xw = [], [], []
        for img_path in record["images"]:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)

            if self.training:
                xs.append(self.strong_transform(image=image)["image"])
                xw.append(self.weak_transform(image=image)["image"])
            else:
                x.append(self.test_transform(image=image)["image"])

        if self.training:
            xs = torch.stack(xs)
            xw = torch.stack(xw)
            return {
                "xs": xs,
                "xw": xw,
                "duration": record["duration"],
                "event": record["event"],
                "label": record["label"],
                "patient_id": record["patient_id"],
            }
        else:
            x = torch.stack(x)
            return {
                "x": x,
                "duration": record["duration"],
                "event": record["event"],
                "label": record["label"],
                "patient_id": record["patient_id"],
            }

    @staticmethod
    def testing_collate_fn(batch):
        all_imgs, patient_index, durations, events, labels, pids = [], [], [], [], [], []
        for i, item in enumerate(batch):
            imgs = item["x"]
            all_imgs.append(imgs)
            patient_index.append(torch.full((imgs.size(0),), i, dtype=torch.long))
            durations.append(item["duration"])
            events.append(item["event"])
            labels.append(item["label"])
            pids.append(item["patient_id"])

        all_imgs = torch.cat(all_imgs, dim=0)
        patient_index = torch.cat(patient_index)
        durations = torch.tensor(durations, dtype=torch.float)
        events = torch.tensor(events, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "x": all_imgs,
            "batch_patient_index": patient_index,
            "duration": durations,
            "event": events,
            "label": labels,
            "patient_id": pids,
        }

    @staticmethod
    def training_collate_fn(batch):
        all_xs, all_xw, patient_index, durations, events, labels, pids = [], [], [], [], [], [], []
        for i, item in enumerate(batch):
            xs = item["xs"]
            xw = item["xw"]
            all_xs.append(xs)
            all_xw.append(xw)
            patient_index.append(torch.full((xs.size(0),), i, dtype=torch.long))
            durations.append(item["duration"])
            events.append(item["event"])
            labels.append(item["label"])
            pids.append(item["patient_id"])

        all_xs = torch.cat(all_xs, dim=0)
        all_xw = torch.cat(all_xw, dim=0)
        patient_index = torch.cat(patient_index)
        durations = torch.tensor(durations, dtype=torch.float)
        events = torch.tensor(events, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "xs": all_xs,
            "xw": all_xw,
            "batch_patient_index": patient_index,
            "duration": durations,
            "event": events,
            "label": labels,
            "patient_id": pids,
        }
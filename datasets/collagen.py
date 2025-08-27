import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, StratifiedKFold
from .augmentations import Transforms


class CollagenData:
    def __init__(self, data_root, xlsx_path, backbone, n_bins=-1, stratify=True, kfold=5, seed=42, patient_level=False):
        self.data_root = data_root
        self.xlsx_path = xlsx_path
        self.backbone = backbone
        self.n_bins = n_bins
        self.stratify = stratify
        self.patient_level = patient_level
        self.kfold = kfold
        self.seed = seed
        self.n_features = 224 if backbone.startswith(('resnet', 'densenet', 'efficient')) else 1200  # CNNs are fixed to 224. Use original size for Vision Transformers
        transform = Transforms(self.n_features)

        self.strong_transform = transform.strong_transform
        self.weak_transform = transform.weak_transform
        self.test_transform = transform.test_transform

        self.df = pd.read_excel(self.xlsx_path)
        self.df = self.df.dropna(subset=['Overall.Survival.Months', 'Overall.Survival.Status'])
        
        if n_bins > 0:
            self.df['label'] = self.bin_durations(self.df['Overall.Survival.Months'].values, n_bins)
            self.n_classes = n_bins

        else:
            self.df['label'] = self.df['Overall.Survival.Months'].values.astype(np.int64)
            self.n_classes = int(self.df['label'].max() * 1.2) # following deephit
            

    def bin_durations(self, durations, n_bins):
        # Bin duration using quantiles (equal number of samples per bin)
        self.bin_edges = np.quantile(durations, q=np.linspace(0, 1, n_bins + 1))
        return np.digitize(durations, self.bin_edges[1:-1]).astype(np.int64)

    def _duration_to_label(self, duration):
        if self.n_bins > 0:
            return np.digitize(duration, self.bin_edges[1:-1]).astype(np.int64)
        else:
            return duration.astype(np.int64)
    
    def get_kfold_datasets(self):
        _dataset = CollagenPatientDataset if self.patient_level else CollagenDataset
        if self.stratify:
            pid = self.df['BBNumber'].unique()
            patient_event = self.df.groupby('BBNumber')['Overall.Survival.Status'].first().values
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(pid, patient_event):
                train_pid = pid[train_idx]
                test_pid = pid[test_idx]
                train_df = self.df[self.df['BBNumber'].isin(train_pid)]
                test_df = self.df[self.df['BBNumber'].isin(test_pid)]
                yield _dataset(self, train_df, training=True), _dataset(self, test_df, training=False)
        # no stratification but still on patients
        else:
            pid = self.df['BBNumber'].unique()
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(pid):
                train_pid = pid[train_idx]
                test_pid = pid[test_idx]
                train_df = self.df[self.df['BBNumber'].isin(train_pid)]
                test_df = self.df[self.df['BBNumber'].isin(test_pid)]
                yield _dataset(self, train_df, training=True), _dataset(self, test_df, training=False)


class CollagenDataset(Dataset):
    def __init__(self, collagen_data, image_df, training=True):
        super().__init__()
        self.df = image_df.copy()
        self.n_classes = collagen_data.n_classes
        self.n_features = collagen_data.n_features
        self.training = training
        self.root = collagen_data.data_root
        self.strong_transform = collagen_data.strong_transform
        self.weak_transform = collagen_data.weak_transform
        self.test_transform = collagen_data.test_transform
        self._duration_to_label = collagen_data._duration_to_label
        # for training surv estimation
        self.duration = self.df['Overall.Survival.Months'].values
        self.event = self.df['Overall.Survival.Status'].values
        self.n_samples = len(self.df)
        self.n_patients = self.df['BBNumber'].nunique()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['Filename']
        duration = row['Overall.Survival.Months']
        event = row['Overall.Survival.Status']
        label = row['label']
        patient_id = row['BBNumber']

        path_r = os.path.join(self.root, f"{row['Folder']}_R", filename)
        path_g = os.path.join(self.root, f"{row['Folder']}_G", filename)
        path_b = os.path.join(self.root, f"{row['Folder']}_B", filename)

        # Load each grayscale image
        image_r = Image.open(path_r).convert('L')  # Convert to grayscale
        image_g = Image.open(path_g).convert('L')  # Convert to grayscale
        image_b = Image.open(path_b).convert('L')  # Convert to grayscale

        # Convert images to numpy arrays
        image_r = np.array(image_r)
        image_g = np.array(image_g)
        image_b = np.array(image_b)

        # Stack grayscale images to form a 3-channel image
        image = np.stack([image_r, image_g, image_b], axis=-1)  # Shape: (H, W, 3)

        if self.training:
            xs = self.strong_transform(image=image)['image']
            xw = self.weak_transform(image=image)['image']
            return {
                'xs': xs,
                'xw': xw,
                'duration': duration,
                'event': event,
                'label': label,
                'patient_id': patient_id
            }
        else:
            x = self.test_transform(image=image)['image']
            return {
                'x': x,
                'duration': duration,
                'event': event,
                'label': label,
                'patient_id': patient_id
            }

class CollagenPatientDataset(Dataset):
    def __init__(self, collagen_data, image_df, training=True):
        super().__init__()
        self.n_classes = collagen_data.n_classes
        self.n_features = collagen_data.n_features
        self.training = training
        self.root = collagen_data.data_root
        self.strong_transform = collagen_data.strong_transform
        self.weak_transform = collagen_data.weak_transform
        self.test_transform = collagen_data.test_transform
        self._duration_to_label = collagen_data._duration_to_label
        self.n_samples = len(image_df)
        self.n_patients = image_df['BBNumber'].nunique()

        # for training surv estimation
        self.duration = image_df.groupby("BBNumber")["Overall.Survival.Months"].first().values
        self.event = image_df.groupby("BBNumber")["Overall.Survival.Status"].first().values

        # Group dataframe by patient
        self.patients = []
        for pid, df_pid in image_df.groupby("BBNumber"):
            record = {
                "patient_id": pid,
                "images": df_pid.to_dict(orient="records"),
                "duration": df_pid["Overall.Survival.Months"].iloc[0],
                "event": df_pid["Overall.Survival.Status"].iloc[0],
                "label": df_pid["label"].iloc[0]
            }
            self.patients.append(record)

    def __len__(self):
        return self.n_patients

    def __getitem__(self, idx):
        record = self.patients[idx]
        x, xs, xw = [], [], []
        for row in record["images"]:
            # paths
            path_r = os.path.join(self.root, f"{row['Folder']}_R", row["Filename"])
            path_g = os.path.join(self.root, f"{row['Folder']}_G", row["Filename"])
            path_b = os.path.join(self.root, f"{row['Folder']}_B", row["Filename"])

            # load + stack into RGB
            r = np.array(Image.open(path_r).convert("L"))
            g = np.array(Image.open(path_g).convert("L"))
            b = np.array(Image.open(path_b).convert("L"))
            image = np.stack([r, g, b], axis=-1)

            # transform
            if self.training:
                xs.append(self.strong_transform(image=image)["image"])
                xw.append(self.weak_transform(image=image)["image"])
            else:
                x.append(self.test_transform(image=image)["image"])

        if self.training:
            xs = torch.stack(xs)
            xw = torch.stack(xw)
            return {
                'xs': xs,
                'xw': xw,
                'duration': record["duration"],
                'event': record["event"],
                'label': record["label"],
                'patient_id': record["patient_id"],
            }
        else:
            x = torch.stack(x)
            return {
                'x': x,
                'duration': record["duration"],
                'event': record["event"],
                'label': record["label"],
                'patient_id': record["patient_id"],
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

        all_imgs = torch.cat(all_imgs, dim=0)  # [N_total_patches, C, H, W]
        patient_index = torch.cat(patient_index)  # [N_total_patches]
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

        all_xs = torch.cat(all_xs, dim=0)  # [N_total_patches, C, H, W]
        all_xw = torch.cat(all_xw, dim=0)  # [N_total_patches, C, H, W]
        patient_index = torch.cat(patient_index)  # [N_total_patches]
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

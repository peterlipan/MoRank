import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class EyepacsData:
    def __init__(self, root, backbone):
        self.root = root
        self.train_path, self.val_path, self.test_path = os.path.join(root, 'train'), os.path.join(root, 'val'), os.path.join(root, 'test')
        self.train_df = self._read_folder(self.train_path)
        self.val_df = self._read_folder(self.val_path)
        self.test_df = self._read_folder(self.test_path)
        self.n_classes = self.train_df['label'].nunique() + 2  # dummy class for the head and tail of CDF

        self.n_features = 224 if backbone.startswith(('resnet', 'densenet', 'efficient')) else 600  # RNNs are fixed to 224. Use original size for Vision Transformers
        self.train_transform = A.Compose([
            A.Resize(height=self.n_features, width=self.n_features),
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.5),
            A.RandomRotate90(p=.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.5),
            A.OneOf([
                    A.ElasticTransform(p=.5),
                    A.GridDistortion(p=.5),
                    A.OpticalDistortion(p=.5),
                ], p=.5),
            A.OneOf([
                A.RandomGridShuffle(grid=(3, 3), p=.5),
                A.RandomGridShuffle(grid=(7, 7), p=.5),
                A.RandomGridShuffle(grid=(11, 11), p=.5),
            ], p=.5),
            A.ColorJitter(p=.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.ChannelShuffle(p=.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.test_transform = A.Compose([
            A.Resize(height=self.n_features, width=self.n_features),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # log the subfolder file pathes into df
    def _read_folder(self, path):
        grades = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        data = []
        for grade in grades:
            grade_path = os.path.join(path, grade)
            files = [os.path.join(grade_path, f) for f in os.listdir(grade_path) if f.endswith('.jpg')]
            for file in files:
                data.append({'path': file, 'label': int(grade)})
        df = pd.DataFrame(data)        
        df['label'] = df['label'].astype(int) + 1 # dummy class for the head
        return df

    def get_official_train_test(self):
        return (
            EyepacsDataset(self.train_df, self.n_classes, self.n_features, transform=self.train_transform),
            EyepacsDataset(self.test_df, self.n_classes, self.n_features, transform=self.test_transform)
        )


class EyepacsDataset(Dataset):
    def __init__(self, df, n_classes, n_features, transform=None):
        self.df = df
        self.transform = transform
        self.n_classes = n_classes
        self.n_features = n_features

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']

        return {
            'data': image,
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

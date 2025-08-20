import random
import numpy as np


class BasicAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def augment(self, x):
        raise NotImplementedError
    
    def __call__(self, x):
        if random.random() < self.p:
            return self.augment(x)
        return x
    
class RandomGaussianNoise(BasicAugmentation):
    def __init__(self, p=0.5, mean=0, std=0.1):
        super().__init__(p)
        self.mean = mean
        self.std = std

    def augment(self, x):
        return x + np.random.normal(self.mean, self.std, x.shape)

class RandomLaplaceNoise(BasicAugmentation):
    def __init__(self, p=0.5, loc=0.0, scale=0.01):

        super().__init__(p)
        self.loc = loc
        self.scale = scale

    def augment(self, x):
        # Generate Laplace noise
        noise = np.random.laplace(self.loc, self.scale, x.shape)
        return x + noise
    
class RandomFeatureMask(BasicAugmentation):
    def __init__(self, p=0.5, mask_prob=0.1):
        super().__init__(p)
        self.mask_prob = mask_prob

    def augment(self, x):
        mask = np.random.rand(*x.shape) < self.mask_prob
        x_aug = x.copy()
        x_aug[mask] = 0.0
        return x_aug

class RandomScaling(BasicAugmentation):
    def __init__(self, p=0.5, scale_range=(0.9, 1.1)):
        super().__init__(p)
        self.scale_range = scale_range

    def augment(self, x):
        scale = np.random.uniform(*self.scale_range, size=x.shape)
        return x * scale

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
class OneOf:
    def __init__(self, transforms: list, p: float = 0.5):
        self.transforms = transforms
        self.p = p
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, x):
        if random.random() < self.p:
            transform = np.random.choice(self.transforms, p=self.transforms_ps)
            return transform(x)
        return x

weak_aug = Compose([
    OneOf([
        RandomGaussianNoise(p=1.0, std=0.01),
        RandomLaplaceNoise(p=1.0, scale=0.01)
    ], p=0.7),
    RandomFeatureMask(p=0.5, mask_prob=0.05),
])

strong_aug = Compose([
    OneOf([
        RandomGaussianNoise(p=1.0, std=0.05),
        RandomLaplaceNoise(p=1.0, scale=0.05),
        RandomScaling(p=1.0, scale_range=(0.8, 1.2)),
    ], p=0.9),
    RandomFeatureMask(p=0.8, mask_prob=0.2),
])

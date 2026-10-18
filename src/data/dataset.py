import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import get_train_transforms, get_val_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_pairs(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    common = sorted(set(images) & set(masks))
    missing = (set(images) ^ set(masks))
    if missing:
        raise FileNotFoundError(f"Paires image/masque introuvables : {sorted(missing)[:5]}")
    if not common:
        raise FileNotFoundError(f"Aucune paire image/masque dans {images_dir} / {masks_dir}")
    return [(images[stem], masks[stem]) for stem in common]


class KvasirDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform if transform is not None else get_val_transforms()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Lecture impossible : {image_path} / {mask_path}")
        mask = (mask > 127).astype(np.float32)

        sample = self.transform(image=image, mask=mask)
        image = sample["image"]
        mask = sample["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask


def get_datasets(images_dir, masks_dir, train_size=0.7, val_size=0.1, seed=42):
    pairs = discover_pairs(images_dir, masks_dir)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    train_ds = KvasirDataset(train_pairs, transform=get_train_transforms())
    val_ds = KvasirDataset(val_pairs, transform=get_val_transforms())
    test_ds = KvasirDataset(test_pairs, transform=get_val_transforms())
    return train_ds, val_ds, test_ds

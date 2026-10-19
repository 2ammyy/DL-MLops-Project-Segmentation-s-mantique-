import numpy as np
import pytest
import torch

from src.data.dataset import KvasirDataset, discover_pairs, get_datasets


def test_split_sizes(fake_data_dir):
    train_ds, val_ds, test_ds = get_datasets(fake_data_dir["images"], fake_data_dir["masks"])
    assert (len(train_ds), len(val_ds), len(test_ds)) == (7, 1, 2)


def test_split_reproducible(fake_data_dir):
    a = get_datasets(fake_data_dir["images"], fake_data_dir["masks"], seed=42)
    b = get_datasets(fake_data_dir["images"], fake_data_dir["masks"], seed=42)
    assert a[0].pairs == b[0].pairs and a[2].pairs == b[2].pairs


def test_sample_shapes_and_dtype(datasets):
    image, mask = datasets[0][0]
    assert image.shape == (3, 256, 256)
    assert image.dtype == torch.float32
    assert mask.shape == (1, 256, 256)
    assert mask.dtype == torch.float32


def test_mask_binary_values(datasets):
    _, mask = datasets[0][0]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_discover_pairs_missing_mask(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "orphan.jpg").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        discover_pairs(images, masks)


def test_dataset_len(datasets):
    train_ds = datasets[0]
    assert train_ds.__len__() == 7

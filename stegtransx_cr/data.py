"""Dataset pipeline.

Training uses DIV2K train HR only (800 images), as in the paper.

Val/test use one held-out dataset from the paper. The paper lists both
ImageNet and COCO; this implementation uses COCO val2017 only (smaller,
public, no ImageNet licence). Val and test pairs are disjoint in *images*,
not only in pair index, so nothing from the test set is seen at validation.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DIV2K_TRAIN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_TRAIN_DIR = "DIV2K_train_HR"

# Paper val/test lists ImageNet and COCO; we keep COCO only.
COCO_VAL_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_DIR = "val2017"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Download and extraction
# --------------------------------------------------------------------------- #
def _report(block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(block * block_size, total)
    percent = 100.0 * done / total
    sys.stdout.write(f"\r  {percent:5.1f}%  ({done / 1e6:.0f} / {total / 1e6:.0f} MB)")
    sys.stdout.flush()


def download_and_extract(url: str, destination: str, expected_dir: str) -> str:
    """Download a zip archive once and extract it into ``destination``."""
    target = os.path.join(destination, expected_dir)
    if os.path.isdir(target) and list_images(target):
        print(f"  {expected_dir}: already present")
        return target

    os.makedirs(destination, exist_ok=True)
    archive = os.path.join(destination, os.path.basename(url))
    if not os.path.exists(archive):
        print(f"  downloading {os.path.basename(url)}")
        urllib.request.urlretrieve(url, archive, reporthook=_report)
        print()
    print(f"  extracting {os.path.basename(url)}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
    try:
        os.remove(archive)
    except OSError:
        pass

    if os.path.isdir(target) and list_images(target):
        return target
    found = _find_image_dir(destination, expected_dir)
    if found:
        return found
    raise FileNotFoundError(f"extracted {url} but found no images under {destination}")


def _find_image_dir(root: str, preferred: str) -> Optional[str]:
    """Locate ``preferred`` only. Never fall back to a sibling dataset folder."""
    preferred_path = os.path.join(root, preferred)
    if os.path.isdir(preferred_path) and list_images(preferred_path):
        return preferred_path
    for dirpath, _, _ in os.walk(root):
        if os.path.basename(dirpath) == preferred and list_images(dirpath):
            return dirpath
    return None


def download_div2k_train(data_dir: str) -> str:
    """DIV2K train HR only — used exclusively to train the hiding/reveal nets."""
    print("DIV2K train:")
    return download_and_extract(DIV2K_TRAIN_URL, data_dir, DIV2K_TRAIN_DIR)


def download_coco_val(data_dir: str) -> str:
    """COCO val2017 — used exclusively for validation and test pairs."""
    print("COCO val2017 (val/test only):")
    return download_and_extract(COCO_VAL_URL, data_dir, COCO_VAL_DIR)


def list_images(directory: str) -> List[str]:
    files = []
    for root, _, names in os.walk(directory):
        for name in sorted(names):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                files.append(os.path.join(root, name))
    return sorted(files)


# --------------------------------------------------------------------------- #
# Preprocessing and RAM cache
# --------------------------------------------------------------------------- #
def load_and_resize(path: str, size: int) -> np.ndarray:
    """Centre-crop to a square and resize to ``size`` x ``size``."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((size, size), Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def build_cache(
    directory: str,
    cache_path: str,
    size: int = 256,
    limit: Optional[int] = None,
    rebuild: bool = False,
) -> np.ndarray:
    """Return an (N, size, size, 3) uint8 array, building and saving it once."""
    if os.path.exists(cache_path) and not rebuild:
        cache = np.load(cache_path)
        if limit is None or cache.shape[0] >= limit:
            return cache[:limit] if limit else cache

    files = list_images(directory)
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no images found in {directory}")

    print(f"  caching {len(files)} images from {os.path.basename(directory)}")
    stacked = np.zeros((len(files), size, size, 3), dtype=np.uint8)
    for index, path in enumerate(files):
        stacked[index] = load_and_resize(path, size)
        if (index + 1) % 100 == 0:
            print(f"    {index + 1}/{len(files)}")
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, stacked)
    return stacked


def make_synthetic_cache(count: int, size: int = 256, seed: int = 42) -> np.ndarray:
    """Smooth synthetic images, used only when DIV2K cannot be downloaded.

    Results produced on this data are not comparable with the paper; it exists
    so the notebook can be executed end to end without network access.
    """
    rng = np.random.default_rng(seed)
    low = rng.random((count, 3, 16, 16), dtype=np.float32)
    tensor = torch.from_numpy(low)
    upsampled = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bicubic", align_corners=False
    ).clamp(0, 1)
    array = (upsampled.numpy() * 255.0).round().astype(np.uint8)
    return array.transpose(0, 2, 3, 1)


# --------------------------------------------------------------------------- #
# Pair splits
# --------------------------------------------------------------------------- #
def build_pairs(num_images: int, num_pairs: int, seed: int) -> List[Tuple[int, int]]:
    """Deterministic list of distinct (cover, secret) index pairs."""
    rng = np.random.default_rng(seed)
    seen = set()
    pairs: List[Tuple[int, int]] = []
    if num_images < 2:
        raise ValueError("need at least two images to build cover/secret pairs")
    max_unique = num_images * (num_images - 1)
    if num_pairs > max_unique:
        raise ValueError(f"cannot draw {num_pairs} ordered pairs from {num_images} images")
    attempts = 0
    while len(pairs) < num_pairs:
        i, j = int(rng.integers(num_images)), int(rng.integers(num_images))
        attempts += 1
        if attempts > max_unique * 20:
            raise RuntimeError("failed to sample unique cover/secret pairs")
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j))
    return pairs


def save_splits(path: str, splits: Dict[str, List[Tuple[int, int]]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({key: [list(pair) for pair in value] for key, value in splits.items()}, handle, indent=2)


def load_splits(path: str) -> Dict[str, List[Tuple[int, int]]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    out: Dict[str, List[Tuple[int, int]]] = {}
    for key in ("val", "test"):
        out[key] = [tuple(pair) for pair in raw[key]]
    return out


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def _to_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor.float().div_(255.0)


class TrainPairDataset(Dataset):
    """Random cover/secret pairs drawn from the cached training images.

    One epoch visits every image once as a cover; the secret is resampled each
    time, so the model sees a large slice of the N * (N - 1) pair space.
    """

    def __init__(self, cache: np.ndarray, seed: int = 42, augment: bool = False) -> None:
        self.cache = cache
        self.seed = seed
        self.augment = augment
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-seeds the partner sampling so epochs differ but stay reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.cache.shape[0]

    def __getitem__(self, index: int):
        rng = np.random.default_rng((self.seed, self.epoch, index))
        partner = int(rng.integers(self.cache.shape[0] - 1))
        if partner >= index:
            partner += 1

        cover = _to_tensor(self.cache[index])
        secret = _to_tensor(self.cache[partner])

        if self.augment:
            if rng.random() < 0.5:
                cover, secret = cover.flip(-1), secret.flip(-1)
            if rng.random() < 0.5:
                cover, secret = cover.flip(-2), secret.flip(-2)
        return cover, secret


class FixedPairDataset(Dataset):
    """A frozen list of (cover, secret) index pairs for validation and test."""

    def __init__(self, cache: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> None:
        self.cache = cache
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        i, j = self.pairs[index]
        return _to_tensor(self.cache[i]), _to_tensor(self.cache[j])


@dataclass
class DataBundle:
    """Everything the training and evaluation code needs."""

    train: TrainPairDataset
    val: FixedPairDataset
    test: FixedPairDataset
    train_cache: np.ndarray
    eval_cache: np.ndarray
    splits_path: str

    def loaders(
        self,
        batch_size: int,
        num_workers: int = 0,
        eval_batch_size: int = 16,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pin = torch.cuda.is_available()
        drop_last = len(self.train) >= batch_size
        train_loader = DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            self.val, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin
        )
        test_loader = DataLoader(
            self.test, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin
        )
        return train_loader, val_loader, test_loader


def prepare_data(
    data_dir: str,
    image_size: int = 256,
    num_train_images: int = 800,
    num_val_pairs: int = 50,
    num_test_pairs: int = 200,
    num_val_images: int = 80,
    num_test_images: int = 220,
    seed: int = 42,
    augment: bool = False,
    allow_synthetic: bool = False,
) -> DataBundle:
    """Train on DIV2K. Val/test on COCO (not ImageNet, not DIV2K-valid)."""
    os.makedirs(data_dir, exist_ok=True)
    train_cache_path = os.path.join(data_dir, f"div2k_train_{image_size}.npy")
    eval_cache_path = os.path.join(data_dir, f"coco_val_{image_size}.npy")
    splits_path = os.path.join(data_dir, "splits.json")
    need_eval = num_val_images + num_test_images

    try:
        train_dir = download_div2k_train(data_dir)
        train_cache = build_cache(train_dir, train_cache_path, image_size, limit=num_train_images)
        coco_dir = download_coco_val(data_dir)
        eval_cache = build_cache(coco_dir, eval_cache_path, image_size, limit=need_eval)
    except Exception as error:
        if not allow_synthetic:
            raise
        print(f"  download failed ({error}); falling back to synthetic images")
        train_cache = make_synthetic_cache(num_train_images, image_size, seed)
        eval_cache = make_synthetic_cache(need_eval, image_size, seed + 1)

    if eval_cache.shape[0] < need_eval:
        raise ValueError(
            f"need {need_eval} COCO images for val/test, got {eval_cache.shape[0]}"
        )

    val_pairs = build_pairs(num_val_images, num_val_pairs, seed)
    test_local = build_pairs(num_test_images, num_test_pairs, seed + 1)
    test_pairs = [(i + num_val_images, j + num_val_images) for i, j in test_local]
    for i, j in val_pairs + test_pairs:
        if i >= eval_cache.shape[0] or j >= eval_cache.shape[0]:
            raise IndexError("val/test pair index is outside the COCO cache")
    splits = {
        "dataset": "COCO val2017",
        "train": "DIV2K train HR",
        "val_images": f"0:{num_val_images}",
        "test_images": f"{num_val_images}:{need_eval}",
        "val": [list(p) for p in val_pairs],
        "test": [list(p) for p in test_pairs],
    }
    os.makedirs(os.path.dirname(os.path.abspath(splits_path)), exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)

    print(
        f"  train DIV2K {train_cache.shape[0]} images | "
        f"val COCO {num_val_images} images / {num_val_pairs} pairs | "
        f"test COCO {num_test_images} images / {num_test_pairs} pairs"
    )

    return DataBundle(
        train=TrainPairDataset(train_cache, seed=seed, augment=augment),
        val=FixedPairDataset(eval_cache, val_pairs),
        test=FixedPairDataset(eval_cache, test_pairs),
        train_cache=train_cache,
        eval_cache=eval_cache,
        splits_path=splits_path,
    )

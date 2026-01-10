from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from .augmentation import get_augmentation_transforms
from .preprocessing import get_preprocess_transforms, IMAGENET_MEAN, IMAGENET_STD


def _build_transforms(cfg: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns:
      train_tf (with augmentation + preprocess)
      eval_tf  (preprocess only)
    """
    aug_tf = get_augmentation_transforms(cfg)          # may be None
    eval_tf = get_preprocess_transforms(cfg)

    # Shared tail
    tail = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=float(cfg.get("augmentation", {}).get("random_erasing_p", 0.0))),
    ])

    if aug_tf is None:
        train_tf = transforms.Compose([
            transforms.Resize((int(cfg["data"]["resize_size"]), int(cfg["data"]["resize_size"]))),
            transforms.CenterCrop((int(cfg["data"]["image_size"]), int(cfg["data"]["image_size"]))),
            tail,
        ])
    else:
        train_tf = transforms.Compose([
            aug_tf,
            tail,
        ])

    return train_tf, eval_tf



def get_dataloaders(cfg: Dict[str, Any]):
    """
    Returns:
      train_loader, val_loader, test_loader, meta
    """
    data_path = cfg["paths"]["data_path"]
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    val_split = float(cfg["data"]["val_split"])
    pin_memory = bool(cfg["data"].get("pin_memory", True))

    seed = int(cfg["experiment"]["seed"])
    g = torch.Generator().manual_seed(seed)

    train_tf, eval_tf = _build_transforms(cfg)

    # Trainval split from torchvision dataset
    full_trainval_train_tf = OxfordIIITPet(
        root=data_path,
        split="trainval",
        target_types="category",
        download=True,
        transform=train_tf,
    )

    test_ds = OxfordIIITPet(
        root=data_path,
        split="test",
        target_types="category",
        download=True,
        transform=eval_tf,
    )

    # --- Stratified val split (per class) ---
    # Build a "label-only" dataset (no transforms needed) to get targets
    label_ds = OxfordIIITPet(
        root=data_path,
        split="trainval",
        target_types="category",
        download=False,
        transform=None,
    )

    num_classes = int(cfg["model"]["num_classes"])
    indices_by_class = [[] for _ in range(num_classes)]

    for idx in range(len(label_ds)):
        _, y = label_ds[idx]
        indices_by_class[int(y)].append(idx)

    val_indices = []
    train_indices = []

    for c in range(num_classes):
        cls_idx = indices_by_class[c]
        # deterministic shuffle per class
        cls_idx = torch.tensor(cls_idx)
        perm = cls_idx[torch.randperm(len(cls_idx), generator=g)].tolist()

        n_val_c = max(1, int(round(len(perm) * val_split)))
        val_indices.extend(perm[:n_val_c])
        train_indices.extend(perm[n_val_c:])

    # Train subset uses train_tf (aug)
    train_subset = Subset(full_trainval_train_tf, train_indices)

    # Val subset uses eval_tf (no aug)
    full_trainval_eval_tf = OxfordIIITPet(
        root=data_path,
        split="trainval",
        target_types="category",
        download=False,
        transform=eval_tf,
    )
    val_ds = Subset(full_trainval_eval_tf, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    meta = {
        "input_shape": (3, int(cfg["data"]["image_size"]), int(cfg["data"]["image_size"])),
        "num_classes": int(cfg["model"]["num_classes"]),
        "n_train": len(train_subset),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }

    return train_loader, val_loader, test_loader, meta

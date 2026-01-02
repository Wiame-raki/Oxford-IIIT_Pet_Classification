import os
import math
import argparse
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.datasets import OxfordIIITPet


# repo imports
from src.utils import load_config, set_seed, get_device
from src.data_loading import get_dataloaders
from src.model import PetSE_CNN
from src.preprocessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms


def save_grid_image(tensor_chw, out_path, nrow=8, title=None):
    """
    tensor_chw: (B,C,H,W) float tensor in [0,1] OR normalized; we will min-max per image for display.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # For visualization: map each image to [0,1] by per-image min/max (works even if normalized)
    x = tensor_chw.clone()
    b = x.size(0)
    x2 = []
    for i in range(b):
        xi = x[i]
        mn = float(xi.min().item())
        mx = float(xi.max().item())
        if mx - mn < 1e-12:
            x2.append(torch.zeros_like(xi))
        else:
            x2.append((xi - mn) / (mx - mn))
    x = torch.stack(x2, dim=0)

    grid = make_grid(x, nrow=nrow)
    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(12, 8))
    if title:
        plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_class_distribution(labels, num_classes, out_path, title):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.bar(range(num_classes), counts)
    plt.title(title)
    plt.xlabel("Class id")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return counts


@torch.no_grad()
def compute_majority_baseline(loader, num_classes):
    # majority class based on targets in loader
    all_y = []
    for _, y in loader:
        all_y.append(y)
    y = torch.cat(all_y, dim=0).cpu()

    counts = torch.bincount(y, minlength=num_classes)
    maj = int(counts.argmax().item())
    acc = float((y == maj).float().mean().item())
    return maj, acc, counts


@torch.no_grad()
def compute_random_uniform_expected(num_classes):
    acc1 = 1.0 / num_classes
    acc5 = min(5, num_classes) / num_classes
    return acc1, acc5


@torch.no_grad()
def initial_batch_loss(cfg):
    device = get_device()
    train_loader, _, _, _ = get_dataloaders(cfg)

    model = PetSE_CNN(
        num_classes=int(cfg["model"]["num_classes"]),
        reduction=int(cfg["model"]["reduction"]),
        blocks_per_stage=int(cfg["model"]["blocks_per_stage"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)))

    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    loss = float(criterion(model(x), y).item())
    return loss



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out_dir", default="report/figures")
    ap.add_argument("--n_samples", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Dataloaders + meta (sizes)
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)
    num_classes = int(cfg["model"]["num_classes"])

    # Print key metadata for copy/paste
    print("\n=== META ===")
    print(meta)

    # Parameter counts
    model = PetSE_CNN(
        num_classes=num_classes,
        reduction=int(cfg["model"]["reduction"]),
        blocks_per_stage=int(cfg["model"]["blocks_per_stage"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== PARAMS ===")
    print("total_params:", total_params)
    print("trainable_params:", trainable_params)

    # Initial expected loss
    expected = math.log(num_classes)
    observed = initial_batch_loss(cfg)
    print("\n=== INITIAL LOSS ===")
    print("expected ~ log(C):", expected)
    print("observed on 1st batch:", observed)

    # Baselines
    maj_val, majacc_val, counts_val = compute_majority_baseline(val_loader, num_classes)
    maj_test, majacc_test, counts_test = compute_majority_baseline(test_loader, num_classes)
    rand_acc1, rand_acc5 = compute_random_uniform_expected(num_classes)
    print("\n=== BASELINES ===")
    print(f"majority(val): class={maj_val} acc={majacc_val:.4f}")
    print(f"majority(test): class={maj_test} acc={majacc_test:.4f}")
    print(f"random uniform expected: acc1={rand_acc1:.4f} acc5={rand_acc5:.4f}")

    # Class distribution plots (val/test are easy from loaders)
    plot_class_distribution(counts_val.argmax().repeat(0), num_classes, os.path.join(out_dir, "tmp.png"), "tmp")  # no-op to ensure import ok

    # Better: build distributions from raw labels in trainval/test datasets
    label_ds_trainval = OxfordIIITPet(
        root=cfg["paths"]["data_path"],
        split="trainval",
        target_types="category",
        download=False,
        transform=None,
    )
    label_ds_test = OxfordIIITPet(
        root=cfg["paths"]["data_path"],
        split="test",
        target_types="category",
        download=False,
        transform=None,
    )

    trainval_labels = [int(label_ds_trainval[i][1]) for i in range(len(label_ds_trainval))]
    test_labels = [int(label_ds_test[i][1]) for i in range(len(label_ds_test))]

    plot_class_distribution(trainval_labels, num_classes, os.path.join(out_dir, "class_dist_trainval.png"),
                            "Class distribution (trainval)")
    plot_class_distribution(test_labels, num_classes, os.path.join(out_dir, "class_dist_test.png"),
                            "Class distribution (test)")

    # Dataset samples AFTER transforms (preprocess vs augmentation)
    eval_tf = get_preprocess_transforms(cfg)
    aug_tf = get_augmentation_transforms(cfg)

    ds_eval = OxfordIIITPet(
        root=cfg["paths"]["data_path"],
        split="trainval",
        target_types="category",
        download=False,
        transform=eval_tf,
    )

    # For augmentation view, use augmentation + ToTensor/Normalize exactly like train pipeline does
    # We'll reuse your actual train dataset through get_dataloaders by sampling one batch from train_loader.
    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))

    save_grid_image(
        x_train[: args.n_samples],
        os.path.join(out_dir, "samples_train_after_aug.png"),
        nrow=8,
        title="Train samples (after augmentation + normalize)",
    )
    save_grid_image(
        x_val[: args.n_samples],
        os.path.join(out_dir, "samples_val_after_preprocess.png"),
        nrow=8,
        title="Val samples (after preprocess + normalize)",
    )

    # Also save a "raw-ish" look: apply only resize+centercrop+ToTensor (no normalize)
    from torchvision import transforms
    img_size = int(cfg["data"]["image_size"])
    resize_size = int(cfg["data"]["resize_size"])
    eval_nonorm = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ds_nonorm = OxfordIIITPet(
        root=cfg["paths"]["data_path"],
        split="trainval",
        target_types="category",
        download=False,
        transform=eval_nonorm,
    )
    xs = torch.stack([ds_nonorm[i][0] for i in range(min(args.n_samples, len(ds_nonorm)))], dim=0)
    save_grid_image(xs, os.path.join(out_dir, "samples_eval_no_norm.png"), nrow=8, title="Samples (preprocess without normalize)")

    print("\nSaved figures into:", out_dir)
    print(" - class_dist_trainval.png / class_dist_test.png")
    print(" - samples_train_after_aug.png / samples_val_after_preprocess.png / samples_eval_no_norm.png")


if __name__ == "__main__":
    main()

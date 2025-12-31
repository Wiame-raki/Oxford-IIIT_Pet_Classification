import argparse
import math
import os
from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data_loading import get_dataloaders
from .model import PetSE_CNN
from .utils import get_device, get_paths, load_config, set_seed


def lr_finder(cfg: Dict[str, Any], experiment_name: str = "lr_finder") -> None:
    paths = get_paths(cfg)
    device = get_device()

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    run_dir = os.path.join(paths["runs_dir"], experiment_name)
    writer = SummaryWriter(log_dir=run_dir)

    # Use the same training loader (still a bit noisy), but we cap batches and early stop
    train_loader, _, _, _ = get_dataloaders(cfg)

    model = PetSE_CNN(
        num_classes=int(cfg["model"]["num_classes"]),
        reduction=int(cfg["model"]["reduction"]),
        blocks_per_stage=int(cfg["model"]["blocks_per_stage"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    lrcfg = cfg.get("lr_finder", {})
    lr_min = float(lrcfg.get("lr_min", 1e-7))
    lr_max = float(lrcfg.get("lr_max", 3e-1))
    max_batches = int(lrcfg.get("max_batches", 200))
    divergence_factor = float(lrcfg.get("divergence_factor", 4.0))
    warmup_batches = int(lrcfg.get("warmup_batches", 10))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_min,
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    model.train()

    best_loss = float("inf")
    step = 0

    # Determine how many batches we will actually use
    total_batches = min(len(train_loader), max_batches)
    if total_batches < 2:
        raise RuntimeError("Not enough batches for LR finder.")

    pbar = tqdm(train_loader, desc="LR Finder", leave=False)
    for i, (x, y) in enumerate(pbar):
        if i >= total_batches:
            break

        # Exponential schedule lr_min -> lr_max across total_batches
        t = i / (total_batches - 1)
        lr = lr_min * (lr_max / lr_min) ** t
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        if not math.isfinite(loss.item()):
            break

        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        writer.add_scalar("lr_finder/loss", loss_val, i)
        writer.add_scalar("lr_finder/lr", float(lr), i)

        # Track best loss and early stop if diverging badly
        if loss_val < best_loss:
            best_loss = loss_val

        if i >= warmup_batches and loss_val > divergence_factor * best_loss:
            # stop once clearly exploding (cleaner plot)
            break

        step += 1
        pbar.set_postfix(lr=float(lr), loss=loss_val)

    writer.close()
    print(f"LR finder logs written to: {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    lr_finder(cfg)


if __name__ == "__main__":
    main()

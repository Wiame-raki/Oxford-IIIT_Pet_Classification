import argparse
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data_loading import get_dataloaders
from .model import PetSE_CNN
from .utils import (
    get_device,
    get_paths,
    load_config,
    set_seed,
    topk_correct,
)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

        topk = topk_correct(logits, y, ks=(1, 5))
        correct1 += topk[1]
        correct5 += topk[5]

    return (
        total_loss / max(1, total),
        correct1 / max(1, total),
        correct5 / max(1, total),
    )


def train(
    cfg: Dict[str, Any],
    experiment_name: Optional[str] = None,
    overfit_batch: bool = False,
    checkpoint_path: Optional[str] = None,
) -> str:
    paths = get_paths(cfg)
    device = get_device()

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    if experiment_name is None:
        experiment_name = str(cfg["experiment"]["name"])

    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    run_dir = os.path.join(paths["runs_dir"], experiment_name)
    writer = SummaryWriter(run_dir)

    train_loader, val_loader, _, meta = get_dataloaders(cfg)

    model = PetSE_CNN(
        num_classes=int(cfg["model"]["num_classes"]),
        reduction=int(cfg["model"]["reduction"]),
        blocks_per_stage=int(cfg["model"]["blocks_per_stage"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    # Cast label_smoothing safely
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # -------------------------
    # Overfit one batch sanity check
    # -------------------------
    if overfit_batch:
        ckpt = os.path.join(paths["artifacts_dir"], "overfit_batch.ckpt")

        if hasattr(model, "dropout"):
            model.dropout.p = 0.0

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=0.0)

        x0, y0 = next(iter(train_loader))
        x0, y0 = x0.to(device), y0.to(device)

        model.train()
        for ep in range(1, 51):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x0), y0)
            loss.backward()
            optimizer.step()

        torch.save({"model_state": model.state_dict()}, ckpt)
        print("Saved overfit batch checkpoint:", ckpt)
        writer.close()
        return ckpt

    # -------------------------
    # Optimizer (cast everything!)
    # -------------------------
    base_lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # -------------------------
    # Warmup + Cosine scheduler (cast everything!)
    # -------------------------
    epochs = int(cfg["training"]["epochs"])
    warmup_epochs = int(cfg["training"].get("warmup_epochs", 5))
    warmup_start_lr = float(cfg["training"].get("warmup_start_lr", 1e-4))
    min_lr = float(cfg["training"].get("min_lr", 1e-5))

    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    if warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be < total epochs")
    if warmup_start_lr <= 0 or base_lr <= 0:
        raise ValueError("Learning rates must be > 0")
    if warmup_start_lr > base_lr:
        raise ValueError("warmup_start_lr must be <= lr")
    if min_lr < 0:
        raise ValueError("min_lr must be >= 0")

    # Warmup: warmup_start_lr -> base_lr (linear)
    start_factor = warmup_start_lr / base_lr
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Cosine: base_lr -> min_lr over remaining epochs
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=min_lr,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    # -------------------------
    # Checkpoint path
    # -------------------------
    if checkpoint_path is None:
        checkpoint_path = os.path.join(paths["artifacts_dir"], f"{experiment_name}_best.ckpt")

    # -------------------------
    # Training loop
    # -------------------------
    best_val_acc1 = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_loss += float(loss.item()) * bs
            total += bs
            correct += (logits.argmax(dim=1) == y).sum().item()

            pbar.set_postfix(loss=float(loss.item()))

        train_loss = running_loss / max(1, total)
        train_acc1 = correct / max(1, total)

        val_loss, val_acc1, val_acc5 = evaluate_epoch(model, val_loader, criterion, device)

        # Step scheduler once per epoch
        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"lr={lr_now:.2e} | "
            f"train: loss={train_loss:.4f}, acc1={train_acc1:.4f} | "
            f"val: loss={val_loss:.4f}, acc1={val_acc1:.4f}, acc5={val_acc5:.4f}"
        )

        # TensorBoard logs (optional)
        writer.add_scalar("lr", lr_now, epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/acc1", train_acc1, epoch)
        writer.add_scalar("val/acc1", val_acc1, epoch)
        writer.add_scalar("val/acc5", val_acc5, epoch)

        # Save best by val acc@1
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_acc1": float(val_acc1),
                    "val_acc5": float(val_acc5),
                    "meta": meta,
                },
                checkpoint_path,
            )

    print(f"\nBest val acc@1 = {best_val_acc1:.4f} @ epoch {best_epoch}")
    writer.close()
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--overfit_batch", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = train(cfg, experiment_name=args.experiment_name, overfit_batch=args.overfit_batch)
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()

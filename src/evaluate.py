import argparse
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .data_loading import get_dataloaders
from .model import PetSE_CNN
from .utils import get_device, load_config, topk_correct


@torch.no_grad()
def evaluate(cfg: Dict[str, Any], checkpoint_path: str) -> Tuple[float, float, float]:
    device = get_device()
    _, _, test_loader, meta = get_dataloaders(cfg)

    model = PetSE_CNN(
        num_classes=int(cfg["model"]["num_classes"]),
        reduction=int(cfg["model"]["reduction"]),
        blocks_per_stage=int(cfg["model"]["blocks_per_stage"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)


    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    criterion = nn.CrossEntropyLoss()

    num_classes = int(cfg["model"]["num_classes"])
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0

    for x, y in tqdm(test_loader, desc="Evaluating", leave=False):
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

        preds = logits.argmax(dim=1)
        for cls in range(num_classes):
            mask = (y == cls)
            if mask.any():
                class_total[cls] += mask.sum().item()
                class_correct[cls] += (preds[mask] == y[mask]).sum().item()

    avg_loss = total_loss / max(1, total)
    acc1 = correct1 / max(1, total)
    acc5 = correct5 / max(1, total)

    # Per-class accuracy (avoid division by 0)
    cls_accs = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        if class_total[c].item() > 0:
            cls_accs[c] = class_correct[c].float() / class_total[c].float()
        else:
            cls_accs[c] = float("nan")

    print(f"Test size: {meta['n_test']}")
    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test acc@1: {acc1:.4f}")
    print(f"Test acc@5: {acc5:.4f}")

    # Show best/worst classes by accuracy
    k = 5
    valid_mask = torch.isfinite(cls_accs)
    valid_idx = torch.where(valid_mask)[0]

    if valid_idx.numel() > 0:
        valid_accs = cls_accs[valid_idx]
        order = torch.argsort(valid_accs)  # ascending

        # Worst k
        worst_local = order[: min(k, order.numel())]
        worst_idx = valid_idx[worst_local]

        # Best k (flip instead of [::-1])
        best_local = order[-min(k, order.numel()):].flip(0)
        best_idx = valid_idx[best_local]

        print("\nWorst-5 classes (class_id, acc):")
        for c in worst_idx.tolist():
            tot = int(class_total[c].item())
            cor = int(class_correct[c].item())
            acc = float(cls_accs[c].item())
            print(f"  {c:02d}: {acc:.3f}  ({cor}/{tot})")

        print("\nBest-5 classes (class_id, acc):")
        for c in best_idx.tolist():
            tot = int(class_total[c].item())
            cor = int(class_correct[c].item())
            acc = float(cls_accs[c].item())
            print(f"  {c:02d}: {acc:.3f}  ({cor}/{tot})")

    return avg_loss, acc1, acc5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()

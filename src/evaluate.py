# src/evaluate.py
import argparse
import torch
import torch.nn as nn

from .data_loading import get_dataloaders
from .model import PetSE_CNN
from .utils import get_device, load_config
from .metrics import (
    confusion_matrix,
    f1_from_confusion,
    balanced_accuracy_from_confusion,
    expected_calibration_error,
    top_confusion_pairs,
)


@torch.no_grad()
def evaluate(cfg, checkpoint_path: str):
    device = get_device()
    print(f"Using device: {device}")

    # -------------------------
    # Load checkpoint
    # -------------------------
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state"]

    # -------------------------
    # Data
    # -------------------------
    _, val_loader, test_loader, meta = get_dataloaders(cfg)

    num_classes = int(cfg["model"]["num_classes"])

    # -------------------------
    # Model
    # -------------------------
    model = PetSE_CNN(
        num_classes=num_classes,
        reduction=int(cfg["model"]["reduction"]),
        blocks_per_stage=int(cfg["model"]["blocks_per_stage"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Evaluation loop
    # -------------------------
    all_probs = []
    all_targets = []
    all_preds = []

    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # top-k
        top5 = logits.topk(5, dim=1).indices
        correct1 += (preds == y).sum().item()
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())
        all_preds.append(preds.cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = torch.cat(all_preds, dim=0)

    # -------------------------
    # Metrics
    # -------------------------
    cm = confusion_matrix(num_classes, targets, preds)
    macro_f1, weighted_f1, macro_prec, macro_rec = f1_from_confusion(cm)
    bal_acc = balanced_accuracy_from_confusion(cm)
    ece = expected_calibration_error(probs, targets, n_bins=15)
    top_pairs = top_confusion_pairs(cm, k=10)

    # -------------------------
    # Print results
    # -------------------------
    print("\n=== Evaluation results (TEST) ===")
    print(f"Loss           : {total_loss / total:.4f}")
    print(f"Acc@1          : {correct1 / total:.4f}")
    print(f"Acc@5          : {correct5 / total:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Balanced Acc   : {bal_acc:.4f}")
    print(f"Macro Precision: {macro_prec:.4f}")
    print(f"Macro Recall   : {macro_rec:.4f}")
    print(f"ECE            : {ece:.4f}")

    print("\nTop confusion pairs [true, pred, count]:")
    for t, p, c in top_pairs.tolist():
        print(f"  class {t} → {p} : {c}")

    return {
        "loss": total_loss / total,
        "acc1": correct1 / total,
        "acc5": correct5 / total,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_acc": bal_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "ece": ece,
        "confusion_matrix": cm,
        "top_confusions": top_pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()

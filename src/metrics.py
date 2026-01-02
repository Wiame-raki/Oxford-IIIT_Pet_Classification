# src/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@torch.no_grad()
def confusion_matrix(num_classes: int, targets: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """
    Returns cm (C,C) where cm[t, p] += 1
    targets/preds: shape (N,)
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device="cpu")
    t = targets.view(-1).to("cpu")
    p = preds.view(-1).to("cpu")
    for i in range(t.numel()):
        cm[int(t[i]), int(p[i])] += 1
    return cm


@torch.no_grad()
def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Returns: (macro_f1, weighted_f1, macro_precision, macro_recall)
    """
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    support = cm.sum(dim=1)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    macro_f1 = torch.nanmean(f1).item()
    weighted_f1 = (f1 * support / (support.sum() + 1e-12)).sum().item()
    macro_precision = torch.nanmean(precision).item()
    macro_recall = torch.nanmean(recall).item()
    return macro_f1, weighted_f1, macro_precision, macro_recall


@torch.no_grad()
def balanced_accuracy_from_confusion(cm: torch.Tensor) -> float:
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    recall = tp / (support + 1e-12)
    return torch.nanmean(recall).item()


@torch.no_grad()
def expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    probs: (N, C) softmax probabilities
    targets: (N,) int64
    """
    device = probs.device
    conf, pred = probs.max(dim=1)          # (N,)
    correct = (pred == targets).float()    # (N,)

    bins = torch.linspace(0, 1, steps=n_bins + 1, device=device)
    ece = torch.zeros((), device=device)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            w = mask.float().mean()
            ece += w * torch.abs(acc_bin - conf_bin)

    return float(ece.item())


@torch.no_grad()
def top_confusion_pairs(cm: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Returns tensor of shape (k, 3): [true, pred, count] excluding diagonal.
    """
    cm2 = cm.clone()
    cm2.fill_diagonal_(0)
    flat = cm2.view(-1)
    vals, idx = torch.topk(flat, k=min(k, flat.numel()))
    out = []
    C = cm.size(0)
    for v, j in zip(vals.tolist(), idx.tolist()):
        t = j // C
        p = j % C
        out.append([t, p, v])
    return torch.tensor(out, dtype=torch.long)

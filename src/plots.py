# src/plots.py
from __future__ import annotations
from typing import List, Optional
import os

import numpy as np
import matplotlib.pyplot as plt
import torch


def save_confusion_matrix_png(cm: torch.Tensor, out_path: str, class_names: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm_np = cm.cpu().numpy().astype(np.int64)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_np, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    if class_names is not None and len(class_names) == cm_np.shape[0]:
        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=90, fontsize=6)
        plt.yticks(ticks, class_names, fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_reliability_diagram(bin_acc: List[float], bin_conf: List[float], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x = np.array(bin_conf, dtype=np.float32)
    y = np.array(bin_acc, dtype=np.float32)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1])
    plt.plot(x, y, marker="o")
    plt.title("Reliability Diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

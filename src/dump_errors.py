# src/dump_errors.py
import argparse
import os
from typing import Any, Dict, List, Tuple

import torch
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

from .utils import load_config, get_device, get_paths
from .model import PetSE_CNN
from .preprocessing import get_preprocess_transforms


@torch.no_grad()
def dump_worst(cfg: Dict[str, Any], checkpoint: str, out_csv: str, top_n: int = 100) -> None:
    device = get_device()
    paths = get_paths(cfg)

    # IMPORTANT: use deterministic eval transforms and keep access to original samples/indices
    eval_tf = get_preprocess_transforms(cfg)
    ds = OxfordIIITPet(
        root=cfg["paths"]["data_path"],
        split="test",
        target_types="category",
        download=False,
        transform=eval_tf,
    )

    model = PetSE_CNN(
        num_classes=int(cfg["model"]["num_classes"]),
        reduction=int(cfg["model"]["reduction"]),
        blocks_per_stage=int(cfg["model"]["blocks_per_stage"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Collect (loss_proxy, idx, true, pred, conf_true, conf_pred)
    rows: List[Tuple[float, int, int, int, float, float]] = []

    for i in tqdm(range(len(ds)), desc="Scanning test"):
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        y_t = torch.tensor([y], device=device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        pred = int(probs.argmax().item())
        conf_pred = float(probs[pred].item())
        conf_true = float(probs[int(y)].item())

        # Bigger means "worse": low confidence on true class
        score = 1.0 - conf_true
        if pred != int(y):
            score += 1.0  # push misclassified examples to the top

        rows.append((score, i, int(y), pred, conf_true, conf_pred))

    rows.sort(key=lambda r: r[0], reverse=True)
    rows = rows[:top_n]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("rank,test_index,true,pred,prob_true,prob_pred,score\n")
        for rank, (score, idx, y, p, pt, pp) in enumerate(rows, start=1):
            f.write(f"{rank},{idx},{y},{p},{pt:.6f},{pp:.6f},{score:.6f}\n")

    print(f"Saved worst-{top_n} errors CSV to: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--top_n", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    out_csv = os.path.join(paths["artifacts_dir"], "analysis", "worst_errors.csv")
    dump_worst(cfg, args.checkpoint, out_csv, top_n=args.top_n)


if __name__ == "__main__":
    main()

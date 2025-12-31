import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config file: {path}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility (may cost speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    runs_dir = cfg["paths"]["runs_dir"]
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    ensure_dir(runs_dir)
    ensure_dir(artifacts_dir)
    return {"runs_dir": runs_dir, "artifacts_dir": artifacts_dir}



from typing import Dict, Iterable, Tuple
import torch


@torch.no_grad()
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks: Tuple[int, ...] = (1,)) -> Dict[int, int]:
    """
    Returns a dict: {k: number_of_correct_predictions_in_topk}
    """
    maxk = max(ks)
    # topk indices: (N, maxk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    # compare with targets: (N, maxk)
    correct = pred.eq(targets.view(-1, 1))

    out: Dict[int, int] = {}
    for k in ks:
        out[k] = int(correct[:, :k].any(dim=1).sum().item())
    return out

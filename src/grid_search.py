import argparse
import os
from copy import deepcopy
from itertools import product
from typing import Any, Dict

from .train import train
from .utils import load_config


def grid_search(cfg: Dict[str, Any]) -> None:
    reductions = cfg["grid_search"]["reduction"]
    blocks_list = cfg["grid_search"]["blocks_per_stage"]
    epochs_per_run = int(cfg["grid_search"]["epochs_per_run"])

    for r, b in product(reductions, blocks_list):
        run_cfg = deepcopy(cfg)
        run_cfg["model"]["reduction"] = int(r)
        run_cfg["model"]["blocks_per_stage"] = int(b)

        # Short runs for grid
        run_cfg["training"]["epochs"] = epochs_per_run

        exp_name = f"grid_r{r}_b{b}"
        run_cfg["experiment"]["name"] = exp_name

        # Save grid checkpoints separately (don’t overwrite artifacts/best.ckpt)
        grid_ckpt_path = os.path.join(run_cfg["paths"]["artifacts_dir"], f"{exp_name}.ckpt")

        print(f"\n=== Running {exp_name} (epochs={epochs_per_run}) ===")
        train(run_cfg, experiment_name=exp_name, overfit_batch=False, checkpoint_path=grid_ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    grid_search(cfg)


if __name__ == "__main__":
    main()

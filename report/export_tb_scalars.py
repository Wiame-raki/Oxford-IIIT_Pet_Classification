import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def export_scalar(logdir: Path, tag: str, out_png: Path):
    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    scalars = ea.Scalars(tag)
    if not scalars:
        raise ValueError(f"Tag '{tag}' not found in {logdir}")

    steps = [s.step for s in scalars]
    vals = [s.value for s in scalars]

    plt.figure()
    plt.plot(steps, vals)
    plt.title(tag)
    plt.xlabel("step")
    plt.ylabel("value")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Path to a single TB run directory (contains events.out.tfevents...)")
    ap.add_argument("--out_dir", default="report/figures/tb_export")
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    logdir = Path(args.logdir)
    out_dir = Path(args.out_dir)

    for tag in args.tags:
        safe = tag.replace("/", "__")
        export_scalar(logdir, tag, out_dir / f"{safe}.png")
        print("saved:", out_dir / f"{safe}.png")


if __name__ == "__main__":
    main()

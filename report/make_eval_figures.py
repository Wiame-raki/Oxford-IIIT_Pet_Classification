import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import load_config
from src.data_loading import get_dataloaders
from src.model import PetSE_CNN


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_y, all_x = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_y.append(y.cpu())
        all_x.append(x.cpu())
    return torch.cat(all_probs), torch.cat(all_y), torch.cat(all_x)


def plot_per_class_acc(y_true, y_pred, num_classes, out_png):
    accs = []
    for c in range(num_classes):
        idx = (y_true == c)
        accs.append((y_pred[idx] == c).float().mean().item() if idx.any() else 0.0)

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(num_classes), accs)
    plt.title("Per-class accuracy (test)")
    plt.xlabel("class id")
    plt.ylabel("accuracy")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confidence_hist(probs, y_true, out_png):
    conf, pred = probs.max(dim=1)
    correct = (pred == y_true)

    plt.figure(figsize=(8, 4))
    plt.hist(conf[correct].numpy(), bins=20, alpha=0.7, label="correct")
    plt.hist(conf[~correct].numpy(), bins=20, alpha=0.7, label="incorrect")
    plt.title("Max softmax confidence distribution (test)")
    plt.xlabel("confidence")
    plt.ylabel("count")
    plt.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def grid_misclassified(x, y_true, y_pred, probs, out_png, n=24):
    conf, pred = probs.max(dim=1)
    bad = (pred != y_true).nonzero(as_tuple=False).squeeze(1)
    if bad.numel() == 0:
        print("No misclassifications found.")
        return
    bad = bad[torch.argsort(conf[bad], descending=True)][:n]


    imgs = x[bad]
    t = int(np.ceil(np.sqrt(len(bad))))
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(bad):
        im = imgs[i]
        im = im.permute(1, 2, 0).numpy()
        im = np.clip(im, 0, 1)
        plt.subplot(t, t, i + 1)
        plt.imshow(im)
        plt.axis("off")
        plt.title(f"T{y_true[idx].item()} → P{y_pred[idx].item()}\n{conf[idx].item():.2f}", fontsize=9)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.suptitle("Most confident misclassifications (test)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="artifacts/best.ckpt")
    ap.add_argument("--out_dir", default="report/figures/eval_extra")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PetSE_CNN(
        num_classes=meta["num_classes"],
        reduction=cfg["model"].get("reduction", 8) if isinstance(cfg.get("model"), dict) else 8,
        blocks_per_stage=cfg["model"].get("blocks_per_stage", 2) if isinstance(cfg.get("model"), dict) else 2,
        dropout=cfg["model"].get("dropout", 0.2) if isinstance(cfg.get("model"), dict) else 0.2,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # remove possible "model." prefix
    new_state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)

    probs, y_true, x = predict(model, test_loader, device)
    y_pred = probs.argmax(dim=1)

    out_dir = Path(args.out_dir)
    plot_per_class_acc(y_true, y_pred, meta["num_classes"], out_dir / "per_class_accuracy_test.png")
    plot_confidence_hist(probs, y_true, out_dir / "confidence_hist_test.png")
    grid_misclassified(x, y_true, y_pred, probs, out_dir / "misclassified_grid_test.png", n=25)

    print("Saved extra eval figures in:", out_dir)


if __name__ == "__main__":
    main()

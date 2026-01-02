# src/visualize_errors.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_config, get_device
from src.data_loading import get_dataloaders
from src.model import PetSE_CNN


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)


def show_errors():
    cfg = load_config("../configs/config.yaml")
    device = get_device()
    _, _, test_loader, _ = get_dataloaders(cfg)

    # Load Best Model
    model = PetSE_CNN(num_classes=37, reduction=8, blocks_per_stage=2, dropout=0.4).to(device)
    ckpt = torch.load("artifacts/cropping_best.ckpt", map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    errors = []

    # Collect 9 errors
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)

            wrong_idx = (preds != y).nonzero()
            for idx in wrong_idx:
                if len(errors) >= 9: break
                i = idx.item()
                errors.append((x[i], y[i].item(), preds[i].item()))
            if len(errors) >= 9: break

    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, (img, true_y, pred_y) in enumerate(errors):
        ax = axes[i // 3, i % 3]
        ax.imshow(denormalize(img))
        ax.set_title(f"True: {true_y} | Pred: {pred_y}", color='red')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("error_analysis.png")
    print("Saved error_analysis.png")


if __name__ == "__main__":
    show_errors()
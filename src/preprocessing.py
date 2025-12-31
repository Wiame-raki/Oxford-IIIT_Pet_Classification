from typing import Any, Dict
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_preprocess_transforms(config: Dict[str, Any]):
    """
    Deterministic transforms for val/test (and optionally train if augmentation disabled).
    """
    img_size = int(config["data"]["image_size"])
    resize_size = int(config["data"]["resize_size"])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

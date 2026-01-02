from typing import Any, Dict, Optional
from torchvision import transforms

def get_augmentation_transforms(config: Dict[str, Any]) -> Optional[transforms.Compose]:
    aug = config.get("augmentation", {})
    if not bool(aug.get("enabled", True)):
        return None

    img_size = int(config["data"]["image_size"])
    resize_size = int(config["data"]["resize_size"])

    # Basic aug params
    rot = float(aug.get("rotation_deg", 10))
    hflip_p = float(aug.get("hflip_p", 0.5))

    jb = float(aug.get("jitter_brightness", 0.10))
    jc = float(aug.get("jitter_contrast", 0.10))
    js = float(aug.get("jitter_saturation", 0.10))
    jh = float(aug.get("jitter_hue", 0.05))

    # Slightly stronger (but still reasonable)
    use_rrc = bool(aug.get("use_random_resized_crop", True))

    scale_min = float(aug.get("rrc_scale_min", 0.60))
    ratio_min = float(aug.get("rrc_ratio_min", 0.75))
    ratio_max = float(aug.get("rrc_ratio_max", 1.33))


    ops = []
    if use_rrc:
        ops.append(transforms.RandomResizedCrop(
            img_size,
            scale=(scale_min, 1.0),
            ratio=(ratio_min, ratio_max),
        ))
    else:
        ops += [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop((img_size, img_size)),
        ]

    ops += [
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomRotation(degrees=rot),
        transforms.ColorJitter(brightness=jb, contrast=jc, saturation=js, hue=jh),
    ]

    return transforms.Compose(ops)

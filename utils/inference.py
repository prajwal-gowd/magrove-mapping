import torch
from PIL import Image
import torchvision.transforms.functional as TF
import config
import numpy as np


def load_segmentation_model(checkpoint_path=config.SAVE_PATH, device=config.DEVICE):
    """
    Intelligently inspects a checkpoint file and loads the appropriate model architecture:
    - MangroveSegFormer (Option A, modern MiT transformer)
    - HybridTCCFNet / LegacyHybridTCCFNet (legacy ResNet-Transformer hybrid)
    """
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    # Sync optimal threshold if stored in checkpoint
    if isinstance(checkpoint, dict):
        if "best_threshold" in checkpoint:
            config.VAL_THRESHOLD = float(checkpoint["best_threshold"])
        elif "val_threshold" in checkpoint:
            config.VAL_THRESHOLD = float(checkpoint["val_threshold"])

    # Determine architecture from keys
    state_keys = list(state_dict.keys())
    is_segformer = any("model.encoder" in k or "encoder.patch_embed" in k for k in state_keys) or (
        isinstance(checkpoint, dict) and "backbone" in checkpoint and "mit_" in str(checkpoint["backbone"])
    )

    if is_segformer:
        from models.segformer_model import MangroveSegFormer

        backbone = checkpoint.get("backbone", getattr(config, "SEGFORMER_BACKBONE", "mit_b3"))
        model = MangroveSegFormer(backbone=backbone, pretrained=False)
        model.load_state_dict(state_dict)
    else:
        # Check if improved v2 HybridTCCFNet or Legacy
        has_skips = any("up3" in k or "bottleneck" in k for k in state_keys)
        if has_skips:
            from models.hybrid_model import HybridTCCFNet
            model = HybridTCCFNet(pretrained_backbone=False)
        else:
            from models.legacy_hybrid_model import LegacyHybridTCCFNet
            model = LegacyHybridTCCFNet(pretrained_backbone=False)

        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def load_model_checkpoint(model, checkpoint_path, device):
    """Load either a training checkpoint or a raw model state dictionary into an existing model."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(checkpoint, dict):
        if "best_threshold" in checkpoint:
            config.VAL_THRESHOLD = float(checkpoint["best_threshold"])
        elif "val_threshold" in checkpoint:
            config.VAL_THRESHOLD = float(checkpoint["val_threshold"])

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path_or_pil):
    """Preprocesses the input image for the model."""
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")

    image_tensor = TF.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image_tensor = TF.to_tensor(image_tensor)
    image_tensor = TF.normalize(
        image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Add batch dimension
    img_tensor = image_tensor.unsqueeze(0)
    return img_tensor


def predict(model, image_path_or_pil, device=config.DEVICE, use_tta=False):
    """
    Runs inference on a single image and returns the binary segmentation mask.
    Args:
        model: PyTorch model.
        image_path_or_pil: Path to the image or PIL Image object.
        device: CPU or GPU device.
        use_tta: Whether to run Test-Time Augmentation for higher precision.
    Returns:
        mask_array: 2D numpy array containing the predicted binary mask (0 or 255).
    """
    model.eval()
    img_tensor = preprocess_image(image_path_or_pil).to(device)

    with torch.no_grad():
        if use_tta and hasattr(model, "predict_with_tta"):
            probs = model.predict_with_tta(img_tensor)
        else:
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)

        mask = (probs > config.VAL_THRESHOLD).float().squeeze()

    mask_array = (mask.cpu().numpy() * 255).astype(np.uint8)
    return mask_array

import torch
from PIL import Image
import torchvision.transforms.functional as TF
import config
import numpy as np


def load_model_checkpoint(model, checkpoint_path, device):
    """Load either a training checkpoint or a raw model state dictionary."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # `weights_only` was added after older supported PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path_or_pil):
    """
    Preprocesses the input image for the model.
    """
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert('RGB')
    else:
        image = image_path_or_pil.convert('RGB')

    image_tensor = TF.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image_tensor = TF.to_tensor(image_tensor)
    image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Add batch dimension
    img_tensor = image_tensor.unsqueeze(0)
    return img_tensor

def predict(model, image_path_or_pil, device):
    """
    Runs inference on a single image and returns the segmentation mask.
    Args:
        model: PyTorch model.
        image_path_or_pil: Path to the image or PIL Image object.
        device: CPU or GPU device.
    Returns:
        mask_array: 2D numpy array containing the predicted binary mask (0 or 255).
    """
    model.eval()
    img_tensor = preprocess_image(image_path_or_pil).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor) # [1, 1, 224, 224]
        probs = torch.sigmoid(logits)
        mask = (probs > 0.65).float().squeeze() # [224, 224]
        
    mask_array = (mask.cpu().numpy() * 255).astype(np.uint8)
    return mask_array

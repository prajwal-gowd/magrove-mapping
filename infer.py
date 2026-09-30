"""Run mangrove segmentation for one image from the command line."""

import argparse
import os

import numpy as np
from PIL import Image

import config
from models.hybrid_model import HybridTCCFNet
from utils.inference import load_model_checkpoint, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to an RGB input image")
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/predicted_mask.png",
        help="Path for the binary mask PNG (default: outputs/predicted_mask.png)",
    )
    parser.add_argument(
        "--checkpoint",
        default=config.SAVE_PATH,
        help=f"Model checkpoint path (default: {config.SAVE_PATH})",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"Input image does not exist: {args.input}")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    model = HybridTCCFNet(num_classes=config.NUM_CLASSES, pretrained_backbone=False)
    load_model_checkpoint(model, args.checkpoint, config.DEVICE)
    mask = predict(model, Image.open(args.input), config.DEVICE)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(args.output)
    print(f"Saved predicted mask to {args.output}")


if __name__ == "__main__":
    main()

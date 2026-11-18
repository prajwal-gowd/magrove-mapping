"""Run mangrove segmentation for one image from the command line."""

import argparse
import os

import numpy as np
from PIL import Image

import config
from utils.inference import load_segmentation_model, predict


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
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable Test-Time Augmentation",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"Input image does not exist: {args.input}")

    model = load_segmentation_model(args.checkpoint, config.DEVICE)
    use_tta = not args.no_tta
    mask = predict(model, Image.open(args.input), config.DEVICE, use_tta=use_tta)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(args.output)
    print(f"Saved predicted mask to {args.output}")


if __name__ == "__main__":
    main()

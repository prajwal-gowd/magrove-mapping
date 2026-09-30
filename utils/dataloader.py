import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import numpy as np
import config


class SegmentationDataset(Dataset):
    """
    Dataset for paired satellite image / binary mask segmentation patches.

    Augmentation strategy (train only):
      - Geometric: H-flip, V-flip, random rotation, random affine
      - Photometric: ColorJitter, Gaussian blur, random brightness/contrast
      - Spatial: RandomResizedCrop for scale invariance
    All geometric transforms are applied identically to both image and mask.
    """

    def __init__(self, root_dir, split="train", is_train=True):
        self.root_dir = root_dir
        self.split = split
        self.is_train = is_train

        self.image_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")

        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            print(f"Warning: Dataset split '{split}' not found at {self.image_dir}")
            self.images = []
        else:
            self.images = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.images)

    def transform(self, image, mask):
        # Resize to target size
        image = TF.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        mask = TF.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # --- Random Resized Crop (scale invariance) ---
            if random.random() > 0.5:
                scale = random.uniform(0.7, 1.0)
                crop_h = int(config.IMAGE_SIZE * scale)
                crop_w = int(config.IMAGE_SIZE * scale)
                i, j = random.randint(0, config.IMAGE_SIZE - crop_h), random.randint(0, config.IMAGE_SIZE - crop_w)
                image = TF.resized_crop(image, i, j, crop_h, crop_w,
                                        (config.IMAGE_SIZE, config.IMAGE_SIZE))
                mask = TF.resized_crop(mask, i, j, crop_h, crop_w,
                                       (config.IMAGE_SIZE, config.IMAGE_SIZE),
                                       interpolation=TF.InterpolationMode.NEAREST)

            # --- Random horizontal flipping ---
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # --- Random vertical flipping ---
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # --- Random rotation (90°/180°/270° or small angle) ---
            if random.random() > 0.5:
                if random.random() > 0.5:
                    angle = random.choice([90, 180, 270])
                else:
                    angle = random.uniform(-30, 30)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # --- Random affine (slight shear & translate) ---
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                translate = [random.randint(-20, 20), random.randint(-20, 20)]
                scale_factor = random.uniform(0.9, 1.1)
                shear = random.uniform(-10, 10)
                image = TF.affine(image, angle=angle, translate=translate, scale=scale_factor, shear=[shear, 0.0])
                mask = TF.affine(mask, angle=angle, translate=translate,
                                 scale=scale_factor, shear=[shear, 0.0],
                                 interpolation=TF.InterpolationMode.NEAREST)

            # --- Color Jitter (image only — not mask) ---
            if random.random() > 0.5:
                jitter = T.ColorJitter(brightness=0.3, contrast=0.3,
                                       saturation=0.3, hue=0.1)
                image = jitter(image)

            # --- Gaussian Blur (image only) ---
            if random.random() > 0.7:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # To Tensor
        image = TF.to_tensor(image)   # [C, H, W], 0-1
        mask = TF.to_tensor(mask)     # [1, H, W], 0-1

        # Binary thresholding to ensure exactly 0 and 1
        mask = (mask > 0.5).float()

        # Normalize image with ImageNet statistics
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        return image, mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self.transform(image, mask)

        return image, mask


def get_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE):
    train_dataset = SegmentationDataset(root_dir=data_dir, split="train", is_train=True)
    val_dataset = SegmentationDataset(root_dir=data_dir, split="val", is_train=False)

    if len(train_dataset) == 0:
        return None, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader

import os
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from utils.dataloader import get_dataloaders
from utils.metrics import (
    calculate_iou,
    calculate_dice,
    calculate_precision,
    calculate_recall,
    calculate_accuracy,
    calculate_confusion_matrix_elements,
)
from models.segformer_model import MangroveSegFormer, CombinedIoULoss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_and_save_curves(history, output_path="segformer_training_curves.png"):
    """Plot and save training and validation metric curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0, 0].set_title("Loss Progression", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. IoU
    axes[0, 1].plot(epochs, history["val_iou"], label="Validation IoU", color="#2ca02c", lw=2)
    axes[0, 1].axhline(y=max(history["val_iou"]), color="r", linestyle="--", alpha=0.7,
                       label=f"Peak IoU: {max(history['val_iou']):.4f}")
    axes[0, 1].set_title("Validation IoU (Jaccard)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Dice / F1
    axes[1, 0].plot(epochs, history["val_dice"], label="Validation Dice / F1", color="#9467bd", lw=2)
    axes[1, 0].set_title("Validation Dice / F1", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Dice")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Precision vs Recall
    axes[1, 1].plot(epochs, history["val_precision"], label="Precision", color="#d62728", lw=2)
    axes[1, 1].plot(epochs, history["val_recall"], label="Recall", color="#17becf", lw=2)
    axes[1, 1].set_title("Precision vs Recall", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Training curves saved to {output_path}")


def train(
    backbone="mit_b3",
    batch_size=16,
    epochs=50,
    warmup_epochs=4,
    encoder_lr=6e-5,
    decoder_lr=3e-4,
    weight_decay=1e-4,
    bce_weight=0.3,
    dice_weight=0.4,
    lovasz_weight=0.3,
    save_path="checkpoints/segformer_best.pth",
    resume=False,
):
    device = config.DEVICE
    print("=" * 65)
    print(f"🌿 MangroveNet — High-IoU SegFormer ({backbone}) Training")
    print(f"Device: {device} | Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"Loss Configuration: BCE={bce_weight}, Dice={dice_weight}, Lovász={lovasz_weight}")
    print("=" * 65)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # 1. Load Data
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    if train_loader is None or val_loader is None:
        raise RuntimeError("Failed to load dataloaders. Check dataset_patches/ directory.")

    print(f"Data ready: {len(train_loader.dataset)} training images, "
          f"{len(val_loader.dataset)} validation images.")

    # 2. Initialize SegFormer
    model = MangroveSegFormer(backbone=backbone, pretrained=True, num_classes=1)

    # 3. Combined IoU-Targeted Loss
    criterion = CombinedIoULoss(
        bce_w=bce_weight,
        dice_w=dice_weight,
        lovasz_w=lovasz_weight,
    )

    # 4. Optimizer with differential LR
    param_groups = model.get_param_groups(encoder_lr=encoder_lr, decoder_lr=decoder_lr)
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    # 5. Warmup + Cosine Annealing Scheduler
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 1
    best_val_iou = 0.0
    best_threshold = 0.50

    if resume and os.path.exists(save_path):
        print(f"Loading checkpoint for resume from {save_path}...")
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_iou = ckpt.get("best_val_iou", 0.0)
            best_threshold = ckpt.get("best_threshold", 0.50)
            print(f"Resumed from epoch {start_epoch} | Best IoU: {best_val_iou:.4f}")

    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_precision": [],
        "val_recall": [],
        "val_accuracy": [],
    }

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")

        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            cur_enc_lr = optimizer.param_groups[0]["lr"]
            cur_dec_lr = optimizer.param_groups[1]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dec_lr": f"{cur_dec_lr:.1e}"})

        epoch_train_loss = train_loss / len(train_loader.dataset)

        # ---------------------------------------------------------------------
        # Validation Loop with Threshold Sweep
        # ---------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_masks = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]"):
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_masks.append(masks.cpu())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        probs_tensor = torch.cat(all_probs, dim=0)
        masks_tensor = torch.cat(all_masks, dim=0)

        # Evaluate threshold sweep [0.35 to 0.75] to find optimal decision boundary
        epoch_best_iou = 0.0
        epoch_best_th = 0.50
        for th in np.arange(0.35, 0.76, 0.05):
            th = round(float(th), 2)
            preds = (probs_tensor > th).float()
            iou = calculate_iou(preds, masks_tensor)
            if iou > epoch_best_iou:
                epoch_best_iou = iou
                epoch_best_th = th

        best_preds = (probs_tensor > epoch_best_th).float()
        epoch_dice = calculate_dice(best_preds, masks_tensor)
        epoch_precision = calculate_precision(best_preds, masks_tensor)
        epoch_recall = calculate_recall(best_preds, masks_tensor)
        epoch_accuracy = calculate_accuracy(best_preds, masks_tensor)
        tp, tn, fp, fn = calculate_confusion_matrix_elements(best_preds, masks_tensor)

        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_iou"].append(epoch_best_iou)
        history["val_dice"].append(epoch_dice)
        history["val_precision"].append(epoch_precision)
        history["val_recall"].append(epoch_recall)
        history["val_accuracy"].append(epoch_accuracy)

        print(
            f"Epoch {epoch:02d} Results | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val IoU: {epoch_best_iou:.4f} (th={epoch_best_th:.2f}) | "
            f"Dice: {epoch_dice:.4f} | Prec: {epoch_precision:.4f} | Rec: {epoch_recall:.4f}"
        )

        # Checkpoint if new best IoU
        if epoch_best_iou > best_val_iou:
            best_val_iou = epoch_best_iou
            best_threshold = epoch_best_th
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": backbone,
                    "model_state_dict": model.state_dict(),
                    "best_val_iou": best_val_iou,
                    "best_threshold": best_threshold,
                    "dice": epoch_dice,
                    "precision": epoch_precision,
                    "recall": epoch_recall,
                    "accuracy": epoch_accuracy,
                    "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                },
                save_path,
            )
            print(f"  ⭐ New Best Model Checkpoint Saved! IoU: {best_val_iou:.4f} (th={best_threshold:.2f})")

    # Save final curves
    plot_and_save_curves(history, output_path="segformer_training_curves.png")
    print("\n" + "=" * 65)
    print(f"🎉 Training Complete! Peak Validation IoU: {best_val_iou:.4f} at threshold: {best_threshold:.2f}")
    print(f"Saved Checkpoint: {save_path}")
    print("=" * 65)
    return best_val_iou, best_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer on Mangrove patches")
    parser.add_argument("--backbone", type=str, default="mit_b3",
                        choices=["mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=6e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.4)
    parser.add_argument("--lovasz-weight", type=float, default=0.3)
    parser.add_argument("--save-path", type=str, default="checkpoints/segformer_best.pth")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    train(
        backbone=args.backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=args.weight_decay,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        lovasz_weight=args.lovasz_weight,
        save_path=args.save_path,
        resume=args.resume,
    )

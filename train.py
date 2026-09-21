import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from models.hybrid_model import HybridTCCFNet
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_iou, calculate_dice, calculate_precision, calculate_recall, calculate_accuracy, calculate_confusion_matrix_elements, BCEDiceLoss

def train(resume=False):
    print("Initializing training pipeline...")
    
    # Create checkpoints dir
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)
    
    # Load Data
    train_loader, val_loader = get_dataloaders()
    if train_loader is None:
        print("Failed to load dataloaders. Check dataset path.")
        return
        
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")
    
    # Initialize Model
    model = HybridTCCFNet(num_classes=config.NUM_CLASSES)
    
    # Loss, Optimizer, and Scheduler
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_iou = 0.0
    start_epoch = 0

    if resume and os.path.exists(config.SAVE_PATH):
        print(f"Loading checkpoint from {config.SAVE_PATH}...")
        try:
            checkpoint = torch.load(config.SAVE_PATH, map_location="cpu")
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_iou = checkpoint.get('best_val_iou', 0.0)
                print(f"Resumed from epoch {start_epoch} with Best IoU: {best_val_iou:.4f}")
            else:
                model.load_state_dict(checkpoint)
                start_epoch = 1  # Completed epoch 1 according to logs
                best_val_iou = 0.3153
                print("Loaded state_dict from previous best_model.pth (Epoch 1).")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    model.to(config.DEVICE)
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        processed_train_samples = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            try:
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)
                
                optimizer.zero_grad()
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                processed_train_samples += images.size(0)
                pbar.set_postfix({"loss": loss.item()})
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nWARNING: Out of memory, skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e
            
        epoch_train_loss = train_loss / max(processed_train_samples, 1)
        print(f"Train Loss: {epoch_train_loss:.4f}")
        
        # --- Validation ---
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            val_dice = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_accuracy = 0.0
            total_tp = 0
            total_tn = 0
            total_fp = 0
            total_fn = 0
            processed_val_samples = 0
            
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc="Validation"):
                    try:
                        images = images.to(config.DEVICE)
                        masks = masks.to(config.DEVICE)
                        
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                        val_loss += loss.item() * images.size(0)
                        
                        # Calculate metrics
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        val_iou += calculate_iou(preds, masks) * images.size(0)
                        val_dice += calculate_dice(preds, masks) * images.size(0)
                        val_precision += calculate_precision(preds, masks) * images.size(0)
                        val_recall += calculate_recall(preds, masks) * images.size(0)
                        val_accuracy += calculate_accuracy(preds, masks) * images.size(0)
                        
                        tp, tn, fp, fn = calculate_confusion_matrix_elements(preds, masks)
                        total_tp += tp
                        total_tn += tn
                        total_fp += fp
                        total_fn += fn
                        
                        processed_val_samples += images.size(0)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"\nWARNING: Out of memory in validation, skipping batch.")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            raise e
                    
            epoch_val_loss = val_loss / max(processed_val_samples, 1)
            epoch_val_iou = val_iou / max(processed_val_samples, 1)
            epoch_val_dice = val_dice / max(processed_val_samples, 1)
            epoch_val_precision = val_precision / max(processed_val_samples, 1)
            epoch_val_recall = val_recall / max(processed_val_samples, 1)
            epoch_val_accuracy = val_accuracy / max(processed_val_samples, 1)
            
            print(f"Val Loss: {epoch_val_loss:.4f} | Val IoU: {epoch_val_iou:.4f} | Val Dice: {epoch_val_dice:.4f}")
            print(f"Val Precision: {epoch_val_precision:.4f} | Val Recall: {epoch_val_recall:.4f} | Val Accuracy: {epoch_val_accuracy:.4f}")
            print(f"Confusion Matrix (Pixels): TP={int(total_tp)}, TN={int(total_tn)}, FP={int(total_fp)}, FN={int(total_fn)}")
            
            if epoch_val_iou > best_val_iou:
                best_val_iou = epoch_val_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_iou': best_val_iou
                }, config.SAVE_PATH)
                print(f"--> Saved new best model with IoU: {best_val_iou:.4f}")
                
            # Step the scheduler
            scheduler.step(epoch_val_iou)
        else:
            # If no validation set, just save the latest model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou
            }, config.SAVE_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)

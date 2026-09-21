import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config
from models.hybrid_model import HybridTCCFNet
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_confusion_matrix_elements
import os

def generate_cm():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = HybridTCCFNet(num_classes=config.NUM_CLASSES)
    if not os.path.exists(config.SAVE_PATH):
        print(f"Checkpoint {config.SAVE_PATH} not found.")
        return
        
    print(f"Loading checkpoint from {config.SAVE_PATH}...")
    checkpoint = torch.load(config.SAVE_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_iou = checkpoint.get('best_val_iou', 0.0)
        epoch = checkpoint.get('epoch', 0)
    else:
        model.load_state_dict(checkpoint)
        best_val_iou = 0.4716  # From logs if metadata is missing
        epoch = 1
        
    model.to(device)
    model.eval()
    
    # Load data
    _, val_loader = get_dataloaders()
    if not val_loader:
        print("No validation loader found.")
        return

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    
    print(f"Running validation on Best Model (Epoch {epoch+1}, IoU: {best_val_iou:.4f})...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            tp, tn, fp, fn = calculate_confusion_matrix_elements(preds, masks)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    print(f"TP: {total_tp}, TN: {total_tn}, FP: {total_fp}, FN: {total_fn}")
    
    # Plot
    # cm array: [[TN, FP], [FN, TP]]
    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Normalize by row (actual class) for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Labels for the heatmap
    labels = np.array([
        [f"TN\n{total_tn:,.0f}\n({cm_norm[0,0]:.2%})", f"FP\n{total_fp:,.0f}\n({cm_norm[0,1]:.2%})"],
        [f"FN\n{total_fn:,.0f}\n({cm_norm[1,0]:.2%})", f"TP\n{total_tp:,.0f}\n({cm_norm[1,1]:.2%})"]
    ])
    
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', 
                xticklabels=['Non-Mangrove (0)', 'Mangrove (1)'], 
                yticklabels=['Non-Mangrove (0)', 'Mangrove (1)'],
                annot_kws={"size": 14})
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Pixel-wise Confusion Matrix\nBest Model (Epoch {epoch+1}, IoU: {best_val_iou:.4f})', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Saved confusion matrix to confusion_matrix.png")

if __name__ == "__main__":
    generate_cm()

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1e-6):
        # BCE with Logits
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        inputs = torch.sigmoid(inputs)
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_iou(preds, labels, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.
    Args:
        preds: Tensor of shape [B, 1, H, W], binary predictions (0 or 1)
        labels: Tensor of shape [B, 1, H, W], ground truth (0 or 1)
    """
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_dice(preds, labels, smooth=1e-6):
    """
    Calculate Dice Coefficient for binary segmentation.
    """
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    
    intersection = (preds * labels).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + labels.sum() + smooth)
    return dice.item()

def calculate_precision(preds, labels, smooth=1e-6):
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.item()

def calculate_recall(preds, labels, smooth=1e-6):
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    tp = (preds * labels).sum()
    fn = ((1 - preds) * labels).sum()
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.item()

def calculate_accuracy(preds, labels):
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    correct = (preds == labels).sum()
    total = labels.size(0)
    return (correct.float() / total).item()

def calculate_confusion_matrix_elements(preds, labels):
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    tp = (preds * labels).sum().item()
    tn = ((1 - preds) * (1 - labels)).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    return tp, tn, fp, fn

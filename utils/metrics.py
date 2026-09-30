import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard-example mining.

    Applies a modulating factor (1 - p_t)^gamma to the standard cross-entropy
    loss, down-weighting easy examples and focusing training on hard negatives.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss for binary segmentation.

    Weighted sum where Dice directly optimizes spatial overlap (IoU proxy)
    and BCE provides smooth gradients for stable training.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1e-6):
        # BCE with Logits
        bce_loss = self.bce(inputs, targets)

        # Dice Loss
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice loss.

    Replaces BCE with Focal loss for better handling of class imbalance
    and hard-to-classify boundary pixels. Dice component directly
    maximizes spatial overlap.
    """

    def __init__(self, focal_weight=0.3, dice_weight=0.7, alpha=0.75, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, inputs, targets, smooth=1e-6):
        focal_loss = self.focal(inputs, targets)

        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

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

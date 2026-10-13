import torch

def compute_segmentation_metrics(pred_probs, targets, threshold=0.5, smooth=1e-5):
    """
    Computes Sørensen-Dice, IoU, Sensitivity, and Specificity
    matching Section 4.1.2 of the base paper.
    """
    preds = (pred_probs > threshold).float()
    targets = (targets > 0).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity
    }
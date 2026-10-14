import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()
        
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='mean')
        return (self.dice_weight * dice_loss.mean()) + (self.bce_weight * bce_loss)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()

        tp = (preds * targets).sum(dim=(2, 3))
        fp = (preds * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - preds) * targets).sum(dim=(2, 3))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky_index), 1.0 / self.gamma)
        return focal_tversky.mean()


class HybridTverskyBCELoss(nn.Module):
    """
    Experiment 3: Composite loss balancing point-wise cross-entropy stability
    with asymmetric boundary penalization.
    - alpha=0.7: heavier penalty on False Positives to halt boundary leakage
    - beta=0.3:  maintains sensitivity to genuine lesion tissue
    - bce_weight=0.5: provides smooth convex gradient anchor across background
    - tversky_weight=0.5: drives structural boundary convergence
    """
    def __init__(self, alpha=0.7, beta=0.3, bce_weight=0.5, tversky_weight=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()

        # 1. Point-wise BCE Loss (Anchor for background stability)
        bce = F.binary_cross_entropy(preds, targets, reduction='mean')

        # 2. Asymmetric Tversky Loss
        tp = (preds * targets).sum(dim=(2, 3))
        fp = (preds * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - preds) * targets).sum(dim=(2, 3))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = 1.0 - tversky_index

        return (self.bce_weight * bce) + (self.tversky_weight * tversky_loss.mean())
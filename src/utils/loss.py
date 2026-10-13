import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    """
    Combined Soft Dice Loss and Binary Cross Entropy Loss
    as formulated in Section 3.3 of the base paper.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred_probs, targets):
        targets = (targets > 0).float()
        
        bce_loss = self.bce(pred_probs, targets)

        pred_flat = pred_probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (pred_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
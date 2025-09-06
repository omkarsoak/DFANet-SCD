import torch
import torch.nn.functional as F
import torch.nn as nn
################################# Tversky loss for multi-class #################################
def tversky_loss(logits, targets, alpha=0.7, beta=0.3, eps=1e-6):
    """
    logits: [B, C, H, W] (raw logits, not argmaxed)
    targets: [B, H, W] (integer class labels)
    """
    num_classes = logits.shape[1]
    # One-hot targets
    targets_1hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    probs = F.softmax(logits, dim=1)
    # Tversky components
    tp = torch.sum(probs * targets_1hot, dim=(0, 2, 3))
    fp = torch.sum(probs * (1 - targets_1hot), dim=(0, 2, 3))
    fn = torch.sum((1 - probs) * targets_1hot, dim=(0, 2, 3))
    # Tversky index per class, then average
    T = tp / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - torch.mean(T)

class CombinedTverskyLoss(nn.Module):
    """
    Half Cross Entropy, Half Tversky
    Adjust ce_weight / tv_weight to taste
    """
    def __init__(self, alpha=0.7, beta=0.3, ce_weight=0.5, tv_weight=0.5, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.alpha = alpha
        self.beta = beta
        self.ce_weight = ce_weight
        self.tv_weight = tv_weight

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_tv = tversky_loss(logits, targets, alpha=self.alpha, beta=self.beta)
        return self.ce_weight * loss_ce + self.tv_weight * loss_tv

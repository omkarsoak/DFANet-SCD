######### Util functions #########
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def calculate_effective_weights(train_loader, device, num_cd_classes=3, method='square_balanced'):
    """Calculate class weights with different strategies to handle class imbalance

    Args:
        train_loader: DataLoader containing training data
        device: torch device
        num_cd_classes: number of classes (default: 3)
        method: weighting strategy ('balanced', 'square_balanced', or 'custom')
    """
    class_counts = torch.zeros(num_cd_classes)
    total_pixels = 0

    # Count class frequencies (using dictionary access)
    for batch in train_loader:
        labels = batch['cd_mask'].to(device)
        for i in range(num_cd_classes):
            class_counts[i] += (labels == i).sum().item()
        total_pixels += labels.numel()

    class_frequencies = class_counts / total_pixels

    if method == 'balanced':
        # Standard balanced weighting (inverse frequency)
        weights = 1.0 / class_frequencies

    elif method == 'square_balanced':
        # Square root of inverse frequencies (less aggressive balancing)
        weights = torch.sqrt(1.0 / class_frequencies)

    elif method == 'custom':
        # Custom weighting that maintains some natural class distribution
        # Adjust these factors based on your domain knowledge
        base_weights = 1.0 / class_frequencies
        adjustment_factors = torch.tensor([0.7, 1.2, 1.2])
        weights = base_weights * adjustment_factors

    # Normalize weights to sum to num_cd_classes
    weights = weights * (num_cd_classes / weights.sum())

    return weights, class_frequencies

def calculate_metrics(predictions, targets, num_cd_classes=3, weighted_metrics=False):
    """
    Calculate comprehensive metrics for change detection using a single confusion matrix

    Args:
        predictions (np.array): Predicted class labels
        targets (np.array): Ground truth class labels
        num_cd_classes (int): Number of classes in the dataset

    Returns:
        dict: Dictionary of performance metrics
    """
    # Flatten predictions and targets
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Compute confusion matrix once
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(num_cd_classes)))

    # Calculate metrics from confusion matrix
    metrics = {}

    # True positives, false positives, false negatives for each class
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    # Overall accuracy from confusion matrix
    metrics['accuracy'] = np.sum(tp) / np.sum(cm)

    # Per-class precision, recall, F1
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    if weighted_metrics == False:
        # Unweighted averages
        metrics['precision'] = np.average(precision)
        metrics['recall'] = np.average(recall)
        metrics['f1_score'] = np.average(f1)
    elif weighted_metrics == True:
        # Weighted averages
        total = np.sum(cm, axis=1)
        metrics['precision'] = np.average(precision,weights=total)
        metrics['recall'] = np.average(recall, weights=total)
        metrics['f1_score'] = np.average(f1,weights=total)

    # Calculate Kappa directly from confusion matrix
    n = np.sum(cm)
    sum_po = np.sum(np.diag(cm))
    sum_pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / n
    metrics['kappa'] = (sum_po - sum_pe) / (n - sum_pe + 1e-6)

    # IoU from confusion matrix
    iou_per_class = tp / (tp + fp + fn + 1e-6)
    metrics['miou'] = np.mean(iou_per_class)

    return metrics
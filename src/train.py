import torch
import os
from utils.metrics import calculate_metrics, calculate_effective_weights
from models.losses import CombinedTverskyLoss
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import json 

def train_model(model, train_loader, val_loader, num_epochs, device,
                   checkpoint_path, num_cd_classes=3, weighted_metrics=False):
    """Train the model with optimized metrics calculation"""

    # Try to load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']  # Load best validation loss
            print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        print("No checkpoint found. Starting training from scratch")
        start_epoch = 0
        best_val_loss = float('inf')

    sem_criterion = CombinedTverskyLoss(
        alpha=0.7,
        beta=0.3,
        ce_weight=0.5,
        tv_weight=0.5,
        class_weights=None
    ).to(device)
    
    class_weights,_ = calculate_effective_weights(train_loader,num_cd_classes=num_cd_classes,device=device)
    print(class_weights)

    change_criterion = CombinedTverskyLoss(
        alpha=0.7,
        beta=0.3,
        ce_weight=0.5,
        tv_weight=0.5,
        class_weights=class_weights.to(device)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    ################################################################################
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')
    ################################################################################

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch in progress_bar:
            img_t2019 = batch['img_t2019'].to(device)
            img_t2024 = batch['img_t2024'].to(device)
            sem_mask_2019 = batch['sem_mask_2019'].to(device)
            sem_mask_2024 = batch['sem_mask_2024'].to(device)
            cd_mask = batch['cd_mask'].to(device)

            optimizer.zero_grad()
            
            ################################################################################
            with autocast('cuda'):
                sem_out1, sem_out2, change_out = model(img_t2019, img_t2024)
                sem_loss = (sem_criterion(sem_out1, sem_mask_2019) + sem_criterion(sem_out2, sem_mask_2024)) / 2
                change_loss = change_criterion(change_out, cd_mask)
                loss = sem_loss + change_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ################################################################################

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics_list = []

        with torch.no_grad():
            for batch in val_loader:
                img_t2019 = batch['img_t2019'].to(device)
                img_t2024 = batch['img_t2024'].to(device)
                sem_mask_2019 = batch['sem_mask_2019'].to(device)
                sem_mask_2024 = batch['sem_mask_2024'].to(device)
                cd_mask = batch['cd_mask'].to(device)

                sem_out1, sem_out2, change_out = model(img_t2019, img_t2024)

                sem_loss = (sem_criterion(sem_out1, sem_mask_2019) +
                           sem_criterion(sem_out2, sem_mask_2024)) / 2
                change_loss = change_criterion(change_out, cd_mask)
                loss = sem_loss + change_loss

                val_loss += loss.item()

                # Calculate metrics for this batch
                preds = torch.argmax(change_out, dim=1).cpu().numpy()
                targets = cd_mask.cpu().numpy()
                batch_metrics = calculate_metrics(preds, targets, num_cd_classes,
                                                  weighted_metrics=weighted_metrics)
                val_metrics_list.append(batch_metrics)

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        val_metrics = {}
        for key in val_metrics_list[0].keys():
          val_metrics[key] = np.mean([m[key] for m in val_metrics_list])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': float(best_val_loss),
                'train_loss': float(avg_train_loss),
                'val_accuracy': float(val_metrics['accuracy']),
                'val_kappa': float(val_metrics['kappa']),
                'val_miou': float(val_metrics['miou']),
                'val_f1_score': float(val_metrics['f1_score']),
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_metrics'].append(val_metrics)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Metrics - Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Kappa: {val_metrics['kappa']:.4f}, "
              f"mIoU: {val_metrics['miou']:.4f}, "
              f"F1 score: {val_metrics['f1_score']:.4f}, ")

    return model, history

def save_training_history(history, checkpoint_path, save_path, save_path_bestepoch):
    # Convert tensors or arrays in the history to lists for JSON serialization
    processed_history = {}
    for phase, metrics in history.items():
        if isinstance(metrics, list):  # Check if metrics is a list
            processed_history[phase] = [
                {metric: (value.tolist() if hasattr(value, 'tolist') else value)
                 for metric, value in entry.items()} if isinstance(entry, dict) else entry
                for entry in metrics
            ]
        else:  # Handle non-list entries
            processed_history[phase] = metrics.tolist() if hasattr(metrics, 'tolist') else metrics

    # Save processed history to a JSON file
    with open(save_path, 'w') as f:
        json.dump(processed_history, f, indent=4)

    print(f"Training history saved to: {save_path}")

    # Load checkpoint and inspect contents
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # print("\nCheckpoint contents:")
    # for key in checkpoint.keys():
    #     print(f"- {key}")

    epoch_data = {
        'best_epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'val_accuracy': checkpoint['val_accuracy'],
        'val_kappa': checkpoint['val_kappa'],
        'val_miou': checkpoint['val_miou'],
        'val_f1_score': checkpoint['val_f1_score']
    }

    # Save the best epoch info
    with open(save_path_bestepoch, 'w') as f:
        json.dump(epoch_data, f, indent=4)
    print(f"Best epoch info saved to: {save_path_bestepoch}")
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from tqdm import tqdm
from utils.metrics import calculate_metrics

def calculate_computational_metrics(model, test_loader, device='cuda'):
    """
    Calculate parameters, FLOPs, and inference time for the PCC model using the entire test dataset.

    Args:
        model: The PCC model to evaluate
        test_loader: DataLoader containing the test dataset
        device: Device to run the model on

    Returns:
        dict: Dictionary containing the computational metrics
    """
    model.eval()  # Set to evaluation mode

    # Calculate parameters in MB
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #params_mb = params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)

    # Get a sample input for FLOPs calculation
    batch = next(iter(test_loader))
    img_t2019 = batch['img_t2019'][0:1].to(device)
    img_t2024 = batch['img_t2024'][0:1].to(device)

    #Calculate FLOPs per forward pass
    try:
        #Try using fvcore if available
        from fvcore.nn import FlopCountAnalysis

        def get_flops_with_fvcore():
            flops = FlopCountAnalysis(model, (img_t2019, img_t2024))
            flops.set_op_handle("aten::_convolution", None)  # Skip this op to avoid errors
            flops.set_op_handle("aten::addmm", None)
            return flops.total()  # Return raw FLOPs

        flops_per_example = get_flops_with_fvcore()
        gflops = flops_per_example / 10**9  # Convert to GFLOPs for display
    except (ImportError, Exception) as e:
        print(f"Error calculating FLOPs: {e}")
        print("Using a rough estimate of FLOPs instead")
        #Make a rough estimate based on model parameters
        flops_per_example = params * 2  # Very rough approximation
        gflops = flops_per_example / 10**9  # Convert to GFLOPs

    # Measure inference time for the entire test dataset
    try:
        total_examples = 0

        # Warm-up run with a single batch
        with torch.no_grad():
            batch = next(iter(test_loader))
            img_t2019 = batch['img_t2019'].to(device)
            img_t2024 = batch['img_t2024'].to(device)
            _ = model(img_t2019, img_t2024)

        # Make sure GPU operations are completed
        if device == 'cuda':
            torch.cuda.synchronize()

        # Timed run for all test data
        start_time = time.time()

        with torch.no_grad():
            for batch in test_loader:
                img_t2019 = batch['img_t2019'].to(device)
                img_t2024 = batch['img_t2024'].to(device)
                _ = model(img_t2019, img_t2024)
                total_examples += img_t2019.size(0)

        # Make sure GPU operations are completed
        if device == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()

        total_inference_time = end_time - start_time

        # Scale to get s/100e
        inference_time_per_100 = total_inference_time * (100 / total_examples)

        print(f"\nProcessed {total_examples} examples in {total_inference_time:.2f} seconds")

    except Exception as e:
        print(f"Error measuring inference time: {e}")
        inference_time_per_100 = float('nan')
        total_examples = 0
        total_inference_time = 0

    metrics = {
        "params": params,
        "flops_gflops": gflops,
        "inference_time_s_per_100e": inference_time_per_100,
        "total_examples": total_examples,
        "total_inference_time": total_inference_time
    }

    # Print computational metrics
    print("\nComputational Metrics:")
    print(f"Parameters (MB): {metrics['params_mb']:.2f}")
    print(f"FLOPs (GFlops): {metrics['flops_gflops']:.2f}")
    print(f"Inference Time (s/100e): {metrics['inference_time_s_per_100e']:.4f}")

    return metrics

def plot_results_pcc(img1, img2, sem_pred1, sem_pred2, sem_gt1, sem_gt2,
                    change_pred, change_gt):
    """Plot the results from the PCC model in notebook cells"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Plot images
    axes[0, 0].imshow(img1.cpu().permute(1, 2, 0))
    axes[0, 0].set_title('Image 2019')
    axes[0, 1].imshow(img2.cpu().permute(1, 2, 0))
    axes[0, 1].set_title('Image 2024')

    # Plot semantic predictions and ground truth
    axes[0, 2].imshow(sem_pred1.cpu())
    axes[0, 2].set_title('Semantic Pred 2019')
    axes[0, 3].imshow(sem_pred2.cpu())
    axes[0, 3].set_title('Semantic Pred 2024')

    axes[1, 0].imshow(sem_gt1.cpu())
    axes[1, 0].set_title('Semantic GT 2019')
    axes[1, 1].imshow(sem_gt2.cpu())
    axes[1, 1].set_title('Semantic GT 2024')

    # Plot change detection results
    axes[1, 2].imshow(change_pred.cpu())
    axes[1, 2].set_title('Change Prediction')
    axes[1, 3].imshow(change_gt.cpu())
    axes[1, 3].set_title('Change GT')

    plt.tight_layout()
    plt.show()


def test_model_pcc(model, test_loader, checkpoint_path, device, num_samples_to_plot=5,
                   num_cd_classes=3, weighted_metrics=False):
    """Test the model with enhanced metrics for both change detection and semantic segmentation
    Optimized version to reduce RAM usage"""

    # Load checkpoint with weights_only to reduce memory
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    model.eval()

    # Initialize accumulators for metrics instead of storing all batch metrics
    cd_metrics_sum = {
        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
        'f1_score': 0.0, 'miou': 0.0, 'kappa': 0.0
    }
    sem_2019_metrics_sum = cd_metrics_sum.copy()
    sem_2024_metrics_sum = cd_metrics_sum.copy()

    # Keep track of total processed samples
    total_processed = 0

    # Store only a few samples for visualization instead of all predictions
    visualization_samples = []

    # Process each batch separately
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            img_t2019 = batch['img_t2019'].to(device)
            img_t2024 = batch['img_t2024'].to(device)
            sem_mask_2019 = batch['sem_mask_2019'].to(device)
            sem_mask_2024 = batch['sem_mask_2024'].to(device)
            cd_mask = batch['cd_mask'].to(device)

            batch_size = img_t2019.size(0)
            total_processed += batch_size

            # Forward pass
            sem_out1, sem_out2, change_out = model(img_t2019, img_t2024)

            # Get predictions
            sem_pred1 = torch.argmax(sem_out1, dim=1)
            sem_pred2 = torch.argmax(sem_out2, dim=1)
            change_pred = torch.argmax(change_out, dim=1)

            # Calculate metrics for all tasks
            # Move tensors to CPU and convert to numpy in smaller chunks to reduce memory
            batch_cd_metrics = calculate_metrics(
                change_pred.cpu().numpy(),
                cd_mask.cpu().numpy(),
                num_cd_classes,
                weighted_metrics=weighted_metrics
            )

            batch_sem_2019_metrics = calculate_metrics(
                sem_pred1.cpu().numpy(),
                sem_mask_2019.cpu().numpy(),
                num_cd_classes,
                weighted_metrics=weighted_metrics
            )

            batch_sem_2024_metrics = calculate_metrics(
                sem_pred2.cpu().numpy(),
                sem_mask_2024.cpu().numpy(),
                num_cd_classes,
                weighted_metrics=weighted_metrics
            )

            # Add to accumulators (weighted by batch size)
            for key in cd_metrics_sum:
                cd_metrics_sum[key] += batch_cd_metrics[key] * batch_size
                sem_2019_metrics_sum[key] += batch_sem_2019_metrics[key] * batch_size
                sem_2024_metrics_sum[key] += batch_sem_2024_metrics[key] * batch_size

            # Store a sample for visualization if needed
            if len(visualization_samples) < num_samples_to_plot:
                # Randomly decide whether to keep this sample
                if np.random.rand() < 0.1:  # 10% chance to keep any given batch
                    # Only keep the first image from the batch to save memory
                    visualization_samples.append({
                        'img_t2019': img_t2019[0].cpu(),
                        'img_t2024': img_t2024[0].cpu(),
                        'sem_pred1': sem_pred1[0].cpu(),
                        'sem_pred2': sem_pred2[0].cpu(),
                        'sem_mask_2019': sem_mask_2019[0].cpu(),
                        'sem_mask_2024': sem_mask_2024[0].cpu(),
                        'change_pred': change_pred[0].cpu(),
                        'cd_mask': cd_mask[0].cpu()
                    })

            # Explicitly free memory
            del img_t2019, img_t2024, sem_mask_2019, sem_mask_2024, cd_mask
            del sem_out1, sem_out2, change_out, sem_pred1, sem_pred2, change_pred
            del batch_cd_metrics, batch_sem_2019_metrics, batch_sem_2024_metrics

            # Force garbage collection periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

    # Calculate final averages
    final_cd_metrics = {k: v / total_processed for k, v in cd_metrics_sum.items()}
    final_sem_2019_metrics = {k: v / total_processed for k, v in sem_2019_metrics_sum.items()}
    final_sem_2024_metrics = {k: v / total_processed for k, v in sem_2024_metrics_sum.items()}

    # Calculate average semantic segmentation metrics
    avg_sem_metrics = {}
    for key in final_sem_2019_metrics.keys():
        avg_sem_metrics[key] = (final_sem_2019_metrics[key] + final_sem_2024_metrics[key]) / 2

    # Print detailed metrics
    print("\n=== Change Detection Metrics ===")
    print(f"Overall Accuracy: {final_cd_metrics['accuracy']:.4f}")
    print(f"Kappa Score: {final_cd_metrics['kappa']:.4f}")
    print(f"mIoU: {final_cd_metrics['miou']:.4f}")
    print(f"F1 score: {final_cd_metrics['f1_score']:.4f}")

    print("\n=== 2019 Semantic Segmentation Metrics ===")
    print(f"Overall Accuracy: {final_sem_2019_metrics['accuracy']:.4f}")
    print(f"Kappa Score: {final_sem_2019_metrics['kappa']:.4f}")
    print(f"mIoU: {final_sem_2019_metrics['miou']:.4f}")
    print(f"F1 score: {final_sem_2019_metrics['f1_score']:.4f}")

    print("\n=== 2024 Semantic Segmentation Metrics ===")
    print(f"Overall Accuracy: {final_sem_2024_metrics['accuracy']:.4f}")
    print(f"Kappa Score: {final_sem_2024_metrics['kappa']:.4f}")
    print(f"mIoU: {final_sem_2024_metrics['miou']:.4f}")
    print(f"F1 score: {final_sem_2024_metrics['f1_score']:.4f}")

    print("\n=== Average Semantic Segmentation Metrics ===")
    print(f"Overall Accuracy: {avg_sem_metrics['accuracy']:.4f}")
    print(f"Kappa Score: {avg_sem_metrics['kappa']:.4f}")
    print(f"mIoU: {avg_sem_metrics['miou']:.4f}")
    print(f"F1 score: {avg_sem_metrics['f1_score']:.4f}")

    # Calculate computational metrics on a smaller subset to save memory
    print("\nCalculating computational metrics on the test dataset...")
    computational_metrics = calculate_computational_metrics(model, test_loader, device)

    # Plot random samples that we've collected
    print(f"\nPlotting {len(visualization_samples)} samples...")
    for sample in visualization_samples:
        plot_results_pcc(
            sample['img_t2019'],
            sample['img_t2024'],
            sample['sem_pred1'],
            sample['sem_pred2'],
            sample['sem_mask_2019'],
            sample['sem_mask_2024'],
            sample['change_pred'],
            sample['cd_mask']
        )
        # Clear sample after plotting to free memory
        del sample

    # Collect results and return
    results = {
        'change_detection': final_cd_metrics,
        'semantic_2019': final_sem_2019_metrics,
        'semantic_2024': final_sem_2024_metrics,
        'semantic_average': avg_sem_metrics,
        'computational_metrics': computational_metrics
    }

    return results

def save_test_metrics(history, save_path):
    # Convert tensors or arrays in the history to lists for JSON serialization
    def process_value(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.int32, np.int64)):
            return int(value)
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(item) for item in value]
        else:
            return value

    processed_history = {}
    for phase, metrics in history.items():
        processed_history[phase] = process_value(metrics)

    # Save processed history to a JSON file
    with open(save_path, 'w') as f:
        json.dump(processed_history, f, indent=4)

    print(f"Test metrics saved to: {save_path}")


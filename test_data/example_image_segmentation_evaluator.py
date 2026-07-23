"""
Example Custom Evaluator for Image Segmentation Tasks

This evaluator demonstrates how to evaluate image segmentation submissions
where participants submit ZIP files containing segmentation masks.

The evaluator computes:
- Intersection over Union (IoU) / Jaccard Index
- Dice Coefficient
- Pixel Accuracy

Requirements:
- Ground truth contains image IDs and paths to ground truth masks
- Submission ZIP contains predicted segmentation masks (PNG/JPG files)
- Mask filenames should match the image IDs in ground truth

Example ground truth format:
[
    {"image_id": "img001", "mask_path": "masks/img001_gt.png", "category": "person"},
    {"image_id": "img002", "mask_path": "masks/img002_gt.png", "category": "car"},
    ...
]

Example submission structure:
submission.zip/
    img001.png
    img002.png
    ...
"""

import os
import glob
import numpy as np
from PIL import Image


def load_mask(mask_path):
    """
    Load a segmentation mask from file.
    
    Args:
        mask_path: Path to mask image file
        
    Returns:
        Binary numpy array (0 or 1)
    """
    try:
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask_array = np.array(mask)
        # Binarize: any non-zero pixel is considered foreground
        binary_mask = (mask_array > 0).astype(np.uint8)
        return binary_mask
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU) / Jaccard Index.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        IoU score (0-1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        # Both masks are empty
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """
    Compute Dice Coefficient (F1 Score for segmentation).
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Dice score (0-1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_pixels = pred_mask.sum() + gt_mask.sum()
    
    if total_pixels == 0:
        # Both masks are empty
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / total_pixels


def compute_pixel_accuracy(pred_mask, gt_mask):
    """
    Compute pixel-wise accuracy.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Accuracy (0-1)
    """
    correct_pixels = (pred_mask == gt_mask).sum()
    total_pixels = pred_mask.size
    
    return correct_pixels / total_pixels if total_pixels > 0 else 0.0


def compute_scores(extraction_path, ground_truth_df):
    """
    Evaluate image segmentation submissions.
    
    Args:
        extraction_path: Path to extracted ZIP contents with predicted masks
        ground_truth_df: DataFrame with columns: image_id, mask_path, (optional) category
        
    Returns:
        Tuple of (partial_score, partial_metric, complete_score, complete_metric)
    """
    # Find all predicted mask files in the extraction directory
    pred_mask_files = {}
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        for file_path in glob.glob(os.path.join(extraction_path, ext)):
            # Extract image ID from filename (without extension)
            image_id = os.path.splitext(os.path.basename(file_path))[0]
            pred_mask_files[image_id] = file_path
    
    print(f"Found {len(pred_mask_files)} predicted masks in submission")
    print(f"Ground truth contains {len(ground_truth_df)} images")
    
    # Compute metrics for each image
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    
    matched_count = 0
    missing_predictions = []
    
    for idx, row in ground_truth_df.iterrows():
        image_id = row['image_id']
        gt_mask_path = row.get('mask_path', '')
        
        # Check if prediction exists for this image
        if image_id not in pred_mask_files:
            missing_predictions.append(image_id)
            # Assign zero scores for missing predictions
            iou_scores.append(0.0)
            dice_scores.append(0.0)
            pixel_accuracies.append(0.0)
            continue
        
        # Load ground truth mask
        # Note: In a real scenario, ground truth masks would be loaded from a secure location
        # For this example, we assume gt_mask_path is relative to a known directory
        # or we generate synthetic ground truth
        
        # For demonstration, we'll create a dummy ground truth
        # In production, you would load from: gt_mask = load_mask(gt_mask_path)
        # Here we create a simple synthetic mask for testing
        gt_mask = np.random.randint(0, 2, size=(256, 256), dtype=np.uint8)
        
        # Load predicted mask
        pred_mask_path = pred_mask_files[image_id]
        pred_mask = load_mask(pred_mask_path)
        
        if pred_mask is None:
            # Failed to load prediction
            iou_scores.append(0.0)
            dice_scores.append(0.0)
            pixel_accuracies.append(0.0)
            continue
        
        # Ensure masks have the same shape
        if pred_mask.shape != gt_mask.shape:
            print(f"Warning: Shape mismatch for {image_id}. "
                  f"Predicted: {pred_mask.shape}, Ground truth: {gt_mask.shape}")
            # Resize predicted mask to match ground truth
            pred_mask_img = Image.fromarray(pred_mask * 255)
            pred_mask_img = pred_mask_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
            pred_mask = (np.array(pred_mask_img) > 0).astype(np.uint8)
        
        # Compute metrics
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        pixel_acc = compute_pixel_accuracy(pred_mask, gt_mask)
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_accuracies.append(pixel_acc)
        matched_count += 1
    
    if missing_predictions:
        print(f"Warning: {len(missing_predictions)} predictions missing")
        print(f"Missing image IDs: {missing_predictions[:10]}...")  # Show first 10
    
    # Compute average metrics
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    mean_pixel_acc = np.mean(pixel_accuracies) if pixel_accuracies else 0.0
    
    print(f"\nEvaluation Results:")
    print(f"  Matched predictions: {matched_count}/{len(ground_truth_df)}")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
    
    # Convert to scores (0-100 scale)
    # Use IoU as the primary metric
    partial_score = mean_iou * 100.0
    partial_metric = mean_iou
    
    # Complete score uses Dice coefficient
    complete_score = mean_dice * 100.0
    complete_metric = mean_dice
    
    return (partial_score, partial_metric, complete_score, complete_metric)

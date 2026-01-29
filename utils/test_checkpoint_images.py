#!/usr/bin/env python3
"""
Distill3R Inference Script

Loads a trained Distill3R student model and runs inference on a directory
of images to generate colored 3D point clouds.

Usage:
    python utils/test_checkpoint_images.py path/to/images/ \\
        --checkpoint checkpoints/distill3r/last.ckpt \\
        --output-dir results/

Options:
    --size              Image preprocessing size (default: 518)
    --conf-percentile   Confidence threshold percentile (default: 10, keeps top 90%)
    --device            Device to use (default: cuda)

Output:
    {scene_name}_student.ply    Colored point cloud
    {scene_name}_info.txt       Inference metadata
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import open3d as o3d
import glob

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Fast3R imports for image loading
sys.path.insert(0, str(project_root / "external" / "fast3r"))
from fast3r.dust3r.utils.image import load_images

# Distill3R imports
from distill3r.student.distillation_module import DistillationLitModule


def run_timed(func, *args, **kwargs):
    """Run function with timing."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def save_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path):
    """Save point cloud to PLY file."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pc, compressed=True)


def load_student_model(checkpoint_path: str, device):
    """Load trained student model from checkpoint."""
    print(f"Loading student model from {checkpoint_path}")
    
    # Check if this is a directory or file
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        # Find the .ckpt file in the directory
        ckpt_files = list(ckpt_path.glob("*.ckpt"))
        if not ckpt_files:
            raise ValueError(f"No .ckpt files found in {checkpoint_path}")
        if len(ckpt_files) > 1:
            print(f"Multiple checkpoints found, using: {ckpt_files[0]}")
        checkpoint_path = str(ckpt_files[0])
    
    # Load the Lightning module checkpoint with map_location
    # strict=False to ignore old teacher_filter keys that were removed
    lit_module = DistillationLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False
    )
    lit_module = lit_module.to(device)
    lit_module.eval()
    
    # Extract the student model and ensure eval mode
    student_model = lit_module.student
    student_model = student_model.to(device)
    student_model.eval()

    # CRITICAL FIX: Disable random image IDs at inference time
    # Random IDs are for training augmentation, but break geometric consistency at inference
    if hasattr(student_model, 'decoder') and hasattr(student_model.decoder, 'random_image_idx_embedding'):
        print(f"BEFORE: random_image_idx_embedding = {student_model.decoder.random_image_idx_embedding}")
        student_model.decoder.random_image_idx_embedding = False
        print(f"AFTER:  random_image_idx_embedding = {student_model.decoder.random_image_idx_embedding}")
        print("  → Using SEQUENTIAL image IDs [0,1,2,3,...] for geometric consistency")

    print(f"Student model loaded successfully")
    print(f"Total parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # Quick sanity check
    with torch.no_grad():
        test_view = {
            'img': torch.randn(1, 3, 224, 518, device=device),
            'instance': 'test',
            'label': 'test',
            'true_shape': torch.tensor([[224, 518]], device=device)
        }
        try:
            test_out = student_model([test_view])
            if test_out and test_out[0]:
                for k, v in test_out[0].items():
                    if isinstance(v, torch.Tensor) and (torch.isnan(v).any() or torch.isinf(v).any()):
                        print(f"WARNING: Model outputs NaN/Inf in {k}!")
        except Exception as e:
            print(f"Warning during sanity check: {e}")
    
    return student_model


def load_images_from_directory(image_dir: Path, target_size=518):
    """Load all images from directory using Fast3R's load_images (same as training)."""
    import torchvision.transforms.functional as TF

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(image_dir / ext)))
        image_paths.extend(glob.glob(str(image_dir / ext.upper())))

    image_paths = sorted(image_paths)  # Sort for consistent ordering

    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")

    print(f"Found {len(image_paths)} images in {image_dir}")
    for i, path in enumerate(image_paths[:5]):  # Show first 5
        print(f"  {i+1}: {Path(path).name}")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")

    # Load original images for coloring
    original_images = [Image.open(path).convert('RGB') for path in image_paths]

    # CRITICAL: Use Fast3R's load_images (same preprocessing as training!)
    # This matches what CachedSampleDataset uses during training
    views = load_images([str(p) for p in image_paths], size=target_size, square_ok=False, verbose=False)

    # Resize to exact DUNE-compatible dimensions (224x518)
    # Fast3R's load_images gives us properly preprocessed images, we just need to resize
    for view in views:
        if view['img'].shape[-2:] != (224, 518):
            # Resize to 224x518 (DUNE patch_size=14 compatible: 224/14=16, 518/14=37)
            view['img'] = TF.resize(view['img'], [224, 518], antialias=True)
            view['true_shape'] = torch.tensor([[224, 518]], device=view['img'].device)

    return views, original_images, image_paths


def run_student_inference(student_model, views, device):
    """Run inference using the student model with Fast3R's inference function."""
    # CRITICAL: Use Fast3R's inference function (same as Fast3R test)
    # This handles the proper batching and collation
    from fast3r.dust3r.inference_multiview import inference

    with torch.no_grad():
        out = inference(views, student_model, device=device, dtype=torch.float32, verbose=False)
        predictions = out["preds"]

        # DIAGNOSTIC: Check if global = local (no transform learned)
        print("\n" + "="*60)
        print("DIAGNOSTIC: Checking if model learned coordinate transforms")
        print(f"Analyzing ALL {len(predictions)} views")
        print("="*60)

        all_diffs = []
        for i, pred in enumerate(predictions):  # Check ALL views
            if 'pts3d_in_other_view' in pred and 'pts3d_local' in pred:
                global_pts = pred['pts3d_in_other_view'][0].cpu().numpy()
                local_pts = pred['pts3d_local'][0].cpu().numpy()

                diff = np.abs(global_pts - local_pts)
                valid = np.isfinite(diff).all(axis=-1)

                if valid.sum() > 0:
                    mean_diff = diff[valid].mean()
                    max_diff = diff[valid].max()

                    all_diffs.append(mean_diff)

                    print(f"\nView {i:2d}:")
                    print(f"  Valid points: {valid.sum():,}/{valid.size:,}")
                    print(f"  Mean |global - local|: {mean_diff:.6f}")
                    print(f"  Max |global - local|: {max_diff:.6f}")

                    if i == 0:
                        print(f"  Expected: ~0.000000 (view 0 is anchor, global = local)")
                        if mean_diff > 0.01:
                            print(f"  ❌ CRITICAL: View 0 should have global ≈ local!")
                        else:
                            print(f"  ✓ Correct")
                    else:
                        print(f"  Expected: LARGE (view {i} should be transformed to view 0's frame)")
                        if mean_diff < 0.5:
                            print(f"  ⚠️  Suspiciously small transform")
                        else:
                            print(f"  ✓ Appears transformed")
                else:
                    print(f"\nView {i:2d}: No valid points to compare")
            else:
                print(f"\nView {i:2d}: Missing pts3d_in_other_view or pts3d_local")

        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS:")
        print("="*60)
        if len(all_diffs) > 1:
            all_diffs = np.array(all_diffs)
            print(f"Total views analyzed: {len(all_diffs)}")
            print(f"\nDifference statistics (mean |global - local|):")
            print(f"  View 0: {all_diffs[0]:.6f} (should be ~0)")
            print(f"  Views 1+: min={all_diffs[1:].min():.6f}, max={all_diffs[1:].max():.6f}, mean={all_diffs[1:].mean():.6f}")
            print(f"  Overall variance: {all_diffs.var():.6f}")
            print(f"\nAll view differences: {[f'{d:.4f}' for d in all_diffs]}")

            # Diagnose the problem
            print("\n" + "-"*60)
            print("DIAGNOSIS:")
            if all_diffs[0] > 0.01:
                print("❌ CRITICAL BUG: View 0 global ≠ local!")
                print("   This violates the fundamental constraint that view 0 is the anchor.")
                print("   Possible causes:")
                print("   1. Teacher cache was generated incorrectly")
                print("   2. Student model never learned the identity transform for view 0")

            if all_diffs.std() < 0.05:
                print("❌ CRITICAL BUG: All views have UNIFORM difference!")
                print(f"   Mean diff: {all_diffs.mean():.4f} ± {all_diffs.std():.4f}")
                print("   This means the model is NOT learning geometric transforms.")
                print("   It's applying a constant offset to all views.")

            if all_diffs[0] < 0.01 and all_diffs[1:].mean() > 1.0:
                print("✓ Model appears to be learning transforms correctly")

            print("="*60 + "\n")
        else:
            print("Not enough views to analyze")
            print("="*60 + "\n")

    return {"preds": predictions}


def extract_point_cloud_from_preds(preds, original_images, confidence_percentile=10):
    """Extract colored point cloud from student predictions using percentile-based filtering.
    
    Args:
        preds: Model predictions
        original_images: Original images for coloring
        confidence_percentile: Percentile threshold (10 = keep top 90% most confident points)
    """
    points = []
    colors = []
    
    for view_idx, pred in enumerate(preds):
        # Get 3D coordinates - use Fast3R key format
        xyz_tensor = None
        if 'pts3d_in_other_view' in pred:
            xyz_tensor = pred['pts3d_in_other_view']
        elif 'pts3d' in pred:
            xyz_tensor = pred['pts3d']
        
        if xyz_tensor is None:
            print(f"No 3D points found in prediction for view {view_idx}")
            continue
        
        # Convert to numpy - handle batch dimension
        if xyz_tensor.dim() == 4:  # [B, H, W, 3]
            xyz = xyz_tensor[0].cpu().numpy()
        elif xyz_tensor.dim() == 3:  # [H, W, 3]
            xyz = xyz_tensor.cpu().numpy()
        else:
            print(f"Unexpected xyz shape for view {view_idx}: {xyz_tensor.shape}")
            continue
        
        # Get confidence - student model outputs 'conf_local'
        conf_tensor = pred.get('conf_local')
        if conf_tensor is None:
            # Fallback to 'conf' for compatibility
            conf_tensor = pred.get('conf')
            if conf_tensor is None:
                print(f"No confidence found for view {view_idx}")
                continue
        
        # Convert confidence to numpy
        if conf_tensor.dim() == 3:  # [B, H, W]
            conf = conf_tensor[0].cpu().numpy()
        elif conf_tensor.dim() == 2:  # [H, W]
            conf = conf_tensor.cpu().numpy()
        else:
            print(f"Unexpected conf shape for view {view_idx}: {conf_tensor.shape}")
            continue
        
        H, W, _ = xyz.shape
        
        # Apply Fast3R-style filtering
        # 1. Finite coordinate check
        finite_mask = np.isfinite(xyz).all(axis=-1)
        
        # 2. Confidence threshold using percentile (like Fast3R does)
        # Calculate the percentile threshold for this view
        conf_threshold = np.percentile(conf, confidence_percentile)
        conf_mask = conf > conf_threshold
        
        # 3. Check for additional masks (geometric, photometric)
        additional_masks = []
        for mask_key in ('mask_photometric', 'mask_geometric', 'mask_global'):
            if mask_key in pred:
                mask_tensor = pred[mask_key]
                if mask_tensor.dim() == 3:  # [B, H, W]
                    mask = mask_tensor[0].cpu().numpy().astype(bool)
                else:
                    mask = mask_tensor.cpu().numpy().astype(bool)
                additional_masks.append(mask)
        
        # Combine all masks
        valid_mask = finite_mask & conf_mask
        for mask in additional_masks:
            valid_mask = valid_mask & mask
        
        # Debug info
        print(f"View {view_idx} filtering:")
        print(f"  Confidence range: {conf.min():.3f} - {conf.max():.3f} (mean: {conf.mean():.3f})")
        print(f"  Finite pixels: {finite_mask.sum()}/{finite_mask.size}")
        print(f"  High confidence (top {100-confidence_percentile:.0f}%): {conf_mask.sum()}/{conf_mask.size}")
        print(f"  Final valid: {valid_mask.sum()}/{valid_mask.size}")
        
        if not valid_mask.any():
            print(f"  No valid points for view {view_idx}")
            continue
        
        # Extract valid points
        valid_xyz = xyz[valid_mask]
        
        # Get real image colors (like Fast3R does)
        if view_idx < len(original_images):
            # Resize original image to match xyz dimensions
            orig_img = original_images[view_idx].resize((W, H), Image.LANCZOS)
            rgb_array = np.asarray(orig_img)
            valid_colors = rgb_array[valid_mask]
        else:
            # Fallback to view colors if original images not available
            view_colors = np.array([
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255]
            ])
            color = view_colors[view_idx % len(view_colors)]
            valid_colors = np.tile(color, (len(valid_xyz), 1))
        
        points.append(valid_xyz)
        colors.append(valid_colors)
        
        print(f"  Added {len(valid_xyz):,} points to point cloud")
    
    if points:
        all_points = np.concatenate(points, axis=0)
        all_colors = np.concatenate(colors, axis=0)
        return all_points, all_colors
    else:
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Test student model checkpoint inference on directory of images")
    parser.add_argument("image_dir", type=str,
                        help="Directory containing images to process")
    parser.add_argument("--checkpoint", default="checkpoints/rtx6000_balanced_loss/final_model.ckpt", 
                        help="Path to student model checkpoint")
    parser.add_argument("--output-dir", default="inference_results",
                        help="Output directory for point clouds")
    parser.add_argument("--size", type=int, default=518,
                        help="Image size for Fast3R preprocessing (default=518, training uses 518)")
    parser.add_argument("--conf-percentile", type=float, default=10,
                        help="Confidence percentile threshold (10 = keep top 90% most confident points)")
    parser.add_argument("--device", default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    image_dir = Path(args.image_dir)
    
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    print("=" * 60)
    print("DISTILL3R CHECKPOINT IMAGE INFERENCE TEST")
    print("=" * 60)
    
    # Load student model
    student_model = load_student_model(args.checkpoint, device)
    
    # Load images from directory (using Fast3R's preprocessing)
    views, original_images, image_paths = load_images_from_directory(
        image_dir, target_size=args.size
    )

    print(f"\nProcessing {len(views)} images from {image_dir}")
    print(f"Image size: {args.size} (Fast3R preprocessing)")
    print(f"Confidence percentile: {args.conf_percentile}% (keeping top {100-args.conf_percentile:.0f}% of points)")
    
    # Run student inference
    print(f"\nRunning student inference...")
    out, inference_time = run_timed(
        run_student_inference, student_model, views, device
    )
    
    print(f"Student inference time: {inference_time:.2f}s")
    
    # Extract point cloud with original images for coloring
    print(f"\nExtracting point cloud...")
    points, colors = extract_point_cloud_from_preds(
        out["preds"], original_images, confidence_percentile=args.conf_percentile
    )
    
    if points is not None:
        print(f"\nTotal points extracted: {len(points):,}")
        
        # Save point cloud
        scene_name = image_dir.name
        output_path = output_dir / f"{scene_name}_student.ply"
        save_ply(points, colors, output_path)
        print(f"Saved point cloud: {output_path}")
        
        # Save timing and info
        info_path = output_dir / f"{scene_name}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Image directory: {image_dir}\n")
            f.write(f"Number of images: {len(views)}\n")
            f.write(f"Image size: 224x518 (landscape, matches training)\n")
            f.write(f"Confidence percentile: {args.conf_percentile}% (kept top {100-args.conf_percentile:.0f}% of points)\n")
            f.write(f"Inference time: {inference_time:.2f}s\n")
            f.write(f"Total points: {len(points):,}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"\nImage files processed:\n")
            for i, path in enumerate(image_paths):
                f.write(f"  {i+1}: {Path(path).name}\n")
        
        print(f"Saved info: {info_path}")
    else:
        print("No valid points extracted")
    
    print(f"\nInference test completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
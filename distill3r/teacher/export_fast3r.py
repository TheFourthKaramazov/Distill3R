DEBUG = False  # Set to True to enable demo visualization (view-drop, PnP, fused PLY)
import csv
import json
import numpy as np
import yaml
import torch
from tqdm import tqdm
from pathlib import Path
import time
import pandas as pd
import sys
import open3d as o3d
import os

# Add external fast3r to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "external" / "fast3r"))

# Add project root to path for distill3r imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference

from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from distill3r.teacher.rle_helpers import encode_rle, decode_rle

from PIL import Image
import open3d as o3d


def run_timed(func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def save_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64)/255.0)
    o3d.io.write_point_cloud(str(path), pc, compressed=True)


def sample_views_temporally(group, max_views=20, seed=42, sample_id=0):
    """Sample exactly max_views (20) CONSECUTIVE views from group.

    TEMPORAL LOGIC PRESERVED:
    - Sorts by sample_id to maintain temporal/chronological order
    - Uses consecutive chunks (np.arange) for sequential frames
    - Each sample_id gets a different temporal window

    Always returns exactly max_views by sampling with replacement if needed.
    """
    total_views = len(group)

    # TEMPORAL STEP 1: Sort by sample_id to ensure temporal/chronological order
    if 'sample_id' in group.columns:
        group_sorted = group.sort_values('sample_id').reset_index(drop=True)
    else:
        group_sorted = group.reset_index(drop=True)

    np.random.seed(seed + sample_id)

    if total_views >= max_views:
        # TEMPORAL STEP 2: Select CONSECUTIVE chunk for temporal locality
        # Each sample gets different temporal window:
        # sample_id=0 → frames [0:20], sample_id=1 → frames [20:40], etc.
        start_idx = sample_id * max_views

        # Handle end of sequence
        if start_idx + max_views > total_views:
            start_idx = max(0, total_views - max_views)

        # CRITICAL: Use np.arange for CONSECUTIVE indices (not random!)
        indices = np.arange(start_idx, start_idx + max_views)
    else:
        # Not enough views: sample with replacement
        indices = np.random.choice(total_views, max_views, replace=True)
        # TEMPORAL STEP 3: Sort to maintain temporal order even with replacement
        indices = np.sort(indices)

    sampled = group_sorted.iloc[indices].reset_index(drop=True)
    return sampled


def determine_samples_per_scene(total_views, max_samples=100, required_views_per_sample=20):
    """Calculate how many 20-view samples can be created from a scene.

    Each sample uses exactly 20 consecutive views for temporal locality.

    Args:
        total_views: Number of views available in the scene
        max_samples: Maximum samples to create per scene
        required_views_per_sample: Views per sample (fixed at 20)

    Returns:
        Number of samples that can be created
    """
    if total_views < 2:
        return 0

    # Each sample needs exactly 20 consecutive views
    num_samples = int(total_views / required_views_per_sample)

    return min(max_samples, num_samples)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate teacher cache from Fast3R')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (PLY, poses)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing cache')
    args = parser.parse_args()

    # Configuration from environment variables (set by script) or defaults
    manifest_path = Path(os.getenv('TEACHER_MANIFEST_PATH', 'processed_data/images/manifest.csv'))
    base_cache_dir = Path(os.getenv('TEACHER_CACHE_DIR', 'caches/teacher_cache'))
    max_views_per_sample = int(os.getenv('TEACHER_MAX_VIEWS', '20'))
    image_size = int(os.getenv('TEACHER_IMAGE_SIZE', '512'))
    target_height = int(os.getenv('TEACHER_TARGET_HEIGHT', '224'))
    target_width = int(os.getenv('TEACHER_TARGET_WIDTH', '518'))
    debug_mode = args.debug or os.getenv('TEACHER_DEBUG', 'False').lower() == 'true'
    device_name = os.getenv('TEACHER_DEVICE', 'auto')
    resume_mode = args.resume or os.getenv('TEACHER_RESUME', 'False').lower() == 'true'
    max_samples_per_scene = int(os.getenv('TEACHER_MAX_SAMPLES_PER_SCENE', '10000')) # set to high number to use all data (10k)
    
    # Override global DEBUG with environment setting
    global DEBUG
    DEBUG = debug_mode
    
    # Device selection
    if device_name == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    dtype = torch.float16
    timings = {}

    # Ensure manifest exists
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run 'python scripts/generate_manifest.py' first")
        return
    
    base_cache_dir.mkdir(exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Max views per sample: {max_views_per_sample}")
    print(f"Max samples per scene: {max_samples_per_scene}")
    print(f"Image size: {image_size}")
    print(f"Target output resolution: {target_height}x{target_width}")
    print(f"Debug mode: {DEBUG}")
    print(f"Resume mode: {resume_mode}")

    # Load model & lit_module
    print("Loading Fast3R model...")
    (model, timings["load_model"]) = run_timed(Fast3R.from_pretrained, "jedyang97/Fast3R_ViT_Large_512")
    model.to(device).eval()

    # Use Fast3R's default: random_image_idx_embedding=True for better data augmentation
    print(f"Decoder setting: random_image_idx_embedding={model.decoder.random_image_idx_embedding}")

    (lit, timings["load_lit"]) = run_timed(MultiViewDUSt3RLitModule.load_for_inference, model)
    lit.eval()

    # Read full manifest and group by dataset first, then scene_id
    df = pd.read_csv(manifest_path)

    print(f"Loaded manifest with {len(df)} images across {df['dataset'].nunique()} datasets")

    # Track total samples created across all datasets
    total_samples_created = 0

    for dataset_name, dataset_group in df.groupby("dataset"):
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Create dataset-specific cache directory
        dataset_cache_dir = base_cache_dir / dataset_name
        dataset_cache_dir.mkdir(exist_ok=True)
        
        scene_groups = list(dataset_group.groupby("scene_id"))
        
        for scene_id, full_group in tqdm(scene_groups, desc=f"{dataset_name} Scenes"):
            total_views = len(full_group)
            
            # Determine how many samples we can create
            num_samples = determine_samples_per_scene(
                total_views, 
                max_samples_per_scene, 
                max_views_per_sample  # Require exactly this many views per sample
            )
            
            if num_samples == 0:
                print(f"Skipping scene {scene_id}: only {total_views} views (need ≥2)")
                continue

            if num_samples > 1:
                print(f"Scene {scene_id}: {total_views} views -> {num_samples} samples (20 views each)")

            # Track samples created for this scene
            scene_samples_created = 0

            # Create multiple samples from this scene
            for sample_idx in range(num_samples):
                # Create sample-specific directory
                if num_samples > 1:
                    scene_cache_dir = dataset_cache_dir / f"{scene_id}_sample{sample_idx:02d}"
                else:
                    scene_cache_dir = dataset_cache_dir / scene_id
                
                scene_cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Skip if resuming and sample already exists
                if resume_mode and (scene_cache_dir / "sampled_views.json").exists():
                    existing_npz = len(list(scene_cache_dir.glob("*.npz")))
                    if existing_npz > 0:
                        continue

                # Sample subset of views for this specific sample
                group = sample_views_temporally(
                    full_group,
                    max_views=max_views_per_sample,
                    seed=42,
                    sample_id=sample_idx
                )
                
                # Save the sampled view information for student training
                sampled_view_info = {
                    'dataset': dataset_name,
                    'scene_id': scene_id,
                    'sample_id': sample_idx,
                    'total_samples': num_samples,
                    'original_count': total_views,
                    'sampled_count': len(group),
                    'sampled_image_paths': group["image_path"].tolist(),
                    'sampled_indices': group.index.tolist()
                }
                
                # Save to JSON for student training reference
                with open(scene_cache_dir / "sampled_views.json", 'w') as f:
                    json.dump(sampled_view_info, f, indent=2)
                
                # Load all views for this sample
                image_paths = [Path(p) for p in group["image_path"]]
                
                (views, timings["load_images"]) = run_timed(
                    load_images, [str(p) for p in image_paths], size=image_size, verbose=False
                )

                # Clear GPU cache before inference
                torch.cuda.empty_cache()

                # Inference on all views of the sample
                (out, timings["inference"]) = run_timed(
                    inference, views, model, device=device, dtype=dtype, verbose=False, profiling=False
                )
                preds = out["preds"]

                # sampling with replacement already guarantees exactly max_views_per_sample views
                # so preds, views, and image_paths should all have exactly 20 elements

                # Compute cross-view geometric consistency masks
                (_, timings["align_pts"]) = run_timed(
                    lit.align_local_pts3d_to_global,
                    preds=preds,
                    views=views,
                    min_conf_thr_percentile=85
                )

                # Estimate camera poses via PnP and save
                if DEBUG:
                    (pose_batch, timings["pose_pnp"]) = run_timed(
                        MultiViewDUSt3RLitModule.estimate_camera_poses,
                        preds,
                        niter_PnP=100,
                        focal_length_estimation_method="first_view_from_global_head"
                    )
                    poses = pose_batch[0]
                    np.save(scene_cache_dir / "poses_c2w.npy", np.stack(poses))

                # Collect all views into lists for stacking (CONSOLIDATED FORMAT)
                # Maintain exact order from zip(preds, image_paths) to match sampled_views.json
                xyz_local_list = []
                xyz_global_list = []
                conf_local_list = []
                conf_global_list = []
                mask_rle_list = []

                for pred, image_path in zip(preds, image_paths):
                    xyz_local = pred["pts3d_local"][0].cpu().numpy()
                    xyz_global = pred["pts3d_local_aligned_to_global"][0].cpu().numpy()
                    conf_local = pred["conf_local"][0].cpu().numpy()
                    conf_global = pred["conf"][0].cpu().numpy()

                    # Resize outputs to DUNE-compatible resolution (both dims divisible by 14)
                    current_h, current_w = xyz_local.shape[:2]

                    if (current_h, current_w) != (target_height, target_width):
                        # Convert to torch tensors for interpolation
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                        # Resize xyz arrays: [H, W, 3] -> [1, 3, H, W] -> interpolate -> [H, W, 3]
                        xyz_local_t = torch.from_numpy(xyz_local.astype(np.float32)).to(device)
                        xyz_local_t = xyz_local_t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                        xyz_local_t = F.interpolate(xyz_local_t, size=(target_height, target_width), mode='bilinear', align_corners=False)
                        xyz_local = xyz_local_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

                        xyz_global_t = torch.from_numpy(xyz_global.astype(np.float32)).to(device)
                        xyz_global_t = xyz_global_t.permute(2, 0, 1).unsqueeze(0)
                        xyz_global_t = F.interpolate(xyz_global_t, size=(target_height, target_width), mode='bilinear', align_corners=False)
                        xyz_global = xyz_global_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

                        # Resize confidence arrays: [H, W] -> [1, 1, H, W] -> interpolate -> [H, W]
                        conf_local_t = torch.from_numpy(conf_local.astype(np.float32)).to(device)
                        conf_local_t = conf_local_t.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                        conf_local_t = F.interpolate(conf_local_t, size=(target_height, target_width), mode='bilinear', align_corners=False)
                        conf_local = conf_local_t.squeeze(0).squeeze(0).cpu().numpy()

                        conf_global_t = torch.from_numpy(conf_global.astype(np.float32)).to(device)
                        conf_global_t = conf_global_t.unsqueeze(0).unsqueeze(0)
                        conf_global_t = F.interpolate(conf_global_t, size=(target_height, target_width), mode='bilinear', align_corners=False)
                        conf_global = conf_global_t.squeeze(0).squeeze(0).cpu().numpy()

                    # Photometric confidence mask (local) - use consistent threshold
                    conf_threshold = 0.30  # Consistent with PLY generation
                    mask = conf_local > conf_threshold

                    # Encode combined mask (only photometric for now)
                    mask_rle = encode_rle(mask)

                    # Prune low-confidence pixels
                    low_mask = conf_local < conf_threshold
                    xyz_local[low_mask] = np.nan
                    xyz_global[low_mask] = np.nan
                    conf_local[low_mask] = 0
                    conf_global[low_mask] = 0

                    # Apply quantization with explicit dtype enforcement
                    xyz_local_f16 = xyz_local.astype(np.float16)
                    xyz_global_f16 = xyz_global.astype(np.float16)
                    conf_local_f16 = conf_local.astype(np.float16)
                    conf_global_f16 = conf_global.astype(np.float16)

                    # Free original float32 arrays immediately
                    del xyz_local, xyz_global, conf_local, conf_global

                    # Append to lists (maintains order from sampled_views.json)
                    xyz_local_list.append(xyz_local_f16)
                    xyz_global_list.append(xyz_global_f16)
                    conf_local_list.append(conf_local_f16)
                    conf_global_list.append(conf_global_f16)
                    mask_rle_list.append(mask_rle)

                # Stack all views into consolidated arrays [N, H, W, ...]
                xyz_local_stacked = np.stack(xyz_local_list, axis=0)    # [N, 224, 518, 3]
                xyz_global_stacked = np.stack(xyz_global_list, axis=0)  # [N, 224, 518, 3]
                conf_local_stacked = np.stack(conf_local_list, axis=0)  # [N, 224, 518]
                conf_global_stacked = np.stack(conf_global_list, axis=0)  # [N, 224, 518]
                masks_array = np.array(mask_rle_list, dtype=object)  # [N] object array of RLE

                # Save single consolidated file
                consolidated_path = scene_cache_dir / "consolidated.npz"
                np.savez_compressed(
                    consolidated_path,
                    xyz_local=xyz_local_stacked,
                    xyz_global=xyz_global_stacked,
                    conf_local=conf_local_stacked,
                    conf_global=conf_global_stacked,
                    masks=masks_array,
                    num_views=len(preds),
                    inference_time=np.float32(timings.get("inference", 0))
                )

                # Clear GPU cache after processing sample
                torch.cuda.empty_cache()

                # Fuse and save colored point cloud for this sample
                if DEBUG:
                    xyz_list, rgb_list = [], []
                    for pred, img_path in zip(preds, image_paths):
                        try:
                            orig = Image.open(img_path).convert("RGB")
                            xyz = pred.get("local_pts3d_global_est", pred["pts3d_in_other_view"])[0].cpu().numpy()
                            H, W = xyz.shape[:2]
                            rgb = np.array(orig.resize((W, H), Image.LANCZOS))
                            mask = np.isfinite(xyz).all(-1) & (pred["conf"][0].cpu().numpy() > 0.30)
                            if mask.any():
                                xyz_list.append(xyz[mask])
                                rgb_list.append(rgb[mask])
                        except Exception as e:
                            print(f"Warning: Failed to process image {img_path}: {e}")
                            continue
                            
                    if xyz_list:
                        xyz_all = np.concatenate(xyz_list, 0)
                        rgb_all = np.concatenate(rgb_list, 0)
                        save_ply(xyz_all, rgb_all, scene_cache_dir / "fused_fast3r.ply")

                # Sample successfully created
                scene_samples_created += 1

            # Scene completed - print stats
            if num_samples > 1:
                print(f"Scene {scene_id}: created {scene_samples_created}/{num_samples} samples")

            # Accumulate to global totals
            total_samples_created += scene_samples_created

    # All datasets completed - print final summary
    print("\n" + "=" * 80)
    print("CACHE GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total samples created: {total_samples_created:,}")
    print(f"Views per sample: 20 (consecutive temporal frames)")
    print(f"Cache saved to: {base_cache_dir}")


if __name__ == "__main__":
    main()

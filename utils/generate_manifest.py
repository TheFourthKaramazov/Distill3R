#!/usr/bin/env python3
"""
Generate Multi-Dataset Manifest Script

This script creates a unified manifest (CSV file) containing metadata for all configured
datasets. The manifest includes image paths, scene IDs, intrinsics, and other metadata
needed for training and teacher cache generation.

Usage:
    python scripts/generate_manifest.py [--config CONFIG_PATH] [--output OUTPUT_PATH]

Arguments:
    --config CONFIG_PATH    Path to data configuration YAML (default: configs/data_paths.yaml)
    --output OUTPUT_PATH    Output CSV path (default: data_manifest.csv)

The script will:
1. Load dataset configurations from the YAML file
2. Process each enabled dataset according to its pattern and sample_frac
3. Generate processed images and metadata JSONs in prep.out_dir
4. Create a unified CSV manifest with columns:
   - dataset: Dataset name (CO3D, ARKitScenes, MegaDepth, BlendedMVS)
   - split: Train/val split designation
   - sample_id: Unique identifier for each image
   - scene_id: Scene grouping identifier
   - image_path: Path to processed PNG file
   - K: Camera intrinsic matrix (3x3)
   - H, W: Image height and width

Configuration:
    Edit configs/data_paths.yaml to adjust:
    - sample_frac: Controls how many scenes to sample from each dataset
    - resize_max: Maximum edge size for processed images
    - enabled: Whether to include a dataset (defaults to true if not specified)

Examples:
    # Generate manifest with default settings
    python scripts/generate_manifest.py
    
    # Use custom config and output paths
    python scripts/generate_manifest.py --config my_config.yaml --output my_manifest.csv
    
    # Check the generated manifest
    python -c "import pandas as pd; df=pd.read_csv('data_manifest.csv'); print(df.groupby('dataset').size())"
"""

import argparse
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
from distill3r.data.datasets import get_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-dataset manifest for Distill3r training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/data_paths.yaml",
        help="Path to data configuration YAML file (default: configs/data_paths.yaml)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data_manifest.csv",
        help="Output CSV manifest path (default: data_manifest.csv)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    print("Generating multi-dataset manifest...")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print()
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Generate manifest using the get_manifest function
        print("Processing datasets and generating manifest...")
        df = get_manifest(config_path)
        
        # Get the actual output location from the config
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text())
        prep_config = cfg.get("prep", {})
        actual_out_dir = Path(prep_config.get("out_dir", "processed_data/images"))
        actual_manifest = actual_out_dir / "manifest.csv"
        
        print(f"Manifest saved to: {actual_manifest}")
        
        # Only copy if user explicitly specified a non-default output path
        output_path = Path(args.output)
        if args.output != "data_manifest.csv" and actual_manifest != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(actual_manifest, output_path)
            print(f"Manifest also copied to: {output_path}")
        
        print(f"Total samples: {len(df)}")
        
        # Show breakdown by dataset
        print("\nDataset breakdown:")
        dataset_stats = df.groupby('dataset').agg({
            'scene_id': 'nunique',  # Number of unique scenes
            'sample_id': 'count'    # Number of images
        }).rename(columns={'scene_id': 'scenes', 'sample_id': 'images'})
        
        for dataset, stats in dataset_stats.iterrows():
            print(f"  {dataset:12}: {stats['scenes']:2d} scenes, {stats['images']:4d} images")
        
        total_scenes = dataset_stats['scenes'].sum()
        total_images = dataset_stats['images'].sum()
        print(f"\nTotal: {total_scenes} scenes, {total_images} images")
        
        if args.verbose:
            # Show sample paths to verify structure
            print("\nSample paths by dataset:")
            for dataset in df['dataset'].unique():
                sample_path = df[df['dataset'] == dataset]['image_path'].iloc[0]
                print(f"  {dataset:12}: {sample_path}")
        
        print("\nReady for teacher cache export!")
        
    except Exception as e:
        print(f"Error generating manifest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
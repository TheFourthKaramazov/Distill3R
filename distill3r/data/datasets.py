"""
distill3r.data.datasets
-----------------------

Utilities for assembling Distill3R’s image manifest and preparing
960‑pixel PNG copies with adjusted intrinsics.

Public API is Indempotent: calling `get_manifest` multiple times
will not overwrite existing PNGs or JSONs. It will only copy
and resize images that are missing.

The manifest is a CSV file with the following columns:
- dataset: name of the dataset (e.g. CO3D)
- split: train / val / test
- sample_id: unique identifier for the image (e.g. 0001)
- scene_id: unique identifier for the scene/sequence (e.g. CO3D_apple_110_13051_23361)
- image_path: path to the resized PNG
- K: camera intrinsics (3x3 matrix) (place holder to be changed when values are estimated)
- H: height of the image
- W: width of the image


Typical usage
~~~~~~~~~~~~~
>>> from distill3r.data.datasets import get_manifest
>>> df = get_manifest("configs/data_paths.yaml")
>>> print(df.head())

The first invocation copies / resizes raw frames and writes:

    data/images/<DATASET>/<SPLIT>/<sample_id>.png
    data/images/<DATASET>/<SPLIT>/<sample_id>.json   # intrinsics

It also stores a manifest CSV at:

    data/images/manifest.csv
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# Restrict imports to avoid circular dependencies
__all__ = [
    "DatasetSpec",
    "PrepCfg",
    "prepare_split",
    "get_manifest",
]

# -----------------------------------------------------------------------------#
# Configuration dataclasses
# -----------------------------------------------------------------------------#


@dataclass
class DatasetSpec:
    """Specification for *one* subset (e.g. CO3D train)."""

    name: str  # CO3D, ARK, …
    root: Path  # directory containing raw files
    pattern: str  # glob underneath root, e.g. "**/*.jpg"
    sample_frac: float  # 0.03 → 3 %
    split: str  # train / val / test


@dataclass
class PrepCfg:
    """Generic preprocessing settings."""

    resize_max: int = 960  # longest edge after scaling
    out_dir: Path = Path("data/images")  # where to write PNGs


# -----------------------------------------------------------------------------#
# Scene ID extraction functions
# -----------------------------------------------------------------------------#


def _extract_scene_id(file_path: Path, dataset_name: str) -> str:
    """Extract scene ID from file path based on dataset structure.
    
    Parameters
    ----------
    file_path : Path
        Path to the image file
    dataset_name : str
        Name of the dataset (CO3D, ARKitScenes, MegaDepth, BlendedMVS, Habitat, ScanNetPP)
        
    Returns
    -------
    scene_id : str
        Unique scene identifier that groups related images
    """
    parts = file_path.parts
    
    if dataset_name == "CO3D" or dataset_name == "CO3D_single":
        # Structure: dataset/co3d/category/sequence/images/frame_*.jpg
        # OR: dataset/co3d_single_sequence/category/sequence/images/frame_*.jpg
        # Scene ID: CO3D_category_sequence or CO3D_single_category_sequence
        for i, part in enumerate(parts):
            if (part == "co3d" or part == "co3d_single_sequence") and i + 2 < len(parts):
                category = parts[i + 1]
                sequence = parts[i + 2]
                prefix = "CO3D_single" if part == "co3d_single_sequence" else "CO3D"
                return f"{prefix}_{category}_{sequence}"
        
    elif dataset_name == "ARKitScenes":
        # Structure: raw_data/arkitscenes/3dod/Training/scene_id/scene_id_frames/lowres_wide/*.png
        # Scene ID: ARKitScenes_scene_id
        for i, part in enumerate(parts):
            if part in ["Training", "Validation"] and i + 1 < len(parts):
                scene_id = parts[i + 1]
                return f"ARKitScenes_{scene_id}"
                
    elif dataset_name == "MegaDepth":
        # Structure: dataset/megadepth/scene_id/dense*/imgs/*.jpg
        # Scene ID: MegaDepth_scene_id_dense_dir
        for i, part in enumerate(parts):
            if part == "megadepth" and i + 2 < len(parts):
                scene_id = parts[i + 1]
                dense_dir = parts[i + 2]  # e.g., dense0, dense1
                return f"MegaDepth_{scene_id}_{dense_dir}"
                
    elif dataset_name == "BlendedMVS":
        # Structure: raw_data/blendedmvs/scene_id/blended_images/*.jpg
        # Scene ID: BlendedMVS_scene_id
        for i, part in enumerate(parts):
            if part == "blendedmvs" and i + 1 < len(parts):
                scene_id = parts[i + 1]
                return f"BlendedMVS_{scene_id}"

    elif dataset_name == "Habitat":
        # Structure: dataset/habitat/scene_id/matterport_color_images/*.jpg
        # Scene ID: Habitat_scene_id
        for i, part in enumerate(parts):
            if part == "habitat" and i + 1 < len(parts):
                scene_id = parts[i + 1]
                return f"Habitat_{scene_id}"

    elif dataset_name == "ScanNetPP":
        # Structure: dataset/scannetpp/scene_id/dslr/resized_images/*.JPG
        # Scene ID: ScanNetPP_scene_id
        for i, part in enumerate(parts):
            if part == "scannetpp" and i + 1 < len(parts):
                scene_id = parts[i + 1]
                return f"ScanNetPP_{scene_id}"

    # Fallback: use parent directory name
    return f"{dataset_name}_{file_path.parent.name}"


def _group_files_by_scene(files: List[Path], dataset_name: str) -> Dict[str, List[Path]]:
    """Group files by scene ID.
    
    Parameters
    ----------
    files : List[Path]
        List of image file paths
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    scene_groups : Dict[str, List[Path]]
        Dictionary mapping scene IDs to lists of file paths
    """
    scene_groups = {}
    for file_path in files:
        scene_id = _extract_scene_id(file_path, dataset_name)
        if scene_id not in scene_groups:
            scene_groups[scene_id] = []
        scene_groups[scene_id].append(file_path)
    return scene_groups


def _sample_scenes(scene_groups: Dict[str, List[Path]], sample_frac: float, split: str, val_scene_split: bool = False) -> List[Path]:
    """Sample scenes and return all images from selected scenes.
    
    Parameters
    ----------
    scene_groups : Dict[str, List[Path]]
        Dictionary mapping scene IDs to lists of file paths
    sample_frac : float
        Fraction of scenes to sample
    split : str
        Split type (train/val)
    val_scene_split : bool
        Whether this dataset uses scene-level validation splitting
        
    Returns
    -------
    selected_files : List[Path]
        List of file paths from selected scenes
    """
    scene_ids = list(scene_groups.keys())
    scene_ids.sort()  # Ensure reproducible ordering
    
    if val_scene_split and len(scene_ids) > 1:
        # Scene-level splitting: deterministically assign scenes to train/val
        random.seed(42)  # Fixed seed for reproducible splits
        random.shuffle(scene_ids)
        
        if split == "val":
            # Take the last 15% of scenes for validation
            val_count = max(1, int(len(scene_ids) * 0.15))
            selected_scene_ids = scene_ids[-val_count:]
            # Apply sample_frac to validation scenes
            keep_count = max(1, int(len(selected_scene_ids) * sample_frac + 0.5))
            selected_scene_ids = selected_scene_ids[:keep_count]
        else:  # train
            # Take the first 85% of scenes for training
            train_count = len(scene_ids) - max(1, int(len(scene_ids) * 0.15))
            train_scene_ids = scene_ids[:train_count]
            # Apply sample_frac to training scenes
            keep_count = max(1, int(len(train_scene_ids) * sample_frac + 0.5))
            selected_scene_ids = train_scene_ids[:keep_count]
    else:
        # Regular sampling: just take sample_frac of all scenes
        random.seed(0)  # For backwards compatibility
        random.shuffle(scene_ids)
        keep_count = max(1, int(len(scene_ids) * sample_frac + 0.5))
        selected_scene_ids = scene_ids[:keep_count]
    
    # Collect all files from selected scenes
    selected_files = []
    for scene_id in selected_scene_ids:
        selected_files.extend(scene_groups[scene_id])
    
    return selected_files


# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#


def _resize_and_save(src: Path, dst: Path, max_edge: int) -> tuple[int, int, float]:
    """Load `src`, down‑scale so max(H, W)=`max_edge`, save as 8‑bit PNG.

    Returns
    -------
    H, W : int
        New height and width.
    scale : float
        Multiplicative scale that was applied (new / old).
    """


    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src)
    h0, w0 = img.shape[:2]

    # Downscale if necessary (compute scale factor)
    scale = min(1.0, max_edge / max(h0, w0))

    # Use cv2.INTER_LANCZOS4 for downscaling, highest quality but SLOWER
    # if speed is a concern, change to cv2.INTER_LINEAR 
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w0 * scale)), int(round(h0 * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    # Store and compress by 30%
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    h, w = img.shape[:2]
    return h, w, scale

# Should not be necessary, but used to avoid error 
def _default_intrinsics(h: int, w: int) -> np.ndarray:
    """Fallback pinhole K—principal point at centre, f ≈ max(H,W)."""
    f = max(h, w)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)

# Used for COLMAP comparison or other use cases
def _save_intrinsics(json_path: Path, K: np.ndarray) -> None:
    """Save camera intrinsics to a JSON file."""
    json_path.write_text(json.dumps({"K": K.tolist()}, indent=2))


# -----------------------------------------------------------------------------#
# Split preparation
# -----------------------------------------------------------------------------#


def prepare_split(spec: DatasetSpec, prep: PrepCfg, val_scene_split: bool = False) -> List[Dict]:
    """Copy & resize a split, return manifest rows (dicts).
    Parameters
    ----------
    spec : DatasetSpec
        Specification for the split to prepare.
    prep : PrepCfg
        Preprocessing configuration (resize, output directory).
    val_scene_split : bool
        Whether to use scene-level splitting for train/val
    Returns
    -------
    rows : List[Dict]
        List of dictionaries with keys: dataset, split, sample_id, scene_id, image_path, K, H, W.
        Each dict corresponds to one image.
    
    """

    # Recurse under root and get files
    files = sorted(spec.root.glob(spec.pattern))
    if not files:
        print(f"  No files matched for {spec.name} ({spec.root})", file=sys.stderr)
        return []

    # Group files by scene and sample scenes (not individual images)
    scene_groups = _group_files_by_scene(files, spec.name)
    keep = _sample_scenes(scene_groups, spec.sample_frac, spec.split, val_scene_split)

    # One dict per image
    # (dataset, split, sample_id, image_path, K, H, W)
    rows: List[Dict] = []
    for fp in tqdm(keep, desc=f"{spec.name}-{spec.split}", unit="img"):
        # Extract scene ID for this image
        scene_id = _extract_scene_id(fp, spec.name)
        
        # Create unique sample_id by combining scene and frame
        # This prevents collisions between different scenes
        unique_sample_id = f"{scene_id}_{fp.stem}"
        
        # output paths
        out_png = (
            prep.out_dir / spec.name / spec.split / f"{unique_sample_id.lower()}.png"
        )
        out_json = out_png.with_suffix(".json")

        # Avoid second disk read for cached PNG
        if out_png.exists():
            h, w = cv2.imread(str(out_png), cv2.IMREAD_COLOR).shape[:2]
            # faster: read H,W from JSON too:
            meta = json.loads(out_json.read_text())
            h, w = meta["H"], meta["W"]
            K = np.array(meta["K"])
        else:
            # create resized PNG and derive placeholder intrinsics
            h, w, scale = _resize_and_save(fp, out_png, prep.resize_max)
            K = _default_intrinsics(int(h / scale), int(w / scale)) * scale

            # save intrinsics + image size in the JSON side‑car
            meta = {"K": K.tolist(), "H": h, "W": w}
            out_json.write_text(json.dumps(meta, indent=2))


        rows.append(
            {
                "dataset": spec.name,
                "split": spec.split,
                "sample_id": unique_sample_id,  # Use unique sample_id
                "scene_id": scene_id,
                "image_path": str(out_png),
                "K": K,
                "H": h,
                "W": w,
            }
        )

    return rows


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#


def get_manifest(cfg_path: str | Path = "configs/data_paths.yaml") -> pd.DataFrame:
    """Prepare all datasets listed in the YAML and return a manifest DataFrame.

    Subsequent calls are incremental: existing PNGs / JSONs are reused.

    Parameters
    ----------
    cfg_path : str or Path
        Path to the YAML configuration file. The default is "configs/data_paths.yaml".
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the manifest with columns: dataset, split, sample_id, scene_id, image_path, K, H, W.
        Each row corresponds to one image.
 
    """

    # Load YAML config and fill prepcfg class
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    
    # Convert out_dir to Path if it's a string
    prep_config = cfg.get("prep", {})
    if "out_dir" in prep_config and isinstance(prep_config["out_dir"], str):
        prep_config["out_dir"] = Path(prep_config["out_dir"])
    
    prep = PrepCfg(**prep_config)

    all_rows: List[Dict] = []

    # prepare each split with DatasetSpec
    for ds in cfg["datasets"]:
        spec = DatasetSpec(
            name=ds["name"],
            root=Path(ds["root"]).expanduser(),
            pattern=ds["pattern"],
            sample_frac=float(ds["sample_frac"]),
            split=ds["split"],
        )
        val_scene_split = ds.get("val_scene_split", False)
        all_rows.extend(prepare_split(spec, prep, val_scene_split)) # make manifest

    df = pd.DataFrame(all_rows)
    out_csv = prep.out_dir / "manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Manifest saved to: {out_csv}")
    return df
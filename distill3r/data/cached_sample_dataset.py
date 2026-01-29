
"""
Cached Sample Dataset for Distill3R



This dataset ensures exact consistency between teacher cache generation and student training

by reading the sampled_views.json files created during cache generation and using the

exact same image paths and groupings.

"""



import json

import random

from pathlib import Path

from typing import Dict, List, Tuple, Optional

import torch

import torch.nn.functional as F

from torch.utils.data import Dataset

import numpy as np

from PIL import Image

import torchvision.transforms as transforms



# Fast3R imports for image loading and processing

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "external" / "fast3r"))

from fast3r.dust3r.utils.image import load_images





class CachedSampleDataset(Dataset):

    """

    Dataset that reads exact view samples from teacher cache generation.

    

    Ensures perfect consistency between teacher cache and student training

    by using the same image paths and groupings saved in sampled_views.json.

    """

    

    def __init__(

        self,

        teacher_cache_dir: str,

        image_size: int = 512,

        num_views: int = 6,

        transform: Optional[transforms.Compose] = None,

        max_samples: Optional[int] = None,

        encoder_type: str = "dune"

    ):

        self.teacher_cache_dir = Path(teacher_cache_dir)

        self.image_size = image_size

        self.num_views = num_views

        self.transform = transform

        self.max_samples = max_samples

        self.encoder_type = encoder_type

        

        # Collect all samples from cache directories

        self.samples = self._collect_samples()

        

        if self.max_samples and len(self.samples) > self.max_samples:

            self.samples = self.samples[:self.max_samples]

            print(f"Limited to {len(self.samples)} samples (max_samples={self.max_samples}) from {teacher_cache_dir}")

        else:

            print(f"Found {len(self.samples)} cached samples in {teacher_cache_dir}")

    

    def _collect_samples(self) -> List[Dict]:

        """Collect all sample information from sampled_views.json files."""

        samples = []

        

        if not self.teacher_cache_dir.exists():

            raise ValueError(f"Teacher cache directory not found: {self.teacher_cache_dir}")

        

        # Traverse dataset directories

        for dataset_dir in self.teacher_cache_dir.iterdir():

            if not dataset_dir.is_dir():

                continue

                

            # Traverse scene/sample directories

            for scene_dir in dataset_dir.iterdir():

                if not scene_dir.is_dir():

                    continue

                

                sampled_views_file = scene_dir / "sampled_views.json"

                if not sampled_views_file.exists():

                    continue

                

                try:

                    with open(sampled_views_file, 'r') as f:

                        sample_info = json.load(f)

                    

                    # Use final_image_paths if available (after confidence filtering)

                    image_paths = sample_info.get('final_image_paths', 

                                                sample_info.get('sampled_image_paths', []))

                    

                    if len(image_paths) < 2:  # Need at least 2 views

                        continue

                    

                    sample_data = {

                        'dataset': sample_info['dataset'],

                        'scene_id': sample_info['scene_id'],

                        'sample_id': sample_info.get('sample_id', 0),

                        'image_paths': image_paths,

                        'cache_dir': scene_dir,

                        'total_views': len(image_paths)

                    }

                    

                    samples.append(sample_data)

                    

                except Exception as e:

                    print(f"Warning: Failed to load sample info from {sampled_views_file}: {e}")

                    continue

        

        return samples

    

    def __len__(self) -> int:

        return len(self.samples)

    

    def __getitem__(self, idx: int) -> Dict:

        """

        Get a training sample with exact consistency to teacher cache.

        

        Returns:

            Dictionary containing:

            - 'images': List of PIL Images

            - 'paths': List of image paths  

            - 'dataset': Dataset name

            - 'scene_id': Scene identifier

            - 'sample_id': Sample identifier

            - 'cache_dir': Path to cache directory for this sample

        """

        sample = self.samples[idx]



        # Get the image paths for this sample

        all_image_paths = sample['image_paths']



        # Use all views from cache (always exactly 20 consecutive views)

        selected_paths = all_image_paths

        # No padding needed - cache generation ensures exactly 20 views



        # Load images using Fast3R's image loading

        try:

            # Convert to absolute paths if they're relative

            abs_paths = []

            for path_str in selected_paths:

                path = Path(path_str)

                if not path.is_absolute():

                    # Assume relative to project root

                    path = Path.cwd() / path

                abs_paths.append(str(path))

            

            # Use Fast3R's load_images function with same settings as teacher cache generation

            fast3r_images = load_images(abs_paths, size=self.image_size, square_ok=False, verbose=False)

            

            # Extract PIL images from Fast3R format

            images = []

            for img_data in fast3r_images:

                if isinstance(img_data, dict) and 'img' in img_data:

                    images.append(img_data['img'])  # Extract PIL image

                elif isinstance(img_data, Image.Image):

                    images.append(img_data)

                else:

                    # Fallback: load image directly

                    img_path = abs_paths[len(images)]

                    images.append(Image.open(img_path).convert('RGB'))

            

        except Exception as e:

            print(f"Error loading images for sample {idx}: {e}")

            print(f"Paths: {selected_paths}")

            # Return dummy data to avoid training crash

            dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='black')

            images = [dummy_image] * min(len(selected_paths), self.num_views)

        

        return {

            'images': images,

            'paths': selected_paths,

            'dataset': sample['dataset'],

            'scene_id': sample['scene_id'],

            'sample_id': sample['sample_id'],

            'cache_dir': sample['cache_dir'],

            'num_views': len(images),

            'encoder_type': self.encoder_type

        }

    

    def get_cache_info(self, idx: int) -> Dict:

        """Get cache directory and metadata for a sample."""

        sample = self.samples[idx]

        return {

            'cache_dir': sample['cache_dir'],

            'dataset': sample['dataset'],

            'scene_id': sample['scene_id'],

            'sample_id': sample['sample_id']

        }









def load_teacher_cache_for_sample(cache_dir: Path, image_paths: List[str]) -> Dict:

    """Load teacher cache from consolidated format, repeating views to match padded image_paths."""

    teacher_data = {}



    # Load single consolidated file

    consolidated_path = cache_dir / "consolidated.npz"

    if not consolidated_path.exists():

        raise FileNotFoundError(f"Consolidated cache not found: {consolidated_path}")



    try:

        cache_data = np.load(consolidated_path, allow_pickle=True)



        # Load stacked arrays

        xyz_local_all = cache_data['xyz_local']     # [N, 224, 518, 3]

        xyz_global_all = cache_data['xyz_global']   # [N, 224, 518, 3]

        conf_local_all = cache_data['conf_local']   # [N, 224, 518]

        conf_global_all = cache_data['conf_global'] # [N, 224, 518]

        masks_all = cache_data['masks']             # [N] object array of RLE

        num_views = int(cache_data['num_views'])



        # All samples have exactly 20 views - direct indexing (no modulo wrapping)

        for i in range(len(image_paths)):

            # Direct index since cache always has 20 views matching image_paths

            cache_idx = i



            view_data = {

                'pts3d_local': torch.from_numpy(xyz_local_all[cache_idx].astype(np.float32)),

                'pts3d_global': torch.from_numpy(xyz_global_all[cache_idx].astype(np.float32)),

                'conf_local': torch.from_numpy(conf_local_all[cache_idx].astype(np.float32)),

                'conf_global': torch.from_numpy(conf_global_all[cache_idx].astype(np.float32))

            }



            # Decode mask if needed

            if len(masks_all) > cache_idx:

                mask_rle = list(masks_all[cache_idx])

                from distill3r.teacher.rle_helpers import decode_rle

                H, W = xyz_local_all.shape[1:3]

                mask = decode_rle(mask_rle, (H, W))

                view_data['mask'] = torch.from_numpy(mask).bool()



            teacher_data[f'view_{i}'] = view_data



        cache_data.close()



    except Exception as e:

        raise RuntimeError(f"Failed to load consolidated cache from {consolidated_path}: {e}")



    return teacher_data





def create_cached_samples_collate_fn():

    """

    Factory function to create a collate function.

    

    Returns:

        Collate function configured for cached samples

    """

    def collate_cached_samples_with_config(batch: List[Dict]) -> Dict:

        return collate_cached_samples(batch)

    

    return collate_cached_samples_with_config





def collate_cached_samples(batch: List[Dict]) -> Dict:

    """

    Collate function for CachedSampleDataset.



    Handles variable number of views per sample and loads corresponding teacher cache data.

    """

    # Determine target resolution based on encoder type

    encoder_type = "dune"  # default

    for sample in batch:

        if 'encoder_type' in sample:

            encoder_type = sample['encoder_type']

            break



    if encoder_type == "dinov3":

        target_h, target_w = 224, 512  # DINOv3-compatible (patch_size=16)

    else:

        target_h, target_w = 224, 518  # DUNE-compatible (patch_size=14)



    # Stack images into tensor batches

    all_images = []

    all_paths = []

    all_datasets = []

    all_scene_ids = []

    all_sample_ids = []

    all_cache_dirs = []

    all_num_views = []

    all_teacher_data = []



    # All samples have exactly 20 views - no padding needed

    max_views = 20



    for sample in batch:

        images = sample['images']



        # Verify all samples have exactly 20 views (no padding)

        if len(images) != max_views:

            raise ValueError(f"Expected exactly {max_views} views per sample, got {len(images)}. "

                           f"Cache regeneration required with fixed 20-view logic.")



        # Convert PIL images to tensors and resize

        image_tensors = []

        for img in images[:max_views]:

            if isinstance(img, Image.Image):

                img_tensor = transforms.ToTensor()(img)  # [C, H, W]

            else:

                # Fast3R tensor format [1, C, H, W] -> squeeze to [C, H, W]

                img_tensor = img.squeeze(0) if img.dim() == 4 and img.shape[0] == 1 else img



            # Ensure we have [C, H, W] format

            if img_tensor.dim() != 3:

                raise ValueError(f"Expected 3D tensor [C, H, W], got {img_tensor.shape}")



            # Ensure landscape mode (W >= H) for Fast3R compatibility

            if img_tensor.shape[-1] < img_tensor.shape[-2]:  # W < H (portrait)

                img_tensor = img_tensor.transpose(-1, -2)  # Transpose to landscape



            # Resize to target dimensions BEFORE stacking

            if img_tensor.shape[-2:] != (target_h, target_w):

                img_tensor = F.interpolate(

                    img_tensor.unsqueeze(0), size=(target_h, target_w),

                    mode='bilinear', align_corners=False

                ).squeeze(0)



            image_tensors.append(img_tensor)



        # Load teacher cache data for this sample

        teacher_data = load_teacher_cache_for_sample(sample['cache_dir'], sample['paths'])



        all_images.append(torch.stack(image_tensors))

        all_paths.append(sample['paths'])

        all_datasets.append(sample['dataset'])

        all_scene_ids.append(sample['scene_id'])

        all_sample_ids.append(sample['sample_id'])

        all_cache_dirs.append(sample['cache_dir'])

        all_num_views.append(sample['num_views'])

        all_teacher_data.append(teacher_data)

    

    return {

        'images': torch.stack(all_images),  # [B, N, C, H, W]

        'paths': all_paths,

        'datasets': all_datasets,

        'scene_ids': all_scene_ids,

        'sample_ids': all_sample_ids,

        'cache_dirs': all_cache_dirs,

        'num_views': all_num_views,

        'max_views': max_views,

        'teacher_data': all_teacher_data,  # List of teacher cache data per sample

        'true_shape': (target_h, target_w)  # Resized image dimensions for correct DPT head output

    }
"""
Distillation Lightning Module for Distill3r

Extends MultiViewDUSt3RLitModule to add knowledge distillation capabilities
while reusing all the existing training infrastructure, metrics, and evaluation.
"""

import sys
import warnings
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
import lightning.pytorch as pl
import numpy as np

# Add external fast3r to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "external" / "fast3r"))

from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from distill3r.student.model import CompressedFast3R
from distill3r.student.loss import DistillLoss


class DistillationLitModule(pl.LightningModule):
    """
    Knowledge distillation Lightning module that extends MultiViewDUSt3RLitModule.
    
    Inherits all training/validation infrastructure while adding:
    - Teacher cache loading and online teacher generation
    - Distillation loss computation  
    - Memory-efficient training
    """
    
    def __init__(
        self,
        # All parameters required from config - NO DEFAULTS
        alpha_g: float,
        alpha_l: float,
        gamma: float,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        max_steps: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        encoder_depth: int,
        encoder_heads: int,
        decoder_depth: int,
        decoder_heads: int,
        encoder_type: str,
        max_views: int,
        max_parallel_views_for_head: int,
        landscape_only: bool,
        teacher_cache_dir: str,
        memory_cleanup_freq: int,
        log_memory_usage: bool,
        gpu_memory_threshold: float,
        # Parameters with defaults
        loss_type: str = 'default',  # 'default' or 'no_conf_weight' for ablation studies
        min_lr: float = 5e-5,  # Default for backward compatibility with old checkpoints
    ):
        # Store configuration for later use
        self._student_config = {
            'img_size': img_size,
            'patch_size': patch_size,
            'embed_dim': embed_dim,
            'encoder_depth': encoder_depth,
            'encoder_heads': encoder_heads,
            'decoder_depth': decoder_depth,
            'decoder_heads': decoder_heads,
            'encoder_type': encoder_type,
            'max_views': max_views,
            'max_parallel_views_for_head': max_parallel_views_for_head,
            'landscape_only': landscape_only
        }
        
        # Initialize parent LightningModule
        super().__init__()
        
        # Store configuration first - ensure numeric types
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.min_lr = float(min_lr)
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Build student model
        self.student = CompressedFast3R(**self._student_config)

        # Build distillation loss (conditional based on loss_type for ablation studies)
        if loss_type == 'no_conf_weight':
            from distill3r.student.loss_no_conf_weight import DistillLossNoConfWeight
            self.distill_loss = DistillLossNoConfWeight(
                alpha_g=alpha_g, alpha_l=alpha_l, gamma=gamma
            )
            print(f"Using ablation loss: DistillLossNoConfWeight (no confidence weighting)")
        elif loss_type == 'student_conf_weight':
            from distill3r.student.loss_student_conf_weight import DistillLossStudentConfWeight
            self.distill_loss = DistillLossStudentConfWeight(
                alpha_g=alpha_g, alpha_l=alpha_l, gamma=gamma
            )
            print(f"Using ablation loss: DistillLossStudentConfWeight (student confidence weighting)")
        else:
            self.distill_loss = DistillLoss(
                alpha_g=alpha_g, alpha_l=alpha_l, gamma=gamma
            )
            print(f"Using standard loss: DistillLoss (with confidence weighting)")
            
        # Filtering is done at export time (conf > 0.30) in export_fast3r.py
        self.teacher_cache_dir = teacher_cache_dir
        self.memory_cleanup_freq = memory_cleanup_freq
        self.log_memory_usage = log_memory_usage
        self.gpu_memory_threshold = gpu_memory_threshold
        
        # Memory management
        self._step_count = 0

        # Online teacher (loaded lazily if needed)
        self._teacher_model = None

        # Throughput tracking
        self._epoch_start_time = None
        self._epoch_samples = 0
        
    def configure_optimizers(self):
        """Configure optimizer for all trainable parameters."""
        # Get all trainable parameters
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Create optimizer with single parameter group
        optimizer = AdamW(
            trainable_params, 
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95), 
            eps=1e-8
        )
        
        # Create scheduler - max_steps must be provided from config
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        total_steps = self.max_steps

        # Warmup + Cosine Annealing
        if self.warmup_steps > 0:
            # Linear warmup from 0 to learning_rate
            def warmup_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

            # Cosine annealing after warmup
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=self.min_lr
            )

            # Combine: warmup then cosine
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
            # No warmup - just cosine
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.min_lr
            )

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}

    def on_train_epoch_start(self):
        """Track epoch start time for throughput calculation."""
        self._epoch_start_time = time.time()
        self._epoch_samples = 0

    def on_train_epoch_end(self):
        """Log throughput metrics at epoch end."""
        if self._epoch_start_time is not None:
            epoch_time = time.time() - self._epoch_start_time

            if self._epoch_samples > 0 and epoch_time > 0:
                samples_per_sec = self._epoch_samples / epoch_time

                # Log throughput metrics
                self.log('train/samples_per_sec', samples_per_sec, sync_dist=True)
                self.log('train/epoch_time_min', epoch_time / 60, sync_dist=True)

                # Print for visibility
                if self.trainer.is_global_zero:
                    print(f"\nEpoch {self.current_epoch} throughput: {samples_per_sec:.2f} samples/sec ({epoch_time/60:.1f} min)")

    # Removed: _load_teacher_model - no longer using online teacher generation
    
    def _load_teacher_cache_for_scene(self, cache_dir):
        """Load all teacher cache files for a scene."""
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return None
        
        scene_cache = {}
        
        # Load all .npz files in the cache directory
        for cache_file in cache_path.glob("*.npz"):
            view_name = cache_file.stem
            
            try:
                cache_data = np.load(cache_file)
                
                # Extract teacher outputs from cache
                teacher_view = {}
                
                # XYZ coordinates
                if 'xyz_l' in cache_data:
                    teacher_view['pts3d_local'] = torch.from_numpy(cache_data['xyz_l']).float()
                if 'xyz_g' in cache_data:
                    teacher_view['pts3d_global'] = torch.from_numpy(cache_data['xyz_g']).float()
                
                # Confidence scores
                if 'conf_local' in cache_data:
                    teacher_view['conf_local'] = torch.from_numpy(cache_data['conf_local']).float()
                if 'conf_global' in cache_data:
                    teacher_view['conf_global'] = torch.from_numpy(cache_data['conf_global']).float()
                
                # Valid mask (decode RLE if present)
                if 'mask' in cache_data:
                    from ..teacher.rle_helpers import decode_rle
                    mask_rle = cache_data['mask'].item()
                    mask = decode_rle(mask_rle, teacher_view['xyz_local'].shape[:2])
                    teacher_view['mask'] = torch.from_numpy(mask).bool()
                else:
                    # Create default mask from confidence if no explicit mask
                    if 'conf_local' in teacher_view:
                        conf = teacher_view['conf_local']
                        teacher_view['mask'] = (conf > 0.3).bool()  # Use confidence threshold
                    else:
                        # Create mask from valid xyz coordinates
                        if 'xyz_local' not in teacher_view:
                            raise ValueError(f"xyz_local missing from teacher cache {cache_file} - required for mask generation")
                        xyz = teacher_view['xyz_local']
                        teacher_view['mask'] = torch.isfinite(xyz).all(dim=-1).bool()
                
                scene_cache[view_name] = teacher_view
                
            except Exception as e:
                print(f"Warning: Failed to load cache file {cache_file}: {e}")
                continue
        
        return scene_cache if scene_cache else None

    def forward(self, views):
        """Forward pass through student model."""
        return self.student(views)
    
    def training_step(self, batch, batch_idx):
        """Training step with distillation loss from teacher cache."""
        # Only accept cached sample format - no fallbacks
        if not isinstance(batch, dict) or 'images' not in batch:
            raise ValueError(f"Expected batch with 'images' key from CachedSampleDataset, got: {batch.keys() if isinstance(batch, dict) else type(batch)}")
        
        if 'teacher_data' not in batch:
            raise ValueError("Expected batch with 'teacher_data' key from CachedSampleDataset")
        
        images = batch['images']  # [B, N, C, H, W]
        teacher_data_list = batch['teacher_data']  # List of teacher data per sample
        
        # Convert to Fast3R format for student forward pass
        batch_size, num_views = images.shape[:2]
        fast3r_views = []
        teacher_targets = []
        
        for b in range(batch_size):
            # Extract views for this batch item (scene)
            scene_views = []
            for v in range(num_views):
                # Convert tensor to Fast3R format (dict with 'img' key)
                img_tensor = images[b, v]  # [C, H, W]
                # Ensure consistent dtype for mixed precision
                if img_tensor.dtype == torch.float16:
                    img_tensor = img_tensor.float()  # Convert half to float for processing
                # Add batch dimension for Fast3R: [C, H, W] -> [1, C, H, W]
                img_tensor = img_tensor.unsqueeze(0)
                view_dict = {
                    'img': img_tensor,
                    'instance': f'view_{v}',
                    'label': f'scene_{b}',
                    'true_shape': torch.tensor([batch['true_shape']]).expand(1, 2)  # [1, 2] format expected by student
                }
                scene_views.append(view_dict)
            fast3r_views.append(scene_views)
            
            # Require teacher data for every scene
            if b >= len(teacher_data_list):
                raise ValueError(f"Missing teacher data for scene {b}")
            
            scene_teacher_cache = teacher_data_list[b]
            if scene_teacher_cache is None or len(scene_teacher_cache) == 0:
                raise ValueError(f"Empty teacher cache for scene {b}")
            
            teacher_targets.append(scene_teacher_cache)
        
        # Process each scene separately
        scene_losses = []
        total_loss_details = {}
        
        for scene_idx, scene_views in enumerate(fast3r_views):
            # Student forward pass on this scene
            student_outputs = self.forward(scene_views)
            
            # Get teacher outputs for this scene (guaranteed to exist)
            scene_teacher_targets = teacher_targets[scene_idx]
            
            # Convert teacher cache format to match student outputs
            formatted_teacher_targets = self._format_teacher_cache_for_distillation(
                scene_teacher_targets, scene_views
            )
            
            if formatted_teacher_targets is None:
                raise ValueError(f"Failed to format teacher targets for scene {scene_idx}")
            
            # Compute distillation loss
            loss, loss_details = self._compute_distillation_loss(
                student_outputs, formatted_teacher_targets, scene_views
            )
            
            # Accumulate loss details
            for key, value in loss_details.items():
                if key not in total_loss_details:
                    total_loss_details[key] = []
                total_loss_details[key].append(value)
            
            scene_losses.append(loss)
        
        # Average loss across scenes
        if not scene_losses:
            raise ValueError("No valid scene losses computed - check data pipeline")
        total_loss = torch.stack(scene_losses).mean()
        
        # Log averaged metrics
        for key, values in total_loss_details.items():
            if values:
                # Values might be floats or tensors
                if isinstance(values[0], torch.Tensor):
                    avg_value = torch.stack(values).mean()
                else:
                    avg_value = sum(values) / len(values)
                self.log(f"train/{key}", avg_value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Memory management and logging (every step)
        self._step_count += 1
        
        # Safety check: Monitor BOTH GPU and system memory
        try:
            import pynvml
            import psutil
            
            # GPU memory check
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            actual_gpu_memory_gb = info.used / 1024**3
            total_gpu_memory_gb = info.total / 1024**3
            gpu_utilization = (info.used / info.total) * 100
            
            # System RAM check - CRITICAL for preventing freezes
            system_memory = psutil.virtual_memory()
            system_ram_percent = system_memory.percent
            # Calculate used memory similar to 'free -h' (total - available)
            system_ram_gb = (system_memory.total - system_memory.available) / 1024**3
            
            # CPU check - detect abnormal usage (use cached value to avoid blocking)
            cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking, uses cached value
            
            # Get more accurate load average for sustained CPU usage
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            cpu_cores = psutil.cpu_count()
            load_percent = (load_avg / cpu_cores) * 100 if cpu_cores else 0
            
            # Log every 100 steps
            if self._step_count % 100 == 0:
                print(f"[Step {self._step_count}] GPU: {actual_gpu_memory_gb:.1f}GB ({gpu_utilization:.1f}%), "
                      f"RAM: {system_ram_gb:.1f}GB ({system_ram_percent:.1f}%), "
                      f"CPU: {cpu_percent:.1f}% (Load: {load_percent:.1f}%)")
            
            # Stop if system RAM is critically high (prevent freeze)
            if system_ram_percent > 85.0:
                print(f"\nSTOPPING TRAINING: HIGH SYSTEM RAM: {system_ram_gb:.1f}GB ({system_ram_percent:.1f}%)")
                print("System freeze imminent - stopping training gracefully")
                self.trainer.should_stop = True
                return total_loss
            
            # Stop if GPU memory too high
            if gpu_utilization > self.gpu_memory_threshold:
                print(f"\nSTOPPING TRAINING: HIGH GPU MEMORY: {actual_gpu_memory_gb:.1f}GB / {total_gpu_memory_gb:.1f}GB ({gpu_utilization:.1f}%)")
                raise RuntimeError(f"GPU memory exceeded threshold: {gpu_utilization:.1f}% > {self.gpu_memory_threshold}%")
                
        except ImportError as e:
            # Fallback protection using PyTorch memory if pynvml not available
            pytorch_memory_gb = torch.cuda.memory_allocated() / 1024**3
            # Try to get total memory from torch
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            pytorch_utilization = (pytorch_memory_gb / total_memory_gb) * 100
            if pytorch_utilization > self.gpu_memory_threshold:
                print(f"\nSTOPPING TRAINING: HIGH GPU MEMORY: {pytorch_memory_gb:.1f}GB / {total_memory_gb:.1f}GB ({pytorch_utilization:.1f}%)")
                raise RuntimeError(f"GPU memory exceeded threshold: {pytorch_utilization:.1f}% > {self.gpu_memory_threshold}%")
        
        # Clean memory at specified frequency
        if self._step_count % self.memory_cleanup_freq == 0:
            torch.cuda.empty_cache()

        # Track samples for throughput calculation
        self._epoch_samples += batch_size

        return total_loss
    
    def _compute_distillation_loss(self, student_outputs, teacher_targets, views):
        """Compute distillation loss between student and teacher."""
        # Use teacher data directly (already filtered at export time)

        # Prepare outputs for loss computation
        pred_dict = self._format_outputs_for_loss(student_outputs)
        target_dict = self._format_outputs_for_loss(teacher_targets)
        
        # Generate proper validity mask
        mask = self._generate_validity_mask(pred_dict, target_dict)
        
        # Validate tensor shapes are compatible
        if pred_dict.get('xyz_global') is not None and target_dict.get('xyz_global') is not None:
            if pred_dict['xyz_global'].shape != target_dict['xyz_global'].shape:
                # Check if shapes are just transposed (H,W swapped)
                pred_shape = pred_dict['xyz_global'].shape
                target_shape = target_dict['xyz_global'].shape
                if (pred_shape[0] == target_shape[0] and pred_shape[1] == target_shape[1] and 
                    pred_shape[2] == target_shape[3] and pred_shape[3] == target_shape[2] and
                    pred_shape[4] == target_shape[4]):
                    # Transpose target to match pred (swap H and W dimensions)
                    for key in ['xyz_global', 'xyz_local']:
                        if key in target_dict:
                            target_dict[key] = target_dict[key].transpose(2, 3)
                    for key in ['conf', 'conf_local', 'conf_global']:
                        if key in target_dict:
                            target_dict[key] = target_dict[key].transpose(2, 3)
                else:
                    raise ValueError(f"Shape mismatch between pred and target xyz_global: "
                                   f"{pred_dict['xyz_global'].shape} vs {target_dict['xyz_global'].shape}")
        
        # Compute distillation loss
        loss, loss_details = self.distill_loss(pred_dict, target_dict, mask)
        
        return loss, loss_details
    
    def _generate_validity_mask(self, pred_dict: Dict, target_dict: Dict) -> torch.Tensor:
        """Generate validity mask for loss computation."""
        # Get dimensions from the first available tensor
        reference_tensor = None
        for key in ['xyz_global', 'xyz_local']:
            if key in pred_dict:
                reference_tensor = pred_dict[key]
                break
            elif key in target_dict:
                reference_tensor = target_dict[key]
                break
        
        if reference_tensor is None:
            # Fallback to any available tensor
            if pred_dict:
                reference_tensor = list(pred_dict.values())[0]
            elif target_dict:
                reference_tensor = list(target_dict.values())[0]
            else:
                raise ValueError("No tensors available to generate validity mask from pred_dict or target_dict")
        
        # Create mask based on valid (finite) values - handle missing dimensions
        if reference_tensor.dim() == 5:  # [B, N, H, W, 3] - correct format
            B, N, H, W, _ = reference_tensor.shape
            mask_shape = (B, N, H, W)
        elif reference_tensor.dim() == 4:  # [B, N, H, W] - correct format
            mask_shape = reference_tensor.shape
        elif reference_tensor.dim() == 3:  # [H, W, 3] - missing B, N
            H, W, _ = reference_tensor.shape
            # This is wrong - we need to fix the source of this tensor
            raise ValueError(f"Tensor missing batch/view dimensions: expected [B, N, H, W, 3], got [H, W, 3] = {reference_tensor.shape}")
        elif reference_tensor.dim() == 2:  # [H, W] - missing B, N  
            H, W = reference_tensor.shape
            # This is wrong - we need to fix the source of this tensor
            raise ValueError(f"Tensor missing batch/view dimensions: expected [B, N, H, W], got [H, W] = {reference_tensor.shape}")
        else:
            raise ValueError(f"Unexpected reference tensor dimensions: {reference_tensor.shape}. Expected 4D or 5D tensor.")
        
        # Start with all valid mask
        mask = torch.ones(mask_shape, dtype=torch.bool, device=reference_tensor.device)
        
        # Mask out invalid values in coordinate tensors
        for key in ['xyz_global', 'xyz_local']:
            if key in target_dict:
                coords = target_dict[key]
                if coords.dim() == 5:  # [B, N, H, W, 3]
                    # Check for finite values across xyz coordinates
                    valid_coords = torch.isfinite(coords).all(dim=-1)  # [B, N, H, W]
                    if mask.shape != valid_coords.shape:
                        # Check if shapes are just transposed
                        if (mask.shape[0] == valid_coords.shape[0] and 
                            mask.shape[1] == valid_coords.shape[1] and
                            mask.shape[2] == valid_coords.shape[3] and 
                            mask.shape[3] == valid_coords.shape[2]):
                            # Transpose valid_coords to match mask
                            valid_coords = valid_coords.transpose(2, 3)
                        else:
                            print(f"DEBUG: Processing {key} tensor")
                            print(f"  coords.shape: {coords.shape}")
                            print(f"  valid_coords.shape: {valid_coords.shape}")
                            print(f"  mask.shape: {mask.shape}")
                            print(f"  reference_tensor.shape: {reference_tensor.shape}")
                            raise ValueError(f"Mask shape mismatch: {mask.shape} != {valid_coords.shape}")
                    mask = mask & valid_coords
        
        # Apply confidence threshold if available
        confidence_threshold = 0.1  # Minimum confidence threshold
        for conf_key in ['conf', 'conf_local', 'conf_global']:
            if conf_key in target_dict:
                conf = target_dict[conf_key]
                if conf.dim() == 4:  # [B, N, H, W]
                    valid_conf = conf > confidence_threshold
                    if mask.shape != valid_conf.shape:
                        # Check if shapes are just transposed
                        if (mask.shape[0] == valid_conf.shape[0] and 
                            mask.shape[1] == valid_conf.shape[1] and
                            mask.shape[2] == valid_conf.shape[3] and 
                            mask.shape[3] == valid_conf.shape[2]):
                            # Transpose valid_conf to match mask
                            valid_conf = valid_conf.transpose(2, 3)
                        else:
                            raise ValueError(f"Confidence mask shape mismatch: {mask.shape} != {valid_conf.shape}")
                    mask = mask & valid_conf
                break  # Use first available confidence
        
        return mask
    
    # Removed: _compute_reconstruction_loss - only using distillation loss from teacher cache
    
    def _format_teacher_cache_for_distillation(self, teacher_cache_data, scene_views):
        """
        Convert teacher cache format to match student output format for distillation.
        
        Teacher cache format: {'view_0': {data}, 'view_1': {data}, ...}
        Student output format: list of dicts with keys like 'pts3d_in_other_view', 'pts3d_local'
        
        Returns formatted data that can be compared with student outputs.
        """
        if not teacher_cache_data or not isinstance(teacher_cache_data, dict):
            return None
        
        # Extract view data from teacher cache
        view_keys = [k for k in teacher_cache_data.keys() if k.startswith('view_')]
        if not view_keys:
            return None
        
        # Sort views to ensure consistent ordering
        view_keys.sort(key=lambda x: int(x.split('_')[1]))
        
        formatted_outputs = []
        
        for i, view_key in enumerate(view_keys):
            view_data = teacher_cache_data[view_key]
            if not isinstance(view_data, dict):
                continue
            
            # Format each view's data to match student output format
            formatted_view = {}
            
            # Global coordinates -> pts3d_in_other_view (this is what Fast3R compares)
            if 'pts3d_global' in view_data:
                formatted_view['pts3d_in_other_view'] = view_data['pts3d_global']
            
            # Local coordinates -> pts3d_local  
            if 'pts3d_local' in view_data:
                formatted_view['pts3d_local'] = view_data['pts3d_local']
            
            # Map local confidence from teacher cache
            if 'conf_local' in view_data:
                formatted_view['conf_local'] = view_data['conf_local']
            
            # Map global confidence from teacher cache
            # Student model outputs 'conf' from global head, teacher stores as 'conf_global'
            if 'conf_global' in view_data:
                formatted_view['conf_global'] = view_data['conf_global']
                formatted_view['conf'] = view_data['conf_global']  # Map to student's naming convention
            
            # VGGT-specific: Add camera pose encoding if present
            if 'pose_enc' in view_data:
                formatted_view['pose_enc'] = view_data['pose_enc']
            
            formatted_outputs.append(formatted_view)
        
        return formatted_outputs
    
    def _format_teacher_outputs(self, teacher_outputs):
        """Convert teacher model outputs to expected format for distillation."""
        if isinstance(teacher_outputs, list):
            # Convert list of view outputs to batched format
            formatted = {}
            for key in ['pts3d_in_other_view', 'pts3d_local', 'conf', 'conf_local']:
                if all(key in out for out in teacher_outputs):
                    formatted[key] = torch.stack([out[key] for out in teacher_outputs], dim=0)
            return formatted
        else:
            return teacher_outputs
    
    def _validate_input_compatibility(self, student_outputs, teacher_targets):
        """Validate that student and teacher outputs have compatible formats."""
        if not teacher_targets:
            return True
            
        # Check key compatibility
        student_keys = set()
        teacher_keys = set(teacher_targets.keys()) if isinstance(teacher_targets, dict) else set()
        
        if isinstance(student_outputs, list) and student_outputs:
            student_keys = set(student_outputs[0].keys())
        elif isinstance(student_outputs, dict):
            student_keys = set(student_outputs.keys())
            
        common_keys = student_keys.intersection(teacher_keys)
        if not common_keys:
            warnings.warn(f"No common keys between student {student_keys} and teacher {teacher_keys}")
            return False
            
        # Check tensor shapes and types
        if isinstance(teacher_targets, dict) and isinstance(student_outputs, list):
            for key in common_keys:
                if key in teacher_targets and key in student_outputs[0]:
                    teacher_tensor = teacher_targets[key]
                    student_tensor = student_outputs[0][key]
                    
                    if not isinstance(teacher_tensor, torch.Tensor):
                        teacher_targets[key] = torch.from_numpy(teacher_tensor).float().to(self.device)
                    elif teacher_tensor.device != self.device:
                        teacher_targets[key] = teacher_tensor.to(self.device)
                        
                    if not isinstance(student_tensor, torch.Tensor):
                        print(f"Warning: Student output {key} is not a tensor: {type(student_tensor)}")
                        
        return True
    
    def _format_outputs_for_loss(self, outputs):
        """Format model outputs for loss computation."""
        if not outputs:
            return {}
        
        # Handle both list of view outputs and single dict of tensors
        if isinstance(outputs, list):
            # List of view outputs - stack across views
            formatted = {}
            
            # Global coordinates - use the aligned global coordinates, fallback to pts3d_in_other_view
            global_key = 'pts3d_local_aligned_to_global' if all('pts3d_local_aligned_to_global' in out for out in outputs) else 'pts3d_in_other_view'
            if all(global_key in out for out in outputs):
                view_tensors = []
                for out in outputs:
                    tensor = out[global_key]
                    # Add batch dimension if missing [H, W, 3] -> [1, H, W, 3]
                    if tensor.dim() == 3:
                        # Check if tensor dimensions suggest landscape vs portrait orientation
                        # Fast3R uses landscape mode, so width > height is expected
                        h, w = tensor.shape[0], tensor.shape[1]
                        if w < h:  # Likely transposed (portrait instead of landscape)
                            tensor = tensor.transpose(0, 1)  # [W, H, 3] -> [H, W, 3]
                        tensor = tensor.unsqueeze(0)
                    view_tensors.append(tensor)
                formatted['xyz_global'] = torch.stack(view_tensors, dim=1)  # [B, N, H, W, 3]
            
            # Local coordinates 
            if all('pts3d_local' in out for out in outputs):
                view_tensors = []
                for out in outputs:
                    tensor = out['pts3d_local']
                    # Add batch dimension if missing [H, W, 3] -> [1, H, W, 3]
                    if tensor.dim() == 3:
                        # Check if tensor dimensions suggest landscape vs portrait orientation
                        h, w = tensor.shape[0], tensor.shape[1]
                        if w < h:  # Likely transposed (portrait instead of landscape)
                            tensor = tensor.transpose(0, 1)  # [W, H, 3] -> [H, W, 3]
                        tensor = tensor.unsqueeze(0)
                    view_tensors.append(tensor)
                formatted['xyz_local'] = torch.stack(view_tensors, dim=1)  # [B, N, H, W, 3]
            
            # Global confidence - map 'conf' to both 'conf' and 'conf_global' 
            if all('conf' in out for out in outputs):
                view_tensors = []
                for out in outputs:
                    tensor = out['conf']
                    # Add batch dimension if missing [H, W] -> [1, H, W]
                    if tensor.dim() == 2:
                        # Check if tensor dimensions suggest landscape vs portrait orientation
                        h, w = tensor.shape[0], tensor.shape[1]
                        if w < h:  # Likely transposed (portrait instead of landscape)
                            tensor = tensor.transpose(0, 1)  # [W, H] -> [H, W]
                        tensor = tensor.unsqueeze(0)
                    view_tensors.append(tensor)
                global_conf = torch.stack(view_tensors, dim=1)  # [B, N, H, W]
                formatted['conf'] = global_conf  # For backward compatibility
                formatted['conf_global'] = global_conf  # For loss function consistency
            
            # Local confidence
            if all('conf_local' in out for out in outputs):
                view_tensors = []
                for out in outputs:
                    tensor = out['conf_local']
                    # Add batch dimension if missing [H, W] -> [1, H, W]
                    if tensor.dim() == 2:
                        # Check if tensor needs transposing based on shape
                        if tensor.shape[0] == 512 and tensor.shape[1] == 288:  # Common teacher resolution
                            tensor = tensor.transpose(0, 1)  # [W, H] -> [H, W]
                        tensor = tensor.unsqueeze(0)
                    view_tensors.append(tensor)
                formatted['conf_local'] = torch.stack(view_tensors, dim=1)  # [B, N, H, W]
            
                
            return formatted
        else:
            # Already formatted dict - return as is
            return outputs
    
    # Removed validation_step override - causes error with read-only current_epoch property
    # Reconstruction evaluation is controlled by parent class based on epoch number
    
    def on_train_epoch_end(self):
        """Clean up memory at the end of each epoch to prevent accumulation."""
        # Force aggressive garbage collection
        import gc
        import psutil
        
        # Log system state before cleanup
        if hasattr(self, '_step_count'):
            system_memory = psutil.virtual_memory()
            print(f"\n[Epoch {self.current_epoch} End] Pre-cleanup RAM: {system_memory.used/1024**3:.1f}GB ({system_memory.percent:.1f}%)")
        
        # CRITICAL: Force DataLoader worker memory reset every 5 epochs
        if (self.current_epoch + 1) % 5 == 0:
            print(f"[Epoch {self.current_epoch}] Forcing DataLoader worker reset to prevent memory accumulation")
            # Note: Lightning will automatically recreate dataloaders when needed
            # We just need to clear any cached references
        
        # Multiple rounds of garbage collection for thorough cleanup
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force Python to release memory back to OS
        gc.collect(2)  # Full collection including oldest generation
        
        # Log system state after cleanup
        if hasattr(self, '_step_count'):
            system_memory = psutil.virtual_memory()
            print(f"[Epoch {self.current_epoch} End] Post-cleanup RAM: {system_memory.used/1024**3:.1f}GB ({system_memory.percent:.1f}%)\n")
            
        # Log memory status
        if self.log_memory_usage:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            print(f"Epoch end - GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            self.log("memory/epoch_end_allocated_gb", memory_allocated, on_epoch=True)
            self.log("memory/epoch_end_reserved_gb", memory_reserved, on_epoch=True)
    
    @classmethod 
    def load_for_inference(cls, checkpoint_path: str):
        """Load model from checkpoint for inference."""
        model = cls.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model

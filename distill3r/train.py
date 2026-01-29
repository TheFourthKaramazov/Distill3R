#!/usr/bin/env python3
"""
Prototype Training Script with Memory and Time Tracking

Comprehensive training script for Distill3r prototype with:
- Memory usage monitoring and alerts
- Time tracking with TQDM progress bars
- Checkpoint saving and loading
- Performance metrics for scaling estimates
"""

import os
import sys
import time
import psutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
import yaml
from tqdm import tqdm
import numpy as np

# ── Add the project root (Distill3r/) ────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]    
sys.path.insert(0, str(ROOT))                  
sys.path.insert(0, str(ROOT / "external" / "fast3r"))

from distill3r.student.distillation_module import DistillationLitModule
from distill3r.data.cached_sample_dataset import CachedSampleDataset, create_cached_samples_collate_fn
from torch.utils.data import DataLoader


class MemoryTracker:
    """Track GPU and system memory usage throughout training."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.start_time = time.time()
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write("timestamp,elapsed_sec,gpu_memory_gb,gpu_reserved_gb,system_memory_gb,step\n")
    
    def log_memory(self, step: int = 0):
        """Log current memory usage."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # GPU memory - get actual usage from nvidia-ml-py
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = info.used / 1024**3  # Convert to GB
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # Keep PyTorch reserved
            except ImportError:
                # Fallback to PyTorch if pynvml not available
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_memory = gpu_reserved = 0.0
        
        # System memory
        system_memory = psutil.virtual_memory().used / 1024**3
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{current_time},{elapsed:.1f},{gpu_memory:.2f},{gpu_reserved:.2f},{system_memory:.2f},{step}\n")
        
        return {
            'gpu_memory_gb': gpu_memory,
            'gpu_reserved_gb': gpu_reserved, 
            'system_memory_gb': system_memory,
            'elapsed_sec': elapsed
        }


class ConfidenceMonitorCallback(Callback):
    """Monitor confidence values during training to detect learning issues."""
    
    def __init__(self, log_interval: int = 100, enabled: bool = True):
        self.log_interval = log_interval
        self.enabled = enabled
        self.step_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Analyze confidence values after each training step."""
        if not self.enabled:
            return

        # Only run on rank 0 in DDP mode
        if trainer.global_rank != 0:
            return

        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            # Get a sample prediction from the student
            with torch.no_grad():
                # Use the first sample from the batch
                if isinstance(batch, dict) and 'images' in batch:
                    images = batch['images'][:1]  # Just first sample
                    B, N, C, H, W = images.shape
                    
                    # Convert to Fast3R format
                    views = []
                    for v in range(min(N, 3)):  # Just check first 3 views
                        view_dict = {
                            'img': images[0, v].unsqueeze(0),
                            'instance': f'view_{v}',
                            'label': 'debug',
                            'true_shape': torch.tensor([[H, W]]).to(images.device)
                        }
                        views.append(view_dict)
                    
                    # Get student predictions
                    student_outputs = pl_module.student(views)
                    
                    # Analyze both local and global confidence
                    conf_local_stats = []
                    conf_global_stats = []
                    for i, output in enumerate(student_outputs):
                        # Check local confidence
                        if 'conf_local' in output:
                            conf_local = output['conf_local']
                            local_stats = {
                                'min': conf_local.min().item(),
                                'max': conf_local.max().item(),
                                'mean': conf_local.mean().item(),
                                'std': conf_local.std().item()
                            }
                            conf_local_stats.append(local_stats)
                        
                        # Check global confidence (output as 'conf')
                        if 'conf' in output:
                            conf_global = output['conf']
                            global_stats = {
                                'min': conf_global.min().item(),
                                'max': conf_global.max().item(),
                                'mean': conf_global.mean().item(),
                                'std': conf_global.std().item()
                            }
                            conf_global_stats.append(global_stats)
                    
                    if conf_local_stats or conf_global_stats:
                        # Print summary
                        print(f"\n[Step {self.step_count}] Confidence Monitor:")
                        
                        if conf_global_stats:
                            print(f"  Student GLOBAL confidence (first {len(conf_global_stats)} views):")
                            for i, stats in enumerate(conf_global_stats):
                                print(f"    View {i}: [{stats['min']:.3f}, {stats['max']:.3f}] "
                                      f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                        
                        if conf_local_stats:
                            print(f"  Student LOCAL confidence (first {len(conf_local_stats)} views):")
                            for i, stats in enumerate(conf_local_stats):
                                print(f"    View {i}: [{stats['min']:.3f}, {stats['max']:.3f}] "
                                      f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                        
                        # Check if learning is happening
                        all_stats = conf_local_stats + conf_global_stats
                        if all_stats:
                            avg_std = np.mean([s['std'] for s in all_stats])
                            if avg_std < 0.01:
                                print("  WARNING: Very low variance in confidence predictions!")
                        
                        # Also check teacher values if available
                        if 'teacher_data' in batch:
                            teacher_data = batch['teacher_data'][0]  # First sample
                            if isinstance(teacher_data, dict) and 'view_0' in teacher_data:
                                # Check teacher global confidence
                                print(f"  Teacher GLOBAL confidence (first 3 views):")
                                for v in range(min(3, N)):
                                    view_key = f'view_{v}'
                                    if view_key in teacher_data and 'conf_global' in teacher_data[view_key]:
                                        t_conf_g = teacher_data[view_key]['conf_global']
                                        print(f"    View {v}: [{t_conf_g.min():.3f}, {t_conf_g.max():.3f}] "
                                              f"mean={t_conf_g.mean():.3f}, std={t_conf_g.std():.3f}")
                                
                                # Check teacher local confidence
                                print(f"  Teacher LOCAL confidence (first 3 views):")
                                for v in range(min(3, N)):
                                    view_key = f'view_{v}'
                                    if view_key in teacher_data and 'conf_local' in teacher_data[view_key]:
                                        t_conf_l = teacher_data[view_key]['conf_local']
                                        print(f"    View {v}: [{t_conf_l.min():.3f}, {t_conf_l.max():.3f}] "
                                              f"mean={t_conf_l.mean():.3f}, std={t_conf_l.std():.3f}")


class MemoryCallback(pl.Callback):
    """Lightning callback for memory tracking."""
    
    def __init__(self, tracker: MemoryTracker, alert_threshold_gb: float = 20.0):
        self.tracker = tracker
        self.alert_threshold = alert_threshold_gb
        self.step_count = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        
        # Log memory every 100 steps (roughly every 2-3 minutes)
        if self.step_count % 100 == 0:
            memory_info = self.tracker.log_memory(self.step_count)
            
            # Alert if approaching memory limit (no duplicate print - main monitoring in distillation_module.py)
            if memory_info['gpu_memory_gb'] > self.alert_threshold:
                print(f"\n  HIGH MEMORY USAGE: {memory_info['gpu_memory_gb']:.1f}GB GPU memory")


def create_enhanced_datamodule(config: Dict[str, Any]) -> pl.LightningDataModule:
    """Create simplified data module with cached samples."""
    from distill3r.data.cached_sample_dataset import CachedSampleDataset, create_cached_samples_collate_fn
    
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    class SimpleCachedDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.cache_dir = data_config.get('cache_dir', 'teacher_cache_tiny')
            self.batch_size = training_config.get('batch_size', 1)
            self.num_workers = data_config.get('num_workers', 16)
            # Get max_views from model config
            self.num_views = config.get('model', {}).get('max_views', 6)
            
        def setup(self, stage: Optional[str] = None):
            if stage == 'fit' or stage is None:
                self.train_dataset = CachedSampleDataset(
                    teacher_cache_dir=self.cache_dir,
                    image_size=512,
                    num_views=self.num_views,  # Use config value
                    max_samples=None,  # Use all available samples
                    encoder_type=config.get('model', {}).get('encoder_type', 'dune')
                )
                
        def train_dataloader(self):
            # Read DataLoader config
            pin_memory = data_config.get('pin_memory', False)
            persistent_workers = data_config.get('persistent_workers', False) and self.num_workers > 0
            prefetch_factor = data_config.get('prefetch_factor', 2) if self.num_workers > 0 else None
            
            print(f"Creating DataLoader with num_workers={self.num_workers}, batch_size={self.batch_size}, "
                  f"pin_memory={pin_memory}, persistent_workers={persistent_workers}")
            
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                collate_fn=create_cached_samples_collate_fn(),
                drop_last=True
            )
            print(f"DataLoader created successfully with {len(self.train_dataset)} samples")
            return dataloader
    
    return SimpleCachedDataModule()


def create_enhanced_callbacks(config: Dict[str, Any], memory_tracker: MemoryTracker):
    """Create callbacks with enhanced monitoring."""
    callbacks = []
    
    # Memory tracking
    memory_callback = MemoryCallback(
        memory_tracker, 
        alert_threshold_gb=config.get('memory', {}).get('max_memory_gb', 20.0)
    )
    callbacks.append(memory_callback)
    
    # Confidence monitoring - enabled by default
    confidence_monitor = ConfidenceMonitorCallback(
        log_interval=500,  # Every 500 steps (reduces verbosity for full training)
        enabled=config.get('monitoring', {}).get('monitor_confidence', True)
    )
    callbacks.append(confidence_monitor)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get('output_dir', 'checkpoints/distill3r'),
        filename='distill3r-{epoch:02d}-{step:06d}-{train/total_epoch:.3f}',
        monitor='train/total_epoch',
        mode='min',
        save_top_k=config.get('checkpointing', {}).get('save_top_k', 5),
        save_last=True,
        every_n_epochs=config.get('checkpointing', {}).get('save_every_n_epochs', 10),
        save_on_train_epoch_end=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping (optional for prototype)
    if config['training'].get('early_stop_patience'):
        early_stop = EarlyStopping(
            monitor='train/total_epoch',
            patience=config['training']['early_stop_patience'],
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop)
    
    return callbacks




def main():
    """Main training function with comprehensive monitoring."""
    parser = argparse.ArgumentParser(description='Distill3r training with monitoring')
    parser.add_argument('--config', type=str, default='configs/distill3r.yaml',
                       help='Training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--no-confidence-monitor', action='store_true',
                       help='Disable confidence monitoring during training')
    parser.add_argument('--dry-run', action='store_true',
                       help='Setup only, no training')
    
    args = parser.parse_args()
    
    log_dir = Path('logs/distill3r')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("DISTILL3R TRAINING")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output: {config.get('output_dir', 'checkpoints/distill3r')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print()
    
    # Set random seed
    pl.seed_everything(config.get('seed', 42))
    
    # Override confidence monitor setting from command line
    if 'monitoring' not in config:
        config['monitoring'] = {}
    config['monitoring']['monitor_confidence'] = not args.no_confidence_monitor
    
    # Setup memory tracking
    log_dir = Path(config.get('logging', {}).get('log_dir', 'logs/distill3r'))
    memory_tracker = MemoryTracker(log_dir / 'memory_usage.csv')
    
    # Create data module
    print("Creating data module...")
    try:
        data_module = create_enhanced_datamodule(config)
        data_module.setup('fit')
        
        # Get data size for estimates
        train_loader = data_module.train_dataloader()
        data_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 100
        print(f"Training data size: {data_size} samples")

    except Exception as e:
        print(f"Error creating data module: {e}")
        print("Make sure manifest and cache are generated first!")
        return 1

    # Create model
    print("Creating student model...")
    model_config = config.get('model', {})
    loss_config = config.get('loss', {})
    training_config = config.get('training', {})

    # Calculate total training steps for LR scheduler (fix cosine annealing bug)
    if 'batch_size' not in training_config:
        raise ValueError("training.batch_size must be specified in config")
    if 'accumulate_grad_batches' not in training_config:
        raise ValueError("training.accumulate_grad_batches must be specified in config")
    if 'max_epochs' not in training_config:
        raise ValueError("training.max_epochs must be specified in config")

    batch_size = training_config['batch_size']
    accumulate_grad_batches = training_config['accumulate_grad_batches']
    max_epochs = training_config['max_epochs']
    steps_per_epoch = data_size / (batch_size * accumulate_grad_batches)
    calculated_max_steps = int(steps_per_epoch * max_epochs)
    print(f"Calculated max_steps for LR scheduler: {calculated_max_steps:,} "
          f"({steps_per_epoch:.0f} steps/epoch × {max_epochs} epochs)")
    
    lit_module = DistillationLitModule(
        # All parameters from config - NO DEFAULTS
        alpha_g=loss_config['alpha_g'],
        alpha_l=loss_config['alpha_l'],
        gamma=loss_config['gamma'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        max_steps=calculated_max_steps,
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        embed_dim=model_config['embed_dim'],
        encoder_depth=model_config['encoder_depth'],
        encoder_heads=model_config['encoder_heads'],
        decoder_depth=model_config['decoder_depth'],
        decoder_heads=model_config['decoder_heads'],
        encoder_type=model_config['encoder_type'],
        max_views=model_config['max_views'],
        max_parallel_views_for_head=model_config['max_parallel_views_for_head'],
        landscape_only=model_config['landscape_only'],
        teacher_cache_dir=config['data']['cache_dir'],
        memory_cleanup_freq=config['memory']['memory_cleanup_freq'],
        log_memory_usage=config['memory']['log_memory_usage'],
        gpu_memory_threshold=config['memory']['gpu_memory_threshold'],
        # Parameters with defaults
        loss_type=loss_config.get('loss_type', 'default'),  # For ablation studies
        min_lr=training_config['min_lr'],
    )

    # Print model statistics
    model_stats = lit_module.student.get_model_stats()
    print(f"Model created:")
    print(f"  Total parameters: {model_stats['total_parameters']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_parameters']:,}")
    print(f"  Trainable ratio: {model_stats['trainable_ratio']:.1%}")
    print(f"  Dataset size: {data_size} samples")
    
    if args.dry_run:
        print("\nDry run complete - setup successful!")
        return 0
    
    # Create logger
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name=config.get('logging', {}).get('experiment_name', 'distill3r')
    )
    
    # Create trainer with monitoring
    num_gpus = training_config.get('num_gpus', 1)
    gpu_id = training_config.get('gpu_id', 1)  # Default to GPU 1 (second GPU)

    # Multi-GPU fix: disable NCCL P2P to avoid deadlocks
    if num_gpus > 1:
        os.environ['NCCL_P2P_DISABLE'] = '1'

    strategy = 'ddp_find_unused_parameters_true' if num_gpus > 1 else 'auto'

    # Set devices based on configuration
    if num_gpus == 1:
        # Single GPU - use specific GPU ID
        devices = [gpu_id]  # List with single GPU ID
        print(f"Training configuration:")
        print(f"  Using GPU: {gpu_id}")
        print(f"  Strategy: {strategy}")
    else:
        # Multi-GPU - use specified number
        devices = num_gpus
        print(f"Training configuration:")
        print(f"  GPUs: {num_gpus}")
        print(f"  Strategy: {strategy}")
    
    trainer = pl.Trainer(
        max_epochs=training_config.get('max_epochs', 50),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        strategy=strategy,
        precision=training_config.get('precision', 32),  # Read precision from config
        gradient_clip_val=training_config.get('max_grad_norm', 1.0),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 4),
        val_check_interval=training_config.get('val_check_interval', 0.25),
        log_every_n_steps=training_config.get('log_every_n_steps', 5),
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=create_enhanced_callbacks(config, memory_tracker),
        logger=csv_logger,
        deterministic=False,
        benchmark=True,
    )
    
    # Start training with time tracking
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    start_time = time.time()
    memory_tracker.log_memory(0)  # Initial memory state
    
    try:
        # Print resume information if resuming
        if args.resume:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            print(f"RESUMING from checkpoint:")
            print(f"  Checkpoint: {args.resume}")
            print(f"  Epoch: {ckpt.get('epoch', 'Unknown')}")
            print(f"  Global step: {ckpt.get('global_step', 'Unknown')}")
            print(f"  Will continue from epoch {ckpt.get('epoch', 0) + 1}")
            print()
        
        trainer.fit(
            model=lit_module,
            datamodule=data_module,
            ckpt_path=args.resume
        )
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total training time: {training_duration:.1f}s ({training_duration/60:.1f}m)")
        print(f"Average time per epoch: {training_duration/trainer.current_epoch:.1f}s")
        
        # Final memory log
        final_memory = memory_tracker.log_memory(-1)
        print(f"Final GPU memory: {final_memory['gpu_memory_gb']:.1f}GB")
        
        # Save final model
        final_path = Path(config.get('output_dir', 'checkpoints/distill3r')) / 'final_model.ckpt'
        trainer.save_checkpoint(final_path)
        print(f"Final model saved: {final_path}")
        
        print(f"Memory log saved: {memory_tracker.log_file}")
        print(f"Training logs: {csv_logger.log_dir}")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
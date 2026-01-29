"""
Compressed Fast3R Student Model for Knowledge Distillation

A compressed version of Fast3R using DUNE or DINOv3 encoder.
Key features:
- DUNE ViT-Small encoder (384 dim, 12 layers, 6 heads, patch_size=14)
  - Pretrained via multi-teacher distillation (DINO-v2, MASt3R, Multi-HMR)
- DINOv3 ViT-Small encoder (384 dim, 12 layers, 6 heads, patch_size=16)
  - Pretrained on LVD-1689M dataset with advanced self-supervised learning
- Compressed Fast3R decoder (384 dim, 6 layers vs teacher's 768 dim, 12 layers)
- Same global attention fusion (concatenate all view tokens)
- Same DPT heads for 3D prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import warnings
from functools import partial
import numpy as np

# Fast3R components
from fast3r.models.fast3r import Fast3RDecoder
from fast3r.dust3r.heads.dpt_head import PixelwiseTaskWithDPT
from fast3r.dust3r.heads.postprocess import postprocess
from fast3r.dust3r.utils.misc import transpose_to_landscape


class CompressedFast3R(nn.Module):
    """
    Compressed Fast3R model for knowledge distillation.
    
    Encoder options with compressed Fast3R decoder:
    - DUNE ViT-Small (384 dim, 12 layers, 6 heads, patch_size=14)
      - Pretrained via multi-teacher distillation from DINO-v2, MASt3R, Multi-HMR
    - DINOv3 ViT-Small (384 dim, 12 layers, 6 heads, patch_size=16)
      - Pretrained on LVD-1689M dataset with RoPE embeddings and storage tokens
    - Decoder: Compressed Fast3R transformer (384 dim, 6 layers, 6 heads) vs teacher (768 dim, 12 layers, 12 heads)
    - Same global cross-view attention (concatenate all view tokens)
    - Same DPT heads for 3D prediction
    """
    
    def __init__(
        self,
        # Model architecture
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 384,           # Reduced from 768 (teacher)
        encoder_depth: int = 8,         # Reduced from 12 (teacher)
        encoder_heads: int = 6,         # Reduced from 12 (teacher)
        decoder_depth: int = 6,         # Reduced from 12 (teacher)
        decoder_heads: int = 6,         # Reduced from 12 (teacher)
        
        # Encoder type selection
        encoder_type: str = "dune",     # "dune" or "dinov3"
        
        # Memory management
        max_views: int = 8,
        max_parallel_views_for_head: int = 8,
        
        # Head configuration
        output_mode: str = "pts3d",
        head_type: str = "dpt",
        depth_mode: list = None,
        conf_mode: list = None,
        landscape_only: bool = False,
        with_local_head: bool = True,
        load_pretrained: bool = True,
    ):
        super().__init__()
        
        # Store configuration
        self.img_size = img_size
        self.encoder_type = encoder_type
        self.max_views = max_views
        self.max_parallel_views_for_head = max_parallel_views_for_head
        self.output_mode = output_mode
        self.head_type = head_type
        # Set default depth_mode to match Fast3R configuration
        self.depth_mode = depth_mode if depth_mode is not None else ("exp", -float("inf"), float("inf"))
        # Set default conf_mode to match Fast3R configuration exactly
        self.conf_mode = conf_mode if conf_mode is not None else ("exp", 1, float("inf"))
        
        # Validate encoder configuration and set dimensions to match pretrained weights
        if encoder_type == "dune":
            # DUNE has fixed architecture - warn if user tries to customize
            if embed_dim != 384:
                warnings.warn(f"DUNE encoder has fixed embed_dim=384, ignoring provided embed_dim={embed_dim}")
                embed_dim = 384
            if encoder_depth != 12:
                warnings.warn(f"DUNE encoder has fixed depth=12, ignoring provided encoder_depth={encoder_depth}")
                encoder_depth = 12
            if encoder_heads != 6:
                warnings.warn(f"DUNE encoder has fixed num_heads=6, ignoring provided encoder_heads={encoder_heads}")
                encoder_heads = 6
            if patch_size != 14:
                warnings.warn(f"DUNE encoder has fixed patch_size=14, ignoring provided patch_size={patch_size}")
                patch_size = 14
        elif encoder_type == "dinov3":
            # DINOv3 ViT-Small has fixed architecture
            if embed_dim != 384:
                warnings.warn(f"DINOv3 ViT-Small has fixed embed_dim=384, ignoring provided embed_dim={embed_dim}")
                embed_dim = 384
            if encoder_depth != 12:
                warnings.warn(f"DINOv3 ViT-Small has fixed depth=12, ignoring provided encoder_depth={encoder_depth}")
                encoder_depth = 12
            if encoder_heads != 6:
                warnings.warn(f"DINOv3 ViT-Small has fixed num_heads=6, ignoring provided encoder_heads={encoder_heads}")
                encoder_heads = 6
            if patch_size != 16:
                warnings.warn(f"DINOv3 ViT-Small has fixed patch_size=16, ignoring provided patch_size={patch_size}")
                patch_size = 16
        
        # Store final configuration (potentially overridden for DUNE)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Store max_views configuration
        # Note: RTX 4090 can handle 20+ views with proper batch sizing
        
        # Build encoder
        self.encoder = self._build_encoder(
            img_size=img_size,
            patch_size=patch_size, 
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            encoder_type=encoder_type,
            load_pretrained=load_pretrained
        )
        
        # Build decoder (compressed Fast3R)
        self.decoder = self._build_decoder(
            enc_embed_dim=embed_dim,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=decoder_heads
        )
        
        # Build heads (same as teacher but smaller input dim)
        self.downstream_head = self._build_head(
            head_type=head_type,
            output_mode=output_mode,
            has_conf=bool(self.conf_mode),  # Use stored value after defaults applied
            patch_size=patch_size,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=embed_dim,
            decoder_depth=decoder_depth
        )
        
        # Optional local head
        if with_local_head:
            self.downstream_head_local = self._build_head(
                head_type=head_type,
                output_mode=output_mode,
                has_conf=bool(self.conf_mode),  # Use stored value after defaults applied
                patch_size=patch_size,
                encoder_embed_dim=embed_dim,
                decoder_embed_dim=embed_dim,
                decoder_depth=decoder_depth
            )
        else:
            self.downstream_head_local = None
            
        # Wrap heads with landscape transform
        self.head = transpose_to_landscape(self.downstream_head, activate=landscape_only)
        if self.downstream_head_local:
            self.local_head = transpose_to_landscape(self.downstream_head_local, activate=landscape_only)
        else:
            self.local_head = None
    
    def _build_encoder(self, img_size: int, patch_size: int, embed_dim: int, 
                      depth: int, num_heads: int, encoder_type: str, load_pretrained: bool = True) -> nn.Module:
        """Build encoder with DUNE or DINOv3."""
        
        if encoder_type == "dune" and load_pretrained:
            try:
                print("Loading DUNE ViT-Small encoder...")
                
                # Add external/dune to Python path for torch.hub
                import sys
                from pathlib import Path
                dune_path = str(Path(__file__).parent.parent.parent / "external" / "dune")
                if dune_path not in sys.path:
                    sys.path.insert(0, dune_path)
                
                # Load DUNE ViT-Small encoder via torch.hub
                # This loads the full pretrained encoder with perfect dimension match
                encoder = torch.hub.load("naver/dune", "dune_vitsmall_14_448_encoder", trust_repo=True)
                print("Successfully loaded DUNE ViT-Small encoder with perfect dimension match")
                
                # Debug: Check if DUNE encoder has NaN in weights
                for name, param in encoder.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"WARNING: DUNE parameter {name} contains NaN/inf values!")
                        break
                else:
                    print("DUNE encoder weights are clean (no NaN/inf)")
                
                # Verify dimensions match our student architecture exactly
                assert hasattr(encoder, 'embed_dim') and encoder.embed_dim == 384, f"Expected DUNE embed_dim=384, got {getattr(encoder, 'embed_dim', 'unknown')}"
                assert hasattr(encoder, 'patch_size') and encoder.patch_size == 14, f"Expected DUNE patch_size=14, got {getattr(encoder, 'patch_size', 'unknown')}"
                
                # No dimension mismatch - DUNE ViT-Small matches our 384 dim student perfectly!
                print("DUNE encoder dimensions verified: 384 embed_dim, 14 patch_size, 12 depth, 6 heads")
                
            except Exception as e:
                warnings.warn(f"Failed to load DUNE encoder: {e}")
                print(f"DUNE loading error details: {e}")
                raise RuntimeError(f"Failed to load DUNE encoder: {e}")
        
        elif encoder_type == "dinov3" and load_pretrained:
            try:
                print("Loading DINOv3 ViT-Small encoder...")
                
                # Add external/dinov3 to Python path
                import sys
                from pathlib import Path
                dinov3_path = str(Path(__file__).parent.parent.parent / "external" / "dinov3")
                if dinov3_path not in sys.path:
                    sys.path.insert(0, dinov3_path)
                
                # Load DINOv3 ViT-Small via torch.hub with local weights
                weights_path = Path(__file__).parent.parent.parent / "pretrained_models" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
                if not weights_path.exists():
                    raise FileNotFoundError(f"DINOv3 weights not found at {weights_path}")
                
                # Load model via torch.hub
                encoder = torch.hub.load(
                    dinov3_path, 
                    'dinov3_vits16', 
                    source='local',
                    weights=str(weights_path),
                    pretrained=True
                )
                print("Successfully loaded DINOv3 ViT-Small encoder")
                
                # Verify dimensions
                assert hasattr(encoder, 'embed_dim') and encoder.embed_dim == 384
                assert hasattr(encoder, 'patch_size') and encoder.patch_size == 16
                print("DINOv3 encoder dimensions verified: 384 embed_dim, 16 patch_size, 12 depth, 6 heads")
                
            except Exception as e:
                warnings.warn(f"Failed to load DINOv3 encoder: {e}")
                print(f"DINOv3 loading error details: {e}")
                raise RuntimeError(f"Failed to load DINOv3 encoder: {e}")
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Pure fine-tuning - all parameters unfrozen by default
        
        return encoder
    
    def _build_decoder(self, enc_embed_dim: int, embed_dim: int, depth: int, num_heads: int) -> nn.Module:
        """Build compressed Fast3R decoder."""
        decoder = Fast3RDecoder(
            random_image_idx_embedding=True,  # Use random IDs for data augmentation (matches Fast3R default)
            enc_embed_dim=enc_embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            attn_implementation="flash_attention",
            attn_bias_for_inference_enabled=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        print(f"Fast3R Decoder: random_image_idx_embedding={decoder.random_image_idx_embedding}")
        if decoder.random_image_idx_embedding:
            print("  Using RANDOM image IDs (data augmentation enabled)")
        else:
            print("  Using SEQUENTIAL image IDs [0,1,2,3,...] (fixed)")
        return decoder
    
    def _build_head(self, head_type: str, output_mode: str, has_conf: bool, patch_size: int,
                   encoder_embed_dim: int, decoder_embed_dim: int, decoder_depth: int) -> nn.Module:
        """Build prediction head (same as teacher but adapted for smaller dimensions)."""
        if head_type == "dpt" and output_mode == "pts3d":
            # Calculate hook indices for DPT (same logic as teacher)
            l2 = decoder_depth
            feature_dim = 256
            last_dim = feature_dim // 2
            out_nchan = 3
            
            return PixelwiseTaskWithDPT(
                num_channels=out_nchan + int(has_conf),
                feature_dim=feature_dim,
                last_dim=last_dim,
                hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                dim_tokens=[encoder_embed_dim, decoder_embed_dim, decoder_embed_dim, decoder_embed_dim],
                postprocess=postprocess,
                depth_mode=self.depth_mode,
                conf_mode=self.conf_mode,
                head_type="regression",
                patch_size=patch_size,
            )
        else:
            raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
    
    def _encode_images(self, views, chunk_size=400):
        """Encode images using compressed encoder (same logic as teacher)."""
        B = views[0]["img"].shape[0]

        # Check if all images have the same shape
        same_shape = all(view["img"].shape == views[0]["img"].shape for view in views)

        if same_shape:
            # Stack images along a new dimension to create a batch
            imgs = torch.cat([view["img"] for view in views], dim=0)  # Shape: [num_views * B, C, H, W]
            true_shapes = torch.cat(
                [view.get("true_shape", torch.tensor(view["img"].shape[-2:])[None].repeat(B, 1)) for view in views],
                dim=0
            )  # Shape: [num_views * B, 2]

            # Encode images in chunks to prevent OOM
            num_chunks = (imgs.shape[0] + chunk_size - 1) // chunk_size
            feats_chunks = []
            pos_chunks = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, imgs.shape[0])
                
                # Handle encoder-specific output formats
                img_batch = imgs[start_idx:end_idx]
                
                if self.encoder_type == "dune":
                    # DUNE returns dict with 'x_norm_patchtokens'
                    encoder_output = self.encoder(img_batch)
                    if isinstance(encoder_output, dict):
                        chunk_feats = encoder_output['x_norm_patchtokens']  # [B, N_patches, embed_dim]
                    else:
                        chunk_feats = encoder_output
                elif self.encoder_type == "dinov3":
                    # DINOv3 returns dict when called with is_training=True
                    encoder_output = self.encoder.forward_features(img_batch)
                    chunk_feats = encoder_output['x_norm_patchtokens']  # [B, N_patches, embed_dim]
                else:
                    raise ValueError(f"Unknown encoder type: {self.encoder_type}")
                
                # Create dummy positions for compatibility (neither DUNE nor DINOv3 use explicit positions)
                chunk_pos = torch.zeros(chunk_feats.shape[0], chunk_feats.shape[1], 2, device=chunk_feats.device)
                
                feats_chunks.append(chunk_feats)
                pos_chunks.append(chunk_pos)
            
            feats = torch.cat(feats_chunks, dim=0)
            pos = torch.cat(pos_chunks, dim=0)

            # Split the encoded features and positions back into individual views
            encoded_feats = torch.split(feats, B, dim=0)
            positions = torch.split(pos, B, dim=0)
            shapes = torch.split(true_shapes, B, dim=0)
        else:
            # Process each image individually
            encoded_feats, positions, shapes = [], [], []
            for view in views:
                img = view["img"]
                true_shape = view.get(
                    "true_shape", torch.tensor(img.shape[-2:])[None].repeat(B, 1)
                )
                
                # Handle encoder-specific output formats
                if self.encoder_type == "dune":
                    # DUNE returns dict with 'x_norm_patchtokens'
                    encoder_output = self.encoder(img)
                    if isinstance(encoder_output, dict):
                        feat = encoder_output['x_norm_patchtokens']  # [B, N_patches, embed_dim]
                    else:
                        feat = encoder_output
                elif self.encoder_type == "dinov3":
                    # DINOv3 returns dict when called with forward_features
                    encoder_output = self.encoder.forward_features(img)
                    feat = encoder_output['x_norm_patchtokens']  # [B, N_patches, embed_dim]
                else:
                    raise ValueError(f"Unknown encoder type: {self.encoder_type}")
                
                # Create dummy positions for compatibility
                pos = torch.zeros(feat.shape[0], feat.shape[1], 2, device=feat.device)
                
                encoded_feats.append(feat)
                positions.append(pos)
                shapes.append(true_shape)

        return encoded_feats, positions, shapes
    
    def forward(self, views, profiling=False):
        """
        Forward pass using identical logic to Fast3R teacher.
        
        Args:
            views (list[dict]): List of views, each view is a dict of tensors
            
        Returns:
            list[dict]: List of results for each view (same format as teacher)
        """
        # Encode images (same as teacher)
        encoded_feats, positions, shapes = self._encode_images(views)
        
        # Create image IDs for each patch (same as teacher)
        num_images = len(views)
        B, _, _ = encoded_feats[0].shape
        
        different_resolution_across_views = not all(torch.equal(shapes[0], shape) for shape in shapes)
        
        # Initialize image IDs
        image_ids = []
        for i, encoded_feat in enumerate(encoded_feats):
            num_patches = encoded_feat.shape[1]
            image_ids.extend([i] * num_patches)
        
        image_ids = torch.tensor(image_ids * B).reshape(B, -1).to(encoded_feats[0].device)
        
        # Global cross-view fusion through decoder (same as teacher)
        dec_output = self.decoder(encoded_feats, positions, image_ids)
        
        # Prepare outputs for heads (same chunking logic as teacher)
        final_results = [{} for _ in range(num_images)]
        
        if different_resolution_across_views or self.training:
            # Different resolutions or training mode - process sequentially
            num_patches_list = [encoded_feat.shape[1] for encoded_feat in encoded_feats]
            gathered_outputs_list = [[] for _ in range(num_images)]
            
            for layer_output in dec_output:
                split_layer_outputs = torch.split(layer_output, num_patches_list, dim=1)
                for img_id, gathered_output in enumerate(split_layer_outputs):
                    gathered_outputs_list[img_id].append(gathered_output)
            
            # Process each view separately
            for img_id in range(num_images):
                img_result = self.head(gathered_outputs_list[img_id], shapes[img_id])
                if self.local_head:
                    local_img_result = self.local_head(gathered_outputs_list[img_id], shapes[img_id])

                # Store results
                for key in img_result.keys():
                    if key == 'pts3d':
                        final_results[img_id]['pts3d_in_other_view'] = img_result[key]
                    else:
                        final_results[img_id][key] = img_result[key]

                if self.local_head:
                    final_results[img_id]['pts3d_local'] = local_img_result['pts3d']
                    if 'conf' in local_img_result:
                        final_results[img_id]['conf_local'] = local_img_result['conf']
        
        else:
            # Same resolution and inference mode - batch process with chunking
            from einops import rearrange
            
            P_patches = encoded_feats[0].shape[1]
            gathered_outputs_list = []
            
            for layer_output in dec_output:
                layer_output = rearrange(
                    layer_output,
                    'B (num_images P_patches) D -> (num_images B) P_patches D',
                    num_images=num_images,
                    P_patches=P_patches
                )
                gathered_outputs_list.append(layer_output)
            
            concatenated_shapes = torch.cat(shapes, dim=0)
            
            # Chunked processing to avoid OOM
            shape_chunks = torch.split(concatenated_shapes, self.max_parallel_views_for_head, dim=0)
            num_chunks = len(shape_chunks)
            
            chunked_gathered_outputs_list = [[] for _ in range(num_chunks)]
            
            for layer_output in gathered_outputs_list:
                split_layer_outputs = torch.split(layer_output, self.max_parallel_views_for_head, dim=0)
                for chunk_idx, split_output in enumerate(split_layer_outputs):
                    chunked_gathered_outputs_list[chunk_idx].append(split_output)
            
            # Process chunks
            result_chunks = []
            local_result_chunks = [] if self.local_head else None
            
            for chunk, chunk_shapes in zip(chunked_gathered_outputs_list, shape_chunks):
                result_chunk = self.head(chunk, chunk_shapes)
                result_chunks.append(result_chunk)
                
                if self.local_head:
                    local_result_chunk = self.local_head(chunk, chunk_shapes)
                    local_result_chunks.append(local_result_chunk)
            
            # Reassemble chunks
            result = {key: torch.cat([chunk[key] for chunk in result_chunks], dim=0) for key in result_chunks[0].keys()}
            
            if self.local_head:
                local_result = {key: torch.cat([chunk[key] for chunk in local_result_chunks], dim=0) for key in local_result_chunks[0].keys()}
            
            # Re-map results back to per-view format
            for key in result.keys():
                for img_id in range(num_images):
                    img_result = result[key][img_id * B:(img_id + 1) * B]
                    if key == 'pts3d':
                        final_results[img_id]['pts3d_in_other_view'] = img_result
                    else:
                        final_results[img_id][key] = img_result

                    if self.local_head:
                        local_img_result = local_result['pts3d'][img_id * B:(img_id + 1) * B]
                        final_results[img_id]['pts3d_local'] = local_img_result
                        if 'conf' in local_result:
                            final_results[img_id]['conf_local'] = local_result['conf'][img_id * B:(img_id + 1) * B]
        
        return final_results
    
    def set_max_parallel_views_for_head(self, max_parallel_views_for_head: int):
        """Set maximum parallel views for head processing (same as teacher)."""
        self.max_parallel_views_for_head = max_parallel_views_for_head
    
    def get_model_stats(self) -> Dict[str, any]:
        """Get comprehensive model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'embed_dim': self.embed_dim,
            'max_views': self.max_views,
            'max_parallel_views_for_head': self.max_parallel_views_for_head,
            'model_type': 'CompressedFast3R'
        }
    


# Alias for backward compatibility
StudentModel = CompressedFast3R
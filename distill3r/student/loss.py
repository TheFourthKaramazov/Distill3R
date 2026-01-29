"""
Memory-Efficient Knowledge Distillation Loss for Distill-3R

Optimized for single GPU training on RTX 4090 with minimal memory overhead.
Avoids large tensor flattening and uses chunked processing where needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np


class DistillLoss(nn.Module):
    """
    Knowledge distillation loss with Fast3R-style cross-view normalization.

    CRITICAL: Uses cross-view normalization for global loss (all views share scale)
    and per-view normalization for local loss, exactly matching Fast3R training.
    """

    def __init__(
        self,
        alpha_g: float = 1.0,
        alpha_l: float = 0.5,
        gamma: float = 0.1,
        norm_mode: str = "avg_dis"  # Match Fast3R default
    ):
        super().__init__()
        self.alpha_g = alpha_g
        self.alpha_l = alpha_l
        self.gamma = gamma
        self.norm_mode = norm_mode

    def normalize_pointcloud_from_views(self, pts_list, norm_mode="avg_dis", valid_list=None):
        """
        Normalize point clouds from multiple views - EXACT Fast3R implementation.

        All views are concatenated and normalized by a SINGLE scale factor,
        forcing the model to learn consistent cross-view alignment.
        """
        assert all(pts.ndim >= 3 and pts.shape[-1] == 3 for pts in pts_list)

        norm_mode, dis_mode = norm_mode.split("_")
        # Concatenate all point clouds and valid masks if provided
        all_pts = torch.cat(pts_list, dim=1)
        all_pts = all_pts.view(all_pts.shape[0], -1, 3)
        if valid_list is not None:
            all_valid = torch.cat(valid_list, dim=1)
            all_valid = all_valid.view(all_valid.shape[0], -1)
            all_pts[all_valid == 0] = float('nan')  # mask out invalid points with nan

        valid_pts = all_pts

        # Compute the distance to the origin for valid points
        dis = valid_pts.norm(dim=-1)

        # Apply distance transformation based on dis_mode
        if dis_mode == "dis":
            pass  # Do nothing
        elif dis_mode == "log1p":
            dis = torch.log1p(dis)
        elif dis_mode == "warp-log1p":
            log_dis = torch.log1p(dis)
            warp_factor = log_dis / dis.clip(min=1e-8)
            all_pts = all_pts * warp_factor.view(-1, 1)  # Warp the points with the warp factor
            dis = log_dis  # The final distance is now the log-transformed distance
        else:
            raise ValueError(f"Unsupported distance mode: {dis_mode}")

        # Apply different normalization modes
        if norm_mode == "avg":
            norm_factor = dis.nanmean(dim=-1)   # Compute mean distance of valid points
        elif norm_mode == "median":
            norm_factor = dis.nanmedian(dim=-1)  # Compute median distance of valid points
        else:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")

        norm_factor = norm_factor.clip(min=1e-8)  # Prevent division by zero

        # Normalize all point clouds by the SAME factor
        normalized_pts = [pts / norm_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                          for pts in pts_list]

        return normalized_pts, norm_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def normalize_pointcloud_per_view(self, pts_list, norm_mode="avg_dis", valid_list=None):
        """
        Normalize point clouds on a per-view basis - EXACT Fast3R implementation.

        Each view gets its own scale factor (for local loss).
        """
        norm_mode, dis_mode = norm_mode.split("_")

        normed_pts_list = []
        for pts, valid in zip(pts_list, valid_list):
            valid_pts = pts.clone()
            valid_pts[valid == 0] = float('nan')

            # Compute distance
            dis = valid_pts.norm(dim=-1)

            # Apply distance transformation
            if dis_mode == "dis":
                pass
            elif dis_mode == "log1p":
                dis = torch.log1p(dis)
            elif dis_mode == "warp-log1p":
                log_dis = torch.log1p(dis)
                warp_factor = log_dis / dis.clip(min=1e-8)
                pts = pts * warp_factor.unsqueeze(-1)
                dis = log_dis
            else:
                raise ValueError(f"Unsupported distance mode: {dis_mode}")

            # Apply normalization per view
            if norm_mode == "avg":
                norm_factor = dis.nanmean(dim=(-1, -2), keepdim=True)
            elif norm_mode == "median":
                norm_factor = dis.nanmedian(dim=(-1, -2), keepdim=True).values
            else:
                raise ValueError(f"Unsupported normalization mode: {norm_mode}")

            norm_factor = norm_factor.clip(min=1e-8)
            normed_pts = pts / norm_factor.unsqueeze(-1)
            normed_pts_list.append(normed_pts)

        return normed_pts_list

    def _global_loss(self, pred: Dict, target: Dict, mask: torch.Tensor) -> torch.Tensor:
        """
        Global loss with cross-view normalization - EXACT Fast3R approach.

        All views normalized by SINGLE scale factor to enforce consistent alignment.
        """
        if 'conf_global' not in target:
            raise ValueError("conf_global missing from target")

        # Input shape: [B, N, H, W, 3] where N = num views
        B, N, H, W, _ = pred['xyz_global'].shape

        # Convert to list of [B, H, W, 3] for each view (Fast3R format)
        pred_pts_list = [pred['xyz_global'][:, i] for i in range(N)]
        targ_pts_list = [target['xyz_global'][:, i] for i in range(N)]
        valid_list = [mask[:, i] for i in range(N)]

        # Apply cross-view normalization (all views share ONE scale)
        pred_pts_list, _ = self.normalize_pointcloud_from_views(
            pred_pts_list, self.norm_mode, valid_list
        )
        targ_pts_list, _ = self.normalize_pointcloud_from_views(
            targ_pts_list, self.norm_mode, valid_list
        )

        # Compute loss for each view separately on normalized points
        total_loss = 0.0
        for i in range(N):
            conf = target['conf_global'][:, i]  # [B, H, W]
            view_loss = self._confidence_weighted_loss(
                pred_pts_list[i], targ_pts_list[i],
                conf, valid_list[i], weight=1.0
            )
            total_loss += view_loss

        return total_loss / N  # Average over views

    def _local_loss(self, pred: Dict, target: Dict, mask: torch.Tensor) -> torch.Tensor:
        """
        Local loss with per-view normalization - EXACT Fast3R approach.

        Each view normalized independently (local coordinate systems).
        """
        if 'conf_local' not in target:
            raise ValueError("conf_local missing from target")

        # Input shape: [B, N, H, W, 3] where N = num views
        B, N, H, W, _ = pred['xyz_local'].shape

        # Convert to list of [B, H, W, 3] for each view
        pred_pts_list = [pred['xyz_local'][:, i] for i in range(N)]
        targ_pts_list = [target['xyz_local'][:, i] for i in range(N)]
        valid_list = [mask[:, i] for i in range(N)]

        # Apply per-view normalization (each view has own scale)
        pred_pts_list = self.normalize_pointcloud_per_view(
            pred_pts_list, self.norm_mode, valid_list
        )
        targ_pts_list = self.normalize_pointcloud_per_view(
            targ_pts_list, self.norm_mode, valid_list
        )

        # Compute loss for each view separately
        total_loss = 0.0
        for i in range(N):
            conf = target['conf_local'][:, i]  # [B, H, W]
            view_loss = self._confidence_weighted_loss(
                pred_pts_list[i], targ_pts_list[i],
                conf, valid_list[i], weight=1.0
            )
            total_loss += view_loss

        return total_loss / N  # Average over views

    def _confidence_loss(self, pred: Dict, target: Dict, mask: torch.Tensor) -> torch.Tensor:
        # Supervise both local and global confidence maps
        device = mask.device if mask is not None else pred.get('xyz_global', pred.get('xyz_local')).device
        total_conf_loss = torch.tensor(0.0, device=device)
        
        # Local confidence supervision
        if 'conf_local' not in pred or 'conf_local' not in target:
            raise ValueError("Missing conf_local in pred or target for local confidence supervision")
        
        diff_local = (pred['conf_local'] - target['conf_local']).abs()
        if mask is not None:
            masked_local = diff_local * mask.float()
            loss_local = masked_local.sum() / (mask.sum() + 1e-8)
        else:
            loss_local = diff_local.mean()
        
        # Global confidence supervision
        # Student outputs 'conf' from global head, teacher has 'conf_global'
        if 'conf' not in pred:
            if 'conf_global' in pred:  # Try alternative naming
                pred_global = pred['conf_global']
            else:
                raise ValueError("Missing global confidence ('conf' or 'conf_global') in pred")
        else:
            pred_global = pred['conf']
            
        if 'conf_global' not in target:
            raise ValueError("Missing conf_global in target for global confidence supervision")
        
        diff_global = (pred_global - target['conf_global']).abs()
        if mask is not None:
            masked_global = diff_global * mask.float()
            loss_global = masked_global.sum() / (mask.sum() + 1e-8)
        else:
            loss_global = diff_global.mean()
        
        # Average of both confidence losses
        total_conf_loss = (loss_local + loss_global) / 2.0
        
        return total_conf_loss


    def _confidence_weighted_loss(self, pred: torch.Tensor, target: torch.Tensor,
                                confidence: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
        """
        Compute confidence-weighted L2 loss directly in spatial format.

        Args:
            pred: [B, H, W, 3] or [B, H, W] predictions (single view)
            target: [B, H, W, 3] or [B, H, W] targets (single view)
            confidence: [B, H, W] teacher confidence scores
            mask: [B, H, W] valid mask
            weight: Loss weight multiplier
        """
        # Ensure same shape - teacher outputs should already match student resolution
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} != target {target.shape}. " +
                           "Teacher outputs should be generated at student resolution.")

        diff = pred - target
        if diff.shape[-1] == 3:  # XYZ case - last dim is 3D coordinates
            loss = (diff * diff).sum(dim=-1)  # [B, H, W] - sum over XYZ
        else:  # confidence case
            loss = diff * diff

        # Apply confidence weighting to focus on high-confidence regions
        if confidence is not None:
            if confidence.shape != loss.shape:
                raise ValueError(f"Confidence shape {confidence.shape} must match loss shape {loss.shape}")
            conf_weight = torch.clamp(confidence / 10.0, min=0.1, max=1.0)
            loss = loss * conf_weight

        # Apply mask and reduce
        if mask is not None:
            if mask.shape != loss.shape:
                raise ValueError(f"Mask shape {mask.shape} incompatible with loss shape {loss.shape}")
            masked_loss = loss * mask.float()
            return masked_loss.sum() / (mask.sum() + 1e-8) * weight
        else:
            return loss.mean() * weight


    def forward(self, pred: Dict, target: Dict, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Memory-efficient forward pass."""
        # All inputs in spatial format: [B, N, H, W, ...], mask: [B, N, H, W]
        
        # Get device from pred or target tensors if mask is None
        device = mask.device if mask is not None else (
            list(pred.values())[0].device if pred else list(target.values())[0].device
        )
        
        losses = {}

        # Global 3D loss - no flattening
        if 'xyz_global' not in pred:
            raise ValueError("xyz_global missing from pred - required for global geometry loss")
        if 'xyz_global' not in target:
            raise ValueError("xyz_global missing from target - required for global geometry loss")
        losses['global'] = self.alpha_g * self._global_loss(pred, target, mask)

        # Local 3D loss - no flattening
        if 'xyz_local' not in pred:
            raise ValueError("xyz_local missing from pred - required for local geometry loss")
        if 'xyz_local' not in target:
            raise ValueError("xyz_local missing from target - required for local geometry loss")
        losses['local'] = self.alpha_l * self._local_loss(pred, target, mask)
            
        # Confidence loss - no flattening
        losses['confidence'] = self.gamma * self._confidence_loss(pred, target, mask)
            
        total_loss = sum(losses.values())
        
        # Convert to float for logging - handle potential scalar values
        loss_dict = {}
        for k, v in losses.items():
            if hasattr(v, 'item'):
                loss_dict[k] = v.item()
            else:
                loss_dict[k] = float(v)
        
        if hasattr(total_loss, 'item'):
            loss_dict['total'] = total_loss.item()
        else:
            loss_dict['total'] = float(total_loss)
        
        return total_loss, loss_dict

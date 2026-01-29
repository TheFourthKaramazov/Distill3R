"""
RLE (Run-Length Encoding) helpers for compressing boolean masks.

Used to efficiently store large boolean masks in teacher cache files.
"""

import numpy as np
from typing import List, Tuple


def encode_rle(mask: np.ndarray) -> List[int]:
    """
    Encode a boolean mask using run-length encoding.
    
    Args:
        mask: Boolean array of shape (H, W)
        
    Returns:
        List of integers representing RLE encoding.
        Format: [start1, length1, start2, length2, ...]
        where start_i is the starting position of the i-th run of True values
        and length_i is the length of that run.
    """
    # Flatten mask to 1D
    flat_mask = mask.flatten()
    
    # Find runs of True values
    diff = np.diff(np.concatenate(([False], flat_mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Convert to RLE format
    rle = []
    for start, end in zip(starts, ends):
        rle.extend([int(start), int(end - start)])
    
    return rle


def decode_rle(rle: List[int], shape: Tuple[int, int]) -> np.ndarray:
    """
    Decode an RLE-encoded mask back to a boolean array.
    
    Args:
        rle: List of integers in RLE format [start1, length1, start2, length2, ...]
        shape: Target shape (H, W) for the decoded mask
        
    Returns:
        Boolean array of shape (H, W)
    """
    # Create flat mask
    flat_mask = np.zeros(shape[0] * shape[1], dtype=bool)
    
    # Fill in True runs
    for i in range(0, len(rle), 2):
        if i + 1 < len(rle):
            start = rle[i]
            length = rle[i + 1]
            flat_mask[start:start + length] = True
    
    # Reshape to target shape
    return flat_mask.reshape(shape)


def compress_mask(mask: np.ndarray) -> dict:
    """
    Compress a boolean mask for storage.
    
    Args:
        mask: Boolean array of shape (H, W)
        
    Returns:
        Dictionary with compressed mask data and metadata
    """
    return {
        'rle': encode_rle(mask),
        'shape': mask.shape,
        'dtype': str(mask.dtype)
    }


def decompress_mask(compressed: dict) -> np.ndarray:
    """
    Decompress a stored mask.
    
    Args:
        compressed: Dictionary from compress_mask()
        
    Returns:
        Boolean array of original shape
    """
    return decode_rle(compressed['rle'], compressed['shape'])

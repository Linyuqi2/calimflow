import numpy as np
from typing import Tuple, Optional, Union
from scipy import ndimage

def correct_offset_round_error(
    xc: np.ndarray,
    xi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Corrects rounding errors in window coordinates."""
    if np.diff(xc) > np.diff(xi):
        xc[0] = xc[0] + (np.diff(xc) - np.diff(xi))
    elif np.diff(xc) < np.diff(xi):
        xi[0] = xi[0] + (np.diff(xi) - np.diff(xc))
    return xc, xi

def pseudo_rand_sample_2d(
    sz: Tuple[int, int],
    n_samps: int,
    width: float = 2.0,
    weight: float = 1.0,
    pdf: Optional[np.ndarray] = None,
    max_it: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample pseudo-randomly within a 2D matrix with exclusion zones.
    
    Args:
        sz: Size of input matrix (height, width)
        n_samps: Number of locations to sample
        width: Width of Gaussian exclusionary zone
        weight: Weight of Gaussian (0-1, 1 fully excludes previously sampled locations)
        pdf: Probability density function of allowable locations
        max_it: Maximum number of iterations to look for new positions
    
    Returns:
        pos: Sampled positions (n_samps x 2)
        pdf: Updated probability density function
    """
    if pdf is None:
        pdf = np.ones(sz, dtype=np.float32)
        
    # Create Gaussian PDF
    x, y = np.meshgrid(
        np.arange(-int(np.ceil(2*width)), int(np.ceil(2*width))+1),
        np.arange(-int(np.ceil(2*width)), int(np.ceil(2*width))+1)
    )
    gpdf = np.float32(-weight * np.exp(-(x**2 + y**2)/(width**2)))
    
    # Initialize position matrix
    pos = np.zeros((n_samps, 2))
    i = 0
    num_it = 0
    
    while i < n_samps and num_it < max_it:
        num_it += 1
        rnd_pt = np.floor(np.random.rand(2) * np.array(sz)).astype(int)
        
        if (0 <= rnd_pt[0] < sz[0] and 0 <= rnd_pt[1] < sz[1] and 
            (pdf[rnd_pt[0], rnd_pt[1]] - np.random.rand()) > 0):
            # Calculate window coordinates
            xc = np.array([
                max(0, 2*width + 1 - rnd_pt[0]),
                min(0, sz[0] - rnd_pt[0] - 2*width) + 4*width
            ], dtype=np.int32)
            
            yc = np.array([
                max(0, 2*width + 1 - rnd_pt[1]),
                min(0, sz[1] - rnd_pt[1] - 2*width) + 4*width
            ], dtype=np.int32)
            
            # Calculate image coordinates
            xi = np.array([
                max(0, rnd_pt[0] - 2*width),
                min(sz[0]-1, rnd_pt[0] + 2*width)
            ], dtype=np.int32)
            
            yi = np.array([
                max(0, rnd_pt[1] - 2*width),
                min(sz[1]-1, rnd_pt[1] + 2*width)
            ], dtype=np.int32)
            
            # Correct for rounding errors
            xc, xi = correct_offset_round_error(xc, xi)
            yc, yi = correct_offset_round_error(yc, yi)
            
            # Update PDF
            pdf[xi[0]:xi[1]+1, yi[0]:yi[1]+1] += gpdf[
                xc[0]:xc[1]+1,
                yc[0]:yc[1]+1
            ]
            
            pos[i] = rnd_pt
            i += 1
            num_it = 0
            
    return pos, pdf

def pseudo_rand_sample_3d(
    sz: Tuple[int, int, int],
    n_samps: int,
    width: float = 2.0,
    weight: float = 1.0,
    pdf: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample pseudo-randomly within a 3D array with exclusion zones.
    
    Args:
        sz: Size of input array (depth, height, width)
        n_samps: Number of locations to sample
        width: Width of Gaussian exclusionary zone
        weight: Weight of Gaussian (0-1, 1 fully excludes previously sampled locations)
        pdf: Probability density function of allowable locations
    
    Returns:
        pos: Sampled positions (n_samps x 3)
        pdf: Updated probability density function
    """
    if pdf is None:
        pdf = np.ones(sz, dtype=np.float32)
        
    # Create 3D Gaussian PDF
    x, y, z = np.meshgrid(
        np.arange(-int(np.ceil(2*width)), int(np.ceil(2*width))+1),
        np.arange(-int(np.ceil(2*width)), int(np.ceil(2*width))+1),
        np.arange(-int(np.ceil(2*width)), int(np.ceil(2*width))+1)
    )
    gpdf = np.float32(-weight * np.exp(-(x**2 + y**2 + z**2)/(width**2)))
    
    # Initialize position matrix
    pos = np.zeros((n_samps, 3))
    i = 0
    
    while i < n_samps:
        rnd_pt = np.floor(np.random.rand(3) * np.array(sz)).astype(int)
        
        if (0 <= rnd_pt[0] < sz[0] and 0 <= rnd_pt[1] < sz[1] and 
            0 <= rnd_pt[2] < sz[2] and
            (pdf[rnd_pt[0], rnd_pt[1], rnd_pt[2]] - np.random.rand()) > 0):
            xc = np.array([
                max(0, 2*width + 1 - rnd_pt[0]),
                min(0, sz[0] - rnd_pt[0] - 2*width) + 4*width
            ], dtype=np.int32)
            
            yc = np.array([
                max(0, 2*width + 1 - rnd_pt[1]),
                min(0, sz[1] - rnd_pt[1] - 2*width) + 4*width
            ], dtype=np.int32)
            
            zc = np.array([
                max(0, 2*width + 1 - rnd_pt[2]),
                min(0, sz[2] - rnd_pt[2] - 2*width) + 4*width
            ], dtype=np.int32)
            
            # Calculate image coordinates
            xi = np.array([
                max(0, rnd_pt[0] - 2*width),
                min(sz[0]-1, rnd_pt[0] + 2*width)
            ], dtype=np.int32)
            
            yi = np.array([
                max(0, rnd_pt[1] - 2*width),
                min(sz[1]-1, rnd_pt[1] + 2*width)
            ], dtype=np.int32)
            
            zi = np.array([
                max(0, rnd_pt[2] - 2*width),
                min(sz[2]-1, rnd_pt[2] + 2*width)
            ], dtype=np.int32)

            xc, xi = correct_offset_round_error(xc, xi)
            yc, yi = correct_offset_round_error(yc, yi)
            zc, zi = correct_offset_round_error(zc, zi)
            
            # Update PDF using 0-based indexing
            pdf[xi[0]:xi[1]+1, yi[0]:yi[1]+1, zi[0]:zi[1]+1] += gpdf[
                xc[0]:xc[1]+1,
                yc[0]:yc[1]+1,
                zc[0]:zc[1]+1
            ]
            
            pos[i] = rnd_pt
            i += 1
            
    return pos, pdf

def sample_random_location(
    vol_size: Union[np.ndarray, Tuple[int, int, int]],
    n_samples: int,
    min_dist: float,
    excluded_regions: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Sample random locations in a volume with minimum distance constraints.

    Args:
        vol_size: Size of volume (3D array)
        n_samples: Number of locations to sample
        min_dist: Minimum distance between samples
        excluded_regions: Binary array indicating forbidden regions (same size as volume)
        weights: Weight map for biasing sample locations (same size as volume)

    Returns:
        locations: Array of sampled locations (n_samples x 3)
    """
    # Convert inputs to numpy arrays
    vol_size = np.asarray(vol_size)
    
    # Initialize masks
    if excluded_regions is None:
        valid_mask = np.ones(vol_size, dtype=bool)
    else:
        valid_mask = ~excluded_regions
        
    # Initialize weights
    if weights is None:
        weights = np.ones(vol_size)
    weights = weights * valid_mask
    if weights.sum() > 0:
        weights = weights / weights.sum()
    
    # Initialize output
    locations = []
    
    # Create exclusion kernel
    x, y, z = np.meshgrid(
        np.arange(-min_dist, min_dist + 1),
        np.arange(-min_dist, min_dist + 1),
        np.arange(-min_dist, min_dist + 1)
    )
    exclusion_kernel = (x**2 + y**2 + z**2) <= min_dist**2
    
    while len(locations) < n_samples and np.any(valid_mask):
        # Get valid indices and their probabilities
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) == 0:
            break
            
        # Calculate sampling probabilities
        probs = weights[valid_mask]
        if probs.sum() > 0:
            probs = probs / probs.sum()
            
        # Sample location
        idx = np.random.choice(len(probs), p=probs)
        loc = np.array([
            valid_indices[0][idx],
            valid_indices[1][idx],
            valid_indices[2][idx]
        ])
        
        # Add location
        locations.append(loc)
        
        # Update exclusion zone
        x_start = max(0, loc[0] - min_dist)
        x_end = min(vol_size[0], loc[0] + min_dist + 1)
        y_start = max(0, loc[1] - min_dist)
        y_end = min(vol_size[1], loc[1] + min_dist + 1)
        z_start = max(0, loc[2] - min_dist)
        z_end = min(vol_size[2], loc[2] + min_dist + 1)
        
        k_x_start = max(0, min_dist - loc[0])
        k_y_start = max(0, min_dist - loc[1])
        k_z_start = max(0, min_dist - loc[2])
        
        k_x_end = min(2*min_dist + 1, min_dist + vol_size[0] - loc[0])
        k_y_end = min(2*min_dist + 1, min_dist + vol_size[1] - loc[1])
        k_z_end = min(2*min_dist + 1, min_dist + vol_size[2] - loc[2])
        
        valid_mask[x_start:x_end, y_start:y_end, z_start:z_end] &= ~exclusion_kernel[
            k_x_start:k_x_end,
            k_y_start:k_y_end,
            k_z_start:k_z_end
        ]
        
        # Update weights
        weights = weights * valid_mask
        if weights.sum() > 0:
            weights = weights / weights.sum()
    
    if not locations:
        return np.zeros((0, 3))
        
    return np.array(locations)
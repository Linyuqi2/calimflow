import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial import ConvexHull

def rotation_matrix(angle: float, axis: int) -> np.ndarray:
    """Generate rotation matrix for given angle and axis."""
    c = np.cos(np.deg2rad(angle))
    s = np.sin(np.deg2rad(angle))
    if axis == 0:  # x-axis
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:  # y-axis
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # z-axis
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_matrix_vector(v: np.ndarray, deg: float) -> np.ndarray:
    """Generate rotation matrix for rotation around vector v by deg degrees."""
    rad = np.deg2rad(deg)
    if rad == 0:
        return np.eye(3)
        
    v = v / np.linalg.norm(v)
    v1, v2, v3 = v
    ca = np.cos(rad)
    sa = np.sin(rad)
    
    return np.array([
        [ca + v1*v1*(1-ca), v1*v2*(1-ca)-v3*sa, v1*v3*(1-ca)+v2*sa],
        [v2*v1*(1-ca)+v3*sa, ca+v2*v2*(1-ca), v2*v3*(1-ca)-v1*sa],
        [v3*v1*(1-ca)-v2*sa, v3*v2*(1-ca)+v1*sa, ca+v3*v3*(1-ca)]
    ])

def teardrop_projection(vertices: np.ndarray, p: float = 1.0) -> np.ndarray:
    """Project sphere points onto a teardrop shape."""
    # Calculate radii and angles
    rr = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    tt = np.pi - np.arctan2(rr, np.abs(vertices[:, 2])) - (vertices[:, 2] > 0) * np.pi
    
    # Create teardrop shape
    v_tear = np.zeros_like(vertices)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_tear[:, 0] = (vertices[:, 0]/rr) * np.sin(tt) * (np.sin(0.5*tt)**p)
        v_tear[:, 1] = (vertices[:, 1]/rr) * np.sin(tt) * (np.sin(0.5*tt)**p)
    v_tear[:, 2] = -np.cos(tt)
    
    # Handle NaN values (occurs at poles where division by zero happens)
    v_tear = np.nan_to_num(v_tear)
    
    return v_tear 

def create_disk_structure(radius: int) -> np.ndarray:
    """
    Create a disk-shaped structuring element (similar to MATLAB's strel('disk', r)).
    
    Args:
        radius: Radius of the disk in pixels
        
    Returns:
        2D boolean array with disk shape
    """
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = x**2 + y**2 <= radius**2
    return disk

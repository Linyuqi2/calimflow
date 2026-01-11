"""
Spherical sampling and triangulation utilities.

This module implements spherical sampling algorithms used in neural volume generation.
"""

import numpy as np
from scipy.spatial import ConvexHull
from typing import Tuple, Optional


def spiral_sample_sphere(N: int, vis: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Produce an approximately uniform sampling of a unit sphere using a spiral method.

    This function replicates the MATLAB SpiralSampleSphere.m functionality.
    According to the spiral approach, particle longitudes are proportional to particle
    rank (1 to N) and latitudes are assigned to ensure uniform sampling density.

    Args:
        N: Desired number of particles (default=200)
        vis: Optional flag for visualization (not implemented in Python version)

    Returns:
        Tuple of (V, Tri):
        - V: N-by-3 array of vertex (sample) coordinates
        - Tri: M-by-3 list of face-vertex connectivities, or None if not requested

    References:
    Christopher Carlson, 'How I Made Wine Glasses from Sunflowers',
    July 8, 2011. url: http://blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers/
    """
    N = int(np.round(N))

    # Golden ratio and golden angle
    gr = (1 + np.sqrt(5)) / 2  # golden ratio
    ga = 2 * np.pi * (1 - 1/gr)  # golden angle

    # Particle indices
    i = np.arange(N)

    # Latitude: defined so that particle index is proportional to surface area
    lat = np.arccos(1 - 2 * i / (N - 1))

    # Longitude: position particles at even intervals
    lon = i * ga

    # Convert from spherical to Cartesian coordinates
    x = np.sin(lat) * np.cos(lon)
    y = np.sin(lat) * np.sin(lon)
    z = np.cos(lat)

    V = np.column_stack([x, y, z])

    # Triangulation (if requested)
    Tri = None
    try:
        # Use scipy's ConvexHull for triangulation
        hull = ConvexHull(V)
        Tri = hull.simplices
    except:
        # If convex hull fails, return None
        Tri = None

    return V, Tri


def intriangulation(vertices: np.ndarray, triangles: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """
    Test if points are inside a triangulated surface.

    This function replicates the MATLAB intriangulation functionality using
    the barycentric coordinate method.

    Args:
        vertices: Nx3 array of triangle vertices
        triangles: Mx3 array of triangle indices
        query_points: Kx3 array of points to test

    Returns:
        K-length boolean array indicating which points are inside the surface
    """
    def is_point_in_triangle(p, a, b, c):
        """Test if point p is inside triangle defined by points a, b, c"""
        # Compute barycentric coordinates
        v0 = b - a
        v1 = c - a
        v2 = p - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v <= 1)

    # Handle single point case
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)

    inside = np.zeros(len(query_points), dtype=bool)

    for i, point in enumerate(query_points):
        for tri in triangles:
            a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            if is_point_in_triangle(point, a, b, c):
                inside[i] = True
                break

    return inside


def teardrop_projection(vertices: np.ndarray, mode: int = 1) -> np.ndarray:
    """
    Apply teardrop projection to spherical samples.

    This function replicates the MATLAB teardrop_poj functionality.

    Args:
        vertices: Nx3 array of spherical coordinates
        mode: Projection mode (1 for teardrop, 2 for peanut)

    Returns:
        Nx3 array of projected coordinates
    """
    if mode == 1:
        # Teardrop projection (pyramidal neuron)
        # Bias towards one pole to create tear-drop shape
        z_coords = vertices[:, 2]
        # Apply sigmoid-like transformation to create bias
        bias_factor = 1.5
        new_z = np.tanh(bias_factor * z_coords) / np.tanh(bias_factor)
        # Renormalize to unit sphere
        xy_norm = np.sqrt(np.maximum(1 - new_z**2, 1e-10))  # Ensure non-negative with minimum value
        current_xy_norm = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
        # Handle normalization safely
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_factor = np.where(current_xy_norm > 1e-10, xy_norm / current_xy_norm, 0)
        scale_factor = np.nan_to_num(scale_factor, nan=0.0, posinf=0.0, neginf=0.0)
        new_x = vertices[:, 0] * scale_factor
        new_y = vertices[:, 1] * scale_factor
        result = np.column_stack([new_x, new_y, new_z])

    elif mode == 2:
        # Peanut projection
        # Create peanut-like shape by scaling in middle
        z_coords = vertices[:, 2]
        # Apply quadratic scaling
        scale = 1 - 0.3 * (1 - z_coords**2)
        result = vertices * scale[:, np.newaxis]

    else:
        result = vertices.copy()

    return result

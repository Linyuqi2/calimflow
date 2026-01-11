"""
Neural body generation using Gaussian Processes.

This module implements the generation of individual neural soma shapes using
isotropic Gaussian processes on spherical surfaces.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from typing import Tuple, Optional, Dict, Any
from .sphere_sampling import spiral_sample_sphere, teardrop_projection


def geodesic_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Compute geodesic distances on a sphere between points.

    Args:
        points: Nx3 array of points on unit sphere

    Returns:
        NxN matrix of geodesic distances
    """
    # Compute pairwise Euclidean distances
    euclidean_dist = squareform(pdist(points))

    # Convert to geodesic distances using spherical law of cosines
    # For unit sphere, geodesic distance = arccos(dot product)
    cos_angles = np.clip(np.dot(points, points.T), -1, 1)
    geodesic_dist = np.arccos(cos_angles)

    return geodesic_dist


def generate_neural_body(neur_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single neural soma shape using Gaussian process.

    This function replicates the MATLAB generateNeuralBody.m functionality.

    Args:
        neur_params: Neuron parameters dictionary

    Returns:
        Tuple of (Vcell, Vnuc, Tri, rotAng):
        - Vcell: Nx3 array of soma surface points
        - Vnuc: Nx3 array of nucleus surface points
        - Tri: Mx3 triangulation array
        - rotAng: 1x3 rotation angles [Rx, Ry, Rz]
    """
    # Get spherical sampling
    if 'S_samp' in neur_params and neur_params['S_samp'] is not None:
        V = neur_params['S_samp']
        Tri = neur_params['Tri']
    else:
        V, Tri = spiral_sample_sphere(neur_params['n_samps'], False)

    # Apply neuron type-specific transformations
    if neur_params['neur_type'] == 'pyr':
        Vtear = teardrop_projection(V, 1)  # Teardrop for pyramidal
    elif neur_params['neur_type'] == 'peanut':
        Vtear = teardrop_projection(V, 2)  # Peanut shape
    else:
        Vtear = V.copy()

    # Compute covariance matrix based on geodesic distances
    if 'dists' in neur_params and neur_params['dists'] is not None:
        dists = neur_params['dists']
    else:
        dists = geodesic_distance_matrix(V)
        # Apply Gaussian covariance
        pwr = 1  # GP power (fixed in MATLAB)
        dists = neur_params['p_scale'] * np.exp(-(dists / neur_params['l_scale']) ** pwr)

    # Set up mean function for neuron type
    if neur_params['neur_type'] == 'pyr':
        Rtear = np.sqrt(np.sum(Vtear**2, axis=1))
    elif neur_params['neur_type'] == 'peanut':
        Rtear = np.ones(len(V))  # Isotropic for peanut
    else:
        Rtear = np.ones(len(V))

    # Ensure positive definite covariance matrix
    eigenvals = np.linalg.eigvals(dists)
    min_eigenval = np.min(eigenvals.real)
    if min_eigenval < 0:
        dists = dists + abs(min_eigenval) * np.eye(dists.shape[0])

    # Generate GP sample
    x_bnds = np.array(neur_params['exts']) * neur_params['avg_rad']
    try:
        # Sample from multivariate normal
        x_base = np.random.multivariate_normal(np.zeros(len(Rtear)), dists)
        x_base = x_base.astype(float)
    except:
        # Fallback for numerical issues
        x_base = np.random.randn(len(Rtear)) * np.sqrt(np.diag(dists))

    # Center and scale the radii
    x = x_base - np.mean(x_base) + neur_params['avg_rad'] * Rtear
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x_bnds[1] - x_bnds[0]) * (x - x_min) / (x_max - x_min) + x_bnds[0]

    # Generate nucleus radii
    if neur_params['neur_type'] == 'pyr':
        x2 = x_base - np.mean(x_base) + neur_params['avg_rad']
        x2_min = np.min(x2)
        x2_max = np.max(x2)
        x2 = (x_bnds[1] - x_bnds[0]) * (x2 - x2_min) / (x2_max - x2_min) + x_bnds[0]
    else:
        x2 = x.copy()

    # Generate elliptical shapes
    eccens = np.ones(3) + neur_params['eccen'] * (np.random.rand(3) - 0.5)
    eccens = eccens / (np.prod(eccens) ** (1/3))  # Normalize

    if neur_params['neur_type'] == 'pyr':
        Vetear = Vtear * eccens
        Vetear = Vetear / np.sqrt(np.mean(np.sum(Vetear**2, axis=1)))
    else:
        Vetear = V * eccens
        Vetear = Vetear / np.sqrt(np.mean(np.sum(Vetear**2, axis=1)))

    # Create soma and nucleus surfaces
    nucoff = 3  # Nucleus offset (from MATLAB)
    Vcell = Vetear * x[:, np.newaxis]
    Vcell = Vcell + np.array([0, 0, -nucoff])  # Shift soma down

    # Create nucleus
    Vnuc = Vetear * x2[:, np.newaxis]
    Vnuc = Vnuc + np.array([0, 0, -nucoff])  # Shift nucleus down

    # Apply nucleus smoothing/shrinking (from MATLAB nexts parameters)
    Vnorms2 = np.sqrt(np.sum(Vnuc**2, axis=1))
    Vnorms = np.sqrt(np.sum(Vcell**2, axis=1))

    # Shrink and smooth nucleus
    nexts = neur_params['nexts']
    Vnorms2_smooth = nexts[1] * (nexts[0] * (Vnorms2 - np.min(Vnorms2)) / (np.max(Vnorms2) - np.min(Vnorms2)) +
                                  (1 - nexts[0]) * np.max(Vnorms2))

    # Ensure nucleus fits inside soma with minimum thickness
    Vnorms2_smooth = Vnorms2_smooth + np.min(Vnorms - Vnorms2_smooth) - neur_params['min_thic'][1]

    # Apply elliptical scaling to nucleus
    Vnuc = Vnuc * eccens * (Vnorms2_smooth / np.sqrt(np.sum(Vnuc**2, axis=1)))[:, np.newaxis]

    # Add lateral shift to nucleus
    lat_ang = np.random.rand() * 2 * np.pi
    lat_shft = neur_params['min_thic'][1] * np.array([np.sin(lat_ang), np.cos(lat_ang), 0])
    Vnuc = Vnuc + lat_shft

    # Shift both soma and nucleus up
    Vcell = Vcell + np.array([0, 0, nucoff])
    Vnuc = Vnuc + np.array([0, 0, nucoff])

    # Optional nuclear radius adjustment
    if 'nuc_rad' in neur_params and neur_params['nuc_rad'] is not None:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(Vnuc)
            VnucSz = hull.volume * 3/(4*np.pi)  # Approximate volume
            nucsz = (4/3) * np.pi * (neur_params['nuc_rad'][0]**3)

            if len(neur_params['nuc_rad']) > 1:
                scale_factor = ((nucsz / VnucSz) ** (1/3)) ** (1/neur_params['nuc_rad'][1])
            else:
                scale_factor = (nucsz / VnucSz) ** (1/3)

            Vnuc = Vnuc * scale_factor
        except:
            pass  # Skip if convex hull fails

    # Apply random rotations
    max_ang = neur_params.get('max_ang', 20.0)
    rotAng = -max_ang + 2 * max_ang * np.random.rand(3)

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(np.radians(rotAng[0])), -np.sin(np.radians(rotAng[0]))],
                   [0, np.sin(np.radians(rotAng[0])), np.cos(np.radians(rotAng[0]))]])

    Ry = np.array([[np.cos(np.radians(rotAng[1])), 0, np.sin(np.radians(rotAng[1]))],
                   [0, 1, 0],
                   [-np.sin(np.radians(rotAng[1])), 0, np.cos(np.radians(rotAng[1]))]])

    Rz = np.array([[np.cos(np.radians(rotAng[2])), -np.sin(np.radians(rotAng[2])), 0],
                   [np.sin(np.radians(rotAng[2])), np.cos(np.radians(rotAng[2])), 0],
                   [0, 0, 1]])

    # Apply rotations
    Vnuc = Vnuc @ Rx.T @ Ry.T @ Rz.T
    Vcell = Vcell @ Rx.T @ Ry.T @ Rz.T

    return Vcell, Vnuc, Tri, rotAng

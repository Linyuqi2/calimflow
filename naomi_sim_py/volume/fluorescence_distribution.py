"""
Fluorescence distribution algorithms for neural volume simulation.

This module implements fluorescence value assignment for cellular components
using Gaussian processes and distance-based decay models.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import cdist


def masked_3d_gp(size: Tuple[int, int, int], length_scale: float,
                variance: float, mask_value: float = 0) -> np.ndarray:
    """
    Generate 3D Gaussian Process with masking.

    This is a simplified implementation of the MATLAB masked_3DGP_v2 function.

    Args:
        size: Tuple of (nx, ny, nz) for the GP field size
        length_scale: GP length scale parameter
        variance: GP variance parameter
        mask_value: Value to use for masking (not implemented in this version)

    Returns:
        3D numpy array with GP values
    """
    nx, ny, nz = size

    # Create coordinate grids
    x = np.linspace(-nx//2, nx//2, nx)
    y = np.linspace(-ny//2, ny//2, ny)
    z = np.linspace(-nz//2, nz//2, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

    # Calculate squared distances
    dist_sq = cdist(coords, coords, 'sqeuclidean')

    # Gaussian covariance matrix
    cov_matrix = variance * np.exp(-dist_sq / (2 * length_scale**2))

    # Add small diagonal term for numerical stability
    cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])

    try:
        # Generate GP sample
        gp_values = np.random.multivariate_normal(np.zeros(len(coords)), cov_matrix)
        gp_field = gp_values.reshape((nx, ny, nz))
    except:
        # Fallback for numerical issues
        gp_field = np.random.randn(nx, ny, nz) * np.sqrt(variance)

    return gp_field


def set_cell_fluorescence(vol_params: Dict[str, Any],
                         neur_params: Dict[str, Any],
                         dend_params: Dict[str, Any],
                         neur_num: np.ndarray,
                         neur_soma: np.ndarray,
                         neur_num_AD: np.ndarray,
                         neur_locs: np.ndarray,
                         neur_vol: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Set fluorescence distributions for all cells.

    This function implements a simplified version of the MATLAB setCellFluoresence.m
    functionality for assigning fluorescence values to cellular components.

    Args:
        vol_params: Volume parameters
        neur_params: Neuron parameters
        dend_params: Dendrite parameters
        neur_num: Neuron number array
        neur_soma: Neuron soma array
        neur_num_AD: Apical dendrite array
        neur_locs: Neuron locations
        neur_vol: Current fluorescence volume

    Returns:
        Tuple of (gp_vals, neur_vol):
        - gp_vals: List of fluorescence distributions per neuron
        - neur_vol: Updated fluorescence volume
    """
    if vol_params.get('verbose', 1) >= 1:
        print('Setting fluorescence distributions...')

    vol_sz = vol_params['vol_sz']
    N_neur = vol_params['N_neur']
    vres = vol_params['vres']

    # Initialize gp_vals structure
    gp_vals = []
    for i in range(N_neur + vol_params.get('N_den', 10)):
        gp_vals.append({
            'locations': np.array([], dtype=np.int32),
            'fluorescence': np.array([], dtype=np.float32),
            'is_soma': np.array([], dtype=bool)
        })

    # Count voxels per neuron/component
    num_vox = np.zeros(N_neur + vol_params.get('N_den', 10), dtype=int)
    for idx in range(len(neur_num.flat)):
        comp_id = neur_num.flat[idx]
        if 1 <= comp_id <= len(num_vox):
            num_vox[comp_id - 1] += 1

    # Initialize location arrays
    for i in range(len(gp_vals)):
        gp_vals[i]['locations'] = np.zeros(num_vox[i], dtype=np.int32)
        gp_vals[i]['fluorescence'] = np.zeros(num_vox[i], dtype=np.float32)
        gp_vals[i]['is_soma'] = np.zeros(num_vox[i], dtype=bool)

    # Fill location arrays
    curr_vox = np.zeros(len(num_vox), dtype=int)
    for idx in range(len(neur_num.flat)):
        comp_id = neur_num.flat[idx]
        if 1 <= comp_id <= len(num_vox):
            comp_idx = comp_id - 1
            gp_vals[comp_idx]['locations'][curr_vox[comp_idx]] = idx
            curr_vox[comp_idx] += 1

    # Set fluorescence for each neuron
    for kk in range(N_neur):
        if vol_params.get('verbose', 1) >= 1:
            print(f'  Processing neuron {kk+1}/{N_neur}')

        # Generate GP for soma fluorescence variation (use smaller, fixed size)
        gp_size = (20, 20, 20)  # Fixed reasonable size to avoid memory issues
        gp_field = masked_3d_gp(gp_size, neur_params.get('fluor_dist', [1.0, 1.0])[0] * vres,
                              neur_params.get('fluor_dist', [1.0, 1.0])[1])

        # Get neuron locations
        neur_locs_kk = gp_vals[kk]['locations']
        if len(neur_locs_kk) == 0:
            continue

        # Find soma locations
        soma_locs = neur_locs_kk[neur_soma.flat[neur_locs_kk] == (kk + 1)]
        ad_locs = neur_locs_kk[neur_num_AD.flat[neur_locs_kk]]

        # Convert to coordinates
        vol_size_pixels = (np.array(vol_sz) * vres).astype(int)
        soma_coords = np.array(np.unravel_index(soma_locs, vol_size_pixels)).T
        all_coords = np.array(np.unravel_index(neur_locs_kk, vol_size_pixels)).T

        # Calculate distances from neuron center
        neur_center = neur_locs[kk] * vres
        distances = np.sqrt(np.sum((all_coords - neur_center)**2, axis=1))

        # Get weight parameters (simplified)
        wtSc = dend_params.get('weightScale', [1.0, 0.5, 0.8])

        # Calculate base fluorescence with distance decay
        base_fluorescence = wtSc[1] * np.exp(-distances / (vres * wtSc[0])) + (1 - wtSc[1])

        # Apply soma-specific fluorescence variation
        soma_fluorescence = np.zeros(len(neur_locs_kk))

        # For soma voxels, apply GP-based variation
        if len(soma_locs) > 0:
            soma_coords_local = (soma_coords - neur_center).astype(int)
            gp_center = np.array(gp_size) // 2

            for i, soma_coord in enumerate(soma_coords_local):
                # Map to GP coordinates (with bounds checking)
                gp_x = np.clip(gp_center[0] + soma_coord[0], 0, gp_size[0] - 1)
                gp_y = np.clip(gp_center[1] + soma_coord[1], 0, gp_size[1] - 1)
                gp_z = np.clip(gp_center[2] + soma_coord[2], 0, gp_size[2] - 1)

                gp_value = gp_field[gp_x, gp_y, gp_z]

                # Normalize GP value to [-1, 1] range and shift to [0, 2]
                normalized_gp = (gp_value - np.mean(gp_field)) / (np.max(np.abs(gp_field - np.mean(gp_field))) + 1e-6)
                soma_fluorescence[i] = 0.5 * normalized_gp + 1.0

            # Mark soma locations
            gp_vals[kk]['is_soma'][np.isin(neur_locs_kk, soma_locs)] = True

        # Mark apical dendrite locations
        if len(ad_locs) > 0:
            gp_vals[kk]['fluorescence'][np.isin(neur_locs_kk, ad_locs)] = 1.0

        # Combine fluorescence values
        fluorescence = base_fluorescence.copy()

        # Apply soma fluorescence where applicable
        if len(soma_locs) > 0:
            soma_mask = np.isin(neur_locs_kk, soma_locs)
            fluorescence[soma_mask] = soma_fluorescence[soma_mask]

        # Apply apical dendrite fluorescence
        if len(ad_locs) > 0:
            ad_mask = np.isin(neur_locs_kk, ad_locs)
            fluorescence[ad_mask] = 1.0

        # Store fluorescence values
        gp_vals[kk]['fluorescence'] = fluorescence.astype(np.float32)

        # Update volume if provided
        if neur_vol is not None and len(neur_vol.flat) == len(neur_num.flat):
            neur_vol.flat[neur_locs_kk] = fluorescence

    # Process dendrites (simplified - just set uniform fluorescence)
    for kk in range(N_neur, N_neur + vol_params.get('N_den', 0)):
        if kk < len(gp_vals) and len(gp_vals[kk]['locations']) > 0:
            gp_vals[kk]['fluorescence'] = np.ones(len(gp_vals[kk]['locations']), dtype=np.float32)

    if vol_params.get('verbose', 1) >= 1:
        print('Fluorescence distribution completed.')

    return gp_vals, neur_vol

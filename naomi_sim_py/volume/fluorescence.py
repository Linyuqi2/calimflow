"""
Fluorescence distribution assignment for neural volumes.

This module implements the setCellFluorescence functionality that assigns
non-uniform fluorescence values to neural components.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import ndimage


def masked_3dgp_v2(grid_sz: np.ndarray, l_scale: float, p_scale: float, mu: float = 0) -> np.ndarray:
    """
    Generate a 3D Gaussian Process sample.

    This function replicates the MATLAB masked_3DGP_v2.m functionality.

    Args:
        grid_sz: Grid dimensions [x, y, z]
        l_scale: Length scale of the GP
        p_scale: Covariance scaling parameter
        mu: Mean value (default=0)

    Returns:
        3D array of GP sample values
    """
    # Ensure grid_sz is array
    if np.isscalar(grid_sz):
        grid_sz = np.array([grid_sz, grid_sz, grid_sz])
    elif len(grid_sz) == 1:
        grid_sz = np.array([grid_sz[0], grid_sz[0], grid_sz[0]])

    # Ensure l_scale is array
    if np.isscalar(l_scale):
        l_scale = np.array([l_scale, l_scale, l_scale])

    # Generate frequency domain coordinates
    freq_coords = []
    for i, sz in enumerate(grid_sz):
        coords = np.fft.fftfreq(sz)
        freq_coords.append(coords)

    kx, ky, kz = np.meshgrid(freq_coords[0], freq_coords[1], freq_coords[2], indexing='ij')

    # Compute power spectrum (Fourier transform of covariance)
    k_squared = kx**2 / l_scale[0]**2 + ky**2 / l_scale[1]**2 + kz**2 / l_scale[2]**2

    # Avoid division by zero at k=0
    power_spectrum = p_scale / (1 + k_squared)**2  # Using squared exponential kernel in frequency domain
    power_spectrum[0, 0, 0] = 0  # Set DC component to zero

    # Generate complex Gaussian random variables
    noise_real = np.random.normal(0, 1, grid_sz)
    noise_imag = np.random.normal(0, 1, grid_sz)

    # Apply power spectrum
    fft_noise = np.fft.fftn(noise_real + 1j * noise_imag) * np.sqrt(power_spectrum)

    # Inverse FFT to get spatial domain
    gp_sample = np.real(np.fft.ifftn(fft_noise))

    # Normalize and add mean
    gp_sample = gp_sample / np.std(gp_sample) * np.sqrt(p_scale) + mu

    return gp_sample.astype(np.float32)


def set_cell_fluorescence(vol_params: Dict[str, Any],
                         neur_params: Dict[str, Any],
                         dend_params: Dict[str, Any],
                         neur_num: np.ndarray,
                         neur_soma: np.ndarray,
                         neur_num_AD: np.ndarray,
                         neur_locs: np.ndarray,
                         neur_vol: np.ndarray) -> Tuple[List[List], np.ndarray]:
    """
    Set non-uniform fluorescence distribution for neural cells.

    This function replicates the MATLAB setCellFluoresence.m functionality.

    Args:
        vol_params: Volume parameters
        neur_params: Neuron parameters
        dend_params: Dendrite parameters
        neur_num: Array with neuron IDs at each voxel
        neur_soma: Array with neuron IDs at soma voxels
        neur_num_AD: Array with apical dendrite IDs at each voxel
        neur_locs: Neuron locations [N_neur, 3]
        neur_vol: Current fluorescence volume array

    Returns:
        Tuple of (gp_vals, neur_vol):
        - gp_vals: List of [locations, fluorescence_values, is_soma_mask] for each component
        - neur_vol: Updated fluorescence volume
    """
    vol_sz = np.array(vol_params['vol_sz'])
    N_neur = vol_params['N_neur']
    vres = vol_params['vres']

    # Extract fluorescence parameters - match MATLAB exactly
    wtSc = dend_params.get('weightScale', np.array([150, 1, 0.8]))  # Default weights [distance, weight, variation]
    flSc = neur_params.get('fluor_dist', np.array([1, 0.2]))  # Default fluor params [length, weight]

    if vol_params.get('verbose', 1) >= 1:
        print('Setting fluorescence distributions...')

    # Count voxels per component
    numcomps = N_neur + vol_params.get('N_den', 0)
    neur_num_flat = neur_num.flatten()
    numvox = np.zeros(numcomps, dtype=int)

    for kk in range(1, numcomps + 1):  # MATLAB uses 1-based indexing
        numvox[kk-1] = np.sum(neur_num_flat == kk)

    # Initialize gp_vals as list of lists - match MATLAB exactly
    gp_vals = []
    for kk in range(numcomps):
        gp_vals.append([np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=bool)])

    # Populate gp_vals with locations - match MATLAB loop logic
    for idx in range(len(neur_num_flat)):
        component_id = neur_num_flat[idx]
        if 1 <= component_id <= numcomps:
            # MATLAB uses 1-based indexing, convert to 0-based for Python list
            list_idx = component_id - 1
            gp_vals[list_idx][0] = np.append(gp_vals[list_idx][0], idx)
            gp_vals[list_idx][1] = np.append(gp_vals[list_idx][1], 1.0)
            gp_vals[list_idx][2] = np.append(gp_vals[list_idx][2], False)

    # Process neural components (neurons)
    for kk in range(N_neur):
        if vol_params.get('verbose', 1) >= 1:
            print(f'  Processing neuron {kk+1}/{N_neur}')

        component_idx = kk  # 0-based indexing for Python list

        # Generate 3D GP for soma fluorescence distribution - match MATLAB exactly
        gp_size_val = np.round(neur_params['avg_rad'] * 6 * vres)
        gp_grid = np.full(3, gp_size_val, dtype=int)  # Create [size, size, size] array

        # Set fixed seed for reproducible GP generation (match MATLAB behavior)
        np.random.seed(42 + kk)  # Use neuron index for consistent but different seeds

        TMP = masked_3dgp_v2(gp_grid, flSc[0] * vres, flSc[1], 0)

        # Get locations for this neuron
        TMP_loc = gp_vals[component_idx][0]

        # Find soma locations for this neuron
        soma_mask = neur_soma.flatten()[TMP_loc] == (kk + 1)
        TMP_soma = TMP_loc[soma_mask]

        # Find apical dendrite locations for this neuron
        ad_mask = neur_num_AD.flatten()[TMP_loc] == (kk + 1)
        TMP_AD = TMP_loc[ad_mask]

        if len(TMP_soma) > 0:
            # Convert linear indices to 3D coordinates - match MATLAB ind2sub
            soma_coords = np.array(np.unravel_index(TMP_soma, vol_sz * vres))

            # Calculate distances from neuron center - match MATLAB exactly
            neuron_center = np.floor(vres * neur_locs[kk]).astype(int)
            gp_center = np.floor(gp_grid[0] / 2).astype(int)

            TMP_dist = soma_coords - neuron_center[:, np.newaxis] + gp_center

            # Ensure indices are within bounds - match MATLAB bsxfun(@max,@min)
            TMP_dist = np.maximum(TMP_dist, np.array([1, 1, 1])[:, np.newaxis])
            TMP_dist = np.minimum(TMP_dist, gp_grid[:, np.newaxis])

            # MATLAB uses 1-based indexing, convert to 0-based
            TMP_dist = TMP_dist - 1

            # Get GP values at soma locations
            TMP_vals = TMP[TMP_dist[0], TMP_dist[1], TMP_dist[2]]

            # Normalize GP values - exact MATLAB formula
            if len(TMP_vals) > 1:
                tmp_mean = np.mean(TMP_vals)
                tmp_diff = TMP_vals - tmp_mean
                max_abs_diff = np.max(np.abs(tmp_diff))
                if max_abs_diff > 0:
                    TMP_vals = 0.5 * tmp_diff / max_abs_diff + 1
                else:
                    TMP_vals = np.ones_like(TMP_vals)
            TMP_vals = np.nan_to_num(TMP_vals, nan=1.0)

            # Apply soma fluorescence values
            gp_vals[component_idx][1][soma_mask] = TMP_vals
            gp_vals[component_idx][2][soma_mask] = True  # Mark as soma

        # Apply apical dendrite values (constant)
        if len(TMP_AD) > 0:
            gp_vals[component_idx][1][ad_mask] = 1.0

        # Calculate distance-based decay for all points in this component - match MATLAB exactly
        if len(TMP_loc) > 0:
            # Convert linear indices to coordinates
            coords = np.array(np.unravel_index(TMP_loc, vol_sz * vres))

            # Calculate distances from neuron center - match MATLAB TMP_sep calculation
            distances = np.sqrt(np.sum((coords - vres * neur_locs[kk, :, np.newaxis])**2, axis=0))

            # Apply exponential decay - use correct wtSc parameter order: wtSc[1]=weight, wtSc[0]=scale, wtSc[2]=variation
            decay_factor = wtSc[1] * np.exp(-distances / (vres * wtSc[0])) + (1 - wtSc[1])
            decay_factor *= (1 - wtSc[2])

            # Apply decay but preserve soma values - this matches MATLAB logic
            gp_vals[component_idx][1] = decay_factor * (1 - gp_vals[component_idx][2]) + \
                                       gp_vals[component_idx][1] * gp_vals[component_idx][2]

    # Process dendrite components (if any)
    for kk in range(N_neur, numcomps):
        component_idx = kk
        # Dendrites get constant fluorescence
        gp_vals[component_idx][1][:] = 1.0
        gp_vals[component_idx][2][:] = False

    # Update volume with fluorescence values (match MATLAB conservative approach)
    total_assigned = 0
    fluorescence_threshold = 0.0  # No threshold - keep all fluorescence values

    for kk in range(numcomps):
        if len(gp_vals[kk][0]) > 0:
            fluorescence_vals = gp_vals[kk][1]
            # Apply threshold to match MATLAB's conservative fluorescence distribution
            significant_mask = fluorescence_vals >= fluorescence_threshold
            if np.any(significant_mask):
                significant_indices = gp_vals[kk][0][significant_mask]
                significant_values = fluorescence_vals[significant_mask]
                neur_vol.flat[significant_indices] = significant_values
                total_assigned += len(significant_indices)

    if vol_params.get('verbose', 1) >= 1:
        print(f'Fluorescence distribution completed. Assigned significant fluorescence (>={fluorescence_threshold}) to {total_assigned} voxels across {numcomps} components.')

    return gp_vals, neur_vol


def update_volume_fluorescence(gp_vals: List[List], neur_vol: np.ndarray) -> np.ndarray:
    """
    Update volume fluorescence from gp_vals.

    Args:
        gp_vals: List of [locations, fluorescence_values, is_soma_mask] for each component
        neur_vol: Volume array to update

    Returns:
        Updated volume array
    """
    for component_vals in gp_vals:
        if len(component_vals[0]) > 0:
            neur_vol.flat[component_vals[0]] = component_vals[1]

    return neur_vol

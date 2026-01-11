"""
Generate neural volume by placing neurons in 3D space.

This module implements the placement of neural somas and nuclei in a volume
array, replicating the MATLAB generateNeuralVolume.m functionality.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from .sphere_sampling import intriangulation


def generate_neural_volume(neur_params: Dict[str, Any],
                          vol_params: Dict[str, Any],
                          neur_locs: np.ndarray,
                          Vcell: List[np.ndarray],
                          Vnuc: List[np.ndarray],
                          neur_ves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[np.ndarray]]:
    """
    Place neural somas and nuclei in a volume.

    This function replicates the MATLAB generateNeuralVolume.m functionality.

    Args:
        neur_params: Neuron parameters
        vol_params: Volume parameters
        neur_locs: Nx3 array of neuron locations
        Vcell: List of soma surface meshes
        Vnuc: List of nucleus surface meshes
        neur_ves: 3D boolean array of vessel locations

    Returns:
        Tuple of (neur_soma, neur_vol, gp_nuc, gp_soma):
        - neur_soma: Volume array with neuron IDs at soma locations
        - neur_vol: Volume array with fluorescence values
        - gp_nuc: List of nuclear fluorescence distributions
        - gp_soma: List of soma location indices
    """
    # Initialize volume arrays
    vol_sz = vol_params['vol_sz']
    vres = vol_params['vres']

    neur_soma = np.zeros((int(vol_sz[0] * vres),
                         int(vol_sz[1] * vres),
                         int(vol_sz[2] * vres)), dtype=np.uint16)
    neur_vol = np.zeros_like(neur_soma, dtype=np.float32)

    # Initialize GP storage
    gp_nuc = []
    gp_soma = []

    # Get triangulation for all neurons (assume same triangulation)
    _, Tri = Vcell[0].shape if len(Vcell) > 0 else (None, None)
    if Tri is None:
        # Create simple triangulation if not available
        n_points = len(Vcell[0]) if len(Vcell) > 0 else 0
        Tri = np.array([[0, 1, 2]])  # Dummy triangulation

    # Setup taken points array (copy vessel locations)
    taken_pts = neur_ves.copy()

    # Process depth offset
    vol_depth = vol_params['vol_depth'] * vres
    taken_pts = taken_pts[:, :, int(vol_depth):int(vol_depth + vol_sz[2] * vres)]

    N_neur = len(neur_locs)

    for kk in range(N_neur):
        # Calculate bounding box for current neuron
        Vcell_kk = Vcell[kk]
        max_ext = np.ceil(np.max(np.sqrt(np.sum((Vcell_kk - neur_locs[kk])**2, axis=1))))
        mExtRes = int(np.ceil(max_ext * vres))

        idx_pos = np.round(vres * neur_locs[kk]).astype(int)

        # Define local region bounds (match MATLAB 1-based indexing)
        vol_size_pixels = (int(vol_sz[0] * vres), int(vol_sz[1] * vres), int(vol_sz[2] * vres))

        idxX_start = max(1, idx_pos[0] - mExtRes)
        idxX_end = min(idx_pos[0] + mExtRes, vol_size_pixels[0])
        idxX = slice(idxX_start - 1, idxX_end)  # Convert to 0-based slice

        idxY_start = max(1, idx_pos[1] - mExtRes)
        idxY_end = min(idx_pos[1] + mExtRes, vol_size_pixels[1])
        idxY = slice(idxY_start - 1, idxY_end)  # Convert to 0-based slice

        idxZ_start = max(1, idx_pos[2] - mExtRes)
        idxZ_end = min(idx_pos[2] + mExtRes, vol_size_pixels[2])
        idxZ = slice(idxZ_start - 1, idxZ_end)  # Convert to 0-based slice

        # Create local meshgrid (match MATLAB exactly)
        local_shape = (idxX.stop - idxX.start,
                      idxY.stop - idxY.start,
                      idxZ.stop - idxZ.start)

        if local_shape[0] <= 0 or local_shape[1] <= 0 or local_shape[2] <= 0:
            continue  # Skip if region is invalid

        # Create local coordinates (match MATLAB meshgrid logic)
        x_coords = np.arange(idxX_start, idxX_end + 1)  # MATLAB 1-based coordinates
        y_coords = np.arange(idxY_start, idxY_end + 1)  # MATLAB 1-based coordinates
        z_coords = np.arange(idxZ_start, idxZ_end + 1)  # MATLAB 1-based coordinates

        x_local = (x_coords - idx_pos[0]) / vres
        y_local = (y_coords - idx_pos[1]) / vres
        z_local = (z_coords - idx_pos[2]) / vres

        mesh_x, mesh_y, mesh_z = np.meshgrid(x_local, y_local, z_local, indexing='ij')
        mesh_x = mesh_x.T  # Match MATLAB permute
        mesh_y = mesh_y.T
        mesh_z = mesh_z.T

        # Create test points (match MATLAB idx_tri calculation)
        mesh_x_flat = mesh_x.flatten()
        mesh_y_flat = mesh_y.flatten()
        mesh_z_flat = mesh_z.flatten()

        test_points = np.column_stack([
            mesh_x_flat + neur_locs[kk, 0] + 1/(2*vres),  # Add 1/vres/2 offset
            mesh_y_flat + neur_locs[kk, 1] + 1/(2*vres),
            mesh_z_flat + neur_locs[kk, 2] + 1/(2*vres)
        ])

        # Test which points are inside the soma
        try:
            TMP1 = intriangulation(Vcell[kk], Tri, test_points)
            TMP1 = TMP1.reshape(local_shape)
        except:
            # Fallback: use distance-based approximation
            distances = np.sqrt(np.sum((test_points - neur_locs[kk])**2, axis=1))
            max_rad = np.max(np.sqrt(np.sum((Vcell[kk] - neur_locs[kk])**2, axis=1)))
            TMP1 = (distances <= max_rad).reshape(local_shape)

        # Test which points are inside the nucleus
        try:
            TMP = intriangulation(Vnuc[kk], Tri, test_points)
            TMP = TMP.reshape(local_shape)
        except:
            # Fallback: use distance-based approximation
            distances = np.sqrt(np.sum((test_points - neur_locs[kk])**2, axis=1))
            max_rad = np.max(np.sqrt(np.sum((Vnuc[kk] - neur_locs[kk])**2, axis=1)))
            TMP = (distances <= max_rad).reshape(local_shape)

        # Find valid soma points (inside soma but not nucleus, not occupied)
        neur_idx = TMP1 & (~TMP) & (~taken_pts[idxX, idxY, idxZ])

        # Update occupied points
        taken_pts[idxX, idxY, idxZ] = taken_pts[idxX, idxY, idxZ] | neur_idx

        # Find linear indices for soma points (match MATLAB exactly)
        soma_indices = np.where(neur_idx)
        if len(soma_indices[0]) > 0:
            # Convert local indices to global 1-based indices (match MATLAB)
            iX = soma_indices[0] + idxX_start  # MATLAB uses 1-based global indices
            iY = soma_indices[1] + idxY_start
            iZ = soma_indices[2] + idxZ_start

            # Convert to 0-based for numpy array indexing
            neur_idx2 = np.ravel_multi_index((iX - 1, iY - 1, iZ - 1), neur_soma.shape)

            # Mark soma locations
            neur_soma.flat[neur_idx2] = kk + 1  # MATLAB uses 1-based indexing

            # Store soma indices
            gp_soma.append(neur_idx2.astype(np.int32))

        # Process nucleus points
        neur_idx_nuc = TMP.copy()

        # Find linear indices for nucleus points (match MATLAB)
        nuc_indices = np.where(neur_idx_nuc)
        if len(nuc_indices[0]) > 0:
            iX = nuc_indices[0] + idxX_start  # MATLAB 1-based global indices
            iY = nuc_indices[1] + idxY_start
            iZ = nuc_indices[2] + idxZ_start

            neur_idx2 = np.ravel_multi_index((iX - 1, iY - 1, iZ - 1), neur_soma.shape)  # Convert to 0-based

            # Store nucleus information
            gp_nuc.append({
                'locations': neur_idx2.astype(np.int32),
                'fluorescence': neur_params['nuc_fluorsc']
            })

            # Set nuclear fluorescence (match MATLAB)
            neur_vol.flat[neur_idx2] = neur_params['nuc_fluorsc']

    return neur_soma, neur_vol, gp_nuc, gp_soma

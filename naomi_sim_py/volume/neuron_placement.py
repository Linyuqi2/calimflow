"""
Neuron placement with rejection sampling.

This module implements the placement of neurons in a volume using rejection
sampling to avoid overlaps with vasculature and maintain minimum distances.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, List, Dict, Any, Optional
from .sphere_sampling import spiral_sample_sphere
from .neural_body import generate_neural_body


def sample_dense_neurons(neur_params: Dict[str, Any],
                        vol_params: Dict[str, Any],
                        neur_ves: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Sample neuron shapes and locations in a volume.

    This function replicates the MATLAB sampleDenseNeurons.m functionality,
    using rejection sampling to place neurons while avoiding vasculature.

    Args:
        neur_params: Neuron parameters
        vol_params: Volume parameters
        neur_ves: 3D boolean array of vessel locations

    Returns:
        Tuple of (neur_locs, Vcell, Vnuc, Tri, rotAng):
        - neur_locs: Nx3 array of neuron locations
        - Vcell: List of N soma surface meshes
        - Vnuc: List of N nucleus surface meshes
        - Tri: Triangulation for surface meshes
        - rotAng: Nx3 array of rotation angles
    """
    # Initialize outputs
    Vcell = []
    Vnuc = []
    rotAng = []

    # Setup volume parameters
    vol_sz = vol_params['vol_sz']
    vres = vol_params['vres']

    # Create volume grid for sampling
    vol_depth = vol_params['vol_depth'] * vres

    # Create morphological structuring element for vessel dilation
    x, y, z = np.mgrid[-np.ceil(vol_params['min_dist']/2):np.ceil(vol_params['min_dist']/2)+1,
                       -np.ceil(vol_params['min_dist']/2):np.ceil(vol_params['min_dist']/2)+1,
                       -np.ceil(vol_params['min_dist']/2):np.ceil(vol_params['min_dist']/2)+1]
    se = np.sqrt(x**2 + y**2 + z**2) <= vol_params['min_dist']/2

    # Dilate vessels to create exclusion zones
    neur_ves_trunc = ndimage.binary_dilation(neur_ves, se)

    # Isolate valid region (within volume and not in vessels)
    # MATLAB: idx_good = neur_ves_trunc(:,: ,1+vol_depth:vol_depth+vol_sz(3)*vol_params.vres);
    vol_depth_int = int(vol_depth)
    vol_slice_start = vol_depth_int + 1  # MATLAB uses 1-based indexing
    vol_slice_end = vol_depth_int + int(vol_sz[2] * vres)
    vol_slice_end = min(neur_ves_trunc.shape[2], vol_slice_end)

    # MATLAB indexing is 1-based, but we're working with 0-based numpy arrays
    valid_region = neur_ves_trunc[:, :, vol_depth_int:vol_slice_end]

    # MATLAB: idx_good = ~idx_good;
    idx_good = ~valid_region
    idx_bad = idx_good.copy()  # MATLAB: idx_bad = idx_good  # MATLAB: idx_bad = idx_good

    # Get spherical sampling for all neurons
    V, Tri = spiral_sample_sphere(neur_params['n_samps'], False)
    neur_params['S_samp'] = V
    neur_params['Tri'] = Tri

    # Create volume grid for candidate point generation (same size as idx_good)
    # MATLAB: [mesh_x, mesh_y, mesh_z] = meshgrid(single(linspace(0,vol_sz(1),vol_sz(1)*vol_params.vres)), ...)
    grid_shape = idx_good.shape
    x_coords = np.linspace(0, vol_sz[0], grid_shape[0], dtype=np.float32)
    y_coords = np.linspace(0, vol_sz[1], grid_shape[1], dtype=np.float32)
    z_coords = np.linspace(0, vol_sz[2], grid_shape[2], dtype=np.float32)

    mesh_x, mesh_y, mesh_z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    # MATLAB: mesh_x = permute(mesh_x,[2 1 3]); mesh_y = permute(mesh_y,[2 1 3]); mesh_z = permute(mesh_z,[2 1 3]);
    # This rotates the mesh to match volume dimensions
    mesh_x = np.transpose(mesh_x, (1, 0, 2))
    mesh_y = np.transpose(mesh_y, (1, 0, 2))
    mesh_z = np.transpose(mesh_z, (1, 0, 2))

    # Initialize neuron locations
    neur_locs = np.array([[np.inf, np.inf, np.inf]])
    kk = 0
    eta = 1.1  # Distance scaling factor

    # Sample neurons
    # MATLAB: while (sum(idx_good(:))>1)&&(size(Vcell,3)-isempty(Vcell)<vol_params.N_neur)
    while (np.sum(idx_good) > 1 and
           len(Vcell) < vol_params['N_neur']):

        kk += 1

        # Generate neuron shape (MATLAB does this before trying to place)
        V_tmp, Vnuc_tmp, _, rotAng_tmp = generate_neural_body(neur_params)

        Vcell.append(V_tmp)
        Vnuc.append(Vnuc_tmp)
        rotAng.append(rotAng_tmp)

        # Sample location
        if np.sum(idx_good) > 0:
            # Pick random valid location
            flat_good = idx_good.flatten()
            good_indices = np.where(flat_good)[0]
            selected_idx = np.random.choice(good_indices)
            # MATLAB: mesh_x is y-coords, mesh_y is x-coords after permute
            new_pt = np.array([mesh_y.flat[selected_idx],  # x coordinate
                              mesh_x.flat[selected_idx],  # y coordinate
                              mesh_z.flat[selected_idx]]) # z coordinate
        else:
            # Fallback to any available location
            flat_bad = idx_bad.flatten()
            bad_indices = np.where(flat_bad)[0]
            selected_idx = np.random.choice(bad_indices)
            # MATLAB: mesh_x is y-coords, mesh_y is x-coords after permute
            new_pt = np.array([mesh_y.flat[selected_idx],  # x coordinate
                              mesh_x.flat[selected_idx],  # y coordinate
                              mesh_z.flat[selected_idx]]) # z coordinate

        # Handle single neuron case
        if vol_params['N_neur'] == 1:
            new_pt = np.array(vol_sz) / 2

        # Check distance to existing neurons
        if len(neur_locs) > 1:
            distances = np.sqrt(np.sum((new_pt - neur_locs[:-1])**2, axis=1))
            tmp_dist = np.min(distances) if len(distances) > 0 else np.inf
        else:
            tmp_dist = np.inf

        # Add to location list
        neur_locs = np.vstack([neur_locs, new_pt])

        # Update exclusion zones
        # MATLAB: mesh_x is actually y-coords, mesh_y is x-coords after permute
        distances_to_new = np.sqrt((mesh_y - new_pt[0])**2 +  # mesh_y is x-coords
                                  (mesh_x - new_pt[1])**2 +  # mesh_x is y-coords
                                  (mesh_z - new_pt[2])**2)

        # Exclude points too close to current neuron
        idx_good[distances_to_new <= eta * vol_params['min_dist']] = False
        idx_bad[distances_to_new <= vol_params['min_dist']] = False

        # MATLAB: idx_good = ~(idx_good|~idx_bad);
        # This ensures idx_good only contains positions that are both not too close to vessels
        # and not too close to other neurons
        idx_good = ~(idx_good | ~idx_bad)

    # Clean up neuron locations (remove initialization row)
    neur_locs = neur_locs[1:]

    # Adjust neuron count if we couldn't fit all requested neurons
    actual_N_neur = len(Vcell)

    # Shift neuron meshes to their final locations
    neur_locs_single = neur_locs.astype(float)
    for i in range(actual_N_neur):
        Vcell[i] = Vcell[i] + neur_locs_single[i]
        Vnuc[i] = Vnuc[i] + neur_locs_single[i]

    # Convert to numpy arrays
    Vcell_out = []
    Vnuc_out = []
    for i in range(actual_N_neur):
        Vcell_out.append(Vcell[i].astype(np.float32))
        Vnuc_out.append(Vnuc[i].astype(np.float32))

    rotAng_out = np.array(rotAng).astype(np.float32)

    return neur_locs_single, Vcell_out, Vnuc_out, Tri, rotAng_out

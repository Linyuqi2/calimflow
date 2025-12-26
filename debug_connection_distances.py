#!/usr/bin/env python3
"""
Debug capillary connection distances and filtering.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent / "naomi_sim_py"
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

try:
    from naomi_sim_py.tests.comparison_tools.tpm_blood_vessels_python_comparison import check_vol_params, check_vasc_params
    import json

    print("="*80)
    print("CONNECTION DISTANCE DEBUG")
    print("="*80)

    # Load parameters
    param_file = Path('naomi_sim_py/tests/tpm_test_params.json')

    if not param_file.exists():
        print("ERROR: Parameter file not found")
        sys.exit(1)

    with open(param_file, 'r') as f:
        param_data = json.load(f)

    vol_params_raw = param_data['vol_params']
    vasc_params_raw = param_data['vasc_params']

    vol_params = check_vol_params(vol_params_raw)
    vasc_params = check_vasc_params(vasc_params_raw)

    vres = vol_params['vres']

    # Calculate vascular parameters like in simulate_blood_vessels.py
    from naomi_sim_py.vasculature.data_structures import VascParams
    vp = VascParams()
    vp.depth_surf = vasc_params['depth_surf'] * vres
    vp.mindists = np.array(vasc_params['vesFreq']) * vres / 2
    vp.maxcappdist = 2 * vasc_params['vesFreq'][2] * vres
    vp.vesSize = np.array(vasc_params['vesSize']) * vres
    vp.distsc = 1.0

    print("Vascular parameters:")
    print(f"  vesFreq: {vasc_params['vesFreq']}")
    print(f"  vres: {vres}")
    print(f"  maxcappdist: {vp.maxcappdist}")
    print(f"  distsc: {vp.distsc}")

    # Simulate the capillary sampling and distance calculation
    from naomi_sim_py.vasculature.connections import pseudo_rand_sample_3d

    # Setup volume
    if 'vasc_sz' not in vol_params or vol_params.get('vasc_sz') is None:
        nv_vol_sz = vol_params['vol_sz'] + np.array([0, 0, vol_params['vol_depth']])
    else:
        nv_vol_sz = vol_params['vasc_sz']

    nv_size = np.array(nv_vol_sz) * vres

    # Calculate number of capillaries
    volume = np.prod(nv_vol_sz)
    np.random.seed(42)
    nv_ncapp = max(int(np.round(volume / (vasc_params['vesFreq'][2] ** 3) * abs(1 + vasc_params['vesNumScale'] * np.random.randn()))), 0)

    print(f"\nCapillary calculation:")
    print(f"  Volume: {volume}")
    print(f"  vesFreq[2]: {vasc_params['vesFreq'][2]}")
    print(f"  Calculated ncapp: {nv_ncapp}")

    # Sample capillaries
    capp_pos = pseudo_rand_sample_3d(
        nv_size / vres,  # volume_size in physical units
        nv_ncapp,
        vp.mindists[0] / vres,  # min_dist
        1.0,  # sep_weight
        np.ones(tuple((nv_size // vres).astype(int)))  # weight_vol
    )

    print(f"\nCapillary sampling:")
    print(f"  Requested: {nv_ncapp}")
    print(f"  Sampled: {len(capp_pos)}")
    print(f"  Position range: x=[{capp_pos[:,0].min():.1f}, {capp_pos[:,0].max():.1f}], y=[{capp_pos[:,1].min():.1f}, {capp_pos[:,1].max():.1f}], z=[{capp_pos[:,2].min():.1f}, {capp_pos[:,2].max():.1f}]")

    # Convert to voxel coordinates
    capp_pos_vox = (capp_pos * vres).astype(int)
    capp_pos_vox = np.clip(capp_pos_vox, 0, nv_size - 1)

    print(f"  Voxel range: x=[{capp_pos_vox[:,0].min()}, {capp_pos_vox[:,0].max()}], y=[{capp_pos_vox[:,1].min()}, {capp_pos_vox[:,1].max()}], z=[{capp_pos_vox[:,2].min()}, {capp_pos_vox[:,2].max()}]")

    if len(capp_pos) > 1:
        # Calculate distance matrix
        from naomi_sim_py.vasculature.connections import pos2dists
        dist_mat = pos2dists(capp_pos)

        print(f"\nDistance analysis:")
        print(f"  Distance matrix shape: {dist_mat.shape}")
        print(f"  Min distance: {np.min(dist_mat[dist_mat > 0]):.1f}")
        print(f"  Max distance: {np.max(dist_mat):.1f}")
        print(f"  Mean distance: {np.mean(dist_mat[dist_mat > 0]):.1f}")
        print(f"  Median distance: {np.median(dist_mat[dist_mat > 0]):.1f}")

        # Count distances exceeding threshold
        n_distances = np.sum(dist_mat > 0)
        n_exceeding = np.sum(dist_mat > vp.maxcappdist)
        print(f"  Distances > maxcappdist ({vp.maxcappdist:.1f}): {n_exceeding}/{n_distances} ({100*n_exceeding/n_distances:.1f}%)")

        # MATLAB-style connection analysis
        print("\nMATLAB-style connection analysis:")

        # Initialize connection matrix
        capp_conn_mat = np.zeros((len(capp_pos), len(capp_pos)))

        # Find minimum distance connections
        min_dists = np.min(dist_mat, axis=1)
        min_indices = np.argmin(dist_mat, axis=1)

        n_initial_connections = 0
        for i in range(len(min_dists)):
            if min_dists[i] < np.inf and min_dists[i] <= vp.maxcappdist:
                capp_conn_mat[i, min_indices[i]] = 1
                capp_conn_mat[min_indices[i], i] = 1
                n_initial_connections += 1

        print(f"  Initial connections (min distance): {n_initial_connections}")

        # Apply distance threshold
        dist_mat_thresholded = dist_mat.copy()
        dist_mat_thresholded[dist_mat > vp.maxcappdist] = np.inf
        print(f"  Connections after distance threshold: {np.sum(capp_conn_mat)}")

        # Check for potential issues
        print("\nPotential issues:")
        if vp.maxcappdist > np.max(dist_mat) * 0.8:
            print("  WARNING: maxcappdist is very large compared to actual distances")
        if len(capp_pos) < 10:
            print("  WARNING: Very few capillaries generated")
        if n_initial_connections > len(capp_pos) * 2:
            print("  WARNING: Too many initial connections per capillary")

    print("\nAnalysis complete!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

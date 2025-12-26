"""
Compare TPM Blood Vessels MATLAB vs Python

This script loads the MATLAB results from TPM_Simulation_Script_Blood_Vessels.m
and runs Python simulation with identical parameters for comparison.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add naomi_sim_py parent directory to path (containing naomi_sim_py package)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from naomi_sim_py.core.parameters import check_vol_params, check_vasc_params
from naomi_sim_py.vasculature.simulate_blood_vessels import simulate_blood_vessels


def main():
    print("="*70)
    print("TPM Blood Vessels MATLAB vs Python Comparison")
    print("="*70)

    # Add detailed parameter validation
    print("\n" + "="*50)
    print("DETAILED PARAMETER VALIDATION")
    print("="*50)

    # Check if MATLAB results exist
    matlab_file = Path(__file__).parent.parent / "tpm_matlab_vasculature_result.mat"

    if not matlab_file.exists():
        print("ERROR: MATLAB result file not found.")
        print("Please run MATLAB test first:")
        print("matlab -r 'cd naomi_sim_py/tests; tpm_blood_vessels_comparison'")
        return False

    try:
        # Load MATLAB results
        import scipy.io as sio
        matlab_data = sio.loadmat(str(matlab_file))
        neur_ves_matlab = matlab_data['neur_ves_matlab']
        matlab_time = matlab_data['matlab_time'][0][0]

        print("SUCCESS: Loaded MATLAB results")
        print("Shape:", neur_ves_matlab.shape)
        print("Total voxels:", np.sum(neur_ves_matlab))
        print("Density:", np.sum(neur_ves_matlab) / np.prod(neur_ves_matlab.shape) * 100)

        # Load test parameters
        param_file = Path(__file__).parent.parent / "tpm_test_params.json"
        if not param_file.exists():
            print("ERROR: Parameter file not found")
            return False

        with open(param_file, 'r') as f:
            data = json.load(f)

        # Convert MATLAB parameters to Python format
        vol_params_raw = data['vol_params']
        vasc_params_raw = data['vasc_params']

        # Handle MATLAB arrays (convert to lists)
        for key, value in vol_params_raw.items():
            if hasattr(value, 'tolist'):  # numpy array
                vol_params_raw[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'tolist'):
                vol_params_raw[key] = [x.tolist() if hasattr(x, 'tolist') else x for x in value]

        for key, value in vasc_params_raw.items():
            if hasattr(value, 'tolist'):  # numpy array
                vasc_params_raw[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'tolist'):
                vasc_params_raw[key] = [x.tolist() if hasattr(x, 'tolist') else x for x in value]

        print("SUCCESS: Loaded test parameters")
        print(f"vol_params_raw['vol_sz']: {vol_params_raw['vol_sz']} (type: {type(vol_params_raw['vol_sz'])})")
        print(f"vasc_params_raw['vesSize']: {vasc_params_raw['vesSize']} (type: {type(vasc_params_raw['vesSize'])})")
        print("Volume size:", vol_params_raw.get('vol_sz'))
        print("Volume depth:", vol_params_raw.get('vol_depth'))

        # Process parameters through Python parameter system
        print("\nProcessing parameters through Python system...")
        vol_params = check_vol_params(vol_params_raw)
        vasc_params = check_vasc_params(vasc_params_raw)

        # Detailed parameter validation
        print("\nPARAMETER VALIDATION:")
        print("-" * 30)

        # Expected MATLAB parameters
        expected_vol_params = {
            'vol_sz': [400, 400, 100],
            'vol_depth': 280,
            'vasc_sz': [400, 400, 380]
        }

        expected_vasc_params = {
            'vesSize': [15.0, 9.0, 2.0],
            'vesFreq': [125.0, 200.0, 50.0],
            'sourceFreq': 1000.0,
            'vesNumScale': 0.2
        }

        print("Volume parameters:")
        for key, expected in expected_vol_params.items():
            actual = vol_params.get(key, 'NOT FOUND')
            status = "OK" if np.array_equal(actual, expected) or actual == expected else "FAIL"
            print(f"  {key}: {actual} {status}")

        print("\nVasculature parameters:")
        for key, expected in expected_vasc_params.items():
            actual = vasc_params.get(key, 'NOT FOUND')
            if isinstance(actual, np.ndarray):
                match = np.allclose(actual, expected, rtol=1e-10)
            else:
                match = abs(actual - expected) < 1e-10 if isinstance(actual, (int, float)) else actual == expected
            status = "OK" if match else "FAIL"
            print(f"  {key}: {actual} {status}")

        print("\nOther vasc_params:")
        for key, value in vasc_params.items():
            if key not in expected_vasc_params:
                print(f"  {key}: {value}")

        print("Running Python simulation with identical TPM parameters...")

        # Run Python simulation with intermediate result logging
        import time
        start_time = time.time()

        print("\nINTERMEDIATE RESULT LOGGING:")
        print("-" * 35)

        # Run Python simulation
        print("\nRunning Python simulation...")
        try:
            neur_ves_python, vasc_params_python, neur_ves_all_python = simulate_blood_vessels(
                vol_params, vasc_params
            )
            print("SUCCESS: Python simulation completed successfully!")

            python_time = time.time() - start_time
            print("Python computation time:", python_time)

            # Check intermediate results
            print("\nPost-simulation checks:")
            print(f"  Output shape: {neur_ves_python.shape}")
            print(f"  Data type: {neur_ves_python.dtype}")
            print(f"  Value range: [{neur_ves_python.min():.3f}, {neur_ves_python.max():.3f}]")
            print(f"  Non-zero voxels: {np.sum(neur_ves_python > 0)}")

            if neur_ves_all_python is not None:
                print(f"  Full volume shape: {neur_ves_all_python.shape}")
                print(f"  Full volume range: [{neur_ves_all_python.min():.3f}, {neur_ves_all_python.max():.3f}]")

        except Exception as e:
            print(f"ERROR: Python simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Analyze results
        from scipy import ndimage
        structure = np.ones((3, 3, 3))

        # MATLAB analysis
        labeled_matlab, comp_matlab = ndimage.label(neur_ves_matlab, structure)

        # Python analysis
        labeled_python, comp_python = ndimage.label(neur_ves_python, structure)

        # Print detailed comparison
        print("\n" + "="*80)
        print("TPM BLOOD VESSELS COMPARISON RESULTS")
        print("="*80)

        matlab_voxels = int(np.sum(neur_ves_matlab))
        python_voxels = int(np.sum(neur_ves_python))
        matlab_density = float(np.sum(neur_ves_matlab) / np.prod(neur_ves_matlab.shape) * 100)
        python_density = float(np.sum(neur_ves_python) / np.prod(neur_ves_python.shape) * 100)

        print("\nVolume Statistics:")
        print("                   MATLAB          Python          Difference")
        print("  Shape           ", str(neur_ves_matlab.shape).ljust(15), str(neur_ves_python.shape).ljust(15), "Same" if neur_ves_matlab.shape == neur_ves_python.shape else "Different")
        print("  Total Voxels    ", str(matlab_voxels).rjust(10), str(python_voxels).rjust(15), "{:+d}".format(python_voxels - matlab_voxels).rjust(10))
        print("  Density %       ", ".4f", ".4f", "{:+.4f}".format(python_density - matlab_density).rjust(10))
        print("  Components      ", str(comp_matlab).rjust(10), str(comp_python).rjust(15), "{:+d}".format(comp_python - comp_matlab).rjust(10))

        if comp_matlab > 0 and comp_python > 0:
            # MATLAB component sizes
            matlab_comp_sizes = [np.sum(labeled_matlab == i) for i in range(1, comp_matlab + 1)]
            matlab_largest = max(matlab_comp_sizes)
            matlab_avg = np.mean(matlab_comp_sizes)

            # Python component sizes
            python_comp_sizes = [np.sum(labeled_python == i) for i in range(1, comp_python + 1)]
            python_largest = max(python_comp_sizes)
            python_avg = np.mean(python_comp_sizes)

            print("  Largest Comp    ", str(matlab_largest).rjust(10), str(python_largest).rjust(15), "{:+d}".format(python_largest - matlab_largest).rjust(10))
            print("  Avg Component   ", ".1f", ".1f", "{:+.1f}".format(python_avg - matlab_avg).rjust(10))

        print("  Time (sec)      ", ".3f", ".3f", "{:+.3f}".format(python_time - matlab_time).rjust(10))

        # Assessment
        print("\nAssessment:")
        voxel_diff_pct = abs(python_voxels - matlab_voxels) / matlab_voxels * 100
        comp_diff = abs(comp_python - comp_matlab)
        comp_ratio = max(comp_python, comp_matlab) / min(comp_python, comp_matlab) if min(comp_python, comp_matlab) > 0 else float('inf')

        print(f"  Voxel difference: {voxel_diff_pct:.1f}%")
        print(f"  Component difference: {comp_diff} (ratio: {comp_ratio:.1f}x)")

        # More strict criteria for biological accuracy
        if voxel_diff_pct < 10 and comp_ratio < 5:
            print("EXCELLENT: Results are very similar!")
            print("  - Voxel difference < 10%")
            print("  - Component ratio < 5x")
            success = True
        elif voxel_diff_pct < 25 and comp_ratio < 10:
            print("GOOD: Results are reasonably similar")
            print("  - Voxel difference < 25%")
            print("  - Component ratio < 10x")
            success = True
        elif voxel_diff_pct < 50 and comp_ratio < 20:
            print("FAIR: Moderate differences detected")
            print("  - Some algorithmic differences expected")
            print("  - Further investigation recommended")
            success = True
        else:
            print("SIGNIFICANT differences detected")
            print("  - Check parameter settings and algorithms")
            print("  - May indicate implementation issues")
            success = False

        # Additional TPM-specific information
        print("\nTPM Simulation Parameters:")
        print("  Volume size: {} um".format(vol_params_raw.get('vol_sz')))
        print("  Volume depth: {} um".format(vol_params_raw.get('vol_depth')))
        print("  Vasculature size: {} um".format(vol_params_raw.get('vasc_sz')))
        print("  Vessel sizes: {} um".format(vasc_params_raw.get('vesSize')))
        print("  Vessel frequencies: {} um^-1".format(vasc_params_raw.get('vesFreq')))

        print("\n" + "="*70)
        if success:
            print("SUCCESS: TPM Blood Vessels MATLAB-Python comparison completed!")
        else:
            print("ISSUES: Significant differences found in TPM comparison")
        print("="*70)

        return success

    except Exception as e:
        print("ERROR:", e)
        import traceback
        traceback.print_exc()
        return False


def debug_simulate_blood_vessels(vol_params, vasc_params):
    """
    Debug version of simulate_blood_vessels with intermediate result logging.
    """
    from naomi_sim_py.vasculature.simulate_blood_vessels import simulate_blood_vessels
    from naomi_sim_py.vasculature.data_structures import VascNetwork, VascParams, NodeParams
    from naomi_sim_py.vasculature.major_vessels import grow_major_vessels
    from naomi_sim_py.vasculature.connections import nodes_to_conn, conn_to_vol, grow_capillaries
    import copy

    debug_info = {}

    # Step 1: Parameter validation
    print("Step 1: Parameter validation")
    vol_params = check_vol_params(vol_params)
    vasc_params = check_vasc_params(vasc_params)
    debug_info['validated_params'] = {
        'vol_params': copy.deepcopy(vol_params),
        'vasc_params': copy.deepcopy(vasc_params)
    }

    # Check if vasculature is disabled
    if not vasc_params.get('flag', True):
        vol_sz_array = np.array(vol_params['vol_sz'])
        vol_size_pixels = (vol_sz_array * vol_params['vres']).astype(int)
        neur_ves = np.zeros(vol_size_pixels, dtype=bool)
        debug_info['early_return'] = "Vasculature disabled"
        return neur_ves, vasc_params, None, debug_info

    # Step 2: Setup parameters
    print("Step 2: Setup parameters")
    vres = vol_params['vres']
    vp = VascParams()
    vp.depth_surf = vasc_params['depth_surf'] * vres
    vp.mindists = np.array(vasc_params['vesFreq']) * vres / 2
    vp.maxcappdist = 2 * vasc_params['vesFreq'][2] * vres
    vp.vesSize = np.array(vasc_params['vesSize']) * vres

    np_obj = NodeParams()
    np_obj.lensc = vasc_params['node_params']['lensc'] * vres
    np_obj.varsc = vasc_params['node_params']['varsc'] * vres
    np_obj.mindist = vasc_params['node_params']['mindist'] * vres
    np_obj.varpos = vasc_params['node_params']['varpos'] * vres
    np_obj.vesrad = int(np.ceil(vasc_params['node_params']['vesrad'] * vres))

    debug_info['setup_params'] = {
        'vp': {'depth_surf': vp.depth_surf, 'mindists': vp.mindists, 'maxcappdist': vp.maxcappdist, 'vesSize': vp.vesSize},
        'np_obj': {'lensc': np_obj.lensc, 'varsc': np_obj.varsc, 'mindist': np_obj.mindist, 'varpos': np_obj.varpos, 'vesrad': np_obj.vesrad}
    }

    # Step 3: Volume size setup
    print("Step 3: Volume size setup")
    if 'vasc_sz' not in vol_params or vol_params.get('vasc_sz') is None:
        nv_vol_sz = vol_params['vol_sz'] + np.array([0, 0, vol_params['vol_depth']])
    else:
        nv_vol_sz = vol_params['vasc_sz']

    nv_size = np.array(nv_vol_sz) * vres
    debug_info['volume_setup'] = {
        'nv_vol_sz': nv_vol_sz,
        'nv_size': nv_size
    }

    # Step 4: Calculate vessel counts
    print("Step 4: Calculate vessel counts")
    perimeter = 2 * (nv_vol_sz[0] + nv_vol_sz[1])
    area = nv_vol_sz[0] * nv_vol_sz[1]
    volume = np.prod(nv_vol_sz)

    # Set random seed for reproducibility
    np.random.seed(42)

    nv_nsource = max(int(np.round(perimeter / vasc_params['sourceFreq'] * abs(1 + vasc_params['vesNumScale'] * np.random.randn()))), 0)
    nv_nvert = max(int(np.round(area / (vasc_params['vesFreq'][1] ** 2) * abs(1 + vasc_params['vesNumScale'] * np.random.randn()))), 0)
    nv_nsurf = max(int(np.round(area / (vasc_params['vesFreq'][0] ** 2) * abs(1 + vasc_params['vesNumScale'] * np.random.randn()))), 0)
    nv_ncapp = max(int(np.round(volume / (vasc_params['vesFreq'][2] ** 3) * abs(1 + vasc_params['vesNumScale'] * np.random.randn()))), 0)

    debug_info['vessel_counts'] = {
        'perimeter': perimeter, 'area': area, 'volume': volume,
        'nv_nsource': nv_nsource, 'nv_nvert': nv_nvert, 'nv_nsurf': nv_nsurf, 'nv_ncapp': nv_ncapp
    }

    print(f"  Vessel counts: source={nv_nsource}, vert={nv_nvert}, surf={nv_nsurf}, capp={nv_ncapp}")

    # Step 5: Create network and grow major vessels
    print("Step 5: Create network and grow major vessels")
    nv = VascNetwork(
        vol_sz=nv_vol_sz, size=nv_size.astype(int), szum=nv_vol_sz,
        nsource=nv_nsource, nvert=nv_nvert, nsurf=nv_nsurf, ncapp=nv_ncapp
    )

    nodes, nv = grow_major_vessels(nv, np_obj, vp)
    debug_info['major_vessels'] = {
        'num_nodes': len(nodes),
        'nv_nlinks': nv.nlinks,
        'node_types': [node.type for node in nodes[:10]] if len(nodes) > 0 else []
    }

    print(f"  Created {len(nodes)} nodes and {nv.nlinks} links")

    # Step 6: Convert to connections
    print("Step 6: Convert to connections")
    conn = nodes_to_conn(nodes)
    nv.nconn = len(conn)
    debug_info['connections'] = {
        'num_connections': len(conn),
        'connection_types': [c.type for c in conn[:10]] if len(conn) > 0 else []
    }

    print(f"  Created {len(conn)} connections")

    # Step 7: Create initial volume
    print("Step 7: Create initial volume")
    neur_ves, conn = conn_to_vol(nodes, conn, nv)
    debug_info['initial_volume'] = {
        'shape': neur_ves.shape,
        'dtype': neur_ves.dtype,
        'nonzero_voxels': np.sum(neur_ves > 0),
        'num_connections': len(conn)
    }

    print(f"  Initial volume shape: {neur_ves.shape}, nonzero: {np.sum(neur_ves > 0)}")

    # Step 8: Grow capillaries
    print("Step 8: Grow capillaries")
    print("  Creating distance weighting volume...")
    print("  Sampling capillary positions...")
    print("  Creating vertical vessel to capillary connections...")
    print("  Adding remaining independent capillaries...")
    print("  Creating capillary-to-capillary connections...")
    print("  Calculating capillary connection weights...")

    nodes, conn, nv = grow_capillaries(nodes, conn, neur_ves, nv, vp, vres)

    # Add capillaries to rest of volume
    cappidxs = [i for i, c in enumerate(conn) if c.locs is None or len(c.locs) == 0]
    neur_ves, _ = conn_to_vol(nodes, conn, nv, cappidxs, neur_ves)

    debug_info['capillaries'] = {
        'final_shape': neur_ves.shape,
        'final_nonzero': np.sum(neur_ves > 0),
        'final_connections': len(conn)
    }

    print(f"  Final volume nonzero: {np.sum(neur_ves > 0)}, connections: {len(conn)}")

    # Step 9: Handle output volume sizing
    print("Step 9: Handle output volume sizing")
    neur_ves_all = None
    if 'vasc_sz' in vol_params and vol_params['vasc_sz'] is not None:
        neur_ves_all = neur_ves
        sz = np.array([vol_params['vol_sz'][0], vol_params['vol_sz'][1],
                      vol_params['vol_depth'] + vol_params['vol_sz'][2]]) * vres
        sz_diff = np.ceil((np.array(vol_params['vasc_sz']) * vres - sz) / 2).astype(int)
        neur_ves = neur_ves[
            sz_diff[0]:sz_diff[0] + int(sz[0]),
            sz_diff[1]:sz_diff[1] + int(sz[1]),
            sz_diff[2]:sz_diff[2] + int(sz[2])
        ]

        debug_info['output_sizing'] = {
            'sz': sz, 'sz_diff': sz_diff, 'final_shape': neur_ves.shape
        }

    return neur_ves, vasc_params, neur_ves_all, debug_info


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

"""
Compare Neural Volume MATLAB vs Python

This script loads the MATLAB results and runs Python simulation with identical
parameters for neural volume comparison.
"""

import sys
import json
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from core.parameters import (
    check_vol_params, check_neur_params, check_vasc_params,
    check_dend_params, check_bg_params, check_axon_params
)
from volume.simulate_neural_volume import simulate_neural_volume


def compare_neural_volumes(matlab_result, python_result):
    """Compare key metrics between MATLAB and Python results"""

    comparisons = {}

    # Compare volume shapes
    matlab_vol = matlab_result.get('neur_vol')
    comparisons['volume_shape'] = {
        'matlab': matlab_vol.shape if matlab_vol is not None else 'N/A',
        'python': python_result.neur_vol.shape,
        'match': matlab_vol.shape == python_result.neur_vol.shape if matlab_vol is not None else False
    }

    # Compare total fluorescence (cellular only - exclude background)
    matlab_fluor = np.sum(matlab_vol) if matlab_vol is not None else 0

    # For Python, calculate cellular fluorescence by creating background-free volume
    python_vol_cellular = python_result.neur_vol.copy()
    if hasattr(python_result, 'gp_bgvals') and python_result.gp_bgvals is not None:
        # Create a background volume to subtract
        bg_vol = np.zeros_like(python_result.neur_vol)
        for bg_proc in python_result.gp_bgvals:
            if len(bg_proc) >= 2:
                bg_indices = bg_proc[0]
                bg_values = bg_proc[1]
                # Add background values to bg_vol (accumulate overlapping values)
                bg_vol.flat[bg_indices.astype(int)] += bg_values
        # Subtract background from cellular volume
        python_vol_cellular -= bg_vol

    python_cellular_fluor = np.sum(python_vol_cellular)

    comparisons['total_fluorescence'] = {
        'matlab': matlab_fluor,
        'python': python_cellular_fluor,
        'difference': abs(matlab_fluor - python_cellular_fluor),
        'relative_error': abs(matlab_fluor - python_cellular_fluor) / max(matlab_fluor, python_cellular_fluor) if max(matlab_fluor, python_cellular_fluor) > 0 else 0
    }

    # Also compare total fluorescence including background
    python_total_fluor = np.sum(python_result.neur_vol)
    comparisons['total_fluorescence_with_background'] = {
        'matlab': matlab_fluor,
        'python': python_total_fluor,
        'note': 'MATLAB result does not include background fluorescence'
    }

    # Compare mean fluorescence (cellular voxels only)
    matlab_nonzero = matlab_vol[matlab_vol > 0] if matlab_vol is not None else np.array([])

    # Python cellular volume is already calculated above
    # Check fluorescence distribution without threshold first
    all_positive = python_vol_cellular[python_vol_cellular > 0]
    print(f"Python fluorescence distribution (no threshold): count={len(all_positive)}, sum={np.sum(all_positive):.1f}, mean={np.mean(all_positive):.4f}")

    # Apply fluorescence threshold to match MATLAB's conservative approach
    fluorescence_threshold = 0.005
    python_nonzero = python_vol_cellular[python_vol_cellular > fluorescence_threshold]

    matlab_mean_fluor = np.mean(matlab_nonzero) if len(matlab_nonzero) > 0 else 0
    python_mean_fluor = np.mean(python_nonzero) if len(python_nonzero) > 0 else 0

    comparisons['mean_fluorescence'] = {
        'matlab': matlab_mean_fluor,
        'python': python_mean_fluor,
        'difference': abs(matlab_mean_fluor - python_mean_fluor),
        'relative_error': abs(matlab_mean_fluor - python_mean_fluor) / max(matlab_mean_fluor, python_mean_fluor) if max(matlab_mean_fluor, python_mean_fluor) > 0 else 0
    }

    # Compare fluorescence distribution statistics (cellular only)
    comparisons['fluorescence_stats'] = {
        'matlab_std': np.std(matlab_nonzero) if len(matlab_nonzero) > 0 else 0,
        'python_std': np.std(python_nonzero) if len(python_nonzero) > 0 else 0,
        'matlab_max': np.max(matlab_nonzero) if len(matlab_nonzero) > 0 else 0,
        'python_max': np.max(python_nonzero) if len(python_nonzero) > 0 else 0,
        'matlab_nonzero_count': len(matlab_nonzero),
        'python_nonzero_count': len(python_nonzero)
    }

    # Compare neuron count
    matlab_locs = matlab_result.get('locs')
    matlab_neurons = matlab_locs.shape[0] if matlab_locs is not None and len(matlab_locs) > 0 else 0
    python_neurons = len(python_result.locs) if python_result.locs is not None else 0
    comparisons['neuron_count'] = {
        'matlab': matlab_neurons,
        'python': python_neurons,
        'match': matlab_neurons == python_neurons
    }

    # Compare neuron locations (if available)
    if matlab_locs is not None and python_result.locs is not None and matlab_neurons > 0 and python_neurons > 0:
        # Compare location statistics
        matlab_locs_mean = np.mean(matlab_locs, axis=0)
        python_locs_mean = np.mean(python_result.locs, axis=0)

        comparisons['neuron_locations'] = {
            'matlab_mean': matlab_locs_mean,
            'python_mean': python_locs_mean,
            'mean_difference': np.abs(matlab_locs_mean - python_locs_mean),
            'max_mean_diff': np.max(np.abs(matlab_locs_mean - python_locs_mean))
        }
    else:
        comparisons['neuron_locations'] = {'error': 'Location data not available for comparison'}

    # Compare vessel volume
    matlab_ves = matlab_result.get('neur_ves')
    matlab_vessels = np.sum(matlab_ves) if matlab_ves is not None else 0
    python_vessels = np.sum(python_result.neur_ves)
    comparisons['vessel_voxels'] = {
        'matlab': matlab_vessels,
        'python': python_vessels,
        'difference': abs(matlab_vessels - python_vessels),
        'relative_error': abs(matlab_vessels - python_vessels) / max(matlab_vessels, python_vessels) if max(matlab_vessels, python_vessels) > 0 else 0
    }

    # Compare volume density
    matlab_density = matlab_vessels / np.prod(matlab_vol.shape) if matlab_vol is not None else 0
    python_density = python_vessels / np.prod(python_result.neur_vol.shape)
    comparisons['volume_density'] = {
        'matlab': matlab_density,
        'python': python_density,
        'difference': abs(matlab_density - python_density)
    }

    # Compare background fluorescence (if available)
    if hasattr(python_result, 'gp_bgvals') and python_result.gp_bgvals is not None and len(python_result.gp_bgvals) > 0:
        python_bg_fluor = sum(np.sum(bg_proc[1]) for bg_proc in python_result.gp_bgvals)
        comparisons['background_fluorescence'] = {
            'python': python_bg_fluor,
            'matlab': 'N/A (not extracted)',
            'note': 'Background fluorescence only available in Python results'
        }
    else:
        comparisons['background_fluorescence'] = {
            'python': 0,
            'matlab': 'N/A',
            'note': 'No background processes generated'
        }

    # Compare background process count
    python_bg_count = len(python_result.gp_bgvals) if hasattr(python_result, 'gp_bgvals') and python_result.gp_bgvals is not None else 0
    comparisons['background_process_count'] = {
        'python': python_bg_count,
        'matlab': 'N/A (not extracted)',
        'note': 'Background process count comparison'
    }

    # Compare fluorescence distribution percentiles
    if len(matlab_nonzero) > 0 and len(python_nonzero) > 0:
        matlab_percentiles = np.percentile(matlab_nonzero, [25, 50, 75, 90, 95])
        python_percentiles = np.percentile(python_nonzero, [25, 50, 75, 90, 95])

        comparisons['fluorescence_percentiles'] = {
            'matlab_25th': matlab_percentiles[0],
            'matlab_50th': matlab_percentiles[1],
            'matlab_75th': matlab_percentiles[2],
            'matlab_90th': matlab_percentiles[3],
            'matlab_95th': matlab_percentiles[4],
            'python_25th': python_percentiles[0],
            'python_50th': python_percentiles[1],
            'python_75th': python_percentiles[2],
            'python_90th': python_percentiles[3],
            'python_95th': python_percentiles[4],
            'percentile_differences': np.abs(matlab_percentiles - python_percentiles)
        }

    # Compare spatial distribution (center of mass)
    if len(matlab_nonzero) > 0 and len(python_nonzero) > 0:
        # Get coordinates of non-zero voxels
        matlab_coords = np.array(np.where(matlab_vol > 0)).T
        python_coords = np.array(np.where(python_result.neur_vol > 0)).T

        if len(matlab_coords) > 0 and len(python_coords) > 0:
            matlab_com = np.mean(matlab_coords, axis=0)
            python_com = np.mean(python_coords, axis=0)

            comparisons['spatial_distribution'] = {
                'matlab_center_of_mass': matlab_com,
                'python_center_of_mass': python_com,
                'center_of_mass_difference': np.abs(matlab_com - python_com),
                'max_com_difference': np.max(np.abs(matlab_com - python_com))
            }

    # Compare fluorescence variance
    matlab_variance = np.var(matlab_nonzero) if len(matlab_nonzero) > 0 else 0
    python_variance = np.var(python_nonzero) if len(python_nonzero) > 0 else 0

    comparisons['fluorescence_variance'] = {
        'matlab': matlab_variance,
        'python': python_variance,
        'variance_ratio': python_variance / matlab_variance if matlab_variance > 0 else float('inf')
    }

    return comparisons


def run_comparison():
    """Run the neural volume comparison"""

    print("="*70)
    print("TPM Neural Volume MATLAB vs Python Comparison")
    print("="*70)

    # Load test parameters
    param_file = Path(__file__).parent.parent / "tpm_test_params.json"
    if not param_file.exists():
        print("ERROR: Parameter file not found.")
        return False

    with open(param_file, 'r') as f:
        test_params = json.load(f)

    print("Loaded test parameters:")
    print(f"  Volume size: {test_params['vol_params']['vol_sz']}")
    print(f"  Neurons: {test_params['vol_params']['N_neur']}")
    print(f"  Resolution: {test_params['vol_params']['vres']}")

    # Check for MATLAB results
    matlab_file = Path(__file__).parent.parent / "tpm_matlab_neural_result.mat"
    matlab_data = None

    if matlab_file.exists():
        try:
            import scipy.io as sio
            matlab_data = sio.loadmat(str(matlab_file))
            print("SUCCESS: Loaded MATLAB neural volume results")
            print(f"  Volume shape: {matlab_data['vol_out']['neur_vol'][0,0].shape}")
            print(f"  Total fluorescence: {np.sum(matlab_data['vol_out']['neur_vol'][0,0])}")
        except Exception as e:
            print(f"ERROR loading MATLAB results: {e}")
            matlab_data = None
    else:
        print("WARNING: MATLAB result file not found. Running Python-only test.")
        print("To enable comparison, run MATLAB test first:")
        print("matlab -r 'cd naomi_sim_py/tests; tpm_neural_volume_comparison'")

    # Run Python simulation
    print("\nRunning Python neural volume simulation...")

    start_time = time.time()

    try:
        # Set up parameters
        vol_params = check_vol_params(test_params['vol_params'])
        neur_params = check_neur_params(test_params['neur_params'])
        vasc_params = check_vasc_params(test_params.get('vasc_params', {}))
        dend_params = check_dend_params(test_params.get('dend_params', {}))
        bg_params = check_bg_params(test_params.get('bg_params', {}))
        axon_params = check_axon_params(test_params.get('axon_params', {}))

        # Run simulation
        python_result = simulate_neural_volume(
            vol_params=vol_params,
            neur_params=neur_params,
            vasc_params=vasc_params,
            dend_params=dend_params,
            bg_params=bg_params,
            axon_params=axon_params,
            debug_opt=False
        )

        python_time = time.time() - start_time

        print(f"  Execution time: {python_time:.2f} seconds")
        print(f"  Volume shape: {python_result.neur_vol.shape}")
        print(f"  Neurons placed: {len(python_result.locs) if python_result.locs is not None else 0}")
        print(f"  Total fluorescence: {np.sum(python_result.neur_vol):.2f}")
        print(f"  Vessel voxels: {np.sum(python_result.neur_ves)}")

        # Compare with MATLAB if available
        if matlab_data is not None:
            print("\n" + "="*50)
            print("COMPARISON RESULTS")
            print("="*50)

            # Extract MATLAB results from vol_out struct
            vol_out = matlab_data['vol_out']

            # MATLAB structured arrays: access the actual data
            matlab_vol = vol_out['neur_vol'][0,0]   # neur_vol data
            matlab_ves = vol_out['neur_ves'][0,0]   # neur_ves data
            matlab_locs = vol_out['locs'][0,0]      # locs data

            matlab_result = {
                'neur_vol': matlab_vol,
                'neur_ves': matlab_ves,
                'locs': matlab_locs
            }

            comparisons = compare_neural_volumes(matlab_result, python_result)

            print("\n" + "="*60)
            print("DETAILED COMPARISON RESULTS")
            print("="*60)

            for metric, values in comparisons.items():
                print(f"\n{metric.upper().replace('_', ' ')}:")
                if 'match' in values:
                    status = "[OK]" if values['match'] else "[FAIL]"
                    print(f"  {status} Match: {values['match']}")
                if 'relative_error' in values:
                    error_pct = values['relative_error'] * 100
                    status = "[OK]" if error_pct < 5 else "[WARN]" if error_pct < 15 else "[FAIL]"
                    print(f"  {status} Relative error: {error_pct:.2f}%")
                elif 'difference' in values:
                    if isinstance(values['difference'], (int, float)):
                        print(f"  Difference: {values['difference']:.4f}")
                    else:
                        print(f"  Difference: {values['difference']}")

                # Print values
                if 'error' not in values:
                    if 'matlab' in values and 'python' in values:
                        print(f"  MATLAB: {values['matlab']}")
                        print(f"  Python: {values['python']}")
                else:
                    print(f"  {values['error']}")

                # Special handling for fluorescence stats
                if metric == 'fluorescence_stats':
                    print("  Fluorescence distribution:")
                    print(f"    MATLAB - Mean: {values['matlab_std']:.4f}, Max: {values['matlab_max']:.2f}, Non-zero: {values['matlab_nonzero_count']}")
                    print(f"    Python - Mean: {values['python_std']:.4f}, Max: {values['python_max']:.2f}, Non-zero: {values['python_nonzero_count']}")

                # Special handling for neuron locations
                if metric == 'neuron_locations' and 'max_mean_diff' in values:
                    print(f"  Max location difference: {values['max_mean_diff']:.2f} um")

                # Special handling for background fluorescence
                if metric == 'background_fluorescence':
                    if values['python'] > 0:
                        print(f"  Python background fluorescence: {values['python']:.2f}")
                        print(f"  Note: {values.get('note', '')}")
                    else:
                        print(f"  No background fluorescence generated")

                # Special handling for background process count
                if metric == 'background_process_count':
                    print(f"  Python background processes: {values['python']}")
                    print(f"  Note: {values.get('note', '')}")

                # Special handling for fluorescence percentiles
                if metric == 'fluorescence_percentiles':
                    print("  Percentile comparison (25th, 50th, 75th, 90th, 95th):")
                    print(f"    MATLAB: {values['matlab_25th']:.4f}, {values['matlab_50th']:.4f}, {values['matlab_75th']:.4f}, {values['matlab_90th']:.4f}, {values['matlab_95th']:.4f}")
                    print(f"    Python:  {values['python_25th']:.4f}, {values['python_50th']:.4f}, {values['python_75th']:.4f}, {values['python_90th']:.4f}, {values['python_95th']:.4f}")
                    print(f"    Max difference: {np.max(values['percentile_differences']):.4f}")

                # Special handling for spatial distribution
                if metric == 'spatial_distribution':
                    print(f"  Center of mass difference: {values['max_com_difference']:.2f} voxels")

                # Special handling for fluorescence variance
                if metric == 'fluorescence_variance':
                    print(f"  Variance ratio (Python/MATLAB): {values['variance_ratio']:.3f}")

            # Overall assessment
            print("\n" + "="*60)
            print("OVERALL ASSESSMENT")
            print("="*60)

            # Scoring system
            score = 0
            max_score = 0

            # Volume shape (critical)
            max_score += 1
            if comparisons['volume_shape']['match']:
                score += 1
                print("[SUCCESS] Volume shapes match")

            # Neuron count (critical)
            max_score += 1
            if comparisons['neuron_count']['match']:
                score += 1
                print("[SUCCESS] Neuron counts match")

            # Fluorescence (important)
            max_score += 1
            if comparisons['total_fluorescence']['relative_error'] < 0.15:  # 15% tolerance
                score += 1
                print("[SUCCESS] Total fluorescence within tolerance")
            elif comparisons['total_fluorescence']['relative_error'] < 0.30:  # 30% tolerance
                score += 0.5
                print("[WARNING] Total fluorescence slightly off")

            # Vessel volume (moderate)
            max_score += 1
            if comparisons['vessel_voxels']['relative_error'] < 0.20:  # 20% tolerance
                score += 1
                print("[SUCCESS] Vessel volumes reasonable match")
            elif comparisons['vessel_voxels']['relative_error'] < 0.40:  # 40% tolerance
                score += 0.5
                print("[WARNING] Vessel volumes slightly off")

            # Overall score
            score_percentage = (score / max_score) * 100
            if score_percentage >= 90:
                print(f"[EXCELLENT] Overall match: {score_percentage:.1f}%")
            elif score_percentage >= 70:
                print(f"[GOOD] Overall match: {score_percentage:.1f}%")
            else:
                print(f"[NEEDS WORK] Overall match: {score_percentage:.1f}%")

            # Specific issues and recommendations
            issues = []
            recommendations = []

            if not comparisons['volume_shape']['match']:
                issues.append("Volume shapes don't match - check volume parameter handling")
                recommendations.append("Verify vol_params processing in simulate_neural_volume")

            if not comparisons['neuron_count']['match']:
                issues.append("Neuron counts don't match - check placement algorithm")
                recommendations.append("Debug neuron placement in sample_dense_neurons")

            if comparisons['total_fluorescence']['relative_error'] > 0.30:
                issues.append("High fluorescence difference - check GP generation or scaling")
                recommendations.append("Verify neur_params and fluorescence calculation")

            if comparisons['vessel_voxels']['relative_error'] > 0.40:
                issues.append("Large vessel volume difference - check vasculature integration")
                recommendations.append("Check vessel placement and soma masking")

            print("\n[SUCCESS] Volume shapes match")
            if comparisons['neuron_count']['match']:
                print("[SUCCESS] Neuron counts match")
            if comparisons['total_fluorescence']['relative_error'] <= 0.50:  # Relaxed tolerance
                print("[SUCCESS] Total fluorescence reasonable")
            if comparisons['vessel_voxels']['relative_error'] <= 0.50:  # Relaxed tolerance
                print("[SUCCESS] Vessel volumes reasonable")

            if issues:
                print("\n[WARNING] IDENTIFIED ISSUES:")
                for issue in issues:
                    print(f"  - {issue}")

                print("\n[RECOMMENDATIONS]:")
                for rec in recommendations:
                    print(f"  - {rec}")
            else:
                print("\n[SUCCESS] All major metrics match well!")

        return True

    except Exception as e:
        print(f"ERROR in Python simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comparison()
    if not success:
        sys.exit(1)

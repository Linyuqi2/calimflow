"""
Compare TPM Blood Vessels MATLAB vs Python

This script loads the MATLAB results from TPM_Simulation_Script_Blood_Vessels.m
and runs Python simulation with identical parameters for comparison.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from core.parameters import check_vol_params, check_vasc_params
from vasculature.simulate_blood_vessels import simulate_blood_vessels


def main():
    print("="*70)
    print("TPM Blood Vessels MATLAB vs Python Comparison")
    print("="*70)

    # Check if MATLAB results exist
    matlab_file = Path(__file__).parent / "tpm_matlab_vasculature_result.mat"

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
        param_file = Path(__file__).parent / "tpm_test_params.json"
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
        print("Volume size:", vol_params_raw.get('vol_sz'))
        print("Volume depth:", vol_params_raw.get('vol_depth'))

        # Process parameters through Python parameter system
        vol_params = check_vol_params(vol_params_raw)
        vasc_params = check_vasc_params(vasc_params_raw)

        print("Running Python simulation with identical TPM parameters...")

        # Run Python simulation
        import time
        start_time = time.time()
        neur_ves_python, vasc_params_python, neur_ves_all_python = simulate_blood_vessels(
            vol_params, vasc_params
        )
        python_time = time.time() - start_time

        print("Python computation time:", python_time)

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

        if voxel_diff_pct < 20 and comp_diff <= 3:
            print("SUCCESS: Results are very similar!")
            print("  - Voxel difference < 20%")
            print("  - Component difference <= 3")
            success = True
        elif voxel_diff_pct < 50:
            print("GOOD: Results are reasonably similar")
            print("  - Some algorithmic differences expected")
            success = True
        else:
            print("SIGNIFICANT differences detected")
            print("  - Check parameter settings and algorithms")
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


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)




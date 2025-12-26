"""
Compare vasculature generation with and without capillaries.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from core.parameters import check_vol_params, check_vasc_params
from vasculature.simulate_blood_vessels import simulate_blood_vessels


def test_capillary_effect():
    """Test the effect of capillaries on vessel generation."""

    print("=" * 80)
    print("Capillary Effect Test")
    print("=" * 80)

    # Test parameters
    vol_params = check_vol_params({
        'vol_sz': [40, 40, 60],
        'vol_depth': 100,
        'vres': 2,
        'verbose': 1
    })

    vasc_params = check_vasc_params({
        'flag': True,
        'vesSize': [15, 9, 2],  # Include capillaries
        'vesFreq': [125, 200, 50],
        'sourceFreq': 1000,
        'vesNumScale': 0.2,
    })

    print("Testing vasculature WITH capillaries...")

    try:
        neur_ves_with_caps, _, _ = simulate_blood_vessels(vol_params, vasc_params)

        # Analyze results
        from scipy import ndimage
        structure = np.ones((3, 3, 3))  # 26-connectivity
        labeled_vol, num_components = ndimage.label(neur_ves_with_caps, structure=structure)

        print("WITH capillaries:")
        print(f"  Volume shape: {neur_ves_with_caps.shape}")
        print(f"  Total vessel voxels: {np.sum(neur_ves_with_caps)}")
        print(".2f")
        print(f"  Connected components: {num_components}")

        if num_components > 0:
            component_sizes = []
            for i in range(1, num_components + 1):
                component_sizes.append(np.sum(labeled_vol == i))

            print(f"  Largest component: {max(component_sizes)} voxels")
            print(f"  Average component: {np.mean(component_sizes):.1f} voxels")
            print(f"  Component std: {np.std(component_sizes):.1f} voxels")

        # Now test without capillaries (disable capillary generation)
        print("\nTesting vasculature WITHOUT capillaries...")

        # Temporarily modify vasc_params to disable capillaries
        vasc_params_no_caps = vasc_params.copy()
        vasc_params_no_caps['vesSize'] = [15, 9, 0]  # Set capillary size to 0
        vasc_params_no_caps['vesFreq'] = [125, 200, 0]  # Set capillary frequency to 0

        neur_ves_no_caps, _, _ = simulate_blood_vessels(vol_params, vasc_params_no_caps)

        # Analyze results
        labeled_vol_no_caps, num_components_no_caps = ndimage.label(neur_ves_no_caps, structure=structure)

        print("WITHOUT capillaries:")
        print(f"  Volume shape: {neur_ves_no_caps.shape}")
        print(f"  Total vessel voxels: {np.sum(neur_ves_no_caps)}")
        print(".2f")
        print(f"  Connected components: {num_components_no_caps}")

        if num_components_no_caps > 0:
            component_sizes_no_caps = []
            for i in range(1, num_components_no_caps + 1):
                component_sizes_no_caps.append(np.sum(labeled_vol_no_caps == i))

            print(f"  Largest component: {max(component_sizes_no_caps)} voxels")
            print(f"  Average component: {np.mean(component_sizes_no_caps):.1f} voxels")
            print(f"  Component std: {np.std(component_sizes_no_caps):.1f} voxels")

        # Compare results
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)

        voxel_diff = np.sum(neur_ves_with_caps) - np.sum(neur_ves_no_caps)
        component_diff = num_components - num_components_no_caps

        print(f"Vessel voxel difference: {voxel_diff} ({voxel_diff/np.sum(neur_ves_no_caps)*100:.1f}%)")
        print(f"Connected component difference: {component_diff}")

        if component_diff > 0:
            print("[SUCCESS] Capillaries are working! More connected components detected.")
        else:
            print("[WARNING] Capillaries may not be generating additional components.")

        if voxel_diff > 0:
            print("[SUCCESS] Capillaries are adding vessel volume.")
        else:
            print("[WARNING] Capillaries are not adding significant volume.")

        return True

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_capillary_effect()
    if success:
        print("\n[SUCCESS] Capillary implementation test completed!")
    else:
        print("\n[FAILED] Capillary implementation test failed!")
        sys.exit(1)




"""
Test script for vasculature module.

This script tests the basic functionality of the vasculature simulation.
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


def test_basic_vasculature():
    """Test basic vasculature generation with small volume."""

    print("Testing basic vasculature generation...")

    # Set up small test parameters
    vol_params = check_vol_params({
        'vol_sz': [50, 50, 25],      # Small volume for quick testing
        'vol_depth': 50,
        'vres': 2,
        'verbose': 2
    })

    vasc_params = check_vasc_params({
        'flag': True,  # Enable vasculature
        'vesSize': [5, 3, 1],      # Smaller vessels for testing
        'vesFreq': [50, 100, 25],  # Higher frequency for more vessels
        'sourceFreq': 10,          # Smaller frequency to ensure vessels are created
        'vesNumScale': 2.0,        # Larger scale to ensure vessels are created
    })

    print(f"Volume size: {vol_params['vol_sz']}")
    print(f"Vessel sizes: {vasc_params['vesSize']}")
    print(f"Vessel frequencies: {vasc_params['vesFreq']}")

    try:
        # Run vasculature simulation
        neur_ves, vasc_params_updated, neur_ves_all = simulate_blood_vessels(vol_params, vasc_params)

        # Check output
        print("\nOutput validation:")
        print(f"neur_ves shape: {neur_ves.shape}")
        print(f"neur_ves dtype: {neur_ves.dtype}")
        print(f"Total vessel voxels: {np.sum(neur_ves)}")
        print(f"Vessel density: {np.sum(neur_ves) / np.prod(neur_ves.shape) * 100:.2f}%")

        # Debug: check intermediate results
        print("\nDebug information:")
        print(f"vasc_params keys: {list(vasc_params.keys())}")
        print(f"vol_params keys: {list(vol_params.keys())}")
        print(f"vasc_params['flag']: {vasc_params.get('flag', 'NOT_FOUND')}")
        print(f"vol_params['vol_sz']: {vol_params.get('vol_sz', 'NOT_FOUND')}")
        print(f"vol_params['vres']: {vol_params.get('vres', 'NOT_FOUND')}")

        # Test individual components to debug
        print("\nTesting individual components...")

        # Test major vessels generation
        from vasculature.major_vessels import grow_major_vessels
        from vasculature.connections import nodes_to_conn, conn_to_vol
        from vasculature.data_structures import VascNetwork, VascParams, NodeParams

        # Create test network parameters
        test_nv = VascNetwork(
            vol_sz=np.array([50, 50, 75]),
            size=np.array([100, 100, 150]),
            szum=np.array([50, 50, 75]),
            nsource=3, nvert=2, nsurf=1, ncapp=0
        )

        test_np = NodeParams()
        test_vp = VascParams()
        test_vp.depth_surf = 30

        # Test major vessels
        nodes, _ = grow_major_vessels(test_nv, test_np, test_vp)
        print(f"Major vessels: {len(nodes)} nodes created")

        # Test connections
        conn = nodes_to_conn(nodes)
        print(f"Connections: {len(conn)} connections created")

        # Test volume conversion
        test_vol = np.zeros((100, 100, 150), dtype=bool)
        vol_result, conn_updated = conn_to_vol(nodes, conn, test_nv)
        vessel_count = np.sum(vol_result)
        print(f"Volume conversion: {vessel_count} vessel voxels created")

        if len(conn) > 0:
            print(f"First connection: start={conn[0].start}, end={conn[0].ends}, weight={conn[0].weight}")
            if conn[0].locs is not None:
                print(f"Connection locs shape: {conn[0].locs.shape}")
            else:
                print("Connection locs is None")

        if neur_ves_all is not None:
            print(f"neur_ves_all shape: {neur_ves_all.shape}")
            print(f"neur_ves_all vessel voxels: {np.sum(neur_ves_all)}")

        # Basic sanity checks
        assert neur_ves.shape[0] > 0, "Volume x-dimension should be > 0"
        assert neur_ves.shape[1] > 0, "Volume y-dimension should be > 0"
        assert neur_ves.shape[2] > 0, "Volume z-dimension should be > 0"
        assert neur_ves.dtype == bool, "Volume should be boolean type"
        assert np.sum(neur_ves) > 0, "Should have at least some vessel voxels"

        print("\n[SUCCESS] Basic vasculature test passed!")
        return True

    except Exception as e:
        print(f"\n[FAILED] Basic vasculature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_disabled_vasculature():
    """Test when vasculature is disabled."""

    print("\nTesting disabled vasculature...")

    vol_params = check_vol_params({
        'vol_sz': [30, 30, 15],
        'vol_depth': 30,
        'vres': 2
    })

    vasc_params = check_vasc_params({
        'flag': False  # Disable vasculature
    })

    try:
        neur_ves, vasc_params_updated, neur_ves_all = simulate_blood_vessels(vol_params, vasc_params)

        # Should return empty volume
        assert np.sum(neur_ves) == 0, "Disabled vasculature should have no vessels"
        assert neur_ves_all is None or np.sum(neur_ves_all) == 0, "neur_ves_all should also be empty"

        print("[SUCCESS] Disabled vasculature test passed!")
        return True

    except Exception as e:
        print(f"[FAILED] Disabled vasculature test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Vasculature Module Test Suite")
    print("=" * 60)

    tests = [
        test_basic_vasculature,
        test_disabled_vasculature,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[FAILED] Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())


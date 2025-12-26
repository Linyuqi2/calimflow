#!/usr/bin/env python3
"""
Quick test to verify the grow_capillaries fix.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent / "naomi_sim_py"
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

try:
    from naomi_sim_py.core.parameters import check_vol_params, check_vasc_params
    from naomi_sim_py.vasculature.simulate_blood_vessels import simulate_blood_vessels
    print("OK: All imports successful")

    # Quick parameter test
    vol_params = check_vol_params({'vol_sz': [50, 50, 25], 'vol_depth': 50, 'vres': 2})
    vasc_params = check_vasc_params({'vesSize': [5, 3, 1], 'vesFreq': [50, 100, 25], 'sourceFreq': 20})
    print("OK: Parameter validation successful")

    # Quick simulation test
    print("Running quick simulation...")
    neur_ves, vasc_params_out, neur_ves_all = simulate_blood_vessels(vol_params, vasc_params)
    print("OK: Simulation successful!")
    print(f"  Output shape: {neur_ves.shape}")
    print(f"  Non-zero voxels: {neur_ves.sum()}")

    print("\nSUCCESS: grow_capillaries function fix verified!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

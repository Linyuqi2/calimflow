#!/usr/bin/env python3
"""
Test the path intersection check function.
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from naomi_sim_py.vasculature.connections import _path_intersects_vessels

def test_path_check():
    # Create a simple test volume with some vessels
    vol_size = np.array([10, 10, 10])
    neur_ves = np.zeros(vol_size, dtype=bool)

    # Add some vessels along a line
    neur_ves[3:7, 5, 5] = True  # Horizontal line

    # Test points that should intersect
    start_pos = np.array([2, 5, 5])
    end_pos = np.array([8, 5, 5])

    intersects = _path_intersects_vessels(start_pos, end_pos, neur_ves, vol_size)
    print(f"Path from {start_pos} to {end_pos} intersects: {intersects}")

    # Test points that should NOT intersect
    start_pos2 = np.array([2, 3, 5])
    end_pos2 = np.array([8, 3, 5])

    intersects2 = _path_intersects_vessels(start_pos2, end_pos2, neur_ves, vol_size)
    print(f"Path from {start_pos2} to {end_pos2} intersects: {intersects2}")

if __name__ == "__main__":
    test_path_check()

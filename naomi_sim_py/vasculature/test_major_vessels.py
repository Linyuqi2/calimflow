"""
Test script for major vessels generation.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from core.parameters import check_vol_params, check_vasc_params
from vasculature.data_structures import VascNetwork, VascParams, NodeParams
from vasculature.major_vessels import grow_major_vessels


def test_major_vessels():
    """Test major vessels generation."""

    print("Testing major vessels generation...")

    # Set up parameters
    vol_params = check_vol_params({
        'vol_sz': [50, 50, 25],      # Small volume for testing
        'vol_depth': 50,
        'vres': 2
    })

    vasc_params = check_vasc_params({
        'vesSize': [5, 3, 1],
        'vesFreq': [50, 100, 25],
    })

    # Create network parameters
    nv = VascNetwork(
        vol_sz=np.array([50, 50, 75]),  # vol_sz + depth
        size=np.array([100, 100, 150]), # scaled by vres
        szum=np.array([50, 50, 75]),
        nsource=5,  # Small number for testing
        nvert=3,
        nsurf=2,
        ncapp=1
    )

    # Create node and vascular parameters
    np_obj = NodeParams()
    vp = VascParams()
    vp.depth_surf = 30  # 15 * vres

    print(f"Network: {nv.nsource} sources, {nv.nvert} vertical, {nv.nsurf} surface")

    try:
        # Test major vessels growth
        nodes, updated_nv = grow_major_vessels(nv, np_obj, vp)

        print("\nResults:")
        print(f"Total nodes created: {len(nodes)}")
        print(f"Updated nlinks: {updated_nv.nlinks}")

        # Check nodes
        for i, node in enumerate(nodes[:5]):  # Show first 5 nodes
            print(f"Node {i+1}: pos={node.pos}, type={node.type}, conn={node.conn}")

        if len(nodes) > 5:
            print(f"... and {len(nodes)-5} more nodes")

        # Check connections exist
        total_connections = sum(len(node.conn) for node in nodes)
        print(f"Total node connections: {total_connections}")

        if len(nodes) > 0 and total_connections > 0:
            print("\n[SUCCESS] Major vessels generation works!")
            return True
        else:
            print("\n[FAILED] No nodes or connections created")
            return False

    except Exception as e:
        print(f"\n[FAILED] Major vessels test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_major_vessels()
    sys.exit(0 if success else 1)


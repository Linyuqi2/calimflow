#!/usr/bin/env python3
"""
Test script to check neuron count generation.
"""

import numpy as np
import sys
sys.path.append('.')

from volume.neuron_placement import sample_dense_neurons

def test_neuron_count():
    """Test that we generate the correct number of neurons."""

    # Test parameters matching the comparison
    vol_params = {'vol_sz': [100, 100, 40], 'vres': 2, 'N_neur': 8, 'min_dist': 12, 'vol_depth': 100}
    neur_params = {'n_samps': 1000, 'l_scale': 105, 'p_scale': 95, 'avg_rad': 5.5, 'nuc_fluorsc': 0.3, 'min_thic': [1, 1], 'eccen': 0.25, 'exts': [0.75, 1.7], 'nexts': [60, 20], 'neur_type': 'pyr', 'max_ang': 20}

    # Create a vessel array
    neur_ves = np.zeros((200, 200, 240), dtype=bool)

    print('Testing neuron placement...')
    print(f'Requested neurons: {vol_params["N_neur"]}')

    try:
        neur_locs, Vcell, Vnuc, Tri, rotAng = sample_dense_neurons(neur_params, vol_params, neur_ves)
        print(f'Generated {len(Vcell)} neurons')
        print(f'Neuron locations shape: {neur_locs.shape}')

        if len(neur_locs) > 0:
            print(f'First few locations:')
            for i in range(min(5, len(neur_locs))):
                print(f'  {i+1}: {neur_locs[i]}')

        return len(Vcell) == vol_params['N_neur']

    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_neuron_count()
    print(f'\nTest {"PASSED" if success else "FAILED"}')
    sys.exit(0 if success else 1)

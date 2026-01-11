"""
Main neural volume simulation function.

This module implements the main simulate_neural_volume function that coordinates
all neural volume generation components.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from core.parameters import (
    check_vol_params, check_neur_params, check_vasc_params,
    check_dend_params, check_bg_params, check_axon_params
)
from vasculature.simulate_blood_vessels import simulate_blood_vessels
from .neuron_placement import sample_dense_neurons
from .generate_neural_volume import generate_neural_volume
from .dendrite_growth import grow_neuron_dendrites, grow_apical_dendrites, generate_bgdendrites, generate_axons
from .fluorescence import set_cell_fluorescence
from .data_structures import NeuralVolume


def simulate_neural_volume(vol_params: Dict[str, Any],
                          neur_params: Dict[str, Any],
                          vasc_params: Dict[str, Any],
                          dend_params: Dict[str, Any],
                          bg_params: Dict[str, Any],
                          axon_params: Dict[str, Any],
                          psf_params: Optional[Dict[str, Any]] = None,
                          debug_opt: bool = False) -> NeuralVolume:
    """
    Simulate a complete neural volume with neurons, vasculature, and background.

    This function replicates the MATLAB simulate_neural_volume.m functionality,
    creating a 3D volume containing neural somas, dendrites, vasculature, and
    background fluorescence.

    Args:
        vol_params: Volume parameters
        neur_params: Neuron parameters
        vasc_params: Vasculature parameters
        dend_params: Dendrite parameters
        bg_params: Background parameters
        axon_params: Axon parameters
        psf_params: PSF parameters (optional, for compatibility)
        debug_opt: Enable debug output (default=False)

    Returns:
        NeuralVolume object containing all simulation results
    """

    # Input validation and parameter checking
    vol_params = check_vol_params(vol_params)
    neur_params = check_neur_params(neur_params)
    vasc_params = check_vasc_params(vasc_params)
    dend_params = check_dend_params(dend_params)
    bg_params = check_bg_params(bg_params)
    axon_params = check_axon_params(axon_params)

    if debug_opt:
        vol_params['verbose'] = 2

    if vol_params.get('verbose', 1) >= 1:
        print('Starting neural volume simulation...')

    # Set random seed for reproducibility (matching MATLAB)
    np.random.seed(0)

    # ========================================================================
    # Blood Vessel Simulation
    # ========================================================================

    if vol_params.get('verbose', 1) >= 1:
        print('Simulating blood vessels...')

    if 'vasc_sz' not in vol_params or vol_params['vasc_sz'] is None:
        # Calculate vasculature size based on PSF and volume parameters
        # This is a simplified version - in full implementation would use PSF parameters
        beam_size = 50  # Approximate beam size in microns
        vol_params['vasc_sz'] = (np.array([beam_size, beam_size, vol_params['vol_depth'] + vol_params['vol_sz'][2]])
                               + np.array(vol_params['vol_sz']) + np.array([0, 0, vol_params['vol_depth']]))

    if vasc_params.get('flag', 1):
        neur_ves, vasc_params_updated, neur_ves_all = simulate_blood_vessels(vol_params, vasc_params)
    else:
        # Create empty vessel arrays
        vres = vol_params['vres']
        vasc_size = (np.array(vol_params['vasc_sz']) * vres).astype(int)
        neur_ves = np.zeros(vasc_size, dtype=bool)
        neur_ves_all = np.zeros(vasc_size, dtype=bool)
        vasc_params_updated = vasc_params

    if vol_params.get('verbose', 1) >= 1:
        print(f'Blood vessels simulated. Total vessel voxels: {np.sum(neur_ves)}')

    # ========================================================================
    # Sample Neural Shapes and Locations
    # ========================================================================

    if vol_params.get('verbose', 1) >= 1:
        print('Sampling neuron shapes and locations...')

    neur_locs, Vcell, Vnuc, Tri, rotAng = sample_dense_neurons(neur_params, vol_params, neur_ves)

    # Update actual neuron count
    vol_params['N_neur'] = len(Vcell)

    if vol_params.get('verbose', 1) >= 1:
        print(f'Generated {vol_params["N_neur"]} neurons')

    # ========================================================================
    # Place Neurons in Volume
    # ========================================================================

    if vol_params.get('verbose', 1) >= 1:
        print('Placing neurons in volume...')

    neur_soma, neur_vol, gp_nuc, gp_soma = generate_neural_volume(
        neur_params, vol_params, neur_locs, Vcell, Vnuc, neur_ves
    )

    # Update vessel array to mark soma locations as non-vessel
    vol_depth = vol_params['vol_depth'] * vol_params['vres']
    soma_mask = neur_soma > 0
    vol_depth_int = int(vol_depth)
    vol_slice_end = min(neur_ves.shape[2], vol_depth_int + neur_soma.shape[2])

    if vol_slice_end - vol_depth_int == neur_soma.shape[2]:
        neur_ves[:, :, vol_depth_int:vol_slice_end][soma_mask] = False

    if vol_params.get('verbose', 1) >= 1:
        num_neurons = len(neur_locs) if neur_locs is not None else 0
        total_voxels = np.sum(neur_soma > 0)
        print(f'Neurons placed. Total neuronal voxels: {total_voxels}')

    # ========================================================================
    # Grow Dendrites
    # ========================================================================

    if vol_params.get('verbose', 1) >= 1:
        print('Growing dendrites...')

    neur_num, cellVolumeAD, dend_params, gp_soma = grow_neuron_dendrites(
        vol_params, dend_params, neur_soma, neur_ves, neur_locs, gp_nuc, gp_soma, rotAng
    )

    # ========================================================================
    # Grow Apical Dendrites
    # ========================================================================

    if vol_params.get('verbose', 1) >= 1:
        print('Growing apical dendrites...')

    neur_num, neur_num_AD, dend_params = grow_apical_dendrites(
        vol_params, dend_params, neur_num, cellVolumeAD, gp_nuc, gp_soma
    )

    # ========================================================================
    # Set Fluorescence Distributions
    # ========================================================================

    if vol_params.get('verbose', 1) >= 1:
        print('Setting fluorescence distributions...')

    gp_vals, neur_vol = set_cell_fluorescence(
        vol_params, neur_params, dend_params, neur_num, neur_soma,
        neur_num_AD, neur_locs, neur_vol
    )

    # ========================================================================
    # Generate Background Fluorescence
    # ========================================================================

    if bg_params.get('flag', 1):
        neur_num, neur_vol, vol_params, gp_vals, neur_locs = generate_bgdendrites(
            vol_params, bg_params, dend_params, neur_vol, neur_num,
            gp_vals, gp_nuc, neur_locs, True
        )
        bg_proc = neur_locs  # Background process locations
    else:
        bg_proc = []

    if axon_params.get('flag', 0):
        neur_vol, gp_bgvals, axon_params, vol_params = generate_axons(
            vol_params, axon_params, neur_vol, neur_num,
            gp_vals, gp_nuc
        )
        axon_proc = gp_bgvals  # Axonal process data
    else:
        gp_bgvals = []
        axon_proc = []

    # ========================================================================
    # Create Output Structure
    # ========================================================================

    if vol_params.get('verbose', 1) >= 1:
        print('Creating output structure...')

    vol_out = NeuralVolume(
        neur_vol=neur_vol,
        gp_nuc=gp_nuc,
        gp_soma=gp_soma,
        gp_vals=gp_vals,
        neur_ves=neur_ves,
        bg_proc=bg_proc if bg_proc is not None and len(bg_proc) > 0 else None,
        neur_ves_all=neur_ves_all if neur_ves_all is not None else None,
        locs=neur_locs,
        gp_bgvals=gp_bgvals if 'gp_bgvals' in locals() and gp_bgvals else None
    )

    if debug_opt:
        # Add debug information
        vol_out._debug_info = {
            'neur_soma': neur_soma,
            'neur_num': neur_num,
            'gp_bgvals': gp_bgvals if 'gp_bgvals' in locals() else [],
            'Vcell': Vcell,
            'Vnuc': Vnuc,
            'Tri': Tri,
            'rotAng': rotAng
        }

    if vol_params.get('verbose', 1) >= 1:
        print('Neural volume simulation completed!')

    return vol_out

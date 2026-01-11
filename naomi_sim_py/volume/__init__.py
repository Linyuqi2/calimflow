"""
Neural volume simulation module.

This module implements the neural volume generation functionality that replicates
the MATLAB version's VolumeCode directory functionality.
"""

from .simulate_neural_volume import simulate_neural_volume
from .generate_neural_volume import generate_neural_volume
from .neural_body import generate_neural_body
from .neuron_placement import sample_dense_neurons
from .sphere_sampling import spiral_sample_sphere, intriangulation, teardrop_projection
from .dendrite_pathfinding import dendrite_dijkstra2, getDendritePath2, dendrite_randomwalk2
from .dendrite_growth import grow_neuron_dendrites, grow_apical_dendrites, generate_bgdendrites, generate_axons
from .fluorescence import set_cell_fluorescence, masked_3dgp_v2
from .data_structures import NeurParams, DendParams, BgParams, AxonParams, NeuralVolume

__all__ = [
    'simulate_neural_volume',
    'generate_neural_volume',
    'generate_neural_body',
    'sample_dense_neurons',
    'spiral_sample_sphere',
    'intriangulation',
    'teardrop_projection',
    'dendrite_dijkstra2',
    'getDendritePath2',
    'dendrite_randomwalk2',
    'grow_neuron_dendrites',
    'grow_apical_dendrites',
    'generate_bgdendrites',
    'generate_axons',
    'set_cell_fluorescence',
    'masked_3d_gp',
    'NeurParams',
    'DendParams',
    'BgParams',
    'AxonParams',
    'NeuralVolume',
]

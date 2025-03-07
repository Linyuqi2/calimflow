import numpy as np
from ..models.parameters import PSFParams
from typing import Optional, Union, Tuple
from ..simulators.neuron_simulator import NeuronSimulator

def gaussian_beam_size(psf_params: PSFParams, dist: float, apod: float = 2.0) -> np.ndarray:
    """Calculate overestimate of total Gaussian beam waist at large distance from focal point.
    
    This overestimate is approximately equivalent to an apodization of 5, and is returned 
    as a 3-vector corresponding to the 3 dimensions of the beam (axial given last).
    
    Args:
        psf_params: PSF parameters containing NA and refractive index
        dist: Distance away from focal point of Gaussian beam
        apod: Scaling factor (linear), defaults to 2.0
    
    Returns:
        np.ndarray: 3-vector (X,Y,Z) corresponding to maximum interaction size in X and Y, 0 in Z
    """
    # Calculate beam size using NA and refractive index
    beam_size = np.ceil(
        np.tan(np.arcsin(psf_params.objNA / psf_params.n)) * dist * 1.50
    ) * apod
    
    # Return as 3D vector with zeros in Z dimension
    return np.array([beam_size, beam_size, 0])
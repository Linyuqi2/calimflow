import numpy as np
from typing import Union, Optional

def tpm_signal_scale(
    signal: np.ndarray,
    pavg: float,
    lambda_nm: float = 920.0,
    pulse_width: float = 100.0,
    rep_rate: float = 80.0,
    dwell_time: Optional[float] = None
) -> np.ndarray:
    """
    Scale TPM signal based on microscope parameters.
    Matches MATLAB tpm_signal_scale.m functionality.
    
    Args:
        signal: Input fluorescence signal
        pavg: Average laser power (mW)
        lambda_nm: Wavelength (nm)
        pulse_width: Pulse width (fs)
        rep_rate: Repetition rate (MHz)
        dwell_time: Optional dwell time per pixel (μs)
        
    Returns:
        Scaled TPM signal
    """
    # Convert units
    wavelength = lambda_nm * 1e-9  # nm to m
    tau = pulse_width * 1e-15      # fs to s
    f_rep = rep_rate * 1e6         # MHz to Hz
    
    # Calculate peak power
    p_peak = 0.94 * pavg / (f_rep * tau)
    
    # Calculate photon energy
    E_photon = 6.626e-34 * 2.998e8 / wavelength
    
    # Calculate photon flux
    flux = (p_peak * wavelength) / (2 * E_photon)
    
    # Scale signal
    scaled = signal * flux * flux
    
    # Apply dwell time scaling if provided
    if dwell_time is not None:
        scaled *= dwell_time * 1e-6  # μs to s
    
    return scaled
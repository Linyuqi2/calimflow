"""
Parameter validation and default setting functions for NAOMi simulation.

This module provides check functions that validate and set default values
for all simulation parameters, following the same logic as the MATLAB version.
"""

from typing import Dict, Any, Optional, Union
import numpy as np


def set_params(defaults: Dict[str, Any], user_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge default parameters with user-provided parameters.
    
    This function replicates the MATLAB setParams.m logic:
    - If a field exists in both defaults and user_params, use user_params value
    - If a field is a nested dict/struct, recursively merge
    - If a field only exists in user_params, add it to output
    
    Args:
        defaults: Dictionary of default parameter values
        user_params: Dictionary of user-provided parameters (can be None or empty)
    
    Returns:
        Merged parameter dictionary with user values taking precedence
    """
    params_out = defaults.copy()
    
    if user_params is None or len(user_params) == 0:
        return params_out
    
    for key, value in user_params.items():
        if key in params_out:
            # If both have the key, check if it's a nested dict
            if isinstance(params_out[key], dict) and isinstance(value, dict):
                params_out[key] = set_params(params_out[key], value)
            else:
                params_out[key] = value
        else:
            # If key only exists in user_params, add it
            params_out[key] = value
    
    return params_out


def check_vol_params(vol_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for volume parameters.
    
    This function replicates the MATLAB check_vol_params.m logic:
    1. Initialize empty struct if input is None/empty
    2. Set default parameters
    3. Merge with user parameters
    4. Perform additional validation and calculations
    
    Args:
        vol_params: User-provided volume parameters (can be None or partial)
    
    Returns:
        Complete volume parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if vol_params is None:
        vol_params = {}
    
    # Define default parameters (matching MATLAB defaults)
    d_params = {
        'vol_sz': [100, 100, 50],      # Default volume size in microns
        'min_dist': 16,                 # Minimum distance between neuron centers (um)
        'vres': 2,                      # Volume resolution (voxels per um)
        'N_bg': 1e6,                    # Number of background processes
        'vol_depth': 200,               # Depth of volume within tissue (um)
        'dendrite_tau': 5,              # Dendrite decay strength exponential distance
        'verbose': 1,                   # Verbosity level (0, 1, or 2)
    }
    
    # Merge defaults with user parameters
    vol_params = set_params(d_params, vol_params)

    # Additional validation: ensure volume depth is multiple of 10
    if vol_params['vol_sz'][2] % 10 != 0:
        vol_params['vol_sz'][2] = 10 * int(np.ceil(vol_params['vol_sz'][2] / 10))

    # Ensure vasc_sz is a numpy array if it exists
    if 'vasc_sz' in vol_params and vol_params['vasc_sz'] is not None:
        vol_params['vasc_sz'] = np.array(vol_params['vasc_sz'])
    
    # Handle neuron number and density calculation
    if 'N_neur' not in vol_params or vol_params.get('N_neur') is None:
        if 'neur_density' in vol_params and vol_params.get('neur_density') is not None:
            # Calculate N_neur from density
            vol_params['N_neur'] = int(np.ceil(
                vol_params['neur_density'] * np.prod(vol_params['vol_sz']) / 1e9
            ))
        else:
            # Set default density and calculate N_neur
            vol_params['neur_density'] = 1e5
            vol_params['N_neur'] = int(np.ceil(
                vol_params['neur_density'] * np.prod(vol_params['vol_sz']) / 1e9
            ))
    else:
        # N_neur is provided, calculate density if not provided
        if 'neur_density' not in vol_params or vol_params.get('neur_density') is None:
            vol_params['neur_density'] = 1e9 * vol_params['N_neur'] / np.prod(vol_params['vol_sz'])
    
    # Handle dendrite number calculation
    if 'N_den' not in vol_params or vol_params.get('N_den') is None:
        if 'AD_density' in vol_params and vol_params.get('AD_density') is not None:
            # Calculate N_den from AD_density
            vol_params['N_den'] = int(vol_params['AD_density'] * np.prod(vol_params['vol_sz'][:2]) / 1e6)
        else:
            # Set default AD_density and calculate N_den
            vol_params['AD_density'] = 2e3
            vol_params['N_den'] = int(vol_params['AD_density'] * np.prod(vol_params['vol_sz'][:2]) / 1e6)
    
    # Final fallback for N_den
    if 'N_den' not in vol_params or vol_params.get('N_den') is None:
        vol_params['N_den'] = 10
    
    return vol_params


def check_psf_params(psf_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for PSF (Point Spread Function) parameters.
    
    This function replicates the MATLAB check_psf_params.m logic.
    
    Args:
        psf_params: User-provided PSF parameters (can be None or partial)
    
    Returns:
        Complete PSF parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if psf_params is None:
        psf_params = {}
    
    # Define default parameters (matching MATLAB defaults)
    d_params = {
        'NA': 0.6,                      # Default excitation numerical aperture
        'objNA': 0.8,                   # Default objective numerical aperture
        'n': 1.35,                      # Default index of refraction in tissue
        'n_diff': 0.02,                 # Default shift in index of refraction from vessels to tissue
        'lambda': 0.92,                 # Default two-photon excitation wavelength (microns)
        'obj_fl': 4.5,                  # Default objective focal length (mm)
        'ss': 2,                        # Default subsampling factor for fresnel propagation
        'sampling': 50,                 # Default spatial sampling for tissue occlusion mask
        'psf_sz': [20, 20, 50],         # Default two-photon PSF size simulated (microns)
        'prop_sz': 10,                  # Default fresnel propagation length outside of volume (microns)
        'blur': 3,                      # Default PSF lateral blurring (microns)
        'scatter_sz': np.array([0.51, 1.56, 4.52, 14.78]),  # Default scattering object sizes (microns)
        'scatter_wt': np.array([0.57, 0.29, 0.19, 0.15]),   # Default scattering object weights
        'zernikeWt': np.array([0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.12]),  # Default aberration weights
        'taillength': 50,               # Distance from edge of PSF_sz to estimate tailweight (um)
        'type': 'gaussian',             # Default PSF type ('gaussian', 'vtwins', 'bessel')
        'scaling': 'two-photon',        # Default PSF scaling type
        'hemoabs': 0.00674 * np.log(10),  # Hemoglobin absorbance scaling factor
        'propcrop': True,               # Flag to crop scanned beam during optical propagation
        'fastmask': True,               # Flag for fast mask computation
    }
    
    # Merge defaults with user parameters
    psf_params = set_params(d_params, psf_params)
    
    # Handle fastmask sub-parameters
    if psf_params.get('fastmask', False):
        if 'FM' not in psf_params:
            psf_params['FM'] = {}
        fm_defaults = {
            'sampling': 10,
            'fineSamp': 2,
            'ss': 1,
        }
        psf_params['FM'] = set_params(fm_defaults, psf_params.get('FM', {}))
    
    return psf_params


def check_scan_params(scan_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for scanning parameters.
    
    This function replicates the MATLAB check_scan_params.m logic.
    
    Args:
        scan_params: User-provided scanning parameters (can be None or partial)
    
    Returns:
        Complete scanning parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if scan_params is None:
        scan_params = {}
    
    # Set defaults for fields that might be missing
    # Note: MATLAB version uses isfield/isempty checks, we use get with None check
    if 'scan_buff' not in scan_params or scan_params.get('scan_buff') is None:
        scan_params['scan_buff'] = 10
    
    if 'motion' not in scan_params or scan_params.get('motion') is None:
        scan_params['motion'] = True
    
    if 'scan_avg' not in scan_params or scan_params.get('scan_avg') is None:
        scan_params['scan_avg'] = 2
    
    if 'sfrac' not in scan_params or scan_params.get('sfrac') is None:
        scan_params['sfrac'] = 2
    
    if 'verbose' not in scan_params or scan_params.get('verbose') is None:
        scan_params['verbose'] = 1
    
    return scan_params


def check_spike_opts(spike_opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for spike/fluorescence simulation options.
    
    This function replicates the MATLAB check_spike_opts.m logic.
    
    Args:
        spike_opts: User-provided spike options (can be None or partial)
    
    Returns:
        Complete spike options dictionary with all required fields
    """
    # Initialize empty dict if None
    if spike_opts is None:
        spike_opts = {}
    
    # Set defaults using the same logic as MATLAB (checking isfield/isempty)
    defaults = {
        'K': 30,                        # Default to 30 neurons
        'mu': 0,                         # Default mean of normal r.v. used in log-normal
        'sig': 1,                        # Default std-dev of normal r.v. used in log-normal
        'dyn_type': 'Ca_DE',            # Default dynamics type
        'rate_dist': 'gamma',           # Default rate distribution
        'dt': 1/30,                     # Default sampling rate is 30 Hz
        'nt': 1000,                     # Default number of time-steps
        'rate': 1e-3,                   # Default inverse average of 1s between bursts
        'N_bg': 0,                      # Default to only one background component
        'prot': 'GCaMP6',               # Default protein type
        'alpha': 1,                     # Default Gamma distribution parameter (Exponential)
        'burst_mean': 10,               # Default mean of Poisson for spikes per burst
        'smod_flag': 'hawkes',          # Default simulation model
        'p_off': 0.2,                   # Default parameter
        'selfact': 1.2,                 # Default parameter
        'min_mod': [0.4, 2.53],         # Shape parameter estimate
        'spikeflag': 1,                 # Flag to save spike data
        'dendflag': 1,                  # Flag to simulate dendrite traces
        'axonflag': 1,                  # Flag to simulate axon traces
    }
    
    # Apply defaults only for missing or None values
    for key, default_value in defaults.items():
        if key not in spike_opts or spike_opts.get(key) is None:
            spike_opts[key] = default_value
    
    return spike_opts


def check_noise_params(noise_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for noise model parameters.
    
    This function replicates the MATLAB check_noise_params.m logic.
    
    Args:
        noise_params: User-provided noise parameters (can be None or partial)
    
    Returns:
        Complete noise parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if noise_params is None:
        noise_params = {}
    
    # Set defaults (matching MATLAB check_noise_params.m)
    defaults = {
        'mu': 100,                      # Default mean measurement increase per photon
        'mu0': 0,                       # Default electronics offset
        'sigma': 2300,                  # Default variance increase per photon
        'sigma0': 2.7,                  # Default electronics base noise variance
        'darkcount': 0.05,              # Default PMT dark count rate
        'sigscale': 2e-7,               # Default signal magnitude scale
        'bleedp': 0.3,                  # Default electronics pixel bleed through probability
        'bleedw': 0.4,                  # Default electronics average pixel bleed-through
    }
    
    # Apply defaults only for missing or None values
    for key, default_value in defaults.items():
        if key not in noise_params or noise_params.get(key) is None:
            noise_params[key] = default_value
    
    # Note: MATLAB code has duplicate checks for bleedp and bleedw with different defaults
    # The later ones (0.2 and 0.5) would overwrite, but we'll use the first set (0.3, 0.4)
    # as they appear first in the logical flow
    
    return noise_params


def check_vasc_params(vasc_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for vasculature parameters.

    This function replicates the MATLAB check_vasc_params.m logic.

    Args:
        vasc_params: User-provided vasculature parameters (can be None or partial)

    Returns:
        Complete vasculature parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if vasc_params is None:
        vasc_params = {}

    # Define default parameters (matching MATLAB check_vasc_params.m)
    d_params = {
        'flag': 1,                                    # On/off flag for vasculature simulation
        'ves_shift': np.array([5.0, 15.0, 5.0]),     # Amount of wobble allowed for blood vessels (um)
        'depth_vasc': 200.0,                          # Depth into tissue for vasculature simulation (um)
        'depth_surf': 15.0,                           # Depth into tissue of surface vasculature (um)
        'distWeightScale': 2.0,                       # Scaling factor for node distance weights
        'randWeightScale': 0.1,                       # Scaling factor for node variability weights
        'cappAmpScale': 0.5,                          # Scaling factor for capillary lateral weights
        'cappAmpZscale': 0.5,                         # Scaling factor for capillary axial weights
        'vesSize': np.array([15.0, 9.0, 2.0]),       # Vessel radii [surface, axial, capillaries] (um)
        'vesFreq': np.array([125.0, 200.0, 50.0]),   # Vessel frequencies [surface, axial, capillaries] (um)
        'sourceFreq': 1000.0,                         # Rate of source node generation (um/node)
        'vesNumScale': 0.2,                           # Vessel number random scaling factor
        'sepweight': 0.75,                            # Weight guiding node placement distance
        'distsc': 4.0,                                # Strength of local capillary connections
    }

    # Merge defaults with user parameters
    vasc_params = set_params(d_params, vasc_params)

    # Handle node_params sub-structure
    if 'node_params' not in vasc_params or vasc_params.get('node_params') is None:
        # Create default node_params
        vasc_params['node_params'] = {
            'maxit': 25,                              # Maximum iterations to place nodes
            'lensc': 50.0,                            # Average distance between branch points (um)
            'varsc': 15.0,                            # Standard deviation of distances (um)
            'mindist': 10.0,                          # Minimum inter-node distance (um)
            'varpos': 5.0,                            # Standard deviation of placement (um)
            'dirvar': np.pi/8,                        # Maximum branching angle (radians)
            'branchp': 0.02,                          # Probability of branching surface vasculature
            'vesrad': 25.0,                           # Radius of surface vasculature (um)
        }
    else:
        # Check and fill individual node_params fields
        node_defaults = {
            'maxit': 25,
            'lensc': 50.0,
            'varsc': 15.0,
            'mindist': 10.0,
            'varpos': 5.0,
            'dirvar': np.pi/8,
            'branchp': 0.02,
            'vesrad': 25.0,
        }

        for key, default_value in node_defaults.items():
            if key not in vasc_params['node_params'] or vasc_params['node_params'].get(key) is None:
                vasc_params['node_params'][key] = default_value

    return vasc_params


def check_tpm_params(tpm_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for TPM (Two-Photon Microscopy) parameters.
    
    This function replicates the MATLAB check_tpm_params.m logic.
    
    Args:
        tpm_params: User-provided TPM parameters (can be None or partial)
    
    Returns:
        Complete TPM parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if tpm_params is None:
        tpm_params = {}
    
    # Define default parameters (matching MATLAB check_tpm_params.m)
    d_params = {
        'nidx': 1.33,                   # Index of refraction (water)
        'nac': 0.8,                     # Objective NA
        'phi': None,                    # Will be calculated if None
        'eta': 0.6,                     # eGFP quantum yield
        'conc': 10,                     # Fluorophore concentration (uM)
        'delta': 2,                     # Two-photon abs. cross section (GM) - estimate at resting calcium levels (saturated GCaMP is 35)
        'gp': 0.588,                    # Pulse-shape temporal coherence
        'f': 80,                        # Ti:S laser rep rate (MHz)
        'tau': 150,                     # Ti:S laser pulse-width (fs)
        'pavg': 40,                     # Laser power (mW)
        'lambda': 0.92,                 # Wavelength for GCaMP excitation
    }
    
    # Merge defaults with user parameters
    tpm_params = set_params(d_params, tpm_params)
    
    # Calculate phi if not provided
    # Formula from MATLAB: 0.8*((1-sqrt(1-(nac/nidx)^2))/2)*0.4
    # This represents: 80% transmission * solid angle * 40% PMT QE
    if tpm_params.get('phi') is None:
        nac = tpm_params.get('nac', 0.8)
        nidx = tpm_params.get('nidx', 1.33)
        # Calculate solid angle: (1-sqrt(1-(nac/nidx)^2))/2
        solid_angle = (1 - np.sqrt(1 - (nac / nidx) ** 2)) / 2
        tpm_params['phi'] = 0.8 * solid_angle * 0.4
    
    return tpm_params


def check_neur_params(neur_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for neuron generation parameters.

    This function replicates the MATLAB check_neur_params.m logic.

    Args:
        neur_params: User-provided neuron parameters (can be None or partial)

    Returns:
        Complete neuron parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if neur_params is None:
        neur_params = {}

    # Define default parameters (matching MATLAB check_neur_params.m)
    d_params = {
        'n_samps': 1000,                # Number of sphere samples
        'l_scale': 105.0,               # Length-scale for GP bumpiness
        'p_scale': 95.0,                # Variance for GP shape
        'avg_rad': 5.5,                 # Average neuron radius (um)
        'nuc_fluorsc': 0.3,             # Nuclear fluorescence
        'min_thic': np.array([1.0, 1.0]), # Minimum cytoplasmic thickness [soma, nucleus]
        'eccen': 0.25,                  # Maximum eccentricity
        'exts': np.array([0.75, 1.7]),  # Radius bounds [min, max]
        'nexts': np.array([60.0, 20.0]), # Nuclear shrink/smooth params
        'neur_type': 'pyr',             # Neuron type ('pyr' or 'peanut')
        'nuc_rad': None,                # Nuclear radius scaling (optional)
        'max_ang': 20.0,                # Maximum rotation angle
        'fluor_dist': np.array([1.0, 0.2]),  # Fluorescence distribution [length_scale, variance]
    }

    # Merge defaults with user parameters
    neur_params = set_params(d_params, neur_params)

    # Ensure array parameters are numpy arrays
    for key in ['min_thic', 'exts', 'nexts']:
        if neur_params[key] is not None and not isinstance(neur_params[key], np.ndarray):
            neur_params[key] = np.array(neur_params[key])

    return neur_params


def check_dend_params(dend_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for dendrite generation parameters.

    This function replicates the MATLAB check_dend_params.m logic.

    Args:
        dend_params: User-provided dendrite parameters (can be None or partial)

    Returns:
        Complete dendrite parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if dend_params is None:
        dend_params = {}

    # Define default parameters (matching MATLAB check_dend_params.m)
    d_params = {
        'dtParams': np.array([35.0, 100.0, 50.0, 1.0, 10.0]),  # Dendritic tree params
        'atParams': np.array([1.0, 5.0, 2.0, 2.0, 4.0]),       # Apical dendrite params
        'dweight': 10.0,                # Path planning randomness weight
        'bweight': 50.0,                # Obstruction avoidance weight
        'thicknessScale': 0.75,         # Thickness scaling
        'dims': np.array([20.0, 20.0, 20.0]),        # Spatial dimensions
        'dimsSS': np.array([10.0, 10.0, 10.0]),      # Subsampling factors
        'rallexp': 2.0,                 # Relaxation exponent
        'weightScale': np.array([150.0, 1.0, 0.8]), # Fluorescence weights [distance_scale, weight, variation]
    }

    # Merge defaults with user parameters
    dend_params = set_params(d_params, dend_params)

    # Ensure array parameters are numpy arrays
    for key in ['dtParams', 'atParams', 'dims', 'dimsSS']:
        if dend_params[key] is not None and not isinstance(dend_params[key], np.ndarray):
            dend_params[key] = np.array(dend_params[key])

    return dend_params


def check_bg_params(bg_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for background generation parameters.

    This function replicates the MATLAB check_bg_params.m logic.

    Args:
        bg_params: User-provided background parameters (can be None or partial)

    Returns:
        Complete background parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if bg_params is None:
        bg_params = {}

    # Define default parameters (matching MATLAB check_bg_params.m)
    d_params = {
        'flag': 1,                      # Enable/disable background generation
        'distvar': 5.0,                 # Distance variation
        'distscale': 2.0,               # Distance scaling
        'nanchors': 5,                  # Number of anchor points
        'numptssc': 20,                 # Points scaling factor
    }

    # Merge defaults with user parameters
    bg_params = set_params(d_params, bg_params)

    return bg_params


def check_axon_params(axon_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Check and set default values for axon generation parameters.

    This function replicates the MATLAB check_axon_params.m logic.

    Args:
        axon_params: User-provided axon parameters (can be None or partial)

    Returns:
        Complete axon parameters dictionary with all required fields
    """
    # Initialize empty dict if None
    if axon_params is None:
        axon_params = {}

    # Define default parameters (matching MATLAB check_axon_params.m)
    d_params = {
        'flag': 0,                      # Enable/disable axon generation (default disabled)
        'distvar': 5.0,                 # Distance variation
        'distscale': 2.0,               # Distance scaling
        'nanchors': 5,                  # Number of anchor points
        'numptssc': 20,                 # Points scaling factor
    }

    # Merge defaults with user parameters
    axon_params = set_params(d_params, axon_params)

    return axon_params


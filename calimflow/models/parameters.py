from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict
import numpy as np

@dataclass
class VolumeParams:
    """Parameters for the overall volume simulation."""
    # User-defined parameters
    size: np.ndarray = field(default_factory=lambda: np.array([100, 100, 30]))  # 3-element vector with size (in um)
    res: float = 2.0  # resolution to simulate volume (samples/um)
    depth: float = 100.0  # Depth under brain surface (um)
    verbose: int = 1  # Verbosity level (0,1,2)
    # Runtime parameters
    _size: np.ndarray = field(init=False)  # size of volume [px]
    _depth: float = field(init=False)  # depth of volume [px]

@dataclass
class VascParams:
    """Parameters for vasculature simulation."""
    # User-defined parameters
    depth_vasc: float = 200.0  # depth for vasculature simulation (um)
    depth_surf: float = 15.0  # depth of surface vasculature (um)
    source_freq: float = 1000.0  # source node generation rate
    ves_size: np.ndarray = field(default_factory=lambda: np.array([15, 9, 2]))  # vessel radius (surface, axial, capillaries) (um)
    ves_shift: np.ndarray = field(default_factory=lambda: np.array([5, 15, 5]))  # vessel wobble allowed (um)
    ves_freq: np.ndarray = field(default_factory=lambda: np.array([125, 200, 50]))  # vessel frequency
    ves_num_scale: float = 0.2  # vessel number scaling
    dist_weight_scale: float = 2.0  # scaling for node distance
    rand_weight_scale: float = 0.1  # additional node variability
    capp_amp_l_scale: float = 0.5  # scaling for capillaries (lateral)
    capp_amp_a_scale: float = 0.5  # scaling for capillaries (axial)
    sep_weight: float = 0.75  # node placement weight
    dist_sc: float = 4.0  # capillary connection strength
    maxcappdist: float = 100.0  # maximum distance between capillaries (um)
    # Runtime parameters
    _depth_vasc: float = field(init=False)  # depth of vasculature volume [px]
    _depth_surf: float = field(init=False)  # depth of surface vasculature [px]
    _size: np.ndarray = field(init=False)  # size of vasculature volume [px] 
    _szum: np.ndarray = field(init=False)  # size of vasculature volume [um] 
    _ves_size: np.ndarray = field(init=False)  # vessel radius [px]
    _ves_shift: np.ndarray = field(init=False)  # vessel wobble allowed [px]
    _mindists: np.ndarray = field(init=False)  # minimum distance between nodes [px]
    _maxcappdist: np.ndarray = field(init=False)  # maximum distance between capillaries [px]
    _nsource: int = field(init=False)  # number of source vessels from volume surface
    _ncapp: int = field(init=False)  # number of capillaries
    _nvert: int = field(init=False)  # number of penetrating vessels
    _nsurf: int = field(init=False)  # number of surface vessels
    _nnodes: int = field(init=False)  # number of nodes
    _nconn: int = field(init=False)  # number of edges
    _nlinks: int = field(init=False)  # number of links
    _nvert_sum: int = field(init=False)  # number of penetrating vessel connections to capillaries

@dataclass
class NodeParams:
    """Parameters for vessel node generation."""
    # User-defined parameters
    maxit: int = 20  # Maximum iteration to place nodes
    lensc: float = 50.0  # Average distance between vasculature branch points (um)
    varsc: float = 15.0  # Standard deviation of distances between branch points (um)
    mindist: float = 10.0  # Minimum inter-node distance (um)
    varpos: float = 5.0  # Standard deviation of vasculature placement (um)
    dirvar: float = np.pi/8  # Maximum branching angle (radians)
    branchp: float = 0.02  # Probability of branching surface vasculature
    # Runtime parameters
    _lensc: float = field(init=False)  # Average distance between vasculature branch points [px]
    _varsc: float = field(init=False)  # Standard deviation of distances between branch points [px]
    _mindist: float = field(init=False)  # Minimum inter-node distance [px]
    _varpos: float = field(init=False)  # Standard deviation of vasculature placement [px]

@dataclass
class NeuronParams:
    """Parameters for neuron generation."""
    # User-defined parameters
    n_samps: int = 1000  # Number of sphere samples for mesh
    l_scale: float = 5.0  # Length-scale for GP of soma shapes
    p_scale: float = 3.4  # Variance of GP of soma shape
    avg_rad: float = 5.5  # Average radius of neuron in um
    nuc_fluorsc: float = 0.3  # Nucleus fluorescence
    min_thic: float = 1.0  # Minimum cytoplasmic thickness
    eccen: float = 0.25  # Maximum eccentricity of neuron
    neur_type: str = 'default'  # Type of neuron ('default', 'pyr', 'peanut')
    fluor_dist: Optional[dict] = None  # Fluorescence distribution parameters
    
    # Runtime parameters (initialized during simulation)
    S_samp: Optional[np.ndarray] = field(default=None, init=False)  # Sphere sampling points
    Tri: Optional[np.ndarray] = field(default=None, init=False)  # Triangulation for mesh
    dists: Optional[np.ndarray] = field(default=None, init=False)  # Geodesic distances

@dataclass
class DendriteParams:
    """Parameters for dendrite generation."""
    dt_params: np.ndarray = field(default_factory=lambda: np.array([40, 150, 50, 1, 10]))  # [num_trees, radius_xy, radius_z, width_scale, variation]
    at_params: np.ndarray = field(default_factory=lambda: np.array([1, 5, 2, 2, 4]))  # [num_apical, radius_xy, radius_z, offset]
    d_weight: float = 10.0  # Weight for path planning randomness
    b_weight: float = 50.0  # Weight for obstruction
    thickness_scale: float = 0.75  # Scaling for dendrite thickness
    dims: np.ndarray = field(default_factory=lambda: np.array([20, 20, 20]))  # Dimensions for sampling
    dims_ss: np.ndarray = field(default_factory=lambda: np.array([10, 10, 10]))  # Subsampling dimensions
    rall_exp: float = 1.5  # Rall exponent for branching

@dataclass
class NeuronBody:
    """Represents a single neuron's physical structure."""
    vertices: np.ndarray  # Soma vertices
    nucleus_vertices: np.ndarray  # Nucleus vertices
    rotation: np.ndarray  # Rotation angles (Rx, Ry, Rz)
    location: np.ndarray  # 3D location

@dataclass
class NeuralVolume:
    """Contains the complete neural volume data."""
    soma: np.ndarray  # Volume indicating neuron locations
    fluorescence: np.ndarray  # Volume containing fluorescence levels
    nucleus_data: List[Tuple[np.ndarray, float]]  # Locations and fluorescence values for nuclei
    soma_data: List[np.ndarray]  # Locations for cell bodies
@dataclass
class PSFParams:
    """Parameters for PSF generation."""
    objNA: float = 0.8  # Objective numerical aperture
    NA: float = 0.6  # PSF numerical aperture
    lambda_: float = 920.0  # Wavelength in nm
    n: float = 1.33  # Refractive index
    res_lateral: float = 0.5  # Lateral resolution in microns
    res_axial: float = 2.0  # Axial resolution in microns

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not isinstance(self.scatter_sizes, np.ndarray):
            self.scatter_sizes = np.array(self.scatter_sizes)
        if not isinstance(self.scatter_weights, np.ndarray):
            self.scatter_weights = np.array(self.scatter_weights)
            
        # Ensure scatter_sizes is 2D with shape (N,1)
        if self.scatter_sizes.ndim == 1:
            self.scatter_sizes = self.scatter_sizes.reshape(-1, 1)
            
        # Ensure scatter_weights is 1D with same length as scatter_sizes
        if self.scatter_weights.shape[0] != self.scatter_sizes.shape[0]:
            raise ValueError("scatter_weights must have same length as scatter_sizes")

@dataclass
class CalciumParams:
    """Parameters for calcium dynamics simulation."""
    sat_type: str = 'Ca_DE'      # 'single', 'Ca_DE', or 'double'
    dt: float = 1/100            # Time step
    ext_rate: float = 292.3      # Extrusion rate
    ca_bind: float = 110         # Calcium binding ratio
    ca_rest: float = 50e-9       # Resting calcium concentration
    ind_con: float = 200e-6      # Indicator concentration
    ca_dis: float = 290e-9       # Calcium dissociation constant
    ca_sat: float = 1.0          # Calcium saturation parameter
    prot_type: str = 'GCaMP6'    # Protein type
    k_on: float = 1.0            # Binding rate
    k_off: float = 0.2           # Unbinding rate
    ca_amp: float = 76.1251      # Calcium transient amplitude
    t_on: float = 0.8535         # Rising time constant
    t_off: float = 98.6173       # Falling time constant

@dataclass
class ActivityParams:
    """Parameters for neural activity simulation."""
    n_timepoints: int = 2500  # Number of timepoints
    dt: float = 1/30  # Time step in seconds
    ca_decay: float = 1.0  # Calcium decay time constant
    noise_std: float = 0.1  # Noise standard deviation
    spike_rate: float = 0.1  # Average spike rate

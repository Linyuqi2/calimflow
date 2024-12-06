from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
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
    n_samps: int = 1000  # Number of sphere samples for mesh
    l_scale: float = 5.0  # Length-scale for GP of soma shapes
    p_scale: float = 3.4  # Variance of GP of soma shape
    avg_rad: float = 5.5  # Average radius of neuron in um
    nuc_fluorsc: float = 0.3  # Nucleus fluorescence
    min_thic: float = 1.0  # Minimum cytoplasmic thickness
    eccen: float = 0.25  # Maximum eccentricity of neuron
    neur_type: str = 'default'  # Type of neuron ('default', 'pyr', 'peanut')
    fluor_dist: Optional[dict] = None  # Fluorescence distribution parameters

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
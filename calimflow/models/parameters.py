from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict
import numpy as np

@dataclass
class VolumeParams:
    """Parameters for the overall volume simulation."""
    # User-defined parameters
    size: np.ndarray = field(default_factory=lambda: np.array([100, 100, 50]))
    #vasc sim only works with ints
    res: int = 2  # resolution to simulate volume (samples/um) 
    depth: int = 200  # Changed to match MATLAB default
    verbose: int = 1  # Verbosity level (0,1,2)
    mine_dist: float = 16.0 #Default minimum distance between neuron centers (Default = 16)
    dendrite_tau: float= 5.0 # Default dendrite decay strength exponential distance is 5
    n_den: Optional[float] = None  # Number of dendrites (will be calculated from AD_density if not provided)
    n_bg: int = int(1e6)  # Number of background/neuropil components
    n_neur: Optional[int] = None  # Number of neurons to generate
    neur_density: float = 1e5  # Neural density (neurons per cubic mm)
    AD_density: float = 2e3  # Apical dendrite density per square mm
    vasc_sz: Optional[np.ndarray] = None  # Size of vasculature volume (calculated from gaussian beam)
    # Runtime parameters
    _size: np.ndarray = field(init=False)  # size of volume [px]

    def __post_init__(self):
        """Calculate dependent parameters after initialization."""
        # Handle n_neur calculation
        if self.n_neur is None:
            volume_mm3 = np.prod(self.size) / 1e9  # convert um^3 to mm^3
            self.n_neur = int(np.ceil(self.neur_density * volume_mm3))
        else:
            self.neur_density = self.n_neur * 1e9 / np.prod(self.size)

        # Handle n_den calculation
        if self.n_den is None:
            area_mm2 = np.prod(self.size[:2]) / 1e6  # convert um^2 to mm^2
            self.n_den = self.AD_density * area_mm2
            if self.n_den == 0:
                self.n_den = 10  # Default fallback value if calculation results in 0

        # Ensure volume depth is multiple of 10 (matching MATLAB)
        if self.size[2] % 10 != 0:
            self.size[2] = (self.size[2] / 10)

@dataclass
class VascParams:
    """Parameters for vasculature simulation."""
    # User-defined parameters
    depth_vasc: float = 200.0  # Matches MATLAB
    depth_surf: float = 15.0  # Matches MATLAB
    source_freq: float = 1000.0  # Matches MATLAB
    ves_size: np.ndarray = field(default_factory=lambda: np.array([15, 9, 2]))  # Matches MATLAB
    ves_shift: np.ndarray = field(default_factory=lambda: np.array([5, 15, 5]))  # Matches MATLAB
    ves_freq: np.ndarray = field(default_factory=lambda: np.array([125, 200, 50])) 
    ves_num_scale: float = 0.2  # Matches MATLAB
    dist_weight_scale: float = 2.0  # scaling for node distance
    rand_weight_scale: float = 0.1  # additional node variability
    capp_amp_l_scale: float = 0.5  # scaling for capillaries (lateral)
    capp_amp_a_scale: float = 0.5  # scaling for capillaries (axial)
    sep_weight: float = 0.75  # node placement weight
    dist_sc: float = 4.0  # capillary connection strength
    maxcappdist: float = 100.0  # Matches MATLAB
    # Runtime parameters
    _depth_vasc: float = field(init=False)  # depth of vasculature volume [px]
    _depth_surf: float = field(init=False)  # depth of surface vasculature [px]
    _size: np.ndarray = field(init=False)  # size of vasculature volume [px] 
    _szum: np.ndarray = field(init=False)  # size of vasculature volume [um] 
    _ves_size: np.ndarray = field(init=False)  # vessel radius [px]
    _ves_shift: np.ndarray = field(init=False)  # vessel wobble allowed [px]
    _mindists: np.ndarray = field(init=False)  # minimum distance between nodes [px]
    _maxcappdist: np.ndarray = field(init=False)  # maximum distance between capillaries [px]
    _nsource: int = field(init=False, default=0)
    _ncapp: int = field(init=False, default=0)
    _nvert: int = field(init=False, default=0)
    _nsurf: int = field(init=False, default=0)
    _nnodes: int = field(init=False, default=0)
    _nconn: int = field(init=False, default=0)
    _nlinks: int = field(init=False, default=0)
    _nvert_sum: int = field(init=False, default=0)
    _nodes: int = field(init=False, default=0)  # Added missing nodes count

    def __post_init__(self):
        """Initialize dependent parameters."""
        # Ensure parameters are properly initialized with defaults if not set
        if not hasattr(self, 'source_freq'):
            self.source_freq = 1000.0  # Default spacing between source vessels
        if not hasattr(self, 'ves_freq'):
            self.ves_freq = np.array([125, 200, 50])  # Default vessel frequencies
        if not hasattr(self, 'ves_num_scale'):
            self.ves_num_scale = 0.2  # Default vessel number scaling

@dataclass
class NodeParams:
    """Parameters for vessel node generation."""
    # User-defined parameters
    maxit: int = 25  # Changed to match MATLAB
    lensc: float = 50.0  # Matches MATLAB
    varsc: float = 15.0  # Matches MATLAB
    mindist: float = 10.0  # Matches MATLAB
    varpos: float = 5.0  # Matches MATLAB
    dirvar: float = np.pi/8  # Matches MATLAB
    branchp: float = 0.02  # Matches MATLAB
    
    # Runtime parameters
    _lensc: float = field(init=False)  # Average distance between vasculature branch points [px]
    _varsc: float = field(init=False)  # Standard deviation of distances between branch points [px]
    _mindist: float = field(init=False)  # Minimum inter-node distance [px]
    _varpos: float = field(init=False)  # Standard deviation of vasculature placement [px]

    def __post_init__(self):
        """Initialize dependent parameters."""
        # Initialize runtime parameters with same values as user parameters
        # These will be scaled by resolution later
        self._lensc = self.lensc
        self._varsc = self.varsc
        self._mindist = self.mindist
        self._varpos = self.varpos

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
    exts =  [0.75,1.7]
    # Runtime parameters (initialized during simulation)
    S_samp: Optional[np.ndarray] = field(default=None, init=False)  # Sphere sampling points
    Tri: Optional[np.ndarray] = field(default=None, init=False)  # Triangulation for mesh
    dists: Optional[np.ndarray] = field(default=None, init=False)  # Geodesic distances

@dataclass
class DendriteParams:
    """Parameters for dendrite generation."""
    # Default parameters matched with MATLAB implementation
    dt_params: np.ndarray = field(default_factory=lambda: np.array([40, 150, 50, 1, 10]))  # dendritic tree [number, radius_xy, radius_z, width_scale, variation]
    at_params: np.ndarray = field(default_factory=lambda: np.array([6, 5, 5, 5, 1]))  # L2/3 apical [number, radius_xy, radius_z, offset, scale]
    at_params2: np.ndarray = field(default_factory=lambda: np.array([1, 5, 5, 5, 4]))  # L5 apical [number, radius_xy, radius_z, offset, scale]
    dweight: float = 10.0  # Weight for path planning randomness
    bweight: float = 5.0  # Weight for obstruction (changed from 50 to 5 to match MATLAB)
    thickness_scale: float = 0.5  # Scaling for dendrite thickness in um^2
    dims: np.ndarray = field(default_factory=lambda: np.array([60, 60, 60]))  # Dimensions at 10um per space
    dims_ss: np.ndarray = field(default_factory=lambda: np.array([5, 5, 5]))  # Subsampling factor (changed from 10 to 5)
    rall_exp: float = 1.5  # Rall exponent for branching
    weight_scale: np.ndarray = field(default_factory=lambda: np.array([150, 1.0, 0.8]))  # [distance scaling, weight, variation]
    dend_var: float = 0.25  # Dendrite variation parameter

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Convert lists to numpy arrays if needed
        if not isinstance(self.dt_params, np.ndarray):
            self.dt_params = np.array(self.dt_params)
        if not isinstance(self.at_params, np.ndarray):
            self.at_params = np.array(self.at_params)
        if not isinstance(self.at_params2, np.ndarray):
            self.at_params2 = np.array(self.at_params2)
        if not isinstance(self.dims, np.ndarray):
            self.dims = np.array(self.dims)
        if not isinstance(self.dims_ss, np.ndarray):
            self.dims_ss = np.array(self.dims_ss)
        if not isinstance(self.weight_scale, np.ndarray):
            self.weight_scale = np.array(self.weight_scale)

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
    # Keep existing parameters
    objNA: float = 0.8  # Objective numerical aperture
    NA: float = 0.6  # PSF numerical aperture
    lambda_: float = 920.0  # Wavelength in nm
    n: float = 1.35  # Refractive index
    res_lateral: float = 0.5  # Lateral resolution in microns
    res_axial: float = 2.0  # Axial resolution in microns
    n_diff: float = 0.02  # Shift in index of refraction from vessels to tissue
    obj_fl: float = 4.5  # Objective focal length (mm)
    ss: int = 2  # Subsampling factor for fresnel propagation
    sampling: int = 50  # Spatial sampling for tissue occlusion mask
    psf_sz: np.ndarray = field(default_factory=lambda: np.array([20, 20, 50]))  # PSF size simulated (microns)
    prop_sz: float = 10.0  # Fresnel propagation length outside of volume (microns)
    blur: float = 3.0  # PSF lateral blurring (microns)
    scatter_sizes: np.ndarray = field(default_factory=lambda: np.array([0.51, 1.56, 4.52, 14.78]))  # Scattering object sizes (microns)
    scatter_weights: np.ndarray = field(default_factory=lambda: np.array([0.57, 0.29, 0.19, 0.15]))  # Scattering object weights
    zernike_wt: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.12]))  # Microscope aberration weights
    taillength: float = 50.0  # Distance from edge of PSF_sz to estimate tailweight (um)
    psf_type: str = 'gaussian'  # PSF type ('gaussian','vtwins','bessel')
    scaling: str = 'two-photon'  # PSF scaling type ('two-photon','three-photon','temporal-focusing')
    hemoabs: float = 0.00674 * np.log(10)  # Hemoglobin absorbance scaling factor
    propcrop: bool = True  # Flag to crop scanned beam during optical propagation
    fastmask: bool = True  # Flag for fast masking
    FM: dict = field(default_factory=lambda: {
        'sampling': 10,
        'fineSamp': 2,
        'ss': 1
    })  # Fast mask parameters

    def __post_init__(self):
        """Initialize dependent parameters."""
        # ...existing validation code...
        
        # Additional initialization for fast mask parameters if needed
        if self.fastmask and not hasattr(self, 'FM'):
            self.FM = {
                'sampling': 10,
                'fineSamp': 2,
                'ss': 1
            }

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

@dataclass
class AxonParams:
    """Parameters for axon generation."""
    # User-defined parameters
    flag: int = 1  # Flag for generation of background dendrites
    distsc: float = 0.5  # Parameter for directed random walk (higher = more directed)
    fillweight: float = 100.0  # Maximum length for single process branch (um)
    maxlength: float = 200.0  # Maximum length for background processes (um)
    minlength: float = 10.0  # Minimum length for background processes (um)
    maxdist: float = 100.0  # Maximum distance to the end of a process (um)
    maxel: int = 8  # Max number of axons per voxel
    varfill: float = 0.3  # Variation in filling weight (std around 1)
    maxvoxel: int = 6  # Maximum number of elements per voxel
    padsize: int = 20  # Background padding size (for smoothness in background)
    numbranches: int = 20  # Number of allowable branches for a single process
    varbranches: int = 5  # Standard deviation of branches per process
    maxfill: float = 0.5  # Voxel maximum occupation fraction
    n_proc: int = 10  # Number of background components
    l: float = 25.0  # GP length scale for correlation
    rho: float = 0.1  # GP variance parameter for correlation

    # Runtime parameters
    _size: np.ndarray = field(init=False)  # Size in pixels
    _resolution: float = field(init=False)  # Resolution in pixels/um

    def __post_init__(self):
        """Initialize dependent parameters."""
        if not isinstance(self.maxel, int):
            self.maxel = int(self.maxel)
        if not isinstance(self.maxvoxel, int):
            self.maxvoxel = int(self.maxvoxel)
        if not isinstance(self.numbranches, int):
            self.numbranches = int(self.numbranches)
        if not isinstance(self.varbranches, int):
            self.varbranches = int(self.varbranches)
        if not isinstance(self.n_proc, int):
            self.n_proc = int(self.n_proc)

@dataclass
class BgParams:
    """Parameters for background process generation."""
    # User-defined parameters
    flag: int = 1  # Flag for generation of background dendrites
    distsc: float = 0.5  # Parameter for directed random walk (higher = more directed)
    fillweight: float = 100.0  # Maximum length for single process branch (um)
    maxlength: float = 200.0  # Maximum length for background processes (um)
    minlength: float = 10.0  # Minimum length for background processes (um)
    maxdist: float = 100.0  # Maximum distance to the end of a process (um)
    maxel: int = 1  # Max number of axons per voxel

    # Runtime parameters
    _size: np.ndarray = field(init=False)  # Size in pixels
    _resolution: float = field(init=False)  # Resolution in pixels/um

    def __post_init__(self):
        """Initialize dependent parameters."""
        if not isinstance(self.maxel, int):
            self.maxel = int(self.maxel)
        if not isinstance(self.flag, int):
            self.flag = int(self.flag)

"""
Data structures for neural volume simulation.

This module defines the core data structures used in neural volume generation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class NeurParams:
    """
    Neuron generation parameters.

    Attributes:
        n_samps: Number of sphere samples for mesh generation (default=1000)
        l_scale: Length-scale for GP shape bumpiness (default=105)
        p_scale: Variance for GP shape (default=95)
        avg_rad: Average neuron radius in um (default=5.5)
        nuc_fluorsc: Nuclear fluorescence (default=0.3)
        min_thic: Minimum cytoplasmic thickness (default=0.25)
        eccen: Maximum eccentricity (default=0.25)
        exts: Radius bounds [min, max] (default=[0.75, 1.7])
        nexts: Nuclear shrink/smooth params (default=[60, 20])
        neur_type: Neuron type ('pyr' or 'peanut', default='pyr')
        nuc_rad: Nuclear radius scaling (optional)
        max_ang: Maximum rotation angle (default=20)
    """
    n_samps: int = 1000
    l_scale: float = 105.0
    p_scale: float = 95.0
    avg_rad: float = 5.5
    nuc_fluorsc: float = 0.3
    min_thic: np.ndarray = None  # Will be initialized in __post_init__
    eccen: float = 0.25
    exts: np.ndarray = None  # Will be initialized in __post_init__
    nexts: np.ndarray = None  # Will be initialized in __post_init__
    neur_type: str = 'pyr'
    nuc_rad: Optional[np.ndarray] = None
    max_ang: float = 20.0

    def __post_init__(self):
        """Initialize array parameters"""
        if self.min_thic is None:
            self.min_thic = np.array([1.0, 1.0])
        elif not isinstance(self.min_thic, np.ndarray):
            self.min_thic = np.array(self.min_thic)

        if self.exts is None:
            self.exts = np.array([0.75, 1.7])
        elif not isinstance(self.exts, np.ndarray):
            self.exts = np.array(self.exts)

        if self.nexts is None:
            self.nexts = np.array([60.0, 20.0])
        elif not isinstance(self.nexts, np.ndarray):
            self.nexts = np.array(self.nexts)


@dataclass
class DendParams:
    """
    Dendrite generation parameters.

    Attributes:
        dtParams: Dendritic tree params [num_branches, rad_xy, rad_z, width_scale, var_num]
        atParams: Apical dendrite params [num, rad_xy, rad_z, offset, spacing]
        dweight: Path planning randomness weight (default=10)
        bweight: Obstruction avoidance weight (default=50)
        thicknessScale: Thickness scaling (default=0.75)
        dims: Spatial dimensions [x, y, z] (default=[20, 20, 20])
        dimsSS: Subsampling factors [x, y, z] (default=[10, 10, 10])
        rallexp: Relaxation exponent (default=2)
    """
    dtParams: np.ndarray = None
    atParams: np.ndarray = None
    dweight: float = 10.0
    bweight: float = 50.0
    thicknessScale: float = 0.75
    dims: np.ndarray = None
    dimsSS: np.ndarray = None
    rallexp: float = 2.0

    def __post_init__(self):
        """Initialize array parameters"""
        if self.dtParams is None:
            self.dtParams = np.array([35.0, 100.0, 50.0, 1.0, 10.0])
        elif not isinstance(self.dtParams, np.ndarray):
            self.dtParams = np.array(self.dtParams)

        if self.atParams is None:
            self.atParams = np.array([1.0, 5.0, 2.0, 2.0, 4.0])
        elif not isinstance(self.atParams, np.ndarray):
            self.atParams = np.array(self.atParams)

        if self.dims is None:
            self.dims = np.array([20.0, 20.0, 20.0])
        elif not isinstance(self.dims, np.ndarray):
            self.dims = np.array(self.dims)

        if self.dimsSS is None:
            self.dimsSS = np.array([10.0, 10.0, 10.0])
        elif not isinstance(self.dimsSS, np.ndarray):
            self.dimsSS = np.array(self.dimsSS)


@dataclass
class BgParams:
    """
    Background/neuropil generation parameters.

    Attributes:
        flag: Enable/disable background generation (default=1)
        distvar: Distance variation (default=5)
        distscale: Distance scaling (default=2)
        nanchors: Number of anchor points (default=5)
        numptssc: Points scaling factor (default=20)
    """
    flag: int = 1
    distvar: float = 5.0
    distscale: float = 2.0
    nanchors: int = 5
    numptssc: int = 20


@dataclass
class AxonParams:
    """
    Axon generation parameters.

    Attributes:
        flag: Enable/disable axon generation (default=0)
        distvar: Distance variation (default=5)
        distscale: Distance scaling (default=2)
        nanchors: Number of anchor points (default=5)
        numptssc: Points scaling factor (default=20)
    """
    flag: int = 0
    distvar: float = 5.0
    distscale: float = 2.0
    nanchors: int = 5
    numptssc: int = 20


@dataclass
class NeuralVolume:
    """
    Output structure for neural volume simulation.

    Attributes:
        neur_vol: Fluorescence volume array
        gp_nuc: Nuclear fluorescence distributions (cell array)
        gp_soma: Soma fluorescence distributions (cell array)
        gp_vals: Full cellular fluorescence distributions
        neur_ves: Vessel locations
        bg_proc: Background processes
        neur_ves_all: Full vasculature volume
        locs: Neuron locations in microns
        gp_bgvals: Background/axonal fluorescence distributions
    """
    neur_vol: np.ndarray
    gp_nuc: List[Dict[str, Any]]
    gp_soma: List[np.ndarray]
    gp_vals: List[Dict[str, Any]]
    neur_ves: np.ndarray
    bg_proc: Optional[List] = None
    neur_ves_all: Optional[np.ndarray] = None
    locs: np.ndarray = None
    gp_bgvals: Optional[List] = None

    def __post_init__(self):
        """Ensure arrays are numpy arrays"""
        if not isinstance(self.neur_vol, np.ndarray):
            self.neur_vol = np.array(self.neur_vol)
        if not isinstance(self.neur_ves, np.ndarray):
            self.neur_ves = np.array(self.neur_ves)
        if self.neur_ves_all is not None and not isinstance(self.neur_ves_all, np.ndarray):
            self.neur_ves_all = np.array(self.neur_ves_all)
        if self.locs is not None and not isinstance(self.locs, np.ndarray):
            self.locs = np.array(self.locs)

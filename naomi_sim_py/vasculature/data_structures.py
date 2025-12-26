"""
Data structures for vasculature simulation.

This module defines the core data structures used in blood vessel network simulation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class VascNode:
    """
    Blood vessel node structure.

    Attributes:
        num: Identifying number
        root: Root node index (0 means no root)
        conn: List of connected node indices
        pos: 3D position [x, y, z] in pixels
        type: Node type ('edge', 'surf', 'vert', 'sfvt', 'capp')
        misc: Miscellaneous parameters (varies by node type)
    """
    num: int
    root: int
    conn: List[int]
    pos: np.ndarray  # shape: (3,)
    type: str
    misc: Optional[np.ndarray] = None

    def __post_init__(self):
        """Ensure pos is numpy array"""
        if not isinstance(self.pos, np.ndarray):
            self.pos = np.array(self.pos, dtype=float)
        if self.misc is not None and not isinstance(self.misc, np.ndarray):
            self.misc = np.array(self.misc)


@dataclass
class VascConnection:
    """
    Blood vessel connection structure.

    Attributes:
        start: Starting node index
        ends: Ending node index
        weight: Connection weight
        locs: Optional array of connection path points
        type: Connection type
    """
    start: int
    ends: int
    weight: float
    locs: Optional[np.ndarray] = None
    type: str = 'vessel'

    def __post_init__(self):
        """Ensure locs is numpy array if provided"""
        if self.locs is not None and not isinstance(self.locs, np.ndarray):
            self.locs = np.array(self.locs, dtype=float)


@dataclass
class VascNetwork:
    """
    Vasculature network parameters.

    Attributes:
        vol_sz: Volume size in microns [x, y, z]
        size: Volume size in pixels [x, y, z] (vol_sz * vres)
        szum: Volume size in microns (alias for vol_sz)
        nsource: Number of source vessels from volume surface
        nvert: Number of penetrating (vertical) vessels
        nsurf: Number of surface vessels
        ncapp: Number of capillaries
        nconn: Number of connections
        nlinks: Number of links (intermediate nodes)
    """
    vol_sz: np.ndarray  # shape: (3,)
    size: np.ndarray    # shape: (3,)
    szum: np.ndarray    # shape: (3,) - alias for vol_sz
    nsource: int
    nvert: int
    nsurf: int
    ncapp: int
    nconn: int = 0
    nlinks: int = 0

    def __post_init__(self):
        """Ensure arrays are numpy arrays"""
        for attr in ['vol_sz', 'size', 'szum']:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, np.ndarray):
                setattr(self, attr, np.array(val, dtype=int))


@dataclass
class NodeParams:
    """
    Node placement parameters.

    Attributes:
        maxit: Maximum iterations to place nodes
        lensc: Average distance between vasculature branch points (pixels)
        varsc: Standard deviation of distances between branch points (pixels)
        mindist: Minimum inter-node distance (pixels)
        varpos: Standard deviation of vasculature placement (pixels)
        dirvar: Maximum branching angle (radians)
        branchp: Probability of branching surface vasculature
        vesrad: Radius of surface vasculature (pixels)
    """
    maxit: int = 25
    lensc: float = 50.0
    varsc: float = 15.0
    mindist: float = 10.0
    varpos: float = 5.0
    dirvar: float = np.pi/8
    branchp: float = 0.02
    vesrad: float = 25.0


@dataclass
class VascParams:
    """
    Vasculature simulation parameters.

    Attributes:
        ves_shift: 3-vector of wobble allowed for blood vessels [x, y, z] (um)
        depth_vasc: Depth into tissue for vasculature simulation (pixels)
        depth_surf: Depth into tissue of surface vasculature (pixels)
        distWeightScale: Scaling factor for node distance weights
        randWeightScale: Scaling factor for node variability weights
        cappAmpScale: Scaling factor for capillary lateral weights
        cappAmpZscale: Scaling factor for capillary axial weights
        vesSize: Vessel radii [surface, axial, capillaries] (pixels)
        vesFreq: Blood vessel frequencies [surface, axial, capillaries] (um)
        vesNumScale: Blood vessel number random scaling factor
        sourceFreq: Rate of source node generation (um/node)
        sepweight: Weight guiding node placement distance (0-1)
        distsc: Strength of local capillary connections
        mindists: Minimum distances between nodes (pixels)
        maxcappdist: Maximum capillary distance (pixels)
        node_params: Node placement parameters
    """
    ves_shift: np.ndarray = None
    depth_vasc: float = 200.0
    depth_surf: float = 15.0
    distWeightScale: float = 2.0
    randWeightScale: float = 0.1
    cappAmpScale: float = 0.5
    cappAmpZscale: float = 6.0
    vesSize: np.ndarray = None
    vesFreq: np.ndarray = None
    vesNumScale: float = 0.2
    sourceFreq: float = 1000.0
    sepweight: float = 0.75
    distsc: float = 4.0
    mindists: Optional[np.ndarray] = None
    maxcappdist: Optional[float] = None
    node_params: NodeParams = None

    def __post_init__(self):
        """Initialize default arrays"""
        if self.ves_shift is None:
            self.ves_shift = np.array([30.0, 15.0, 15.0])
        if self.vesSize is None:
            self.vesSize = np.array([15.0, 6.0, 2.0])
        if self.vesFreq is None:
            self.vesFreq = np.array([125.0, 200.0, 50.0])
        if self.node_params is None:
            self.node_params = NodeParams()


# Utility functions for data structure creation
def create_vasc_node(num: int, root: int = 0, conn: List[int] = None,
                    pos: np.ndarray = None, node_type: str = '',
                    misc: np.ndarray = None) -> VascNode:
    """
    Create a new VascNode with default values.

    Args:
        num: Node identifying number
        root: Root node index (0 for no root)
        conn: List of connected node indices
        pos: 3D position array
        node_type: Node type string
        misc: Miscellaneous parameters

    Returns:
        New VascNode instance
    """
    if conn is None:
        conn = []
    if pos is None:
        pos = np.zeros(3)
    if misc is None:
        misc = np.array([])

    return VascNode(
        num=num,
        root=root,
        conn=conn,
        pos=pos,
        type=node_type,
        misc=misc
    )


def create_vasc_connection(start: int, ends: int, weight: float = 1.0,
                          locs: np.ndarray = None, conn_type: str = 'vessel') -> VascConnection:
    """
    Create a new VascConnection.

    Args:
        start: Starting node index
        ends: Ending node index
        weight: Connection weight
        locs: Optional path points array
        conn_type: Connection type

    Returns:
        New VascConnection instance
    """
    return VascConnection(
        start=start,
        ends=ends,
        weight=weight,
        locs=locs,
        type=conn_type
    )


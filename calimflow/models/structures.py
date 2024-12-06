from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np

@dataclass
class Node:
    """Represents a node in the vasculature network."""
    num: int
    root: int
    conn: List[int]
    pos: np.ndarray
    node_type: str
    misc: Optional[Union[List[bool], float]] = None

@dataclass
class Connection:
    """Represents a connection between nodes in the vasculature network."""
    start: int
    ends: int
    weight: float
    misc: Optional[str] = None  # Additional information about the connection
    locs: Optional[np.ndarray] = None  # Added to store vessel locations 
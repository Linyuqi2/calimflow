"""
Dendrite pathfinding algorithms using Dijkstra and random walk.

This module implements the pathfinding algorithms used for dendrite growth
in neural volume simulation.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def dendrite_dijkstra2(M: np.ndarray, dims: np.ndarray, root: np.ndarray) -> np.ndarray:
    """
    Dijkstra algorithm for dendrite pathfinding.

    This function replicates the MATLAB dendrite_dijkstra2.m functionality.
    It finds shortest paths in a 6-connected 3D grid using Dijkstra's algorithm.

    Args:
        M: Cost matrix of shape (prod(dims), 6) where each row contains costs
           for the 6 directions: [right, left, up, down, top, bottom]
        dims: 3-element array [nx, ny, nz] specifying grid dimensions
        root: 3-element array specifying the root node coordinates

    Returns:
        pathfrom: Array of shape (prod(dims),) containing the predecessor
                 of each node in the shortest path tree
    """
    nx, ny, nz = dims.astype(int)
    total_nodes = nx * ny * nz

    # Reshape M to be (total_nodes, 6)
    if M.shape[0] != total_nodes:
        M = M.reshape(total_nodes, 6)

    # Create adjacency matrix for 6-connected 3D grid
    # Directions: [right, left, up, down, top, bottom]
    directions = np.array([
        [1, 0, 0],   # right
        [-1, 0, 0],  # left
        [0, 1, 0],   # up
        [0, -1, 0],  # down
        [0, 0, 1],   # top
        [0, 0, -1]   # bottom
    ])

    # Initialize sparse adjacency matrix
    row_indices = []
    col_indices = []
    data = []

    for node_idx in range(total_nodes):
        # Convert linear index to 3D coordinates
        x = node_idx % nx
        y = (node_idx // nx) % ny
        z = node_idx // (nx * ny)

        for dir_idx, direction in enumerate(directions):
            # Calculate neighbor coordinates
            nx_coord = x + direction[0]
            ny_coord = y + direction[1]
            nz_coord = z + direction[2]

            # Check bounds
            if (0 <= nx_coord < nx and
                0 <= ny_coord < ny and
                0 <= nz_coord < nz):

                neighbor_idx = nx_coord + ny_coord * nx + nz_coord * nx * ny

                # Add edge with cost from M
                row_indices.append(node_idx)
                col_indices.append(neighbor_idx)
                data.append(M[node_idx, dir_idx])

    # Create sparse adjacency matrix
    adjacency = csr_matrix((data, (row_indices, col_indices)),
                          shape=(total_nodes, total_nodes))

    # Convert root coordinates to linear index
    root_idx = int(root[0] + root[1] * nx + root[2] * nx * ny)

    # Run Dijkstra from root
    dist_matrix, predecessors = dijkstra(adjacency, indices=[root_idx],
                                        return_predecessors=True)

    # Extract predecessors array
    pathfrom = predecessors[0]

    return pathfrom


def getDendritePath2(pathfrom: np.ndarray, end: np.ndarray,
                    root: np.ndarray, dims: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract path from Dijkstra predecessor array.

    This function replicates the MATLAB getDendritePath2.m functionality.

    Args:
        pathfrom: Predecessor array from Dijkstra (1D array with predecessor indices)
        end: End point coordinates [x, y, z]
        root: Root point coordinates [x, y, z]
        dims: Grid dimensions [nx, ny, nz] (optional, for coordinate conversion)

    Returns:
        path: Nx3 array of path coordinates from root to end
    """
    path = []

    # Convert end coordinates to linear index
    if dims is not None:
        end_idx = np.ravel_multi_index(end.astype(int), dims)
        root_idx = np.ravel_multi_index(root.astype(int), dims)
    else:
        # Assume coordinates are already indices
        end_idx = end.astype(int)
        root_idx = root.astype(int)

    current_idx = end_idx

    # Trace back from end to root
    max_iterations = 10000  # Prevent infinite loops
    iterations = 0

    visited = set()  # Prevent cycles

    while iterations < max_iterations and current_idx.item() not in visited:
        visited.add(current_idx.item())

        # Convert linear index back to coordinates
        if dims is not None:
            current_coords = np.array(np.unravel_index(current_idx, dims))
        else:
            current_coords = np.array([current_idx, 0, 0])  # Simplified for 1D case

        path.append(current_coords)

        # Check if we've reached the root
        if current_idx == root_idx:
            break

        # Get predecessor
        if current_idx < len(pathfrom) and pathfrom[current_idx] >= 0:
            current_idx = int(pathfrom[current_idx])
        else:
            # No valid predecessor, path is invalid
            return np.array([])

        iterations += 1

    # Reverse path to go from root to end
    if path:
        path_array = np.array(path[::-1])

        # Remove root from path if it's duplicated
        if len(path_array) > 1 and np.allclose(path_array[0], root, atol=1e-6):
            path_array = path_array[1:]

        return path_array
    else:
        return np.array([])


def dendrite_randomwalk2(M: np.ndarray, root: np.ndarray, target: np.ndarray,
                        distsc: float, maxlength: float, fillweight: float,
                        maxel: int, minlength: float) -> np.ndarray:
    """
    Random walk algorithm for dendrite growth.

    This function replicates the MATLAB dendrite_randomwalk2.m functionality.
    It performs a directed random walk from root to target with constraints.

    Args:
        M: Cost matrix
        root: Starting point coordinates [x, y, z]
        target: Target point coordinates [x, y, z]
        distsc: Directionality parameter (higher = more directed)
        maxlength: Maximum path length
        fillweight: Filling weight parameter
        maxel: Maximum elements per voxel
        minlength: Minimum path length

    Returns:
        path: Nx3 array of path coordinates
    """
    # This is a simplified implementation of the random walk algorithm
    # The full MATLAB implementation is quite complex with many edge cases

    path = [root.copy()]
    current = root.copy()
    total_length = 0

    max_iterations = int(maxlength * 2)  # Prevent infinite loops

    for iteration in range(max_iterations):
        # Calculate direction to target
        direction = target - current
        distance = np.linalg.norm(direction)

        if distance < 1.0:
            # Close enough to target
            path.append(target.copy())
            break

        # Normalize direction
        if distance > 0:
            direction = direction / distance

        # Add some randomness based on distsc
        random_component = np.random.randn(3) * (1.0 / (1.0 + distsc))
        direction = direction + random_component
        direction = direction / np.linalg.norm(direction)

        # Take a step
        step_size = 1.0
        next_point = current + direction * step_size

        # Check bounds (simplified)
        next_point = np.clip(next_point, [0, 0, 0], [1000, 1000, 1000])  # Large bounds

        # Add to path
        path.append(next_point.copy())
        current = next_point
        total_length += step_size

        # Check length constraints
        if total_length > maxlength:
            break

    # Convert to numpy array
    if len(path) >= 2:
        return np.array(path[1:])  # Exclude starting point
    else:
        return np.array([])

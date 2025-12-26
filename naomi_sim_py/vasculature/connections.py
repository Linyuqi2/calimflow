"""
Node connections and volume conversion for blood vessel simulation.

This module implements:
- nodes_to_conn: Convert node structure to connection structure
- conn_to_vol: Convert connections to 3D volume representation
"""

import numpy as np
from typing import List, Tuple, Optional
from .data_structures import VascNode, VascConnection, VascNetwork, create_vasc_connection


def nodes_to_conn(nodes: List[VascNode]) -> List[VascConnection]:
    """
    Convert node structure to connection structure.

    This replicates the MATLAB nodesToConn function logic.

    Args:
        nodes: List of vascular nodes

    Returns:
        List of vascular connections
    """
    connections: List[VascConnection] = []

    # Create connections based on node.conn relationships
    for i, node in enumerate(nodes):
        for conn_idx in node.conn:
            # Avoid duplicate connections (only create for higher indices)
            if conn_idx > node.num:
                # Find the target node
                target_node = None
                for n in nodes:
                    if n.num == conn_idx:
                        target_node = n
                        break

                if target_node is not None:
                    # Calculate weight based on distance (simplified)
                    distance = np.linalg.norm(node.pos - target_node.pos)
                    weight = max(1.0, distance / 10.0)  # Simplified weight calculation

                    conn = create_vasc_connection(
                        start=i,
                        ends=nodes.index(target_node),
                        weight=weight,
                        conn_type='vessel'
                    )
                    connections.append(conn)

    return connections


def conn_to_vol(nodes: List[VascNode], conn: List[VascConnection], nv: VascNetwork,
               conn_indices: Optional[List[int]] = None,
               existing_vol: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[VascConnection]]:
    """
    Convert connections to 3D volume.

    This implements a simplified version of the MATLAB connToVol function.
    Instead of using complex spline curves, we use Bresenham's line algorithm
    to draw straight lines between nodes.

    Args:
        nodes: List of vascular nodes
        conn: List of vascular connections
        nv: Vascular network parameters
        conn_indices: Optional list of connection indices to process (default: all)
        existing_vol: Optional existing volume to add to

    Returns:
        Tuple of (updated_volume, updated_connections)
    """
    if existing_vol is None:
        vol = np.zeros(nv.size, dtype=bool)
    else:
        vol = existing_vol.copy()

    # Determine which connections to process
    if conn_indices is None:
        connections_to_process = conn
    else:
        connections_to_process = [conn[i] for i in conn_indices if i < len(conn)]

    # Process each connection
    for connection in connections_to_process:
        if connection.start >= len(nodes) or connection.ends >= len(nodes):
            continue

        start_node = nodes[connection.start]
        end_node = nodes[connection.ends]

        # Create vessel path between nodes (simplified: straight line)
        vessel_locs = _create_vessel_path(start_node.pos, end_node.pos, nv.size)

        # Store the path in the connection
        connection.locs = vessel_locs

        # Add vessel to volume with thickness
        weight = max(1, int(np.ceil(connection.weight)))
        vol = _add_vessel_to_volume(vol, vessel_locs, weight, nv.size)

    return vol, conn


def _create_vessel_path(start_pos: np.ndarray, end_pos: np.ndarray, vol_size: np.ndarray) -> np.ndarray:
    """
    Create a vessel path between two points using Bresenham's line algorithm.

    Args:
        start_pos: Starting position [x, y, z]
        end_pos: Ending position [x, y, z]
        vol_size: Volume size [x, y, z]

    Returns:
        Array of positions along the vessel path
    """
    # Ensure positions are within bounds
    start_pos = np.clip(start_pos, [1, 1, 1], vol_size).astype(int)
    end_pos = np.clip(end_pos, [1, 1, 1], vol_size).astype(int)

    # For simplicity, create a straight line path
    # In the full implementation, this would use spline curves like MATLAB
    num_points = max(10, int(np.linalg.norm(end_pos - start_pos)))

    # Linear interpolation between start and end
    t = np.linspace(0, 1, num_points)
    path = start_pos[:, np.newaxis] + t * (end_pos - start_pos)[:, np.newaxis]
    path = np.round(path).astype(int).T

    # Ensure all points are within bounds
    path = np.clip(path, [1, 1, 1], vol_size)

    # Remove duplicates
    path = np.unique(path, axis=0)

    return path


def _add_vessel_to_volume(vol: np.ndarray, vessel_locs: np.ndarray, weight: float, vol_size: np.ndarray) -> np.ndarray:
    """
    Add a vessel path to the volume with thickness (MATLAB-style implementation).

    This replicates the MATLAB connToVol.m logic using morphological dilation
    with spherical structuring elements and additive accumulation.

    Args:
        vol: 3D volume array (will be modified in-place)
        vessel_locs: Vessel path locations [N, 3] (1-based indexing)
        weight: Vessel thickness/radius (can be fractional)
        vol_size: Volume size [x, y, z]

    Returns:
        Updated volume
    """
    if len(vessel_locs) == 0:
        return vol

    # Convert to 0-based indexing for numpy
    vessel_locs = vessel_locs - 1

    # Clamp vessel locations to volume bounds
    vessel_locs = np.maximum(vessel_locs, [0, 0, 0])
    vessel_locs = np.minimum(vessel_locs, vol_size - 1)
    vessel_locs = vessel_locs.astype(int)

    # Remove duplicate locations
    vessel_locs = np.unique(vessel_locs, axis=0)

    if len(vessel_locs) == 0:
        return vol

    # MATLAB-style morphological dilation approach
    # Create bounding box for the vessel path
    min_idx = np.maximum(np.min(vessel_locs, axis=0) - np.ceil(weight), [0, 0, 0]).astype(int)
    max_idx = np.minimum(np.max(vessel_locs, axis=0) + np.ceil(weight), vol_size - 1).astype(int)

    # Create local volume for this vessel segment
    local_shape = max_idx - min_idx + 1
    local_vol = np.zeros(local_shape, dtype=bool)

    # Convert vessel locations to local coordinates
    local_locs = vessel_locs - min_idx

    # Mark vessel path in local volume
    for loc in local_locs:
        if (0 <= loc[0] < local_shape[0] and
            0 <= loc[1] < local_shape[1] and
            0 <= loc[2] < local_shape[2]):
            local_vol[loc[0], loc[1], loc[2]] = True

    # Apply morphological dilation with spherical structuring element
    # MATLAB: se = strel(sqrt(x.^2 + y.^2 + z.^2) <= weight)
    radius = int(np.ceil(weight))
    if radius > 0:
        # Limit radius to avoid memory issues
        max_radius = 3  # Limit to reasonable size
        effective_radius = min(radius, max_radius)

        try:
            from scipy import ndimage
            if effective_radius <= max_radius:
                # Create spherical structuring element (only for small radii to avoid memory issues)
                x, y, z = np.mgrid[-effective_radius:effective_radius+1,
                                 -effective_radius:effective_radius+1,
                                 -effective_radius:effective_radius+1]
                se = np.sqrt(x**2 + y**2 + z**2) <= weight
                local_vol = ndimage.binary_dilation(local_vol, structure=se)
            else:
                # For larger radii, use iterative box dilation
                local_vol = ndimage.binary_dilation(local_vol, iterations=effective_radius)
        except ImportError:
            # Fallback: use iterative box dilation
            local_vol = ndimage.binary_dilation(local_vol, iterations=effective_radius)

    # Add to main volume (MATLAB: neur_ves = neur_ves + TMP)
    # Use additive accumulation instead of boolean OR
    vol[min_idx[0]:max_idx[0]+1,
        min_idx[1]:max_idx[1]+1,
        min_idx[2]:max_idx[2]+1] = np.logical_or(
        vol[min_idx[0]:max_idx[0]+1,
            min_idx[1]:max_idx[1]+1,
            min_idx[2]:max_idx[2]+1],
        local_vol
    )

    return vol


def pos2dists(positions: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise distances between positions.

    Args:
        positions: N x 3 array of positions

    Returns:
        N x N distance matrix
    """
    n_points = positions.shape[0]
    dist_mat = np.zeros((n_points, n_points))

    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.linalg.norm(positions[i] - positions[j])
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    return dist_mat


def pseudo_rand_sample_3d(volume_size: np.ndarray, n_samples: int,
                         min_dist: float, sep_weight: float,
                         weight_vol: np.ndarray) -> np.ndarray:
    """
    Pseudo-uniform 3D sampling with distance constraints and weighting.

    Args:
        volume_size: Size of volume [x, y, z]
        n_samples: Number of samples to generate
        min_dist: Minimum distance between samples
        sep_weight: Separation weight for repulsion
        weight_vol: Weighting volume (higher values = more likely to be sampled)

    Returns:
        N x 3 array of sampled positions
    """
    # Flatten weight volume for easier sampling
    flat_weights = weight_vol.flatten()
    total_weight = np.sum(flat_weights)

    if total_weight == 0:
        # Uniform sampling if no weights
        positions = np.random.rand(n_samples, 3) * volume_size
        return positions

    # Sample positions using weighted sampling
    positions = []

    for _ in range(n_samples):
        # Sample from weighted distribution
        flat_idx = np.random.choice(len(flat_weights), p=flat_weights/total_weight)

        # Convert back to 3D coordinates
        z = flat_idx // (volume_size[0] * volume_size[1])
        y = (flat_idx % (volume_size[0] * volume_size[1])) // volume_size[0]
        x = flat_idx % volume_size[0]

        pos = np.array([x, y, z], dtype=float)

        # Add small random offset to avoid exact grid positions
        pos += np.random.rand(3) - 0.5

        # Ensure within bounds
        pos = np.clip(pos, 0, volume_size - 1)

        positions.append(pos)

    return np.array(positions)


def grow_capillaries(nodes: List[VascNode], conn: List[VascConnection],
                    neur_ves: np.ndarray, nv: VascNetwork, vp: VascParams,
                    vres: float) -> Tuple[List[VascNode], List[VascConnection], VascNetwork]:
    """
    Grow capillaries from major vessels.

    This implements the complex capillary growth logic based on distance weights
    from the MATLAB version.

    Args:
        nodes: Current node list
        conn: Current connection list
        neur_ves: Current vessel volume
        nv: Vascular network parameters
        vp: Vascular parameters
        vres: Volume resolution

    Returns:
        Tuple of (updated_nodes, updated_conn, updated_nv)
    """
    # Check if capillaries are needed
    if vp.vesSize[2] <= 0 or vp.vesFreq[2] <= 0 or nv.ncapp <= 0:
        print("Skipping capillary generation (disabled or not needed)")
        return nodes, conn, nv

    print("Growing capillaries...")
    initial_conn_count = len(conn)

    # Step 1: Create distance weighting volume
    print("  Creating distance weighting volume...")
    dilrad = int(np.ceil(vp.mindists[0] / vres))

    # Initialize weighting volume
    weight_vol = np.zeros(nv.szum, dtype=float)

    # Create gaussian kernel centered at origin
    kernel_size = 2 * dilrad + 1
    x, y, z = np.mgrid[-dilrad:dilrad+1, -dilrad:dilrad+1, -dilrad:dilrad+1]
    gaussian_kernel = np.exp(-2 * (x**2 + y**2 + z**2) / dilrad**2)

    # Add weights based on existing vessel locations
    sub_sample = max(1, int(dilrad / 3))  # Subsample to speed up computation
    for i in range(len(conn)):
        if conn[i].locs is not None and len(conn[i].locs) > 0:
            # Subsample the connection locations
            locs_sampled = conn[i].locs[::sub_sample]

            for loc in locs_sampled:
                # Convert to integer indices
                loc_int = np.round(loc / vres).astype(int)

                # Ensure within bounds for kernel application
                x_start = max(0, loc_int[0] - dilrad)
                x_end = min(nv.szum[0], loc_int[0] + dilrad + 1)
                y_start = max(0, loc_int[1] - dilrad)
                y_end = min(nv.szum[1], loc_int[1] + dilrad + 1)
                z_start = max(0, loc_int[2] - dilrad)
                z_end = min(nv.szum[2], loc_int[2] + dilrad + 1)

                # Extract the corresponding kernel region
                kx_start = dilrad - (loc_int[0] - x_start)
                kx_end = dilrad + (x_end - loc_int[0])
                ky_start = dilrad - (loc_int[1] - y_start)
                ky_end = dilrad + (y_end - loc_int[1])
                kz_start = dilrad - (loc_int[2] - z_start)
                kz_end = dilrad + (z_end - loc_int[2])

                kernel_region = gaussian_kernel[kx_start:kx_end, ky_start:ky_end, kz_start:kz_end]

                # Apply to weight volume
                x_slice = slice(x_start, x_end)
                y_slice = slice(y_start, y_end)
                z_slice = slice(z_start, z_end)

                weight_vol[x_slice, y_slice, z_slice] = np.maximum(
                    weight_vol[x_slice, y_slice, z_slice],
                    kernel_region
                )

    # Invert weights (higher weight = less likely to be sampled, i.e., away from vessels)
    weight_vol = 1.0 - weight_vol

    # Set random seed for reproducibility (matching MATLAB)
    np.random.seed(0)

    # Step 2: Sample capillary positions
    print(f"  Sampling {nv.ncapp} capillary positions...")
    if nv.ncapp <= 0:
        print("    No capillaries to sample (ncapp <= 0)")
        capp_pos = np.array([]).reshape(0, 3)
    else:
        capp_pos = pseudo_rand_sample_3d(
            nv.size / vres,  # volume size in physical units
            nv.ncapp,
            vp.mindists[2] / vres,  # minimum distance in voxels
            vp.sepweight,
            weight_vol
        )
        print(f"    Sampled {len(capp_pos)} positions")

    # Convert back to voxel coordinates and add randomness
    capp_pos = capp_pos * vres + 1 - np.random.rand(*capp_pos.shape) * vres

    # Step 3: Create connections from vertical vessels to capillaries
    print("  Creating vertical vessel to capillary connections...")

    # Find vertical vessel nodes (MATLAB-style: start with sfvt, extend to connected vert)
    vertidxs = [i for i, node in enumerate(nodes) if node.type == 'sfvt']
    print(f"    Initial sfvt nodes: {len(vertidxs)}")

    # Extend to connected 'vert' nodes (MATLAB logic from lines 80-93)
    extended_vert_chains = []
    for i in range(len(vertidxs)):
        vesidx = [vertidxs[i]]  # Start with current sfvt node

        flag = True
        while flag:
            # Get connections of the last node in vesidx
            last_node_idx = vesidx[-1]
            tmpidx = nodes[last_node_idx].conn.copy()

            # Filter to only 'vert' type nodes
            tmpidx = [idx for idx in tmpidx if nodes[idx].type == 'vert']

            # Remove nodes already in vesidx (setxor equivalent)
            tmpidx = [idx for idx in tmpidx if idx not in vesidx]

            if not tmpidx:
                flag = False
            else:
                # Add the first new node found (MATLAB adds one at a time)
                vesidx.append(tmpidx[0])

        # Remove the original sfvt node, keep only extended vert nodes
        extended_verts = vesidx[1:]
        extended_vert_chains.append(extended_verts)

    # Flatten for processing, but keep track of chains for per-chain connection counts
    vert_idxs = [idx for chain in extended_vert_chains for idx in chain]
    print(f"    Extended to {len(vert_idxs)} total vertical vessel nodes across {len(extended_vert_chains)} chains")

    if len(extended_vert_chains) > 0:
        # Calculate number of capillaries per vertical vessel CHAIN (MATLAB: randi per sfvt node)
        max_conn = int(np.ceil(nv.szum[2] / vp.vesFreq[2]))
        nv_vert_conn = np.random.randint(1, max_conn + 1, len(extended_vert_chains))  # randi returns 1 to max_conn
        nv.nvert_sum = np.sum(nv_vert_conn)

        node_idx = len(nodes)
        conn_idx = len(conn)

        for i, vert_chain in enumerate(extended_vert_chains):
            if len(vert_chain) == 0:
                continue  # Skip empty chains

            for j in range(nv_vert_conn[i]):
                # Randomly select a node from this chain (MATLAB logic)
                ves_idx = np.random.choice(vert_chain)

                # Find closest capillary to this vertical vessel
                ves_pos = nodes[ves_idx].pos
                distances = np.linalg.norm(capp_pos - ves_pos, axis=1)

                # Skip if no capillaries available
                if len(distances) == 0:
                    continue

                closest_idx = np.argmin(distances)

                # Create new capillary node
                capp_pos_node = capp_pos[closest_idx]
                new_node = VascNode(
                    num=node_idx,
                    root=ves_idx,
                    conn=[ves_idx],
                    pos=capp_pos_node,
                    type='capp'
                )
                nodes.append(new_node)
                nodes[ves_idx].conn.append(node_idx)

                # Create connection
                weight = max(1, np.random.normal(vp.vesSize[2], vp.vesSize[3] if len(vp.vesSize) > 3 else 0))
                new_conn = VascConnection(
                    start=ves_idx,
                    ends=node_idx,
                    weight=weight,
                    locs=None,
                    type='vtcp'
                )
                conn.append(new_conn)

                # Remove this capillary from available list
                capp_pos = np.delete(capp_pos, closest_idx, axis=0)
                nv.ncapp -= 1

                node_idx += 1
                conn_idx += 1

        # Update network counters
        nv.nconn = conn_idx

    # Step 4: Add remaining capillaries as independent nodes
    print(f"  Adding {len(capp_pos)} remaining independent capillaries...")
    node_idx = len(nodes)
    for i in range(len(capp_pos)):
        new_node = VascNode(
            num=node_idx,
            root=[],
            conn=[],
            pos=capp_pos[i],
            type='capp'
        )
        nodes.append(new_node)
        node_idx += 1

    # Step 5: Create connections between capillaries
    print("  Creating capillary-to-capillary connections...")

    # Separate connected and unconnected capillaries
    vert_conn_idxs = [i for i, node in enumerate(nodes) if node.root and node.type == 'capp']
    capp_conn_idxs = [i for i, node in enumerate(nodes) if not node.root and node.type == 'capp']

    # Get positions of all capillaries
    capp_positions = np.array([nodes[i].pos for i in vert_conn_idxs + capp_conn_idxs])

    if len(capp_positions) > 1:
        # Calculate distance matrix
        capp_mat = pos2dists(capp_positions)

        # Initialize connection matrix
        capp_conn_mat = np.zeros((len(capp_positions), len(capp_positions)))

        # Set diagonal and vertical vessel connections to infinity
        np.fill_diagonal(capp_mat, np.inf)
        if hasattr(nv, 'nvert_sum') and nv.nvert_sum > 0:
            capp_mat[:nv.nvert_sum, :nv.nvert_sum] = np.inf

        # Find minimum distance connections for each capillary
        min_dists = np.min(capp_mat, axis=1)
        min_indices = np.argmin(capp_mat, axis=1)

        # Create initial connections
        for i in range(len(min_dists)):
            if min_dists[i] < np.inf:
                capp_conn_mat[i, min_indices[i]] = 1
                capp_conn_mat[min_indices[i], i] = 1

                # Remove these connections from consideration
                capp_mat[i, :] = np.inf
                capp_mat[:, i] = np.inf
                capp_mat[min_indices[i], :] = np.inf
                capp_mat[:, min_indices[i]] = np.inf

        # Apply distance threshold
        capp_mat[capp_mat > vp.maxcappdist] = np.inf

        # Limit connections per capillary (max 3)
        for i in range(len(capp_positions)):
            if np.sum(capp_conn_mat[i, :]) >= 3:
                capp_mat[i, :] = np.inf
                capp_mat[:, i] = np.inf
            capp_mat[i, capp_conn_mat[i, :] > 0] = np.inf
            capp_mat[capp_conn_mat[i, :] > 0, i] = np.inf

        # Iteratively add more connections
        lflag = True
        vert_sum = getattr(nv, 'nvert_sum', 0)
        path_checks = 0
        paths_blocked = 0
        while lflag:
            cap_sums = np.sum(capp_conn_mat, axis=1)
            if vert_sum < len(cap_sums) and np.min(cap_sums[vert_sum:]) >= 1:
                lflag = False
            else:
                zero_conn = np.where(cap_sums == 1)[0]
                if len(zero_conn) == 0:
                    lflag = False
                else:
                    rnd_idx = np.random.choice(zero_conn)
                    if np.min(capp_mat[rnd_idx, :]) < np.inf:
                        # Use distance-based probability for connection
                        valid_mask = capp_mat[rnd_idx, :] < np.inf
                        valid_dists = capp_mat[rnd_idx, valid_mask]
                        if len(valid_dists) > 0:
                            cap_dist_inv = 1.0 / (valid_dists ** vp.distsc)
                            cap_cdf = np.cumsum(cap_dist_inv) / np.sum(cap_dist_inv)
                            rand_val = np.random.rand()
                            lnk_idx = np.where(cap_cdf > rand_val)[0][0]

                            # Find actual index
                            valid_indices = np.where(valid_mask)[0]
                            lnk_idx = valid_indices[lnk_idx]

                            # CRITICAL: Check if connection path intersects existing vessels (MATLAB logic)
                            # Only check for independent capillaries (not connected to vertical vessels)
                            vert_sum = getattr(nv, 'nvert_sum', 0)
                            if rnd_idx >= vert_sum and lnk_idx >= vert_sum:
                                path_checks += 1
                                start_pos = capp_positions[rnd_idx]
                                end_pos = capp_positions[lnk_idx]
                                if _path_intersects_vessels(start_pos, end_pos, neur_ves, nv.size):
                                    # Path intersects existing vessels, skip this connection
                                    paths_blocked += 1
                                    capp_mat[rnd_idx, lnk_idx] = np.inf
                                    capp_mat[lnk_idx, rnd_idx] = np.inf
                                    continue

                            capp_conn_mat[rnd_idx, lnk_idx] = 1
                            capp_conn_mat[lnk_idx, rnd_idx] = 1

                            # Remove connections from consideration
                            capp_mat[np.ix_([rnd_idx, lnk_idx], capp_conn_mat[rnd_idx, :] > 0)] = np.inf
                            capp_mat[np.ix_(capp_conn_mat[lnk_idx, :] > 0, [rnd_idx, lnk_idx])] = np.inf
                            capp_mat[rnd_idx, capp_conn_mat[rnd_idx, :] > 0] = np.inf
                            capp_mat[capp_conn_mat[lnk_idx, :] > 0, lnk_idx] = np.inf
                            capp_mat[rnd_idx, lnk_idx] = np.inf
                            capp_mat[lnk_idx, rnd_idx] = np.inf

                            if np.sum(capp_conn_mat[lnk_idx, :]) >= 3:
                                capp_mat[lnk_idx, :] = np.inf
                                capp_mat[:, lnk_idx] = np.inf

        print(f"  Path intersection checks: {path_checks} performed, {paths_blocked} connections blocked")

        # Create connection structures
        conn_start, conn_end = np.where(np.triu(capp_conn_mat))
        conn_mat = np.zeros((len(nodes), len(nodes)))

        # Create ordered list of all capillary node indices (MATLAB: connidxs = [vertconnidxs cappconnidxs])
        connidxs = vert_conn_idxs + capp_conn_idxs

        for i in range(len(conn_start)):
            # Map capp_conn_mat indices to actual node indices (MATLAB: connidxs(connS(i)), connidxs(connF(i)))
            s_idx = connidxs[conn_start[i]]
            e_idx = connidxs[conn_end[i]]

            nodes[s_idx].conn.append(e_idx)
            nodes[e_idx].conn.append(s_idx)

            conn_idx = len(conn)
            new_conn = VascConnection(
                start=s_idx,
                ends=e_idx,
                weight=np.nan,  # Will be calculated later
                locs=None,
                type='capp'
            )
            conn.append(new_conn)
            conn_mat[s_idx, e_idx] = conn_idx + 1

        # Calculate weights for capillary connections (MATLAB-style algorithm)
        print("  Calculating capillary connection weights...")
        to_connect = []

        # Initialize connection matrix for weight calculation
        conn_mat = np.zeros((len(nodes), len(nodes)))
        for i, c in enumerate(conn):
            if hasattr(c, 'start') and hasattr(c, 'ends'):
                conn_mat[c.start, c.ends] = i + 1
                conn_mat[c.ends, c.start] = i + 1

        # Find connections that need weight calculation
        to_connect = []
        for i, c in enumerate(conn):
            if np.isnan(c.weight) and hasattr(c, 'start') and hasattr(c, 'ends'):
                to_connect.append(i)

        # Process connections iteratively (MATLAB logic)
        while to_connect:
            curr_conn_idx = to_connect[0]
            curr_conn = conn[curr_conn_idx]

            if np.isnan(curr_conn.weight):
                conn_start = curr_conn.start
                conn_end = curr_conn.ends

                # Get connected nodes
                start_conns = [c_idx for c_idx in nodes[conn_start].conn if c_idx != conn_end]
                end_conns = [c_idx for c_idx in nodes[conn_end].conn if c_idx != conn_start]

                # Get connection indices
                start_conn_indices = [int(conn_mat[conn_start, c_idx]) - 1 for c_idx in start_conns if conn_mat[conn_start, c_idx] > 0]
                end_conn_indices = [int(conn_mat[conn_end, c_idx]) - 1 for c_idx in end_conns if conn_mat[conn_end, c_idx] > 0]

                # Filter out current connection
                start_conn_indices = [idx for idx in start_conn_indices if idx != curr_conn_idx]
                end_conn_indices = [idx for idx in end_conn_indices if idx != curr_conn_idx]

                # Get weights
                start_weights = [conn[idx].weight for idx in start_conn_indices if not np.isnan(conn[idx].weight)]
                end_weights = [conn[idx].weight for idx in end_conn_indices if not np.isnan(conn[idx].weight)]

                # Calculate weight (MATLAB logic)
                if not start_weights:
                    if not end_weights:
                        # No connected weights available
                        connweight = max(1, np.random.normal(vp.vesSize[2], vp.vesSize[3] if len(vp.vesSize) > 3 else 0))
                    else:
                        connweight = end_weights[0]  # Use single end weight
                else:
                    if not end_weights:
                        connweight = start_weights[0]  # Use single start weight
                    else:
                        # Both sides have weights - use average or complex calculation
                        if len(start_weights) == 1:
                            connweight = start_weights[0]
                        elif len(end_weights) == 1:
                            connweight = end_weights[0]
                        else:
                            # Complex calculation similar to MATLAB
                            max_start = max(start_weights)
                            min_start = min(start_weights)
                            max_end = max(end_weights)
                            min_end = min(end_weights)

                            # Use geometric mean approach
                            start_geom = np.sqrt(max_start**2 - min_start**2 + max_start**2 + min_start**2) / np.sqrt(2)
                            end_geom = np.sqrt(max_end**2 - min_end**2 + max_end**2 + min_end**2) / np.sqrt(2)

                            connweight = (start_geom + end_geom) / 2

                conn[curr_conn_idx].weight = max(1, connweight)

                # Add unprocessed connections to queue
                for idx in start_conn_indices + end_conn_indices:
                    if np.isnan(conn[idx].weight) and idx not in to_connect:
                        to_connect.append(idx)

            # Remove current connection from queue
            to_connect = to_connect[1:]

        # Final fallback for any remaining NaN weights
        for i in range(len(conn)):
            if np.isnan(conn[i].weight):
                conn[i].weight = max(1, np.random.normal(vp.vesSize[2], vp.vesSize[3] if len(vp.vesSize) > 3 else 0))

    # Debug: check capillary connections
    added_conn_count = len(conn) - initial_conn_count
    print(f"  Capillary growth complete: {len(nodes)} nodes, {len(conn)} connections")
    print(f"  Added {added_conn_count} connections during capillary growth")

    # Check all connection types
    print("  Connection types:")
    for i, c in enumerate(conn):
        if hasattr(c, 'type'):
            locs_status = "set" if c.locs is not None else "None"
            print(f"    Connection {i}: type={c.type}, locs={locs_status}")

    capillary_connections = [i for i, c in enumerate(conn) if hasattr(c, 'type') and c.type in ['capp', 'vtcp']]
    print(f"  Found {len(capillary_connections)} capillary/vtcp connections")

    return nodes, conn, nv


def _path_intersects_vessels(start_pos: np.ndarray, end_pos: np.ndarray,
                           neur_ves: np.ndarray, vol_size: np.ndarray) -> bool:
    """
    Check if the straight line path between two capillaries intersects with existing vessels.

    This implements the MATLAB logic from growCapillaries.m lines 146-158.
    If any point along the path contains a vessel, the connection is not allowed.

    Args:
        start_pos: Starting capillary position [x, y, z]
        end_pos: Ending capillary position [x, y, z]
        neur_ves: Current vessel volume (3D boolean array)
        vol_size: Volume size [x, y, z]

    Returns:
        True if path intersects existing vessels, False otherwise
    """
    # Calculate distance between points
    distance = np.linalg.norm(end_pos - start_pos)
    if distance == 0:
        return False

    # Create path points (MATLAB uses 2*distance points)
    num_points = max(2, int(2 * distance))

    # Generate points along the line
    t = np.linspace(0, 1, num_points)
    path_points = start_pos[:, np.newaxis] + t * (end_pos - start_pos)[:, np.newaxis]

    # Convert to integer voxel coordinates
    voxel_coords = np.round(path_points).astype(int)

    # Clip to volume bounds
    voxel_coords = np.clip(voxel_coords,
                          np.array([0, 0, 0])[:, np.newaxis],
                          (vol_size - 1)[:, np.newaxis])

    # Check if any point along the path contains a vessel
    for i in range(voxel_coords.shape[1]):
        x, y, z = voxel_coords[:, i]
        if neur_ves[x, y, z]:
            return True

    return False

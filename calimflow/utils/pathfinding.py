import numpy as np
from typing import Tuple, Optional
from scipy import sparse, ndimage


def get_dendrite_path2(M: np.ndarray, node: np.ndarray, root: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Retrieve the path of a dendrite from the full paths matrix until it hits the root node.
    Direct port of MATLAB's getDendritePath2 function.

    Args:
        M: pathfrom matrix (matrix containing path information)
        node: End node location in the volume [x,y,z]
        root: Starting location for the dendrite path [x,y,z]

    Returns:
        path: Retrieved path (list of positions)
        pathM: (Optional) Full path matrix (binary matrix of path)
    """
    # Initialize path with starting node
    path = node.copy()
    curr_node = node.copy()

    if len(node) == 2:  # 2D case
        while not np.array_equal(curr_node, root):
            try:
                curr_node = M[int(curr_node[0]), int(curr_node[1]), :].reshape(2)
                path = np.vstack((path, curr_node))
            except:
                path = np.array([])
                break

        # Create path matrix if additional output requested
        pathM = None
        if M.ndim == 3:
            pathM = np.zeros((M.shape[0], M.shape[1]), dtype=bool)
            if len(path) > 0:
                pathM[path[:, 0].astype(int), path[:, 1].astype(int)] = True

    elif len(node) == 3:  # 3D case
        # Pre-allocate maximum possible path length
        max_length = sum(M.shape)
        path = np.zeros((max_length, 3))
        path[0] = curr_node
        idx = 0

        try:
            while not np.array_equal(curr_node, root):
                idx += 1
                curr_node = M[
                    int(curr_node[0]), 
                    int(curr_node[1]), 
                    int(curr_node[2]), 
                    :
                ].reshape(3)
                path[idx] = curr_node

            # Trim path to actual length
            path = path[:idx + 1]

        except:
            # Return empty path if error occurs
            return np.array([]), None

        # Create path matrix if additional output requested
        pathM = None
        if M.ndim == 4:
            pathM = np.zeros((M.shape[0], M.shape[1], M.shape[2]), dtype=bool)
            if len(path) > 0:
                pathM[
                    path[:, 0].astype(int),
                    path[:, 1].astype(int),
                    path[:, 2].astype(int)
                ] = True

    else:
        raise ValueError('Number of dimensions of node must be 2 or 3')

    return path, pathM

def dendrite_dijkstra(M: np.ndarray, pe: np.ndarray, root: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized implementation of Dijkstra's algorithm for dendrite path finding.
    Python version of MATLAB's dendrite_dijkstra_cpp.
    
    Args:
        M: Matrix reshaped to (N,6) containing edge weights
        pe: Position shifts for 6 directions [R,L,U,D,T,B]
        root: Starting node index
        
    Returns:
        distance: Array of distances from root
        pathfrom: Array indicating path taken
    """
    # Initialize arrays
    n_points = M.shape[0]
    distance = np.full(n_points, np.inf, dtype=np.float32)
    pathfrom = np.zeros(n_points, dtype=np.int32)
    distance[root] = 0
    
    # Initialize priority queue with root
    to_visit = np.ones(n_points, dtype=bool)
    queue = np.zeros(n_points, dtype=np.float32)
    queue[root] = 1
    
    while np.any(queue > 0):
        # Find current node (minimum distance unvisited node)
        curr_idx = np.argmax(queue)
        if queue[curr_idx] == 0:
            break
            
        # Mark as visited
        to_visit[curr_idx] = False
        queue[curr_idx] = 0
        
        # Check each direction
        for i, shift in enumerate(pe):
            next_idx = curr_idx + int(shift)
            
            # Check bounds
            if next_idx < 0 or next_idx >= n_points:
                continue
                
            # Only process unvisited nodes
            if to_visit[next_idx]:
                # Calculate new distance
                new_dist = distance[curr_idx] + M[curr_idx, i]
                
                # Update if new path is shorter
                if new_dist < distance[next_idx]:
                    distance[next_idx] = new_dist
                    pathfrom[next_idx] = curr_idx + 1  # +1 to match MATLAB 1-based indexing
                    queue[next_idx] = max(np.finfo(float).eps, -new_dist)
    
    return distance, pathfrom

def dendrite_dijkstra2(M: np.ndarray, dims: np.ndarray, root: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Dijkstra's algorithm for growing dendrites.
    Python implementation of MATLAB's dendrite_dijkstra2 function.
    
    Args:
        M: Matrix containing blockages (reshaped to N x 6)
        dims: Size of volume to grow dendrites in [x,y,z]
        root: Starting point [x,y,z]
    
    Returns:
        distance: Distance the path has traveled
        pathfrom: Path through volume that dendrite takes
    """
    try:
        # Convert root position to linear index
        root2 = np.ravel_multi_index((int(root[0]), int(root[1]), int(root[2])), dims)
    except:
        raise ValueError(f"Invalid root position: {root}")
    
    # Convert M to float32 if needed
    if M.dtype != np.float32:
        M = M.astype(np.float32)
    
    # Set adjacent edges: Right, Left, Up, Down, Top, Bottom
    e = np.array([
        [1, 0, 0], [-1, 0, 0],  # R, L
        [0, 1, 0], [0, -1, 0],  # U, D
        [0, 0, 1], [0, 0, -1]   # T, B
    ])
    
    # Calculate position shifts based on volume dimensions
    pe = e @ np.array([1, dims[0], dims[0]*dims[1]])
    
    # Run optimized Dijkstra's algorithm
    distance, pathfrom_tmp = dendrite_dijkstra(M, pe, root2)
    
    # Reshape distance array
    distance = distance.reshape(dims)
    
    # Convert pathfrom to subscript form
    pathfrom = np.zeros((np.prod(dims), 3))
    valid_paths = pathfrom_tmp > 0
    if np.any(valid_paths):
        path_indices = np.where(valid_paths)[0]
        path_subs = np.array(np.unravel_index(pathfrom_tmp[valid_paths]-1, dims))  # -1 to convert back to 0-based
        pathfrom[path_indices] = path_subs.T
    
    # Reshape pathfrom to match volume dimensions
    pathfrom = pathfrom.reshape((*dims, 3))
    
    return distance, pathfrom

def smooth_cell_body(all_paths: list, cell_body: np.ndarray, fdims: np.ndarray) -> np.ndarray:
    """
    Smooth cell body regions where dendrites connect.
    
    Args:
        all_paths: List of dendrite paths
        cell_body: Array of cell body indices
        fdims: Dimensions of volume [x,y,z]
        
    Returns:
        Output indices for smoothed regions
    """
    from scipy.interpolate import CubicSpline
    
    # Find intersection points of paths with cell body
    conn_idx_root = np.zeros((len(all_paths), 3))
    empty_idxs = np.zeros(len(all_paths), dtype=bool)
    
    for i, path in enumerate(all_paths):
        if len(path) > 0:
            path_ind = np.ravel_multi_index((path[:, 0], path[:, 1], path[:, 2]), fdims)
            path_intersect = np.isin(path_ind, cell_body)
            try:
                conn_idx_root[i] = path[np.where(path_intersect)[0][0]]
            except:
                empty_idxs[i] = True
        else:
            empty_idxs[i] = True
    
    # Calculate distance matrix
    dist_mat = np.sqrt(np.sum((conn_idx_root[:, None] - conn_idx_root[None, :])**2, axis=2))
    dist_mat = (dist_mat == 0).astype(float)
    dist_mat[empty_idxs] = np.nan
    
    # Group dendrites
    dend_groups = []
    for i in range(len(dist_mat)):
        if not np.isnan(dist_mat[i,i]):
            group = np.where(dist_mat[i])[0]
            dist_mat[group] = np.nan
            dend_groups.append(group)
    
    # Process each group
    offset = 2
    conn_idx = np.zeros((len(dend_groups), 3))
    conn_roots = np.zeros((len(dend_groups), 3))
    
    for i, group in enumerate(dend_groups):
        path = all_paths[group[0]]
        path_ind = np.ravel_multi_index((path[:, 0], path[:, 1], path[:, 2]), fdims)
        path_intersect = np.isin(path_ind, cell_body)
        try:
            offset_idx = max(0, np.where(path_intersect)[0][0] - round(offset*np.sqrt(len(group))))
            conn_idx[i] = path[offset_idx]
            conn_roots[i] = path[np.where(path_intersect)[0][0]]
        except:
            conn_idx[i] = path[0]
            conn_roots[i] = path[0]
    
    # Get cell body bounds and create matrix
    cell_ind = np.array(np.unravel_index(cell_body, fdims)).T
    cell_min = np.min(cell_ind, axis=0)
    cell_max = np.max(cell_ind, axis=0)
    
    cell_mat = np.zeros(fdims, dtype=bool)
    cell_mat.ravel()[cell_body] = True
    
    # Extract relevant portion
    cell_crop = cell_mat[
        cell_min[0]:cell_max[0]+1,
        cell_min[1]:cell_max[1]+1,
        cell_min[2]:cell_max[2]+1
    ]
    
    # Calculate cell borders
    cell_diff = (
        np.pad(cell_crop[:-2, 1:-1, 1:-1], ((0,0),(1,1),(1,1))) +
        np.pad(cell_crop[2:, 1:-1, 1:-1], ((0,0),(1,1),(1,1))) +
        np.pad(cell_crop[1:-1, :-2, 1:-1], ((1,1),(0,0),(1,1))) +
        np.pad(cell_crop[1:-1, 2:, 1:-1], ((1,1),(0,0),(1,1))) +
        np.pad(cell_crop[1:-1, 1:-1, :-2], ((1,1),(1,1),(0,0))) +
        np.pad(cell_crop[1:-1, 1:-1, 2:], ((1,1),(1,1),(0,0)))
    )
    
    cell_borders = cell_crop.copy()
    cell_borders[1:-1,1:-1,1:-1] &= (cell_diff > 0) & (cell_diff < 6)
    
    cell_borders2 = np.zeros(fdims, dtype=bool)
    cell_borders2[
        cell_min[0]:cell_max[0]+1,
        cell_min[1]:cell_max[1]+1,
        cell_min[2]:cell_max[2]+1
    ] = cell_borders
    
    borders_sub = np.array(np.where(cell_borders2)).T
    
    # Process each group
    cell_processed = np.zeros(fdims, dtype=bool)
    test_dist = np.array([0, 4, 10])
    num_samp = 20
    
    for j in range(len(conn_roots)):
        dist_off = min(max(test_dist[1], round(offset*np.sqrt(len(dend_groups[j])))), test_dist[2])
        border_dist = borders_sub - conn_roots[j]
        border_dist = np.sqrt(np.sum(border_dist**2, axis=1))
        test_idx = np.where((border_dist < dist_off) & (border_dist > test_dist[0]))[0]
        
        test_sub = []
        for idx in test_idx:
            pts = np.vstack((conn_roots[j], conn_idx[j], borders_sub[idx]))
            t = np.linspace(0, 1, num_samp)
            cs = CubicSpline(np.arange(len(pts)), pts)
            dpts = np.round(cs(t)).astype(int)
            test_sub.extend(dpts)
            
        if test_sub:
            test_sub = np.array(test_sub)
            test_sub = np.clip(test_sub, [0,0,0], np.array(fdims)-1)
            test_ind = np.ravel_multi_index(
                (test_sub[:,0], test_sub[:,1], test_sub[:,2]), 
                fdims
            )
            
            # Create cell bump
            cell_bump = cell_borders2.copy()
            cell_bump.ravel()[cell_body] = True
            cell_bump.ravel()[test_ind] = True
            
            # Iteratively fill gaps
            num_diff = np.inf
            while num_diff > 0:
                tmp = np.sum(cell_bump)
                cell_bump = ndimage.binary_dilation(cell_bump)
                cell_bump &= (
                    ndimage.binary_dilation(cell_bump, structure=np.ones((3,3,3))) >= 4
                )
                num_diff = np.sum(cell_bump) - tmp
                
            cell_processed |= cell_bump
            
    return np.where(cell_processed)[0]

def dilate_dendrite_path_all(paths: np.ndarray, path_nums: np.ndarray, obstruction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simultaneously dilate all dendrite paths in volume.
    
    Args:
        paths: Full set of simulated paths
        path_nums: Corresponding cell numbers
        obstruction: Occupied space in volume
        
    Returns:
        paths: Updated simulated paths
        path_nums: Updated cell numbers
    """
    max_dist = 20  # Maximum dilation distance
    
    # Create distance grid
    x, y, z = np.meshgrid(
        np.arange(-max_dist, max_dist+1),
        np.arange(-max_dist, max_dist+1), 
        np.arange(-max_dist, max_dist+1)
    )
    dists = x**2 + y**2 + z**2
    dsz = dists.shape
    
    # Sort distances
    didx = np.argsort(dists.ravel())
    dpos = np.where(np.diff(dists.ravel()[didx]))[0]
    
    # Convert to float32 and handle obstructions
    paths = paths.astype(np.float32)
    paths[obstruction] = np.nan
    dims = paths.shape
    pdims = np.prod(dims)
    
    # Calculate shifts for 6 directions
    dshifts = np.array([
        -dims[0]*dims[1],  # up
        dims[0]*dims[1],   # down  
        -dims[0],          # left
        dims[0],           # right
        -1,                # back
        1                  # front
    ])
    
    # Get indices to process
    idxs = np.where(paths > 1)[0]
    i = 1
    
    while i < max_dist**2 and len(idxs) > 0:
        # Get shifts for current distance
        dx = np.unravel_index(didx[dpos[i-1]+1:dpos[i]+1], dsz)[0] - max_dist
        dy = np.unravel_index(didx[dpos[i-1]+1:dpos[i]+1], dsz)[1] - max_dist
        dz = np.unravel_index(didx[dpos[i-1]+1:dpos[i]+1], dsz)[2] - max_dist
        jidxs = dz*dims[0]*dims[1] + dy*dims[0] + dx
        
        # Process each location
        for j in range(len(idxs)):
            # Get valid neighbor indices
            pidxs = idxs[j] + jidxs
            pidxs = pidxs[(pidxs >= 0) & (pidxs < pdims)]
            pidxs = pidxs[paths.ravel()[pidxs] == 0]
            
            # Check connectivity
            didxt = np.zeros(len(pidxs), dtype=bool)
            num_val = path_nums.ravel()[idxs[j]]
            
            for k, pidx in enumerate(pidxs):
                # Get neighbor indices
                didxs = pidx + dshifts
                didxs = didxs[(didxs >= 0) & (didxs < pdims)]
                
                # Check if connected to existing path
                if np.any(path_nums.ravel()[didxs] == num_val):
                    didxt[k] = True
                    
            pidxs = pidxs[didxt]
            
            # Distribute path weight to neighbors
            while paths.ravel()[idxs[j]] > 1 and len(pidxs) > 0:
                ridx = np.random.randint(len(pidxs))
                pidx = pidxs[ridx]
                pidxs = np.delete(pidxs, ridx)
                
                paths.ravel()[idxs[j]] -= 1
                paths.ravel()[pidx] = 1
                path_nums.ravel()[pidx] = num_val
                
        # Get remaining locations to process
        idxs = np.where(paths.ravel() > 1)[0]
        i += 1
        
    return paths, path_nums

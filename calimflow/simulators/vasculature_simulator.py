from typing import Tuple, List, Optional
import numpy as np
from scipy import ndimage, sparse
from skimage.morphology import disk
from scipy.interpolate import splprep, splev
import random
import copy
import json
import matplotlib.pyplot as plt

from ..models.parameters import VolumeParams, VascParams, NodeParams
from ..models.structures import Node, Connection
from ..utils.geometry import rotation_matrix, create_disk_structure
from ..utils.sampling import pseudo_rand_sample_2d, pseudo_rand_sample_3d


class VasculatureSimulator:
    def __init__(
        self,
        vol_params: VolumeParams,
        vasc_params: VascParams,
        node_params: Optional[NodeParams] = None,
    ):
        """Initialize VasculatureSimulator."""
        self.vol_params = copy.deepcopy(vol_params)
        self.vasc_params = copy.deepcopy(vasc_params)
        self.node_params = copy.deepcopy(node_params) if node_params else NodeParams()
        self._nodes = []
        self._conn = []
        self._neur_ves = None
        self._neur_ves_all = None

    def simulate(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Main method to simulate vasculature."""
        # Adjust volume parameters
        self.vol_params._size = np.ceil(self.vol_params.size * self.vol_params.res)
        self.vol_params._depth = np.ceil(self.vol_params.depth * self.vol_params.res)

        # Scale vasculature parameters
        self.vasc_params._depth_surf = self.vasc_params.depth_surf * self.vol_params.res
        self.vasc_params._depth_vasc = (
            self.vasc_params.depth_vasc * self.vol_params.res
        )  # TODO: Deprecated
        self.vasc_params._mindists = self.vasc_params.ves_freq * self.vol_params.res / 2
        self.vasc_params._maxcappdist = self.vasc_params.maxcappdist * self.vol_params.res
        self.vasc_params._ves_size = self.vasc_params.ves_size * self.vol_params.res
        self.vasc_params._ves_shift = self.vasc_params.ves_shift * self.vol_params.res
        self.vasc_params._size = (
            self.vol_params.size + np.array([0, 0, self.vol_params.depth])
        ) * self.vol_params.res
        self.vasc_params._szum = self.vasc_params._size + np.array([0, 0, self.vol_params.depth])
        self.vasc_params._nsource = max(
            round(
                (
                    2
                    * (self.vol_params.size[0] + self.vol_params.size[1])
                    / self.vasc_params.source_freq
                )
                * abs(1 + self.vasc_params.ves_num_scale * np.random.randn())
            ),
            0,
        )
        self.vasc_params._nvert = max(
            round(
                (
                    self.vol_params.size[0]
                    * self.vol_params.size[1]
                    / (self.vasc_params.ves_freq[1] ** 2)
                )
                * abs(1 + self.vasc_params.ves_num_scale * np.random.randn())
            ),
            0,
        )
        self.vasc_params._nsurf = max(
            round(
                (
                    self.vol_params.size[0]
                    * self.vol_params.size[1]
                    / (self.vasc_params.ves_freq[0] ** 2)
                )
                * abs(1 + self.vasc_params.ves_num_scale * np.random.randn())
            ),
            0,
        )
        self.vasc_params._ncapp = max(
            round(
                (np.prod(self.vol_params.size) / (self.vasc_params.ves_freq[2] ** 3))
                * abs(1 + self.vasc_params.ves_num_scale * np.random.randn())
            ),
            0,
        )

        # Scale node parameters
        self.node_params._lensc = self.node_params.lensc * self.vol_params.res
        self.node_params._varsc = self.node_params.varsc * self.vol_params.res
        self.node_params._mindist = self.node_params.mindist * self.vol_params.res
        self.node_params._varpos = self.node_params.varpos * self.vol_params.res

        # Grow major vessels
        self._grow_major_vessels()

        # Convert nodes to connections
        self._conn = self._nodes_to_conn()
        self.vasc_params._nconn = len(self._conn)

        # Adjust surface vessel locations
        for c in self._conn:
            # Adjust start node if it's a surface vessel
            if self._nodes[c.start].node_type in ["edge", "surf", "sfvt"]:
                conn_length = len(self._nodes[c.start].conn)
                if conn_length > 0:  # Avoid division by zero
                    z_adjustment = np.ceil(c.weight / conn_length)
                    self._nodes[c.start].pos[2] = min(
                        self._nodes[c.start].pos[2] + z_adjustment, self.vasc_params._depth_surf
                    )

            # Adjust end node if it's a surface vessel
            if self._nodes[c.ends].node_type in ["edge", "surf", "sfvt"]:
                conn_length = len(
                    self._nodes[c.start].conn
                )  # Using start connections as per MATLAB
                if conn_length > 0:  # Avoid division by zero
                    z_adjustment = np.ceil(c.weight / conn_length)
                    self._nodes[c.ends].pos[2] = min(
                        self._nodes[c.ends].pos[2] + z_adjustment, self.vasc_params._depth_surf
                    )

        # Create initial volume with major vessels
        self._neur_ves = np.zeros(self.vasc_params._size, dtype=bool)
        self._conn_to_vol()

        # Store full vessel array before adding capillaries
        self._neur_ves_all = self._neur_ves.copy() if self.vol_params.verbose > 1 else None

        # Add capillaries
        print("Generate Capillaries")
        self._grow_capillaries()

        # Add capillaries to rest of volume
        capp_idxs = [i for i, c in enumerate(self._conn) if c.locs is None]
        self._conn_to_vol(capp_idxs)

        return self._neur_ves.copy(), (
            self._neur_ves_all.copy() if self.vol_params.verbose > 1 else None
        )

    def _grow_major_vessels(self) -> None:
        """
        Grows the major blood vessels (surface and vertical).
        """
        # Initialize source nodes and mark them as "edge"
        for i in range(self.vasc_params._nsource):
            # Randomly generate "edge" position along the border
            tmp_idx = [
                random.random()
                >= self.vasc_params._size[1]
                / (self.vasc_params._size[0] + self.vasc_params._size[1]),
                random.random() >= 0.5,
            ]

            if tmp_idx[0] and tmp_idx[1]:
                tmp_pos = np.array(
                    [
                        np.round((self.vasc_params._size[0] - 1) * random.random()),
                        0,
                        self.vasc_params._depth_surf,
                    ]
                )
            elif tmp_idx[0] and not tmp_idx[1]:
                tmp_pos = np.array(
                    [
                        np.round((self.vasc_params._size[0] - 1) * random.random()),
                        self.vasc_params._size[1],
                        self.vasc_params._depth_surf,
                    ]
                )
            elif not tmp_idx[0] and tmp_idx[1]:
                tmp_pos = np.array(
                    [
                        0,
                        np.round((self.vasc_params._size[1] - 1) * random.random()),
                        self.vasc_params._depth_surf,
                    ]
                )
            else:
                tmp_pos = np.array(
                    [
                        self.vasc_params._size[0],
                        np.round((self.vasc_params._size[1] - 1) * random.random()),
                        self.vasc_params._depth_surf,
                    ]
                )

            self._nodes.append(
                Node(num=i, root=-1, conn=[], pos=tmp_pos, node_type="edge", misc=tmp_idx)
            )

        # Grow surface vessels
        surf_ves = np.zeros(
            (int(self.vasc_params._size[0]), int(self.vasc_params._size[1])), dtype=bool
        )
        for i in range(self.vasc_params._nsource):
            if self._nodes[i].misc[0] and self._nodes[i].misc[1]:
                rand_dir = 0.5 * np.pi + random.random() * self.node_params.dirvar
            elif self._nodes[i].misc[0] and not self._nodes[i].misc[1]:
                rand_dir = 1.5 * np.pi + random.random() * self.node_params.dirvar
            elif not self._nodes[i].misc[0] and self._nodes[i].misc[1]:
                rand_dir = 0.0 * np.pi + random.random() * self.node_params.dirvar
            else:
                rand_dir = 1.0 * np.pi + random.random() * self.node_params.dirvar

            # Grow the surface vessel from the source node
            surf_ves = self._branch_grow_nodes(surf_ves, i, rand_dir)

        # Store number of nodes added after growing surface vessels
        self.vasc_params._nlinks = len(self._nodes) - self.vasc_params._nsource

        # Sample additional locations for diving vessels
        se = disk(self.vasc_params._ves_size[0])
        surf_ves = ndimage.binary_dilation(surf_ves, se)

        # Sample surface positions
        surf_pos, _ = pseudo_rand_sample_2d(
            self.vasc_params._size[:2],
            self.vasc_params._nsurf,
            self.vasc_params._mindists[0],
            self.vasc_params.sep_weight,
            (1 - surf_ves).astype(np.float32),
        )

        # Add z-coordinate
        surf_pos = np.column_stack(
            [surf_pos, np.full(self.vasc_params._nsurf, self.vasc_params._depth_surf)]
        )

        # Combine with existing nodes
        existing_pos = np.array([node.pos for node in self._nodes])
        surf_pos = np.vstack([existing_pos, surf_pos])

        # Calculate distances between all points
        surf_mat = self._pos2dists(surf_pos)

        # Set up connection matrix
        surf_mat[
            : self.vasc_params._nlinks + self.vasc_params._nsource,
            : self.vasc_params._nlinks + self.vasc_params._nsource,
        ] = np.inf
        surf_mat[: self.vasc_params._nsource, : self.vasc_params._nsource] = 0

        for i in range(self.vasc_params._nlinks + self.vasc_params._nsource):
            if self._nodes[i].root >= 0:
                dist = np.linalg.norm(self._nodes[i].pos - self._nodes[self._nodes[i].root].pos)
                surf_mat[self._nodes[i].root, i] = dist
                surf_mat[i, self._nodes[i].root] = dist

        surf_mat[
            self.vasc_params._nlinks + self.vasc_params._nsource :,
            : self.vasc_params._nlinks + self.vasc_params._nsource,
        ] = np.inf

        # Apply distance weighting
        tmp_surf_mat = (surf_mat**self.vasc_params.dist_sc) * (
            1 + self.vasc_params.rand_weight_scale * np.random.randn(*surf_mat.shape)
        )

        # Find paths using Dijkstra's algorithm
        _, surf_path = self._vessel_dijkstra(tmp_surf_mat, 0)
        surf_path[: self.vasc_params._nsource] = np.arange(self.vasc_params._nsource)

        # Connect to nearest source for disconnected nodes
        for i in range(self.vasc_params._nsurf + self.vasc_params._nsource):
            if surf_path[i] == 0:
                surf_path[i] = np.argmin(surf_mat[i, : self.vasc_params._nsource])

        # Create nodes for surface positions
        for i in range(
            self.vasc_params._nlinks + self.vasc_params._nsource,
            self.vasc_params._nlinks + self.vasc_params._nsource + self.vasc_params._nsurf,
        ):
            pos = surf_pos[i]
            is_edge = (pos[0] in [1, self.vasc_params._size[0]]) or (
                pos[1] in [1, self.vasc_params._size[1]]
            )

            self._nodes.append(
                Node(
                    num=i,
                    root=int(surf_path[i]),
                    conn=[int(surf_path[i])],
                    pos=pos,
                    node_type="edge" if is_edge else "surf",
                )
            )

        # Update node connections
        self.vasc_params._nnodes = (
            self.vasc_params._nlinks + self.vasc_params._nsource + self.vasc_params._nsurf
        )
        for i in range(self.vasc_params._nnodes):
            if self._nodes[i].root >= 0:
                self._nodes[self._nodes[i].root].conn = list(
                    set(self._nodes[self._nodes[i].root].conn + [i])
                )

        # Prune surface vasculature and select diving vessels
        se = disk(round(self.vasc_params._mindists[0] * 2))
        neur_vert = np.zeros(self.vasc_params._size[:2], dtype=bool)

        for i in range(self.vasc_params._nnodes):
            if self._nodes[i].node_type == "surf" and len(self._nodes[i].conn) == 1:
                pos = self._nodes[i].pos[:2].astype(int)
                if not neur_vert[pos[0], pos[1]]:
                    self._nodes[i].node_type = "sfvt"
                    tmp = np.zeros(self.vasc_params._size[:2], dtype=bool)
                    tmp[pos[0], pos[1]] = 1
                    neur_vert = np.logical_or(neur_vert, ndimage.binary_dilation(tmp, se))
                else:
                    self._delnode(i)

        surf_idx = [i for i, node in enumerate(self._nodes) if node.node_type == "surf"]
        surf_pos = np.array(
            [node.pos[:2] for node in self._nodes if node.node_type == "surf"]
        ).astype(int)
        while sum(
            1 for node in self._nodes if node.node_type == "sfvt"
        ) < self.vasc_params._nvert and np.any(neur_vert[surf_pos[:, 0], surf_pos[:, 1]] == 0):
            tmp_idx = np.where(neur_vert[surf_pos[:, 0], surf_pos[:, 1]] == 0)[0]
            tmp_idx = surf_idx[np.random.choice(tmp_idx)]
            self._nodes[tmp_idx].node_type = "sfvt"
            tmp = np.zeros(self.vasc_params._size[:2], dtype=bool)
            tmp[
                self._nodes[tmp_idx].pos[0].astype(int), self._nodes[tmp_idx].pos[1].astype(int)
            ] = 1
            neur_vert = np.logical_or(neur_vert, ndimage.binary_dilation(tmp, se))

        # Grow diving vessels to bottom of volume
        vert_idx = [i for i, node in enumerate(self._nodes) if node.node_type == "sfvt"]
        curr_idx = self.vasc_params._nnodes

        for i in vert_idx:
            curr_node = i
            while self._nodes[curr_node].pos[2] < self.vasc_params._size[2] - 1:
                node_pos = self._nodes[curr_node].pos + np.ceil(
                    [
                        np.random.randn() * self.node_params.varpos,
                        np.random.randn() * self.node_params.varpos,
                        max(
                            self.node_params.varsc * np.random.randn() + self.node_params.lensc,
                            self.node_params.mindist,
                        ),
                    ]
                )
                node_pos = np.clip(node_pos, 0, self.vasc_params._size - 1)

                self._nodes.append(
                    Node(
                        num=curr_idx,
                        root=curr_node,
                        conn=[curr_node],
                        pos=node_pos,
                        node_type="vert",
                    )
                )
                self._nodes[curr_node].conn.append(curr_idx)
                curr_node = curr_idx
                curr_idx += 1

        # Update vasculature parameters
        self.vasc_params._nvert = sum(1 for node in self._nodes if node.node_type == "sfvt")
        self.vasc_params._nnodes = curr_idx

        # Initialize end node sizes
        ends = [i for i, node in enumerate(self._nodes) if len(node.conn) == 1]
        for i in ends:
            # Use gamma distribution for vessel sizes
            shape = 3
            scale = (self.vasc_params._ves_size[1] - self.vasc_params._ves_size[2]) / 3
            self._nodes[i].misc = self.vasc_params._ves_size[2] + np.random.gamma(shape, scale)

    def _branch_grow_nodes(
        self,
        neur_ves: np.ndarray,
        idx: int,
        direction: float,
    ) -> np.ndarray:
        """
        Grow surface vasculature from a starting node.

        Args:
            neur_ves: Surface vasculature occupancy (for crossover avoidance)
            idx: Starting branch node index
            direction: Starting branch node growth direction (in radians)

        Returns:
            neur_ves: Updated surface vasculature occupancy
        """
        border_flag = True  # Flag to indicate being at a border
        overlap_flag = False  # Whether to allow overlapping paths
        num_it = 0  # Iteration counter
        prev_pos = self._nodes[idx].pos[:2]  # Previous position (x,y only)
        prev_num = self._nodes[idx].num  # Previous node number
        test_idxs2 = None  # Store previous test indices
        nv_size = np.array(neur_ves.shape)  # Size of vessel array
        branch_p = 0  # Branch probability

        # Dilate the node to avoid overlapping paths
        se = disk(self.vasc_params._ves_size[0])
        neur_ves2 = ndimage.binary_dilation(neur_ves, se)

        while num_it < self.node_params.maxit and border_flag:
            # Check for branching
            if random.random() < branch_p:
                branch_p = 0
                dir_b = direction - abs((0.5 + random.random()) * self.node_params.dirvar)
                direction = direction + abs((0.5 + random.random()) * self.node_params.dirvar)
                neur_ves = self._branch_grow_nodes(neur_ves, prev_num, dir_b)
                overlap_flag = True
            else:
                branch_p = branch_p + self.node_params.branchp

            # Calculate new position
            dir_vect = np.array([np.cos(direction), np.sin(direction)])
            ves_dist = max(
                self.node_params.varsc * np.random.randn() + self.node_params.lensc,
                self.node_params.mindist,
            )
            node_pos = dir_vect * ves_dist + prev_pos + self.node_params.varpos * np.random.randn(2)

            # Check boundaries
            if np.any(node_pos < 1) or np.any(node_pos > nv_size[:2]):
                node_pos = np.clip(node_pos, 0, nv_size[:2] - 1)
                border_flag = False

            # Linear interpolation between points
            test_subs = self._vec_linspace(node_pos, prev_pos, int(np.ceil(ves_dist)))

            # Add adjacent positions
            test_subs_padded = np.hstack(
                [
                    test_subs,
                    test_subs + np.array([[0, 1]]).reshape(2, 1),
                    test_subs + np.array([[1, 0]]).reshape(2, 1),
                ]
            )

            # Clip to boundaries
            test_subs_padded[0, :] = np.clip(test_subs_padded[0, :], a_min=0, a_max=nv_size[0] - 1)
            test_subs_padded[1, :] = np.clip(test_subs_padded[1, :], a_min=0, a_max=nv_size[1] - 1)

            # Convert subscripts to indices
            test_idxs = np.ravel_multi_index(
                multi_index=(
                    test_subs_padded[0, :].astype(int),
                    test_subs_padded[1, :].astype(int),
                ),
                dims=nv_size[:2],
            )

            if np.sum(neur_ves2.flat[test_idxs]) == 0 or overlap_flag:
                # Create new node
                node_num = len(self._nodes)
                self._nodes.append(
                    Node(
                        num=node_num,
                        root=prev_num,
                        conn=[prev_num],
                        pos=np.append(node_pos, self.vasc_params._depth_surf),
                        node_type="surf",
                    )
                )

                prev_pos = node_pos
                prev_num = node_num
                num_it += 1

                if test_idxs2 is not None:
                    neur_ves.flat[test_idxs2] = 1

                test_idxs2 = test_idxs
            else:
                border_flag = False
                test_idxs2 = None

            overlap_flag = False

        if test_idxs2 is not None:
            neur_ves.flat[test_idxs2] = 1

        return neur_ves

    def _vec_linspace(self, v1: np.ndarray, v2: np.ndarray, n: int) -> np.ndarray:
        """
        A vectorized version of linspace that creates a sequence of n vectors
        between v1 and v2.

        Args:
            v1: Start vector
            v2: End vector
            n: Number of points

        Returns:
            Array of shape (len(v1), n) containing the sequence of vectors
        """
        out = np.zeros((len(v1), n))
        for i in range(len(v1)):
            out[i, :] = np.linspace(v1[i], v2[i], n)
        return out

    def _pos2dists(self, positions: np.ndarray) -> np.ndarray:
        """
        Convert a series of points into a matrix of distances between all points.

        Args:
            positions: NxD array of N D-dimensional points

        Returns:
            NxN matrix of distances between points
        """
        num_points = positions.shape[0]
        distances = np.zeros((num_points, num_points))

        for i in range(num_points):
            diff = positions - positions[i]
            distances[:, i] = np.sqrt(np.sum(diff**2, axis=1))

        return distances

    def _vessel_dijkstra(self, dist_mat: np.ndarray, proot: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Dijkstra's algorithm for growing vasculature.

        Args:
            dist_mat: Distance matrix between nodes
            proot: Starting root node

        Returns:
            distance: Minimum path distance from all nodes to root
            pathfrom: Path from root to each node
        """
        dims = dist_mat.shape[0]
        to_visit = np.ones(dims, dtype=bool)
        unvisited = np.zeros(dims)
        unvisited[proot] = 1
        distance = np.full(dims, np.inf)
        distance[proot] = 0
        pathfrom = np.full(dims, np.nan)
        cn = proot  # current node

        while np.any(unvisited):
            to_visit[cn] = False
            next_idx = np.where(unvisited)[0]
            idx = np.argmin(unvisited[unvisited > 0])
            cn = next_idx[idx]
            unvisited[cn] = 0

            for nn in range(dims):
                if to_visit[nn]:
                    ndist = distance[cn] + dist_mat[cn, nn]
                    if ndist < distance[nn]:
                        unvisited[nn] = max(np.finfo(float).eps, ndist)
                        distance[nn] = ndist
                        pathfrom[nn] = cn

        return distance, pathfrom

    def _delnode(self, num: int) -> None:
        """
        Delete a single node and update connections.

        Args:
            num: Index of node to delete

        Returns:
            Updated list of nodes
        """
        # Update connections in other nodes
        for i in self._nodes[num].conn:
            self._nodes[i].conn = [x for x in self._nodes[i].conn if x != num]
            if self._nodes[i].root == num:
                self._nodes[i].root = -1

        # Replace deleted node with empty node
        self._nodes[num] = Node(num=num, root=-1, conn=[], pos=np.array([0, 0, 0]), node_type="")

    def _nodes_to_conn(self) -> List[Connection]:
        """Converts node structure to connection structure."""
        # Find end nodes (nodes with only one connection)
        ends = [i for i, node in enumerate(self._nodes) if len(node.conn) == 1]

        # Initialize sparse connection matrix
        n = len(self._nodes)
        conn_mat = sparse.lil_matrix((n, n))

        # Build connection matrix
        for end_idx in ends:
            curr_node = end_idx
            while self._nodes[curr_node].root >= 0:
                weight = self._nodes[end_idx].misc if self._nodes[end_idx].misc is not None else 1.0
                conn_mat[self._nodes[curr_node].num, self._nodes[curr_node].root] = np.sqrt(
                    conn_mat[self._nodes[curr_node].num, self._nodes[curr_node].root] ** 2
                    + weight**2
                )
                curr_node = self._nodes[curr_node].root

        # Convert to connections list
        connections = []
        rows, cols, weights = sparse.find(conn_mat)

        for i in range(len(rows)):
            connections.append(
                Connection(start=int(rows[i]), ends=int(cols[i]), weight=float(weights[i]))
            )

        return connections

    def _conn_to_vol(
        self,
        idxs: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[Connection]]:
        """
        Add vasculature connectivity to volume.

        Args:
            idxs: Indices of vessels to generate (default: all)
        Returns:
            conn: Updated connections with location information
        """
        if idxs is None:
            idxs = range(len(self._conn))

        for i in idxs:
            if self._conn[i].start == self._conn[i].ends:
                continue

            # Find connected nodes for spline interpolation
            tmp_l = np.setxor1d(self._nodes[self._conn[i].start].conn, [self._conn[i].ends])
            tmp_u = np.setxor1d(self._nodes[self._conn[i].ends].conn, [self._conn[i].start])

            # Randomly select one connection if multiple exist
            if tmp_l.size > 0:
                tmp_l = tmp_l[np.random.randint(len(tmp_l))]
            if tmp_u.size > 0:
                tmp_u = tmp_u[np.random.randint(len(tmp_u))]

            # Sample points based on distance
            numsamp = int(
                2
                * np.linalg.norm(
                    self._nodes[self._conn[i].start].pos - self._nodes[self._conn[i].ends].pos
                )
            )

            if tmp_l.size > 0 and tmp_u.size > 0:
                # Spline interpolation including TMPL and TMPU
                spts = np.vstack(
                    [
                        self._nodes[tmp_l].pos,
                        self._nodes[self._conn[i].start].pos,
                        self._nodes[self._conn[i].ends].pos,
                        self._nodes[tmp_u].pos,
                    ]
                )
            elif tmp_l.size > 0:
                # Spline interpolation including only TMPL
                spts = np.vstack(
                    [
                        self._nodes[tmp_l].pos,
                        self._nodes[self._conn[i].start].pos,
                        self._nodes[self._conn[i].ends].pos,
                    ]
                )
            elif tmp_u.size > 0:
                # Spline interpolation including only TMPU
                spts = np.vstack(
                    [
                        self._nodes[self._conn[i].start].pos,
                        self._nodes[self._conn[i].ends].pos,
                        self._nodes[tmp_u].pos,
                    ]
                )
            else:
                # Spline interpolation without TMPL or TMPU
                spts = np.vstack(
                    [self._nodes[self._conn[i].start].pos, self._nodes[self._conn[i].ends].pos]
                )

            spts = np.unique(spts, axis=0)
            tck, u = splprep((spts[:, 0], spts[:, 1], spts[:, 2]), s=0, k=min(3, spts.shape[0] - 1))
            u_eval = np.linspace(u[0], u[-1], int(numsamp))
            ves_loc = np.ceil(np.array(splev(u_eval, tck)).T).astype(int)

            ves_loc = np.clip(np.ceil(ves_loc), 0, self.vasc_params._size - 1)
            ves_loc = np.unique(ves_loc, axis=0)
            self._conn[i].locs = ves_loc

            # Calculate dilation bounds
            min_idx = np.maximum(
                np.min(ves_loc, axis=0) - np.ceil(self._conn[i].weight), [0, 0, 0]
            ).astype(int)
            max_idx = np.minimum(
                np.max(ves_loc, axis=0) + np.ceil(self._conn[i].weight), self.vasc_params._size - 1
            ).astype(int)
            ves_loc = ves_loc - min_idx

            # Create 3D volume and dilate
            tmp = np.zeros((max_idx - min_idx + 1).astype(int), dtype=bool)
            tmp[ves_loc[:, 0].astype(int), ves_loc[:, 1].astype(int), ves_loc[:, 2].astype(int)] = (
                True
            )
            x, y, z = np.meshgrid(
                np.arange(-np.ceil(self._conn[i].weight), np.ceil(self._conn[i].weight) + 1),
                np.arange(-np.ceil(self._conn[i].weight), np.ceil(self._conn[i].weight) + 1),
                np.arange(-np.ceil(self._conn[i].weight), np.ceil(self._conn[i].weight) + 1),
            )
            se = np.sqrt(x**2 + y**2 + z**2) <= self._conn[i].weight
            tmp = ndimage.binary_dilation(tmp, se)

            # Add dilated vessel to main volume
            self._neur_ves[
                min_idx[0] : max_idx[0] + 1,
                min_idx[1] : max_idx[1] + 1,
                min_idx[2] : max_idx[2] + 1,
            ] = np.logical_or(
                self._neur_ves[
                    min_idx[0] : max_idx[0] + 1,
                    min_idx[1] : max_idx[1] + 1,
                    min_idx[2] : max_idx[2] + 1,
                ],
                tmp,
            )

    def _grow_capillaries(self) -> None:
        """
        Sample capillary locations and setup nodes and connections.
        """
        # Initialize pseudouniform sampling for capillaries
        dilrad = int(np.ceil(self.vasc_params._mindists[2] / self.vol_params.res))
        x, y, z = np.meshgrid(
            np.arange(-dilrad, dilrad + 1),
            np.arange(-dilrad, dilrad + 1),
            np.arange(-dilrad, dilrad + 1),
            indexing="ij",
        )
        se = np.exp(-2 * (x**2 + y**2 + z**2) / dilrad**2)
        tmp_vol = np.zeros(self.vasc_params._szum, dtype=np.float32)

        # Process each connection
        for i in range(len(self._conn)):
            tmp_pos = np.round(self._conn[i].locs / self.vol_params.res)
            tmp_pos = tmp_pos[:: max(1, dilrad // 3)]

            for pos in tmp_pos:
                # Calculate bounds
                tmp_l = pos - dilrad
                tmp_l = (tmp_l < 0) * (-tmp_l)
                tmp_u = pos + dilrad
                tmp_u = (tmp_u > self.vasc_params._szum - 1) * (
                    tmp_u - (self.vasc_params._szum - 1)
                )

                # Extract relevant portion of structural element
                tmp = se[
                    int(tmp_l[0]) : se.shape[0] - int(tmp_u[0]),
                    int(tmp_l[1]) : se.shape[1] - int(tmp_u[1]),
                    int(tmp_l[2]) : se.shape[2] - int(tmp_u[2]),
                ]

                # Calculate volume indices
                tmp_l = pos - dilrad + tmp_l
                tmp_u = pos + dilrad - tmp_u

                # Update volume using maximum operation
                tmp_vol[
                    int(tmp_l[0]) : int(tmp_u[0]) + 1,
                    int(tmp_l[1]) : int(tmp_u[1]) + 1,
                    int(tmp_l[2]) : int(tmp_u[2]) + 1,
                ] = np.maximum(
                    tmp_vol[
                        int(tmp_l[0]) : int(tmp_u[0]) + 1,
                        int(tmp_l[1]) : int(tmp_u[1]) + 1,
                        int(tmp_l[2]) : int(tmp_u[2]) + 1,
                    ],
                    tmp,
                )

        # Sample capillary positions
        capp_pos, _ = pseudo_rand_sample_3d(
            self.vasc_params._size / self.vol_params.res,
            self.vasc_params._ncapp,
            self.vasc_params._mindists[2] / self.vol_params.res,
            self.vasc_params.sep_weight,
            1 - tmp_vol.astype(np.float32),
        )
        capp_pos = (
            capp_pos * self.vol_params.res
            + 1
            - np.random.randint(max(1, int(self.vol_params.res)), size=capp_pos.shape)
        )

        # Setup connections from diving vessels to capillaries
        nv_vert_conn = np.random.randint(
            0,
            self.vasc_params._szum[2] // self.vasc_params.ves_freq[2],
            size=self.vasc_params._nvert,
        )
        self.vasc_params._nvert_sum = np.sum(nv_vert_conn)
        node_idx = self.vasc_params._nnodes
        conn_idx = self.vasc_params._nconn

        # Find vertical vessel indices
        vert_idxs = [i for i, n in enumerate(self._nodes) if n.node_type == "sfvt"]

        # Ensure vessel size array has enough elements
        if len(self.vasc_params._ves_size) < 4:
            self.vasc_params._ves_size = np.append(
                self.vasc_params._ves_size, self.vasc_params._ves_size[2] * 0
            )

        # Process each vertical vessel
        for i, vert_idx in enumerate(vert_idxs):
            ves_idx = [vert_idx]

            # Find all connected vertical vessels
            while True:
                tmp_idx = [
                    n for n in self._nodes[ves_idx[-1]].conn if self._nodes[n].node_type == "vert"
                ]
                tmp_idx = list(set(tmp_idx) - set(ves_idx))
                if not tmp_idx:
                    break
                ves_idx.extend(tmp_idx)

            ves_idx = ves_idx[1:]  # Remove first element (surface vessel)

            # Connect capillaries to vertical vessels
            for _ in range(nv_vert_conn[i]):
                tmp_idx = np.random.choice(ves_idx)

                # Find closest capillary
                dists = np.sum((capp_pos - self._nodes[tmp_idx].pos) ** 2, axis=1)
                tmp = np.nanargmin(dists)

                # Create new node and connection
                self._nodes.append(
                    Node(
                        num=node_idx,
                        root=tmp_idx,
                        conn=[tmp_idx],
                        pos=capp_pos[tmp].copy(),
                        node_type="capp",
                    )
                )
                self._nodes[tmp_idx].conn.append(node_idx)

                # Create connection with random weight
                weight = max(
                    1,
                    np.random.normal(self.vasc_params._ves_size[2], self.vasc_params._ves_size[3]),
                )
                self._conn.append(
                    Connection(start=node_idx, ends=tmp_idx, weight=weight, misc="vtcp")
                )

                # Update indices
                node_idx += 1
                conn_idx += 1

                # Mark capillary as used
                capp_pos[tmp] = np.nan

        # Update vasculature parameters
        self.vasc_params._nnodes = node_idx
        self.vasc_params._nconn = conn_idx

        for i in range(capp_pos.shape[0]):
            if not np.isnan(capp_pos[i]).any():
                self._nodes.append(
                    Node(num=node_idx, root=-1, conn=[], pos=capp_pos[i].copy(), node_type="capp")
                )
                node_idx += 1
        # Update vasculature parameters
        self.vasc_params._nnodes = node_idx

        # Setup capillary-to-capillary connections
        vert_conn_idxs = [
            i
            for i, n in enumerate(self._nodes)
            if self._nodes[i].root != -1 and self._nodes[i].node_type == "capp"
        ]
        capp_conn_idxs = [
            i
            for i, n in enumerate(self._nodes)
            if self._nodes[i].root == -1 and self._nodes[i].node_type == "capp"
        ]
        conn_idxs = vert_conn_idxs + capp_conn_idxs

        # Get positions for distance calculation
        capp_pos = np.vstack([self._nodes[i].pos for i in conn_idxs])

        # Calculate distance matrix and setup initial connections
        capp_mat = self._pos2dists(capp_pos)
        capp_mat[np.eye(self.vasc_params._ncapp, dtype=bool)] = np.inf
        capp_mat[: self.vasc_params._nvert_sum, : self.vasc_params._nvert_sum] = np.inf

        # Initialize connection matrix and find minimum distances
        capp_conn_mat = np.zeros((self.vasc_params._ncapp, self.vasc_params._ncapp))
        min_capp = np.unravel_index(np.argmin(capp_mat, axis=0), capp_mat.shape)

        # Set initial connections using array indexing
        idx = np.arange(self.vasc_params._ncapp)
        capp_conn_mat[idx, min_capp] = 1
        capp_conn_mat[min_capp, idx] = 1
        capp_mat[idx, min_capp] = np.inf
        capp_mat[min_capp, idx] = np.inf

        # Remove distant connections
        capp_mat[capp_mat > self.vasc_params._maxcappdist] = np.inf

        # Initial processing of connections
        for i in range(self.vasc_params._ncapp):
            if np.sum(capp_conn_mat[i, :]) >= 3:
                capp_mat[i, :] = np.inf
                capp_mat[:, i] = np.inf
            capp_mat[i, capp_conn_mat[i, :].astype(bool)] = np.inf
            capp_mat[capp_conn_mat[i, :].astype(bool), i] = np.inf

        # Check vessel intersections
        for i in range(self.vasc_params._nvert_sum, self.vasc_params._ncapp):
            for j in range(i + 1, self.vasc_params._ncapp):
                if capp_mat[i, j] < np.inf:
                    num_points = int(2 * capp_mat[i, j])
                    x = np.linspace(capp_pos[i, 0], capp_pos[j, 0], num_points)
                    y = np.linspace(capp_pos[i, 1], capp_pos[j, 1], num_points)
                    z = np.linspace(capp_pos[i, 2], capp_pos[j, 2], num_points)
                    xpix = np.ceil(np.column_stack([x, y, z]))

                    # Convert to indices and check for vessel intersection
                    indices = (
                        (xpix[:, 0]).astype(int),
                        (xpix[:, 1]).astype(int),
                        (xpix[:, 2]).astype(int),
                    )
                    if np.any(self._neur_ves[indices]):
                        capp_mat[i, j] = np.inf
                        capp_mat[j, i] = np.inf

        print("End vessel intersection processing")

        # Main connection loop
        while True:
            capp_sum = np.sum(capp_conn_mat, axis=1)
            if np.min(capp_sum[self.vasc_params._nvert_sum :]) > 1:
                break
            idxs = np.where(capp_sum == 1)[0]
            if np.min(np.min(capp_mat[:, idxs])) == np.inf:
                break

            rnd_idx = np.random.choice(idxs)
            # Calculate connection probabilities
            cap_dist_inv = 1.0 / (capp_mat[rnd_idx, :] ** self.vasc_params._dist_sc)
            cap_dist_inv[np.isinf(cap_dist_inv)] = 0

            # Find connection using CDF
            cap_cdf = np.concatenate(([0], np.cumsum(cap_dist_inv) / np.sum(cap_dist_inv, axis=1)))
            lnk_idx = np.where(np.diff(cap_cdf > np.random.random()))[0][0]

            # Update connections
            capp_conn_mat[rnd_idx, lnk_idx] = 1
            capp_conn_mat[lnk_idx, rnd_idx] = 1

            # Update distance matrix
            mask_rnd = capp_conn_mat[rnd_idx, :].astype(bool)
            mask_lnk = capp_conn_mat[lnk_idx, :].astype(bool)
            capp_mat[mask_rnd, lnk_idx] = np.inf
            capp_mat[mask_lnk, rnd_idx] = np.inf
            capp_mat[lnk_idx, mask_rnd] = np.inf
            capp_mat[rnd_idx, mask_lnk] = np.inf
            capp_mat[rnd_idx, lnk_idx] = np.inf
            capp_mat[lnk_idx, rnd_idx] = np.inf

            if np.sum(capp_conn_mat[lnk_idx, :]) >= 3:
                capp_mat[lnk_idx, :] = np.inf
                capp_mat[:, lnk_idx] = np.inf

        # Get final connections
        conn_s, conn_f = np.where(np.triu(capp_conn_mat))
        conn_mat = sparse.lil_matrix((len(self._nodes), len(self._nodes)))
        conn_idx = self.vasc_params._nconn
        for s, f in zip(conn_s, conn_f):
            self._nodes[conn_idxs[s]].conn.append(conn_idxs[f])
            self._nodes[conn_idxs[f]].conn.append(conn_idxs[s])
            self._conn.append(
                Connection(start=conn_idxs[s], ends=conn_idxs[f], weight=np.nan, misc="capp")
            )
            conn_mat[conn_idxs[s], conn_idxs[f]] = conn_idx
            conn_idx += 1

        # Update vasculature parameters
        self.vasc_params._nconn = conn_idx

        # Process weights
        nodes_to_connect = [c.start for c in self._conn if c.misc == "vtcp"]
        to_connect = []

        # Find connections to process
        for node_idx in nodes_to_connect:
            tmp_conn = self._nodes[node_idx].conn
            for j in tmp_conn:
                if conn_mat[node_idx, j]:
                    to_connect.append(conn_mat[node_idx, j])

        to_connect = np.array(to_connect).flatten()
        conn_to_connect = [i for i, c in enumerate(self._conn) if c.misc == "vtcp"]

        # Update connection matrix
        for i in conn_to_connect:
            conn_mat[self._conn[i].ends, self._conn[i].start] = i
        conn_mat = conn_mat + conn_mat.T

        # Process weights for connected vessels
        while len(to_connect) > 0:
            curr_conn = int(to_connect[0])
            if np.isnan(self._conn[curr_conn].weight):
                conn_start = self._conn[curr_conn].start
                conn_end = self._conn[int(curr_conn)].ends

                # Get connected vessels
                start_conns = self._nodes[conn_start].conn
                end_conns = self._nodes[conn_end].conn
                start_conns = conn_mat[conn_start, start_conns].toarray().flatten()
                end_conns = conn_mat[conn_end, end_conns].toarray().flatten()

                # Remove current connection and zeros
                start_conns = start_conns[(start_conns != curr_conn) & (start_conns != 0)]
                end_conns = end_conns[(end_conns != curr_conn) & (end_conns != 0)]

                # Get weights
                start_weights = np.array([self._conn[int(i)].weight for i in start_conns])
                end_weights = np.array([self._conn[int(i)].weight for i in end_conns])
                end_flag = False
                start_flag = False

                # Process start weights
                if np.any(np.isnan(start_weights)):
                    weight1 = np.nan
                else:
                    if len(start_weights) == 1:
                        start_flag = True
                        weight1 = start_weights[0]
                    else:
                        tmp1 = max(start_weights) ** 2 - min(start_weights) ** 2
                        tmp2 = max(start_weights) ** 2 + min(start_weights) ** 2
                        weight1 = np.sqrt(np.random.random() * (tmp2 - tmp1) + tmp1)

                # Process end weights
                if np.any(np.isnan(end_weights)):
                    weight2 = np.nan
                else:
                    if len(end_weights) == 1:
                        end_flag = True
                        weight2 = end_weights[0]
                    else:
                        tmp1 = max(end_weights) ** 2 - min(end_weights) ** 2
                        tmp2 = max(end_weights) ** 2 + min(end_weights) ** 2
                        weight2 = np.sqrt(np.random.random() * (tmp2 - tmp1) + tmp1)

                # Calculate final weight
                if np.isnan(weight1):
                    if np.isnan(weight2):
                        conn_weight = max(
                            1,
                            np.random.normal(
                                self.vasc_params._ves_size[2], self.vasc_params._ves_size[3]
                            ),
                        )
                    else:
                        conn_weight = weight2
                else:
                    if np.isnan(weight2):
                        conn_weight = weight1
                    else:
                        if start_flag:
                            conn_weight = weight1
                        elif end_flag:
                            conn_weight = weight2
                        else:
                            conn_weight = (weight1 + weight2) / 2

                self._conn[curr_conn].weight = conn_weight

                # Add new connections to process
                to_connect = np.concatenate(
                    (
                        to_connect,
                        end_conns[np.isnan(end_weights)],
                        start_conns[np.isnan(start_weights)],
                    )
                )

            to_connect = to_connect[1:]

        print("End weight processing")

        # Set remaining weights
        for i in range(self.vasc_params._nconn):
            if self._conn[i].weight is None or np.isnan(self._conn[i].weight):
                self._conn[i].weight = max(
                    1,
                    np.random.normal(self.vasc_params._ves_size[2], self.vasc_params._ves_size[3]),
                )

    def plot_vasculature(self, save_path: str = None, show: bool = False) -> None:
        """
        Plot the vasculature network with vessels as curves.

        Args:
            save_path: Path to save the figure. If None, figure is not saved
            show: Whether to display the plot
        """
        # Create high resolution figure
        fig = plt.figure(figsize=(15, 15), dpi=300)
        ax = fig.add_subplot(111, projection="3d")

        # Plot each vessel connection as a curve
        colors = ["red", "blue", "green"]  # Different colors for vessel types

        # Create dummy lines for legend
        surface_line = plt.Line2D([0], [0], color=colors[0], alpha=0.6, linewidth=2)
        vertical_line = plt.Line2D([0], [0], color=colors[1], alpha=0.6, linewidth=2)
        capillary_line = plt.Line2D([0], [0], color=colors[2], alpha=0.2, linewidth=1)

        vessel_nums = np.zeros(3)
        for conn in self._conn:
            if conn.locs is not None:  # Skip connections without location data
                # Get vessel type for coloring and opacity
                if self._nodes[conn.start].node_type in ["edge", "surf", "sfvt"]:
                    vessel_nums[0] += 1
                    color = colors[0]  # Surface vessels in red
                    alpha = 0.6
                elif self._nodes[conn.start].node_type == "vert":
                    vessel_nums[1] += 1
                    color = colors[1]  # Vertical vessels in blue
                    alpha = 0.6
                else:
                    vessel_nums[2] += 1
                    color = colors[2]  # Capillaries in green
                    alpha = 0.2  # Reduced opacity for capillaries

                # Plot vessel as a line with thickness proportional to vessel weight
                linewidth = max(0.5, min(2.0, conn.weight / 2))
                ax.plot(
                    conn.locs[:, 0],
                    conn.locs[:, 1],
                    conn.locs[:, 2],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                )

        print(f"Vessel counts:")
        print(f" - Surface vessels: {int(vessel_nums[0])}")
        print(f" - Vertical vessels: {int(vessel_nums[1])}")
        print(f" - Capillaries: {int(vessel_nums[2])}")

        # Enhance plot appearance
        ax.set_title("Vasculature Network", fontsize=14, pad=20)
        ax.set_xlabel("X axis", fontsize=12)
        ax.set_ylabel("Y axis", fontsize=12)
        ax.set_zlabel("Depth (µm)", fontsize=12)

        # Set axis limits based on volume size
        ax.set_xlim(0, self.vasc_params._size[0])
        ax.set_ylim(0, self.vasc_params._size[1])
        ax.set_zlim(0, self.vasc_params._size[2])  # Normal z limits

        # Invert z-axis so 0 is at the top
        ax.invert_zaxis()

        # Set equal aspect ratio
        ax.set_box_aspect(
            [
                self.vasc_params._size[0] / max(self.vasc_params._size),
                self.vasc_params._size[1] / max(self.vasc_params._size),
                self.vasc_params._size[2] / max(self.vasc_params._size),
            ]
        )

        # Add grid and legend with correct colors and opacities
        ax.grid(True, alpha=0.3)
        ax.legend(
            [surface_line, vertical_line, capillary_line],
            ["Surface Vessels", "Vertical Vessels", "Capillaries"],
            loc="upper right",
            fontsize=10,
        )

        # Adjust view angle for better visualization
        ax.view_init(elev=20, azim=45)

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close()

    def vessel_network_to_json(self) -> str:
        """
        Convert vessel network data to JSON format suitable for Three.js visualization.

        Returns:
            JSON string containing vessel network data
        """

        def convert_ndarray(arr):
            """Convert numpy array to list for JSON serialization"""
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr

        # Prepare vessel data structure
        vessel_data = {
            "metadata": {
                "version": 0.0,
                "type": "VesselNetwork",
                "generator": "VasculatureSimulator",
            },
            "vessels": [],
        }

        # Process each connection (vessel)
        for conn in self._conn:
            if conn.locs is not None:  # Skip connections without location data
                # Get vessel type and properties
                if self._nodes[conn.start].node_type in ["edge", "surf", "sfvt"]:
                    vessel_type = "surface"
                    color = [1, 0, 0]  # Red
                elif self._nodes[conn.start].node_type == "vert":
                    vessel_type = "vertical"
                    color = [0, 0, 1]  # Blue
                else:
                    vessel_type = "capillary"
                    color = [0, 1, 0]  # Green

                # Create vessel object
                vessel = {
                    "type": vessel_type,
                    "start": {
                        "id": int(conn.start),
                        "position": convert_ndarray(self._nodes[conn.start].pos),
                    },
                    "end": {
                        "id": int(conn.ends),
                        "position": convert_ndarray(self._nodes[conn.ends].pos),
                    },
                    "path": convert_ndarray(conn.locs),
                    "radius": float(conn.weight / 2),
                    "color": color,
                }

                vessel_data["vessels"].append(vessel)

        return json.dumps(vessel_data)

    def save_vessel_network(self, save_path: str) -> None:
        """
        Save vessel network data to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        json_data = self.vessel_network_to_json()
        with open(save_path, "w") as f:
            f.write(json_data)

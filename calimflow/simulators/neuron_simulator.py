from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import copy

from ..models.parameters import (
    VolumeParams, NeuronParams, DendriteParams, NeuralVolume, NeuronBody
)
from ..utils.geometry import (
    rotation_matrix,
    teardrop_projection,
    create_disk_structure
)
from ..utils.mesh import (
    spiral_sample_sphere,
    in_triangulation
)

class NeuronSimulator:
    def __init__(
        self,
        vol_params: VolumeParams,
        neur_params: NeuronParams,
        dend_params: DendriteParams,
        bg_params: Optional[Dict] = None,
        axon_params: Optional[Dict] = None
    ):
        """Initialize NeuronSimulator."""
        # Store deep copies of parameters to avoid modifying originals
        self.vol_params = copy.deepcopy(vol_params)
        self.neur_params = copy.deepcopy(neur_params)
        self.dend_params = copy.deepcopy(dend_params)
        self.bg_params = bg_params or {'flag': False}
        self.axon_params = axon_params or {'flag': False}
        self.neuron_bodies: List[NeuronBody] = []

        # Calculate runtime parameters
        # Volume parameters
        self.vol_params._size = np.ceil(self.vol_params.size * self.vol_params.res).astype(int)
        self.vol_params._depth = np.ceil(self.vol_params.depth * self.vol_params.res)
        
        # Dendrite parameters
        self.dend_params.dt_params = self.dend_params.dt_params * self.vol_params.res
        self.dend_params.at_params = self.dend_params.at_params * self.vol_params.res
        self.dend_params.thickness_scale = self.dend_params.thickness_scale * self.vol_params.res * self.vol_params.res
        
        # Calculate number of neurons based on volume size (1 neuron per 1000 cubic microns)
        vol_size_um = np.prod(self.vol_params.size)
        self.n_neurons = max(1, int(vol_size_um / 1000))
        
        # Calculate minimum distance between neurons (in pixels)
        self.min_dist = 15 * self.vol_params.res  # 15 microns minimum distance

    def sample_locations(self, vessel_volume: np.ndarray) -> np.ndarray:
        """Sample neuron locations avoiding vessels."""
        # Convert vessel volume to match full volume size if needed
        if vessel_volume.shape != tuple(self.vol_params._size):
            vessel_volume = np.zeros(tuple(self.vol_params._size), dtype=bool)
            
        # Generate neuron locations using _sample_dense_neurons
        neur_locs, v_cell, v_nuc, tri, _ = self._sample_dense_neurons(
            vessel_volume,
            self.n_neurons,
            self.min_dist
        )
        
        # Store generated neuron bodies
        self._store_neuron_bodies(neur_locs, v_cell, v_nuc, tri)
        
        return neur_locs

    def generate_neurons(
        self,
        neur_locs: np.ndarray,
        neur_ves: np.ndarray
    ) -> Tuple[NeuralVolume, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate neural bodies and processes.
        
        Args:
            neur_locs: Locations of neurons
            neur_ves: Blood vessel volume
            
        Returns:
            neural_volume: NeuralVolume object containing soma and fluorescence data
            neur_locs: Updated neuron locations
            v_cell: Vertices defining soma shapes
            v_nuc: Vertices defining nucleus shapes
            tri: Triangulation for surface mesh
            neur_soma: Binary volume indicating soma locations
        """
        # Generate initial neural volume
        neur_soma, neur_vol, gp_nuc, gp_soma = self._generate_neural_volume(
            neur_locs,
            neur_ves
        )

        # Generate dendrites if enabled
        if self.dend_params is not None:
            # Generate random rotation angles for each neuron
            rot_ang = np.random.rand(len(neur_locs), 3) * 2 * np.pi
            neur_num, neur_num_ad = self._grow_neuron_dendrites(
                neur_soma,
                neur_ves,
                neur_locs,
                gp_nuc,
                gp_soma,
                rot_ang
            )
        else:
            neur_num = np.zeros_like(neur_soma)
            neur_num_ad = np.zeros_like(neur_soma)

        # Set cellular fluorescence
        neur_vol, gp_vals = self._set_cell_fluorescence(
            neur_num,
            neur_soma,
            neur_num_ad,
            neur_locs,
            neur_vol
        )

        # Generate background components if enabled
        if self.bg_params['flag']:
            neur_vol, gp_vals, neur_num = self._generate_background_fluorescence(
                neur_vol,
                neur_num,
                gp_vals,
                gp_nuc
            )

        # Extract vertices and triangulation from neuron bodies
        v_cell = np.array([nb.vertices for nb in self.neuron_bodies])
        v_nuc = np.array([nb.nucleus_vertices for nb in self.neuron_bodies])
        # Get triangulation from first neuron (same for all)
        tri = spiral_sample_sphere(self.neur_params.n_samps)[1]

        return (
            NeuralVolume(
                soma=neur_soma,
                fluorescence=neur_vol,
                nucleus_data=gp_nuc,
                soma_data=gp_soma
            ),
            neur_locs,
            v_cell,
            v_nuc,
            tri,
            neur_soma
        )

    def _sample_dense_neurons(
        self,
        vessel_volume: np.ndarray,
        n_neurons: int,
        min_dist: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample shapes and locations for all somas in a volume.
        
        Args:
            vessel_volume: Array delineating areas occupied by vasculature
            n_neurons: Number of neurons to sample
            min_dist: Minimum distance between neurons (in pixels)
            
        Returns:
            neur_locs: Kx3 array of 3D locations for all K neurons
            v_cell: Vertices defining soma shapes
            v_nuc: Vertices defining nucleus shapes
            tri: Triangulation for surface mesh grids
            rot_ang: Rotation angles of cells
        """
        eta = 1.1  # Expansion factor for minimum distance
        
        # Create exclusion zone around vessels
        x, y, z = np.meshgrid(
            np.arange(-np.ceil(min_dist/2), np.ceil(min_dist/2)+1),
            np.arange(-np.ceil(min_dist/2), np.ceil(min_dist/2)+1),
            np.arange(-np.ceil(min_dist/2), np.ceil(min_dist/2)+1)
        )
        se = np.sqrt(x**2 + y**2 + z**2) <= min_dist/2
        neur_ves_trunc = ndimage.binary_dilation(vessel_volume, structure=se)
        
        # Sample sphere for mesh generation
        v_samp, tri = spiral_sample_sphere(self.neur_params.n_samps)
        self.neur_params.S_samp = v_samp
        self.neur_params.Tri = tri
        
        # Initialize storage
        v_cell = []
        v_nuc = []
        rot_ang = []
        vol_sz = self.vol_params._size
        
        # Create volume meshgrid
        mesh_x, mesh_y, mesh_z = np.meshgrid(
            np.linspace(0, vol_sz[0], int(vol_sz[0] * self.vol_params.res)),
            np.linspace(0, vol_sz[1], int(vol_sz[1] * self.vol_params.res)),
            np.linspace(0, vol_sz[2], int(vol_sz[2] * self.vol_params.res)),
            indexing='ij'
        )
        
        # Get valid volume region
        vol_depth = self.vol_params._depth
        idx_good = ~neur_ves_trunc[:, :, :int(vol_sz[2]*self.vol_params.res)]
        idx_bad = idx_good.copy()
        
        # Sample neuron locations
        neur_locs = np.array([[np.inf, np.inf, np.inf]])
        k = 0
        
        while idx_good.any() and (len(v_cell) < n_neurons):
            k += 1
            
            # Generate neural body
            v_tmp, v_nuc_tmp, _, rot_ang_tmp = self._generate_neural_body(self.neur_params)
            v_cell.append(v_tmp)
            v_nuc.append(v_nuc_tmp)
            rot_ang.append(rot_ang_tmp)
            
            # Sample new location
            if idx_good.any():
                idx_now = np.random.choice(np.where(idx_good.ravel())[0])
            else:
                idx_now = np.random.choice(np.where(idx_bad.ravel())[0])
            
            new_pt = np.array([
                mesh_x.ravel()[idx_now],
                mesh_y.ravel()[idx_now],
                mesh_z.ravel()[idx_now]
            ])
            
            # Center single neuron volumes if requested
            if n_neurons == 1:
                new_pt = np.round(vol_sz / 2)
            
            # Update exclusion zones
            tmp_dist = np.min(np.sqrt(np.sum((new_pt - neur_locs)**2, axis=1)))
            neur_locs = np.vstack((neur_locs, new_pt))
            
            # Update valid locations
            dist_mask = np.sqrt(
                (mesh_x - new_pt[0])**2 +
                (mesh_y - new_pt[1])**2 +
                (mesh_z - new_pt[2])**2
            )
            idx_good[dist_mask <= eta * min_dist] = False
            idx_bad[dist_mask <= min_dist] = False
            idx_good = ~(idx_good | ~idx_bad)
        
        # Format outputs
        neur_locs = neur_locs[1:].astype(np.float32)  # Remove initialization row
        n_neur = len(v_cell)
        
        # Convert lists to arrays and shift to final locations
        v_cell = np.array(v_cell, dtype=np.float32)
        v_nuc = np.array(v_nuc, dtype=np.float32)
        rot_ang = np.array(rot_ang, dtype=np.float32)
        
        for i in range(n_neur):
            v_cell[i] += neur_locs[i]
            v_nuc[i] += neur_locs[i]
        
        return neur_locs, v_cell, v_nuc, tri, rot_ang

    def _generate_neural_body(
        self,
        neur_params: NeuronParams
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate neural body shape using Gaussian process on sphere."""
        # Constants
        pwr = 1  # GP power (sensitive parameter)
        nuc_off = 3  # Nucleus offset

        # Get sphere sampling using mesh utility
        if not hasattr(neur_params, 'S_samp') or not hasattr(neur_params, 'Tri'):
            v_samp, tri = spiral_sample_sphere(neur_params.n_samps)
            neur_params.S_samp = v_samp
            neur_params.Tri = tri
        else:
            v_samp = neur_params.S_samp
            tri = neur_params.Tri

        # Calculate teardrop projection using geometry utility
        if neur_params.neur_type == 'pyr':
            v_tear = teardrop_projection(v_samp, 1)
        elif neur_params.neur_type == 'peanut':
            v_tear = teardrop_projection(v_samp, 2)
        else:
            v_tear = v_samp.copy()

        # Calculate geodesic distances
        if neur_params.dists is None:
            diffs = v_samp[:, np.newaxis, :] - v_samp[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            dists = 2 * np.arcsin(dists/2)  # geodesic distance
            neur_params.dists = dists

        dists = neur_params.p_scale * np.exp(-(neur_params.dists/neur_params.l_scale)**pwr)

        # Ensure positive definite covariance
        min_eig = np.linalg.eigvalsh(dists)[0] * 1.03
        if min_eig < 0:
            dists = dists + abs(min_eig) * np.eye(dists.shape[0])

        # Generate shapes using GP
        exts = np.array([0.8, 1.2])  # Define extent ratios for soma size if not provided
        x_bounds = exts * neur_params.avg_rad
        x_base = np.abs(np.random.multivariate_normal(np.zeros(len(v_tear)), dists))
        x = x_base - np.mean(x_base) + neur_params.avg_rad

        # Normalize radii
        x_min = min(np.min(x), x_bounds[0])
        x = (x_bounds[1] - x_bounds[0]) * (x - x_min) / (max(np.max(x), x_bounds[1]) - x_min) + x_bounds[0]

        # Generate nucleus shape
        if neur_params.neur_type == 'pyr':
            x2 = x_base - np.mean(x_base) + neur_params.avg_rad
            x2_min = min(np.min(x2), x_bounds[0])
            x2 = (x_bounds[1] - x_bounds[0]) * (x2 - x2_min) / (max(np.max(x2), x_bounds[1]) - x2_min) + x_bounds[0]
        else:
            x2 = x.copy()

        # Create elliptical shapes
        eccens = 1 + neur_params.eccen * (np.random.rand(3) - np.array([0.5, 0.5, 0]))
        eccens = eccens / np.power(np.prod(eccens), 1/3)

        # Generate cell shape
        if neur_params.neur_type == 'pyr':
            v_e_tear = v_tear * eccens
            v_e_tear = v_e_tear / np.sqrt(np.mean(np.sum(v_e_tear**2, axis=1)))
        else:
            v_e_tear = v_samp * eccens
            v_e_tear = v_e_tear / np.sqrt(np.mean(np.sum(v_samp**2, axis=1)))

        # Generate final shapes
        v_cell = v_e_tear * x[:, np.newaxis]
        v_cell = v_cell + np.array([0, 0, -nuc_off])
        v_norms = np.sqrt(np.sum(v_cell**2, axis=1))

        # Generate nucleus
        v_nuc = v_samp * np.array([1, 1, -1]) * x2[:, np.newaxis]
        v_norms2 = np.sqrt(np.sum(v_nuc**2, axis=1))
        
        # Define nucleus extents and thickness
        nexts = [0.7, 0.8]  # Default nucleus size parameters if not provided
        min_thic = [1.0, 2.0]  # Default minimum thickness parameters if not provided
        
        # Shrink and smooth nucleus
        v_norms2 = nexts[1] * (
            nexts[0] * (v_norms2 - np.min(v_norms2)) + 
            (1 - nexts[0]) * np.max(v_norms2)
        )
        v_norms2 = v_norms2 + np.min(v_norms - v_norms2) - min_thic[0]
        v_nuc = v_nuc * eccens * (v_norms2 / np.sqrt(np.sum(v_nuc**2, axis=1)))[:, np.newaxis]

        # Apply lateral shift to nucleus
        lat_ang = np.random.rand() * 2 * np.pi
        lat_shift = (1 - abs(np.random.rand() - np.random.rand())) * min_thic[1]
        lat_shift = lat_shift * np.array([np.sin(lat_ang), np.cos(lat_ang)])

        # Apply final offsets
        v_cell = v_cell + np.array([0, 0, nuc_off])
        v_nuc = v_nuc + np.array([lat_shift[0], lat_shift[1], nuc_off])

        # Optional nucleus size scaling
        if hasattr(neur_params, 'nuc_rad') and neur_params.nuc_rad is not None:
            hull = ConvexHull(v_nuc)
            nuc_sz = (4/3) * np.pi * (neur_params.nuc_rad[0]**3)
            if len(neur_params.nuc_rad) > 1:
                v_nuc = v_nuc * (((nuc_sz/hull.volume)**(1/3))**(1/neur_params.nuc_rad[1]))
            else:
                v_nuc = v_nuc * (nuc_sz/hull.volume)**(1/3)

        # Apply rotations using geometry utility
        max_ang = 20  # Default maximum angle if not provided
        rot_ang = -abs(max_ang) + 2 * abs(max_ang) * np.random.rand(3)
        
        for ang, axis in zip(rot_ang, range(3)):
            R = rotation_matrix(ang, axis)
            v_nuc = v_nuc @ R
            v_cell = v_cell @ R

        return v_cell, v_nuc, tri, rot_ang

    def _store_neuron_bodies(
        self,
        neur_locs: np.ndarray,
        v_cell: np.ndarray,
        v_nuc: np.ndarray,
        tri: np.ndarray
    ) -> None:
        """Store generated neuron body information."""
        for i in range(len(neur_locs)):
            self.neuron_bodies.append(
                NeuronBody(
                    vertices=v_cell[i],
                    nucleus_vertices=v_nuc[i],
                    rotation=np.zeros(3),  # Will be set later
                    location=neur_locs[i]
                )
            )

    def _generate_neural_volume(
        self,
        neur_locs: np.ndarray,
        v_cell: np.ndarray,
        v_nuc: np.ndarray,
        neur_ves: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, float]], List[np.ndarray]]:
        """
        Place neural soma in volume and generate fluorescence values.
        
        Args:
            neur_locs: Nx3 array of neuron locations
            v_cell: Surface points of cell somas
            v_nuc: Surface points of cell nuclei
            neur_ves: Binary array indicating blood vessel locations
            
        Returns:
            neur_soma: Binary volume indicating soma locations
            neur_vol: Volume containing fluorescence levels
            gp_nuc: List of nucleus indices and fluorescence values
            gp_soma: List of soma indices
        """
        if self.vol_params.verbose >= 1:
            print('Setting up volume...', end='')
            
        # Initialize volume arrays
        vol_sz = self.vol_params._size
        vres = self.vol_params.res
        full_size = tuple(vol_sz * vres)
        
        neur_soma = np.zeros(full_size, dtype=np.uint16)
        neur_vol = np.zeros(full_size, dtype=np.float32)
        gp_nuc = [(None, None) for _ in range(len(neur_locs))]
        gp_soma = [None for _ in range(len(neur_locs))]
        
        # Setup volume boundaries
        taken_pts = neur_ves if neur_ves is not None else np.zeros(full_size, dtype=bool)
        vol_depth = int(self.vol_params.depth * vres)
        taken_pts = taken_pts[:, :, vol_depth:vol_depth + int(vol_sz[2]*vres)]
        
        # Get sphere triangulation
        _, tri = spiral_sample_sphere(self.neur_params.n_samps)
        
        if self.vol_params.verbose >= 1:
            print('done.\nFinding interior points...')
            
        # Process each neuron
        for k in range(len(neur_locs)):
            # Calculate extent of neuron
            max_ext = np.ceil(np.max(np.sqrt(np.sum(
                (v_cell[:, :, k] - neur_locs[k])**2, axis=1))))
            m_ext_res = int(np.ceil(max_ext * vres))
            
            # Calculate indices for local volume
            idx_pos = np.round(vres * neur_locs[k]).astype(int)
            idx_ranges = [
                np.arange(
                    max(1, pos - m_ext_res),
                    min(pos + m_ext_res + 1, sz * vres)
                )
                for pos, sz in zip(idx_pos, vol_sz)
            ]
            
            # Create local meshgrid
            mesh = np.meshgrid(*[
                (idx - pos)/vres
                for idx, pos in zip(idx_ranges, idx_pos)
            ], indexing='ij')
            
            # Find points to test
            idx_to_test = np.sqrt(sum(m**2 for m in mesh)) <= max_ext
            
            if not idx_to_test.any():
                continue
            
            # Get test points
            test_points = np.column_stack([
                m[idx_to_test].flatten() for m in mesh
            ])
            
            # Test points against soma mesh
            soma_points = in_triangulation(
                v_cell[:, :, k],
                tri,
                test_points,
                heavy_test=1
            )
            
            # Test points against nucleus mesh
            nuc_points = in_triangulation(
                v_nuc[:, :, k],
                tri,
                test_points,
                heavy_test=1
            )
            
            # Get global indices
            global_indices = np.array([
                idx[idx_to_test] for idx in np.meshgrid(*idx_ranges, indexing='ij')
            ]).T
            
            # Store soma points
            soma_mask = soma_points > 0
            if soma_mask.any():
                soma_indices = tuple(global_indices[soma_mask].T)
                neur_soma[soma_indices] = k + 1
                gp_soma[k] = soma_indices
            
            # Store nucleus points
            nuc_mask = nuc_points > 0
            if nuc_mask.any():
                nuc_indices = tuple(global_indices[nuc_mask].T)
                gp_nuc[k] = (nuc_indices, self.neur_params.nuc_fluorsc)
            
            if self.vol_params.verbose >= 2:
                print(f'Processed neuron {k+1}/{len(neur_locs)}')
            
        if self.vol_params.verbose >= 1:
            print('done.')
            
        return neur_soma, neur_vol, gp_nuc, gp_soma

    def _grow_neuron_dendrites(
        self,
        vol_params: VolumeParams,
        dend_params: DendriteParams,
        neur_soma: np.ndarray,
        neur_ves: np.ndarray,
        neur_locs: np.ndarray,
        gp_nuc: List,
        gp_soma: List,
        rot_ang: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, DendriteParams, List]:
        """
        Grow dendrites for neurons in the volume.
        
        Args:
            vol_params: Volume parameters
            dend_params: Dendrite parameters
            neur_soma: Array where k-th neural soma locations are represented by value k
            neur_ves: Array delineating vasculature locations
            neur_locs: Nx3 array of neuron locations
            gp_nuc: List of nucleus indices for each neuron
            gp_soma: List of soma indices for each neuron
            rot_ang: Optional Nx3 array of rotation angles (Rx,Ry,Rz)
            
        Returns:
            neur_num: Array where k-th neuron's locations (soma + dendrites) are marked by k
            dend_num_ad: Volume array containing apical dendrite numbers
            dend_params: Updated dendrite parameters
            gp_soma: Updated soma points
        """
        # Input parsing
        if rot_ang is None:
            rot_ang = np.zeros((len(neur_locs), 3))

        # Extract parameters
        vres = vol_params.vres
        vol_sz = np.array(vol_params.vol_sz)
        N_neur = vol_params.N_neur
        vol_depth = vol_params.vol_depth * vres
        full_dims = tuple(vol_sz * vres)

        # Setup dendrite parameters
        dt_params = dend_params.dt_params  # dendritic tree parameters
        at_params = dend_params.at_params  # apical dendrite parameters
        dims = np.minimum(dend_params.dims, vol_sz / dend_params.dims_ss)
        dims = dims * vres
        
        # Scale parameters
        dt_params[1:3] = dt_params[1:3] * vres
        at_params[1:4] = at_params[1:4] * vres
        thickness_scale = dend_params.thickness_scale * vres * vres

        # Initialize volumes
        cell_volume = np.zeros(full_dims, dtype=np.float32)
        cell_volume_idx = np.zeros(full_dims, dtype=np.uint16)
        cell_volume_val = np.zeros(full_dims, dtype=np.float32)
        cell_volume_ad = np.zeros(full_dims, dtype=np.uint16)

        # Add vessels to cell volume
        cell_volume = (
            neur_soma.astype(np.float32) + 
            (vol_params.N_den + vol_params.N_neur + vol_params.N_bg + 1) * 
            neur_ves[:, :, vol_depth:vol_depth+vol_sz[2]*vres]
        )

        if vol_params.verbose == 1:
            print('Growing out dendrites...', end='', flush=True)
        elif vol_params.verbose > 1:
            print('Growing out dendrites...')

        # Process each neuron
        for j in range(N_neur):
            if vol_params.verbose > 1:
                print(f'Processing neuron {j}...')

            # Get paths for current neuron
            paths = self._generate_dendrite_paths(
                j, neur_locs, dend_params, cell_volume, 
                neur_ves, full_dims, rot_ang[j] if rot_ang is not None else None
            )

            # Get cell body indices
            cell_body = gp_soma[j]
            cell_body_smoothed = self._smooth_cell_body(paths, cell_body, full_dims)

            # Update volumes with paths
            fine_paths_idx, fine_paths_val, fine_paths_ad = self._process_paths(
                paths, cell_body, thickness_scale
            )

            # Update cell volumes
            self._update_cell_volumes(
                cell_volume, cell_volume_idx, cell_volume_val, cell_volume_ad,
                fine_paths_idx, fine_paths_val, fine_paths_ad,
                j, neur_locs[j], full_dims
            )

            # Update soma information
            if len(gp_soma[j]) == 2:  # If gp_soma has space for smoothed body
                gp_soma[j][1] = cell_body_smoothed
            else:
                gp_soma[j] = [gp_soma[j], cell_body_smoothed]

            if vol_params.verbose == 1:
                print('.', end='', flush=True)
            elif vol_params.verbose > 1:
                print(f'Neuron {j} complete')

        # Convert volumes to final types
        cell_volume_val = np.floor(cell_volume_val) + (
            np.mod(cell_volume_val, 1) > np.random.rand(*cell_volume_val.shape)
        ).astype(np.uint16)
        cell_volume_idx = cell_volume_idx.astype(np.uint16)
        cell_volume_ad = cell_volume_ad.astype(np.uint16)
        cell_volume_bd = (~cell_volume_ad.astype(bool)).astype(np.uint16)

        # Dilate paths
        _, dend_num_ad = self._dilate_dendrite_paths(
            cell_volume_val * cell_volume_ad,
            cell_volume_idx * cell_volume_ad,
            neur_soma
        )
        _, dend_num_bd = self._dilate_dendrite_paths(
            cell_volume_val * cell_volume_bd,
            cell_volume_idx * cell_volume_bd,
            neur_soma
        )

        # Clear nuclei and set somas
        for k in range(N_neur):
            dend_num_ad[gp_nuc[k]] = 0
            dend_num_bd[gp_nuc[k]] = 0
            dend_num_ad[gp_soma[k][0]] = 0  # Using first element of gp_soma
            dend_num_bd[gp_soma[k][0]] = 0

        # Combine dendrites
        dend_num_bd[dend_num_ad > 0] = dend_num_ad[dend_num_ad > 0]
        neur_num = np.zeros_like(neur_soma)
        neur_num[dend_num_bd > 0] = dend_num_bd[dend_num_bd > 0]

        # Final cleanup
        for k in range(N_neur):
            neur_num[gp_nuc[k]] = 0
            neur_num[gp_soma[k][0]] = k + 1

        if vol_params.verbose >= 1:
            print('done.')

        return neur_num, dend_num_ad, dend_params, gp_soma

    def _dendrite_dijkstra(
        self,
        M: np.ndarray,
        dims: Tuple[int, ...],
        root: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Python implementation of dendrite_dijkstra_cpp.
        
        Args:
            M: Cost matrix for path finding
            dims: Volume dimensions
            root: Starting point (3D coordinates)
            
        Returns:
            distance: Distance matrix
            pathfrom: Path tracking matrix
        """
        import heapq
        
        # Convert root to linear index
        root_idx = np.ravel_multi_index(root, dims)
        
        # Initialize arrays
        pdims = np.prod(dims)
        distance = np.full(pdims, np.inf, dtype=np.float32)
        pathfrom = np.full(pdims, -1, dtype=np.int32)
        distance[root_idx] = 0
        
        # Setup edge directions (R,L,U,D,F,B)
        edges = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        pe = edges @ np.array([1, dims[0], dims[0]*dims[1]])
        
        # Priority queue for Dijkstra's algorithm
        pq = [(0, root_idx)]
        visited = np.zeros(pdims, dtype=bool)
        
        while pq:
            dist, current = heapq.heappop(pq)
            
            if visited[current]:
                continue
            
            visited[current] = True
            
            # Check neighbors
            for i, offset in enumerate(pe):
                neighbor = current + offset
                
                if 0 <= neighbor < pdims and not visited[neighbor]:
                    new_dist = dist + M[neighbor]
                    
                    if new_dist < distance[neighbor]:
                        distance[neighbor] = new_dist
                        pathfrom[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
        
        # Reshape outputs
        distance = distance.reshape(dims)
        pathfrom = np.stack(np.unravel_index(pathfrom, dims), axis=-1)
        
        return distance, pathfrom

    def _dilate_dendrite_paths(
        self,
        paths: np.ndarray,
        pathnums: np.ndarray,
        obstruction: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Python implementation of dilateDendritePathAll.
        
        Args:
            paths: Full set of simulated paths
            pathnums: Corresponding cell numbers
            obstruction: Occupied space in volume
            
        Returns:
            paths: Updated paths
            pathnums: Updated cell numbers
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
        dval = np.sort(dists.ravel())
        didx = np.argsort(dists.ravel())
        dpos = np.where(np.diff(dval))[0]
        
        # Initialize arrays
        paths = paths.astype(np.float32)
        paths[obstruction] = np.nan
        dims = paths.shape
        pdims = np.prod(dims)
        
        # Calculate shifts for neighbor checking
        dshifts = np.array([
            -dims[0]*dims[1], dims[0]*dims[1],
            -dims[0], dims[0],
            -1, 1
        ])
        
        # Process paths
        idxs = np.where(paths > 1)[0]
        i = 0
        
        while i < max_dist**2 and len(idxs) > 0:
            # Get shifts for current distance
            dx, dy, dz = np.unravel_index(
                didx[dpos[i]+1:dpos[i+1]+1],
                [2*max_dist+1]*3
            )
            dx -= max_dist
            dy -= max_dist
            dz -= max_dist
            
            # Calculate shifts
            jidxs = dz*dims[0]*dims[1] - dy*dims[0] - dx
            
            for j in range(len(idxs)):
                # Get valid neighbor indices
                pidxs = idxs[j] + jidxs
                pidxs = pidxs[(pidxs > 0) & (pidxs < pdims)]
                pidxs = pidxs[paths.ravel()[pidxs] == 0]
                
                if len(pidxs) == 0:
                    continue
                    
                # Check connectivity
                numval = pathnums.ravel()[idxs[j]]
                didxt = np.zeros(len(pidxs), dtype=bool)
                
                for k, pidx in enumerate(pidxs):
                    didxs = pidx + dshifts
                    didxs = didxs[(didxs > 0) & (didxs < pdims)]
                    if np.any(pathnums.ravel()[didxs] == numval):
                        didxt[k] = True
                
                pidxs = pidxs[didxt]
                
                # Dilate paths
                while paths.ravel()[idxs[j]] > 1 and len(pidxs) > 0:
                    ridx = np.random.randint(len(pidxs))
                    pidx = pidxs[ridx]
                    pidxs = np.delete(pidxs, ridx)
                    paths.ravel()[idxs[j]] -= 1
                    paths.ravel()[pidx] = 1
                    pathnums.ravel()[pidx] = numval
            
            idxs = np.where(paths.ravel() > 1)[0]
            i += 1
        
        return paths, pathnums

    def _smooth_cell_body(
        self,
        allpaths: List[np.ndarray],
        cell_body: np.ndarray,
        fdims: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Python implementation of smoothCellBody.
        
        Args:
            allpaths: List of dendrite paths
            cell_body: Indices of cell body points
            fdims: Full dimensions of volume
            
        Returns:
            output: Smoothed cell body indices
        """
        from scipy.interpolate import CubicSpline
        
        # Find connection points
        conn_idx_root = np.zeros((len(allpaths), 3))
        empty_idxs = np.zeros(len(allpaths), dtype=bool)
        
        for i, path in enumerate(allpaths):
            if len(path) > 0:
                path_ind = np.ravel_multi_index(path.T, fdims)
                path_intersect = np.isin(path_ind, cell_body)
                try:
                    conn_idx_root[i] = path[np.where(path_intersect)[0][0]]
                except:
                    empty_idxs[i] = True
            else:
                empty_idxs[i] = True
        
        # Calculate distance matrix
        dist_mat = np.sqrt(np.sum(
            (conn_idx_root[:, None] - conn_idx_root[None, :])**2,
            axis=2
        ))
        dist_mat = (dist_mat == 0).astype(float)
        dist_mat[empty_idxs] = np.nan
        
        # Group dendrites
        dend_groups = []
        for i in range(len(dist_mat)):
            if not np.isnan(dist_mat[i,i]):
                group = np.where(dist_mat[i])[0]
                dend_groups.append(group)
                dist_mat[group] = np.nan
        
        # Process each group
        offset = 2
        conn_idx = np.zeros((len(dend_groups), 3))
        conn_roots = np.zeros((len(dend_groups), 3))
        
        for i, group in enumerate(dend_groups):
            path = allpaths[group[0]]
            path_ind = np.ravel_multi_index(path.T, fdims)
            path_intersect = np.where(np.isin(path_ind, cell_body))[0]
            
            try:
                idx = path_intersect[0] - round(offset * np.sqrt(len(group)))
                conn_idx[i] = path[max(0, idx)]
            except:
                conn_idx[i] = path[0]
            conn_roots[i] = path[path_intersect[0]]
        
        # Create cell matrix
        cell_coords = np.array(np.unravel_index(cell_body, fdims)).T
        cell_min = np.min(cell_coords, axis=0)
        cell_max = np.max(cell_coords, axis=0)
        
        cell_mat = np.zeros(fdims, dtype=bool)
        cell_mat.ravel()[cell_body] = True
        
        # Process borders
        cell_processed = np.zeros(fdims, dtype=bool)
        test_dist = [0, 4, 10]
        num_samp = 20
        
        for j in range(len(conn_roots)):
            dist_off = min(max(test_dist[1], round(offset * np.sqrt(len(dend_groups[j])))), test_dist[2])
            
            # Find border points within distance
            border_dist = np.sqrt(np.sum((conn_roots[j] - cell_coords)**2, axis=1))
            test_idx = np.where((border_dist < dist_off) & (border_dist > test_dist[0]))[0]
            
            # Interpolate paths
            test_sub = []
            for idx in test_idx:
                points = np.vstack([conn_roots[j], conn_idx[j], cell_coords[idx]])
                t = np.arange(len(points))
                cs = CubicSpline(t, points)
                
                t_new = np.linspace(0, len(points)-1, num_samp)
                dpts = np.round(cs(t_new)).astype(int)
                test_sub.append(dpts)
                
            if len(test_sub) > 0:
                test_sub = np.vstack(test_sub)
                test_sub = np.clip(test_sub, [1,1,1], np.array(fdims)-1)
                test_ind = np.ravel_multi_index(test_sub.T, fdims)
                
                # Update cell processed
                cell_bump = cell_mat.copy()
                cell_bump.ravel()[test_ind] = True
                
                # Smooth cell bump
                while True:
                    old_sum = np.sum(cell_bump)
                    cell_bump = self._smooth_volume(cell_bump)
                    if np.sum(cell_bump) == old_sum:
                        break
                    
                cell_processed |= cell_bump
        
        return np.where(cell_processed)[0]

    def _smooth_volume(self, volume: np.ndarray) -> np.ndarray:
        """Helper function to smooth 3D volume by neighbor count."""
        padded = np.pad(volume, 1, mode='constant')
        neighbors = np.zeros_like(volume)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    neighbors += padded[1+dx:1+dx+volume.shape[0],
                                      1+dy:1+dy+volume.shape[1],
                                      1+dz:1+dz+volume.shape[2]]
        
        return (neighbors >= 4) | volume

    def _get_dendrite_path(
        self,
        M: np.ndarray,
        node: np.ndarray,
        root: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Retrieve the path of a dendrite from the full paths matrix until root node.
        
        Args:
            M: Path tracking matrix from Dijkstra's algorithm
            node: End node location in the volume
            root: Starting location for the dendrite path
            
        Returns:
            path: Retrieved path (list of positions)
            pathM: Optional binary matrix indicating path locations
        """
        if len(node) == 2:
            # 2D case
            path = [node]
            current = node
            
            while not np.array_equal(current, root):
                current = M[current[0], current[1]]
                path.append(current)
            
            path = np.array(path)
            
            # Create binary path matrix if requested
            pathM = np.zeros((M.shape[0], M.shape[1]), dtype=bool)
            pathM[path[:, 0], path[:, 1]] = True
            
            return path, pathM
            
        elif len(node) == 3:
            # 3D case
            max_path_length = sum(M.shape)
            path = np.zeros((max_path_length, 3), dtype=int)
            path[0] = node
            i = 0
            current = node
            
            try:
                while not np.array_equal(current, root):
                    i += 1
                    current = M[current[0], current[1], current[2]]
                    path[i] = current
                    
                # Trim path to actual length
                path = path[:i+1]
                
                # Create binary path matrix if requested
                pathM = np.zeros(M.shape[:3], dtype=bool)
                if len(path) > 0:
                    pathM[path[:, 0], path[:, 1], path[:, 2]] = True
                
                return path, pathM
                
            except:
                # Return empty path if path finding fails
                return np.array([]), np.zeros(M.shape[:3], dtype=bool)
                
        else:
            raise ValueError("Number of dimensions of node must be 2 or 3")

    def _generate_dendrite_paths(
        self,
        neuron_idx: int,
        neur_locs: np.ndarray,
        dend_params: DendriteParams,
        cell_volume: np.ndarray,
        neur_ves: np.ndarray,
        full_dims: Tuple[int, ...],
        rot_ang: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Generate dendrite paths for a single neuron using Dijkstra's algorithm.
        
        Args:
            neuron_idx: Index of current neuron
            neur_locs: Nx3 array of neuron locations
            dend_params: Dendrite parameters
            cell_volume: Current cell volume array
            neur_ves: Blood vessel volume array
            full_dims: Full dimensions of volume
            rot_ang: Optional rotation angles [Rx,Ry,Rz]
            
        Returns:
            paths: List of dendrite paths
        """
        # Extract parameters
        dt_params = dend_params.dt_params  # dendritic tree parameters
        at_params = dend_params.at_params  # apical dendrite parameters
        d_weight = dend_params.d_weight
        b_weight = dend_params.b_weight
        
        # Get current neuron location
        neuron_loc = neur_locs[neuron_idx]
        
        # Initialize paths list
        paths = []
        
        # Generate rotation matrix if angles provided
        if rot_ang is not None:
            R = rotation_matrix(rot_ang[0], rot_ang[1], rot_ang[2])
        else:
            R = np.eye(3)
        
        # Sample points on sphere for basal dendrites
        n_basal = int(dt_params[0] + np.random.randn() * dt_params[4])
        basal_points = spiral_sample_sphere(n_basal)[0]
        
        # Scale basal points by radius
        basal_points = basal_points * np.array([dt_params[1], dt_params[1], dt_params[2]])
        
        # Rotate points
        basal_points = basal_points @ R.T
        
        # Add basal dendrite paths
        for point in basal_points:
            # Get target location
            target = np.round(neuron_loc + point).astype(int)
            
            # Ensure target is within bounds
            if not np.all((target >= 0) & (target < full_dims)):
                continue
                
            # Create cost matrix for Dijkstra
            cost_matrix = (
                d_weight * np.random.rand(*full_dims) + 
                b_weight * (cell_volume > 0)
            ).astype(np.float32)
            
            # Run Dijkstra's algorithm
            distance, pathfrom = self._dendrite_dijkstra(
                cost_matrix,
                full_dims,
                neuron_loc.astype(int)
            )
            
            # Get path
            path, _ = self._get_dendrite_path(
                pathfrom,
                target,
                neuron_loc.astype(int)
            )
            
            if len(path) > 0:
                paths.append(path)
        
        # Add apical dendrites if specified
        if at_params[0] > 0:
            n_apical = int(at_params[0])
            apical_points = np.zeros((n_apical, 3))
            
            # Create apical points (typically going upward)
            apical_points[:, 2] = at_params[2]  # z-height
            apical_points[:, :2] = np.random.randn(n_apical, 2) * at_params[1]  # xy spread
            
            # Add offset
            apical_points += np.array([0, 0, at_params[3]])
            
            # Rotate points
            apical_points = apical_points @ R.T
            
            # Add apical dendrite paths
            for point in apical_points:
                target = np.round(neuron_loc + point).astype(int)
                
                if not np.all((target >= 0) & (target < full_dims)):
                    continue
                    
                cost_matrix = (
                    d_weight * np.random.rand(*full_dims) + 
                    b_weight * (cell_volume > 0)
                ).astype(np.float32)
                
                distance, pathfrom = self._dendrite_dijkstra(
                    cost_matrix,
                    full_dims,
                    neuron_loc.astype(int)
                )
                
                path, _ = self._get_dendrite_path(
                    pathfrom,
                    target,
                    neuron_loc.astype(int)
                )
                
                if len(path) > 0:
                    paths.append(path)
        
        return paths

    def _update_cell_volumes(
        self,
        cell_volume: np.ndarray,
        cell_volume_idx: np.ndarray,
        cell_volume_val: np.ndarray,
        cell_volume_ad: np.ndarray,
        fine_paths_idx: np.ndarray,
        fine_paths_val: np.ndarray,
        fine_paths_ad: np.ndarray,
        neuron_idx: int,
        neuron_loc: np.ndarray,
        full_dims: Tuple[int, ...]
    ) -> None:
        """
        Update cell volumes with new dendrite paths.
        
        Args:
            cell_volume: Current cell volume array
            cell_volume_idx: Array tracking cell indices
            cell_volume_val: Array tracking cell values
            cell_volume_ad: Array tracking apical dendrites
            fine_paths_idx: Fine resolution path indices
            fine_paths_val: Fine resolution path values
            fine_paths_ad: Fine resolution apical dendrite markers
            neuron_idx: Current neuron index
            neuron_loc: Current neuron location
            full_dims: Full dimensions of volume
        """
        # Get dimensions
        fdims = fine_paths_idx.shape
        small_z = full_dims[2] <= fdims[2]
        
        # Get unique indices
        fine_idxs = np.where(fine_paths_idx > 0)
        
        # Convert to subscript indices
        xi, yi, zi = np.unravel_index(fine_idxs[0], fdims)
        
        # Calculate offsets based on neuron location
        all_roots = np.ceil(np.maximum(self.vol_params.vres * neuron_loc, 1e-4))
        
        if small_z:
            # Handle case where z dimension is small
            xi = xi + all_roots[0] - fdims[0]//2 - 1
            yi = yi + all_roots[1] - fdims[1]//2 - 1
            
            # Check bounds
            valid_idx = (
                (xi < full_dims[0]) & (yi < full_dims[1]) & (zi < full_dims[2]) &
                (xi >= 0) & (yi >= 0) & (zi >= 0)
            )
            
            # Get valid indices
            xi = xi[valid_idx]
            yi = yi[valid_idx]
            zi = zi[valid_idx]
            
            # Convert to linear indices
            dest_idxs = np.ravel_multi_index((xi, yi, zi), full_dims)
            fine_idxs = fine_idxs[0][valid_idx]
            
        else:
            # Handle normal case
            xi = xi + all_roots[0] - fdims[0]//2 - 1
            yi = yi + all_roots[1] - fdims[1]//2 - 1
            zi = zi + all_roots[2] - fdims[2]//2 - 1
            
            # Check bounds
            valid_idx = (
                (xi < full_dims[0]) & (yi < full_dims[1]) & (zi < full_dims[2]) &
                (xi >= 0) & (yi >= 0) & (zi >= 0)
            )
            
            # Get valid indices
            xi = xi[valid_idx]
            yi = yi[valid_idx]
            zi = zi[valid_idx]
            
            # Convert to linear indices
            dest_idxs = np.ravel_multi_index((xi, yi, zi), full_dims)
            fine_idxs = fine_idxs[0][valid_idx]
        
        # Update volumes
        cell_volume.ravel()[dest_idxs] += fine_paths_idx.ravel()[fine_idxs]
        cell_volume_idx.ravel()[dest_idxs] += fine_paths_idx.ravel()[fine_idxs]
        cell_volume_val.ravel()[dest_idxs] += fine_paths_val.ravel()[fine_idxs]
        cell_volume_ad.ravel()[dest_idxs] += fine_paths_ad.ravel()[fine_idxs]

    def _set_cell_fluorescence(
        self,
        vol_params: Dict,
        neur_params: Dict,
        dend_params: DendriteParams,
        neur_num: np.ndarray,
        neur_soma: np.ndarray,
        neur_num_ad: np.ndarray,
        neur_locs: np.ndarray,
        neur_vol: Optional[np.ndarray] = None
    ) -> Tuple[List, np.ndarray]:
        """
        Set fluorescence values for cells and dendrites.
        
        Args:
            vol_params: Volume parameters
            neur_params: Neuron parameters
            dend_params: Dendrite parameters
            neur_num: Neuron volume
            neur_soma: Soma volume
            neur_num_ad: Apical dendrite volume
            neur_locs: Neuron locations
            neur_vol: Optional existing neuron volume
            
        Returns:
            gp_vals: Cell array with point locations and fluorescence values
            neur_vol: Updated neuron volume
        """
        N_neur = vol_params['N_neur']
        vres = vol_params['vres']
        
        # Initialize storage
        gp_vals = [(None, None, None) for _ in range(N_neur + vol_params['N_den'])]
        
        # Process each neuron
        for i in range(N_neur):
            # Generate 3D Gaussian process for fluorescence
            fluo_dist = self._generate_fluorescence_distribution(
                neur_params['avg_rad'],
                vres,
                neur_params['fluor_dist']
            )
            
            # Get cell points
            cell_points = np.where(neur_num == i+1)
            soma_points = np.where(neur_soma == i+1)
            dend_points = np.where(neur_num_ad == i+1)
            
            # Calculate fluorescence values
            values = self._calculate_fluorescence_values(
                cell_points,
                soma_points,
                dend_points,
                fluo_dist,
                neur_locs[i],
                vres,
                dend_params.weight_scale
            )
            
            # Store results
            gp_vals[i] = (cell_points, values, soma_points)
            
            if neur_vol is not None:
                neur_vol[cell_points] = values
                
        return gp_vals, neur_vol
    
    def _generate_background_fluorescence(
        self,
        neur_vol: np.ndarray,
        neur_num: np.ndarray,
        gp_vals: List[Tuple[np.ndarray, np.ndarray]],
        gp_nuc: List[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Generate background non-uniform fluorescence.
        
        Args:
            neur_vol: Neural volume array
            neur_num: Array indicating neuron assignments
            gp_vals: List of (indices, values) for each neuron
            gp_nuc: List of (indices, fluorescence) for each nucleus
            
        Returns:
            neur_vol: Updated neural volume
            gp_vals: Updated neuron values including background
            neur_num: Updated neuron assignments
        """
        if not self.bg_params.flag:
            return neur_vol, gp_vals, neur_num

        # Get background pixels (where no neurons/vessels exist)
        bg_pix = (neur_num == 0)
        for nuc_idx, _ in gp_nuc:
            bg_pix[nuc_idx] = 0
            
        # Initialize volume arrays
        vres = self.vol_params.vres
        vol_sz = np.array(self.vol_params.vol_sz)
        full_size = tuple(vol_sz * vres)
        
        # Generate background dendrites
        neur_num, neur_vol, gp_vals, _ = self._generate_bgdendrites(
            bg_pix,
            neur_vol,
            neur_num,
            gp_vals,
            full_size
        )
        
        # Generate axons if enabled
        if self.axon_params.flag:
            neur_vol, gp_bgvals, _ = self._generate_axons(
                bg_pix,
                neur_vol,
                neur_num,
                gp_vals,
                gp_nuc
            )
            
            # Sort axons into processes
            cell_pos = np.array([nv.location for nv in self.neuron_bodies]) * vres
            bg_proc = self._sort_axons(gp_bgvals, cell_pos)
            
            # Add background processes to final values
            for proc_idx, proc_val in bg_proc:
                if proc_idx is not None:
                    neur_vol[proc_idx] += proc_val
                    
        return neur_vol, gp_vals, neur_num
    def _sort_axons(
        self,
        gp_bgvals: List[Tuple[np.ndarray, np.ndarray]],
        cell_pos: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sort axons into N_proc bins based on their spatial relationship to neurons.
        
        Args:
            gp_bgvals: List of tuples containing (locations, values) for each axon
            cell_pos: Array of shape (n_neurons, 3) containing neuron positions
            
        Returns:
            bg_proc: List of tuples containing sorted (locations, values) for each process
        """
        # Calculate volume size in voxels
        vol_sz = (self.vol_params.size * self.vol_params.res).astype(int)
        
        if self.vol_params.verbose > 0:
            print('Sorting axons...')

        # Initialize output list
        N_proc = self.axon_params.N_proc
        bg_proc = [(np.array([]), np.array([])) for _ in range(N_proc)]
        
        # Get number of neurons and dendrites from volume size
        N_neur = len(cell_pos)  # Number of neurons is length of cell positions
        N_den = int(self.vol_params.size[0] * 
                    self.vol_params.size[1] * 
                    self.vol_params.size[2] / 1000)  # Rough estimate of dendrites
        
        # If more processes than neurons + dendrites
        if N_proc > N_neur + N_den:
            N_comps = N_neur + N_den
            
            # Calculate mean position of each axon
            gp_bgpos = np.zeros((len(gp_bgvals), 3))
            for kk, (locs, _) in enumerate(gp_bgvals):
                if len(locs) > 0:
                    # Convert linear indices to 3D coordinates
                    coords = np.unravel_index(locs, vol_sz)
                    gp_bgpos[kk] = np.mean(np.column_stack(coords), axis=0)
            
            cell_pos2 = cell_pos[:N_comps]
            
            # Calculate distances between axons and cell bodies
            dist_mat = np.sqrt(
                ((cell_pos2[:, None] - gp_bgpos[None, :]) ** 2).sum(axis=2)
            )
            
            # Assign closest axons to cells
            idxlist = np.zeros(N_comps, dtype=int)
            for ii in range(N_comps):
                idx = np.argmin(dist_mat[ii])
                dist_mat[:, idx] = np.inf
                bg_proc[ii] = gp_bgvals[idx]
                idxlist[ii] = idx
                
            # Randomly distribute remaining axons
            for kk, (locs, vals) in enumerate(gp_bgvals):
                if kk not in idxlist:
                    index = N_comps + int((N_proc - N_comps) * np.random.random())
                    curr_locs, curr_vals = bg_proc[index]
                    bg_proc[index] = (
                        np.concatenate([curr_locs, locs]) if len(curr_locs) > 0 else locs,
                        np.concatenate([curr_vals, vals]) if len(curr_vals) > 0 else vals
                    )
                
        # If fewer processes than neurons + dendrites
        else:
            for locs, vals in gp_bgvals:
                index = int(np.ceil(N_proc * np.random.random())) - 1
                curr_locs, curr_vals = bg_proc[index]
                bg_proc[index] = (
                    np.concatenate([curr_locs, locs]) if len(curr_locs) > 0 else locs,
                    np.concatenate([curr_vals, vals]) if len(curr_vals) > 0 else vals
                )
        
        if self.vol_params.verbose > 0:
            print('done.')
        
        return bg_proc

    def _generate_axons(
        self,
        bg_pix: np.ndarray,
        neur_vol: np.ndarray,
        neur_num: np.ndarray,
        gp_vals: List[Tuple[np.ndarray, np.ndarray]],
        gp_nuc: List[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Generate axon/background processes.
        
        Args:
            bg_pix: Binary array indicating available background pixels
            neur_vol: Neural volume array
            neur_num: Array indicating neuron assignments
            gp_vals: List of (indices, values) for each neuron
            gp_nuc: List of (indices, fluorescence) for each nucleus
            
        Returns:
            neur_vol: Updated neural volume
            gp_bgvals: List of (indices, values) for background processes
            neur_num: Updated neuron assignments
        """
        if self.vol_params.verbose >= 1:
            print('Generating background fluorescence...', end='')

        # Remove nucleus locations from background pixels
        for kk in range(len(gp_nuc)):
            if gp_nuc[kk][0] is not None:
                bg_pix[gp_nuc[kk][0]] = 0

        # Calculate fill number for background processes
        fillnum = round(
            (self.axon_params.maxfill * 
             self.axon_params.maxel * 
             np.sum(bg_pix))
        )

        # Get volume dimensions
        vol_sz = (self.vol_params.size * 
                  self.vol_params.res).astype(int)
        N_bg = self.vol_params.N_bg

        # Initialize background values array
        gp_bgvals = [(None, None) for _ in range(N_bg)]

        # Initialize volume if needed
        if self.vol_params.verbose > 1:
            print('Initializing volume')
        
        neur_vol = np.zeros_like(neur_vol, dtype=np.float32)
        for kk in range(len(gp_vals)):
            if gp_vals[kk][0] is not None:
                neur_vol[gp_vals[kk][0]] = gp_vals[kk][1]
            if kk < len(gp_nuc) and gp_nuc[kk][0] is not None:
                neur_vol[gp_nuc[kk][0]] = gp_nuc[kk][1]
            if self.vol_params.verbose >= 1:
                print('.', end='', flush=True)

        if self.vol_params.verbose > 1:
            print()

        # Setup padded volume for process generation
        padsize = self.axon_params.padsize
        volpad = vol_sz + 2 * padsize

        # Initialize cost matrix
        M = np.random.rand(*volpad).astype(np.float32)
        padded_bg = np.pad(bg_pix == 0, padsize, mode='constant', constant_values=False)
        M[padded_bg] = np.finfo(np.float32).max

        if self.vol_params.verbose > 1:
            from time import time
            start_time = time()

        # Generate background processes
        j = 0  # Process counter
        numit2 = 0
        nummax = 10000

        while (fillnum > 0) and (j < N_bg) and (numit2 < nummax):
            bgpts = []  # Initialize empty path
            numit2 = 0  # Reset stuck counter

            # Try to generate a valid path
            while len(bgpts) < self.axon_params.minlength and numit2 < nummax:
                numit2 += 1

                # Generate random root point
                root = np.random.randint(1, volpad-1, 3)
                while M[tuple(root)] > (self.axon_params.fillweight * 
                                      self.axon_params.maxel):
                    root = np.random.randint(1, volpad-1, 3)

                # Generate random end point
                ends = np.clip(
                    root + np.random.randint(
                        -2 * self.axon_params.maxdist * self.vol_params.res,
                        2 * self.axon_params.maxdist * self.vol_params.res + 1,
                        3
                    ),
                    1, volpad-1
                )

                # Generate random walk between points
                bgpts = self._dendrite_randomwalk2(
                    M, root, ends,
                    self.axon_params.distsc,
                    self.axon_params.maxlength,
                    self.axon_params.fillweight,
                    self.axon_params.maxel,
                    self.axon_params.minlength
                )

            if bgpts:  # If valid path found
                # Generate branches
                nbranches = max(0, round(
                    self.axon_params.numbranches + 
                    self.axon_params.varbranches * np.random.randn()
                ))

                # Add branches
                for i in range(nbranches):
                    bgpts2 = []
                    numit = 0

                    while len(bgpts2) < self.axon_params.minlength and numit < 100:
                        numit += 1
                        
                        # Select random point on existing path as root
                        root = bgpts[np.random.randint(len(bgpts))]
                        while (root[0] in (1, volpad[0]-1) or 
                               root[1] in (1, volpad[1]-1) or 
                               root[2] in (1, volpad[2]-1)):
                            root = bgpts[np.random.randint(len(bgpts))]

                        # Generate random end point
                        ends = np.clip(
                            root + np.random.randint(
                                -2 * self.axon_params.maxdist * self.vol_params.res,
                                2 * self.axon_params.maxdist * self.vol_params.res + 1,
                                3
                            ),
                            1, volpad-1
                        )

                        # Generate branch path
                        bgpts2 = self._dendrite_randomwalk2(
                            M, root, ends,
                            self.axon_params.distsc,
                            self.axon_params.maxlength,
                            self.axon_params.fillweight,
                            self.axon_params.maxel,
                            self.axon_params.minlength
                        )

                    # Add branch points to main path
                    if len(bgpts2) > 0:
                        bgpts = np.vstack([bgpts, bgpts2])

                # Remove padding from points
                bgpts = bgpts - padsize

                # Remove points outside volume
                valid_mask = ~(
                    (bgpts[:, 0] <= 0) | (bgpts[:, 0] > vol_sz[0]) |
                    (bgpts[:, 1] <= 0) | (bgpts[:, 1] > vol_sz[1]) |
                    (bgpts[:, 2] <= 0) | (bgpts[:, 2] > vol_sz[2])
                )
                bgpts = bgpts[valid_mask]

                if len(bgpts) > 0:
                    # Convert to linear indices
                    linear_idx = np.ravel_multi_index(
                        (bgpts[:, 0], bgpts[:, 1], bgpts[:, 2]),
                        vol_sz
                    )

                    # Generate fluorescence values
                    fluo_vals = (1.0 / self.axon_params.maxel) * np.ones(len(bgpts)) * (
                        1 + self.axon_params.varfill * np.random.randn()
                    )

                    # Store values
                    gp_bgvals[j] = (linear_idx, fluo_vals.astype(np.float32))
                    fillnum -= len(bgpts)
                    neur_vol.ravel()[linear_idx] += fluo_vals
                    j += 1

                if self.vol_params.verbose > 1 and j % 1000 == 0:
                    print(f'{j} ({time() - start_time:.1f} seconds).')

        # Update number of background components generated
        self.vol_params.N_bg = j
        gp_bgvals = gp_bgvals[:j]

        if self.vol_params.verbose >= 1:
            print('done.')

        return neur_vol, gp_bgvals, neur_num

    def _generate_bg_dendrites(
        self,
        bg_pix: np.ndarray,
        neural_volume: NeuralVolume,
        gp_nuc: List[Tuple[np.ndarray, float]],
        neur_locs: Optional[np.ndarray] = None
    ) -> Tuple[NeuralVolume, List[Tuple[np.ndarray, float, bool]], np.ndarray]:
        """Generate background dendrites.
        
        Args:
            bg_pix: Binary array indicating available background pixels
            neural_volume: NeuralVolume object containing soma and fluorescence data
            gp_nuc: List of (indices, fluorescence) for each nucleus
            neur_locs: Optional array of neuron locations (in microns)
            
        Returns:
            neural_volume: Updated NeuralVolume object
            gp_vals: List of (indices, values, is_soma) for each component
            neur_locs: Updated neuron locations
        """
        if self.vol_params.verbose == 1:
            print('Generating background fluorescence.', end='')
        elif self.vol_params.verbose > 1:
            print('Generating background fluorescence...')

        # Initialize or get neuron locations
        if neur_locs is None:
            neur_locs = np.array([])

        # Remove nucleus locations from background pixels
        for nuc_idx, _ in gp_nuc:
            if nuc_idx is not None:
                bg_pix[nuc_idx] = 0

        # Get parameters
        vres = self.vol_params.res
        dt_params = self.dend_params.dt_params
        thickness_scale = self.dend_params.thickness_scale
        
        # Scale parameters to resolution
        dt_params[1:3] = dt_params[1:3] * vres
        thickness_scale = thickness_scale * vres * vres
        
        # Get volume dimensions
        vol_sz = (self.vol_params.size * vres).astype(int)

        if self.vol_params.verbose > 1:
            print('Initializing volume')

        # Initialize volume arrays
        neural_volume.fluorescence = np.zeros_like(neural_volume.fluorescence, dtype=np.float32)
        
        # Add existing fluorescence
        for soma_idx, soma_val in neural_volume.soma_data:
            if soma_idx is not None:
                neural_volume.fluorescence[soma_idx] = soma_val
        for nuc_idx, nuc_val in gp_nuc:
            if nuc_idx is not None:
                neural_volume.fluorescence[nuc_idx] = nuc_val
            if self.vol_params.verbose >= 1:
                print('.', end='', flush=True)

        if self.vol_params.verbose > 1:
            print()

        # Initialize cost matrix M
        M = np.random.rand(*vol_sz).astype(np.float32)
        M[bg_pix == 0] = np.finfo(np.float32).max
        M[0, :, :] = M[:, 0, :] = M[:, :, 0] = np.finfo(np.float32).max
        M[-1, :, :] = M[:, -1, :] = M[:, :, -1] = np.finfo(np.float32).max

        if self.vol_params.verbose > 1:
            from time import time
            start_time = time()

        # Get dendrite variation parameter
        dend_var = getattr(self.dend_params, 'dend_var', 0.25)

        # Initialize volume arrays
        idx_vol = np.zeros(vol_sz, dtype=np.uint16)
        num_vol = np.zeros(vol_sz, dtype=np.float32)

        # Set parameters
        dt_size = np.array([dt_params[1], dt_params[1], dt_params[2]])
        num_pts = 0
        idx = 0
        shift_dist = 3

        # Calculate number of dendrites to generate
        n_dends = int(((np.prod(vol_sz + 2*dt_size) / np.prod(vol_sz)) - 1) * 
                      self.vol_params.N_neur)

        # Generate dendrites
        for j in range(n_dends):
            dend_pts = []
            
            # Generate root point outside volume
            root = np.floor(np.random.rand(3) * (vol_sz + 2*dt_size) - dt_size)
            while (0 < root[0] < vol_sz[0] and 
                   0 < root[1] < vol_sz[1] and 
                   0 < root[2] < vol_sz[2]):
                root = np.floor(np.random.rand(3) * (vol_sz + 2*dt_size) - dt_size)
            
            # Add root to neuron locations
            neur_locs = np.vstack([neur_locs, root/vres]) if len(neur_locs) > 0 else np.array([root/vres])

            # Generate dendrites from root
            for i in range(int(dt_params[0])):
                theta = np.random.rand() * 2 * np.pi
                r = np.sqrt(np.random.rand()) * dt_params[1]
                
                # Calculate end point
                dends = np.floor([
                    r * np.cos(theta) + root[0],
                    r * np.sin(theta) + root[1],
                    2 * dt_params[2] * (np.random.rand() - 0.5) + root[2]
                ])

                if (0 < dends[0] < vol_sz[0] and 
                    0 < dends[1] < vol_sz[1] and 
                    0 < dends[2] < vol_sz[2]):
                    
                    # Calculate shift for boundary conditions
                    max_shift, shift_loc = self._calculate_boundary_shift(root, dends, vol_sz)
                    bg_pts = []
                    numit = 0

                    while len(bg_pts) == 0 and numit < 30:
                        numit += 1
                        root2 = np.round(max_shift * (dends - root) + root)
                        
                        # Apply random shift based on boundary location
                        root2 = self._apply_boundary_shift(root2, shift_loc, shift_dist, vol_sz)
                        
                        # Generate random walk
                        bg_pts = self._dendrite_randomwalk2(
                            M, root2, dends,
                            self.bg_params.distsc,
                            self.bg_params.maxlength,
                            self.bg_params.fillweight,
                            self.bg_params.maxel,
                            self.bg_params.minlength
                        )

                        if len(bg_pts) > 0:
                            bg_pts = np.vstack([root2, bg_pts])
                            try:
                                # Calculate dendrite size variation
                                dend_sz = max(0, np.random.normal(1, dend_var)) ** 2
                                
                                # Calculate weights
                                if len(bg_pts) > 2:
                                    diffs = np.abs(np.diff(np.abs(np.diff(bg_pts, axis=0)), axis=0))
                                    weights = dend_sz * (1 - (1 - 1/np.sqrt(2)) * 
                                               np.concatenate([[0], np.sum(diffs, axis=1)/2, [0]]))
                                else:
                                    weights = dend_sz * np.ones(len(bg_pts))
                                
                                # Convert to linear indices
                                bg_pts_idx = np.ravel_multi_index(
                                    (bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2]),
                                    vol_sz
                                )
                                dend_pts.extend(bg_pts_idx)
                                num_vol.ravel()[bg_pts_idx] = weights
                                
                            except Exception as e:
                                print(f"Error processing dendrite points: {e}")
                                continue

            if dend_pts:
                idx += 1
                num_pts += len(dend_pts)
                idx_vol.ravel()[dend_pts] = idx
                num_vol.ravel()[dend_pts] *= thickness_scale * dt_params[3]

        # Dilate dendrite paths
        _, path_num = self._dilate_dendrite_paths(num_vol, idx_vol, ~bg_pix)

        # Update component numbers
        n_comps = self.vol_params.N_neur + self.vol_params.N_den
        path_num[path_num > 0] += n_comps
        neural_volume.soma += path_num

        # Update weight scale parameters
        wt_sc = self.dend_params.weight_scale
        
        # Create new components list
        new_comps = []
        for i in range(n_comps + 1, n_comps + idx + 1):
            comp_idx = np.where(neural_volume.soma == i)[0]
            if len(comp_idx) > 0:
                comp_vals = (wt_sc[1] * np.exp(-(dt_params[1]/vres)/wt_sc[0]) + 
                            (1 - wt_sc[1])) * (1 - wt_sc[2] * np.random.rand(len(comp_idx)))
                new_comps.append((comp_idx, comp_vals, False))
                neural_volume.fluorescence[comp_idx] = comp_vals

        # Update number of dendrites generated
        self.vol_params.N_den2 = idx

        if self.vol_params.verbose >= 1:
            print('done.')

        return neural_volume, new_comps, neur_locs

    def _calculate_boundary_shift(
        self,
        root: np.ndarray,
        dends: np.ndarray,
        vol_sz: np.ndarray
    ) -> Tuple[float, int]:
        """Calculate shift needed to handle boundary conditions."""
        shifts = np.zeros((2, 3))
        shifts[0] = (root < 1) * (1 - root) / (dends - root)
        shifts[1] = (root > vol_sz) * (vol_sz - root) / (dends - root)
        
        max_shift = np.nanmax(shifts)
        shift_loc = np.unravel_index(np.nanargmax(shifts), shifts.shape)[1]
        
        return max_shift, shift_loc

    def _apply_boundary_shift(
        self,
        root: np.ndarray,
        shift_loc: int,
        shift_dist: int,
        vol_sz: np.ndarray
    ) -> np.ndarray:
        """Apply random shift based on boundary location."""
        shifts = {
            0: np.array([0, np.random.randint(shift_dist), np.random.randint(shift_dist)]),
            1: np.array([np.random.randint(shift_dist), 0, np.random.randint(shift_dist)]),
            2: np.array([np.random.randint(shift_dist), np.random.randint(shift_dist), 0])
        }
        
        root = root + shifts.get(shift_loc, np.zeros(3))
        return np.clip(root, 1, vol_sz - 1)

    def _dendrite_randomwalk2(
        self,
        M: np.ndarray,
        root: np.ndarray,
        ends: np.ndarray,
        distsc: float,
        maxlength: int,
        fillweight: float,
        maxel: int,
        minlength: int
    ) -> np.ndarray:
        """Generate random walk path through neural volume.
        
        Args:
            M: Matrix indicating difficulty to occupy each location (single precision)
            root: 1x3 start location for random walk path
            ends: Target end location for the random walk
            distsc: Weighting parameter for directed walk (>0)
            maxlength: Maximum length for random path
            fillweight: Weighting value for single step against occupancy rate
            maxel: Maximum number of components within a single voxel
            minlength: Minimum length for random path
            
        Returns:
            path_out: Output path of the random walk as Nx3 array of coordinates
        """
        # Convert inputs to correct types
        M = M.astype(np.float32)
        root = np.asarray(root, dtype=np.int32)
        ends = np.asarray(ends, dtype=np.int32)
        distsc = float(distsc)
        maxlength = int(maxlength)
        fillweight = float(fillweight)
        maxel = int(maxel)
        minlength = int(minlength)

        # Initialize path with root point
        path = [root]
        current = root
        
        # Direction to target
        direction = ends - root
        direction = direction / np.sqrt(np.sum(direction**2))
        
        # Possible steps in 3D (26-connectivity)
        steps = np.array([
            [x, y, z] for x in [-1,0,1] 
                      for y in [-1,0,1] 
                      for z in [-1,0,1] 
                      if not (x == 0 and y == 0 and z == 0)
        ])
        
        # Try to reach target
        for _ in range(maxlength):
            if len(path) >= maxlength:
                break
                
            # Calculate step probabilities
            possible_positions = current + steps
            
            # Remove positions outside volume
            valid_mask = np.all((possible_positions >= 0) & 
                              (possible_positions < np.array(M.shape)), axis=1)
            if not np.any(valid_mask):
                break
                
            valid_steps = steps[valid_mask]
            valid_positions = possible_positions[valid_mask]
            
            # Get occupancy values for valid positions
            occupancy = np.array([M[tuple(pos)] for pos in valid_positions])
            
            # Skip if all positions are occupied
            if np.all(occupancy >= fillweight * maxel):
                break
                
            # Calculate directional preference
            step_directions = valid_steps / np.sqrt(np.sum(valid_steps**2, axis=1))[:, None]
            directional_weight = np.exp(distsc * np.sum(step_directions * direction, axis=1))
            
            # Calculate total weights
            total_weight = directional_weight * (1 / (1 + occupancy/(fillweight * maxel)))
            
            # Normalize probabilities
            probs = total_weight / np.sum(total_weight)
            
            # Choose next step
            next_idx = np.random.choice(len(valid_steps), p=probs)
            next_pos = valid_positions[next_idx]
            
            # Update current position and path
            current = next_pos
            path.append(current)
            
            # Check if we've reached target
            if np.all(current == ends):
                break
                
        path = np.array(path)
        
        # Return empty path if too short
        if len(path) < minlength:
            return np.array([])
            
        return path[1:]  # Exclude root point from return path
    
    def _process_paths(
        self,
        paths: List[np.ndarray],
        cell_body: Tuple[np.ndarray, ...],
        thickness_scale: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process dendrite paths to generate indices, values and apical dendrite markers.
        
        Args:
            paths: List of Nx3 arrays containing path coordinates
            cell_body: Tuple of arrays containing cell body indices
            thickness_scale: Scaling factor for dendrite thickness
            
        Returns:
            fine_paths_idx: Binary array indicating path locations
            fine_paths_val: Array of fluorescence values along paths
            fine_paths_ad: Binary array marking apical dendrites
        """
        vol_sz = np.array(self.vol_params.vol_sz)
        vres = self.vol_params.vres
        full_dims = tuple(vol_sz * vres)
        
        fine_paths_idx = np.zeros(full_dims, dtype=np.uint16)
        fine_paths_val = np.zeros(full_dims, dtype=np.float32)
        fine_paths_ad = np.zeros(full_dims, dtype=np.uint16)
        
        if len(cell_body) == 0:
            return fine_paths_idx, fine_paths_val, fine_paths_ad
            
        cell_center = np.mean(np.array(cell_body).T, axis=1)
        
        for path in paths:
            if len(path) == 0:
                continue
                
            path = np.clip(path, 0, np.array(full_dims) - 1)
            path_indices = tuple(path.T.astype(int))
            
            # Determine if path is apical (growing upward)
            is_apical = np.mean(path[:, 2] - cell_center[2]) > 0
            
            # Calculate path values
            weights = np.ones(len(path))
            if len(path) > 2:
                # Add distance-based attenuation
                distances = np.sqrt(np.sum((path - cell_center)**2, axis=1))
                weights *= np.exp(-distances / (vol_sz.mean() * vres))
            
            # Apply thickness scale and add slight randomness
            weights *= thickness_scale * (0.9 + 0.2 * np.random.rand(len(weights)))
            
            # Update arrays
            fine_paths_idx[path_indices] = 1
            fine_paths_val[path_indices] = weights
            if is_apical:
                fine_paths_ad[path_indices] = 1
        
        return fine_paths_idx, fine_paths_val, fine_paths_ad
    
    def _generate_fluorescence_distribution(
        self,
        avg_radius: float,
        vres: float,
        fluor_dist: str
    ) -> np.ndarray:
        """
        Generate a fluorescence distribution for neurons.
        
        Args:
            avg_radius: Average radius of the neuron in pixels
            vres: Volume resolution scaling factor
            fluor_dist: Type of fluorescence distribution
            
        Returns:
            np.ndarray: 3D array containing the fluorescence distribution
        """
        # Calculate size of distribution
        size = int(np.ceil(2 * avg_radius * vres))
        if size % 2 == 0:
            size += 1
            
        # Create coordinate grid
        x, y, z = np.meshgrid(
            np.arange(size) - size//2,
            np.arange(size) - size//2,
            np.arange(size) - size//2
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Generate uniform distribution within radius
        dist = np.where(r <= avg_radius * vres, 1.0, 0.0)
        
        return dist

    def _calculate_fluorescence_values(
        self,
        cell_points: np.ndarray,
        soma_points: np.ndarray,
        dend_points: np.ndarray,
        fluo_dist: np.ndarray,
        neuron_loc: np.ndarray,
        vres: float,
        weight_scale: float
    ) -> np.ndarray:
        """
        Calculate fluorescence values for neuron points.
        
        Args:
            cell_points: Array of all cell points
            soma_points: Array of soma points
            dend_points: Array of dendrite points
            fluo_dist: Fluorescence distribution array
            neuron_loc: Location of the neuron
            vres: Volume resolution scaling factor
            weight_scale: Scaling factor for dendrite weights
            
        Returns:
            np.ndarray: Array of fluorescence values for each point
        """
        # Initialize values array
        values = np.zeros(len(cell_points))
        
        # Get distribution dimensions
        dist_size = fluo_dist.shape[0]
        dist_center = dist_size // 2
        
        # Calculate values for soma points
        if len(soma_points) > 0:
            # Calculate relative positions to neuron center
            rel_positions = ((soma_points - neuron_loc) * vres).astype(int)
            # Shift positions to distribution center
            dist_positions = rel_positions + dist_center
            
            # Get valid indices within distribution bounds
            valid_mask = np.all((dist_positions >= 0) & (dist_positions < dist_size), axis=1)
            valid_positions = dist_positions[valid_mask]
            
            # Get values from distribution
            soma_values = fluo_dist[valid_positions[:, 0], 
                                valid_positions[:, 1], 
                                valid_positions[:, 2]]
            
            # Find indices of soma points in cell_points
            soma_indices = np.where(np.isin(cell_points, soma_points).all(axis=1))[0]
            soma_indices = soma_indices[valid_mask]
            
            # Assign values
            values[soma_indices] = soma_values
        
        # Calculate values for dendrite points
        if len(dend_points) > 0:
            # Calculate distance from soma center
            dend_distances = np.sqrt(np.sum((dend_points - neuron_loc)**2, axis=1))
            
            # Calculate dendrite values with distance-based attenuation
            max_distance = np.max(dend_distances)
            if max_distance > 0:
                dend_values = weight_scale * np.exp(-dend_distances / max_distance)
                
                # Find indices of dendrite points in cell_points
                dend_indices = np.where(np.isin(cell_points, dend_points).all(axis=1))[0]
                
                # Assign values
                values[dend_indices] = dend_values
        
        return values
    def _generate_bgdendrites(self, bg_pix: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of generate_bgdendrites.m - Generate background dendrites."""
        pass
    
    
    def _grow_apical_dendrites(self, neur_soma: np.ndarray, neur_ves: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of growApicalDendrites.m - Generate apical dendrites."""
        pass

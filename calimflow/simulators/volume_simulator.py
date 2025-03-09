from typing import Tuple, Dict, Optional
import time;
import numpy as np
from scipy import ndimage

from ..models.parameters import (
    VolumeParams, NeuronParams, VascParams, 
    DendriteParams, NeuralVolume
)
from .vasculature_simulator import VasculatureSimulator
from .neuron_simulator import NeuronSimulator
from ..utils import spiral_sample_sphere, teardrop_projection
from ..utils.geometry import create_disk_structure
from ..utils.sampling import sample_random_location
from ..utils.mesh import in_triangulation

from ..utils.pathfinding import dendrite_dijkstra2, get_dendrite_path2, smooth_cell_body, dilate_dendrite_path_all

class VolumeSimulator:
    def __init__(
        self,
        vol_params: VolumeParams,
        neur_params: NeuronParams,
        vasc_params: Optional[VascParams] = None,
        dend_params: Optional[DendriteParams] = None,
        bg_params: Optional[Dict] = None,
        axon_params: Optional[Dict] = None,
        psf_params: Optional[Dict] = None
    ):
        """Initialize VolumeSimulator with all necessary parameters."""
        self.vol_params = vol_params
        self.neur_params = neur_params
        self.vasc_params = vasc_params or VascParams()
        self.dend_params = dend_params or DendriteParams()
        self.bg_params = bg_params or {'flag': False}
        self.axon_params = axon_params or {'flag': False}
        self.psf_params = psf_params or {}

    def simulate_neural_volume(self) -> Dict:
        """Main function to create a neural volume with somas, dendrites, vasculature, and background."""
        if self.vol_params.verbose >= 1:
            print('Starting neural volume simulation...')

        # 1. Simulate blood vessels
        if self.vasc_params is not None:
            vasc_sim = VasculatureSimulator(self.vol_params, self.vasc_params)
            neur_ves, neur_ves_all = vasc_sim.simulate()
        else:
            neur_ves, neur_ves_all = self._create_empty_volumes()
        

        # 2. Initialize neuron simulator
        neuron_sim = NeuronSimulator(
            self.vol_params,
            self.neur_params,
            self.dend_params,
            self.bg_params,
            self.axon_params
        )

        # 3. Sample neuron locations avoiding vessels
        neur_locs = neuron_sim.sample_locations(neur_ves)

        # 4. Generate neural bodies and processes
        neural_volume, neur_locs, v_cell, v_nuc, tri, neur_soma = neuron_sim.generate_neurons(
            neur_locs, 
            neur_ves
        )

        # 5. Clear vessel locations where somas exist
        neur_ves = self._clear_soma_vessels(neur_ves, neur_soma)

        # Store debug info if verbose
        if self.vol_params.verbose > 1:
            self.debug_data.update({
                'neur_locs': neur_locs,
                'v_cell': v_cell,
                'v_nuc': v_nuc,
                'tri': tri
            })

        # 6. Grow dendrites
        neur_num, cell_volume_ad, dend_params, gp_soma = self.grow_neuron_dendrites(
            self.vol_params,
            self.dend_params,
            neur_soma,
            neur_ves,
            neur_locs,
            neural_volume.nucleus_data,
            neural_volume.soma_data
        )

        # 7. Prepare output
        output = self._prepare_output(
            neural_volume=neural_volume,
            neur_ves=neur_ves,
            neur_ves_all=neur_ves_all,
            neur_locs=neur_locs
        )

        if self.vol_params.verbose >= 1:
            print('Neural volume simulation complete.')

        return output
    
    def _create_empty_volumes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create empty volumes when no vasculature is simulated."""
        vol_sz = self.vol_params.vol_sz
        vres = self.vol_params.vres
        depth = self.vol_params.vol_depth
        full_size = tuple((vol_sz + np.array([0, 0, depth])) * vres)
        return np.zeros(full_size, dtype=bool), np.zeros(full_size, dtype=bool)
    
    def _clear_soma_vessels(self, neur_ves: np.ndarray, neur_soma: np.ndarray) -> np.ndarray:
        """Clear vessel locations where somas exist."""
        indices = np.where(neur_soma > 0)
    
        if len(indices[0]) > 0:  # Check if any soma points exist
            # Calculate depth offset
            depth_offset = self.vol_params.vol_depth * self.vol_params.vres
            
            # Create corresponding coordinates in vessel volume
            ves_indices = (indices[0], indices[1], indices[2] + depth_offset)
            
            # Set those locations to 0 (clear vessels)
            neur_ves[ves_indices] = 0
            
        return neur_ves

    def _prepare_output(
        self,
        neural_volume: NeuralVolume,
        neur_ves: np.ndarray,
        neur_ves_all: np.ndarray,
        neur_locs: np.ndarray
    ) -> Dict:
        """Prepare the output dictionary."""
        output = {
            'neur_vol': neural_volume.fluorescence,
            'gp_nuc': neural_volume.nucleus_data,
            'gp_soma': neural_volume.soma_data,
            'neur_ves': neur_ves,
            'neur_ves_all': neur_ves_all,
            'locs': neur_locs,
            'bg_proc': []  # Will be populated if axons are enabled
        }

        return output

    def generate_neural_body(self, neur_params: NeuronParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a neural shape using an Isotropic Gaussian process on a sphere.

        Args:
            neur_params: Neuron generation parameters

        Returns:
            Vcell: Points on the neural surface
            Vnuc: Points on the nucleus surface  
            Tri: Triangulation matrix for surfaces
            a: Rotation angles [Rx,Ry,Rz]
        """
        pwr = 1  # GP power (sensitive parameter)
        nuc_off = 3  # Nucleus offset

        # Get sphere sampling if not provided
        if not hasattr(neur_params, 'S_samp') or not hasattr(neur_params, 'Tri'):
            V, Tri = spiral_sample_sphere(neur_params.n_samps)
            neur_params.S_samp = V
            neur_params.Tri = Tri
        else:
            V = neur_params.S_samp
            Tri = neur_params.Tri

        # Calculate teardrop projection if needed
        if neur_params.neur_type == 'pyr':
            V_tear = teardrop_projection(V, 1)
        elif neur_params.neur_type == 'peanut':
            V_tear = teardrop_projection(V, 2)
        else:
            V_tear = V

        # Calculate geodesic distances
        diffs = V[:, np.newaxis, :] - V[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        dists = 2 * np.arcsin(dists/2)  # geodesic distance
        
        # Create covariance matrix directly without storing in neur_params
        cov_matrix = neur_params.p_scale * np.exp(-(dists/neur_params.l_scale)**pwr)
        
        # Ensure positive definite
        min_eig = np.linalg.eigvalsh(cov_matrix)[0] * 1.03
        if min_eig < 0:
            cov_matrix = cov_matrix + abs(min_eig) * np.eye(cov_matrix.shape[0])

        # Generate shapes using GP - convert exts to numpy array
        exts = np.array(neur_params.exts)  # Convert list to numpy array
        x_bounds = exts * neur_params.avg_rad
        x_base = np.abs(np.random.multivariate_normal(np.zeros(len(V_tear)), cov_matrix))
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
            v_e_tear = V_tear * eccens
            v_e_tear = v_e_tear / np.sqrt(np.mean(np.sum(v_e_tear**2, axis=1)))
        else:
            v_e_tear = V * eccens
            v_e_tear = v_e_tear / np.sqrt(np.mean(np.sum(V**2, axis=1)))

        # Generate final shapes
        Vcell = v_e_tear * x[:, np.newaxis]
        Vcell = Vcell + np.array([0, 0, -nuc_off])
        v_norms = np.sqrt(np.sum(Vcell**2, axis=1))

        # Generate nucleus
        Vnuc = V * np.array([1, 1, -1]) * x2[:, np.newaxis]
        v_norms2 = np.sqrt(np.sum(Vnuc**2, axis=1))

        # Define nucleus extents and thickness
        nexts = neur_params.nexts if hasattr(neur_params, 'nexts') else [0.7, 0.8]
        min_thic = [neur_params.min_thic, 2.0]  # Use single value and default

        # Shrink and smooth nucleus
        v_norms2 = nexts[1] * (nexts[0] * (v_norms2 - np.min(v_norms2)) + (1 - nexts[0]) * np.max(v_norms2))
        v_norms2 = v_norms2 + np.min(v_norms - v_norms2) - min_thic[0]
        Vnuc = Vnuc * eccens * (v_norms2 / np.sqrt(np.sum(Vnuc**2, axis=1)))[:, np.newaxis]

        # Apply lateral shift to nucleus
        lat_ang = np.random.rand() * 2 * np.pi
        lat_shift = (1 - abs(np.random.rand() - np.random.rand())) * min_thic[1]
        lat_shift = lat_shift * np.array([np.sin(lat_ang), np.cos(lat_ang)])

        # Apply final offsets
        Vcell = Vcell + np.array([0, 0, nuc_off])
        Vnuc = Vnuc + np.array([lat_shift[0], lat_shift[1], nuc_off])

        # Optional nucleus size scaling
        if hasattr(neur_params, 'nuc_rad') and neur_params.nuc_rad is not None:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(Vnuc)
            nuc_sz = (4/3) * np.pi * (neur_params.nuc_rad[0]**3)
            if len(neur_params.nuc_rad) > 1:
                Vnuc = Vnuc * ((nuc_sz/hull.volume)**(1/3))**(1/neur_params.nuc_rad[1])
            else:
                Vnuc = Vnuc * (nuc_sz/hull.volume)**(1/3)

        # Apply rotations
        max_ang = 20 if not hasattr(neur_params, 'max_ang') else neur_params.max_ang
        rot_ang = -abs(max_ang) + 2 * abs(max_ang) * np.random.rand(3)
        
        for ang, axis in zip(rot_ang, range(3)):
            R = self._rotation_matrix(ang, axis)
            Vnuc = Vnuc @ R
            Vcell = Vcell @ R

        return Vcell, Vnuc, Tri, rot_ang

    def _rotation_matrix(self, angle: float, axis: int) -> np.ndarray:
        """Generate rotation matrix for given angle and axis."""
        c = np.cos(np.deg2rad(angle))
        s = np.sin(np.deg2rad(angle))
        if axis == 0:  # x-axis
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1:  # y-axis
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:  # z-axis
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def sample_dense_neurons(
        self, 
        neur_params: NeuronParams,
        vol_params: VolumeParams,
        neur_ves: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample shapes and locations for all somas in a volume.

        Args:
            neur_params: Neuron parameters
            vol_params: Volume parameters
            neur_ves: Array delineating areas occupied by vasculature

        Returns:
            neur_locs: Kx3 array of 3D locations for all K neurons
            v_cell: Vertices defining soma shapes
            v_nuc: Vertices defining nucleus shapes
            tri: Triangulation for surface mesh
            rot_ang: Rotation angles [Rx,Ry,Rz]
        """
        # Constants
        eta = 1.1  # Expansion factor for minimum distance

        # Create exclusion zone around vessels
        x, y, z = np.meshgrid(
            np.arange(-np.ceil(vol_params.mine_dist/2), np.ceil(vol_params.mine_dist/2)+1),
            np.arange(-np.ceil(vol_params.mine_dist/2), np.ceil(vol_params.mine_dist/2)+1),
            np.arange(-np.ceil(vol_params.mine_dist/2), np.ceil(vol_params.mine_dist/2)+1)
        )
        se = np.sqrt(x**2 + y**2 + z**2) <= vol_params.mine_dist/2
        neur_ves_trunc = ndimage.binary_dilation(neur_ves, structure=se)

        # Get sphere sampling for mesh generation
        v_samp, tri = spiral_sample_sphere(neur_params.n_samps)
        neur_params.S_samp = v_samp
        neur_params.Tri = tri

        # Initialize storage
        v_cell = []
        v_nuc = []
        rot_ang = []
        vol_sz = vol_params.size
        
        # Create volume meshgrid
        mesh_x, mesh_y, mesh_z = np.meshgrid(
            np.linspace(0, vol_sz[0], int(vol_sz[0] * vol_params.res)),
            np.linspace(0, vol_sz[1], int(vol_sz[1] * vol_params.res)),
            np.linspace(0, vol_sz[2], int(vol_sz[2] * vol_params.res)),
            indexing='ij'
        )

        # Get valid volume region
        vol_depth = vol_params.depth * vol_params.res
        idx_good = ~neur_ves_trunc[:, :, :int(vol_sz[2]*vol_params.res)]
        idx_bad = idx_good.copy()

        if vol_params.verbose == 1:
            print('Sampling random locations for neurons...', end='')
        elif vol_params.verbose > 1:
            print('Sampling random locations for neurons...')

        # Sample neuron locations
        neur_locs = np.array([[np.inf, np.inf, np.inf]])
        k = 0
        
        while idx_good.any() and len(v_cell) < vol_params.n_neur:
            k += 1
            
            # Generate neural body
            v_tmp, v_nuc_tmp, _, rot_ang_tmp = self.generate_neural_body(neur_params)
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
            if vol_params.n_neur == 1:
                new_pt = np.round(vol_params.size / 2)

            # Update exclusion zones
            tmp_dist = np.min(np.sqrt(np.sum((new_pt - neur_locs)**2, axis=1)))
            neur_locs = np.vstack((neur_locs, new_pt))
            
            # Update valid locations
            dist_mask = np.sqrt(
                (mesh_x - new_pt[0])**2 +
                (mesh_y - new_pt[1])**2 +
                (mesh_z - new_pt[2])**2
            )
            idx_good[dist_mask <= eta * vol_params.mine_dist] = False
            idx_bad[dist_mask <= vol_params.mine_dist] = False
            idx_good = ~(idx_good | ~idx_bad)

            if vol_params.verbose > 1:
                print(f'Neuron {k}, distance: {tmp_dist:.2f}')

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

        if vol_params.verbose >= 1:
            print('done.')

        return neur_locs, v_cell, v_nuc, tri, rot_ang

    def generate_neural_volume(
        self,
        neur_params: NeuronParams,
        vol_params: VolumeParams,
        neur_locs: np.ndarray,
        Vcell: np.ndarray,
        Vnuc: np.ndarray,
        neur_ves: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Place neural soma in volume.

        Args:
            neur_params: Neuron parameters
            vol_params: Volume parameters
            neur_locs: Neuron locations (Nx3 array)
            Vcell: Surface points of cell somas
            Vnuc: Surface points of cell nuclei
            neur_ves: Binary array of vessel locations

        Returns:
            neur_soma: Array indicating neuron locations
            neur_vol: Array containing fluorescence levels
            gp_nuc: List of (locations, fluorescence) tuples for nuclei
            gp_soma: List of location arrays for cell bodies
        """
        import time
        from datetime import timedelta
        
        if vol_params.verbose >= 1:
            print('Setting up volume...')

        # Initialize arrays
        vol_sz = vol_params.size
        vres = vol_params.res
        neur_soma = np.zeros(tuple(vol_sz * vres), dtype=np.uint16)
        neur_vol = np.zeros(tuple(vol_sz * vres), dtype=np.float32)
        gp_nuc = []  # Will store (locations, fluorescence) tuples
        gp_soma = []  # Will store location arrays
        taken_pts = neur_ves.copy()
        
        # Get volume depth and trim vasculature
        vol_depth = vol_params.depth * vres
        taken_pts = taken_pts[:, :, vol_depth:vol_depth + int(vol_sz[2] * vres)]

        # Get sphere triangulation
        _, Tri = spiral_sample_sphere(neur_params.n_samps)

        if vol_params.verbose == 1:
            print('done.\nFinding interior points...', end='')
        elif vol_params.verbose > 1:
            print('done.\nFinding interior points...')

        # Initialize progress tracking variables
        total_neurons = len(neur_locs)
        start_time = time.time()
        last_update_time = start_time
        progress_update_interval = 2.0  # Update every 2 seconds
        
        # Process each neuron
        for k in range(total_neurons):
            neuron_start_time = time.time()
            
            # Find maximum extent of soma
            max_ext = np.ceil(np.max(np.sqrt(np.sum((Vcell[k] - neur_locs[k])**2, axis=1))))
            
            # Calculate indices for local region
            idx_pos = np.round(vres * neur_locs[k]).astype(int)
            idx_ranges = [
                np.arange(
                    max(0, pos - int(vres * max_ext)),
                    min(pos + int(vres * max_ext), sz * vres)
                )
                for pos, sz in zip(idx_pos, vol_sz)
            ]
            
            # Create local meshgrid
            mesh_coords = np.meshgrid(*[
                (idx - pos)/vres for idx, pos in zip(idx_ranges, idx_pos)
            ], indexing='ij')
            
            # Find points to test
            idx_to_test = np.sqrt(sum(m**2 for m in mesh_coords)) <= max_ext
            
            # Get test points
            idx_tri = np.column_stack([
                m[idx_to_test] + pos/vres + 1/(2*vres)
                for m, pos in zip(mesh_coords, idx_pos)
            ])
            
            # Test points inside soma and nucleus
            soma_inside = in_triangulation(Vcell[k], Tri, idx_tri)
            nuc_inside = in_triangulation(Vnuc[k], Tri, idx_tri)
            
            # Create full boolean arrays
            soma_mask = np.zeros_like(idx_to_test)
            nuc_mask = np.zeros_like(idx_to_test)
            soma_mask[idx_to_test] = soma_inside
            nuc_mask[idx_to_test] = nuc_inside
            
            # Remove nucleus points and overlapping points
            neur_idx = soma_mask & (~nuc_mask)
            neur_idx = neur_idx & (~taken_pts[
                np.ix_(*idx_ranges)
            ])
            
            # Update taken points
            taken_pts[
                np.ix_(*idx_ranges)
            ] |= neur_idx
            
            # Get linear indices
            ix, iy, iz = np.nonzero(neur_idx)
            ix += idx_ranges[0][0]
            iy += idx_ranges[1][0]
            iz += idx_ranges[2][0]
            lin_idx = np.ravel_multi_index((ix, iy, iz), tuple(vol_sz * vres))
            
            # Store soma points
            neur_soma.ravel()[lin_idx] = k + 1
            gp_soma.append(lin_idx.astype(np.int32))
            
            # Process nucleus points
            ix, iy, iz = np.nonzero(nuc_mask)
            ix += idx_ranges[0][0]
            iy += idx_ranges[1][0]
            iz += idx_ranges[2][0]
            nuc_idx = np.ravel_multi_index((ix, iy, iz), tuple(vol_sz * vres))
            
            # Store nucleus points and fluorescence
            gp_nuc.append((nuc_idx.astype(np.int32), neur_params.nuc_fluorsc))
            neur_vol.ravel()[nuc_idx] = neur_params.nuc_fluorsc
            
            # Update progress
            current_time = time.time()
            if vol_params.verbose == 1:
                print('.', end='', flush=True)
            elif vol_params.verbose > 1:
                neuron_time = current_time - neuron_start_time
                print(f'Neuron {k+1}/{total_neurons} done ({neuron_time:.2f} seconds)')
            
            # Show periodic progress updates
            if (vol_params.verbose >= 1 and 
                current_time - last_update_time > progress_update_interval and 
                k < total_neurons - 1):  # Don't show on last iteration
                
                progress = (k + 1) / total_neurons
                elapsed = current_time - start_time
                estimated_total = elapsed / progress if progress > 0 else 0
                remaining = max(0, estimated_total - elapsed)
                
                # Format time as hh:mm:ss
                remaining_str = str(timedelta(seconds=int(remaining)))
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                
                # Create a clean progress bar
                bar_width = 20
                filled_width = int(round(bar_width * progress))
                bar = '█' * filled_width + '░' * (bar_width - filled_width)
                
                print(f"\r\033[K[{bar}] {progress*100:.1f}% - {k+1}/{total_neurons} neurons processed - ETA: {remaining_str} (elapsed: {elapsed_str})", end="", flush=True)
                last_update_time = current_time

        # Final progress report
        if vol_params.verbose >= 1:
            final_time = time.time() - start_time
            final_time_str = str(timedelta(seconds=int(final_time)))
            print(f"\r\033[KNeural volume generation completed! {total_neurons}/{total_neurons} neurons processed in {final_time_str}.")
            print('done.')

        return neur_soma, neur_vol, gp_nuc, gp_soma

    def grow_neuron_dendrites(
        self,
        vol_params: VolumeParams,
        dend_params: DendriteParams,
        neur_soma: np.ndarray,
        neur_ves: np.ndarray,
        neur_locs: np.ndarray,
        gp_nuc: list,
        gp_soma: list,
        rot_ang: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, DendriteParams, list]:
        """Grow dendrites in neural volume."""
        if vol_params.verbose == 1:
            print('Growing out dendrites...', end='')
        elif vol_params.verbose > 1:
            print('Growing out dendrites...')

        # Extract parameters
        vres = vol_params.res
        n_neur = vol_params.n_neur
        vol_sz = vol_params.size
        dims = np.minimum(dend_params.dims, vol_sz/dend_params.dims_ss)
        full_dims = vol_sz * vres
        dims = dims * vres

        # Scale parameters
        dt_params = dend_params.dt_params.copy()
        at_params = dend_params.at_params.copy()
        dt_params[1:3] = dt_params[1:3] * vres
        at_params[1:4] = at_params[1:4] * vres
        thickness_scale = dend_params.thickness_scale * vres * vres

        # Initialize volumes
        cell_volume = (neur_soma.astype(np.float32) + 
                    (vol_params.n_den + vol_params.n_neur + vol_params.n_bg + 1) * 
                    neur_ves[:, :, vol_params.depth:vol_params.depth + int(vol_sz[2]*vres)])

        # Add nucleus data
        for k in range(n_neur):
            if len(gp_nuc[k][0]) > 0:
                cell_volume.ravel()[gp_nuc[k][0]] = k + 1

        # Initialize outputs
        neur_num = cell_volume.astype(np.uint16)
        cell_volume_ad = np.zeros_like(neur_soma, dtype=bool)
        
        # Get neuron roots
        all_roots = np.ceil(np.maximum(vres * neur_locs, 1e-4))

        # Initialize dimensions and flags
        full_dims = (vol_sz * vres).astype(int)
        dims = (dims * vres).astype(int)
        dimsSS = dend_params.dims_ss
        small_z = full_dims[2] <= dims[2] * dimsSS[2]

        # Process each neuron
        for j in range(n_neur):
            try:
                # Get soma center and calculate apical root
                ap_root = np.array(np.unravel_index(np.min(gp_soma[j]), full_dims))
                num_dt = max(1, int(dt_params[0] + round(dt_params[4] * np.random.randn())))

                # Create boundary flag
                border_flag = 0
                try:
                    if small_z:
                        root_l = np.array([dims[0]//2 + 1, dims[1]//2 + 1, all_roots[j,2]], dtype=int)
                        x_start = all_roots[j,0].astype(int) - dims[0]//2 - 1
                        x_end = x_start + dims[0]
                        y_start = all_roots[j,1].astype(int) - dims[1]//2 - 1
                        y_end = y_start + dims[1]
                        obstruction = cell_volume[x_start:x_end, y_start:y_end, :]
                    else:
                        root_l = np.array([dims[0]//2 + 1, dims[1]//2 + 1, dims[2]//2 + 1], dtype=int)
                        x_start = all_roots[j,0].astype(int) - dims[0]//2 - 1
                        x_end = x_start + dims[0]
                        y_start = all_roots[j,1].astype(int) - dims[1]//2 - 1
                        y_end = y_start + dims[1]
                        z_start = all_roots[j,2].astype(int) - dims[2]//2 - 1
                        z_end = z_start + dims[2]
                        obstruction = cell_volume[x_start:x_end, y_start:y_end, z_start:z_end]

                except Exception as e:
                    # Handle edge cases with proper integer array shapes
                    if small_z:
                        root_l = np.array([dims[0]//2 + 1, dims[1]//2 + 1, all_roots[j,2]], dtype=int)
                        shape = [dims[0], dims[1], cell_volume.shape[2]] if not small_z else dims
                        obstruction = np.zeros(shape, dtype=np.float32)
                        x_lims = [all_roots[j,0] - dims[0]//2, all_roots[j,0] + dims[0]//2 - 1]
                        y_lims = [all_roots[j,1] - dims[1]//2, all_roots[j,1] + dims[1]//2 - 1]
                        
                        # Calculate valid ranges
                        x_valid = slice(
                            max(0, -x_lims[0] + 2),
                            dims[0] + min(0, -x_lims[1] + cell_volume.shape[0])
                        )
                        y_valid = slice(
                            max(0, -y_lims[0] + 2),
                            dims[1] + min(0, -y_lims[1] + cell_volume.shape[1])
                        )
                        
                        # Copy valid portion of volume
                        obstruction[x_valid, y_valid, :] = cell_volume[
                            max(0, x_lims[0]):min(x_lims[1], cell_volume.shape[0]),
                            max(0, y_lims[0]):min(y_lims[1], cell_volume.shape[1]),
                            :
                        ]
                        border_flag = 1
                    else:
                        root_l = np.array([dims[0]//2 + 1, dims[1]//2 + 1, dims[2]//2 + 1], dtype=int)
                        obstruction = np.zeros(dims, dtype=np.float32)
                        x_lims = [all_roots[j,0] - dims[0]//2, all_roots[j,0] + dims[0]//2 - 1]
                        y_lims = [all_roots[j,1] - dims[1]//2, all_roots[j,1] + dims[1]//2 - 1]
                        z_lims = [all_roots[j,2] - dims[2]//2, all_roots[j,2] + dims[2]//2 - 1]
                        
                        # Calculate valid ranges
                        x_valid = slice(
                            max(0, -x_lims[0] + 2),
                            dims[0] + min(0, -x_lims[1] + cell_volume.shape[0])
                        )
                        y_valid = slice(
                            max(0, -y_lims[0] + 2),
                            dims[1] + min(0, -y_lims[1] + cell_volume.shape[1])
                        )
                        z_valid = slice(
                            max(0, -z_lims[0] + 2),
                            dims[2] + min(0, -z_lims[1] + cell_volume.shape[2])
                        )
                        
                        # Copy valid portion of volume
                        obstruction[x_valid, y_valid, z_valid] = cell_volume[
                            max(0, x_lims[0]):min(x_lims[1], cell_volume.shape[0]),
                            max(0, y_lims[0]):min(y_lims[1], cell_volume.shape[1]),
                            max(0, z_lims[0]):min(z_lims[1], cell_volume.shape[2])
                        ]
                        border_flag = 1

                # Find cell body and clear from obstruction
                cell_body = np.where(obstruction == j + 1)[0]
                obstruction.ravel()[cell_body] = 0

                # Create weights matrix and reshape with proper dimension handling
                root = np.ceil(root_l / dimsSS)
                root = np.minimum(root, dims).astype(int)  # Ensure integer values
                
                # Get integer dimensions for M
                M_dims = np.array([dims[0], dims[1], dims[2]], dtype=int)
                
                # Create weight matrix with exact dimensions matching MATLAB code
                M = 1 + dend_params.dweight * np.random.rand(M_dims[0], M_dims[1], M_dims[2], 6)
                M_flat = M.reshape(-1, 6)
                
                # Convert obstruction to match M dimensions
                if obstruction.size == 0:
                    obstruction = np.zeros(M_dims, dtype=np.float32)
                else:
                    # Ensure obstruction has exact dimensions needed
                    obstruction_resized = np.zeros(M_dims, dtype=np.float32)
                    shared_x = min(obstruction.shape[0], M_dims[0])
                    shared_y = min(obstruction.shape[1], M_dims[1])
                    shared_z = min(obstruction.shape[2], M_dims[2])
                    
                    obstruction_resized[:shared_x, :shared_y, :shared_z] = \
                        obstruction[:shared_x, :shared_y, :shared_z]
                    obstruction = obstruction_resized
                
                # Create mask with matching dimensions
                obstruction_mask = obstruction.reshape(-1) > 0
                
                # Now these should match in size
                M_flat[obstruction_mask] = np.inf

                # Add boundary constraints with correct dimensions
                dims_int = dims.astype(int)
                M_flat[0::dims_int[0], 0] = np.inf  # Right edge
                M_flat[dims_int[0]-1::dims_int[0], 1] = np.inf  # Left edge
                M_flat[:dims_int[0]*dims_int[1], 2] = np.inf  # Top edge
                M_flat[-dims_int[0]*dims_int[1]:, 3] = np.inf  # Bottom edge
                M_flat[::dims_int[0]*dims_int[1], 4] = np.inf  # Front face
                M_flat[dims_int[0]*dims_int[1]-1::dims_int[0]*dims_int[1], 5] = np.inf  # Back face

                # Run Dijkstra's algorithm
                distance, pathfrom = dendrite_dijkstra2(M_flat, dims_int, root)

                # Generate endpoints for dendrites
                paths = np.zeros(dims, dtype=bool)
                all_paths = []
                fine_paths_idx = np.zeros(dims, dtype=np.float32)
                fine_idxs = []

                # Generate random endpoints and paths
                for i in range(num_dt):
                    # Generate random endpoint
                    theta = np.random.random() * 2 * np.pi
                    r = np.sqrt(np.random.random()) * dt_params[1]
                    end_pos = np.floor([
                        r * np.cos(theta) + root_l[0],
                        r * np.sin(theta) + root_l[1],
                        2 * dt_params[2] * (np.random.random() - 0.5) + root_l[2]
                    ])
                    
                    # Clip to bounds
                    end_pos = np.clip(end_pos, [1, 1, 1], dims - 1)
                    
                    # Get path using dendrite_dijkstra2
                    path, _ = get_dendrite_path2(pathfrom, end_pos.astype(int), root_l.astype(int))
                    all_paths.append(path)

                    if len(path) > 0:
                        # Calculate dendrite thickness variation
                        dend_sz = max(0, np.random.normal(1, dend_params.dend_var)) ** 2
                        
                        # Calculate path weights based on twists
                        if len(path) > 2:
                            diffs = np.abs(np.diff(np.abs(np.diff(path, axis=0)), axis=0))
                            path_w = dend_sz * (1 - (1 - 1/np.sqrt(2)) * np.concatenate([
                                [0], np.sum(diffs, axis=1)/2, [0]
                            ]))
                        else:
                            path_w = dend_sz * np.ones(len(path))

                        # Add path to volume
                        path_indices = np.ravel_multi_index(
                            (path[:, 0].astype(int), 
                             path[:, 1].astype(int), 
                             path[:, 2].astype(int)), dims)
                        paths.ravel()[path_indices] = True
                        fine_paths_idx.ravel()[path_indices] += path_w
                        fine_idxs.extend(path_indices)

                # Process paths
                fine_idxs = np.unique(fine_idxs)
                if len(fine_idxs) > 0:
                    fine_paths_idx.flat[fine_idxs] = (
                        thickness_scale * dt_params[3] * 
                        (fine_paths_idx.flat[fine_idxs] ** (1/dend_params.rall_exp))
                    )

                # Get cell indices from paths
                cell_idx = np.where(paths)  # Add this line to get indices

                # Update cell volume with proper bounds checking
                if len(cell_idx[0]) > 0:  # Only update if paths exist
                    # Calculate target indices with bounds checking
                    x_shift = int(all_roots[j,0] - dims[0]/2)
                    y_shift = int(all_roots[j,1] - dims[1]/2)
                    
                    # Calculate target coordinates
                    x_coords = cell_idx[0] + x_shift
                    y_coords = cell_idx[1] + y_shift
                    z_coords = cell_idx[2]  # Z coordinates don't need shifting
                    
                    # Create mask for valid indices
                    valid_mask = (
                        (x_coords >= 0) & (x_coords < cell_volume.shape[0]) &
                        (y_coords >= 0) & (y_coords < cell_volume.shape[1]) &
                        (z_coords >= 0) & (z_coords < cell_volume.shape[2])
                    )
                    
                    if np.any(valid_mask):
                        # Extract valid coordinates
                        x_valid = x_coords[valid_mask]
                        y_valid = y_coords[valid_mask]
                        z_valid = z_coords[valid_mask]
                        
                        # Update volumes using only valid coordinates
                        cell_volume[x_valid, y_valid, z_valid] = j + 1
                        neur_num[x_valid, y_valid, z_valid] = j + 1

                if vol_params.verbose == 1:
                    print('.', end='', flush=True)
                elif vol_params.verbose > 1:
                    print(f'Neuron {j+1} processed')

            except Exception as e:
                print(f"Error processing neuron {j}: {str(e)}")
                import traceback
                traceback.print_exc()  # Add better error tracking
                continue

        if vol_params.verbose >= 1:
            print('done.')

        return neur_num, cell_volume_ad, dend_params, gp_soma

    def _get_local_volume(
        self,
        cell_volume: np.ndarray,
        root_pos: np.ndarray,
        f_dims: np.ndarray,
        dims: np.ndarray,
        small_z: bool = True
    ) -> np.ndarray:
        """
        Extract local volume around a neuron for dendrite growth.
        
        Args:
            cell_volume: Full cell volume array
            root_pos: Root position [x,y,z]
            f_dims: Fine dimensions [x,y,z]
            dims: Coarse dimensions [x,y,z]
            small_z: Whether to use reduced z-dimension sampling
            
        Returns:
            Local volume array around root position
        """
        try:
            # Convert dimensions to integers
            root_pos = np.asarray(root_pos, dtype=np.int32)
            f_dims = np.asarray(f_dims, dtype=np.int32)
            dims = np.asarray(dims, dtype=np.int32)
            
            # Calculate volume bounds
            x_start = max(0, root_pos[0] - f_dims[0]//2)
            x_end = min(cell_volume.shape[0], root_pos[0] + (f_dims[0]+1)//2)
            y_start = max(0, root_pos[1] - f_dims[1]//2)
            y_end = min(cell_volume.shape[1], root_pos[1] + (f_dims[1]+1)//2)
            
            if small_z:
                # Return full z-dimension
                return cell_volume[x_start:x_end, y_start:y_end, :]
            else:
                # Calculate z bounds
                z_start = max(0, root_pos[2] - f_dims[2]//2)
                z_end = min(cell_volume.shape[2], root_pos[2] + (f_dims[2]+1)//2)
                return cell_volume[x_start:x_end, y_start:y_end, z_start:z_end]
                
        except Exception as e:
            print(f"Error in _get_local_volume: {str(e)}")
            # Return empty volume of correct size on error
            out_shape = f_dims if not small_z else [f_dims[0], f_dims[1], cell_volume.shape[2]]
            return np.zeros(out_shape, dtype=cell_volume.dtype)

    def _calculate_path_weights(self, path: np.ndarray, dend_sz: float) -> np.ndarray:
        """Helper function to calculate path weights based on path twists."""
        if len(path) > 2:
            diffs = np.abs(np.diff(np.abs(np.diff(path, axis=0)), axis=0))
            return dend_sz * (1 - (1 - 1/np.sqrt(2)) * np.concatenate([
                [0], np.sum(diffs, axis=1)/2, [0]
            ]))
        return dend_sz * np.ones(len(path))


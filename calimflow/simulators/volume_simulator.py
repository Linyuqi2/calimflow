from typing import Tuple, Dict, Optional
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

        # 6. Prepare output
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
from typing import Tuple, Dict, Optional
import numpy as np

from ..models.parameters import (
    VolumeParams, NeuronParams, VascParams, 
    DendriteParams, NeuralVolume
)
from .vasculature_simulator import VasculatureSimulator
from .neuron_simulator import NeuronSimulator

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
    
    def _clear_soma_vessels(
        self,
        neur_ves: np.ndarray,
        neur_soma: np.ndarray
    ) -> np.ndarray:
        """Clear vessel locations where somas exist."""
        depth_offset = self.vol_params.vol_depth * self.vol_params.vres
        vol_size = np.prod(self.vol_params.vol_sz[:2] * self.vol_params.vres)
        soma_indices = np.where(neur_soma > 0)[0] + depth_offset * vol_size
        neur_ves.ravel()[soma_indices] = 0
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
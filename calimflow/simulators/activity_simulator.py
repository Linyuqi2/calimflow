import numpy as np
from typing import Tuple, Dict, Optional
from scipy.signal import resample
from ..models.parameters import ActivityParams, CalciumParams

class ActivitySimulator:
    def __init__(self, params: ActivityParams):
        self.params = params
        self.l_buff = 500  # Buffer length for steady-state
        
    def generate_time_traces(
        self, 
        num_neurons: int, 
        locations: np.ndarray = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Generate synthetic neural activity traces."""
        dt = self.params.dt
        nt = self.params.n_timepoints
        ca_decay = self.params.ca_decay
        noise_std = self.params.noise_std
        spike_rate = self.params.spike_rate

        # Generate spike trains
        spikes = np.random.rand(num_neurons, nt) < spike_rate * dt

        # Generate calcium traces
        traces = {'soma': np.zeros((num_neurons, nt)), 'dend': np.zeros((num_neurons, nt))}
        for i in range(num_neurons):
            for t in range(1, nt):
                traces['soma'][i, t] = traces['soma'][i, t-1] * np.exp(-dt / ca_decay) + spikes[i, t]
                traces['dend'][i, t] = traces['dend'][i, t-1] * np.exp(-dt / ca_decay) + spikes[i, t]

        # Add noise
        traces['soma'] += noise_std * np.random.randn(num_neurons, nt)
        traces['dend'] += noise_std * np.random.randn(num_neurons, nt)

        return traces, spikes
    
    def _generate_hawkes_spikes(
        self, 
        num_neurons: int, 
        nt: int, 
        rate: float,
        locations: np.ndarray = None
    ) -> np.ndarray:
        """Generate correlated spike trains using Hawkes process with stability."""
        spikes = np.zeros((num_neurons, nt), dtype=np.float64)
        dt = 1/100  # 100Hz sampling
        
        # Scale parameters for stability
        base_rate = rate * dt  # Convert to rate per timestep
        alpha = self.params.alpha * dt
        beta = self.params.beta
        
        # Generate correlated spikes
        for t in range(1, nt):
            # Calculate history effect with exponential decay
            history = np.zeros(num_neurons)
            for s in range(max(0, t-int(5/beta/dt)), t):
                history += alpha * np.exp(-beta * (t-s) * dt) * spikes[:, s]
            
            # Calculate total rate with bounds
            total_rate = np.clip(base_rate + history, 0, 1)
            
            # Generate spikes
            spikes[:, t] = np.random.poisson(total_rate)
        
        return spikes
    
    def _generate_compartment_activity(
        self, 
        spikes: np.ndarray, 
        compartment: str
    ) -> np.ndarray:
        """Generate calcium dynamics for different cellular compartments."""
        cal_params = CalciumParams()
        
        # Adjust parameters based on compartment
        if compartment == 'dendrite':
            cal_params.ext_rate /= 2
        elif compartment == 'background':
            cal_params.ext_rate /= 4
        
        # Simulate calcium dynamics
        fluorescence = self._calcium_dynamics(spikes, cal_params)
        
        # Remove buffer period
        return fluorescence[:, self.l_buff:]
    
    def _calcium_dynamics(
        self, 
        spikes: np.ndarray, 
        params: CalciumParams
    ) -> np.ndarray:
        """Simulate calcium dynamics and fluorescence with numerical stability."""
        num_neurons, nt = spikes.shape
        ca = np.zeros_like(spikes, dtype=np.float64)  # Use float64 for better precision
        bound = np.zeros_like(spikes, dtype=np.float64)
        
        # Scale parameters for numerical stability
        k_on = params.k_on * 1e-3  # Scale binding rate
        k_off = params.k_off * 1e-3  # Scale unbinding rate
        ext_rate = params.ext_rate * 1e-3  # Scale extrusion rate
        
        # Add baseline calcium
        ca[:, 0] = params.ca_rest
        bound[:, 0] = k_on * ca[:, 0] / (k_on * ca[:, 0] + k_off)
        
        # Simulate dynamics with stability checks
        for t in range(1, nt):
            # Calcium influx and decay with bounds
            dca = (spikes[:, t] - ext_rate * (ca[:, t-1] - params.ca_rest)) * params.dt
            ca[:, t] = np.clip(ca[:, t-1] + dca, params.ca_rest, 1e-3)
            
            # Protein binding with saturation
            bound_inf = k_on * ca[:, t] / (k_on * ca[:, t] + k_off)
            tau_bound = 1 / (k_on * ca[:, t] + k_off)
            dbound = (bound_inf - bound[:, t-1]) * params.dt / tau_bound
            bound[:, t] = np.clip(bound[:, t-1] + dbound, 0, 1)
        
        return bound
    
    def _post_process_traces(self, traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Post-process fluorescence traces."""
        if self.params.dt != 1/100:
            # Resample to desired frame rate
            r1 = 100
            r2 = int(1/self.params.dt)
            
            for comp in traces:
                if traces[comp] is not None:
                    # Add buffer
                    buff = np.repeat(traces[comp][:, :1], 100, axis=1)
                    traces[comp] = np.concatenate([buff, traces[comp]], axis=1)
                    
                    # Resample
                    traces[comp] = resample(
                        traces[comp], 
                        int(traces[comp].shape[1] * r2/r1), 
                        axis=1
                    )
        
        # Apply expression variation
        mod_vals = self._expression_variation(traces['soma'].shape[0])
        for comp in traces:
            if traces[comp] is not None:
                traces[comp] = traces[comp] * mod_vals[:, None]
        
        return traces
    
    def _expression_variation(self, num_neurons: int) -> np.ndarray:
        """Generate expression variation factors."""
        p_off = getattr(self.params, 'p_off', 0.1)
        min_mod = getattr(self.params, 'min_mod', 0.2)
        
        # Generate modulation values
        mod_vals = np.random.gamma(2, 0.5, num_neurons)
        mod_vals = np.maximum(mod_vals, min_mod)
        
        # Set some neurons to be "off"
        off_neurons = np.random.rand(num_neurons) < p_off
        mod_vals[off_neurons] = 0
        
        return mod_vals

    def _bin_spike_trains(
        self,
        evt: np.ndarray, 
        evm: np.ndarray, 
        n_nodes: int, 
        dt: float, 
        t_total: int
    ) -> np.ndarray:
        """Convert continuous spike events to binned format."""
        if evt.size != evm.size:
            raise ValueError("Number of events must match number of marks")
        if np.max(evm) > n_nodes:
            raise ValueError("Total nodes must exceed highest mark")
        if np.ceil(np.max(evt)/dt) > t_total:
            raise ValueError("Time span must exceed latest event")
            
        S = np.zeros((n_nodes, t_total))
        bin_indices = np.ceil(evt/dt).astype(int) - 1
        for mark, bin_idx in zip(evm, bin_indices):
            S[mark-1, bin_idx] += 1
        return S

    def _mk_double_exp_ker(
        self,
        t_on: float,
        t_off: float,
        amp: float,
        dt: float
    ) -> np.ndarray:
        """Generate double exponential kernel."""
        t = np.arange(0, 5*t_off, dt)
        h = amp * (1 - np.exp(-t/t_on)) * np.exp(-t/t_off)
        return h / np.sum(h)

    def _sat_nonlin(self, CB: np.ndarray, prot_type: str) -> np.ndarray:
        """Apply nonlinear saturation to calcium-bound fluorophore."""
        prot_type = prot_type.lower()
        
        if 'gcamp6' in prot_type:
            F0 = 1
            F = 25.2 * (1./(1 + (290e-9/CB)**2.7))
        elif 'gcamp3' in prot_type:
            F0 = 2
            F = 12 * (1./(1 + (287e-9/CB)**2.52))
        elif 'ogb1' in prot_type:
            F0 = 1
            F = 14 * (1./(1 + 250e-9/CB))
        else:
            F0 = 1
            F = 25.2 * (1./(1 + (290e-9/CB)**2.7))
            
        return F0 + F0*F
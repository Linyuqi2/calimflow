import numpy as np
import matplotlib.pyplot as plt
import time

from calimflow.models.parameters import VolumeParams, NeuronParams, VascParams, DendriteParams
from calimflow.simulators import VolumeSimulator, NeuronSimulator, VasculatureSimulator

def test_vasculature_simulator():
    """Test VasculatureSimulator with visualization."""
    print("Testing VasculatureSimulator...")
    start_time = time.time()
    
    # Initialize parameters
    vol_params = VolumeParams(
        size=np.array([50, 50, 30]),
        res=2,
        depth=100,
        verbose=1
    )
    vasc_params = VascParams(
        ves_freq=np.array([20, 10, 5]),
        source_freq=100,
        ves_size=np.array([6, 3, 1])
    )
    
    # Create simulator and run simulation
    simulator = VasculatureSimulator(vol_params, vasc_params)
    neur_ves, neur_ves_all = simulator.simulate()
    
    # Plot using the new method
    simulator.plot_vasculature(save_path="vasculature.png", show=True)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds.")

def main():
    """Run all tests."""
    test_vasculature_simulator()

if __name__ == "__main__":
    main() 
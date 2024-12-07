import time

import numpy as np

from calimflow.models.parameters import VascParams, VolumeParams
from calimflow.simulators import VasculatureSimulator


def test_vasculature_simulator():
    """Test VasculatureSimulator with visualization."""
    print("Testing VasculatureSimulator...")
    start_time = time.time()

    # Initialize parameters
    vol_params = VolumeParams(size=np.array([50, 50, 30]), res=2, depth=100, verbose=1)
    vasc_params = VascParams(
        ves_freq=np.array([20, 10, 5]), source_freq=100, ves_size=np.array([6, 3, 1])
    )

    # Create vas_simulator and run simulation
    vas_simulator = VasculatureSimulator(vol_params, vasc_params)
    neur_ves, neur_ves_all = vas_simulator.simulate()

    # Plot using the new method
    vas_simulator.plot_vasculature(save_path="../visualization/images/vasculature.png")
    vas_simulator.save_vessel_network(save_path="../visualization/data/vasculature.json")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds.")


def main():
    """Run all tests."""
    test_vasculature_simulator()


if __name__ == "__main__":
    main()

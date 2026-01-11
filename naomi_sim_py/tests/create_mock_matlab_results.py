"""
Create mock MATLAB results for neural volume testing.

This script creates a mock MATLAB result file that mimics the structure
of real MATLAB simulate_neural_volume output for testing purposes.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

def create_mock_matlab_results():
    """Create mock MATLAB results for testing"""

    # Volume parameters
    vol_sz = [100, 100, 40]
    vres = 2

    # Create mock neur_vol (fluorescence volume)
    neur_vol_shape = [s * vres for s in vol_sz]  # [200, 200, 80]
    neur_vol = np.zeros(neur_vol_shape, dtype=np.float32)

    # Add some mock neurons with fluorescence
    # Use same seed as Python simulation for consistency
    np.random.seed(0)

    # Mock neuron locations - use same logic as Python to match neuron count
    min_dist = 12  # From test parameters
    requested_neurons = 8

    neuron_centers = []
    for i in range(requested_neurons):
        # Simple placement - in real MATLAB this would be more complex
        # But for mock data, we'll place them systematically
        x = 15 + (i % 3) * 25  # Spread across X
        y = 15 + (i // 3) * 25  # Spread across Y
        z = 10 + (i % 2) * 15   # Alternate Z levels

        # Ensure within bounds
        x = min(max(x, 10), vol_sz[0]-10)
        y = min(max(y, 10), vol_sz[1]-10)
        z = min(max(z, 6), vol_sz[2]-6)

        center = np.array([x, y, z])
        neuron_centers.append(center)

        # Create Gaussian fluorescence blob for each neuron
        center_pixels = (center * vres).astype(int)

        # Create 3D Gaussian
        x_coords, y_coords, z_coords = np.meshgrid(
            np.arange(max(0, center_pixels[0]-20), min(neur_vol_shape[0], center_pixels[0]+20)),
            np.arange(max(0, center_pixels[1]-20), min(neur_vol_shape[1], center_pixels[1]+20)),
            np.arange(max(0, center_pixels[2]-10), min(neur_vol_shape[2], center_pixels[2]+10)),
            indexing='ij'
        )

        # Gaussian with radius ~10 pixels (5um)
        sigma = 10
        gauss = np.exp(-((x_coords - center_pixels[0])**2 +
                        (y_coords - center_pixels[1])**2 +
                        (z_coords - center_pixels[2])**2) / (2 * sigma**2))

        # Add to volume with some baseline fluorescence
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()
        neur_vol[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] += gauss * 0.5

    n_neurons = len(neuron_centers)  # Will be 8

    # Create mock neur_ves (vessel volume)
    neur_ves = np.zeros(neur_vol_shape, dtype=bool)

    # Add some mock vessels
    # Horizontal vessel at the top
    neur_ves[50:70, :, 10:15] = True
    # Vertical vessel on the side
    neur_ves[10:15, 50:150, :] = True

    # Convert to MATLAB-style arrays (correct shape for MATLAB)
    # MATLAB uses [height, width, depth] = [Y, X, Z] indexing
    neur_vol_matlab = np.transpose(neur_vol, (1, 0, 2))  # [Y, X, Z] -> [200, 200, 100]
    neur_ves_matlab = np.transpose(neur_ves, (1, 0, 2))   # [Y, X, Z] -> [200, 200, 100]
    locs_matlab = np.array(neuron_centers)

    # Create vol_out struct (MATLAB-style)
    # MATLAB uses structs with fields, not nested cell arrays for this case
    vol_out = {
        'neur_vol': neur_vol_matlab,  # Direct array, not cell
        'neur_ves': neur_ves_matlab,
        'locs': locs_matlab,
        'gp_nuc': np.array([]),  # Empty for simplicity
        'gp_soma': np.array([]),  # Empty for simplicity
        'gp_vals': np.array([]),  # Empty for simplicity
        'bg_proc': np.array([]),  # Empty for simplicity
        'neur_ves_all': neur_ves_matlab
    }

    # Save as .mat file
    output_file = Path(__file__).parent / "tpm_matlab_neural_result.mat"
    sio.savemat(str(output_file), {
        'vol_out': vol_out,
        'matlab_time': np.array([5.2])  # Mock computation time
    })

    print(f"Created mock MATLAB results: {output_file}")
    print(f"  Volume shape: {neur_vol_matlab.shape}")
    print(f"  Total fluorescence: {np.sum(neur_vol_matlab):.2f}")
    print(f"  Neuron locations: {len(neuron_centers)}")
    print(f"  Vessel voxels: {np.sum(neur_ves_matlab)}")

    return output_file

if __name__ == "__main__":
    create_mock_matlab_results()

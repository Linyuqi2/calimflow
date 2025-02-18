import numpy as np
import matlab.engine
from pathlib import Path
import sys
import os

# Add the parent directory to system path for imports
sys.path.append(str(Path(__file__).parent.parent))

from calimflow.simulators.neuron_simulator import NeuronSimulator
from calimflow.models.parameters import VolumeParams, NeuronParams, DendriteParams

def create_test_data():
    """Create identical test data for both MATLAB and Python implementations."""
    # Create sample inputs
    vol_params = {
        'vol_sz': np.array([100, 100, 30]),
        'min_dist': 15,
        'N_neur': 50,
        'vres': 2,
        'N_den': 10,
        'N_bg': 50,
        'vol_depth': 100,
        'verbose': 0
    }
    
    axon_params = {
        'distsc': 0.5,
        'fillweight': 100,
        'maxlength': 200,
        'minlength': 10,
        'maxdist': 100,
        'maxel': 8,
        'numbranches': 20,
        'varbranches': 5,
        'maxfill': 0.7,
        'N_proc': 10,
        'l': 25,
        'rho': 0.1
    }
    
    # Create sample background values
    num_axons = 20
    gp_bgvals = []
    vol_size = vol_params['vol_sz'] * vol_params['vres']
    prod_vol = int(np.prod(vol_size))
    
    # Use fixed seed for reproducibility
    np.random.seed(42)
    
    for i in range(num_axons):
        num_points = np.random.randint(10, 50)
        indices = np.random.randint(0, prod_vol, num_points)
        values = np.random.rand(num_points)
        gp_bgvals.append((indices, values))
    
    # Create sample cell positions
    cell_pos = np.random.rand(vol_params['N_neur'], 3) * vol_params['vol_sz']
    
    return vol_params, axon_params, gp_bgvals, cell_pos

def run_matlab_sort_axons(eng, vol_params, axon_params, gp_bgvals, cell_pos):
    """Run the MATLAB implementation."""
    # Convert Python data to MATLAB format
    m_vol_params = eng.struct(
        'vol_sz', matlab.double(vol_params['vol_sz'].tolist()),
        'min_dist', matlab.double([vol_params['min_dist']]),
        'N_neur', matlab.double([vol_params['N_neur']]),
        'vres', matlab.double([vol_params['vres']]),
        'N_den', matlab.double([vol_params['N_den']]),
        'N_bg', matlab.double([vol_params['N_bg']]),
        'vol_depth', matlab.double([vol_params['vol_depth']]),
        'verbose', matlab.double([vol_params['verbose']])
    )
    
    m_axon_params = eng.struct(
        'distsc', matlab.double([axon_params['distsc']]),
        'fillweight', matlab.double([axon_params['fillweight']]),
        'maxlength', matlab.double([axon_params['maxlength']]),
        'minlength', matlab.double([axon_params['minlength']]),
        'maxdist', matlab.double([axon_params['maxdist']]),
        'maxel', matlab.double([axon_params['maxel']]),
        'numbranches', matlab.double([axon_params['numbranches']]),
        'varbranches', matlab.double([axon_params['varbranches']]),
        'maxfill', matlab.double([axon_params['maxfill']]),
        'N_proc', matlab.double([axon_params['N_proc']]),
        'l', matlab.double([axon_params['l']]),
        'rho', matlab.double([axon_params['rho']])
    )
    
    # Convert gp_bgvals to MATLAB cell array
    m_gp_bgvals = eng.cell(len(gp_bgvals), 2)
    for i, (indices, values) in enumerate(gp_bgvals):
        m_gp_bgvals[i][0] = matlab.double(indices.tolist())
        m_gp_bgvals[i][1] = matlab.double(values.tolist())
    
    m_cell_pos = matlab.double(cell_pos.tolist())
    
    # Run MATLAB function
    result = eng.sort_axons(m_vol_params, m_axon_params, m_gp_bgvals, m_cell_pos)
    
    # Convert MATLAB result to Python
    py_result = []
    for i in range(len(result)):
        indices = np.array(result[i][0]) if result[i][0] else np.array([])
        values = np.array(result[i][1]) if result[i][1] else np.array([])
        py_result.append((indices, values))
    
    return py_result

def run_python_sort_axons(vol_params, axon_params, gp_bgvals, cell_pos):
    """Run the Python implementation."""
    # Convert to appropriate parameter objects
    py_vol_params = VolumeParams(
        size=vol_params['vol_sz'],
        min_dist=vol_params['min_dist'],
        n_neurons=vol_params['N_neur'],
        resolution=vol_params['vres'],
        n_dendrites=vol_params['N_den'],
        n_background=vol_params['N_bg'],
        depth=vol_params['vol_depth'],
        verbose=vol_params['verbose']
    )
    
    simulator = NeuronSimulator(
        vol_params=py_vol_params,
        neur_params=NeuronParams(),
        dend_params=DendriteParams(),
        axon_params=axon_params
    )
    
    result = simulator._sort_axons(gp_bgvals, cell_pos)
    return result

def compare_results(matlab_result, python_result):
    """Compare the results from both implementations."""
    if len(matlab_result) != len(python_result):
        print(f"Different number of processes: MATLAB={len(matlab_result)}, Python={len(python_result)}")
        return False
    
    all_match = True
    for i, (m_proc, p_proc) in enumerate(zip(matlab_result, python_result)):
        m_indices, m_values = m_proc
        p_indices, p_values = p_proc
        
        # Compare indices
        if not np.array_equal(m_indices, p_indices):
            print(f"Process {i}: Different indices")
            print(f"MATLAB indices: {m_indices}")
            print(f"Python indices: {p_indices}")
            all_match = False
        
        # Compare values
        if not np.allclose(m_values, p_values, rtol=1e-5, atol=1e-8):
            print(f"Process {i}: Different values")
            print(f"MATLAB values: {m_values}")
            print(f"Python values: {p_values}")
            all_match = False
    
    return all_match

def main():
    """Main test function."""
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    # Create test data
    vol_params, axon_params, gp_bgvals, cell_pos = create_test_data()
    
    # Run both implementations
    matlab_result = run_matlab_sort_axons(eng, vol_params, axon_params, gp_bgvals, cell_pos)
    python_result = run_python_sort_axons(vol_params, axon_params, gp_bgvals, cell_pos)
    
    # Compare results
    success = compare_results(matlab_result, python_result)
    
    # Stop MATLAB engine
    eng.quit()
    
    if success:
        print("Test passed: MATLAB and Python implementations match!")
    else:
        print("Test failed: Implementations produce different results")

if __name__ == "__main__":
    main()

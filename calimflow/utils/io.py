import os
import numpy as np
import json
from typing import Dict, Tuple, Optional, List, Any
import pickle


def save_neural_volume(
    filepath: str,
    neur_soma: np.ndarray,
    neur_vol: np.ndarray,
    neur_locs: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save neural volume data to a file.
    
    Args:
        filepath: Path to save the file
        neur_soma: Neural soma volume (uint16 array with neuron IDs)
        neur_vol: Neural volume with fluorescence values
        neur_locs: Neuron locations (optional)
        metadata: Additional metadata to save (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Default metadata
    if metadata is None:
        metadata = {}
    
    # Prepare data dictionary
    data = {
        'neur_soma': neur_soma,
        'neur_vol': neur_vol,
        'metadata': metadata
    }
    
    if neur_locs is not None:
        data['neur_locs'] = neur_locs
    
    # Save as NPZ file (compressed numpy)
    np.savez_compressed(filepath, **data)
    print(f"Neural volume saved to {filepath}")


def load_neural_volume(filepath: str) -> Dict:
    """
    Load neural volume data from a file.
    
    Args:
        filepath: Path to the saved file
        
    Returns:
        Dictionary containing neural volume data
    """
    # Load NPZ file
    data = np.load(filepath, allow_pickle=True)
    
    # Extract data into dictionary
    result = {
        'neur_soma': data['neur_soma'],
        'neur_vol': data['neur_vol']
    }
    
    # Add optional components if they exist
    if 'neur_locs' in data:
        result['neur_locs'] = data['neur_locs']
    
    if 'metadata' in data:
        result['metadata'] = data['metadata'].item() if data['metadata'].dtype == np.dtype('O') else data['metadata']
    
    return result


def neural_volume_to_json(
    neur_soma: np.ndarray, 
    neur_vol: np.ndarray,
    neur_locs: Optional[np.ndarray] = None,
    intensity_threshold: float = 0.05,
    max_points_per_neuron: int = 10000
) -> str:
    """
    Convert neural volume data to JSON format suitable for 3D visualization.
    
    Args:
        neur_soma: Neural soma volume (uint16 array with neuron IDs)
        neur_vol: Neural volume with fluorescence values
        neur_locs: Neuron locations (optional)
        intensity_threshold: Minimum intensity value to include
        max_points_per_neuron: Maximum number of points per neuron to include
        
    Returns:
        JSON string containing neural volume data
    """
    # Identify unique neurons
    unique_neurons = np.unique(neur_soma)
    unique_neurons = unique_neurons[unique_neurons > 0]  # Remove background
    
    # Prepare neural data structure
    neural_data = {
        "metadata": {
            "version": 0.1,
            "type": "NeuralVolume",
            "generator": "CalimFlow"
        },
        "neurons": []
    }
    
    # Add neuron locations if available
    if neur_locs is not None:
        neural_data["neuron_locations"] = neur_locs.tolist()
    
    # Process each neuron
    for i, neuron_id in enumerate(unique_neurons):
        # Extract soma voxels
        x_soma, y_soma, z_soma = np.where(neur_soma == neuron_id)
        
        # Downsample if necessary
        if len(x_soma) > max_points_per_neuron:
            sample_rate = len(x_soma) // max_points_per_neuron
            x_soma = x_soma[::sample_rate]
            y_soma = y_soma[::sample_rate]
            z_soma = z_soma[::sample_rate]
        
        # Get soma points
        soma_points = np.column_stack([x_soma, y_soma, z_soma]).tolist()
        
        # Find neurite points (fluorescence > threshold but not in soma)
        mask = (neur_vol > intensity_threshold) & (neur_soma != neuron_id)
        
        # If neuron has location, limit search to nearby area
        neurite_points = []
        if neur_locs is not None and i < len(neur_locs):
            # Get neuronal processes near this neuron
            center = neur_locs[i]
            radius = 30  # Reasonable radius to search for processes
            
            # Create process mask within radius of neuron center
            x_grid, y_grid, z_grid = np.meshgrid(
                np.arange(neur_soma.shape[0]), 
                np.arange(neur_soma.shape[1]), 
                np.arange(neur_soma.shape[2]),
                indexing='ij'
            )
            
            distance_mask = (
                ((x_grid - center[0]*2)**2 + 
                 (y_grid - center[1]*2)**2 + 
                 (z_grid - center[2]*2)**2) < radius**2
            )
            
            process_mask = mask & distance_mask
            x_proc, y_proc, z_proc = np.where(process_mask)
            
            if len(x_proc) > max_points_per_neuron:
                sample_rate = len(x_proc) // max_points_per_neuron
                x_proc = x_proc[::sample_rate]
                y_proc = y_proc[::sample_rate]
                z_proc = z_proc[::sample_rate]
            
            neurite_points = np.column_stack([x_proc, y_proc, z_proc]).tolist()
        
        # Create neuron object
        neuron = {
            "id": int(neuron_id),
            "soma": soma_points,
            "neurites": neurite_points
        }
        
        neural_data["neurons"].append(neuron)
    
    return json.dumps(neural_data)


def save_neural_network(
    filepath: str, 
    neur_soma: np.ndarray, 
    neur_vol: np.ndarray,
    neur_locs: Optional[np.ndarray] = None,
    intensity_threshold: float = 0.05
) -> None:
    """
    Save neural network data to a JSON file for 3D visualization.
    
    Args:
        filepath: Path to save the JSON file
        neur_soma: Neural soma volume (uint16 array)
        neur_vol: Neural volume with fluorescence values
        neur_locs: Neuron locations (optional)
        intensity_threshold: Minimum intensity to include
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    json_data = neural_volume_to_json(
        neur_soma, neur_vol, neur_locs, intensity_threshold
    )
    
    with open(filepath, "w") as f:
        f.write(json_data)
    
    print(f"Neural network data saved to {filepath}")

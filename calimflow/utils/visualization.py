import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy import ndimage
import os

def plot_3d_volume(neural_soma, neural_volume=None, save_path=None, show=True, 
                  threshold=0.1, alpha_scale=0.8, elev=20, azim=45, figsize=(12, 10)):
    """
    Visualize a 3D neural volume with somas and optional neural processes.
    
    Args:
        neural_soma: 3D array with soma indices (uint16 or similar)
        neural_volume: 3D array with fluorescence values (optional)
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        threshold: Minimum value to include in visualization (filters noise).
        alpha_scale: Base opacity for the visualization.
        elev: Elevation viewing angle.
        azim: Azimuth viewing angle.
        figsize: Figure size (width, height) in inches.
        
    Returns:
        fig: The matplotlib figure object
    """
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get dimensions
    x_dim, y_dim, z_dim = neural_soma.shape
    
    # Create a colormap for neuron indices
    unique_neurons = np.unique(neural_soma)
    unique_neurons = unique_neurons[unique_neurons > 0]  # Remove zero (background)
    num_neurons = len(unique_neurons)
    
    if num_neurons > 0:
        print(f"Visualizing {num_neurons} neurons")
        
        # Create colormap
        colors = cm.rainbow(np.linspace(0, 1, num_neurons))
        
        # Plot each neuron soma as a distinct color
        for i, neuron_idx in enumerate(unique_neurons):
            # Find the positions of this neuron's soma
            x, y, z = np.where(neural_soma == neuron_idx)
            
            # Downsample points to make visualization faster if needed
            if len(x) > 5000:
                sample_rate = len(x) // 5000
                x, y, z = x[::sample_rate], y[::sample_rate], z[::sample_rate]
            
            # Plot neuron soma as a scatter plot with a distinct color
            ax.scatter(x, y, z, c=[colors[i]], s=5, alpha=alpha_scale, 
                      label=f"Neuron {neuron_idx}")
    else:
        print("No neurons found in soma volume")
    
    # If neural volume is provided, visualize processes
    if neural_volume is not None and neural_volume.max() > 0:
        # Extract process locations (exclude soma locations)
        processes = neural_volume.copy()
        processes[neural_soma > 0] = 0  # Zero out somas to isolate processes
        
        # Only show processes above threshold
        x_proc, y_proc, z_proc = np.where(processes > threshold)
        
        # Downsample points to make visualization faster if needed
        if len(x_proc) > 10000:
            sample_rate = len(x_proc) // 10000
            x_proc, y_proc, z_proc = x_proc[::sample_rate], y_proc[::sample_rate], z_proc[::sample_rate]
        
        if len(x_proc) > 0:
            # Plot processes in a different color (e.g., gray)
            process_colors = processes[x_proc, y_proc, z_proc]
            normalized_colors = process_colors / processes.max() if processes.max() > 0 else process_colors
            ax.scatter(x_proc, y_proc, z_proc, c='gray', s=2, alpha=alpha_scale*0.5)
            print(f"Added {len(x_proc)} process points")
        else:
            print("No neural processes found above threshold")
    
    # Enhance plot appearance
    ax.set_title("Neural Volume Visualization", fontsize=14, pad=20)
    ax.set_xlabel("X axis", fontsize=12)
    ax.set_ylabel("Y axis", fontsize=12)
    ax.set_zlabel("Z axis", fontsize=12)
    
    # Set axis limits
    ax.set_xlim(0, x_dim)
    ax.set_ylim(0, y_dim)
    ax.set_zlim(0, z_dim)
    
    # Set equal aspect ratio for more accurate visualization
    max_dim = max(x_dim, y_dim, z_dim)
    ax.set_box_aspect([x_dim/max_dim, y_dim/max_dim, z_dim/max_dim])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if there are multiple neurons
    if num_neurons > 1:
        ax.legend(loc='upper right', fontsize=10)
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_neural_network(
    neural_soma, neural_volume=None,
    neuron_locs=None, show_somas=True, show_processes=True,
    save_path=None, show=True, threshold=0.05,
    dpi=300, figsize=(15, 15), elev=20, azim=45
):
    """
    Create a high-quality visualization of neural network.
    
    Args:
        neural_soma: 3D array with soma indices (uint16 or similar)
        neural_volume: 3D array with fluorescence values (optional)
        neuron_locs: Array of neuron center locations (optional)
        show_somas: Whether to display somas
        show_processes: Whether to display neural processes
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        threshold: Minimum value to include in visualization (filters noise).
        dpi: DPI for the output figure
        figsize: Figure size in inches (width, height)
        elev: Elevation viewing angle
        azim: Azimuth viewing angle
        
    Returns:
        fig: The matplotlib figure object
    """
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get dimensions
    x_dim, y_dim, z_dim = neural_soma.shape
    
    # Identify unique neurons
    unique_neurons = np.unique(neural_soma)
    unique_neurons = unique_neurons[unique_neurons > 0]  # Remove zero (background)
    num_neurons = len(unique_neurons)
    
    if num_neurons > 0:
        print(f"Visualizing {num_neurons} neurons")
        
        # Create colormap for neurons
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_neurons)))
        if num_neurons > 10:
            colors = np.vstack([colors, plt.cm.Paired(np.linspace(0, 1, num_neurons-10))])
        
        # Keep track of legend handles
        handles = []
        labels = []
        
        if show_somas:
            # Plot each neuron soma as a distinct color
            for i, neuron_idx in enumerate(unique_neurons):
                # Find the positions of this neuron's soma
                x, y, z = np.where(neural_soma == neuron_idx)
                
                # Downsample points to make visualization faster if needed
                if len(x) > 5000:
                    sample_rate = len(x) // 5000
                    x, y, z = x[::sample_rate], y[::sample_rate], z[::sample_rate]
                
                # Plot neuron soma as a scatter plot with a distinct color
                scatter = ax.scatter(
                    x, y, z, 
                    c=[colors[i % len(colors)]], 
                    s=10, 
                    alpha=0.8,
                    label=f"Neuron {neuron_idx}"
                )
                
                # Add to legend (but limit to 10 to avoid clutter)
                if i < 10:
                    handles.append(scatter)
                    labels.append(f"Neuron {neuron_idx}")
    else:
        print("No neurons found in soma volume")
    
    # Add neuron centers if available
    if neuron_locs is not None:
        ax.scatter(
            neuron_locs[:, 0] * neural_soma.shape[0] / x_dim,
            neuron_locs[:, 1] * neural_soma.shape[1] / y_dim,
            neuron_locs[:, 2] * neural_soma.shape[2] / z_dim,
            c='yellow', s=80, alpha=1.0, marker='*'
        )
        
        # Add to legend
        handles.append(plt.Line2D([0], [0], linestyle="none", marker='*', 
                                  color='yellow', markersize=10))
        labels.append("Neuron Centers")
    
    # If neural volume is provided and processes should be shown, visualize processes
    if neural_volume is not None and show_processes:
        # Extract process locations (exclude soma locations)
        processes = neural_volume.copy()
        processes[neural_soma > 0] = 0  # Zero out somas to isolate processes
        
        # Only show processes above threshold
        x_proc, y_proc, z_proc = np.where(processes > threshold)
        
        # Downsample points to make visualization faster if needed
        if len(x_proc) > 10000:
            sample_rate = len(x_proc) // 10000
            x_proc, y_proc, z_proc = x_proc[::sample_rate], y_proc[::sample_rate], z_proc[::sample_rate]
        
        if len(x_proc) > 0:
            # Scale colors by intensity
            process_intensities = processes[x_proc, y_proc, z_proc]
            normalized_intensities = process_intensities / processes.max()
            
            # Plot processes with color based on intensity
            scatter_proc = ax.scatter(
                x_proc, y_proc, z_proc, 
                c='gray', 
                s=3, 
                alpha=0.3,
                marker='.'
            )
            
            # Add to legend
            handles.append(scatter_proc)
            labels.append("Neural Processes")
            
            print(f"Added {len(x_proc)} process points")
        else:
            print("No neural processes found above threshold")
    
    # Enhance plot appearance
    ax.set_title("Neural Network Visualization", fontsize=16, pad=20)
    ax.set_xlabel("X axis (voxels)", fontsize=14)
    ax.set_ylabel("Y axis (voxels)", fontsize=14)
    ax.set_zlabel("Z axis (voxels)", fontsize=14)
    
    # Set axis limits
    ax.set_xlim(0, x_dim)
    ax.set_ylim(0, y_dim)
    ax.set_zlim(0, z_dim)
    
    # Set equal aspect ratio
    max_dim = max(x_dim, y_dim, z_dim)
    ax.set_box_aspect([x_dim/max_dim, y_dim/max_dim, z_dim/max_dim])
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, loc='upper right', fontsize=10)
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_volume_slices(volume, title="Volume Slices", cmap='viridis', 
                      save_path=None, show=True, figsize=(15, 5)):
    """
    Plot orthogonal slices through the center of a 3D volume.
    
    Args:
        volume: 3D numpy array to visualize
        title: Title for the plot
        cmap: Colormap to use
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the plot.
        figsize: Figure size (width, height) in inches.
    
    Returns:
        fig: The matplotlib figure object
    """
    # Get the center slices
    x_center = volume.shape[0] // 2
    y_center = volume.shape[1] // 2
    z_center = volume.shape[2] // 2
    
    # Extract slices
    xy_slice = volume[:, :, z_center]
    xz_slice = volume[:, y_center, :]
    yz_slice = volume[x_center, :, :]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Plot slices
    im0 = axes[0].imshow(xy_slice.T, cmap=cmap, origin='lower')
    axes[0].set_title(f"XY Slice (Z={z_center})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    im1 = axes[1].imshow(xz_slice.T, cmap=cmap, origin='lower')
    axes[1].set_title(f"XZ Slice (Y={y_center})")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    
    im2 = axes[2].imshow(yz_slice.T, cmap=cmap, origin='lower')
    axes[2].set_title(f"YZ Slice (X={x_center})")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")
    
    # Add colorbars
    for i, im in enumerate([im0, im1, im2]):
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Slices saved to {save_path}")
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

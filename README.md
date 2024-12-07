# CalimFlow

CalimFlow is a Python simulator for calcium imaging dynamics, featuring vasculature network simulation and visualization capabilities.


## Installation

### Using conda (recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yaml
conda activate calimflow
```

The environment.yaml already includes the installation of the package in editable mode (`-e .`), so no additional installation steps are needed.


### Alternative: Using pip only


```bash
pip install -e .
```


## Usage

### Basic Vasculature Simulation

See `examples/example_vasculature_simulation.py` for a basic example of how to use the vasculature simulator.

### Viewing the Vasculature Network

1. Start a local server in the project directory:
```bash
python -m http.server
```

2. Open a web browser and navigate to:
```
http://localhost:8000/visualization/vessel_viewer.html
```

3. Use the file input to load your generated JSON file.



## Development

For development work, install additional dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## License

MIT License


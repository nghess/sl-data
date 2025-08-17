# sldata: Neural Spike Data Analysis Library

A Python library for loading and analyzing preprocessed neural spike data and behavioral event data.

## Features

- **SessionData Class**: Core class for loading and managing preprocessed neural session data
- **Behavioral Analysis**: Tools for processing behavioral event data and TTL signals
- **Population Analysis**: Create population raster plots and time-binned activity matrices
- **Event Alignment**: Align neural activity with behavioral events
- **Classifier Examples**: Demo implementations of MLP and LSTM neural classifiers

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/your-username/sl-data.git
cd sl-data

# Install in development mode
pip install -e .
```

### Dependencies

Core dependencies:
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0

Optional dependencies for examples:
- torch >= 1.9.0 (for neural classifier examples)
- scikit-learn >= 1.0.0 (for machine learning examples)

Install with optional dependencies:
```bash
# For classifier examples
pip install -e .[examples]

# For development
pip install -e .[dev]
```

## Quick Start

### Loading Preprocessed Neural Data

```python
from sldata import SessionData

# Load a preprocessed recording session
# (assumes spike_times.npy, spike_templates.npy, etc. already exist)
session = SessionData(
    mouse_id="7001",
    session_id="v1", 
    experiment="my_experiment",
    base_path="/path/to/preprocessed/data",
    min_spikes=50,  # Minimum spikes per cluster
    verbose=True
)

print(f"Loaded {session.n_clusters} clusters")
```

### Creating Population Activity Matrices

```python
# Create population raster with 100ms bins
pop_matrix, time_bins, cluster_ids = session.create_population_raster(
    start_time=0,           # Start at 0 ms
    end_time=30000,         # End at 30 seconds  
    bin_size_ms=100,        # 100 ms time bins
    zscore_neurons=True     # Z-score each neuron
)

print(f"Population matrix shape: {pop_matrix.shape}")  # [neurons x time_bins]
```

### Event-Based Analysis

```python
# Create event colormap for behavioral states
reward_colormap = session.create_event_colormap(
    time_bins, 
    event_column='reward_state',
    aggregation='any'  # True if any event in time bin is True
)

# Filter spikes during specific behavioral periods  
reward_session = session.filter_events('reward_state == 1')
print(f"Spikes during reward: {sum(c['n_spikes'] for c in reward_session.clusters.values())}")
```

### Filtering Clusters

```python
# Filter for olfactory bulb clusters (channels > 16)
ob_session = session.filter_clusters('best_channel > 16')

# Filter for highly active clusters
active_session = session.filter_clusters('n_spikes >= 200')
```

## Examples

### Running Classification Demos

```bash
# Run MLP classification demo
cd examples
python classification_demo.py

# Run LSTM sequence classification demo  
python lstm_demo.py
```

### Custom Analysis

```python
from sldata import SessionData
import matplotlib.pyplot as plt

# Load session
session = SessionData("mouse_001", "session_1", "experiment")

# Create population raster
pop_matrix, time_bins, clusters = session.create_population_raster(
    bin_size_ms=50, zscore_neurons=True
)

# Plot population activity
plt.figure(figsize=(12, 6))
plt.imshow(pop_matrix, aspect='auto', cmap='viridis')
plt.xlabel('Time Bins')
plt.ylabel('Neuron ID') 
plt.title('Population Activity')
plt.colorbar(label='Z-scored Firing Rate')
plt.show()
```

## API Reference

### SessionData Class

The main class for neural session analysis:

#### Key Methods:

- `create_population_raster()`: Create time-binned population activity matrix
- `create_event_colormap()`: Map behavioral events to time bins
- `filter_clusters()`: Filter clusters based on properties
- `filter_events()`: Filter spike times based on behavioral events
- `get_spike_times()`: Extract spike times for specific clusters
- `find_sniff_peaks()`: Detect peaks in sniff signal data

#### Key Properties:

- `clusters`: Dictionary of cluster information
- `events`: Behavioral event dataframe
- `n_clusters`: Number of loaded clusters
- `sniff`: Sniff signal data (if available)

### Utility Modules

- `behavior_utils`: Functions for TTL processing and behavioral data handling

## Testing

Run tests with pytest:

```bash
# Install test dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run with coverage
pytest --cov=sldata tests/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```
Gonzales-Hess, N. (2025). sldata: A Python library for preprocessed neural spike data analysis.
```

## Contact

- Author: Nate Gonzales-Hess
- Email: nhess [at] uoregon [dot] edu
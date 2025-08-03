# SessionData Class Documentation

**Author:** Nate Gonzales-Hess  
**Date:** August 2025  
**Language:** Python 3.7+

## Overview

The `SessionData` class is a standalone tool for loading, storing, and manipulating neural and behavioral data from a single recording session. It provides a unified interface to work with Kilosort spike sorting output, Bonsai behavioral tracking data, and associated signal recordings.

## Key Features

- **Flexible Path Finding**: Automatically locates data files using only experiment name and session identifiers
- **Unified Data Access**: Single object containing neural spikes, behavioral events, and signal traces
- **Population Analysis**: Built-in tools for creating time-binned population activity matrices
- **Cluster Filtering**: Dynamic filtering of neural clusters based on any cluster attribute
- **Sequential Indexing**: Clusters are re-indexed sequentially (0, 1, 2...) while preserving original IDs

## Dependencies

```python
numpy >= 1.18.0
pandas >= 1.0.0
scipy >= 1.4.0
pathlib (standard library)
```

## Installation

Simply copy `session_data.py` to your project directory and import:

```python
from session_data import SessionData
```

## Quick Start

```python
# Load a recording session
session = SessionData(
    mouse_id="7001", 
    session_id="v10", 
    experiment="clickbait-visual",
    base_path="S:/"
)

# Access cluster information
print(f"Loaded {session.n_clusters} clusters")
print(f"Original cluster IDs: {session.get_original_cluster_ids()}")

# Create population activity matrix
pop_matrix, time_bins, cluster_indices = session.create_population_raster(
    start_time=0,
    end_time=10000,  # First 10 seconds
    bin_size_ms=100,
    zscore_neurons=True
)

# Filter clusters by brain region
ca1_session = session.filter_clusters('best_channel <= 16')
ob_session = session.filter_clusters('best_channel > 16')
```

## Class Reference

### Constructor

```python
SessionData(mouse_id, session_id, experiment, base_path="S:\\", 
           sampling_rate=30000.0, min_spikes=50, verbose=True)
```

**Parameters:**
- `mouse_id` (str): Mouse identifier (e.g., '7001')
- `session_id` (str): Session identifier (e.g., 'v10')
- `experiment` (str): Experiment name for path construction
- `base_path` (str): Base directory path for data files
- `sampling_rate` (float): Neural data sampling rate in Hz
- `min_spikes` (int): Minimum spikes required to include a cluster
- `verbose` (bool): Whether to print loading information

### Attributes

#### Core Data
- `clusters` (dict): Neural cluster data with sequential indices (0, 1, 2...)
- `events` (DataFrame): Behavioral events from Bonsai
- `sniff` (array): Sniff signal trace (if available)
- `iti` (array): ITI signal trace (placeholder for future use)
- `reward` (array): Reward signal trace (placeholder for future use)

#### Metadata
- `mouse_id`, `session_id`, `experiment`: Session identifiers
- `n_clusters` (int): Number of loaded clusters
- `sampling_rate` (float): Data sampling rate
- `base_path` (str): Data directory path

#### Cluster Structure
Each cluster entry contains:
```python
{
    'cluster_id': int,           # Original Kilosort cluster ID
    'best_channel': int,         # Channel with max waveform amplitude
    'spike_times': array,        # Spike times in milliseconds
    'waveform_template': array,  # Mean waveform template
    'n_spikes': int             # Total number of spikes
}
```

### Methods

#### Cluster Access

**`get_cluster_ids() -> list`**
Returns sequential cluster indices (0, 1, 2...).

**`get_original_cluster_ids() -> list`**
Returns original Kilosort cluster IDs.

**`get_cluster_info(cluster_index) -> dict`**
Returns complete information for a specific cluster.

**`get_spike_times(cluster_index, start_time=0, end_time=inf) -> array`**
Returns spike times for a cluster within a time window.

#### Population Analysis

**`create_population_raster(start_time=0, end_time=inf, bin_size_ms=50, zscore_neurons=True, cluster_ids=None)`**

Creates time-binned population activity matrix.

**Parameters:**
- `start_time` (float): Start time in milliseconds
- `end_time` (float): End time in milliseconds (inf = full session)
- `bin_size_ms` (float): Time bin size in milliseconds
- `zscore_neurons` (bool): Whether to z-score each neuron's activity
- `cluster_ids` (list): Specific cluster indices to include

**Returns:**
- `population_matrix` (array): [neurons × time_bins] firing rate matrix
- `time_bins` (array): Time bin centers in milliseconds
- `included_clusters` (list): Cluster indices included in matrix

#### Signal Data Access

**`get_sniff_data(start_time_ms=0, end_time_ms=inf) -> array`**
Returns sniff signal within a time window.

**`get_sniff_times() -> array`**
Returns time vector for sniff data in milliseconds.

**`has_sniff_data() -> bool`**
Checks if sniff data is available.

#### Dynamic Filtering

**`filter_clusters(filter_expr) -> SessionData`**

Creates a new SessionData object containing only clusters meeting specified criteria.

**Supported operators:** `>`, `<`, `>=`, `<=`, `==`, `!=`

**Examples:**
```python
# Filter by brain region
ca1_clusters = session.filter_clusters('best_channel <= 16')
ob_clusters = session.filter_clusters('best_channel > 16')

# Filter by activity level
active_clusters = session.filter_clusters('n_spikes >= 200')
sparse_clusters = session.filter_clusters('n_spikes < 100')

# Filter by specific attributes
cluster_5 = session.filter_clusters('cluster_id == 5')
good_channels = session.filter_clusters('best_channel != None')
```

## Directory Structure

The class uses flexible path finding to locate data files. It searches within:
```
base_path/experiment/**/{target_filename}
```

And filters for files containing `mouse_id/session_id` in their path.

**Example directory structures supported:**
```
S:/clickbait-visual/kilosorted/7001/v10/spike_times.npy
S:/clickbait-visual/preprocessed/7001/v10/sniff.npy
S:/clickbait-visual/events/7001/v10/events.csv
S:/my-experiment/any-structure/7001/v10/data.npy
```

## Usage Examples

### Basic Loading and Exploration
```python
# Load session data
session = SessionData("7001", "v10", "clickbait-visual")

# Explore the data
print(f"Session: {session}")
print(f"Clusters loaded: {session.n_clusters}")
print(f"Has sniff data: {session.has_sniff_data()}")
print(f"Events shape: {session.events.shape}")

# Get cluster information
for i in range(min(3, session.n_clusters)):
    info = session.get_cluster_info(i)
    print(f"Cluster {i}: {info['n_spikes']} spikes on channel {info['best_channel']}")
```

### Population Activity Analysis
```python
# Create population raster for first 60 seconds
pop_matrix, time_bins, clusters = session.create_population_raster(
    start_time=0,
    end_time=60000,     # 60 seconds
    bin_size_ms=200,    # 200ms bins
    zscore_neurons=True
)

# Analyze population activity
mean_activity = np.mean(pop_matrix, axis=0)
peak_times = time_bins[np.where(mean_activity > 2)[0]]  # High activity periods

print(f"Population matrix shape: {pop_matrix.shape}")
print(f"Peak activity times: {peak_times[:5]} ms")
```

### Brain Region Analysis
```python
# Separate by brain region (assuming channels 0-16 are CA1, >16 are OB)
ca1_session = session.filter_clusters('best_channel <= 16')
ob_session = session.filter_clusters('best_channel > 16')

print(f"CA1 clusters: {ca1_session.n_clusters}")
print(f"OB clusters: {ob_session.n_clusters}")

# Compare population activity between regions
ca1_matrix, ca1_times, _ = ca1_session.create_population_raster(bin_size_ms=100)
ob_matrix, ob_times, _ = ob_session.create_population_raster(bin_size_ms=100)

ca1_mean = np.mean(ca1_matrix, axis=0)
ob_mean = np.mean(ob_matrix, axis=0)
```

### Signal Analysis
```python
# Work with sniff data
if session.has_sniff_data():
    # Get sniff signal for first 10 seconds
    sniff_segment = session.get_sniff_data(0, 10000)
    sniff_times = session.get_sniff_times()[:len(sniff_segment)]
    
    # Basic analysis
    sniff_rate = len(sniff_segment) / 10.0  # samples per second
    print(f"Sniff sampling rate: {sniff_rate} Hz")
```

### Spike Time Analysis
```python
# Analyze specific cluster
cluster_idx = 0
spike_times = session.get_spike_times(cluster_idx)

# Basic statistics
firing_rate = len(spike_times) / (spike_times[-1] - spike_times[0]) * 1000  # Hz
isi = np.diff(spike_times)  # Inter-spike intervals
mean_isi = np.mean(isi)

print(f"Cluster {cluster_idx}:")
print(f"  Firing rate: {firing_rate:.2f} Hz")
print(f"  Mean ISI: {mean_isi:.2f} ms")
print(f"  Total spikes: {len(spike_times)}")
```

## Error Handling

The class provides robust error handling:

- **Missing files**: Warns about missing data files but continues loading
- **Invalid clusters**: Raises clear errors for invalid cluster indices
- **Filter errors**: Provides detailed warnings for filter expression issues
- **Path errors**: Reports when expected directories don't exist

## Performance Notes

- **File Loading**: Uses efficient numpy loading with flexible path finding
- **Population Matrices**: Memory-efficient creation with optional z-scoring
- **Filtering**: Creates shallow copies where possible to minimize memory usage
- **Path Search**: Targeted searching within experiment directories for speed

## Migration Guide

When moving to a new project:

1. Copy `session_data.py` to your new project directory
2. Update import statements as needed
3. Adjust `base_path` parameter for your data location
4. Modify `experiment` parameter to match your directory structure

## Future Extensions

The class is designed for easy extension:

- Add new signal types by extending `_process_signals()`
- Add new cluster metrics by modifying cluster processing
- Extend filtering with custom operators
- Add new population analysis methods

## License

This code is provided as-is for research purposes. Please credit Nate Gonzales-Hess if used in publications.
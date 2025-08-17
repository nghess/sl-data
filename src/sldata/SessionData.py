"""
SessionData class for storing and manipulating preprocessed neural and behavioral data.

This module provides a standalone class to encapsulate preprocessed neural data from a single recording session,
including spike times, cluster information, and simple population activity binning tools.

Code by Nate Gonzales-Hess, August 2025.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List
from scipy.stats import zscore
from scipy import signal
from scipy.signal import find_peaks
import os
import operator
import copy


class SessionData:
    """
    A class to store and filter Kilosort and Bonsai data for a single recording session.
    
    Attributes:
    -----------
    mouse_id : str
        Mouse identifier
    session_id : str  
        Session identifier
    clusters : dict
        Dictionary with sequential indices (0, 1, 2, ...) as keys, each containing:
        - 'cluster_id': original Kilosort cluster ID
        - 'best_channel': channel with maximum waveform amplitude
        - 'spike_times': spike times in milliseconds
        - 'waveform_template': mean waveform template
        - 'n_spikes': number of spikes
    sampling_rate : float
        Neural data sampling rate in Hz
    n_clusters : int
        Number of clusters loaded (after filtering)
    sniff : np.ndarray
        Sniff signal data (if available)
    sniff_times : np.ndarray
        Peak times of sniff signal in milliseconds (if available)
    """
    
    def __init__(self, mouse_id: str, session_id: str, experiment: str, base_path: str = "S:\\", 
                 sampling_rate: float = 30000.0, min_spikes: int = 50,
                 verbose: bool = True):
        """
        Initialize SessionData object by loading preprocessed neural data.
        
        Parameters:
        -----------
        mouse_id : str
            Mouse identifier (e.g., '7001')
        session_id : str
            Session identifier (e.g., 'v1')
        base_path : str
            Base directory path for preprocessed data files
        sampling_rate : float
            Neural data sampling rate in Hz
        min_spikes : int
            Minimum number of spikes required to include a cluster
        verbose : bool
            Whether to print loading information
        """
        self.mouse_id = mouse_id
        self.session_id = session_id
        self.experiment = experiment
        self.base_path = base_path
        self.sampling_rate = sampling_rate
        self.min_spikes = min_spikes
        self.verbose = verbose
        
        # Initialize containers
        self.clusters = {}
        self.raw_data = {}
        self.n_clusters = 0
        self.events = pd.DataFrame()
        self.duration = np.int32()

        # Initialize signal traces
        self.sniff = np.array([])
        self.sniff_times = np.array([])
        self.iti = np.array([])
        self.reward = np.array([])

        # Load the data
        self._load_ephys_data()
        self._load_events_data()
        self._process_clusters()
        self._process_signals()
        self.find_sniff_peaks(prominence=1000, distance=50)
        
        if verbose:
            print(f"Loaded {self.n_clusters} clusters for {mouse_id}_{session_id}")
            if len(self.sniff) > 0:
                print(f"Loaded sniff data: {len(self.sniff)} samples")
    
    def _get_file_path(self, target_filename: str) -> Optional[Path]:
        """
        Find the path to a target file by searching flexibly within the experiment directory.
        
        Parameters:
        -----------
        target_filename : str
            Target filename to search for (e.g., 'sniff.npy', 'spike_times.npy')
            
        Returns:
        --------
        file_path : Path or None
            Path to the target file, or None if not found
        """
        # Search from experiment directory
        search_dir = Path(self.base_path) / self.experiment
        
        if not search_dir.exists():
            if self.verbose:
                print(f"Experiment directory does not exist: {search_dir}")
            return None
        
        # Search for the target file within the experiment directory
        search_pattern = f"**/{target_filename}"
        matching_files = list(search_dir.glob(search_pattern))
        
        # Filter for files that have mouse_id/session_id in their path
        # Use Path.parts to avoid OS-specific separator issues
        for file_path in matching_files:
            path_parts = file_path.parts
            # Check if both mouse_id and session_id appear in the path parts
            if self.mouse_id in path_parts and self.session_id in path_parts:
                return file_path
        
        if self.verbose:
            print(f"Warning: {target_filename} not found for {self.mouse_id}/{self.session_id} in {search_dir}")
        return None

    def _load_ephys_data(self):
        """Load preprocessed neural data files."""
        try:
            self.raw_data = self._load_kilosort_files(
                self.mouse_id, 
                self.session_id,
                self.experiment, 
                base_path=self.base_path,
                files_needed=['spike_times', 'spike_templates', 'templates', 'sniff']
            )
            
            if self.verbose:
                print(f"Loaded data: {list(self.raw_data.keys())}")
                
        except Exception as e:
            raise ValueError(f"Failed to load preprocessed neural data: {e}")
    
    def _load_kilosort_files(self, mouse_id: str, session_id: str, experiment: str, base_path: str = "S:\\", 
                            files_needed: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load preprocessed spike sorting data for a given mouse and session using flexible path finding.
        
        Parameters:
        -----------
        mouse_id : str
            Mouse identifier (e.g., '7001')
        session_id : str
            Session identifier (e.g., 'v1')
        experiment : str
            Experiment name
        base_path : str
            Base data directory path (default: 'S:\\')
        files_needed : list of str, optional
            Specific files to load. If None, loads all standard files.
            Options: ['spike_times', 'spike_templates', 'templates', 'amplitudes', 
                     'whitening_mat_inv', 'sniff', 'events']
            
        Returns:
        --------
        data : dict
            Dictionary containing loaded data arrays/dataframes
        """
        # Default files to load
        if files_needed is None:
            files_needed = ['spike_times', 'spike_templates', 'templates', 'amplitudes', 
                           'whitening_mat_inv', 'sniff', 'events']
        
        data = {}
        
        # Define filename mapping
        target_files = {
            'spike_times': 'spike_times.npy',
            'spike_templates': 'spike_templates.npy', 
            'templates': 'templates.npy',
            'amplitudes': 'amplitudes.npy',
            'whitening_mat_inv': 'whitening_mat_inv.npy',
            'sniff': 'sniff.npy',
            'events': 'events.csv'
        }
        
        # Load each requested file using flexible path finding
        for key in files_needed:
            if key in target_files:
                target_filename = target_files[key]
                file_path = self._get_file_path(target_filename)
                
                if file_path is not None:
                    try:
                        data[key] = np.load(file_path)
                        if self.verbose:
                            print(f"Successfully loaded {key} from {file_path}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading {key} from {file_path}: {e}")
                else:
                    if self.verbose:
                        print(f"Warning: {target_filename} not found")
        
        return data

    def _load_events_data(self) -> pd.DataFrame:
        """
        Load behavioral events data for a given mouse and session using flexible path finding.

        Returns:
        --------
        events : pd.DataFrame
            Events dataframe with behavioral data from Bonsai
        """
        events_path = self._get_file_path("events.csv")
        
        if events_path is None:
            if self.verbose:
                print("Warning: events.csv not found")
            return
        
        try:
            self.events = pd.read_csv(events_path)
            self.duration = (self.events['timestamp_ms'].iloc[-1] - self.events['timestamp_ms'].iloc[0])
            if self.verbose:
                print(f"Successfully loaded events from {events_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading events from {events_path}: {e}")
    
    def _get_cluster_spike_times(self, spike_times: np.ndarray, spike_templates: np.ndarray, 
                                cluster_id: int, position_times: Optional[np.ndarray] = None,
                                sampling_rate: float = 30000.0, 
                                time_filter: bool = True) -> np.ndarray:
        """
        Extract spike times for a specific cluster with optional time filtering.
        
        Parameters:
        -----------
        spike_times : np.ndarray
            All spike times from kilosort (in samples)
        spike_templates : np.ndarray
            Cluster assignments for each spike
        cluster_id : int
            Target cluster ID
        position_times : np.ndarray, optional
            Position timestamps for filtering (in ms)
        sampling_rate : float
            Sampling rate in Hz (default: 30000)
        time_filter : bool
            Whether to filter spikes to position recording period
            
        Returns:
        --------
        cluster_spike_times : np.ndarray
            Spike times for the cluster (in milliseconds)
        """
        # Get spike indices for this cluster
        cluster_spikes = np.where(spike_templates == cluster_id)[0]
        
        # Extract spike times and convert to milliseconds
        cluster_spike_times = spike_times[cluster_spikes] / (sampling_rate / 1000.0)
        
        # Filter to position recording period if requested
        if time_filter and position_times is not None:
            time_mask = ((cluster_spike_times >= position_times[0]) & 
                        (cluster_spike_times <= position_times[-1]))
            cluster_spike_times = cluster_spike_times[time_mask]
        
        return cluster_spike_times
    
    def _process_clusters(self):
        """Process clusters and extract relevant information."""
        if 'spike_times' not in self.raw_data or 'spike_templates' not in self.raw_data:
            raise ValueError("Missing required spike data")
        
        spike_times = self.raw_data['spike_times']
        spike_templates = self.raw_data['spike_templates']
        templates = self.raw_data.get('templates')
        
        # Get unique cluster IDs
        unique_clusters = np.unique(spike_templates)
        
        # Collect valid clusters first (those meeting min_spikes threshold)
        valid_clusters = []
        
        for cluster_id in unique_clusters:
            # Get spike times for this cluster using internal method
            cluster_spike_times = self._get_cluster_spike_times(
                spike_times, spike_templates, cluster_id, 
                sampling_rate=self.sampling_rate, time_filter=False
            )
            
            # Apply minimum spike threshold
            if len(cluster_spike_times) < self.min_spikes:
                continue
            
            # Get best channel and waveform template
            best_channel = None
            waveform_template = None
            
            if templates is not None and cluster_id < len(templates):
                template = templates[cluster_id]
                # Best channel is the one with maximum absolute amplitude
                best_channel = np.argmax(np.max(np.abs(template), axis=0))
                # Extract waveform for best channel
                waveform_template = template[:, best_channel]
            
            # Store cluster information with original cluster_id
            cluster_info = {
                'cluster_id': int(cluster_id),  # Original cluster ID from Kilosort
                'best_channel': best_channel,
                'spike_times': cluster_spike_times,
                'waveform_template': waveform_template,
                'n_spikes': len(cluster_spike_times)
            }
            
            valid_clusters.append(cluster_info)
        
        # Now store clusters with sequential indices (no gaps)
        self.clusters = {}
        for i, cluster_info in enumerate(valid_clusters):
            self.clusters[i] = cluster_info
        
        self.n_clusters = len(self.clusters)
    
    def _process_signals(self):
        """Process signal traces (sniff, etc.)."""
        # Load sniff data if available
        if 'sniff' in self.raw_data:
            self.sniff = self.raw_data['sniff']
        else:
            self.sniff = np.array([])
            if self.verbose:
                print("No sniff data found")
    
    def get_cluster_ids(self) -> list:
        """
        Get list of cluster indices (sequential 0-based indices).
        
        Returns:
        --------
        cluster_indices : list
            List of cluster indices (0, 1, 2, ...)
        """
        return list(self.clusters.keys())
    
    def get_original_cluster_ids(self) -> list:
        """
        Get list of original Kilosort cluster IDs.
        
        Returns:
        --------
        original_cluster_ids : list
            List of original cluster IDs from Kilosort data
        """
        return [cluster_info['cluster_id'] for cluster_info in self.clusters.values()]
    
    def get_cluster_info(self, cluster_index: int) -> Dict:
        """
        Get information for a specific cluster.
        
        Parameters:
        -----------
        cluster_index : int
            Cluster index (0-based sequential index, not original Kilosort ID)
            
        Returns:
        --------
        cluster_info : dict
            Dictionary with cluster information
        """
        if cluster_index not in self.clusters:
            raise ValueError(f"Cluster index {cluster_index} not found")
        
        return self.clusters[cluster_index].copy()
    
    def get_spike_times(self, cluster_index: int, 
                       start_time: float = 0, 
                       end_time: float = np.inf) -> np.ndarray:
        """
        Get spike times for a cluster within a time window.
        
        Parameters:
        -----------
        cluster_index : int
            Cluster index (0-based sequential index, not original Kilosort ID)
        start_time : float
            Start time in milliseconds
        end_time : float
            End time in milliseconds
            
        Returns:
        --------
        spike_times : np.ndarray
            Filtered spike times
        """
        if cluster_index not in self.clusters:
            raise ValueError(f"Cluster index {cluster_index} not found")
        
        spike_times = self.clusters[cluster_index]['spike_times']
        
        # Apply time filter
        time_mask = (spike_times >= start_time) & (spike_times <= end_time)
        return spike_times[time_mask]
    
    def create_population_raster(self, start_time: float = 0, 
                                end_time: float = np.inf,
                                bin_size_ms: float = 50.0,
                                zscore_neurons: bool = True,
                                cluster_ids: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Create time-binned population activity matrix.
        
        Parameters:
        -----------
        start_time : float
            Start time in milliseconds (default: 0)
        end_time : float
            End time in milliseconds (default: inf, uses session duration)
        bin_size_ms : float
            Time bin size in milliseconds (default: 50)
        zscore_neurons : bool
            Whether to z-score each neuron's activity (default: True)
        cluster_ids : list, optional
            Specific cluster indices to include. If None, uses all clusters.
            
        Returns:
        --------
        population_matrix : np.ndarray
            Population activity matrix [neurons x time_bins]
        time_bins : np.ndarray
            Time bin centers in milliseconds
        included_clusters : list
            List of cluster indices included in the matrix
        """
        # Determine which clusters to include
        if cluster_ids is None:
            included_clusters = self.get_cluster_ids()
        else:
            included_clusters = [cid for cid in cluster_ids if cid in self.clusters]
        
        if len(included_clusters) == 0:
            raise ValueError("No valid clusters found")
        
        # Determine time range
        if end_time == np.inf:
            # Find maximum spike time across all clusters
            max_time = 0
            for cluster_index in included_clusters:
                spike_times = self.clusters[cluster_index]['spike_times']
                if len(spike_times) > 0:
                    max_time = max(max_time, np.max(spike_times))
            end_time = max_time
        
        # Create time bins
        time_edges = np.arange(start_time, end_time + bin_size_ms, bin_size_ms)
        time_bins = time_edges[:-1] + bin_size_ms / 2  # Bin centers
        n_time_bins = len(time_bins)
        
        # Initialize population matrix
        n_neurons = len(included_clusters)
        population_matrix = np.zeros((n_neurons, n_time_bins))
        
        # Fill matrix with spike counts
        for i, cluster_index in enumerate(included_clusters):
            spike_times = self.get_spike_times(cluster_index, start_time, end_time)
            
            if len(spike_times) > 0:
                # Count spikes in each time bin
                spike_counts, _ = np.histogram(spike_times, bins=time_edges)
                population_matrix[i, :] = spike_counts
        
        # Convert to firing rates (spikes/second)
        bin_size_sec = bin_size_ms / 1000.0
        population_matrix = population_matrix / bin_size_sec
        
        # Apply z-scoring if requested
        if zscore_neurons:
            for i in range(n_neurons):
                # Only z-score if there's variance
                if np.std(population_matrix[i, :]) > 0:
                    population_matrix[i, :] = zscore(population_matrix[i, :])
        
        if self.verbose:
            print(f"Created population matrix: {n_neurons} neurons x {n_time_bins} time bins")
            print(f"Time range: {start_time:.1f} - {end_time:.1f} ms")
            print(f"Bin size: {bin_size_ms} ms")
            if zscore_neurons:
                print("Applied z-scoring to neurons")
        
        return population_matrix, time_bins, included_clusters
    
    def create_event_colormap(self, time_bins: np.ndarray, event_column: str, 
                             aggregation: str = 'mean', 
                             timestamp_column: str = 'timestamp_ms') -> np.ndarray:
        """
        Create a color map for time bins based on event data.
        
        Parameters:
        -----------
        time_bins : np.ndarray
            Time bin centers in milliseconds (typically from create_population_raster)
        event_column : str
            Column name from events dataframe to use for color mapping
        aggregation : str
            How to aggregate events within each time bin:
            - 'mean': average value (for continuous variables like speed)
            - 'max': maximum value 
            - 'min': minimum value
            - 'any': True if any event in bin is True (for boolean variables)
            - 'all': True if all events in bin are True (for boolean variables)
            - 'sum': sum of values in bin
            - 'count': number of events in bin
        timestamp_column : str
            Column name containing timestamps (default: 'timestamp_ms')
            
        Returns:
        --------
        color_values : np.ndarray
            Array of values for each time bin, same length as time_bins
            
        Examples:
        ---------
        # Get population raster
        pop_matrix, time_bins, clusters = session.create_population_raster()
        
        # Create color map based on reward state (boolean)
        reward_colors = session.create_event_colormap(time_bins, 'reward_state', 'any')
        
        # Create color map based on average speed
        speed_colors = session.create_event_colormap(time_bins, 'speed', 'mean')
        """
        if self.events.empty:
            if self.verbose:
                print("Warning: No events data available")
            return np.zeros(len(time_bins))
        
        if event_column not in self.events.columns:
            raise ValueError(f"Column '{event_column}' not found in events data. "
                           f"Available columns: {list(self.events.columns)}")
        
        if timestamp_column not in self.events.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in events data")
        
        # Convert timestamps to milliseconds if needed
        if pd.api.types.is_datetime64_any_dtype(self.events[timestamp_column]):
            # Convert datetime to milliseconds from start
            start_time = self.events[timestamp_column].iloc[0]
            event_times_ms = (self.events[timestamp_column] - start_time).dt.total_seconds() * 1000
        else:
            # Assume already in milliseconds - convert to numeric to handle any string values
            event_times_ms = pd.to_numeric(self.events[timestamp_column], errors='coerce')
        
        # Remove any NaN values that might result from conversion
        valid_mask = ~pd.isna(event_times_ms)
        if not valid_mask.all():
            if self.verbose:
                print(f"Warning: {(~valid_mask).sum()} invalid timestamps found and will be ignored")
            event_times_ms = event_times_ms[valid_mask]
            # Also filter the events dataframe to match
            events_subset = self.events[valid_mask]
        else:
            events_subset = self.events
        
        # Calculate time bin edges from bin centers
        bin_size_ms = time_bins[1] - time_bins[0] if len(time_bins) > 1 else 50.0
        time_edges = time_bins - bin_size_ms / 2
        time_edges = np.append(time_edges, time_edges[-1] + bin_size_ms)
        
        # Initialize color values array
        color_values = np.zeros(len(time_bins))
        
        # For each time bin, find events that fall within it and aggregate
        for i, (start_edge, end_edge) in enumerate(zip(time_edges[:-1], time_edges[1:])):
            # Find events within this time bin
            mask = (event_times_ms >= start_edge) & (event_times_ms < end_edge)
            bin_events = events_subset.loc[events_subset.index[mask], event_column]
            
            if len(bin_events) == 0:
                # No events in this bin
                color_values[i] = 0 if aggregation in ['mean', 'max', 'min', 'sum'] else False
                continue
            
            # Apply aggregation function
            if aggregation == 'mean':
                color_values[i] = bin_events.mean()
            elif aggregation == 'max':
                color_values[i] = bin_events.max()
            elif aggregation == 'min':
                color_values[i] = bin_events.min()
            elif aggregation == 'any':
                color_values[i] = bin_events.any()
            elif aggregation == 'all':
                color_values[i] = bin_events.all()
            elif aggregation == 'sum':
                color_values[i] = bin_events.sum()
            elif aggregation == 'count':
                color_values[i] = len(bin_events)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}. "
                               f"Supported: 'mean', 'max', 'min', 'any', 'all', 'sum', 'count'")
        
        if self.verbose:
            print(f"Created color map from '{event_column}' using '{aggregation}' aggregation")
            print(f"Value range: {np.min(color_values):.3f} to {np.max(color_values):.3f}")
        
        return color_values
    
    def find_sniff_peaks(self, height: Optional[float] = None, 
                        prominence: Optional[float] = None, 
                        distance: Optional[int] = None,
                        width: Optional[Union[int, Tuple[int, int]]] = None) -> np.ndarray:
        """
        Find peaks in the sniff signal and store peak times in sniff_times attribute.
        
        Parameters:
        -----------
        height : float, optional
            Required minimum height of peaks. If None, uses default scipy behavior.
        prominence : float, optional
            Required prominence of peaks. If None, uses default scipy behavior.
        distance : int, optional
            Required minimal horizontal distance between peaks in samples.
        width : int or tuple of ints, optional
            Required width of peaks in samples.
            
        Returns:
        --------
        peak_times : np.ndarray
            Array of peak times in milliseconds
            
        Examples:
        ---------
        # Find all peaks with default parameters
        peak_times = session.find_sniff_peaks()
        
        # Find prominent peaks with minimum distance
        peak_times = session.find_sniff_peaks(prominence=0.5, distance=100)
        
        # Find peaks with specific height threshold
        peak_times = session.find_sniff_peaks(height=0.2)
        """
        if len(self.sniff) == 0:
            if self.verbose:
                print("Warning: No sniff data available for peak detection")
            self.sniff_times = np.array([])
            return self.sniff_times
        
        # Determine effective sampling rate after any processing
        effective_sampling_rate = self.sampling_rate
        
        # Check if sniff signal needs filtering and decimation
        if len(self.sniff) > self.duration * 1.5:  # Signal is at higher sample rate than milliseconds
            if self.verbose:
                print("Filtering and decimating sniff signal from 30kHz to 1kHz...")
            
            # Apply 60Hz notch filter
            b, a = signal.iirnotch(60, 10, 30_000)
            self.sniff = signal.filtfilt(b, a, self.sniff)
            
            # Apply low-pass filter (20Hz cutoff)
            sos = signal.butter(3, 40, 'low', fs=30_000, output='sos')
            self.sniff = signal.sosfiltfilt(sos, self.sniff)
            
            # Remove DC offset
            self.sniff -= np.median(self.sniff)
            
            # Decimate by factor of 30 (30kHz -> 1kHz)
            self.sniff = signal.decimate(self.sniff, 30)
            self.sniff = self.sniff.astype(np.int16)
            
            # Update effective sampling rate
            effective_sampling_rate = 1000.0
        
        # Build parameters dictionary for find_peaks
        peak_params = {}
        if height is not None:
            peak_params['height'] = height
        if prominence is not None:
            peak_params['prominence'] = prominence
        if distance is not None:
            peak_params['distance'] = distance
        if width is not None:
            peak_params['width'] = width
        
        # Find peaks in the sniff signal
        peak_indices, _ = find_peaks(self.sniff, **peak_params)
        
        # Convert peak indices to times in milliseconds using correct sampling rate
        self.sniff_times = peak_indices * (1000.0 / effective_sampling_rate)
        
        if self.verbose:
            print(f"Found {len(self.sniff_times)} peaks in sniff signal")
            if len(self.sniff_times) > 0:
                print(f"Peak times range: {self.sniff_times[0]:.1f} - {self.sniff_times[-1]:.1f} ms")
        
        return self.sniff_times
        
    def get_sniff_data(self, start_time_ms: float = 0, 
                      end_time_ms: float = np.inf) -> np.ndarray:
        """
        Get sniff data within a time window.
        
        Parameters:
        -----------
        start_time_ms : float
            Start time in milliseconds (default: 0)
        end_time_ms : float
            End time in milliseconds (default: inf, uses full duration)
            
        Returns:
        --------
        sniff_segment : np.ndarray
            Sniff data for the specified time window
        """
        if len(self.sniff) == 0:
            return np.array([])
        
        # Convert time to samples
        start_sample = int(start_time_ms * self.sampling_rate / 1000.0)
        
        if end_time_ms == np.inf:
            end_sample = len(self.sniff)
        else:
            end_sample = int(end_time_ms * self.sampling_rate / 1000.0)
        
        # Apply bounds checking
        start_sample = max(0, start_sample)
        end_sample = min(len(self.sniff), end_sample)
        
        return self.sniff[start_sample:end_sample]
    
    def has_sniff_data(self) -> bool:
        """
        Check if sniff data is available.
        
        Returns:
        --------
        has_sniff : bool
            True if sniff data is loaded, False otherwise
        """
        return len(self.sniff) > 0
    
    def filter_clusters(self, filter_expr: str) -> 'SessionData':
        """
        Filter clusters based on a condition and return a new SessionData object.
        
        Parameters:
        -----------
        filter_expr : str
            Filter expression string (e.g., 'best_channel > 16', 'n_spikes >= 100')
            Supported operators: >, <, >=, <=, ==, !=
            Supported keys: any key in cluster dictionary
            
        Returns:
        --------
        filtered_session : SessionData
            New SessionData object containing only clusters that meet the criteria
            
        Examples:
        ---------
        # Filter for olfactory bulb clusters (channels > 16)
        ob_session = session.filter_clusters('best_channel > 16')
        
        # Filter for highly active clusters
        active_session = session.filter_clusters('n_spikes >= 200')
        
        # Filter for specific channel
        ch5_session = session.filter_clusters('best_channel == 5')
        """
        
        # Parse the filter expression
        # Order operators by length (longest first) to avoid partial matches
        operators = [
            ('>=', operator.ge),
            ('<=', operator.le),
            ('==', operator.eq),
            ('!=', operator.ne),
            ('>', operator.gt),
            ('<', operator.lt)
        ]
        
        # Find the operator in the expression
        op_found = None
        op_symbol = None
        
        for symbol, op_func in operators:
            if symbol in filter_expr:
                op_found = op_func
                op_symbol = symbol
                break
        
        if op_found is None:
            raise ValueError(f"No valid operator found in filter expression: {filter_expr}")
        
        # Split the expression
        parts = filter_expr.split(op_symbol)
        if len(parts) != 2:
            raise ValueError(f"Invalid filter expression format: {filter_expr}")
        
        key = parts[0].strip()
        value_str = parts[1].strip()
        
        # Convert value to appropriate type
        try:
            # Try to convert to number first
            if '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            # If not a number, treat as string
            value = value_str.strip('\'"')
        
        # Filter clusters
        filtered_clusters = {}
        filtered_indices = []
        
        for cluster_index, cluster_info in self.clusters.items():
            if key not in cluster_info:
                if self.verbose:
                    print(f"Warning: Key '{key}' not found in cluster {cluster_index}")
                continue
            
            cluster_value = cluster_info[key]
            
            # Handle None values
            if cluster_value is None:
                if op_symbol in ['==', '!=']:
                    if (op_symbol == '==' and value is None) or (op_symbol == '!=' and value is not None):
                        filtered_indices.append(cluster_index)
                continue
            
            # Apply the filter
            try:
                if op_found(cluster_value, value):
                    filtered_indices.append(cluster_index)
            except TypeError as e:
                if self.verbose:
                    print(f"Warning: Cannot compare {cluster_value} with {value} for cluster {cluster_index}: {e}")
                continue
        
        # Create new SessionData object
        filtered_session = copy.copy(self)
        filtered_session.clusters = {}
        
        # Re-index filtered clusters sequentially
        for new_index, old_index in enumerate(filtered_indices):
            filtered_session.clusters[new_index] = copy.deepcopy(self.clusters[old_index])
        
        filtered_session.n_clusters = len(filtered_session.clusters)
        
        # Copy signal data (sniff, etc.)
        filtered_session.sniff = self.sniff.copy() if len(self.sniff) > 0 else np.array([])
        filtered_session.sniff_times = self.sniff_times.copy() if len(self.sniff_times) > 0 else np.array([])
        
        if self.verbose:
            print(f"Filtered from {self.n_clusters} to {filtered_session.n_clusters} clusters using: {filter_expr}")
        
        return filtered_session
    
    def filter_events(self, filter_expr: str, timestamp_column: str = 'timestamp_ms', 
                     return_false_condition: bool = False) -> Union['SessionData', Tuple['SessionData', 'SessionData']]:
        """
        Filter spike times based on events data and return a new SessionData object.
        
        Parameters:
        -----------
        filter_expr : str
            Filter expression string (e.g., 'reward_state == 1', 'speed > 5.0')
            Supported operators: >, <, >=, <=, ==, !=
            Supported keys: any column name in events dataframe
        timestamp_column : str
            Column name containing timestamps (default: 'timestamp_ms')
        return_false_condition : bool
            If True, also return a SessionData object containing spikes where condition was False
            
        Returns:
        --------
        filtered_session : SessionData or tuple of SessionData
            New SessionData object(s) with spike times filtered based on events condition.
            If return_false_condition=True, returns (true_condition_session, false_condition_session)
            
        Examples:
        ---------
        # Filter spikes during reward periods
        reward_session = session.filter_events('reward_state == 1')
        
        # Filter spikes during high speed periods, also get low speed periods
        high_speed, low_speed = session.filter_events('speed > 5.0', return_false_condition=True)
        """
        
        if self.events.empty:
            raise ValueError("No events data available for filtering")
        
        if timestamp_column not in self.events.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in events data")
        
        # Parse the filter expression (reuse logic from filter_clusters)
        operators = [
            ('>=', operator.ge),
            ('<=', operator.le),
            ('==', operator.eq),
            ('!=', operator.ne),
            ('>', operator.gt),
            ('<', operator.lt)
        ]
        
        op_found = None
        op_symbol = None
        
        for symbol, op_func in operators:
            if symbol in filter_expr:
                op_found = op_func
                op_symbol = symbol
                break
        
        if op_found is None:
            raise ValueError(f"No valid operator found in filter expression: {filter_expr}")
        
        parts = filter_expr.split(op_symbol)
        if len(parts) != 2:
            raise ValueError(f"Invalid filter expression format: {filter_expr}")
        
        key = parts[0].strip()
        value_str = parts[1].strip()
        
        if key not in self.events.columns:
            raise ValueError(f"Column '{key}' not found in events data. "
                           f"Available columns: {list(self.events.columns)}")
        
        # Convert value to appropriate type
        try:
            if '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            # Handle boolean values
            value_cleaned = value_str.strip('\'"').lower()
            if value_cleaned in ['true', '1']:
                value = True
            elif value_cleaned in ['false', '0']:
                value = False
            else:
                value = value_str.strip('\'"')
        
        # Convert timestamps to milliseconds if needed
        if pd.api.types.is_datetime64_any_dtype(self.events[timestamp_column]):
            start_time = self.events[timestamp_column].iloc[0]
            event_times_ms = (self.events[timestamp_column] - start_time).dt.total_seconds() * 1000
        else:
            event_times_ms = pd.to_numeric(self.events[timestamp_column], errors='coerce')
        
        # Remove any NaN values
        valid_mask = ~pd.isna(event_times_ms)
        if not valid_mask.all():
            if self.verbose:
                print(f"Warning: {(~valid_mask).sum()} invalid timestamps found and will be ignored")
            event_times_ms = event_times_ms[valid_mask]
            events_subset = self.events[valid_mask].copy()
        else:
            events_subset = self.events.copy()
            
        # Apply the condition to events
        try:
            condition_mask = op_found(events_subset[key], value)
        except TypeError as e:
            raise ValueError(f"Cannot apply condition '{filter_expr}': {e}")
        
        
        def _filter_spikes_by_condition(condition_mask):
            """Helper function to filter spike times based on condition periods."""
            filtered_clusters = {}
            
            # Find time periods where condition is true
            if not condition_mask.any():
                # No periods where condition is true
                for cluster_index, cluster_info in self.clusters.items():
                    new_cluster_info = copy.deepcopy(cluster_info)
                    new_cluster_info['spike_times'] = np.array([])
                    new_cluster_info['n_spikes'] = 0
                    filtered_clusters[cluster_index] = new_cluster_info
                return filtered_clusters
            
            # Get time periods where condition is true
            true_times = event_times_ms[condition_mask].values
            
            # Create time intervals where condition is true
            # For continuous data, we need to group consecutive true periods
            time_intervals = []
            if len(true_times) > 0:
                # Sort times to be safe
                true_times = np.sort(true_times)
                
                # Group consecutive time points (within reasonable gap)
                # Assume events are sampled regularly, use median gap as threshold
                if len(true_times) > 1:
                    gaps = np.diff(true_times)
                    median_gap = np.median(gaps)
                    max_gap = median_gap * 2  # Allow up to 2x median gap
                else:
                    max_gap = 100  # Default 100ms gap
                
                start_time = true_times[0]
                end_time = true_times[0]
                
                for i in range(1, len(true_times)):
                    if true_times[i] - end_time <= max_gap:
                        # Extend current interval
                        end_time = true_times[i]
                    else:
                        # Save current interval and start new one
                        time_intervals.append((start_time, end_time))
                        start_time = true_times[i]
                        end_time = true_times[i]
                
                # Don't forget the last interval
                time_intervals.append((start_time, end_time))
            
            # Filter spikes for each cluster
            for cluster_index, cluster_info in self.clusters.items():
                spike_times = cluster_info['spike_times']
                
                if len(spike_times) == 0 or len(time_intervals) == 0:
                    filtered_spike_times = np.array([])
                else:
                    # Find spikes that fall within any of the true time intervals
                    valid_spikes = []
                    for start_time, end_time in time_intervals:
                        # Include spikes within this interval
                        mask = (spike_times >= start_time) & (spike_times <= end_time)
                        valid_spikes.extend(spike_times[mask])
                    
                    filtered_spike_times = np.array(valid_spikes)
                
                # Create new cluster info with filtered spike times
                new_cluster_info = copy.deepcopy(cluster_info)
                new_cluster_info['spike_times'] = filtered_spike_times
                new_cluster_info['n_spikes'] = len(filtered_spike_times)
                filtered_clusters[cluster_index] = new_cluster_info
            
            return filtered_clusters
        
        # Create filtered session for true condition
        true_session = copy.copy(self)
        true_session.clusters = _filter_spikes_by_condition(condition_mask)
        true_session.n_clusters = len(true_session.clusters)
        
        # Copy signal data
        true_session.sniff = self.sniff.copy() if len(self.sniff) > 0 else np.array([])
        true_session.sniff_times = self.sniff_times.copy() if len(self.sniff_times) > 0 else np.array([])
        true_session.events = self.events.copy()
        
        if self.verbose:
            total_true_spikes = sum(cluster['n_spikes'] for cluster in true_session.clusters.values())
            total_original_spikes = sum(cluster['n_spikes'] for cluster in self.clusters.values())
            print(f"Filtered spikes with condition '{filter_expr}': {total_true_spikes}/{total_original_spikes} spikes retained")
        
        if not return_false_condition:
            return true_session
        
        # Create filtered session for false condition
        false_session = copy.copy(self)
        false_session.clusters = _filter_spikes_by_condition(~condition_mask)
        false_session.n_clusters = len(false_session.clusters)
        
        # Copy signal data
        false_session.sniff = self.sniff.copy() if len(self.sniff) > 0 else np.array([])
        false_session.sniff_times = self.sniff_times.copy() if len(self.sniff_times) > 0 else np.array([])
        false_session.events = self.events.copy()
        
        if self.verbose:
            total_false_spikes = sum(cluster['n_spikes'] for cluster in false_session.clusters.values())
            print(f"False condition spikes: {total_false_spikes}/{total_original_spikes} spikes")
        
        return true_session, false_session
    
    def get_session_summary(self) -> Dict:
        """
        Get a summary of the session data.
        
        Returns:
        --------
        summary : dict
            Dictionary containing basic session information
        """
        # Basic session info
        summary = {
            'mouse_id': self.mouse_id,
            'session_id': self.session_id,
            'experiment': self.experiment,
            'n_clusters': self.n_clusters,
        }
        
        # Event columns
        if not self.events.empty:
            summary['event_columns'] = list(self.events.columns)
            
            # Calculate session duration from timestamp_ms column
            if 'timestamp_ms' in self.events.columns:
                timestamps = self.events['timestamp_ms']
                duration_ms = timestamps.max() - timestamps.min()
                summary['session_duration_min'] = round(duration_ms / (1000 * 60), 2)
            else:
                summary['session_duration_min'] = 'N/A (no timestamp_ms column)'
        else:
            summary['event_columns'] = []
            summary['session_duration_min'] = 'N/A (no events data)'
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of SessionData object."""
        return f"SessionData({self.mouse_id}_{self.session_id}, {self.n_clusters} clusters)"
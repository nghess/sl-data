"""
SessionData class for storing and manipulating neural and behavioral data.

This module provides a standalone class to encapsulate neural data from a single recording session,
including spike times, cluster information, and simple population activity binning tools.

Code by Nate Gonzales-Hess, August 2025.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List
from scipy.stats import zscore
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
    """
    
    def __init__(self, mouse_id: str, session_id: str, experiment: str, base_path: str = "S:\\", 
                 sampling_rate: float = 30000.0, min_spikes: int = 50,
                 verbose: bool = True):
        """
        Initialize SessionData object by loading Kilosort data.
        
        Parameters:
        -----------
        mouse_id : str
            Mouse identifier (e.g., '7001')
        session_id : str
            Session identifier (e.g., 'v1')
        base_path : str
            Base directory path for data files
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

        # Initialize signal traces
        self.sniff = np.array([])
        self.iti = np.array([])
        self.reward = np.array([])

        # Load the data
        self._load_kilosort_data()
        self._load_events_data()
        self._process_clusters()
        self._process_signals()
        
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

    def _load_kilosort_data(self):
        """Load raw Kilosort data files."""
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
            raise ValueError(f"Failed to load Kilosort data: {e}")
    
    def _load_kilosort_files(self, mouse_id: str, session_id: str, experiment: str, base_path: str = "S:\\", 
                            files_needed: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load kilosort spike sorting data for a given mouse and session using flexible path finding.
        
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
                     'whitening_mat_inv', 'sniff']
            
        Returns:
        --------
        data : dict
            Dictionary containing loaded data arrays/dataframes
        """
        # Default files to load
        if files_needed is None:
            files_needed = ['spike_times', 'spike_templates', 'templates', 'amplitudes', 
                           'whitening_mat_inv', 'sniff']
        
        data = {}
        
        # Define filename mapping
        target_files = {
            'spike_times': 'spike_times.npy',
            'spike_templates': 'spike_templates.npy', 
            'templates': 'templates.npy',
            'amplitudes': 'amplitudes.npy',
            'whitening_mat_inv': 'whitening_mat_inv.npy',
            'sniff': 'sniff.npy'
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
            raise FileNotFoundError(f"Events file (events.csv) not found for {self.mouse_id}_{self.session_id}")
        
        self.events = pd.read_csv(events_path)
    
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
    
    def get_sniff_times(self) -> np.ndarray:
        """
        Get time vector for sniff data in milliseconds.
        
        Returns:
        --------
        times : np.ndarray
            Time vector in milliseconds
        """
        if len(self.sniff) == 0:
            return np.array([])
        
        return np.arange(len(self.sniff)) * (1000.0 / self.sampling_rate)
    
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
        
        if self.verbose:
            print(f"Filtered from {self.n_clusters} to {filtered_session.n_clusters} clusters using: {filter_expr}")
        
        return filtered_session
    
    def __repr__(self) -> str:
        """String representation of SessionData object."""
        return f"SessionData({self.mouse_id}_{self.session_id}, {self.n_clusters} clusters)"
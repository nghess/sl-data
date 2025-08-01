"""
SessionData class for storing and manipulating Kilosort neural data.

This module provides a standalone class to encapsulate neural data from a single recording session,
including spike times, cluster information, and population activity analysis tools.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List
from scipy.stats import zscore
import os


class SessionData:
    """
    A class to store and manipulate Kilosort data for a single recording session.
    
    This class loads neural data from Kilosort output files and provides methods
    for analyzing population activity patterns across time.
    
    Attributes:
    -----------
    mouse_id : str
        Mouse identifier
    session_id : str  
        Session identifier
    clusters : dict
        Dictionary containing cluster information with keys:
        - cluster_id: dict with 'best_channel', 'spike_times', 'waveform_template'
    sampling_rate : float
        Neural data sampling rate in Hz
    n_clusters : int
        Number of clusters loaded
    """
    
    def __init__(self, mouse_id: str, session_id: str, base_path: str = "S:\\", 
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
        self.base_path = base_path
        self.sampling_rate = sampling_rate
        self.min_spikes = min_spikes
        self.verbose = verbose
        
        # Initialize containers
        self.clusters = {}
        self.raw_data = {}
        self.n_clusters = 0
        
        # Load the data
        self._load_kilosort_data()
        self._process_clusters()
        
        if verbose:
            print(f"Loaded {self.n_clusters} clusters for {mouse_id}_{session_id}")
    
    def _load_kilosort_data(self):
        """Load raw Kilosort data files."""
        try:
            self.raw_data = self._load_kilosort_files(
                self.mouse_id, 
                self.session_id, 
                base_path=self.base_path,
                files_needed=['spike_times', 'spike_templates', 'templates']
            )
            
            if self.verbose:
                print(f"Loaded Kilosort data: {list(self.raw_data.keys())}")
                
        except Exception as e:
            raise ValueError(f"Failed to load Kilosort data: {e}")
    
    def _load_kilosort_files(self, mouse_id: str, session_id: str, base_path: str = "S:\\", 
                            files_needed: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load kilosort spike sorting data for a given mouse and session.
        
        Parameters:
        -----------
        mouse_id : str
            Mouse identifier (e.g., '7001')
        session_id : str
            Session identifier (e.g., 'v1')
        base_path : str
            Base data directory path (default: 'S:\\')
        files_needed : list of str, optional
            Specific files to load. If None, loads all standard files.
            Options: ['spike_times', 'spike_templates', 'templates', 'amplitudes', 
                     'whitening_mat_inv']
            
        Returns:
        --------
        data : dict
            Dictionary containing loaded data arrays/dataframes
        """
        # Handle different base_path formats
        if "kilosorted" in base_path:
            kilosort_path = Path(base_path) / mouse_id / session_id
        else:
            kilosort_path = Path(base_path) / "clickbait-visual" / "kilosorted" / mouse_id / session_id
        
        # Default files to load
        if files_needed is None:
            files_needed = ['spike_times', 'spike_templates', 'templates', 'amplitudes', 
                           'whitening_mat_inv']
        
        data = {}
        
        # Load numpy files
        numpy_files = {
            'spike_times': 'spike_times.npy',
            'spike_templates': 'spike_templates.npy', 
            'templates': 'templates.npy',
            'amplitudes': 'amplitudes.npy',
            'whitening_mat_inv': 'whitening_mat_inv.npy'
        }
        
        for key in files_needed:
            if key in numpy_files:
                file_path = kilosort_path / numpy_files[key]
                if file_path.exists():
                    data[key] = np.load(file_path)
                elif self.verbose:
                    print(f"Warning: {file_path} not found")
        
        
        return data
    
    def _load_events_data(self, mouse_id: str, session_id: str, base_path: str = "S:\\") -> pd.DataFrame:
        """
        Load behavioral events data for a given mouse and session.
        
        Parameters:
        -----------
        mouse_id : str
            Mouse identifier (e.g., '7001')
        session_id : str
            Session identifier (e.g., 'v1')
        base_path : str
            Base data directory path (default: 'S:\\')
            
        Returns:
        --------
        events : pd.DataFrame
            Events dataframe with behavioral data
        """
        # Handle different base_path formats
        if "events" in base_path:
            events_path = Path(base_path) / mouse_id / session_id / "events.csv"
        else:
            events_path = Path(base_path) / "clickbait-visual" / "events" / mouse_id / session_id / "events.csv"
        
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
        
        return pd.read_csv(events_path)
    
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
    
    def _get_session_directories(self, mouse_id: str, base_path: str = "S:\\") -> List[str]:
        """
        Get sorted list of session directories for a mouse.
        
        Parameters:
        -----------
        mouse_id : str
            Mouse identifier (e.g., '7001')
        base_path : str
            Base data directory path
            
        Returns:
        --------
        session_dirs : list of str
            Sorted list of session directory names
        """
        # Handle different base_path formats
        if "kilosorted" in base_path:
            kilosort_mouse_path = Path(base_path) / mouse_id
        else:
            kilosort_mouse_path = Path(base_path) / "clickbait-visual" / "kilosorted" / mouse_id
        
        if not kilosort_mouse_path.exists():
            return []
        
        # Get directories and sort numerically by session number
        session_dirs = [d for d in os.listdir(kilosort_mouse_path) 
                       if os.path.isdir(kilosort_mouse_path / d)]
        
        # Extract numeric part and sort
        session_numbers = [int(''.join(filter(str.isdigit, d))) for d in session_dirs]
        sorted_indices = np.argsort(session_numbers)
        
        return [session_dirs[i] for i in sorted_indices]
    
    def _process_clusters(self):
        """Process clusters and extract relevant information."""
        if 'spike_times' not in self.raw_data or 'spike_templates' not in self.raw_data:
            raise ValueError("Missing required spike data")
        
        spike_times = self.raw_data['spike_times']
        spike_templates = self.raw_data['spike_templates']
        templates = self.raw_data.get('templates')
        
        # Get unique cluster IDs
        unique_clusters = np.unique(spike_templates)
        
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
            
            # Store cluster information
            self.clusters[int(cluster_id)] = {
                'best_channel': best_channel,
                'spike_times': cluster_spike_times,
                'waveform_template': waveform_template,
                'n_spikes': len(cluster_spike_times)
            }
        
        self.n_clusters = len(self.clusters)
    
    def get_cluster_ids(self) -> list:
        """
        Get list of cluster IDs.
        
        Returns:
        --------
        cluster_ids : list
            List of cluster IDs
        """
        return list(self.clusters.keys())
    
    def get_cluster_info(self, cluster_id: int) -> Dict:
        """
        Get information for a specific cluster.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster ID
            
        Returns:
        --------
        cluster_info : dict
            Dictionary with cluster information
        """
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        return self.clusters[cluster_id].copy()
    
    def get_spike_times(self, cluster_id: int, 
                       start_time: float = 0, 
                       end_time: float = np.inf) -> np.ndarray:
        """
        Get spike times for a cluster within a time window.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster ID
        start_time : float
            Start time in milliseconds
        end_time : float
            End time in milliseconds
            
        Returns:
        --------
        spike_times : np.ndarray
            Filtered spike times
        """
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        spike_times = self.clusters[cluster_id]['spike_times']
        
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
            Specific cluster IDs to include. If None, uses all clusters.
            
        Returns:
        --------
        population_matrix : np.ndarray
            Population activity matrix [neurons x time_bins]
        time_bins : np.ndarray
            Time bin centers in milliseconds
        included_clusters : list
            List of cluster IDs included in the matrix
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
            for cluster_id in included_clusters:
                spike_times = self.clusters[cluster_id]['spike_times']
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
        for i, cluster_id in enumerate(included_clusters):
            spike_times = self.get_spike_times(cluster_id, start_time, end_time)
            
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
    
    def get_session_summary(self) -> Dict:
        """
        Get summary statistics for the session.
        
        Returns:
        --------
        summary : dict
            Dictionary with session statistics
        """
        if len(self.clusters) == 0:
            return {'n_clusters': 0}
        
        # Calculate summary statistics
        spike_counts = [info['n_spikes'] for info in self.clusters.values()]
        
        # Get session duration
        all_spike_times = []
        for cluster_info in self.clusters.values():
            all_spike_times.extend(cluster_info['spike_times'])
        
        session_duration = (np.max(all_spike_times) - np.min(all_spike_times)) / 1000.0 if all_spike_times else 0
        
        # Count clusters by brain region (based on channel number)
        ca1_clusters = 0
        ob_clusters = 0
        
        for cluster_info in self.clusters.values():
            if cluster_info['best_channel'] is not None:
                if cluster_info['best_channel'] <= 16:
                    ca1_clusters += 1
                else:
                    ob_clusters += 1
        
        summary = {
            'n_clusters': len(self.clusters),
            'ca1_clusters': ca1_clusters,
            'ob_clusters': ob_clusters,
            'session_duration_sec': session_duration,
            'total_spikes': sum(spike_counts),
            'mean_spikes_per_cluster': np.mean(spike_counts),
            'median_spikes_per_cluster': np.median(spike_counts),
            'cluster_ids': self.get_cluster_ids()
        }
        
        return summary
    
    def load_events_data(self, events_base_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load behavioral events data for this session.
        
        Parameters:
        -----------
        events_base_path : str, optional
            Base path for events data. If None, uses self.base_path with events structure.
            
        Returns:
        --------
        events : pd.DataFrame
            Events dataframe with behavioral data
        """
        if events_base_path is None:
            # Modify base_path for events
            if "kilosorted" in self.base_path:
                events_base_path = self.base_path.replace("kilosorted", "events")
            else:
                events_base_path = self.base_path
        
        return self._load_events_data(self.mouse_id, self.session_id, events_base_path)
    
    def get_session_directories(self) -> List[str]:
        """
        Get sorted list of session directories for this mouse.
        
        Returns:
        --------
        session_dirs : list of str
            Sorted list of session directory names
        """
        return self._get_session_directories(self.mouse_id, self.base_path)
    
    def __repr__(self) -> str:
        """String representation of SessionData object."""
        return f"SessionData({self.mouse_id}_{self.session_id}, {self.n_clusters} clusters)"
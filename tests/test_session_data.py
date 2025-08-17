"""
Test script for SessionData class functionality.

This script demonstrates how to use the SessionData class to load and analyze
neural data from Kilosort output files.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to Python path for testing without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sldata import SessionData

def test_session_data():
    """Test SessionData class with sample mouse and session."""
    
    # Example session
    mouse_id = "7003"
    session_id = "m5"
    
    print("=== Testing SessionData Class ===")
    print(f"Loading data for {mouse_id}_{session_id}...")
    
    try:
        # Initialize SessionData object
        session = SessionData(
            mouse_id=mouse_id,
            session_id=session_id,
            experiment='clickbait-motivate',
            base_path="S:\\",  # Adjust path as needed
            min_spikes=50,
            verbose=True
        )
        
        # Get session summary
        print("\n=== Session Summary ===")
        summary = session.get_session_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Test population raster creation
        print(f"\n=== Creating Population Raster ===")
        
        # Create population matrix for first 10 seconds with 100ms bins
        pop_matrix, time_bins, included_clusters = session.create_population_raster(
            start_time=0,           # Start at 0 ms
            end_time=np.inf,         # End at 10 seconds (10000 ms)
            bin_size_ms=1000,        # 500 ms bins
            zscore_neurons=True,    # Apply z-scoring
            cluster_ids=None        # Use all clusters
        )

        # Create firing rate histograms with 10-second bins for each cluster
        print(f"\n=== Creating Firing Rate Histograms ===")
        
        # Calculate session duration and create 10-second bins
        session_duration_s = time_bins[-1] / 1000
        hist_bin_size_s = 10  # 10 second bins
        hist_bins = np.arange(0, session_duration_s + hist_bin_size_s, hist_bin_size_s)
        
        # Calculate firing rates for each cluster in 10-second bins
        firing_rates = []
        for i, cluster_idx in enumerate(included_clusters):
            spike_times = session.get_spike_times(cluster_idx, 0, np.inf)
            spike_times_s = spike_times / 1000  # Convert to seconds
            
            # Count spikes in each 10-second bin
            spike_counts, _ = np.histogram(spike_times_s, bins=hist_bins)
            # Convert to firing rate (spikes/second)
            cluster_firing_rates = spike_counts / hist_bin_size_s
            firing_rates.append(cluster_firing_rates)
        
        firing_rates = np.array(firing_rates)
        hist_bin_centers = hist_bins[:-1] + hist_bin_size_s / 2
        
        # Create visualization with population raster on top, histograms below
        fig = plt.figure(figsize=(15, 12))
        
        # Population raster (top 1/3 of the figure)
        ax1 = plt.subplot2grid((3, 1), (0, 0))
        im = ax1.imshow(pop_matrix, aspect='auto', cmap='viridis', 
                       extent=[time_bins[0]/1000, time_bins[-1]/1000, 0, len(included_clusters)])
        ax1.set_ylabel('Neuron ID')
        ax1.set_title(f'Population Activity Raster - {mouse_id}_{session_id}')
        plt.colorbar(im, ax=ax1, label='Z-scored Firing Rate')
        
        # Firing rate histograms (bottom 2/3 of the figure)
        ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
        
        # Plot each cluster's firing rate as a line
        for i, cluster_idx in enumerate(included_clusters):
            ax2.plot(hist_bin_centers, firing_rates[i], 
                    alpha=0.7, linewidth=1)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Firing Rate (spikes/s)')
        ax2.set_title(f'Cluster Firing Rates - {len(included_clusters)} clusters (10-second bins)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'test_session_data_{mouse_id}_{session_id}.png', dpi=300, bbox_inches='tight')
        print(f"Saved test plot: test_session_data_{mouse_id}_{session_id}.png")
        plt.show()
        
        # Test spike time extraction for specific cluster
        if cluster_ids:
            test_cluster = cluster_ids[0]
            spike_times = session.get_spike_times(test_cluster, start_time=0, end_time=5000)  # First 5 seconds
            print(f"\n=== Spike Times for Cluster {test_cluster} (first 5s) ===")
            print(f"Number of spikes: {len(spike_times)}")
            print(f"First 10 spike times (ms): {spike_times[:10]}")
        
        print("\n=== Test completed successfully! ===")
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Note: This test requires actual Kilosort data files.")
        print("Adjust the mouse_id, session_id, and base_path parameters to match your data.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    test_session_data()
"""
Test script for SessionData class functionality.

This script demonstrates how to use the SessionData class to load and analyze
neural data from Kilosort output files.
"""

import numpy as np
import matplotlib.pyplot as plt
from session_data import SessionData

def test_session_data():
    """Test SessionData class with sample mouse and session."""
    
    # Example usage - you'll need to adjust these to match your actual data
    mouse_id = "7001"
    session_id = "v10"
    
    print("=== Testing SessionData Class ===")
    print(f"Loading data for {mouse_id}_{session_id}...")
    
    try:
        # Initialize SessionData object
        session = SessionData(
            mouse_id=mouse_id,
            session_id=session_id,
            base_path="S:\\clickbait-visual\\kilosorted",  # Adjust path as needed
            min_spikes=50,
            verbose=True
        )
        
        # Get session summary
        print("\n=== Session Summary ===")
        summary = session.get_session_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Get information for first few clusters
        cluster_ids = session.get_cluster_ids()[:3]  # First 3 clusters
        
        print(f"\n=== Cluster Information (first 3 clusters) ===")
        for cluster_id in cluster_ids:
            info = session.get_cluster_info(cluster_id)
            print(f"Cluster {cluster_id}:")
            print(f"  Best channel: {info['best_channel']}")
            print(f"  Number of spikes: {info['n_spikes']}")
            print(f"  Waveform shape: {info['waveform_template'].shape if info['waveform_template'] is not None else 'None'}")
        
        # Test population raster creation
        print(f"\n=== Creating Population Raster ===")
        
        # Create population matrix for first 10 seconds with 100ms bins
        pop_matrix, time_bins, included_clusters = session.create_population_raster(
            start_time=0,           # Start at 0 ms
            end_time=10000,         # End at 10 seconds (10000 ms)
            bin_size_ms=100,        # 100 ms bins
            zscore_neurons=True,    # Apply z-scoring
            cluster_ids=None        # Use all clusters
        )
        
        print(f"Population matrix shape: {pop_matrix.shape}")
        print(f"Time bins shape: {time_bins.shape}")
        print(f"Number of included clusters: {len(included_clusters)}")
        
        # Create a simple visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot population raster
        im = axes[0].imshow(pop_matrix, aspect='auto', cmap='viridis', 
                           extent=[time_bins[0]/1000, time_bins[-1]/1000, 0, len(included_clusters)])
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Neuron ID')
        axes[0].set_title(f'Population Activity Raster - {mouse_id}_{session_id}')
        plt.colorbar(im, ax=axes[0], label='Z-scored Firing Rate')
        
        # Plot mean population activity over time
        mean_activity = np.mean(pop_matrix, axis=0)
        axes[1].plot(time_bins/1000, mean_activity, 'k-', linewidth=1)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Mean Z-scored Activity')
        axes[1].set_title('Average Population Activity')
        axes[1].grid(True, alpha=0.3)
        
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


def create_synthetic_test():
    """Create a synthetic test when real data is not available."""
    
    print("\n=== Creating Synthetic Test ===")
    
    # Create synthetic spike data
    n_clusters = 5
    session_duration_ms = 30000  # 30 seconds
    
    synthetic_clusters = {}
    
    for cluster_id in range(n_clusters):
        # Generate random spike times
        n_spikes = np.random.randint(100, 500)
        spike_times = np.sort(np.random.uniform(0, session_duration_ms, n_spikes))
        
        # Create synthetic waveform template
        waveform_template = np.random.randn(82)  # Typical Kilosort template length
        
        # Assign random best channel
        best_channel = np.random.randint(0, 32)
        
        synthetic_clusters[cluster_id] = {
            'spike_times': spike_times,
            'waveform_template': waveform_template,
            'best_channel': best_channel,
            'n_spikes': len(spike_times)
        }
    
    print(f"Created {n_clusters} synthetic clusters")
    
    # Test population raster creation with synthetic data
    # (This would require modifying the SessionData class to accept synthetic data)
    print("Synthetic test would require modification of SessionData class to accept synthetic data.")


if __name__ == "__main__":
    # Run the test
    test_session_data()
    
    # If real data test fails, you can uncomment this to see synthetic example
    # create_synthetic_test()
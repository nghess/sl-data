"""
Demo script for spatial analysis of neural activity (place fields).

This script demonstrates how to:
1. Load neural and behavioral data using SessionData
2. Compute and visualize occupancy maps
3. Compute and visualize rate maps for individual neurons
4. Calculate spatial information metrics
5. Compare place fields across different clusters

Code by Nate Gonzales-Hess, December 2024.
"""

import numpy as np
import matplotlib.pyplot as plt

from sldata import SessionData


def main():
    """
    Main demo function showing place field analysis workflow.
    """

    # ============================================================================
    # 1. Load Session Data
    # ============================================================================

    print("=" * 80)
    print("Loading session data...")
    print("=" * 80)

    # Initialize SessionData object
    # Modify these parameters to match your data
    session = SessionData(
        mouse_id='7012',
        session_id='m10',
        experiment='clickbait-motivate',
        base_path='S:/',
        min_spikes=250,
        verbose=True
    )

    print(f"\nLoaded session: {session}")
    print(f"Number of clusters: {session.n_clusters}")
    print(f"Session duration: {session.duration / 1000 / 60:.2f} minutes")

    # Show cluster label distribution
    label_counts = session.get_cluster_label_counts()
    print(f"\nCluster labels:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    # ============================================================================
    # 2. Compute and Plot Occupancy Map
    # ============================================================================

    print("\n" + "=" * 80)
    print("Computing occupancy map...")
    print("=" * 80)

    # Compute occupancy map with metadata
    occupancy_map, occ_metadata = session.compute_occupancy_map(
        bin_size=50.7,  # 1 cm bins
        min_occupancy=0.1,  # minimum 0.1 seconds
        return_metadata=True
    )

    print(f"Occupancy map shape: {occupancy_map.shape}")
    print(f"Number of valid bins: {occ_metadata['n_valid_bins']}")
    print(f"Total time in valid bins: {occ_metadata['total_time']:.2f} seconds")

    # Plot occupancy map
    fig, data = session.plot_occupancy_map(
        bin_size=50.7,
        sigma=1,  # Smooth for visualization
        figsize=(8, 8),
        cbar_min_max_only=True
    )
    plt.show()

    # ============================================================================
    # 3. Compute and Plot Rate Maps for Individual Clusters
    # ============================================================================

    print("\n" + "=" * 80)
    print("Computing rate maps for individual clusters...")
    print("=" * 80)

    # Get list of cluster indices
    cluster_indices = session.get_cluster_ids()

    # Analyze first few clusters (or all if there are few)
    n_clusters_to_analyze = min(3, len(cluster_indices))

    spatial_info_values = []

    for i, cluster_idx in enumerate(cluster_indices[:n_clusters_to_analyze]):
        print(f"\nAnalyzing cluster {cluster_idx}...")

        # Compute rate map with metadata
        rate_map, rate_metadata = session.compute_rate_map(
            cluster_index=cluster_idx,
            bin_size=50.7,
            sigma=1,  # Smooth for visualization
            min_occupancy=0.1,
            return_metadata=True
        )

        spatial_info = rate_metadata['spatial_info']
        mean_rate = rate_metadata['mean_rate']
        total_spikes = rate_metadata['total_spikes']

        spatial_info_values.append(spatial_info)

        print(f"  Spatial Information: {spatial_info:.3f} bits/spike")
        print(f"  Mean firing rate: {mean_rate:.2f} Hz")
        print(f"  Total spikes: {total_spikes}")

        # Plot individual rate map
        session.plot_rate_map(
            cluster_index=cluster_idx,
            bin_size=50.7,
            sigma=1,
            figsize=(6, 6),
            hide_all_ticks=True
        )
        plt.show()

    # ============================================================================
    # 4. Compare Spatial Information Across Clusters
    # ============================================================================

    print("\n" + "=" * 80)
    print("Comparing spatial information across all clusters...")
    print("=" * 80)

    # Compute spatial information for all clusters
    all_spatial_info = []
    all_mean_rates = []

    for cluster_idx in cluster_indices:
        rate_map, metadata = session.compute_rate_map(
            cluster_index=cluster_idx,
            bin_size=50.7,
            sigma=0,  # No smoothing for spatial info calculation
            return_metadata=True
        )
        all_spatial_info.append(metadata['spatial_info'])
        all_mean_rates.append(metadata['mean_rate'])

    # Plot spatial information distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of spatial information
    ax1.hist(all_spatial_info, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Spatial Information (bits/spike)')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title(f'Distribution of Spatial Information\n{session.mouse_id} {session.session_id}')
    ax1.axvline(np.mean(all_spatial_info), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_spatial_info):.3f}')
    ax1.legend()

    # Scatter plot: spatial info vs mean firing rate
    ax2.scatter(all_mean_rates, all_spatial_info, alpha=0.6)
    ax2.set_xlabel('Mean Firing Rate (Hz)')
    ax2.set_ylabel('Spatial Information (bits/spike)')
    ax2.set_title('Spatial Information vs. Firing Rate')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSpatial Information Statistics:")
    print(f"  Mean: {np.mean(all_spatial_info):.3f} bits/spike")
    print(f"  Median: {np.median(all_spatial_info):.3f} bits/spike")
    print(f"  Std: {np.std(all_spatial_info):.3f} bits/spike")
    print(f"  Max: {np.max(all_spatial_info):.3f} bits/spike")
    print(f"  Min: {np.min(all_spatial_info):.3f} bits/spike")

    # Identify top place cells (highest spatial information)
    top_n = min(5, len(cluster_indices))
    top_indices = np.argsort(all_spatial_info)[-top_n:][::-1]

    print(f"\nTop {top_n} clusters by spatial information:")
    for rank, idx in enumerate(top_indices, 1):
        cluster_idx = cluster_indices[idx]
        original_id = session.clusters[cluster_idx]['cluster_id']
        print(f"  {rank}. Cluster {cluster_idx} (original ID: {original_id}): "
              f"{all_spatial_info[idx]:.3f} bits/spike, "
              f"{all_mean_rates[idx]:.2f} Hz")

    # ============================================================================
    # 5. Plot Best Place Cell
    # ============================================================================

    print("\n" + "=" * 80)
    print("Plotting best place cell...")
    print("=" * 80)

    best_cluster_idx = cluster_indices[top_indices[0]]

    fig, data = session.plot_rate_map(
        cluster_index=best_cluster_idx,
        bin_size=50.7,
        sigma=2,  # More smoothing for visualization
        upsample=True,  # High resolution
        figsize=(10, 10),
        cmap='hot'
    )
    plt.show()

    # ============================================================================
    # 6. Compare Rate Maps Across flip_state Conditions
    # ============================================================================

    print("\n" + "=" * 80)
    print("Comparing rate maps across flip_state conditions...")
    print("=" * 80)

    # Check if flip_state column exists
    if 'flip_state' not in session.events.columns:
        print("Warning: 'flip_state' column not found in events data. Skipping flip_state comparison.")
    else:
        # Use the best cluster for comparison
        print(f"Using best cluster (idx={best_cluster_idx}) for flip_state comparison")

        # Compute rate maps for each condition
        print("\nComputing rate map for flip_state=False...")
        rate_map_false, metadata_false = session.compute_rate_map(
            cluster_index=best_cluster_idx,
            bin_size=50.7,
            sigma=2,
            flip_state=False,
            return_metadata=True
        )

        print("Computing rate map for flip_state=True...")
        rate_map_true, metadata_true = session.compute_rate_map(
            cluster_index=best_cluster_idx,
            bin_size=50.7,
            sigma=2,
            flip_state=True,
            return_metadata=True
        )

        print("Computing combined rate map (all data)...")
        rate_map_combined, metadata_combined = session.compute_rate_map(
            cluster_index=best_cluster_idx,
            bin_size=50.7,
            sigma=2,
            filter_flip_state=False,
            return_metadata=True
        )

        # Extract spike counts
        n_spikes_false = metadata_false['total_spikes']
        n_spikes_true = metadata_true['total_spikes']
        n_spikes_combined = metadata_combined['total_spikes']

        print(f"\nSpike counts:")
        print(f"  flip_state=False: {n_spikes_false} spikes")
        print(f"  flip_state=True: {n_spikes_true} spikes")
        print(f"  Combined: {n_spikes_combined} spikes")
        print(f"  Sum of False+True: {n_spikes_false + n_spikes_true} spikes")
        print(f"  Difference: {n_spikes_combined - (n_spikes_false + n_spikes_true)} spikes")

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Get original cluster ID for titles
        original_cluster_id = session.clusters[best_cluster_idx]['cluster_id']

        # Helper function to plot rate map on an axis
        def plot_rate_map_on_axis(ax, rate_map, metadata, title, cmap='hot'):
            # Transpose for display (matching the 90-degree rotation from plot_rate_map)
            valid_bins = metadata['valid_bins']
            rate_map_viz = np.copy(rate_map)
            rate_map_viz[~valid_bins] = np.nan
            rate_map_display = rate_map_viz.T

            # Get spatial info
            spatial_info = metadata['spatial_info']
            n_spikes = metadata['total_spikes']

            # Plot
            im = ax.imshow(rate_map_display, cmap=cmap, aspect='equal', origin='lower',
                          interpolation='bilinear')
            ax.set_title(f'{title}\nSI={spatial_info:.3f} bits/spike, {n_spikes} spikes',
                        fontsize=12)
            ax.axis('off')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Spikes/sec', fontsize=10)
            cbar.outline.set_edgecolor('black')
            cbar.outline.set_linewidth(1)

        # Plot each condition
        plot_rate_map_on_axis(axes[0, 0], rate_map_false, metadata_false,
                              f'Cluster {original_cluster_id} - flip_state=False')
        plot_rate_map_on_axis(axes[0, 1], rate_map_true, metadata_true,
                              f'Cluster {original_cluster_id} - flip_state=True')
        plot_rate_map_on_axis(axes[1, 0], rate_map_combined, metadata_combined,
                              f'Cluster {original_cluster_id} - Combined')

        # Plot spike count comparison in bottom right
        ax_bar = axes[1, 1]
        conditions = ['False', 'True', 'Combined', 'False+True']
        spike_counts = [n_spikes_false, n_spikes_true, n_spikes_combined,
                       n_spikes_false + n_spikes_true]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = ax_bar.bar(conditions, spike_counts, color=colors, alpha=0.7, edgecolor='black')
        ax_bar.set_ylabel('Number of Spikes', fontsize=12)
        ax_bar.set_title('Spike Count Comparison', fontsize=12)
        ax_bar.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, spike_counts):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}',
                       ha='center', va='bottom', fontsize=10)

        plt.suptitle(f'{session.mouse_id} {session.session_id} - flip_state Comparison',
                    fontsize=16, y=0.995)
        plt.tight_layout()
        plt.show()

    print("\nDemo complete!")


if __name__ == "__main__":
    main()

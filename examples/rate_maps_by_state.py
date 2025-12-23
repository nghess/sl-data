"""
Generate flip_state comparison rate maps for all clusters in a session.

This script:
1. Loops through all clusters in a session
2. Computes rate maps for flip_state=False, flip_state=True, and combined
3. Performs statistical testing to compare the two conditions
4. Saves comparison plots to disk (does not display)

Output files are saved as: {mouse_id}_{session_id}_cl{cluster_id}_fs_rate.png

Code by Nate Gonzales-Hess, December 2024.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, wasserstein_distance
from scipy.spatial.distance import cosine

from sldata import SessionData


def spatial_correlation_test(rate_map_1, rate_map_2, valid_bins):
    """
    Test spatial correlation between two rate maps.

    Parameters:
    -----------
    rate_map_1 : np.ndarray
        First rate map
    rate_map_2 : np.ndarray
        Second rate map
    valid_bins : np.ndarray
        Boolean mask of valid bins

    Returns:
    --------
    r : float
        Pearson correlation coefficient
    p : float
        P-value for the correlation
    """
    # Get valid bin values
    vals_1 = rate_map_1[valid_bins].flatten()
    vals_2 = rate_map_2[valid_bins].flatten()

    if len(vals_1) < 3:
        return np.nan, np.nan

    # Compute Pearson correlation
    r, p = pearsonr(vals_1, vals_2)

    return r, p


def earth_movers_distance(rate_map_1, rate_map_2, valid_bins):
    """
    Compute Earth Mover's Distance (Wasserstein distance) between two rate maps.

    This measures the minimum "work" needed to transform one distribution into another.
    Rate maps are z-scored before comparison to make the metric unitless and
    comparable across neurons with different firing rates.
    Lower values indicate more similar distributions.

    Parameters:
    -----------
    rate_map_1 : np.ndarray
        First rate map
    rate_map_2 : np.ndarray
        Second rate map
    valid_bins : np.ndarray
        Boolean mask of valid bins

    Returns:
    --------
    emd : float
        Earth Mover's Distance (z-scored, unitless)
    """
    # Get valid bin values
    vals_1 = rate_map_1[valid_bins].flatten()
    vals_2 = rate_map_2[valid_bins].flatten()

    if len(vals_1) < 3:  # Need at least 3 values for meaningful z-score
        return np.nan

    # Z-score the values to make unitless and comparable across neurons
    from scipy.stats import zscore
    vals_1_z = zscore(vals_1)
    vals_2_z = zscore(vals_2)

    # Compute Wasserstein distance on z-scored values
    emd = wasserstein_distance(vals_1_z, vals_2_z)

    return emd


def cosine_similarity(rate_map_1, rate_map_2, valid_bins):
    """
    Compute cosine similarity between two rate maps.

    Measures the cosine of the angle between the two rate map vectors.
    Range: -1 to 1, where 1 means identical direction.

    Parameters:
    -----------
    rate_map_1 : np.ndarray
        First rate map
    rate_map_2 : np.ndarray
        Second rate map
    valid_bins : np.ndarray
        Boolean mask of valid bins

    Returns:
    --------
    cos_sim : float
        Cosine similarity
    """
    # Get valid bin values
    vals_1 = rate_map_1[valid_bins].flatten()
    vals_2 = rate_map_2[valid_bins].flatten()

    if len(vals_1) < 2:
        return np.nan

    # Compute cosine similarity (1 - cosine distance)
    cos_sim = 1 - cosine(vals_1, vals_2)

    return cos_sim


def plot_flip_state_comparison(session, cluster_idx, output_dir, bin_size=50.7, sigma=1):
    """
    Create and save flip_state comparison plot for a single cluster.

    Parameters:
    -----------
    session : SessionData
        SessionData object
    cluster_idx : int
        Cluster index to analyze
    output_dir : str or Path
        Directory to save plots
    bin_size : float
        Spatial bin size in pixels
    sigma : float
        Gaussian smoothing sigma

    Returns:
    --------
    stats : dict
        Dictionary with statistical test results
    """
    # Get cluster ID for filename
    original_cluster_id = session.clusters[cluster_idx]['cluster_id']

    # Compute rate maps for each condition
    rate_map_false, metadata_false = session.compute_rate_map(
        cluster_index=cluster_idx,
        bin_size=bin_size,
        sigma=sigma,
        flip_state=False,
        return_metadata=True
    )

    rate_map_true, metadata_true = session.compute_rate_map(
        cluster_index=cluster_idx,
        bin_size=bin_size,
        sigma=sigma,
        flip_state=True,
        return_metadata=True
    )

    # Extract spike counts and spatial info
    n_spikes_false = metadata_false['total_spikes']
    n_spikes_true = metadata_true['total_spikes']

    si_false = metadata_false['spatial_info']
    si_true = metadata_true['spatial_info']

    mean_rate_false = metadata_false['mean_rate']
    mean_rate_true = metadata_true['mean_rate']

    # Perform statistical tests
    valid_bins_both = metadata_false['valid_bins'] & metadata_true['valid_bins']

    # 1. Spatial correlation between the two conditions
    r_corr, p_corr = spatial_correlation_test(
        rate_map_false, rate_map_true, valid_bins_both
    )

    # 2. Earth Mover's Distance
    emd = earth_movers_distance(rate_map_false, rate_map_true, valid_bins_both)

    # 3. Cosine Similarity
    cos_sim = cosine_similarity(rate_map_false, rate_map_true, valid_bins_both)

    # Create figure with custom gridspec for better control
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 5.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.3,
                  top=0.85, bottom=0.05)

    ax_false = fig.add_subplot(gs[0, 0])
    ax_true = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[0, 2])

    # Helper function to plot rate map on an axis
    def plot_rate_map_on_axis(ax, rate_map, metadata, title, cmap='hot'):
        # Transpose for display (matching the 90-degree rotation)
        valid_bins = metadata['valid_bins']
        rate_map_viz = np.copy(rate_map)
        rate_map_viz[~valid_bins] = np.nan
        rate_map_display = rate_map_viz.T

        # Get spatial info and spike count
        spatial_info = metadata['spatial_info']
        n_spikes = metadata['total_spikes']
        mean_rate = metadata['mean_rate']

        # Plot
        im = ax.imshow(rate_map_display, cmap=cmap, aspect='equal', origin='lower',
                      interpolation='nearest')
        ax.set_title(f'{title}\nSI={spatial_info:.3f} bits/spike\n'
                    f'{n_spikes} spikes, {mean_rate:.2f} Hz',
                    fontsize=10)
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('Spikes/sec', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)

    # Plot the two conditions
    plot_rate_map_on_axis(ax_false, rate_map_false, metadata_false,
                          f'flip_state = False')
    plot_rate_map_on_axis(ax_true, rate_map_true, metadata_true,
                          f'flip_state = True')

    # Statistics panel
    ax_stats.axis('off')

    # Build statistics text
    stats_text = "Spike Metrics\n"
    stats_text += "=" * 30 + "\n\n"

    stats_text += "Spike Counts:\n"
    stats_text += f"  False: {n_spikes_false}\n"
    stats_text += f"  True:  {n_spikes_true}\n"
    stats_text += f"  Total: {n_spikes_false + n_spikes_true}\n\n"

    stats_text += "Spatial Information:\n"
    stats_text += f"  False: {si_false:.3f} bits/spike\n"
    stats_text += f"  True:  {si_true:.3f} bits/spike\n\n"

    stats_text += "Mean Firing Rate:\n"
    stats_text += f"  False: {mean_rate_false:.2f} Hz\n"
    stats_text += f"  True:  {mean_rate_true:.2f} Hz\n\n"

    stats_text += "=" * 30 + "\n"
    stats_text += "Similarity Metrics\n"
    stats_text += "=" * 30 + "\n\n"

    stats_text += "Spatial Correlation:\n"
    if not np.isnan(r_corr):
        stats_text += f"  r = {r_corr:.3f}\n"
        stats_text += f"  p = {p_corr:.4f}\n"
        if p_corr < 0.001:
            stats_text += "  ***\n"
        elif p_corr < 0.01:
            stats_text += "  **\n"
        elif p_corr < 0.05:
            stats_text += "  *\n"
        else:
            stats_text += "  n.s.\n"
    else:
        stats_text += "  N/A\n"

    stats_text += "\nCosine Similarity:\n"
    if not np.isnan(cos_sim):
        stats_text += f"  {cos_sim:.3f}\n"
    else:
        stats_text += "  N/A\n"

    stats_text += "\nEarth Mover's Distance:\n"
    if not np.isnan(emd):
        stats_text += f"  {emd:.3f}\n"
    else:
        stats_text += "  N/A\n"

    # Display statistics text
    ax_stats.text(0.05, 0.98, stats_text,
                 transform=ax_stats.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 horizontalalignment='left',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.15))

    # Main title with more vertical space
    plt.suptitle(f'{session.mouse_id} {session.session_id} - Cluster {original_cluster_id} - flip_state Comparison',
                fontsize=14, y=0.98)

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = f"{session.mouse_id}_{session.session_id}_cl{original_cluster_id}_fs_rate.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Return statistics
    stats = {
        'cluster_id': original_cluster_id,
        'n_spikes_false': n_spikes_false,
        'n_spikes_true': n_spikes_true,
        'si_false': si_false,
        'si_true': si_true,
        'mean_rate_false': mean_rate_false,
        'mean_rate_true': mean_rate_true,
        'spatial_correlation': r_corr,
        'correlation_pvalue': p_corr,
        'cosine_similarity': cos_sim,
        'earth_movers_distance': emd
    }

    return stats


def rate_map_by_state(mouse_id='7012', session_id='m10', experiment='clickbait-motivate', base_path='D:/data/'):
    """
    Main function to generate flip_state comparison plots for all clusters.
    """

    # ============================================================================
    # Configuration
    # ============================================================================

    # Session parameters
    # mouse_id = '7012'
    # session_id = 'm10'
    # experiment = 'clickbait-motivate'
    # base_path = 'D:/data/'

    # Analysis parameters
    bin_size = 50.7  # 1 cm bins
    sigma = 1  # Smoothing for visualization

    # Output directory
    output_dir = f"./rate_maps/{mouse_id}_{session_id}"

    print("=" * 80)
    print("flip_state Rate Map Comparison")
    print("=" * 80)

    # ============================================================================
    # Load Session Data
    # ============================================================================

    print(f"\nLoading session: {mouse_id} {session_id}...")

    session = SessionData(
        mouse_id=mouse_id,
        session_id=session_id,
        experiment=experiment,
        base_path=base_path,
        min_spikes=250,
        exclude_noise=True,
        verbose=True
    )

    print(f"\nLoaded {session.n_clusters} clusters")

    # Check if flip_state column exists
    if 'flip_state' not in session.events.columns:
        print("Error: 'flip_state' column not found in events data.")
        print("Cannot perform flip_state comparison.")
        return

    # ============================================================================
    # Generate Plots for All Clusters
    # ============================================================================

    print(f"\nGenerating flip_state comparison plots...")
    print(f"Output directory: {output_dir}")
    print("-" * 80)

    cluster_indices = session.get_cluster_ids()
    all_stats = []

    for i, cluster_idx in enumerate(cluster_indices):
        cluster_id = session.clusters[cluster_idx]['cluster_id']
        label = session.clusters[cluster_idx]['label']

        print(f"[{i+1}/{len(cluster_indices)}] Processing cluster {cluster_idx} "
              f"(ID: {cluster_id}, label: {label})...", end=' ')

        try:
            stats = plot_flip_state_comparison(
                session, cluster_idx, output_dir,
                bin_size=bin_size, sigma=sigma
            )
            all_stats.append(stats)

            # Print brief summary
            print(f"✓ ({stats['n_spikes_false']}+{stats['n_spikes_true']} spikes, "
                  f"r={stats['spatial_correlation']:.3f}, "
                  f"cos={stats['cosine_similarity']:.3f})")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    # ============================================================================
    # Summary Statistics
    # ============================================================================

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    if len(all_stats) > 0:
        # Calculate summary statistics
        correlations = [s['spatial_correlation'] for s in all_stats if not np.isnan(s['spatial_correlation'])]
        cosine_sims = [s['cosine_similarity'] for s in all_stats if not np.isnan(s['cosine_similarity'])]
        emds = [s['earth_movers_distance'] for s in all_stats if not np.isnan(s['earth_movers_distance'])]
        sig_correlations = [s for s in all_stats if s['correlation_pvalue'] < 0.05]

        print(f"\nTotal clusters processed: {len(all_stats)}")
        print(f"\nSpatial Correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
        print(f"  Significant (p<0.05): {len(sig_correlations)}/{len(correlations)}")
        print(f"\nCosine Similarity: {np.mean(cosine_sims):.3f} ± {np.std(cosine_sims):.3f}")
        print(f"Earth Mover's Distance: {np.mean(emds):.3f} ± {np.std(emds):.3f}")

        # Find clusters with highest and lowest correlation
        if len(correlations) > 0:
            sorted_stats = sorted(all_stats, key=lambda x: x['spatial_correlation']
                                if not np.isnan(x['spatial_correlation']) else -999, reverse=True)

            print(f"\nHighest correlation: Cluster {sorted_stats[0]['cluster_id']} "
                  f"(r={sorted_stats[0]['spatial_correlation']:.3f}, "
                  f"p={sorted_stats[0]['correlation_pvalue']:.4f})")

            print(f"Lowest correlation: Cluster {sorted_stats[-1]['cluster_id']} "
                  f"(r={sorted_stats[-1]['spatial_correlation']:.3f}, "
                  f"p={sorted_stats[-1]['correlation_pvalue']:.4f})")

    print(f"\nAll plots saved to: {output_dir}")
    print("\nDone!")

"""Get session ids from directory structure."""
def natural_sort_key(path):
    path_str = str(path)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', path_str)]

def get_file_paths(directory: str = '', extension: str = '', keyword: str = '', session_type: str = '', verbose=True) -> list:
    paths = [f for f in Path(directory).glob(f'**/{session_type}*/*.{extension}') if keyword in f.name]
    paths = sorted(paths, key=natural_sort_key)
    if verbose:
        print(f'Found {len(paths)} {keyword}.{extension} files')
    return paths

"""Generate rate maps by flip_state for all sessions"""

base_path='D:/data/'
experiment = "clickbait-motivate"
events_dir = 'bonsai'
session_paths = get_file_paths(f"{base_path}/{experiment}/{events_dir}", 'csv', 'slp', session_type='m')

# Loop through each session and generate rate maps by flip_state
for session_path in session_paths:
    mouse_id = session_path.parent.parent.name
    session_id = session_path.parent.name
    print(f"\nProcessing session: {mouse_id} {session_id}")
    rate_map_by_state(mouse_id=mouse_id, session_id=session_id, experiment=experiment, base_path=base_path)

"""
Demo script for neural population classification using SessionData.

This script demonstrates how to:
1. Load neural data using SessionData
2. Create population activity matrices
3. Prepare behavioral event labels
4. Train a neural classifier
5. Evaluate model performance

Code by Nate Gonzales-Hess, August 2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
import torch
import copy

from sldata import SessionData
from neural_classifier import PopulationClassifier, PopulationDataset


def plot_classifier_results(history: dict, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: np.ndarray, model_name: str = "Classifier"):
    """
    Create standardized plots for classifier evaluation.
    
    Parameters:
    -----------
    history : dict
        Training history with 'train_loss', 'val_loss', 'val_acc'
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Prediction probabilities
    model_name : str
        Name of the model for plot titles
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Training/Validation Loss
    axes[0].plot(history['train_loss'], label='Training Loss', color='blue')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Validation Loss', color='orange')
        
        # Mark the best epoch if available
        if 'best_epoch' in history:
            best_epoch = history['best_epoch']
            best_val_loss = history['val_loss'][best_epoch]
            axes[0].scatter(best_epoch, best_val_loss, color='red', s=100, 
                           marker='*', zorder=5, label=f'Best Model (Epoch {best_epoch + 1})')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy percentages for each cell
    cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create custom annotations with counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentages[i, j]
            row.append(f'{count}\n({percentage:.1f}%)')
        annotations.append(row)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(f'{model_name} - Confusion Matrix')
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Calculate baseline (random classifier performance)
    baseline_precision = np.sum(y_true) / len(y_true)
    
    axes[2].plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
    axes[2].axhline(y=baseline_precision, color='navy', lw=2, linestyle='--', 
                   label=f'Random classifier (AP = {baseline_precision:.3f})')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('Recall (TP/(TP+FN))')
    axes[2].set_ylabel('Precision (TP/(TP+FP))')
    axes[2].set_title(f'{model_name} - Precision-Recall Curve')
    axes[2].legend(loc="lower left")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_data_splits_timeline(session: SessionData, event_column: str, time_bins: np.ndarray, 
                              train_indices: np.ndarray, val_indices: np.ndarray, 
                              test_indices: np.ndarray, bin_size_ms: float, labels: np.ndarray = None):
    """
    Plot timeline showing event_column over time with train/val/test split overlays.
    
    Parameters:
    -----------
    session : SessionData
        Session data containing events
    event_column : str
        Column name for the behavioral event 
    time_bins : np.ndarray
        Time bin centers used for analysis
    train_indices, val_indices, test_indices : np.ndarray
        Indices for each data split
    bin_size_ms : float
        Size of time bins in milliseconds
    labels : np.ndarray, optional
        Labels for time bins (for showing positive/negative)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Convert event times to seconds for plotting
    event_times_s = session.events['timestamp_ms'] / 1000
    time_bins_s = time_bins / 1000
    
    # Plot 1: Event state over time
    event_values = session.events[event_column].astype(int)
    ax1.fill_between(event_times_s, 0, event_values, alpha=0.7, 
                     color='blue' if event_values.iloc[0] else 'red',
                     step='pre', label=f'{event_column}')
    
    # Add color changes for flip states
    flip_changes = np.where(np.diff(event_values))[0]
    for i, change_idx in enumerate(flip_changes):
        change_time = event_times_s.iloc[change_idx]
        next_value = event_values.iloc[change_idx + 1]
        color = 'blue' if next_value else 'red'
        
        # Fill from this change to the next (or end)
        end_idx = flip_changes[i + 1] if i + 1 < len(flip_changes) else len(event_times_s) - 1
        end_time = event_times_s.iloc[end_idx]
        
        ax1.fill_between(event_times_s.iloc[change_idx:end_idx + 1], 0, 
                        event_values.iloc[change_idx:end_idx + 1],
                        alpha=0.7, color=color, step='pre')
    
    ax1.set_ylabel(f'{event_column}')
    ax1.set_title('Behavioral State Over Time')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Data splits overlay with positive/negative labels
    if labels is not None:
        # Separate positive and negative time bins for each split
        train_pos_times = time_bins_s[train_indices][labels[train_indices] == 1]
        train_neg_times = time_bins_s[train_indices][labels[train_indices] == 0]
        val_pos_times = time_bins_s[val_indices][labels[val_indices] == 1]
        val_neg_times = time_bins_s[val_indices][labels[val_indices] == 0]
        test_pos_times = time_bins_s[test_indices][labels[test_indices] == 1]
        test_neg_times = time_bins_s[test_indices][labels[test_indices] == 0]
        
        # Plot each category
        height_offset = 0
        marker_size = 30
        
        # Training data
        if len(train_pos_times) > 0:
            ax2.scatter(train_pos_times, [0.5 + height_offset] * len(train_pos_times), 
                       c='darkgreen', marker='o', s=marker_size, alpha=0.8,
                       label=f'Train Positive ({len(train_pos_times)} bins)')
        height_offset += 0.08
        
        if len(train_neg_times) > 0:
            ax2.scatter(train_neg_times, [0.5 + height_offset] * len(train_neg_times), 
                       c='lightgreen', marker='s', s=marker_size, alpha=0.8,
                       label=f'Train Negative ({len(train_neg_times)} bins)')
        height_offset += 0.08
        
        # Validation data
        if len(val_pos_times) > 0:
            ax2.scatter(val_pos_times, [0.5 + height_offset] * len(val_pos_times), 
                       c='darkorange', marker='o', s=marker_size, alpha=0.8,
                       label=f'Val Positive ({len(val_pos_times)} bins)')
        height_offset += 0.08
        
        if len(val_neg_times) > 0:
            ax2.scatter(val_neg_times, [0.5 + height_offset] * len(val_neg_times), 
                       c='orange', marker='s', s=marker_size, alpha=0.8,
                       label=f'Val Negative ({len(val_neg_times)} bins)')
        height_offset += 0.08
        
        # Test data
        if len(test_pos_times) > 0:
            ax2.scatter(test_pos_times, [0.5 + height_offset] * len(test_pos_times), 
                       c='darkviolet', marker='o', s=marker_size, alpha=0.8,
                       label=f'Test Positive ({len(test_pos_times)} bins)')
        height_offset += 0.08
        
        if len(test_neg_times) > 0:
            ax2.scatter(test_neg_times, [0.5 + height_offset] * len(test_neg_times), 
                       c='plum', marker='s', s=marker_size, alpha=0.8,
                       label=f'Test Negative ({len(test_neg_times)} bins)')
        
        ax2.set_title('Train/Validation/Test Split Distribution (Positive/Negative Labels)')
    
    else:
        # Fallback to original plotting if labels not provided
        split_labels = np.full(len(time_bins), 'unused', dtype='U10')
        split_labels[train_indices] = 'train'
        split_labels[val_indices] = 'validation'  
        split_labels[test_indices] = 'test'
        
        colors = {'train': 'green', 'validation': 'orange', 'test': 'purple', 'unused': 'gray'}
        height = 0
        
        for split_type, color in colors.items():
            mask = split_labels == split_type
            if np.any(mask):
                times = time_bins_s[mask]
                ax2.scatter(times, [0.5 + height] * len(times), c=color, alpha=0.7, 
                           s=50, label=f'{split_type.title()} ({np.sum(mask)} bins)')
                height += .1
        
        ax2.set_title('Train/Validation/Test Split Distribution')
    
    ax2.set_ylabel('Data Split')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, 1.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def prepare_classification_data(session: SessionData, event_column: str, 
                              bin_size_ms: float = 50.0, 
                              min_time_ms: float = 0,
                              max_time_ms: float = None,
                              centroid_y_threshold: float = None) -> tuple:
    """
    Prepare data for classification from SessionData.
    
    Parameters:
    -----------
    session : SessionData
        Loaded session data
    event_column : str
        Column name for the behavioral event to predict
    bin_size_ms : float
        Time bin size in milliseconds
    min_time_ms : float
        Start time for analysis
    max_time_ms : float
        End time for analysis (if None, uses session duration)
    centroid_y_threshold : float, optional
        If provided, only include time bins where centroid_y <= threshold.
        If None, uses 50% of max centroid_y value.
        
    Returns:
    --------
    population_matrix : np.ndarray
        Population activity matrix [time_bins x neurons] - filtered by centroid_y
    labels : np.ndarray
        Binary labels for each time bin - filtered by centroid_y
    time_bins : np.ndarray
        Time bin centers - filtered by centroid_y
    """
    print(f"Preparing data for predicting '{event_column}'...")
    
    # Create population matrix
    if max_time_ms is None:
        max_time_ms = np.inf
    
    pop_matrix, time_bins, cluster_ids = session.create_population_raster(
        start_time=min_time_ms,
        end_time=max_time_ms,
        bin_size_ms=bin_size_ms,
        zscore_neurons=True
    )
    
    # Create event labels - transpose to get [time_bins x neurons]
    population_matrix = pop_matrix.T
    
    # Get behavioral labels for each time bin
    labels = session.create_event_colormap(
        time_bins, 
        event_column, 
        aggregation='any'  # True if any event in time bin is True
    )
    
    # Apply centroid_y filtering if requested
    if 'centroid_y' in session.events.columns:
        # Get centroid_y values for each time bin
        centroid_y_values = session.create_event_colormap(
            time_bins,
            'centroid_y',
            aggregation='mean'  # Use mean centroid_y for each time bin
        )
        
        # Determine threshold
        if centroid_y_threshold is None:
            max_centroid_y = session.events['centroid_y'].max()
            centroid_y_threshold = max_centroid_y * 0.5
        
        # Create mask for time bins below threshold
        centroid_mask = centroid_y_values <= centroid_y_threshold
        
        # Apply filtering
        population_matrix = population_matrix[centroid_mask]
        labels = labels[centroid_mask]
        time_bins = time_bins[centroid_mask]
        
        print(f"Applied centroid_y filter (≤{centroid_y_threshold:.1f}): "
              f"kept {np.sum(centroid_mask)}/{len(centroid_mask)} time bins")
    else:
        print("Warning: No 'centroid_y' column found in events data - skipping position filtering")
    
    print(f"Final data shape: {population_matrix.shape}")
    print(f"Final labels shape: {labels.shape}")
    print(f"Positive class proportion: {np.mean(labels):.3f}")
    print(f"Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f} ms")
    
    return population_matrix, labels, time_bins




def run_classification_demo(mouse_id: str, session_id: str, experiment: str,
                          event_column: str = 'flip_state',
                          base_path: str = "S:\\",
                          bin_size_ms: float = 50.0,
                          test_size: float = 0.2,
                          epochs: int = 100,
                          centroid_y_threshold: float = None,
                          use_best_model: bool = False,
                          cluster_filter: str = None,
                          exclude_final_flip: bool = False):
    """
    Run complete classification demo.
    
    Parameters:
    -----------
    mouse_id : str
        Mouse identifier
    session_id : str  
        Session identifier
    experiment : str
        Experiment name
    event_column : str
        Event column to predict
    base_path : str
        Base data path
    bin_size_ms : float
        Time bin size in milliseconds
    test_size : float
        Fraction of data for testing
    epochs : int
        Number of training epochs
    centroid_y_threshold : float, optional
        Only include time bins where centroid_y <= threshold.
        If None, uses 50% of max centroid_y value.
    use_best_model : bool
        If True, use the model with best validation loss for testing.
        If False, use the final model after all epochs (default).
    cluster_filter : str, optional
        Filter expression for clusters (same syntax as SessionData.filter_clusters()).
        If None, uses all clusters. Examples: 'best_channel >= 17', 'n_spikes > 100'.
    exclude_final_flip : bool
        If True, excludes data from the final flip_state period (when mouse less engaged).
        If False, uses all data (default).
    """
    print("="*60)
    print("NEURAL POPULATION CLASSIFICATION DEMO")
    print("="*60)
    
    # Load session data
    print(f"\n1. Loading session data: {mouse_id}_{session_id}")
    try:
        session = SessionData(mouse_id, session_id, experiment, 
                            base_path=base_path, verbose=True)
    except Exception as e:
        print(f"Error loading session data: {e}")
        print("Using simulated data for demo...")
        return
    
    # Check if event column exists
    if event_column not in session.events.columns:
        print(f"Error: Column '{event_column}' not found in events data.")
        print(f"Available columns: {list(session.events.columns)}")
        return
    
    # Apply cluster filtering if specified
    if cluster_filter is not None:
        print(f"\n2. Filtering clusters with: {cluster_filter}")
        
        original_n_clusters = session.n_clusters
        session = session.filter_clusters(cluster_filter)
        
        print(f"  Original clusters: {original_n_clusters}")
        print(f"  Filtered clusters: {session.n_clusters}")
        
        if session.n_clusters == 0:
            print(f"Error: No clusters found matching: {cluster_filter}")
            return
    
    # Apply final flip_state exclusion if requested
    original_session = session
    if exclude_final_flip and event_column == 'flip_state':
        print(f"\n{3 if cluster_filter is not None else 2}. Excluding final flip_state period...")
        
        # Find the start of the final flip_state period
        flip_changes = np.where(np.diff(session.events['flip_state'].astype(int)))[0]
        if len(flip_changes) > 0:
            final_flip_start_idx = flip_changes[-1] + 1
            final_flip_start_time = session.events['timestamp_ms'].iloc[final_flip_start_idx]
            
            # Filter events to exclude final flip period
            mask = session.events['timestamp_ms'] < final_flip_start_time
            filtered_events = session.events[mask].copy()
            
            # Update session events in place (avoid copying cv2.VideoCapture)
            session.events = filtered_events
            
            print(f"  Excluded final flip period starting at {final_flip_start_time:.0f}ms")
            print(f"  Kept {len(filtered_events)}/{len(original_session.events)} events")
        else:
            print("  No flip_state changes found - using all data")
    
    # Prepare classification data
    step_num = 4 if exclude_final_flip else (3 if cluster_filter is not None else 2)
    print(f"\n{step_num}. Preparing classification data...")
    population_matrix, labels, time_bins = prepare_classification_data(
        session, event_column, bin_size_ms, centroid_y_threshold=centroid_y_threshold
    )
    
    # Check for class balance and diagnose potential issues
    print(f"\n=== Class Balance Diagnostics ===")
    print(f"Original events dataframe:")
    if event_column in session.events.columns:
        original_pos_rate = session.events[event_column].mean()
        print(f"  {event_column} positive rate: {original_pos_rate:.3f}")
        print(f"  {event_column} total events: {len(session.events)}")
    
    print(f"After time binning and filtering:")
    print(f"  Time bin positive rate: {np.mean(labels):.3f}")
    print(f"  Total time bins: {len(labels)}")
    print(f"  Positive time bins: {np.sum(labels)}")
    print(f"  Negative time bins: {len(labels) - np.sum(labels)}")
    
    if np.mean(labels) < 0.01 or np.mean(labels) > 0.99:
        print(f"Warning: Highly imbalanced classes (positive rate: {np.mean(labels):.3f})")
        print("Consider using different aggregation or event column")
    
    # Stratified temporal block split to maintain class balance while avoiding data leakage
    print(f"\n4. Stratified temporal block splitting (test size: {test_size})...")
    n_samples = len(population_matrix)
    
    # Create temporal blocks (e.g., 10 blocks) and sample from each
    n_blocks = 10
    block_size = n_samples // n_blocks
    
    # Initialize split arrays
    train_indices = []
    val_indices = []
    test_indices = []
    
    # For each temporal block, stratify the split
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = (block_idx + 1) * block_size if block_idx < n_blocks - 1 else n_samples
        
        block_samples = np.arange(start_idx, end_idx)
        block_labels = labels[start_idx:end_idx]
        
        # Skip blocks with no variation
        if len(np.unique(block_labels)) < 2:
            # All same class - distribute proportionally
            n_test = int(len(block_samples) * test_size)
            n_val = int(len(block_samples) * 0.16)  # 20% of remaining 80%
            
            test_indices.extend(block_samples[-n_test:])
            val_indices.extend(block_samples[-(n_test+n_val):-n_test])
            train_indices.extend(block_samples[:-(n_test+n_val)])
        else:
            # Stratified split within block
            from sklearn.model_selection import train_test_split
            
            block_train, block_temp = train_test_split(
                block_samples, test_size=test_size + 0.16, random_state=42, 
                stratify=block_labels
            )
            block_val, block_test = train_test_split(
                block_temp, test_size=test_size/(test_size + 0.16), random_state=42,
                stratify=block_labels[block_temp - start_idx]
            )
            
            train_indices.extend(block_train)
            val_indices.extend(block_val)
            test_indices.extend(block_test)
    
    # Create splits using indices
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    X_train = population_matrix[train_indices]
    y_train = labels[train_indices]
    
    X_val = population_matrix[val_indices] 
    y_val = labels[val_indices]
    
    X_test = population_matrix[test_indices]
    y_test = labels[test_indices]
    
    # Create datasets
    train_dataset = PopulationDataset(X_train, y_train)
    val_dataset = PopulationDataset(X_val, y_val)
    test_dataset = PopulationDataset(X_test, y_test)
    
    print(f"Stratified temporal block split:")
    print(f"  Training: {len(train_dataset)} samples, {np.mean(y_train):.3f} positive")
    print(f"  Validation: {len(val_dataset)} samples, {np.mean(y_val):.3f} positive") 
    print(f"  Test: {len(test_dataset)} samples, {np.mean(y_test):.3f} positive")
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples") 
    print(f"Test set: {len(test_dataset)} samples")
    
    # Plot timeline with data splits
    print(f"\nGenerating timeline visualization...")
    plot_data_splits_timeline(session, event_column, time_bins, 
                              train_indices, val_indices, test_indices, bin_size_ms, labels=labels)
    
    # Initialize classifier
    print(f"\n4. Initializing classifier...")
    n_neurons = population_matrix.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Input size: {n_neurons} neurons")
    
    classifier = PopulationClassifier(
        input_size=n_neurons,
        hidden_sizes=[16,8,8],
        dropout_rate=0.3,
        device=device
    )
    
    # Train model
    print(f"\n5. Training MLP model for {epochs} epochs...")
    history = classifier.train_model(
        train_dataset, val_dataset,
        epochs=epochs, batch_size=16, learning_rate=0.0001,
        verbose=True, use_best_model=use_best_model
    )
    
    # Evaluate model
    print(f"\n6. Evaluating MLP model on test set...")
    metrics, y_true, y_pred, y_prob = classifier.evaluate(test_dataset)
    
    print("Test Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    print("\n" + "="*60)
    print("MLP DEMO COMPLETE!")
    print("="*60)
    
    # Create comprehensive plots after training is complete
    print(f"\n7. Generating evaluation plots...")
    plot_classifier_results(history, y_true, y_pred, y_prob, "MLP Classifier")
    
    return classifier, metrics, history


if __name__ == "__main__":
    # Example usage - optimized for neural data with cluster filtering
    run_classification_demo(
        mouse_id="7004",
        session_id="m4", 
        experiment="clickbait-motivate",
        event_column="flip_state",
        base_path="S:\\",
        bin_size_ms=1000.0,                   # Coarser temporal resolution
        test_size=0.2,
        epochs=1000,                          # Training epochs
        centroid_y_threshold=np.inf,         # No position filtering
        use_best_model=True,                # Use final model by default
        #cluster_filter='best_channel < 17',  # Only use channels 17 and above
        exclude_final_flip=False              # Exclude final flip_state period
    )
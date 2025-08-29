"""
Demo script for LSTM-based neural population sequence classification.

This script demonstrates how to:
1. Load neural data using SessionData
2. Create population activity sequences 
3. Prepare behavioral event labels for sequences
4. Train an LSTM classifier
5. Evaluate model performance

Code by Nate Gonzales-Hess, August 2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
import torch

from sldata import SessionData
from neural_classifier import SequenceClassifier, SequenceDataset


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


def create_sequences(population_matrix: np.ndarray, labels: np.ndarray, 
                    sequence_length: int, stride: int = 1) -> tuple:
    """
    Create sliding window sequences from population data.
    
    Parameters:
    -----------
    population_matrix : np.ndarray
        Population activity matrix [time_bins x neurons]
    labels : np.ndarray
        Binary labels for each time bin
    sequence_length : int
        Length of each sequence (number of time bins)
    stride : int
        Stride between sequences (default: 1 for maximum overlap)
        
    Returns:
    --------
    sequences : np.ndarray
        Population sequences [n_sequences x sequence_length x n_neurons]
    sequence_labels : np.ndarray
        Labels for each sequence (label of final time bin)
    """
    n_time_bins, n_neurons = population_matrix.shape
    
    # Calculate number of sequences we can create
    n_sequences = (n_time_bins - sequence_length) // stride + 1
    
    # Initialize arrays
    sequences = np.zeros((n_sequences, sequence_length, n_neurons))
    sequence_labels = np.zeros(n_sequences)
    
    # Create sliding window sequences
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        
        # Population sequence
        sequences[i] = population_matrix[start_idx:end_idx]
        
        # Label is from the final time bin (what we're predicting)
        sequence_labels[i] = labels[end_idx - 1]
    
    print(f"Created {n_sequences} sequences of length {sequence_length}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Positive class proportion: {np.mean(sequence_labels):.3f}")
    
    # Debug: Show how sequence labels are derived
    print(f"Sequence labeling (uses final time bin of each sequence):")
    print(f"  Total time bins: {len(labels)}")
    print(f"  Time bin positive rate: {np.mean(labels):.3f}")
    print(f"  Sequence positive rate: {np.mean(sequence_labels):.3f}")
    
    return sequences, sequence_labels


def prepare_lstm_data(session: SessionData, event_column: str, 
                     sequence_length: int = 10,
                     bin_size_ms: float = 50.0, 
                     stride: int = 1,
                     min_time_ms: float = 0,
                     max_time_ms: float = None,
                     centroid_y_threshold: float = None) -> tuple:
    """
    Prepare sequence data for LSTM classification from SessionData.
    
    Parameters:
    -----------
    session : SessionData
        Loaded session data
    event_column : str
        Column name for the behavioral event to predict
    sequence_length : int
        Length of each sequence (number of time bins)
    bin_size_ms : float
        Time bin size in milliseconds
    stride : int
        Stride between sequences
    min_time_ms : float
        Start time for analysis
    max_time_ms : float
        End time for analysis (if None, uses session duration)
    centroid_y_threshold : float, optional
        Only include time bins where centroid_y <= threshold.
        If None, uses 50% of max centroid_y value.
        
    Returns:
    --------
    sequences : np.ndarray
        Population activity sequences [n_samples x sequence_length x n_neurons] - filtered by centroid_y
    sequence_labels : np.ndarray
        Binary labels for each sequence - filtered by centroid_y
    time_bins : np.ndarray
        Time bin centers - filtered by centroid_y
    """
    print(f"Preparing LSTM data for predicting '{event_column}'...")
    
    # Create population matrix
    if max_time_ms is None:
        max_time_ms = np.inf
    
    pop_matrix, time_bins, cluster_ids = session.create_population_raster(
        start_time=min_time_ms,
        end_time=max_time_ms,
        bin_size_ms=bin_size_ms,
        zscore_neurons=True
    )
    
    # Transpose to get [time_bins x neurons]
    population_matrix = pop_matrix.T
    
    # Get behavioral labels for each time bin
    labels = session.create_event_colormap(
        time_bins, 
        event_column, 
        aggregation='any'  # True if any event in time bin is True
    )
    
    print(f"Time binning results:")
    print(f"  Time bins created: {len(time_bins)}")
    print(f"  Time bin positive rate: {np.mean(labels):.3f}")
    print(f"  Time bin size: {bin_size_ms}ms")
    
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
    
    # Create sequences
    sequences, sequence_labels = create_sequences(
        population_matrix, labels, sequence_length, stride
    )
    
    print(f"Data shape: {population_matrix.shape}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f} ms")
    
    return sequences, sequence_labels, time_bins




def run_lstm_demo(mouse_id: str, session_id: str, experiment: str,
                 event_column: str = 'flip_state',
                 base_path: str = "S:\\",
                 sequence_length: int = 10,
                 bin_size_ms: float = 50.0,
                 stride: int = 5,
                 test_size: float = 0.2,
                 epochs: int = 50,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 centroid_y_threshold: float = None,
                 use_best_model: bool = False,
                 cluster_filter: str = None):
    """
    Run complete LSTM classification demo.
    
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
    sequence_length : int
        Length of input sequences
    bin_size_ms : float
        Time bin size in milliseconds
    stride : int
        Stride between sequences
    test_size : float
        Fraction of data for testing
    epochs : int
        Number of training epochs
    hidden_size : int
        LSTM hidden state size
    num_layers : int
        Number of LSTM layers
    bidirectional : bool
        Whether to use bidirectional LSTM
    use_best_model : bool
        If True, use the model with best validation loss for testing.
        If False, use the final model after all epochs (default).
    cluster_filter : str, optional
        Filter expression for clusters (same syntax as SessionData.filter_clusters()).
        If None, uses all clusters. Examples: 'best_channel >= 17', 'n_spikes > 100'.
    """
    print("="*60)
    print("LSTM NEURAL POPULATION CLASSIFICATION DEMO")
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
    
    # Prepare sequence data
    step_num = 3 if cluster_filter is not None else 2
    print(f"\n{step_num}. Preparing sequence data...")
    sequences, sequence_labels, time_bins = prepare_lstm_data(
        session, event_column, sequence_length, bin_size_ms, stride, 
        centroid_y_threshold=centroid_y_threshold
    )
    
    # Check for class balance and diagnose potential issues
    print(f"\n=== Class Balance Diagnostics ===")
    print(f"Original events dataframe:")
    if event_column in session.events.columns:
        original_pos_rate = session.events[event_column].mean()
        print(f"  {event_column} positive rate: {original_pos_rate:.3f}")
        print(f"  {event_column} total events: {len(session.events)}")
    
    print(f"After time binning and filtering:")
    print(f"  Sequence positive rate: {np.mean(sequence_labels):.3f}")
    print(f"  Total sequences: {len(sequence_labels)}")
    print(f"  Positive sequences: {np.sum(sequence_labels)}")
    print(f"  Negative sequences: {len(sequence_labels) - np.sum(sequence_labels)}")
    
    if np.mean(sequence_labels) < 0.01 or np.mean(sequence_labels) > 0.99:
        print(f"Warning: Highly imbalanced classes (positive rate: {np.mean(sequence_labels):.3f})")
        print("Consider using different aggregation or event column")
    
    # Stratified temporal block split to maintain class balance while avoiding data leakage
    print(f"\n4. Stratified temporal block splitting (test size: {test_size})...")
    n_sequences = len(sequences)
    
    # Create temporal blocks (e.g., 10 blocks) and sample from each
    n_blocks = 10
    block_size = n_sequences // n_blocks
    
    # Initialize split arrays
    train_indices = []
    val_indices = []
    test_indices = []
    
    # For each temporal block, stratify the split
    for block_idx in range(n_blocks):
        start_idx = block_idx * block_size
        end_idx = (block_idx + 1) * block_size if block_idx < n_blocks - 1 else n_sequences
        
        block_sequences = np.arange(start_idx, end_idx)
        block_labels = sequence_labels[start_idx:end_idx]
        
        # Skip blocks with no variation
        if len(np.unique(block_labels)) < 2:
            # All same class - distribute proportionally
            n_test = int(len(block_sequences) * test_size)
            n_val = int(len(block_sequences) * 0.16)  # 20% of remaining 80%
            
            test_indices.extend(block_sequences[-n_test:])
            val_indices.extend(block_sequences[-(n_test+n_val):-n_test])
            train_indices.extend(block_sequences[:-(n_test+n_val)])
        else:
            # Stratified split within block
            from sklearn.model_selection import train_test_split
            
            block_train, block_temp = train_test_split(
                block_sequences, test_size=test_size + 0.16, random_state=42, 
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
    
    X_train = sequences[train_indices]
    y_train = sequence_labels[train_indices]
    
    X_val = sequences[val_indices] 
    y_val = sequence_labels[val_indices]
    
    X_test = sequences[test_indices]
    y_test = sequence_labels[test_indices]
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)
    
    print(f"Stratified temporal block split:")
    print(f"  Training: {len(train_dataset)} sequences, {np.mean(y_train):.3f} positive")
    print(f"  Validation: {len(val_dataset)} sequences, {np.mean(y_val):.3f} positive") 
    print(f"  Test: {len(test_dataset)} sequences, {np.mean(y_test):.3f} positive")
    
    print(f"Training set: {len(train_dataset)} sequences")
    print(f"Validation set: {len(val_dataset)} sequences") 
    print(f"Test set: {len(test_dataset)} sequences")
    
    # Initialize LSTM classifier
    print(f"\n5. Initializing LSTM classifier...")
    n_neurons = sequences.shape[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Input size: {n_neurons} neurons")
    print(f"Sequence length: {sequence_length} time bins")
    print(f"Hidden size: {hidden_size}")
    print(f"Num layers: {num_layers}")
    print(f"Bidirectional: {bidirectional}")
    
    classifier = SequenceClassifier(
        input_size=n_neurons,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=0.2,
        bidirectional=bidirectional,
        device=device
    )
    
    # Train model
    print(f"\n6. Training LSTM model for {epochs} epochs...")
    history = classifier.train_model(
        train_dataset, val_dataset,
        epochs=epochs, batch_size=16, learning_rate=0.0001,
        verbose=True, use_best_model=use_best_model
    )
    
    # Evaluate model
    print(f"\n7. Evaluating LSTM model on test set...")
    metrics, y_true, y_pred, y_prob = classifier.evaluate(test_dataset)
    
    print("Test Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    print("\n" + "="*60)
    print("LSTM DEMO COMPLETE!")
    print("="*60)
    
    # Create comprehensive plots after training is complete
    print(f"\n8. Generating evaluation plots...")
    plot_classifier_results(history, y_true, y_pred, y_prob, "LSTM Classifier")
    
    return classifier, metrics, history


if __name__ == "__main__":
    # Example usage - optimized for neural data with cluster filtering
    run_lstm_demo(
        mouse_id="7004",
        session_id="m4", 
        experiment="clickbait-motivate",
        event_column="flip_state",
        base_path="S:\\",
        sequence_length=2,           # Number of time bins per sequence
        bin_size_ms=500,             # Coarser temporal resolution  
        stride=1,                    # Dense sampling
        test_size=0.2,
        epochs=250,                  # Training epochs
        hidden_size=64,              # Small hidden size 
        num_layers=3,                # Depth of network
        bidirectional=False,
        centroid_y_threshold=np.inf,
        use_best_model=True,         # Use best validation model
        cluster_filter='best_channel < 17'  # Only use channels 17 and above
    )
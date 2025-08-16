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
import torch

from sldata import SessionData
from neural_classifier import SequenceClassifier, SequenceDataset


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
    
    return sequences, sequence_labels


def prepare_lstm_data(session: SessionData, event_column: str, 
                     sequence_length: int = 10,
                     bin_size_ms: float = 50.0, 
                     stride: int = 1,
                     min_time_ms: float = 0,
                     max_time_ms: float = None) -> tuple:
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
        
    Returns:
    --------
    sequences : np.ndarray
        Population activity sequences [n_samples x sequence_length x n_neurons]
    sequence_labels : np.ndarray
        Binary labels for each sequence
    time_bins : np.ndarray
        Time bin centers
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
    
    # Create sequences
    sequences, sequence_labels = create_sequences(
        population_matrix, labels, sequence_length, stride
    )
    
    print(f"Data shape: {population_matrix.shape}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f} ms")
    
    return sequences, sequence_labels, time_bins


def plot_sequence_examples(sequences: np.ndarray, labels: np.ndarray, 
                          time_bins: np.ndarray, n_examples: int = 3):
    """Plot example sequences for visualization."""
    fig, axes = plt.subplots(n_examples, 2, figsize=(15, 4*n_examples))
    
    # Find examples with different labels
    true_indices = np.where(labels == 1)[0]
    false_indices = np.where(labels == 0)[0]
    
    for i in range(n_examples):
        # True example
        if len(true_indices) > i:
            true_idx = true_indices[i]
            im1 = axes[i, 0].imshow(sequences[true_idx].T, aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'True Label Example {i+1}')
            axes[i, 0].set_xlabel('Time Bins')
            axes[i, 0].set_ylabel('Neurons')
            plt.colorbar(im1, ax=axes[i, 0])
        
        # False example
        if len(false_indices) > i:
            false_idx = false_indices[i]
            im2 = axes[i, 1].imshow(sequences[false_idx].T, aspect='auto', cmap='viridis')
            axes[i, 1].set_title(f'False Label Example {i+1}')
            axes[i, 1].set_xlabel('Time Bins')
            axes[i, 1].set_ylabel('Neurons')
            plt.colorbar(im2, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.show()


def plot_lstm_training_history(history: dict):
    """Plot LSTM training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('LSTM Training History')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    if 'val_acc' in history and len(history['val_acc']) > 0:
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('LSTM Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, 'No validation data', 
                    transform=axes[1].transAxes, ha='center', va='center')
        axes[1].set_title('Validation Accuracy')
    
    plt.tight_layout()
    plt.show()


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
                 bidirectional: bool = False):
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
    
    # Prepare sequence data
    print(f"\n2. Preparing sequence data...")
    sequences, sequence_labels, time_bins = prepare_lstm_data(
        session, event_column, sequence_length, bin_size_ms, stride
    )
    
    # Check for class balance
    if np.mean(sequence_labels) < 0.01 or np.mean(sequence_labels) > 0.99:
        print(f"Warning: Highly imbalanced classes (positive rate: {np.mean(sequence_labels):.3f})")
        print("Consider using different aggregation or event column")
    
    # Plot example sequences
    print(f"\n3. Plotting example sequences...")
    plot_sequence_examples(sequences, sequence_labels, time_bins)
    
    # Split data
    print(f"\n4. Splitting data (test size: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, sequence_labels, test_size=test_size, 
        random_state=42, stratify=sequence_labels
    )
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)
    
    # Split training data for validation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    train_dataset = SequenceDataset(X_train_sub, y_train_sub)
    val_dataset = SequenceDataset(X_val, y_val)
    
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
        epochs=epochs, batch_size=16, learning_rate=0.001,
        verbose=True
    )
    
    # Plot training history
    print(f"\n7. Plotting training history...")
    plot_lstm_training_history(history)
    
    # Evaluate model
    print(f"\n8. Evaluating LSTM model on test set...")
    metrics, y_true, y_pred, y_prob = classifier.evaluate(test_dataset)
    
    print("Test Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    # Plot some predictions
    print(f"\n9. Plotting prediction examples...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    # Show 4 random test examples
    random_indices = np.random.choice(len(y_true), 4, replace=False)
    
    for i, idx in enumerate(random_indices):
        # Get the corresponding sequence
        test_seq = X_test[idx]
        true_label = y_true[idx]
        pred_prob = y_prob[idx]
        
        # Plot the sequence
        im = axes[i].imshow(test_seq.T, aspect='auto', cmap='viridis')
        axes[i].set_title(f'True: {int(true_label)}, Pred: {pred_prob:.3f}')
        axes[i].set_xlabel('Time Bins')
        axes[i].set_ylabel('Neurons')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("LSTM DEMO COMPLETE!")
    print("="*60)
    
    return classifier, metrics, history


if __name__ == "__main__":
    # Example usage - modify these parameters for your data
    run_lstm_demo(
        mouse_id="7003",
        session_id="m4", 
        experiment="clickbait-motivate",
        event_column="drinking",
        base_path="S:\\",
        sequence_length=10,  # 10 time bins per sequence
        bin_size_ms=100.0,   # 100ms bins
        stride=5,            # 5 bin stride (50% overlap)
        test_size=0.2,
        epochs=500,
        hidden_size=64,
        num_layers=2,
        bidirectional=False
    )
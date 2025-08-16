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
import torch

from sldata import SessionData
from neural_classifier import PopulationClassifier, PopulationDataset


def prepare_classification_data(session: SessionData, event_column: str, 
                              bin_size_ms: float = 50.0, 
                              min_time_ms: float = 0,
                              max_time_ms: float = None) -> tuple:
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
        
    Returns:
    --------
    population_matrix : np.ndarray
        Population activity matrix [time_bins x neurons]
    labels : np.ndarray
        Binary labels for each time bin
    time_bins : np.ndarray
        Time bin centers
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
    
    print(f"Data shape: {population_matrix.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Positive class proportion: {np.mean(labels):.3f}")
    print(f"Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f} ms")
    
    return population_matrix, labels, time_bins


def plot_training_history(history: dict):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    if 'val_acc' in history and len(history['val_acc']) > 0:
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, 'No validation data', 
                    transform=axes[1].transAxes, ha='center', va='center')
        axes[1].set_title('Validation Accuracy')
    
    plt.tight_layout()
    plt.show()


def plot_classification_results(true_labels: np.ndarray, predicted_labels: np.ndarray, 
                              probabilities: np.ndarray, time_bins: np.ndarray):
    """Plot classification results over time."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))
    
    # True labels
    axes[0].plot(time_bins / 1000, true_labels, 'k-', alpha=0.7, label='True Labels')
    axes[0].set_ylabel('True State')
    axes[0].set_title('Classification Results Over Time')
    axes[0].grid(True)
    axes[0].legend()
    
    # Predicted probabilities
    axes[1].plot(time_bins / 1000, probabilities, 'b-', alpha=0.7, label='Predicted Probability')
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Threshold')
    axes[1].set_ylabel('Probability')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
    axes[1].legend()
    
    # Comparison
    axes[2].plot(time_bins / 1000, true_labels, 'k-', alpha=0.7, label='True')
    axes[2].plot(time_bins / 1000, predicted_labels, 'r-', alpha=0.7, label='Predicted')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('State')
    axes[2].set_title('True vs Predicted')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


def run_classification_demo(mouse_id: str, session_id: str, experiment: str,
                          event_column: str = 'flip_state',
                          base_path: str = "S:\\",
                          bin_size_ms: float = 50.0,
                          test_size: float = 0.2,
                          epochs: int = 100):
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
    
    # Prepare classification data
    print(f"\n2. Preparing classification data...")
    population_matrix, labels, time_bins = prepare_classification_data(
        session, event_column, bin_size_ms
    )
    
    # Check for class balance
    if np.mean(labels) < 0.01 or np.mean(labels) > 0.99:
        print(f"Warning: Highly imbalanced classes (positive rate: {np.mean(labels):.3f})")
        print("Consider using different aggregation or event column")
    
    # Split data
    print(f"\n3. Splitting data (test size: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        population_matrix, labels, test_size=test_size, 
        random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = PopulationDataset(X_train, y_train)
    test_dataset = PopulationDataset(X_test, y_test)
    
    # Split training data for validation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    train_dataset = PopulationDataset(X_train_sub, y_train_sub)
    val_dataset = PopulationDataset(X_val, y_val)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples") 
    print(f"Test set: {len(test_dataset)} samples")
    
    # Initialize classifier
    print(f"\n4. Initializing classifier...")
    n_neurons = population_matrix.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Input size: {n_neurons} neurons")
    
    classifier = PopulationClassifier(
        input_size=n_neurons,
        hidden_sizes=[128, 64],
        dropout_rate=0.2,
        device=device
    )
    
    # Train model
    print(f"\n5. Training model for {epochs} epochs...")
    history = classifier.train_model(
        train_dataset, val_dataset,
        epochs=epochs, batch_size=32, learning_rate=0.001,
        verbose=True
    )
    
    # Plot training history
    print(f"\n6. Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print(f"\n7. Evaluating model on test set...")
    metrics, y_true, y_pred, y_prob = classifier.evaluate(test_dataset)
    
    print("Test Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    # Plot results for test set
    test_indices = np.arange(len(X_test))
    test_time_bins = time_bins[test_indices]  # This is approximate
    
    print(f"\n8. Plotting classification results...")
    plot_classification_results(y_true, y_pred, y_prob, test_time_bins)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    
    return classifier, metrics, history


if __name__ == "__main__":
    # Example usage - modify these parameters for your data
    run_classification_demo(
        mouse_id="7003",
        session_id="m4", 
        experiment="clickbait-motivate",
        event_column="click",
        base_path="S:\\",
        bin_size_ms=100.0,
        test_size=0.2,
        epochs=1000
    )
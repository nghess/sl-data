"""
Demo script for LSTM-based neural population sequence classification with k-fold cross validation.

This script demonstrates how to:
1. Load neural data using SessionData
2. Create population activity sequences
3. Prepare behavioral event labels for sequences
4. Train an LSTM classifier with k-fold cross validation
5. Evaluate model performance across folds

Code by Nate Gonzales-Hess, August 2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import copy
import pandas as pd
import os
from pathlib import Path

from sldata import SessionData
from neural_classifier import SequenceClassifier, SequenceDataset


def plot_classifier_results(history: dict, y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray, model_name: str = "Classifier",
                           mouse_id: str = "", session_id: str = "", event_column: str = "",
                           fold_num: int = None, text_size_scale: float = 1.0, show_legends: bool = True,
                           save_path: str = None):
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
    fold_num : int, optional
        Fold number for display (if using k-fold)
    text_size_scale : float
        Scale factor for text sizes (default: 1.0)
    show_legends : bool
        Whether to show legends on plots (default: True)
    save_path : str, optional
        Path to save the figure. If None, displays instead.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    fold_text = f" (Fold {fold_num})" if fold_num is not None else ""

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

    axes[0].set_xlabel('Epoch', fontsize=12*text_size_scale)
    axes[0].set_ylabel('Loss', fontsize=12*text_size_scale)
    axes[0].set_title(f"{mouse_id}-{session_id} {model_name} Predicting '{event_column}'{fold_text}", fontsize=14*text_size_scale)
    if show_legends:
        axes[0].legend(fontsize=10*text_size_scale)
    axes[0].tick_params(labelsize=10*text_size_scale)
    axes[0].grid(False)

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

    sns.heatmap(cm_percentages, annot=annotations, fmt='', cmap='Blues', ax=axes[1], cbar=False, annot_kws={'size': 10*text_size_scale})
    axes[1].set_xlabel('Predicted', fontsize=12*text_size_scale)
    axes[1].set_ylabel('True', fontsize=12*text_size_scale)
    axes[1].set_title(f'Best Model - Confusion Matrix{fold_text}', fontsize=14*text_size_scale)
    axes[1].tick_params(labelsize=10*text_size_scale)
    axes[1].set_aspect('equal')

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
    axes[2].set_xlabel('Recall (TP/(TP+FN))', fontsize=12*text_size_scale)
    axes[2].set_ylabel('Precision (TP/(TP+FP))', fontsize=12*text_size_scale)
    axes[2].set_title('Precision-Recall Curve', fontsize=14*text_size_scale)
    if show_legends:
        axes[2].legend(loc="lower left", fontsize=10*text_size_scale)
    axes[2].tick_params(labelsize=10*text_size_scale)
    axes[2].grid(False)
    axes[2].set_aspect('equal')

    # 4. ROC Curve
    from sklearn.metrics import roc_curve, auc

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    axes[3].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[3].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random classifier (AUC = 0.500)')
    axes[3].set_xlim([0.0, 1.0])
    axes[3].set_ylim([0.0, 1.05])
    axes[3].set_xlabel('False Positive Rate (1-Specificity)', fontsize=12*text_size_scale)
    axes[3].set_ylabel('True Positive Rate (Sensitivity)', fontsize=12*text_size_scale)
    axes[3].set_title('ROC Curve', fontsize=14*text_size_scale)
    if show_legends:
        axes[3].legend(loc="lower right", fontsize=10*text_size_scale)
    axes[3].tick_params(labelsize=10*text_size_scale)
    axes[3].grid(False)
    axes[3].set_aspect('equal')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()

    return fig


def plot_temporal_predictions(session: SessionData, event_column: str, time_bins: np.ndarray,
                             y_true_bins: np.ndarray, y_pred_bins: np.ndarray, y_prob_bins: np.ndarray,
                             mouse_id: str = "", session_id: str = "", model_name: str = "Classifier",
                             show_trial_boundaries: bool = False, threshold: float = 0.5,
                             text_size_scale: float = 1.0, show_legends: bool = True, save_path: str = None):
    """
    Plot temporal predictions across the session timeline.

    Shows three subplots:
    1. True flip state over time
    2. Predicted binary classifications over time
    3. Prediction confidence/probability over time

    Parameters:
    -----------
    session : SessionData
        Session data containing events
    event_column : str
        Column name for the behavioral event
    time_bins : np.ndarray
        Time bin centers used for analysis
    y_true_bins : np.ndarray
        True labels for each time bin
    y_pred_bins : np.ndarray
        Predicted labels for each time bin
    y_prob_bins : np.ndarray
        Prediction probabilities for each time bin
    mouse_id : str
        Mouse identifier for plot title
    session_id : str
        Session identifier for plot title
    model_name : str
        Model name for plot title
    show_trial_boundaries : bool
        Whether to show trial boundaries as vertical lines
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Convert to seconds for plotting
    event_times_s = session.events['timestamp_ms'] / 1000
    time_bins_s = time_bins / 1000

    # 1. True flip state over time
    event_values = session.events[event_column].astype(int)
    ax1.fill_between(event_times_s, 0, event_values, alpha=0.7,
                     color='blue' if event_values.iloc[0] else 'lightgray',
                     step='pre', label=f'True {event_column}')

    # Add color changes for flip states
    flip_changes = np.where(np.diff(event_values))[0]
    for i, change_idx in enumerate(flip_changes):
        change_time = event_times_s.iloc[change_idx]
        next_value = event_values.iloc[change_idx + 1]
        color = 'blue' if next_value else 'lightgray'

        # Fill from this change to the next (or end)
        end_idx = flip_changes[i + 1] if i + 1 < len(flip_changes) else len(event_times_s) - 1
        end_time = event_times_s.iloc[end_idx]

        ax1.fill_between(event_times_s.iloc[change_idx:end_idx + 1], 0,
                        event_values.iloc[change_idx:end_idx + 1],
                        alpha=0.7, color=color, step='pre')

    ax1.set_ylabel(f'True {event_column}', fontsize=12*text_size_scale)
    ax1.set_title(f"{mouse_id}-{session_id} {model_name} Predictions", fontsize=14*text_size_scale)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.tick_params(labelsize=10*text_size_scale)
    ax1.grid(True, alpha=0.3)

    # 2. Predicted binary classifications over time
    # Create step plot for binary predictions
    correct_mask = y_true_bins == y_pred_bins
    incorrect_mask = ~correct_mask

    # Plot correct predictions
    if np.any(correct_mask):
        ax2.scatter(time_bins_s[correct_mask], y_pred_bins[correct_mask],
                   c='green', marker='|', s=30, alpha=0.8,
                   label=f'Correct ({np.sum(correct_mask)}/{len(y_pred_bins)})')

    # Plot incorrect predictions
    if np.any(incorrect_mask):
        ax2.scatter(time_bins_s[incorrect_mask], y_pred_bins[incorrect_mask],
                   c='red', marker='|', s=30, alpha=0.8,
                   label=f'Incorrect ({np.sum(incorrect_mask)}/{len(y_pred_bins)})')

    ax2.set_ylabel('Predicted Class', fontsize=12*text_size_scale)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    if show_legends:
        ax2.legend(loc='upper right', fontsize=10*text_size_scale)
    ax2.tick_params(labelsize=10*text_size_scale)
    ax2.grid(True, alpha=0.3)

    # 3. Prediction confidence over time
    # Color by correctness
    if np.any(correct_mask):
        ax3.scatter(time_bins_s[correct_mask], y_prob_bins[correct_mask],
                   c='green', marker='o', s=10, alpha=0.6,
                   label='Correct Predictions')

    if np.any(incorrect_mask):
        ax3.scatter(time_bins_s[incorrect_mask], y_prob_bins[incorrect_mask],
                   c='red', marker='o', s=10, alpha=0.6,
                   label='Incorrect Predictions')

    # Add decision threshold line
    ax3.axhline(y=threshold, color='black', linestyle='--', alpha=0.5,
               label=f'Decision Threshold ({threshold:.3f})')

    ax3.set_ylabel('Prediction Probability', fontsize=12*text_size_scale)
    ax3.set_xlabel('Time (s)', fontsize=12*text_size_scale)
    ax3.set_ylim(0, 1)
    if show_legends:
        ax3.legend(loc='upper right', fontsize=10*text_size_scale)
    ax3.tick_params(labelsize=10*text_size_scale)
    ax3.grid(True, alpha=0.3)

    # Add trial boundaries if requested
    if show_trial_boundaries and 'trial_number' in session.events.columns:
        # Find trial boundaries (where trial number changes)
        trial_numbers = session.events['trial_number'].fillna(-1)  # Fill NaN with -1
        trial_changes = np.where(np.diff(trial_numbers))[0]

        # Add vertical lines at trial boundaries to all subplots
        for change_idx in trial_changes:
            boundary_time = event_times_s.iloc[change_idx + 1]  # Start of new trial

            ax1.axvline(x=boundary_time, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax2.axvline(x=boundary_time, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax3.axvline(x=boundary_time, color='red', linestyle=':', alpha=0.5, linewidth=1)

        # Add to legend (just once)
        if len(trial_changes) > 0:
            ax1.axvline(x=-1, color='red', linestyle=':', alpha=0.5, linewidth=1,
                       label=f'Trial boundaries ({len(trial_changes)} changes)')
            if show_legends:
                ax1.legend(loc='upper right')

    # Calculate and display accuracy
    accuracy = np.mean(correct_mask)
    plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f} ({np.sum(correct_mask)}/{len(y_pred_bins)} bins)',
                fontsize=10*text_size_scale, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Set white background
    fig.patch.set_facecolor('white')
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('white')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()

    return fig


def plot_data_splits_timeline(session: SessionData, event_column: str, time_bins: np.ndarray,
                              folds: list, sequence_labels: np.ndarray,
                              sequence_length: int, stride: int,
                              mouse_id: str = "", session_id: str = "",
                              text_size_scale: float = 1.0, show_legends: bool = True,
                              save_path: str = None):
    """
    Plot timeline showing event_column over time with k-fold split overlays.

    Parameters:
    -----------
    session : SessionData
        Session data containing events
    event_column : str
        Column name for the behavioral event
    time_bins : np.ndarray
        Time bin centers used for analysis
    folds : list
        List of fold dictionaries from create_kfold_splits
    sequence_labels : np.ndarray
        Labels for each sequence
    sequence_length : int
        Length of sequences
    stride : int
        Stride between sequences
    mouse_id : str
        Mouse identifier
    session_id : str
        Session identifier
    text_size_scale : float
        Scale factor for text sizes
    show_legends : bool
        Whether to show legends
    save_path : str, optional
        Path to save the figure. If None, displays instead.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Convert event times to seconds for plotting
    event_times_s = session.events['timestamp_ms'] / 1000
    time_bins_s = time_bins / 1000

    # Plot 1: Event state over time
    event_values = session.events[event_column].astype(int)
    ax1.fill_between(event_times_s, 0, event_values, alpha=0.7,
                     color='blue' if event_values.iloc[0] else None,
                     step='pre', label=f'{event_column}')

    # Add color changes for flip states
    flip_changes = np.where(np.diff(event_values))[0]
    for i, change_idx in enumerate(flip_changes):
        change_time = event_times_s.iloc[change_idx]
        next_value = event_values.iloc[change_idx + 1]
        color = 'blue' if next_value else None

        # Fill from this change to the next (or end)
        end_idx = flip_changes[i + 1] if i + 1 < len(flip_changes) else len(event_times_s) - 1
        end_time = event_times_s.iloc[end_idx]

        ax1.fill_between(event_times_s.iloc[change_idx:end_idx + 1], 0,
                        event_values.iloc[change_idx:end_idx + 1],
                        alpha=0.7, color=color, step='pre')

    ax1.set_ylabel(f'{event_column}', fontsize=12*text_size_scale)
    ax1.set_title(f"{mouse_id}-{session_id} '{event_column}' State and K-Fold Splits", fontsize=14*text_size_scale)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([])
    ax1.tick_params(labelsize=10*text_size_scale)
    ax1.grid(False)

    # Plot 2: K-fold splits overlay
    # Collect all test sequences from all folds
    colors = plt.cm.tab10(np.linspace(0, 1, len(folds)))
    height_offset = 0
    marker_size = 1
    start_height = 0.1
    marker_alpha = 1.0

    for fold_idx, fold_info in enumerate(folds):
        test_indices = fold_info['test_indices']
        fold_num = fold_info['fold_num']

        # Convert sequence indices to time bins
        test_times = []
        for seq_idx in test_indices:
            start_bin = seq_idx * stride
            end_bin = start_bin + sequence_length
            for bin_idx in range(start_bin, min(end_bin, len(time_bins))):
                test_times.append(time_bins_s[bin_idx])

        if test_times:
            ax2.scatter(test_times, [start_height + height_offset] * len(test_times),
                       c=[colors[fold_idx]], marker='|', s=marker_size, alpha=marker_alpha,
                       label=f'Fold {fold_num} Test ({len(test_indices)} seqs)')
        height_offset += 0.08

    ax2.set_xlabel('Time (s)', fontsize=12*text_size_scale)
    ax2.set_ylabel('Test Folds', fontsize=12*text_size_scale)
    ax2.set_ylim(0, height_offset + 2*start_height)
    ax2.set_yticks([])
    if show_legends:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10*text_size_scale)
    ax2.tick_params(labelsize=10*text_size_scale)
    ax2.grid(False)

    # Set white background for entire figure
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot to {save_path}")
    else:
        plt.show()

    return fig


def create_kfold_splits(sequence_labels: np.ndarray, time_bins: np.ndarray,
                       event_column_values: np.ndarray, sequence_length: int, stride: int,
                       n_folds: int = 10, random_state: int = 42) -> list:
    """
    Create k-fold cross validation splits that respect event_column state boundaries.

    This function ensures that all time bins within a sequence belong to the same
    event_column state (e.g., all bins are either in flip_state=True or flip_state=False,
    never mixed). Sequences that cross state boundaries are excluded from all folds.

    Parameters:
    -----------
    sequence_labels : np.ndarray
        Labels for each sequence
    time_bins : np.ndarray
        Time bin centers used for analysis
    event_column_values : np.ndarray
        Binary event values for each time bin (e.g., flip_state at each bin)
    sequence_length : int
        Length of each sequence (number of time bins)
    stride : int
        Stride between sequences
    n_folds : int
        Number of folds for cross validation (default: 10)
    random_state : int
        Random seed for reproducible splits

    Returns:
    --------
    folds : list of dict
        List of fold dictionaries, each containing:
        - 'train_indices': sequence indices for training
        - 'val_indices': sequence indices for validation
        - 'test_indices': sequence indices for testing
        - 'fold_num': fold number (0 to n_folds-1)
    """
    np.random.seed(random_state)

    n_sequences = len(sequence_labels)

    # First, identify sequences that cross event boundaries
    # These will be excluded to maintain clean event_column states
    valid_sequences = []
    boundary_crossing_sequences = []

    for seq_idx in range(n_sequences):
        start_bin = seq_idx * stride
        end_bin = start_bin + sequence_length

        if end_bin <= len(event_column_values):
            # Get event values for all bins in this sequence
            seq_event_values = event_column_values[start_bin:end_bin]

            # Check if all bins have the same event state
            if len(np.unique(seq_event_values)) == 1:
                valid_sequences.append(seq_idx)
            else:
                boundary_crossing_sequences.append(seq_idx)

    valid_sequences = np.array(valid_sequences)
    n_valid = len(valid_sequences)

    print(f"\nK-Fold Split Configuration:")
    print(f"  Total sequences: {n_sequences}")
    print(f"  Valid sequences (no boundary crossing): {n_valid}")
    print(f"  Boundary-crossing sequences (excluded): {len(boundary_crossing_sequences)}")
    print(f"  Number of folds: {n_folds}")

    if n_valid < n_folds:
        raise ValueError(f"Not enough valid sequences ({n_valid}) for {n_folds} folds")

    # Shuffle valid sequences
    shuffled_indices = np.random.permutation(valid_sequences)

    # Calculate fold sizes
    base_fold_size = n_valid // n_folds
    remainder = n_valid % n_folds

    # Create folds
    folds = []
    current_idx = 0

    for fold_num in range(n_folds):
        # Calculate this fold's size (distribute remainder across first folds)
        fold_size = base_fold_size + (1 if fold_num < remainder else 0)

        # Extract test indices for this fold
        test_indices = shuffled_indices[current_idx:current_idx + fold_size]

        # All other sequences are for training/validation
        train_val_indices = np.concatenate([
            shuffled_indices[:current_idx],
            shuffled_indices[current_idx + fold_size:]
        ])

        # Split train_val into train and validation (80/20 split)
        n_train_val = len(train_val_indices)
        n_val = max(1, n_train_val // 5)  # 20% for validation

        # Shuffle train_val for random split
        np.random.shuffle(train_val_indices)
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]

        # Store fold information
        folds.append({
            'fold_num': fold_num,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        })

        # Calculate class balance for this fold
        train_labels = sequence_labels[train_indices]
        val_labels = sequence_labels[val_indices]
        test_labels = sequence_labels[test_indices]

        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(train_indices)} sequences ({np.mean(train_labels):.3f} positive)")
        print(f"  Val:   {len(val_indices)} sequences ({np.mean(val_labels):.3f} positive)")
        print(f"  Test:  {len(test_indices)} sequences ({np.mean(test_labels):.3f} positive)")

        current_idx += fold_size

    return folds


def create_sequences(population_matrix: np.ndarray, labels: np.ndarray,
                    sequence_length: int, stride: int = 1) -> tuple:
    """
    Create sliding window sequences from population data.

    Note: This version does NOT filter boundary-crossing sequences.
    That filtering is done in create_kfold_splits() to maintain consistency.

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

    # Calculate maximum number of sequences we can create
    max_sequences = (n_time_bins - sequence_length) // stride + 1

    # Collect all sequences (filtering happens in k-fold split)
    sequences = []
    sequence_labels_list = []

    for i in range(max_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length

        # Population sequence
        sequence = population_matrix[start_idx:end_idx]
        sequences.append(sequence)

        # Label is from the final time bin (what we're predicting)
        sequence_labels_list.append(labels[end_idx - 1])

    # Convert to arrays
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels_list)

    print(f"\nCreated {len(sequences)} sequences of length {sequence_length}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Positive class proportion: {np.mean(sequence_labels):.3f}")

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
        Population activity sequences [n_samples x sequence_length x n_neurons]
    sequence_labels : np.ndarray
        Binary labels for each sequence
    time_bins : np.ndarray
        Time bin centers
    event_column_values : np.ndarray
        Event values for each time bin (needed for boundary detection)
    """
    print(f"\nPreparing LSTM data for predicting '{event_column}'...")

    # Get event timestamp bounds for cropping
    event_min_time = session.events['timestamp_ms'].min()
    event_max_time = session.events['timestamp_ms'].max()

    print(f"Event timestamp bounds: {event_min_time:.1f}ms - {event_max_time:.1f}ms")

    # Create population matrix
    if max_time_ms is None:
        max_time_ms = np.inf

    pop_matrix, time_bins, cluster_ids = session.create_population_raster(
        start_time=min_time_ms,
        end_time=max_time_ms,
        bin_size_ms=bin_size_ms,
        zscore_neurons=False
    )

    # Crop time bins to event timestamp bounds
    crop_mask = (time_bins >= event_min_time) & (time_bins <= event_max_time)
    pop_matrix = pop_matrix[:, crop_mask]
    time_bins = time_bins[crop_mask]

    print(f"Cropped time bins to event bounds: {len(time_bins)} bins remaining")

    # Transpose to get [time_bins x neurons]
    population_matrix = pop_matrix.T

    # Get behavioral labels for each time bin
    labels = session.create_event_colormap(
        time_bins,
        event_column,
        aggregation='any'
    )

    # Get event values for boundary detection
    event_column_values = session.create_event_colormap(
        time_bins,
        event_column,
        aggregation='any'
    )

    print(f"\nTime binning results:")
    print(f"  Time bins created: {len(time_bins)}")
    print(f"  Time bin positive rate: {np.mean(labels):.3f}")
    print(f"  Time bin size: {bin_size_ms}ms")

    # Apply centroid_y filtering if requested
    if 'bonsai_centroid_y' in session.events.columns:
        # Get centroid_y values for each time bin
        centroid_y_values = session.create_event_colormap(
            time_bins,
            'bonsai_centroid_y',
            aggregation='mean'
        )

        # Determine threshold
        if centroid_y_threshold is None:
            max_centroid_y = session.events['bonsai_centroid_y'].max()
            centroid_y_threshold = max_centroid_y * 0.5

        # Create mask for time bins below threshold
        centroid_mask = centroid_y_values <= centroid_y_threshold

        # Apply filtering
        population_matrix = population_matrix[centroid_mask]
        labels = labels[centroid_mask]
        event_column_values = event_column_values[centroid_mask]
        time_bins = time_bins[centroid_mask]

        print(f"Applied centroid_y filter (≤{centroid_y_threshold:.1f}): "
              f"kept {np.sum(centroid_mask)}/{len(centroid_mask)} time bins")
    else:
        print("Warning: No 'bonsai_centroid_y' column found - skipping position filtering")

    # Create sequences
    sequences, sequence_labels = create_sequences(
        population_matrix, labels, sequence_length, stride
    )

    print(f"\nData shape: {population_matrix.shape}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f} ms")

    return sequences, sequence_labels, time_bins, event_column_values


def run_lstm_kfold_demo(mouse_id: str, session_id: str, experiment: str,
                        event_column: str = 'flip_state',
                        base_path: str = "S:\\",
                        sequence_length: int = 10,
                        bin_size_ms: float = 50.0,
                        stride: int = 5,
                        n_folds: int = 10,
                        epochs: int = 50,
                        hidden_size: int = 64,
                        num_layers: int = 2,
                        bidirectional: bool = False,
                        centroid_y_threshold: float = None,
                        use_best_model: bool = False,
                        cluster_filter: str = None,
                        exclude_final_flip: bool = False,
                        random_state: int = 42,
                        text_size_scale: float = 1.0,
                        show_legends: bool = True,
                        plot_each_fold: bool = False):
    """
    Run complete LSTM classification demo with k-fold cross validation.

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
    n_folds : int
        Number of folds for cross validation (default: 10)
    epochs : int
        Number of training epochs
    hidden_size : int
        LSTM hidden state size
    num_layers : int
        Number of LSTM layers
    bidirectional : bool
        Whether to use bidirectional LSTM
    use_best_model : bool
        If True, use the model with best validation loss
    cluster_filter : str, optional
        Filter expression for clusters
    exclude_final_flip : bool
        If True, excludes data from the final flip_state period
    random_state : int
        Random seed for reproducible splits (default: 42)
    plot_each_fold : bool
        If True, plot results for each fold (default: False)
    """
    print("="*60)
    print("LSTM K-FOLD CROSS VALIDATION DEMO")
    print("="*60)

    # Load session data
    print(f"\n1. Loading session data: {mouse_id}_{session_id}")
    try:
        session = SessionData(mouse_id, session_id, experiment,
                            base_path=base_path, verbose=True)
    except Exception as e:
        print(f"Error loading session data: {e}")
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
    if exclude_final_flip and event_column == 'flip_state':
        print(f"\n{3 if cluster_filter is not None else 2}. Excluding final flip_state period...")

        flip_changes = np.where(np.diff(session.events['flip_state'].astype(int)))[0]
        if len(flip_changes) > 0:
            final_flip_start_idx = flip_changes[-1] + 1
            final_flip_start_time = session.events['timestamp_ms'].iloc[final_flip_start_idx]

            mask = session.events['timestamp_ms'] < final_flip_start_time
            filtered_events = session.events[mask].copy()
            session.events = filtered_events

            print(f"  Excluded final flip period starting at {final_flip_start_time:.0f}ms")
            print(f"  Kept {len(filtered_events)} events")

    # Prepare sequence data
    step_num = 4 if exclude_final_flip else (3 if cluster_filter is not None else 2)
    print(f"\n{step_num}. Preparing sequence data...")
    sequences, sequence_labels, time_bins, event_column_values = prepare_lstm_data(
        session, event_column, sequence_length, bin_size_ms, stride,
        centroid_y_threshold=centroid_y_threshold
    )

    # Create k-fold splits
    step_num += 1
    print(f"\n{step_num}. Creating {n_folds}-fold cross validation splits...")
    folds = create_kfold_splits(
        sequence_labels, time_bins, event_column_values,
        sequence_length, stride, n_folds, random_state
    )

    # Create debug plots directory
    debug_dir = Path("debug_plots")
    debug_dir.mkdir(exist_ok=True)
    session_dir = debug_dir / f"{mouse_id}_{session_id}"
    session_dir.mkdir(exist_ok=True)
    print(f"\nDebug plots will be saved to: {session_dir}")

    # Save k-fold splits timeline plot
    print(f"\nGenerating k-fold splits timeline plot...")
    plot_data_splits_timeline(
        session, event_column, time_bins, folds, sequence_labels,
        sequence_length, stride, mouse_id, session_id,
        text_size_scale=text_size_scale, show_legends=show_legends,
        save_path=session_dir / "kfold_splits_timeline.png"
    )

    # Train and evaluate on each fold
    step_num += 1
    print(f"\n{step_num}. Training LSTM model on {n_folds} folds...")

    fold_results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for fold_info in folds:
        fold_num = fold_info['fold_num']
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num + 1}/{n_folds}")
        print(f"{'='*60}")

        # Get data splits for this fold
        train_indices = fold_info['train_indices']
        val_indices = fold_info['val_indices']
        test_indices = fold_info['test_indices']

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

        # Initialize model for this fold
        n_neurons = sequences.shape[2]
        classifier = SequenceClassifier(
            input_size=n_neurons,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=0.75,
            bidirectional=bidirectional,
            device=device
        )

        # Train model
        print(f"\nTraining fold {fold_num + 1}...")
        history = classifier.train_model(
            train_dataset, val_dataset,
            epochs=epochs, batch_size=16, learning_rate=0.0001,
            verbose=False, use_best_model=use_best_model
        )

        # Evaluate model
        print(f"\nEvaluating fold {fold_num + 1}...")
        metrics, y_true, y_pred, y_prob = classifier.evaluate(test_dataset)

        print(f"Fold {fold_num + 1} Test Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

        # Store results
        fold_results.append({
            'fold_num': fold_num,
            'metrics': metrics,
            'history': history,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'test_indices': test_indices
        })

        # Save debug plots for this fold
        print(f"Saving debug plots for fold {fold_num + 1}...")

        # Always save to disk (no interactive display by default)
        plot_classifier_results(
            history, y_true, y_pred, y_prob,
            "LSTM Classifier", mouse_id, session_id, event_column,
            fold_num=fold_num + 1, text_size_scale=text_size_scale,
            show_legends=show_legends,
            save_path=session_dir / f"fold_{fold_num + 1}_classifier_results.png"
        )

    # Compute aggregate statistics across folds
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS ACROSS ALL FOLDS")
    print(f"{'='*60}")

    accuracies = [r['metrics']['accuracy'] for r in fold_results]
    precisions = [r['metrics']['precision'] for r in fold_results]
    recalls = [r['metrics']['recall'] for r in fold_results]
    f1_scores = [r['metrics']['f1'] for r in fold_results]
    aucs = [r['metrics']['auc'] for r in fold_results]

    print(f"\nMean ± Std across {n_folds} folds:")
    print(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  F1-score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"  AUC:       {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Create summary plot across all folds
    print(f"\nGenerating aggregate summary plot...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    metrics_data = {
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores,
        'AUC': aucs
    }

    for idx, (metric_name, metric_values) in enumerate(metrics_data.items()):
        axes[idx].bar(range(1, n_folds + 1), metric_values, color='steelblue', alpha=0.7)
        axes[idx].axhline(y=np.mean(metric_values), color='red', linestyle='--',
                         label=f'Mean: {np.mean(metric_values):.4f}', linewidth=2)
        axes[idx].axhline(y=np.mean(metric_values) + np.std(metric_values),
                         color='orange', linestyle=':', alpha=0.7,
                         label=f'±1 Std', linewidth=1.5)
        axes[idx].axhline(y=np.mean(metric_values) - np.std(metric_values),
                         color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
        axes[idx].set_xlabel('Fold', fontsize=12*text_size_scale)
        axes[idx].set_ylabel(metric_name, fontsize=12*text_size_scale)
        axes[idx].set_title(f'{metric_name} Across Folds', fontsize=14*text_size_scale)
        axes[idx].set_ylim([0, 1.05])
        axes[idx].legend(fontsize=9*text_size_scale)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(labelsize=10*text_size_scale)

    plt.suptitle(f"{mouse_id}-{session_id} K-Fold Cross Validation Summary ({n_folds} folds)",
                 fontsize=16*text_size_scale, y=1.02)
    plt.tight_layout()
    plt.savefig(session_dir / "kfold_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot to {session_dir / 'kfold_summary.png'}")

    print(f"\n{'='*60}")
    print("K-FOLD CROSS VALIDATION COMPLETE!")
    print(f"All plots saved to: {session_dir}")
    print(f"{'='*60}")

    return fold_results


if __name__ == "__main__":
    # Example usage - k-fold cross validation
    fold_results = run_lstm_kfold_demo(
        mouse_id="7010",
        session_id="m10",
        experiment="clickbait-motivate",
        event_column="flip_state",
        base_path="S:\\",
        sequence_length=5,
        bin_size_ms=200,
        stride=5,
        n_folds=10,  # 10-fold cross validation
        epochs=500,
        hidden_size=24,
        num_layers=2,
        bidirectional=False,
        use_best_model=True,
        centroid_y_threshold=np.inf,
        #cluster_filter='best_channel > 16',
        exclude_final_flip=False,
        random_state=42,
        text_size_scale=1,
        show_legends=True,
        plot_each_fold=True  # Set to True to see plots for each fold
    )

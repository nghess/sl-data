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
from torch.utils.data import DataLoader
import copy
import pandas as pd

from sldata import SessionData
from neural_classifier import SequenceClassifier, SequenceDataset


def plot_classifier_results(history: dict, y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray, model_name: str = "Classifier",
                           mouse_id: str = "", session_id: str = "", event_column: str = "",
                           text_size_scale: float = 1.0, show_legends: bool = True):
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
    text_size_scale : float
        Scale factor for text sizes (default: 1.0)
    show_legends : bool
        Whether to show legends on plots (default: True)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
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
    axes[0].set_title(f"{mouse_id}-{session_id} {model_name} Predicting '{event_column}'", fontsize=14*text_size_scale)
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
    axes[1].set_title(f'Best Model - Confusion Matrix', fontsize=14*text_size_scale)
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
    plt.show()
    
    return fig


def plot_temporal_predictions(session: SessionData, event_column: str, time_bins: np.ndarray,
                             y_true_bins: np.ndarray, y_pred_bins: np.ndarray, y_prob_bins: np.ndarray,
                             mouse_id: str = "", session_id: str = "", model_name: str = "Classifier",
                             show_trial_boundaries: bool = False, threshold: float = 0.5,
                             text_size_scale: float = 1.0, show_legends: bool = True):
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
    plt.show()
    
    return fig


def plot_data_splits_timeline(session: SessionData, event_column: str, time_bins: np.ndarray,
                              train_indices: np.ndarray, val_indices: np.ndarray,
                              test_indices: np.ndarray, bin_size_ms: float,
                              sequence_labels: np.ndarray = None, sequence_length: int = None,
                              stride: int = None, train_seq_indices: np.ndarray = None,
                              val_seq_indices: np.ndarray = None, test_seq_indices: np.ndarray = None,
                              mouse_id: str = "", session_id: str = "", show_trial_boundaries: bool = False,
                              text_size_scale: float = 1.0, show_legends: bool = True):
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
        Time bin indices for each data split
    bin_size_ms : float
        Size of time bins in milliseconds
    sequence_labels : np.ndarray, optional
        Labels for each sequence (for showing positive/negative)
    sequence_length : int, optional
        Length of sequences
    stride : int, optional
        Stride between sequences
    train_seq_indices, val_seq_indices, test_seq_indices : np.ndarray, optional
        Sequence indices for each split (for mapping to labels)
    show_trial_boundaries : bool, optional
        If True, show vertical lines at trial boundaries (default: False)
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
    ax1.set_title(f"{mouse_id}-{session_id} '{event_column}' State", fontsize=14*text_size_scale)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([])
    ax1.tick_params(labelsize=10*text_size_scale)
    ax1.grid(False)
    
    # Plot 2: Data splits overlay with positive/negative labels
    if (sequence_labels is not None and sequence_length is not None and stride is not None and
        train_seq_indices is not None and val_seq_indices is not None and test_seq_indices is not None):
        
        # Get time bin labels by mapping from sequence labels
        time_bin_labels = session.create_event_colormap(time_bins, event_column, aggregation='any')
        
        # Create arrays for each split type with positive/negative distinction
        train_pos_times = []
        train_neg_times = []
        val_pos_times = []
        val_neg_times = []
        test_pos_times = []
        test_neg_times = []
        
        # Process training data
        for seq_idx in train_seq_indices:
            start_bin = seq_idx * stride
            end_bin = start_bin + sequence_length
            bin_indices = range(start_bin, min(end_bin, len(time_bins)))
            
            for bin_idx in bin_indices:
                if bin_idx < len(time_bin_labels):
                    if time_bin_labels[bin_idx]:
                        train_pos_times.append(time_bins_s[bin_idx])
                    else:
                        train_neg_times.append(time_bins_s[bin_idx])
        
        # Process validation data
        for seq_idx in val_seq_indices:
            start_bin = seq_idx * stride
            end_bin = start_bin + sequence_length
            bin_indices = range(start_bin, min(end_bin, len(time_bins)))
            
            for bin_idx in bin_indices:
                if bin_idx < len(time_bin_labels):
                    if time_bin_labels[bin_idx]:
                        val_pos_times.append(time_bins_s[bin_idx])
                    else:
                        val_neg_times.append(time_bins_s[bin_idx])
        
        # Process test data
        for seq_idx in test_seq_indices:
            start_bin = seq_idx * stride
            end_bin = start_bin + sequence_length
            bin_indices = range(start_bin, min(end_bin, len(time_bins)))
            
            for bin_idx in bin_indices:
                if bin_idx < len(time_bin_labels):
                    if time_bin_labels[bin_idx]:
                        test_pos_times.append(time_bins_s[bin_idx])
                    else:
                        test_neg_times.append(time_bins_s[bin_idx])
        
        # Plot each category
        height_offset = 0
        marker_size = 1
        start_height = .1
        marker_alpha = 1.0
        
        # Training data
        if train_pos_times:
            ax2.scatter(train_pos_times, [start_height + height_offset] * len(train_pos_times), 
                       c='darkgreen', marker='|', s=marker_size, alpha=marker_alpha,
                       label=f'Train Positive ({len(train_pos_times)} bins)')
        height_offset += 0.08
        
        if train_neg_times:
            ax2.scatter(train_neg_times, [start_height + height_offset] * len(train_neg_times), 
                       c='lightgreen', marker='|', s=marker_size, alpha=marker_alpha,
                       label=f'Train Negative ({len(train_neg_times)} bins)')
        height_offset += 0.08
        
        # Validation data
        if val_pos_times:
            ax2.scatter(val_pos_times, [start_height + height_offset] * len(val_pos_times), 
                       c='darkorange', marker='|', s=marker_size, alpha=marker_alpha,
                       label=f'Val Positive ({len(val_pos_times)} bins)')
        height_offset += 0.08
        
        if val_neg_times:
            ax2.scatter(val_neg_times, [start_height + height_offset] * len(val_neg_times), 
                       c='orange', marker='|', s=marker_size, alpha=marker_alpha,
                       label=f'Val Negative ({len(val_neg_times)} bins)')
        height_offset += 0.08
        
        # Test data
        if test_pos_times:
            ax2.scatter(test_pos_times, [start_height + height_offset] * len(test_pos_times), 
                       c='darkviolet', marker='|', s=marker_size, alpha=marker_alpha,
                       label=f'Test Positive ({len(test_pos_times)} bins)')
        height_offset += 0.08
        
        if test_neg_times:
            ax2.scatter(test_neg_times, [start_height + height_offset] * len(test_neg_times), 
                       c='plum', marker='|', s=marker_size, alpha=marker_alpha,
                       label=f'Test Negative ({len(test_neg_times)} bins)')
        
        ax2.set_title('Train/Val/Test Split', fontsize=14*text_size_scale)
    
    else:
        # Fallback to original plotting if sequence information not provided
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
                ax2.scatter(times, [0.5 + height] * len(times), c=color, alpha=1, 
                           s=50, label=f'{split_type.title()} ({np.sum(mask)} bins)')
                height += .1
        
        ax2.set_title('Train/Validation/Test Split Distribution', fontsize=14*text_size_scale)
    
    # Add trial boundaries if requested
    if show_trial_boundaries and 'trial_number' in session.events.columns:
        # Find trial boundaries (where trial number changes)
        trial_numbers = session.events['trial_number'].fillna(-1)  # Fill NaN with -1
        trial_changes = np.where(np.diff(trial_numbers))[0]
        
        # Add vertical lines at trial boundaries
        for change_idx in trial_changes:
            boundary_time = event_times_s.iloc[change_idx + 1]  # Start of new trial
            
            # Add line to both subplots
            ax1.axvline(x=boundary_time, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax2.axvline(x=boundary_time, color='red', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add trial boundary to legend (just once)
        if len(trial_changes) > 0:
            ax2.axvline(x=-1, color='red', linestyle=':', alpha=0.5, linewidth=1, 
                       label=f'Trial boundaries ({len(trial_changes)} changes)')
    
    ax2.set_xlabel('Time (s)', fontsize=12*text_size_scale)
    ax2.set_ylim(0, height_offset + 2*start_height)
    ax2.set_yticks([])
    ax2.set_ylabel('')
    if show_legends:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10*text_size_scale)
    ax2.tick_params(labelsize=10*text_size_scale)
    ax2.grid(False)
    
    # Set white background for entire figure
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def create_trial_based_split(session: SessionData, sequence_labels: np.ndarray, time_bins: np.ndarray,
                           sequence_length: int, stride: int,
                           train_size: float = 0.6, val_size: float = 0.2, test_size: float = 0.2,
                           random_state: int = 42) -> tuple:
    """
    Create train/val/test split based on trial numbers to maintain temporal structure.
    
    Parameters:
    -----------
    session : SessionData
        Session data containing events with trial_number column
    sequence_labels : np.ndarray
        Labels for each sequence
    time_bins : np.ndarray
        Time bin centers used for analysis
    sequence_length : int
        Length of each sequence (number of time bins)
    stride : int
        Stride between sequences
    train_size : float
        Proportion of trials for training (default: 0.6)
    val_size : float
        Proportion of trials for validation (default: 0.2)
    test_size : float
        Proportion of trials for testing (default: 0.2)
    random_state : int
        Random seed for reproducible splits
        
    Returns:
    --------
    train_indices : np.ndarray
        Sequence indices for training
    val_indices : np.ndarray
        Sequence indices for validation
    test_indices : np.ndarray
        Sequence indices for testing
    trial_assignments : dict
        Mapping of trial numbers to split assignments
    """
    # Validate split proportions
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, got {train_size + val_size + test_size}")
    
    # Check if trial_number column exists
    if 'trial_number' not in session.events.columns:
        raise ValueError("trial_number column not found in session.events")
    
    # Get unique trial numbers
    unique_trials = session.events['trial_number'].unique()
    unique_trials = unique_trials[~pd.isna(unique_trials)]  # Remove NaN values
    unique_trials = sorted(unique_trials)
    
    print(f"Found {len(unique_trials)} unique trials: {unique_trials[:10]}{'...' if len(unique_trials) > 10 else ''}")
    
    # Set random seed for reproducible splits
    np.random.seed(random_state)
    
    # Shuffle trials to ensure random assignment
    shuffled_trials = np.random.permutation(unique_trials)
    
    # Calculate split indices
    n_trials = len(shuffled_trials)
    n_train = int(n_trials * train_size)
    n_val = int(n_trials * val_size)
    n_test = n_trials - n_train - n_val  # Ensure all trials are assigned
    
    # Assign trials to splits
    train_trials = set(shuffled_trials[:n_train])
    val_trials = set(shuffled_trials[n_train:n_train + n_val])
    test_trials = set(shuffled_trials[n_train + n_val:])
    
    trial_assignments = {
        'train': train_trials,
        'val': val_trials,
        'test': test_trials
    }
    
    print(f"Trial split: {len(train_trials)} train, {len(val_trials)} val, {len(test_trials)} test")
    
    # For each time bin, determine which trial it belongs to
    # Create a mapping from time bins to trial numbers
    time_bin_to_trial = {}
    
    # Get trial number for each time bin by finding the closest event
    for i, time_bin in enumerate(time_bins):
        # Find the event closest in time to this bin
        time_diffs = np.abs(session.events['timestamp_ms'] - time_bin)
        closest_event_idx = np.argmin(time_diffs)
        trial_num = session.events['trial_number'].iloc[closest_event_idx]
        
        if not pd.isna(trial_num):
            time_bin_to_trial[i] = trial_num
    
    # Now assign sequences to splits based on their time bins
    train_seq_indices = []
    val_seq_indices = []
    test_seq_indices = []
    
    for seq_idx in range(len(sequence_labels)):
        # Calculate which time bins this sequence spans
        start_time_bin_idx = seq_idx * stride
        end_time_bin_idx = start_time_bin_idx + sequence_length
        
        # Use the center time bin of the sequence as representative
        center_time_bin_idx = start_time_bin_idx + sequence_length // 2
        
        # Make sure we don't go beyond available time bins
        if center_time_bin_idx < len(time_bins) and center_time_bin_idx in time_bin_to_trial:
            trial_num = time_bin_to_trial[center_time_bin_idx]
            
            if trial_num in train_trials:
                train_seq_indices.append(seq_idx)
            elif trial_num in val_trials:
                val_seq_indices.append(seq_idx)
            elif trial_num in test_trials:
                test_seq_indices.append(seq_idx)
    
    train_indices = np.array(train_seq_indices)
    val_indices = np.array(val_seq_indices)
    test_indices = np.array(test_seq_indices)
    
    # Print class balance for each split
    if len(train_indices) > 0:
        train_pos_rate = np.mean(sequence_labels[train_indices])
        print(f"Train: {len(train_indices)} sequences, {train_pos_rate:.3f} positive rate")
    else:
        print("Warning: No training sequences assigned")
        
    if len(val_indices) > 0:
        val_pos_rate = np.mean(sequence_labels[val_indices])
        print(f"Val: {len(val_indices)} sequences, {val_pos_rate:.3f} positive rate")
    else:
        print("Warning: No validation sequences assigned")
        
    if len(test_indices) > 0:
        test_pos_rate = np.mean(sequence_labels[test_indices])
        print(f"Test: {len(test_indices)} sequences, {test_pos_rate:.3f} positive rate")
    else:
        print("Warning: No test sequences assigned")
    
    return train_indices, val_indices, test_indices, trial_assignments


def create_balanced_temporal_split(sequence_labels: np.ndarray, time_bins: np.ndarray,
                                  train_size: float = 0.6, val_size: float = 0.2, test_size: float = 0.2,
                                  n_blocks: int = 10, random_state: int = 42) -> tuple:
    """
    Create train/val/test split with balanced temporal distribution across the session.
    
    This method divides the session into temporal blocks and samples proportionally 
    from each block to ensure even representation across time, regardless of trial density.
    
    Parameters:
    -----------
    sequence_labels : np.ndarray
        Labels for each sequence
    time_bins : np.ndarray
        Time bin centers used for analysis
    train_size : float
        Proportion of data for training (default: 0.6)
    val_size : float
        Proportion of data for validation (default: 0.2)
    test_size : float
        Proportion of data for testing (default: 0.2)
    n_blocks : int
        Number of temporal blocks to create (default: 10)
    random_state : int
        Random seed for reproducible splits
        
    Returns:
    --------
    train_indices : np.ndarray
        Sequence indices for training
    val_indices : np.ndarray
        Sequence indices for validation
    test_indices : np.ndarray
        Sequence indices for testing
    block_info : dict
        Information about the temporal blocks created
    """
    # Validate split proportions
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, got {train_size + val_size + test_size}")
    
    n_sequences = len(sequence_labels)
    
    # Set random seed for reproducible splits
    np.random.seed(random_state)
    
    # Create temporal blocks
    block_size = n_sequences // n_blocks
    remainder = n_sequences % n_blocks
    
    print(f"Creating {n_blocks} balanced temporal blocks from {n_sequences} sequences:")
    print(f"  Base block size: {block_size} sequences")
    if remainder > 0:
        print(f"  {remainder} blocks will have {block_size + 1} sequences")
    
    # Initialize split arrays
    train_indices = []
    val_indices = []
    test_indices = []
    block_info = []
    
    # For each temporal block, sample proportionally for train/val/test
    for block_idx in range(n_blocks):
        # Calculate block boundaries
        start_idx = block_idx * block_size + min(block_idx, remainder)
        if block_idx < remainder:
            end_idx = start_idx + block_size + 1
        else:
            end_idx = start_idx + block_size
        
        block_sequences = np.arange(start_idx, end_idx)
        block_labels = sequence_labels[start_idx:end_idx]
        block_size_actual = len(block_sequences)
        
        print(f"  Block {block_idx}: sequences {start_idx}-{end_idx-1} ({block_size_actual} sequences, {np.mean(block_labels):.3f} positive)")
        
        # Shuffle sequences within this block
        shuffled_indices = np.random.permutation(block_sequences)
        
        # Calculate split sizes for this block
        n_train_block = int(block_size_actual * train_size)
        n_val_block = int(block_size_actual * val_size)
        n_test_block = block_size_actual - n_train_block - n_val_block  # Ensure all assigned
        
        # Assign sequences to splits
        block_train = shuffled_indices[:n_train_block]
        block_val = shuffled_indices[n_train_block:n_train_block + n_val_block]
        block_test = shuffled_indices[n_train_block + n_val_block:]
        
        train_indices.extend(block_train)
        val_indices.extend(block_val)
        test_indices.extend(block_test)
        
        # Store block information
        block_info.append({
            'block_idx': block_idx,
            'start_seq': start_idx,
            'end_seq': end_idx - 1,
            'n_sequences': block_size_actual,
            'n_train': len(block_train),
            'n_val': len(block_val),
            'n_test': len(block_test),
            'positive_rate': np.mean(block_labels)
        })
        
        print(f"    Block {block_idx} split: {len(block_train)} train, {len(block_val)} val, {len(block_test)} test")
    
    # Convert to numpy arrays and sort (to maintain some temporal order within splits)
    train_indices = np.sort(np.array(train_indices))
    val_indices = np.sort(np.array(val_indices))
    test_indices = np.sort(np.array(test_indices))
    
    # Print class balance for each split
    if len(train_indices) > 0:
        train_pos_rate = np.mean(sequence_labels[train_indices])
        print(f"Overall Train: {len(train_indices)} sequences, {train_pos_rate:.3f} positive rate")
    else:
        print("Warning: No training sequences assigned")
        
    if len(val_indices) > 0:
        val_pos_rate = np.mean(sequence_labels[val_indices])
        print(f"Overall Val: {len(val_indices)} sequences, {val_pos_rate:.3f} positive rate")
    else:
        print("Warning: No validation sequences assigned")
        
    if len(test_indices) > 0:
        test_pos_rate = np.mean(sequence_labels[test_indices])
        print(f"Overall Test: {len(test_indices)} sequences, {test_pos_rate:.3f} positive rate")
    else:
        print("Warning: No test sequences assigned")
    
    return train_indices, val_indices, test_indices, block_info


def create_sequences(population_matrix: np.ndarray, labels: np.ndarray, 
                    sequence_length: int, stride: int = 1, filter_boundaries: bool = False) -> tuple:
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
    filter_boundaries : bool
        If True, skip sequences that span state changes (default: False)
        
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
    
    # Collect valid sequences
    valid_sequences = []
    valid_labels = []
    filtered_count = 0
    
    for i in range(max_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        
        # Get labels for this sequence
        seq_labels = labels[start_idx:end_idx]
        
        # Check if we should filter boundary-crossing sequences
        if filter_boundaries:
            # Skip if sequence spans a state change
            if len(np.unique(seq_labels)) > 1:
                filtered_count += 1
                continue
        
        # Population sequence
        sequence = population_matrix[start_idx:end_idx]
        valid_sequences.append(sequence)
        
        # Label is from the final time bin (what we're predicting)
        valid_labels.append(labels[end_idx - 1])
    
    # Convert to arrays
    if len(valid_sequences) == 0:
        raise ValueError("No valid sequences found - try reducing sequence_length or disabling filter_boundaries")
    
    sequences = np.array(valid_sequences)
    sequence_labels = np.array(valid_labels)
    
    print(f"Created {len(sequences)} sequences of length {sequence_length}")
    if filter_boundaries:
        print(f"Filtered out {filtered_count} boundary-crossing sequences ({filtered_count/(max_sequences)*100:.1f}%)")
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
                     centroid_y_threshold: float = None,
                     filter_boundaries: bool = False) -> tuple:
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
    filter_boundaries : bool
        If True, skip sequences that span state changes (default: False)

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
        aggregation='any'  # True if any event in time bin is True
    )
    
    print(f"Time binning results:")
    print(f"  Time bins created: {len(time_bins)}")
    print(f"  Time bin positive rate: {np.mean(labels):.3f}")
    print(f"  Time bin size: {bin_size_ms}ms")
    
    # Apply centroid_y filtering if requested
    if 'bonsai_centroid_y' in session.events.columns:
        # Get centroid_y values for each time bin
        centroid_y_values = session.create_event_colormap(
            time_bins,
            'bonsai_centroid_y',
            aggregation='mean'  # Use mean centroid_y for each time bin
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
        time_bins = time_bins[centroid_mask]
        
        print(f"Applied centroid_y filter (≤{centroid_y_threshold:.1f}): "
              f"kept {np.sum(centroid_mask)}/{len(centroid_mask)} time bins")
    else:
        print("Warning: No 'bonsai_centroid_y' column found in events data - skipping position filtering")
    
    # Create sequences
    sequences, sequence_labels = create_sequences(
        population_matrix, labels, sequence_length, stride, filter_boundaries
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
                 cluster_filter: str = None,
                 exclude_final_flip: bool = False,
                 filter_boundaries: bool = False,
                 use_trial_split: bool = False,
                 use_stratified_temporal: bool = False,
                 train_size: float = 0.6,
                 val_size: float = 0.2,
                 random_state: int = 42,
                 text_size_scale: float = 1.0,
                 show_legends: bool = True):
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
    exclude_final_flip : bool
        If True, excludes data from the final flip_state period (when mouse less engaged).
        If False, uses all data (default).
    filter_boundaries : bool
        If True, skip sequences that span state changes to reduce label noise (default: False).
        If False, includes all sequences and labels them by final time bin.
    use_trial_split : bool
        If True, use trial-based splitting instead of temporal block splitting (default: False).
        Requires trial_number column in events data.
    use_stratified_temporal : bool
        If True, use balanced temporal stratification (default: False).
        Creates temporal blocks and samples proportionally from each for balanced coverage.
        Cannot be used with use_trial_split.
    train_size : float
        Proportion of data for training when using trial-based or stratified temporal split (default: 0.6).
    val_size : float  
        Proportion of data for validation when using trial-based or stratified temporal split (default: 0.2).
        Test size will be 1 - train_size - val_size.
    random_state : int
        Random seed for reproducible splits (default: 42).
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
    
    # Prepare sequence data
    step_num = 4 if exclude_final_flip else (3 if cluster_filter is not None else 2)
    print(f"\n{step_num}. Preparing sequence data...")
    sequences, sequence_labels, time_bins = prepare_lstm_data(
        session, event_column, sequence_length, bin_size_ms, stride, 
        centroid_y_threshold=centroid_y_threshold, filter_boundaries=filter_boundaries
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
    
    # Choose splitting method 
    if use_trial_split and use_stratified_temporal:
        raise ValueError("Cannot use both use_trial_split and use_stratified_temporal. Choose one.")
    
    if use_trial_split:
        print(f"\n4. Trial-based splitting (train: {train_size:.1f}, val: {val_size:.1f}, test: {1-train_size-val_size:.1f})...")
        
        # Use trial-based splitting
        train_indices, val_indices, test_indices, trial_assignments = create_trial_based_split(
            session, sequence_labels, time_bins, sequence_length, stride, 
            train_size, val_size, 1-train_size-val_size, random_state
        )
        
        print(f"Trial-based split completed:")
        print(f"  Trial assignments: {len(trial_assignments['train'])} train, {len(trial_assignments['val'])} val, {len(trial_assignments['test'])} test trials")
        
    elif use_stratified_temporal:
        print(f"\n4. Balanced temporal stratification (train: {train_size:.1f}, val: {val_size:.1f}, test: {1-train_size-val_size:.1f})...")
        
        # Use balanced temporal stratification
        train_indices, val_indices, test_indices, block_info = create_balanced_temporal_split(
            sequence_labels, time_bins, train_size, val_size, 1-train_size-val_size, 
            n_blocks=10, random_state=random_state
        )
        
        print(f"Balanced temporal stratification completed:")
        print(f"  Created {len(block_info)} temporal blocks with balanced sampling")
        
    else:
        print(f"\n4. Stratified temporal block splitting (test size: {test_size})...")
        n_sequences = len(sequences)
        
        # Create temporal blocks and sample from each to ensure full coverage
        n_blocks = 10
        block_size = n_sequences // n_blocks
        
        print(f"Creating {n_blocks} temporal blocks from {n_sequences} sequences:")
        print(f"  Block size: {block_size} sequences per block")
        
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
            
            print(f"  Block {block_idx}: sequences {start_idx}-{end_idx-1} ({len(block_sequences)} sequences, {np.mean(block_labels):.3f} positive)")
            
            # Check if we can stratify this block
            unique_labels, label_counts = np.unique(block_labels, return_counts=True)
            min_class_size = np.min(label_counts)
            
            # Need at least 3 samples per class for stratified split (train/val/test)
            if len(unique_labels) < 2 or min_class_size < 3:
                # Can't stratify - distribute proportionally
                print(f"    Block {block_idx}: Can't stratify (min class size: {min_class_size}), using proportional split")
                n_test = max(1, int(len(block_sequences) * test_size))
                n_val = max(1, int(len(block_sequences) * 0.16))  # 20% of remaining 80%
                
                test_indices.extend(block_sequences[-n_test:])
                val_indices.extend(block_sequences[-(n_test+n_val):-n_test] if n_test+n_val < len(block_sequences) else [])
                train_indices.extend(block_sequences[:-(n_test+n_val)] if n_test+n_val < len(block_sequences) else block_sequences[:-n_test])
            else:
                # Stratified split within block
                from sklearn.model_selection import train_test_split
                
                try:
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
                    print(f"    Block {block_idx}: Stratified split successful")
                    
                except ValueError as e:
                    # Fallback to proportional if stratification fails
                    print(f"    Block {block_idx}: Stratification failed ({e}), using proportional split")
                    n_test = max(1, int(len(block_sequences) * test_size))
                    n_val = max(1, int(len(block_sequences) * 0.16))
                    
                    test_indices.extend(block_sequences[-n_test:])
                    val_indices.extend(block_sequences[-(n_test+n_val):-n_test] if n_test+n_val < len(block_sequences) else [])
                    train_indices.extend(block_sequences[:-(n_test+n_val)] if n_test+n_val < len(block_sequences) else block_sequences[:-n_test])
        
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
    
    # Check for overlapping sequences between splits
    print(f"\n=== Sequence Overlap Analysis ===")
    def get_time_bins_for_sequences(seq_indices, stride, sequence_length):
        """Get all time bin indices covered by sequences"""
        time_bin_set = set()
        for seq_idx in seq_indices:
            start_bin = seq_idx * stride
            end_bin = start_bin + sequence_length
            time_bin_set.update(range(start_bin, end_bin))
        return time_bin_set
    
    train_time_bins = get_time_bins_for_sequences(train_indices, stride, sequence_length)
    val_time_bins = get_time_bins_for_sequences(val_indices, stride, sequence_length)
    test_time_bins = get_time_bins_for_sequences(test_indices, stride, sequence_length)
    
    # Check overlaps
    train_val_overlap = len(train_time_bins & val_time_bins)
    train_test_overlap = len(train_time_bins & test_time_bins)
    val_test_overlap = len(val_time_bins & test_time_bins)
    
    print(f"Time bin overlaps between splits:")
    print(f"  Train-Val overlap: {train_val_overlap} bins")
    print(f"  Train-Test overlap: {train_test_overlap} bins") 
    print(f"  Val-Test overlap: {val_test_overlap} bins")
    
    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        print(f"⚠️ WARNING: Data leakage detected! Overlapping time bins between splits.")
        print(f"   Consider increasing stride or using non-overlapping sequence approach.")
    else:
        print(f"✅ No time bin overlaps detected between splits.")
    
    print(f"Training set: {len(train_dataset)} sequences")
    print(f"Validation set: {len(val_dataset)} sequences") 
    print(f"Test set: {len(test_dataset)} sequences")
    
    # Plot timeline with data splits
    print(f"\nGenerating timeline visualization...")
    # Convert sequence indices to time bin indices for plotting
    train_time_indices = []
    val_time_indices = []
    test_time_indices = []
    
    for seq_idx in train_indices:
        # Each sequence spans multiple time bins
        start_bin = seq_idx * stride
        end_bin = start_bin + sequence_length
        train_time_indices.extend(range(start_bin, min(end_bin, len(time_bins))))
    
    for seq_idx in val_indices:
        start_bin = seq_idx * stride
        end_bin = start_bin + sequence_length
        val_time_indices.extend(range(start_bin, min(end_bin, len(time_bins))))
        
    for seq_idx in test_indices:
        start_bin = seq_idx * stride
        end_bin = start_bin + sequence_length
        test_time_indices.extend(range(start_bin, min(end_bin, len(time_bins))))
    
    # Remove duplicates and convert to arrays
    train_time_indices = np.unique(train_time_indices)
    val_time_indices = np.unique(val_time_indices) 
    test_time_indices = np.unique(test_time_indices)
    
    print(f"Time bin coverage:")
    print(f"  Train: {len(train_time_indices)} time bins")
    print(f"  Validation: {len(val_time_indices)} time bins")
    print(f"  Test: {len(test_time_indices)} time bins")
    print(f"  Total unique: {len(np.unique(np.concatenate([train_time_indices, val_time_indices, test_time_indices])))}/{len(time_bins)}")
    
    plot_data_splits_timeline(session, event_column, time_bins,
                              train_time_indices, val_time_indices, test_time_indices, bin_size_ms,
                              sequence_labels=sequence_labels, sequence_length=sequence_length,
                              stride=stride, train_seq_indices=train_indices,
                              val_seq_indices=val_indices, test_seq_indices=test_indices,
                              mouse_id=mouse_id, session_id=session_id,
                              show_trial_boundaries=use_trial_split, text_size_scale=text_size_scale,
                              show_legends=show_legends)
    
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
        dropout_rate=0.75,
        bidirectional=bidirectional,
        device=device
    )
    
    # Train model
    print(f"\n6. Training LSTM model for {epochs} epochs...")
    history = classifier.train_model(
        train_dataset, val_dataset,
        epochs=epochs, batch_size=128, learning_rate=0.0001,
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
    plot_classifier_results(history, y_true, y_pred, y_prob, "LSTM Classifier", mouse_id, session_id, event_column, text_size_scale, show_legends)
    
    # Generate temporal prediction visualization
    print(f"\n9. Generating temporal prediction visualization...")
    
    # Create bin-wise predictions by mapping TEST sequence predictions back to time bins
    # Only use TEST SET for temporal predictions (to show true out-of-sample performance)
    test_sequences = sequences[test_indices]
    test_seq_labels = sequence_labels[test_indices]
    
    # Get predictions for test sequences only
    test_dataset = SequenceDataset(test_sequences, test_seq_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    classifier.model.eval()
    test_probs = []
    raw_outputs_debug = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(classifier.device)
            outputs = classifier.model(batch_X)
            raw_outputs_debug.extend(outputs.cpu().numpy().flatten())
            # DON'T apply sigmoid - the model already outputs probabilities
            probs = outputs.cpu().numpy().flatten()
            test_probs.extend(probs)
    
    test_probs = np.array(test_probs)
    raw_outputs_debug = np.array(raw_outputs_debug)
    
    # Debug: Check raw model outputs vs processed probabilities
    print(f"Raw model outputs range: {raw_outputs_debug.min():.3f} to {raw_outputs_debug.max():.3f}")
    print(f"After sigmoid range: {test_probs.min():.3f} to {test_probs.max():.3f}")
    print(f"Mean raw output: {raw_outputs_debug.mean():.3f}")
    print(f"Mean probability: {test_probs.mean():.3f}")
    
    # Compare with main evaluation probabilities
    y_prob_array = np.array(y_prob)
    print(f"Main evaluation y_prob range: {y_prob_array.min():.3f} to {y_prob_array.max():.3f}")
    print(f"Main evaluation y_prob mean: {y_prob_array.mean():.3f}")
    print(f"Temporal vs main prob difference: {np.abs(test_probs.mean() - y_prob_array.mean()):.3f}")
    
    # Use standard 0.5 threshold (same as main LSTM evaluation)
    threshold = 0.5
    print(f"Using standard threshold: {threshold}")
    
    # Apply threshold
    test_preds = (test_probs > threshold).astype(int)
    
    # Map TEST sequence predictions back to time bins
    # Initialize with NaN (for bins not covered by test sequences)
    y_pred_bins = np.full(len(time_bins), np.nan)
    y_prob_bins = np.full(len(time_bins), np.nan)
    y_true_bins = session.create_event_colormap(time_bins, event_column, aggregation='any')
    
    # For each TEST sequence, assign its prediction to the corresponding time bins
    for i, seq_idx in enumerate(test_indices):
        # Calculate which time bins this sequence corresponds to
        start_bin = seq_idx * stride
        end_bin = start_bin + sequence_length
        
        # Assign this sequence's prediction to all bins it covers
        for bin_idx in range(start_bin, min(end_bin, len(time_bins))):
            y_pred_bins[bin_idx] = test_preds[i]  # Binary prediction (0 or 1)
            y_prob_bins[bin_idx] = test_probs[i]  # Probability (0.0 to 1.0)
    
    # Remove NaN values (bins not covered by test sequences)
    valid_mask = ~np.isnan(y_pred_bins)
    y_true_bins_clean = y_true_bins[valid_mask]
    y_pred_bins_clean = y_pred_bins[valid_mask]  # Already binary, no need to convert
    y_prob_bins_clean = y_prob_bins[valid_mask]
    time_bins_clean = time_bins[valid_mask]
    
    # Debug: Check what we're actually getting
    print(f"Test predictions range: {test_preds.min():.3f} to {test_preds.max():.3f}")
    print(f"Test probabilities range: {test_probs.min():.3f} to {test_probs.max():.3f}")
    print(f"Unique binary predictions: {np.unique(y_pred_bins_clean)}")
    print(f"Predicted class distribution: {np.bincount(y_pred_bins_clean.astype(int))}")
    
    print(f"Temporal predictions: {len(time_bins_clean)}/{len(time_bins)} bins covered (test set only)")
    
    # Plot temporal predictions
    plot_temporal_predictions(
        session, event_column, time_bins_clean,
        y_true_bins_clean.astype(int), y_pred_bins_clean.astype(int), y_prob_bins_clean,
        mouse_id, session_id, "LSTM Classifier",
        show_trial_boundaries=use_trial_split, threshold=threshold,
        text_size_scale=text_size_scale, show_legends=show_legends
    )
    
    return classifier, metrics, history


if __name__ == "__main__":
    # Example usage - optimized for neural data with cluster filtering
    run_lstm_demo(
        mouse_id="7010",
        session_id="m10", 
        experiment="clickbait-motivate",
        event_column="flip_state",
        base_path="S:\\",
        sequence_length=5,  # Number of time bins per sequence
        bin_size_ms=200,  # Coarser temporal resolution  
        stride=5,  # Dense sampling
        test_size=0.2,
        epochs=1500,  # Training epochs
        hidden_size=24,  # Small hidden size to discourage overfitting
        num_layers=2,  # Depth of network
        bidirectional=False,
        use_best_model=True,  # Use best validation model
        centroid_y_threshold=np.inf,  # Y-axis halfway point = 1968//2
        cluster_filter='best_channel <= 16',  # Filter by cluster attribute
        exclude_final_flip=False,  # Exclude final flip_state period
        filter_boundaries=False,  # Set to True to skip boundary-crossing sequences
        use_trial_split=False,  # Use trial-based splitting for better temporal structure
        use_stratified_temporal=True,
        train_size=0.6,  # 60% of trials for training
        val_size=0.2,  # 20% of trials for validation (20% remaining for test)
        random_state=42,  # For reproducible splits
        text_size_scale=1,  # Make text twice as big for presentations
        show_legends=True  # Hide legends for cleaner presentation slides
    )
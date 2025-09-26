"""
Demo script for SVM-based neural population sequence classification.

This script demonstrates how to:
1. Load neural data using SessionData
2. Create population activity sequences 
3. Prepare behavioral event labels for sequences
4. Train an SVM classifier
5. Evaluate model performance

Code by Nate Gonzales-Hess, September 2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import copy
import pandas as pd
from typing import Tuple, Dict, Optional
import time

from sldata import SessionData


class SVMClassifier:
    """
    SVM wrapper for neural sequence classification with interface similar to SequenceClassifier.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
                 class_weight: str = 'balanced', random_state: int = 42):
        """
        Initialize SVM classifier.
        
        Parameters:
        -----------
        kernel : str
            SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        C : float
            Regularization parameter
        gamma : str or float
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        class_weight : str or dict
            Class weights for handling imbalanced data
        random_state : int
            Random seed for reproducible results
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.history = {}
        
    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten sequence data for SVM input.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences [n_samples, sequence_length, n_features]
            
        Returns:
        --------
        X_flat : np.ndarray
            Flattened features [n_samples, sequence_length * n_features]
        """
        n_samples, seq_len, n_features = X.shape
        return X.reshape(n_samples, seq_len * n_features)
    
    def train_model(self, train_X: np.ndarray, train_y: np.ndarray,
                   val_X: np.ndarray = None, val_y: np.ndarray = None,
                   use_grid_search: bool = True, verbose: bool = True) -> Dict:
        """
        Train SVM model with optional hyperparameter tuning.
        
        Parameters:
        -----------
        train_X : np.ndarray
            Training sequences [n_samples, sequence_length, n_features]
        train_y : np.ndarray
            Training labels [n_samples]
        val_X : np.ndarray, optional
            Validation sequences (for reporting, not used in training)
        val_y : np.ndarray, optional
            Validation labels (for reporting, not used in training)
        use_grid_search : bool
            Whether to use grid search for hyperparameter tuning
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        history : dict
            Training history (for compatibility with LSTM interface)
        """
        if verbose:
            print("Training SVM classifier...")
            
        start_time = time.time()
        
        # Flatten sequences for SVM
        X_train_flat = self._flatten_sequences(train_X)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        if use_grid_search:
            if verbose:
                print("Performing grid search for hyperparameter tuning...")
                
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
            
            # Perform grid search with cross-validation
            self.model = GridSearchCV(
                SVC(class_weight=self.class_weight, random_state=self.random_state, probability=True),
                param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
            )
            self.model.fit(X_train_scaled, train_y)
            
            self.best_params = self.model.best_params_
            if verbose:
                print(f"Best parameters: {self.best_params}")
                print(f"Best CV score: {self.model.best_score_:.4f}")
        else:
            # Train with specified parameters
            self.model = SVC(
                kernel=self.kernel, C=self.C, gamma=self.gamma,
                class_weight=self.class_weight, random_state=self.random_state,
                probability=True
            )
            self.model.fit(X_train_scaled, train_y)
        
        training_time = time.time() - start_time
        
        # Create history dict for compatibility
        self.history = {
            'training_time': training_time,
            'train_accuracy': self.model.score(X_train_scaled, train_y),
            'best_params': self.best_params if use_grid_search else None
        }
        
        # Add validation metrics if provided
        if val_X is not None and val_y is not None:
            X_val_flat = self._flatten_sequences(val_X)
            X_val_scaled = self.scaler.transform(X_val_flat)
            self.history['val_accuracy'] = self.model.score(X_val_scaled, val_y)
        
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Training accuracy: {self.history['train_accuracy']:.4f}")
            if 'val_accuracy' in self.history:
                print(f"Validation accuracy: {self.history['val_accuracy']:.4f}")
        
        return self.history
    
    def evaluate(self, test_X: np.ndarray, test_y: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model on test data.
        
        Parameters:
        -----------
        test_X : np.ndarray
            Test sequences [n_samples, sequence_length, n_features]
        test_y : np.ndarray
            Test labels [n_samples]
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_prob : np.ndarray
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Flatten and scale test data
        X_test_flat = self._flatten_sequences(test_X)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(test_y, y_pred),
            'precision': precision_score(test_y, y_pred, zero_division=0),
            'recall': recall_score(test_y, y_pred, zero_division=0),
            'f1': f1_score(test_y, y_pred, zero_division=0),
            'auc': roc_auc_score(test_y, y_prob) if len(np.unique(test_y)) > 1 else 0.0
        }
        
        return metrics, test_y, y_pred, y_prob


def plot_classifier_results(history: dict, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_prob: np.ndarray, model_name: str = "Classifier", 
                           mouse_id: str = "", session_id: str = "", event_column: str = ""):
    """
    Create standardized plots for classifier evaluation.
    
    Parameters:
    -----------
    history : dict
        Training history with metrics
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Prediction probabilities
    model_name : str
        Name of the model for plot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Training Summary (replace loss plot)
    train_acc = history.get('train_accuracy', 0)
    val_acc = history.get('val_accuracy', None)
    training_time = history.get('training_time', 0)
    
    # Show training summary as text
    summary_text = f"Training Accuracy: {train_acc:.4f}\n"
    if val_acc is not None:
        summary_text += f"Validation Accuracy: {val_acc:.4f}\n"
    summary_text += f"Training Time: {training_time:.2f}s\n"
    
    if 'best_params' in history and history['best_params']:
        summary_text += "\nBest Parameters:\n"
        for param, value in history['best_params'].items():
            summary_text += f"  {param}: {value}\n"
    
    axes[0,0].text(0.1, 0.5, summary_text, transform=axes[0,0].transAxes, fontsize=12,
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[0,0].set_xlim(0, 1)
    axes[0,0].set_ylim(0, 1)
    axes[0,0].set_title(f"{mouse_id}-{session_id} {model_name} Predicting '{event_column}'")
    axes[0,0].axis('off')
    
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
    
    sns.heatmap(cm_percentages, annot=annotations, fmt='', cmap='Blues', ax=axes[0,1], cbar=False)
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('True')
    axes[0,1].set_title(f'{model_name} - Confusion Matrix')
    axes[0,1].set_aspect('equal')
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Calculate baseline (random classifier performance)
    baseline_precision = np.sum(y_true) / len(y_true)
    
    axes[1,0].plot(recall, precision, color='darkorange', lw=2, 
                   label=f'PR curve (AP = {avg_precision:.3f})')
    axes[1,0].axhline(y=baseline_precision, color='navy', lw=2, linestyle='--', 
                     label=f'Random classifier (AP = {baseline_precision:.3f})')
    axes[1,0].set_xlim([0.0, 1.0])
    axes[1,0].set_ylim([0.0, 1.05])
    axes[1,0].set_xlabel('Recall (TP/(TP+FN))')
    axes[1,0].set_ylabel('Precision (TP/(TP+FP))')
    axes[1,0].set_title(f'{model_name} - Precision-Recall Curve')
    axes[1,0].legend(loc="lower left")
    axes[1,0].grid(False)
    axes[1,0].set_aspect('equal')
    
    # 4. ROC Curve
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    axes[1,1].plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                   label='Random classifier (AUC = 0.500)')
    axes[1,1].set_xlim([0.0, 1.0])
    axes[1,1].set_ylim([0.0, 1.05])
    axes[1,1].set_xlabel('False Positive Rate (1-Specificity)')
    axes[1,1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1,1].set_title(f'{model_name} - ROC Curve')
    axes[1,1].legend(loc="lower right")
    axes[1,1].grid(False)
    axes[1,1].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_data_splits_timeline(session: SessionData, event_column: str, time_bins: np.ndarray, 
                              train_indices: np.ndarray, val_indices: np.ndarray, 
                              test_indices: np.ndarray, bin_size_ms: float, 
                              sequence_labels: np.ndarray = None, sequence_length: int = None, 
                              stride: int = None, train_seq_indices: np.ndarray = None,
                              val_seq_indices: np.ndarray = None, test_seq_indices: np.ndarray = None,
                              mouse_id: str = "", session_id: str = "", show_trial_boundaries: bool = False):
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
    
    ax1.set_ylabel(f'{event_column}')
    ax1.set_title(f"{mouse_id}-{session_id} '{event_column}' State")
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([])
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
        
        ax2.set_title('Train/Val/Test Split')
    
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
        
        ax2.set_title('Train/Validation/Test Split Distribution')
    
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
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(0, height_offset + 2*start_height)
    ax2.set_yticks([])
    ax2.set_ylabel('')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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


def prepare_svm_data(session: SessionData, event_column: str, 
                     sequence_length: int = 10,
                     bin_size_ms: float = 50.0, 
                     stride: int = 1,
                     min_time_ms: float = 0,
                     max_time_ms: float = None,
                     centroid_y_threshold: float = None,
                     filter_boundaries: bool = False) -> tuple:
    """
    Prepare sequence data for SVM classification from SessionData.
    
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
    print(f"Preparing SVM data for predicting '{event_column}'...")
    
    # Create population matrix
    if max_time_ms is None:
        max_time_ms = np.inf
    
    pop_matrix, time_bins, cluster_ids = session.create_population_raster(
        start_time=min_time_ms,
        end_time=max_time_ms,
        bin_size_ms=bin_size_ms,
        zscore_neurons=False
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


def run_svm_demo(mouse_id: str, session_id: str, experiment: str,
                 event_column: str = 'flip_state',
                 base_path: str = "S:\\",
                 sequence_length: int = 10,
                 bin_size_ms: float = 50.0,
                 stride: int = 5,
                 test_size: float = 0.2,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 use_grid_search: bool = True,
                 centroid_y_threshold: float = None,
                 cluster_filter: str = None,
                 exclude_final_flip: bool = False,
                 filter_boundaries: bool = False,
                 use_trial_split: bool = False,
                 train_size: float = 0.6,
                 val_size: float = 0.2,
                 random_state: int = 42):
    """
    Run complete SVM classification demo.
    
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
    kernel : str
        SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')
    C : float
        SVM regularization parameter
    gamma : str or float
        SVM kernel coefficient
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    centroid_y_threshold : float, optional
        Only include time bins where centroid_y <= threshold.
        If None, uses 50% of max centroid_y value.
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
    train_size : float
        Proportion of trials for training when using trial-based split (default: 0.6).
    val_size : float  
        Proportion of trials for validation when using trial-based split (default: 0.2).
        Test size will be 1 - train_size - val_size.
    random_state : int
        Random seed for reproducible trial-based splits (default: 42).
    """
    print("="*60)
    print("SVM NEURAL POPULATION CLASSIFICATION DEMO")
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
    sequences, sequence_labels, time_bins = prepare_svm_data(
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
    
    # Choose splitting method based on use_trial_split parameter
    if use_trial_split:
        print(f"\n{step_num+1}. Trial-based splitting (train: {train_size:.1f}, val: {val_size:.1f}, test: {1-train_size-val_size:.1f})...")
        
        # Use trial-based splitting
        train_indices, val_indices, test_indices, trial_assignments = create_trial_based_split(
            session, sequence_labels, time_bins, sequence_length, stride, 
            train_size, val_size, 1-train_size-val_size, random_state
        )
        
        print(f"Trial-based split completed:")
        print(f"  Trial assignments: {len(trial_assignments['train'])} train, {len(trial_assignments['val'])} val, {len(trial_assignments['test'])} test trials")
        
    else:
        print(f"\n{step_num+1}. Simple train/val/test splitting (test size: {test_size})...")
        n_sequences = len(sequences)
        
        # Simple sequential split for SVM (since training is faster than LSTM)
        n_test = int(n_sequences * test_size)
        n_val = int(n_sequences * 0.16)  # 16% for validation
        
        # Take test from end, validation from middle, train from beginning
        test_indices = np.arange(n_sequences - n_test, n_sequences)
        val_indices = np.arange(n_sequences - n_test - n_val, n_sequences - n_test)
        train_indices = np.arange(0, n_sequences - n_test - n_val)
        
        print(f"Sequential split:")
        print(f"  Training: {len(train_indices)} sequences")
        print(f"  Validation: {len(val_indices)} sequences")
        print(f"  Test: {len(test_indices)} sequences")
    
    # Create splits
    X_train = sequences[train_indices]
    y_train = sequence_labels[train_indices]
    
    X_val = sequences[val_indices] 
    y_val = sequence_labels[val_indices]
    
    X_test = sequences[test_indices]
    y_test = sequence_labels[test_indices]
    
    print(f"Final split:")
    print(f"  Training: {len(X_train)} sequences, {np.mean(y_train):.3f} positive")
    print(f"  Validation: {len(X_val)} sequences, {np.mean(y_val):.3f} positive") 
    print(f"  Test: {len(X_test)} sequences, {np.mean(y_test):.3f} positive")
    
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
                              show_trial_boundaries=use_trial_split)
    
    # Initialize SVM classifier
    print(f"\n{step_num+2}. Initializing SVM classifier...")
    n_neurons = sequences.shape[2]
    print(f"Input size: {n_neurons} neurons")
    print(f"Sequence length: {sequence_length} time bins")
    print(f"Flattened feature size: {sequence_length * n_neurons}")
    print(f"Kernel: {kernel}")
    print(f"C: {C}")
    print(f"Gamma: {gamma}")
    print(f"Grid search: {use_grid_search}")
    
    classifier = SVMClassifier(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight='balanced',
        random_state=random_state
    )
    
    # Train model
    print(f"\n{step_num+3}. Training SVM model...")
    history = classifier.train_model(
        X_train, y_train, X_val, y_val,
        use_grid_search=use_grid_search,
        verbose=True
    )
    
    # Evaluate model
    print(f"\n{step_num+4}. Evaluating SVM model on test set...")
    metrics, y_true, y_pred, y_prob = classifier.evaluate(X_test, y_test)
    
    print("Test Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    print("\n" + "="*60)
    print("SVM DEMO COMPLETE!")
    print("="*60)
    
    # Create comprehensive plots after training is complete
    print(f"\n{step_num+5}. Generating evaluation plots...")
    plot_classifier_results(history, y_true, y_pred, y_prob, "SVM Classifier", mouse_id, session_id, event_column)
    
    return classifier, metrics, history


if __name__ == "__main__":
    # Example usage - optimized for neural data with cluster filtering
    run_svm_demo(
        mouse_id="7004",
        session_id="o3", 
        experiment="clickbait-odor",
        event_column="flip_state",
        base_path="S:\\",
        sequence_length=10,  # Number of time bins per sequence
        bin_size_ms=500,  # Coarser temporal resolution  
        stride=10,  # Dense sampling
        test_size=0.2,
        kernel='rbf',  # SVM kernel
        C=1.0,  # Regularization parameter
        gamma='scale',  # Kernel coefficient
        use_grid_search=True,  # Use grid search for hyperparameter tuning
        centroid_y_threshold=np.inf,  # Y-axis halfway point = 1968//2
        cluster_filter='best_channel > 16',  # Filter by cluster attribute
        exclude_final_flip=False,  # Exclude final flip_state period
        filter_boundaries=False,  # Set to True to skip boundary-crossing sequences
        use_trial_split=True,  # Use trial-based splitting for better temporal structure
        train_size=0.6,  # 60% of trials for training
        val_size=0.2,  # 20% of trials for validation (20% remaining for test)
        random_state=42  # For reproducible splits
    )
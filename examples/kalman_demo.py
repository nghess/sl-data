"""
Demo script for neural population position decoding using Kalman filtering.

This script demonstrates how to:
1. Load neural data using SessionData
2. Create population activity matrices
3. Extract animal position from behavioral data
4. Train a Kalman filter for position decoding
5. Evaluate decoding performance

Code by Nate Gonzales-Hess, December 2024.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import seaborn as sns
from scipy.stats import pearsonr

from sldata import SessionData


class KalmanPositionDecoder:
    """
    Simple Kalman filter-inspired position decoder for neural population data.
    
    Uses a linear regression approach with temporal smoothing for position prediction
    from binned population activity.
    """
    
    def __init__(self, alpha=0.01, temporal_window=5):
        """
        Initialize the position decoder.
        
        Parameters:
        -----------
        alpha : float
            Regularization parameter for Ridge regression
        temporal_window : int
            Number of previous time bins to include for temporal smoothing
        """
        self.alpha = alpha
        self.temporal_window = temporal_window
        self.decoder_x = None
        self.decoder_y = None
        self.is_fitted = False
        
    def _create_temporal_features(self, population_matrix):
        """Create temporal features by concatenating previous time bins."""
        n_time_bins, n_neurons = population_matrix.shape
        
        # Create features including previous time bins
        n_features = n_neurons * self.temporal_window
        features = np.zeros((n_time_bins - self.temporal_window + 1, n_features))
        
        for i in range(self.temporal_window, n_time_bins):
            # Concatenate current and previous time bins
            temporal_data = population_matrix[i-self.temporal_window+1:i+1].flatten()
            features[i-self.temporal_window+1] = temporal_data
            
        return features
    
    def fit(self, population_matrix, positions_x, positions_y):
        """
        Train the position decoder.
        
        Parameters:
        -----------
        population_matrix : np.ndarray
            Population activity matrix [time_bins x neurons]
        positions_x : np.ndarray
            X coordinates for each time bin
        positions_y : np.ndarray
            Y coordinates for each time bin
        """
        print("Training Kalman position decoder...")
        
        # Create temporal features
        features = self._create_temporal_features(population_matrix)
        
        # Align positions with temporal features
        aligned_positions_x = positions_x[self.temporal_window-1:]
        aligned_positions_y = positions_y[self.temporal_window-1:]
        
        # Train separate decoders for X and Y
        self.decoder_x = Ridge(alpha=self.alpha)
        self.decoder_y = Ridge(alpha=self.alpha)
        
        self.decoder_x.fit(features, aligned_positions_x)
        self.decoder_y.fit(features, aligned_positions_y)
        
        self.is_fitted = True
        
        print(f"  Decoder trained with {features.shape[0]} samples")
        print(f"  Feature dimensionality: {features.shape[1]} ({self.temporal_window} bins × {features.shape[1]//self.temporal_window} neurons)")
        
    def predict(self, population_matrix):
        """
        Decode positions from population activity.
        
        Parameters:
        -----------
        population_matrix : np.ndarray
            Population activity matrix [time_bins x neurons]
            
        Returns:
        --------
        predicted_x : np.ndarray
            Predicted X coordinates
        predicted_y : np.ndarray  
            Predicted Y coordinates
        """
        if not self.is_fitted:
            raise ValueError("Decoder must be fitted before prediction")
            
        # Create temporal features
        features = self._create_temporal_features(population_matrix)
        
        # Predict positions
        predicted_x = self.decoder_x.predict(features)
        predicted_y = self.decoder_y.predict(features)
        
        return predicted_x, predicted_y
    
    def score(self, population_matrix, true_x, true_y):
        """
        Calculate decoding performance metrics.
        
        Parameters:
        -----------
        population_matrix : np.ndarray
            Population activity matrix [time_bins x neurons]
        true_x : np.ndarray
            True X coordinates
        true_y : np.ndarray
            True Y coordinates
            
        Returns:
        --------
        metrics : dict
            Dictionary containing performance metrics
        """
        pred_x, pred_y = self.predict(population_matrix)
        
        # Align true positions with predictions
        aligned_true_x = true_x[self.temporal_window-1:]
        aligned_true_y = true_y[self.temporal_window-1:]
        
        # Calculate metrics
        mse_x = mean_squared_error(aligned_true_x, pred_x)
        mse_y = mean_squared_error(aligned_true_y, pred_y)
        
        r2_x = r2_score(aligned_true_x, pred_x)
        r2_y = r2_score(aligned_true_y, pred_y)
        
        # Correlation coefficients
        corr_x, _ = pearsonr(aligned_true_x, pred_x)
        corr_y, _ = pearsonr(aligned_true_y, pred_y)
        
        # Combined position error (Euclidean distance)
        position_errors = np.sqrt((aligned_true_x - pred_x)**2 + (aligned_true_y - pred_y)**2)
        mean_position_error = np.mean(position_errors)
        
        return {
            'mse_x': mse_x,
            'mse_y': mse_y,
            'r2_x': r2_x,
            'r2_y': r2_y,
            'correlation_x': corr_x,
            'correlation_y': corr_y,
            'mean_position_error': mean_position_error,
            'position_errors': position_errors
        }


def plot_decoding_results(true_x, true_y, pred_x, pred_y, time_bins, metrics, 
                         mouse_id="", session_id=""):
    """
    Create comprehensive plots for position decoding evaluation.
    
    Parameters:
    -----------
    true_x, true_y : np.ndarray
        True positions
    pred_x, pred_y : np.ndarray  
        Predicted positions
    time_bins : np.ndarray
        Time bin centers
    metrics : dict
        Performance metrics
    mouse_id, session_id : str
        Identifiers for plot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Position trajectory comparison
    axes[0, 0].plot(true_x, true_y, 'b-', alpha=0.7, linewidth=2, label='True trajectory')
    axes[0, 0].plot(pred_x, pred_y, 'r-', alpha=0.7, linewidth=1, label='Predicted trajectory')
    axes[0, 0].scatter(true_x[0], true_y[0], color='blue', s=100, marker='o', label='Start')
    axes[0, 0].scatter(true_x[-1], true_y[-1], color='blue', s=100, marker='s', label='End')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position') 
    axes[0, 0].set_title(f'{mouse_id}-{session_id} Position Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # 2. Time series comparison (X and Y over time)
    time_s = time_bins / 1000  # Convert to seconds
    axes[0, 1].plot(time_s, true_x, 'b-', label='True X', alpha=0.8, linewidth=2)
    axes[0, 1].plot(time_s, pred_x, 'r--', label='Pred X', alpha=0.8, linewidth=1)
    axes[0, 1].plot(time_s, true_y, 'g-', label='True Y', alpha=0.8, linewidth=2)
    axes[0, 1].plot(time_s, pred_y, 'm--', label='Pred Y', alpha=0.8, linewidth=1)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position')
    axes[0, 1].set_title('Position vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction scatter plots
    axes[1, 0].scatter(true_x, pred_x, alpha=0.6, s=20, label=f'X (r={metrics["correlation_x"]:.3f})')
    axes[1, 0].plot([true_x.min(), true_x.max()], [true_x.min(), true_x.max()], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('True X Position')
    axes[1, 0].set_ylabel('Predicted X Position')
    axes[1, 0].set_title('X Position Predictions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    # 4. Error distribution
    position_errors = metrics['position_errors']
    axes[1, 1].hist(position_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(metrics['mean_position_error'], color='red', linestyle='--', 
                      label=f'Mean error: {metrics["mean_position_error"]:.1f}')
    axes[1, 1].set_xlabel('Position Error (distance units)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Position Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def prepare_position_data(session: SessionData, bin_size_ms: float = 100.0,
                         min_time_ms: float = 0, max_time_ms: float = None) -> tuple:
    """
    Prepare data for position decoding from SessionData.
    
    Parameters:
    -----------
    session : SessionData
        Loaded session data
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
    positions_x : np.ndarray
        X coordinates for each time bin
    positions_y : np.ndarray
        Y coordinates for each time bin  
    time_bins : np.ndarray
        Time bin centers
    """
    print(f"Preparing position decoding data...")
    
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
    
    # Extract position data
    if 'nose.x' in session.events.columns and 'nose.y' in session.events.columns:
        position_x_col = 'nose.x'
        position_y_col = 'nose.y'
    # if 'bonsai_centroid_x' in session.events.columns and 'bonsai_centroid_y' in session.events.columns:
    #     position_x_col = 'bonsai_centroid_x'
    #     position_y_col = 'bonsai_centroid_y'
    else:
        raise ValueError("No position data found. Need 'bonsai_centroid_x/y' or 'nose.x/y' columns")
    
    # Get position values for each time bin using mean aggregation
    positions_x = session.create_event_colormap(
        time_bins, position_x_col, aggregation='mean'
    )
    positions_y = session.create_event_colormap(
        time_bins, position_y_col, aggregation='mean'
    )
    
    # Filter out invalid position data (likely tracking failures)
    print(f"Raw position data:")
    print(f"  Position range - X: {positions_x.min():.1f} to {positions_x.max():.1f}")
    print(f"  Position range - Y: {positions_y.min():.1f} to {positions_y.max():.1f}")
    
    # Identify invalid positions (zeros or extreme outliers)
    zero_positions = (positions_x == 0) & (positions_y == 0)
    n_zeros = np.sum(zero_positions)
    print(f"  Found {n_zeros} time bins with (0,0) positions ({n_zeros/len(time_bins)*100:.1f}%)")
    
    if n_zeros > 0:
        print("  Filtering out invalid position data...")
        
        # Create mask for valid positions
        valid_mask = ~zero_positions
        
        # Also filter extreme outliers (optional)
        x_median = np.median(positions_x[valid_mask])
        y_median = np.median(positions_y[valid_mask])
        x_mad = np.median(np.abs(positions_x[valid_mask] - x_median))
        y_mad = np.median(np.abs(positions_y[valid_mask] - y_median))
        
        # Mark as invalid if position is more than 10 MADs from median
        outlier_threshold = 10
        x_outliers = np.abs(positions_x - x_median) > outlier_threshold * x_mad
        y_outliers = np.abs(positions_y - y_median) > outlier_threshold * y_mad
        outlier_mask = x_outliers | y_outliers
        
        n_outliers = np.sum(outlier_mask)
        if n_outliers > 0:
            print(f"  Found {n_outliers} outlier positions")
            valid_mask = valid_mask & ~outlier_mask
        
        # Filter all data to valid positions only
        population_matrix = population_matrix[valid_mask]
        positions_x = positions_x[valid_mask]
        positions_y = positions_y[valid_mask]
        time_bins = time_bins[valid_mask]
        
        print(f"  Kept {np.sum(valid_mask)}/{len(valid_mask)} time bins after filtering")
    
    print(f"Final data preparation results:")
    print(f"  Time bins created: {len(time_bins)}")
    print(f"  Population matrix shape: {population_matrix.shape}")
    print(f"  Position range - X: {positions_x.min():.1f} to {positions_x.max():.1f}")
    print(f"  Position range - Y: {positions_y.min():.1f} to {positions_y.max():.1f}")
    print(f"  Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f} ms")
    
    return population_matrix, positions_x, positions_y, time_bins


def run_kalman_demo(mouse_id: str, session_id: str, experiment: str,
                   base_path: str = "S:\\",
                   bin_size_ms: float = 100.0,
                   test_size: float = 0.2,
                   alpha: float = 0.01,
                   temporal_window: int = 5,
                   cluster_filter: str = None):
    """
    Run complete Kalman position decoding demo.
    
    Parameters:
    -----------
    mouse_id : str
        Mouse identifier
    session_id : str  
        Session identifier
    experiment : str
        Experiment name
    base_path : str
        Base data path
    bin_size_ms : float
        Time bin size in milliseconds
    test_size : float
        Fraction of data for testing
    alpha : float
        Regularization parameter for Ridge regression
    temporal_window : int
        Number of previous time bins to include
    cluster_filter : str, optional
        Filter expression for clusters
    """
    print("="*60)
    print("KALMAN POSITION DECODING DEMO")
    print("="*60)
    
    # Load session data
    print(f"\n1. Loading session data: {mouse_id}_{session_id}")
    try:
        session = SessionData(mouse_id, session_id, experiment, 
                            base_path=base_path, verbose=True)
    except Exception as e:
        print(f"Error loading session data: {e}")
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
    
    # Prepare position decoding data
    step_num = 3 if cluster_filter is not None else 2
    print(f"\n{step_num}. Preparing position decoding data...")
    population_matrix, positions_x, positions_y, time_bins = prepare_position_data(
        session, bin_size_ms
    )
    
    # Split data temporally to avoid leakage
    print(f"\n{step_num+1}. Splitting data temporally (test size: {test_size})...")
    n_samples = len(population_matrix)
    split_idx = int(n_samples * (1 - test_size))
    
    # Training data (first portion)
    X_train = population_matrix[:split_idx]
    y_train_x = positions_x[:split_idx]
    y_train_y = positions_y[:split_idx]
    
    # Test data (last portion)  
    X_test = population_matrix[split_idx:]
    y_test_x = positions_x[split_idx:]
    y_test_y = positions_y[split_idx:]
    time_bins_test = time_bins[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize and train decoder
    print(f"\n{step_num+2}. Training Kalman position decoder...")
    decoder = KalmanPositionDecoder(alpha=alpha, temporal_window=temporal_window)
    decoder.fit(X_train, y_train_x, y_train_y)
    
    # Evaluate decoder
    print(f"\n{step_num+3}. Evaluating position decoder...")
    metrics = decoder.score(X_test, y_test_x, y_test_y)
    pred_x, pred_y = decoder.predict(X_test)
    
    print("Position Decoding Performance:")
    print(f"  X Correlation:  {metrics['correlation_x']:.4f}")
    print(f"  Y Correlation:  {metrics['correlation_y']:.4f}")
    print(f"  X R²:          {metrics['r2_x']:.4f}")
    print(f"  Y R²:          {metrics['r2_y']:.4f}")
    print(f"  Mean Position Error: {metrics['mean_position_error']:.2f} units")
    
    print("\n" + "="*60)
    print("KALMAN DEMO COMPLETE!")
    print("="*60)
    
    # Create comprehensive plots
    print(f"\n{step_num+4}. Generating evaluation plots...")
    
    # Align test data with predictions (account for temporal window)
    aligned_true_x = y_test_x[temporal_window-1:]
    aligned_true_y = y_test_y[temporal_window-1:]
    aligned_time_bins = time_bins_test[temporal_window-1:]
    
    plot_decoding_results(aligned_true_x, aligned_true_y, pred_x, pred_y, 
                         aligned_time_bins, metrics, mouse_id, session_id)
    
    return decoder, metrics


if __name__ == "__main__":
    # Example usage
    run_kalman_demo(
        mouse_id="7004",
        session_id="m4", 
        experiment="clickbait-motivate",
        base_path="S:\\",
        bin_size_ms=33.3,           # Time resolution for decoding
        test_size=0.3,               # Fraction for testing
        alpha=0.1,                   # Regularization strength
        temporal_window=30#,           # Number of time bins for temporal features
        #cluster_filter='best_channel >= 16'  # Filter by cluster quality
    )
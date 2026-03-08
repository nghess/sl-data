"""
Kalman filter position decoding from neural population activity.

Implements a proper linear Kalman filter for decoding animal 2D position
from binned population activity, using 30 ms bins to match the temporal
resolution of the video tracking (≈33 fps).

State-space model
-----------------
    State:       z_t = [x, y, vx, vy]^T          (position + velocity)
    Transition:  z_t  = A * z_{t-1} + w_t,   w_t ~ N(0, W)
    Observation: n_t  = H * z_t   + q_t,     q_t ~ N(0, Q)

where n_t is the N-dimensional population firing-rate vector.

All parameters (A, W, H, Q) are estimated from training data via
ordinary least squares.  The filter is then run forward on held-out
test data using the standard Kalman predict/update recursion.

References
----------
Wu W, et al. (2006) Bayesian Population Decoding of Motor Cortical Activity
Using a Kalman Filter. Neural Computation, 18(1), 80-118.

Code by Nate Gonzales-Hess.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

from sldata import SessionData


# ---------------------------------------------------------------------------
# Kalman filter class
# ---------------------------------------------------------------------------

class KalmanFilterDecoder:
    """
    Linear Kalman filter decoder for 2-D position from population activity.

    Parameters are learned from training data; decoding is done by running
    the standard Kalman predict/update recursion on the test set.

    Parameters
    ----------
    neural_smooth_sigma : float
        Std of Gaussian kernel (in bins) applied to neural activity before
        fitting H and before decoding.  Equivalent to temporal integration
        that improves SNR without sacrificing much spatial resolution.
        Set to 0 to disable.  Default: 2.
    pos_smooth_sigma : float
        Std of Gaussian kernel (in bins) applied to position labels before
        fitting H.  Reduces tracking jitter in the SLEAP output.
        Set to 0 to disable.  Default: 2.

    Attributes
    ----------
    A : np.ndarray, shape (2, 2)
        State transition matrix.
    W : np.ndarray, shape (2, 2)
        Process noise covariance.
    H : np.ndarray, shape (n_neurons, 2)
        Observation matrix.
    Q : np.ndarray, shape (n_neurons, n_neurons)
        Observation noise covariance (diagonal approximation).
    """

    def __init__(self, neural_smooth_sigma: float = 2.0,
                 pos_smooth_sigma: float = 2.0):
        self.neural_smooth_sigma = neural_smooth_sigma
        self.pos_smooth_sigma    = pos_smooth_sigma
        self.A = None
        self.W = None
        self.H = None
        self.Q = None
        # Position normalisation stats (set during fit, used to denormalise decoded output)
        self.pos_x_mean = 0.0
        self.pos_x_std  = 1.0
        self.pos_y_mean = 0.0
        self.pos_y_std  = 1.0
        self.is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, population_matrix: np.ndarray,
            positions_x: np.ndarray,
            positions_y: np.ndarray) -> None:
        """
        Estimate Kalman filter parameters from training data.

        Parameters
        ----------
        population_matrix : np.ndarray, shape (n_time, n_neurons)
            Binned, z-scored population firing rates.
        positions_x : np.ndarray, shape (n_time,)
            Animal nose X coordinate for each time bin (pixels).
        positions_y : np.ndarray, shape (n_time,)
            Animal nose Y coordinate for each time bin (pixels).
        """
        n_time, n_neurons = population_matrix.shape

        # --- Optional smoothing ---
        # Smoothing neural activity reduces per-bin noise (↓ Q) and makes the
        # H estimate more reliable.  Smoothing positions removes SLEAP tracking
        # jitter so that H captures genuine spatial tuning rather than label noise.
        if self.neural_smooth_sigma > 0:
            population_matrix = gaussian_filter1d(
                population_matrix.astype(float),
                sigma=self.neural_smooth_sigma, axis=0
            )
        if self.pos_smooth_sigma > 0:
            positions_x = gaussian_filter1d(positions_x.astype(float),
                                            sigma=self.pos_smooth_sigma)
            positions_y = gaussian_filter1d(positions_y.astype(float),
                                            sigma=self.pos_smooth_sigma)

        # Normalise positions to zero-mean / unit-std so that the state and
        # observation spaces are on the same scale as the z-scored neural data.
        # Without this, H entries are O(1/pixel) ≈ 1e-3, making the Kalman
        # gain negligibly small and letting the state prediction diverge.
        self.pos_x_mean = float(np.mean(positions_x))
        self.pos_x_std  = float(np.std(positions_x))
        self.pos_y_mean = float(np.mean(positions_y))
        self.pos_y_std  = float(np.std(positions_y))

        px = (positions_x - self.pos_x_mean) / self.pos_x_std
        py = (positions_y - self.pos_y_mean) / self.pos_y_std

        # State matrix Z: rows are [x, y] — position only (no velocity).
        # Omitting velocity avoids divergence from noisy finite-difference
        # estimates; the transition model becomes a random walk (A ≈ I).
        Z = np.column_stack([px, py])  # (n_time, 2)

        # ------------------------------------------------------------------
        # Estimate state transition matrix A
        # Model: z_t = A * z_{t-1}  =>  Z[1:] ≈ Z[:-1] @ A.T
        # ------------------------------------------------------------------
        Z_curr = Z[1:]   # (n_time-1, 2)
        Z_prev = Z[:-1]  # (n_time-1, 2)

        A_T, _, _, _ = np.linalg.lstsq(Z_prev, Z_curr, rcond=None)
        self.A = A_T.T   # (2, 2)

        # Process noise covariance W from state-transition residuals
        residuals_state = Z_curr - Z_prev @ A_T   # (n_time-1, 2)
        self.W = np.cov(residuals_state.T) + np.eye(2) * 1e-6  # (2, 2)

        # ------------------------------------------------------------------
        # Estimate observation matrix H via Ridge regression
        # Model: n_t = H * z_t  =>  population_matrix ≈ Z @ H.T
        # Ridge avoids near-zero H entries for neurons with weak spatial tuning.
        # ------------------------------------------------------------------
        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(Z, population_matrix)      # fits all neurons jointly
        self.H = ridge.coef_                 # (n_neurons, 2)

        # Observation noise covariance Q (diagonal: independent neuron noise)
        residuals_obs = population_matrix - Z @ self.H.T   # (n_time, n_neurons)
        obs_var = np.var(residuals_obs, axis=0)        # (n_neurons,)
        obs_var = np.maximum(obs_var, 1e-6)            # avoid zero variance
        self.Q = np.diag(obs_var)                      # (n_neurons, n_neurons)

        self.is_fitted = True
        print(f"Kalman filter fitted: {n_time} training bins, {n_neurons} neurons")
        print(f"  State noise std (x,y): {np.sqrt(self.W[0,0]):.3f}, {np.sqrt(self.W[1,1]):.3f}")
        print(f"  Median obs noise std:  {np.median(np.sqrt(obs_var)):.3f}")

    # ------------------------------------------------------------------
    def decode(self, population_matrix: np.ndarray,
               initial_state: np.ndarray = None,
               initial_cov: np.ndarray = None):
        """
        Decode position from population activity using the Kalman filter.

        Parameters
        ----------
        population_matrix : np.ndarray, shape (n_time, n_neurons)
            Binned, z-scored population firing rates for the test period.
        initial_state : np.ndarray, shape (2,), optional
            Initial [x, y] in normalised units.  Defaults to zeros.
        initial_cov : np.ndarray, shape (2, 2), optional
            Initial state covariance.  Defaults to 10 × I.

        Returns
        -------
        predicted_x : np.ndarray, shape (n_time,)
        predicted_y : np.ndarray, shape (n_time,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before decode().")

        # Apply the same neural smoothing used during fitting
        if self.neural_smooth_sigma > 0:
            population_matrix = gaussian_filter1d(
                population_matrix.astype(float),
                sigma=self.neural_smooth_sigma, axis=0
            )

        n_time = population_matrix.shape[0]
        A, W, H, Q = self.A, self.W, self.H, self.Q
        n_state = A.shape[0]
        I = np.eye(n_state)

        # Initialise state and covariance (in normalised position units)
        z = np.zeros(n_state) if initial_state is None else initial_state.copy()
        P = np.eye(n_state) * 10.0 if initial_cov is None else initial_cov.copy()

        pred_x_norm = np.empty(n_time)
        pred_y_norm = np.empty(n_time)

        for t in range(n_time):
            # --- Predict ---
            z_pred = A @ z
            P_pred = A @ P @ A.T + W

            # --- Update ---
            # Innovation covariance S = H P H^T + Q  (n_neurons × n_neurons)
            HP = H @ P_pred                          # (n_neurons, n_state)
            S  = HP @ H.T + Q                        # (n_neurons, n_neurons)

            # Kalman gain K = P H^T S^{-1}  (n_state × n_neurons)
            K = np.linalg.solve(S, HP).T             # (n_state, n_neurons)

            # State update
            innovation = population_matrix[t] - H @ z_pred
            z = z_pred + K @ innovation

            # Joseph stabilized covariance update — keeps P positive-definite
            # under floating-point errors: P = (I-KH) P (I-KH)^T + K Q K^T
            IKH = I - K @ H
            P = IKH @ P_pred @ IKH.T + K @ Q @ K.T

            pred_x_norm[t] = z[0]
            pred_y_norm[t] = z[1]

        # Denormalise back to pixel coordinates
        predicted_x = pred_x_norm * self.pos_x_std + self.pos_x_mean
        predicted_y = pred_y_norm * self.pos_y_std + self.pos_y_mean

        return predicted_x, predicted_y

    # ------------------------------------------------------------------
    def decode_linear(self, population_matrix: np.ndarray):
        """
        Direct Ridge regression decode (no Kalman smoothing).

        Fits a ridge regression from neural activity → position on the
        training data and applies it directly to the test set.  Use this
        as a diagnostic baseline: if Kalman is not beating this, something
        is wrong with the dynamics model.

        Returns
        -------
        pred_x, pred_y : np.ndarray, shape (n_time,)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before decode_linear().")

        if self.neural_smooth_sigma > 0:
            population_matrix = gaussian_filter1d(
                population_matrix.astype(float),
                sigma=self.neural_smooth_sigma, axis=0
            )

        # W = (H^T Q^{-1} H)^{-1} H^T Q^{-1}  — the Kalman-optimal linear decoder
        # at steady state (P→0).  Here we shortcut by fitting Ridge directly.
        ridge_x = Ridge(alpha=1.0, fit_intercept=True)
        ridge_y = Ridge(alpha=1.0, fit_intercept=True)

        # Use the stored H to construct a decode matrix from the training
        # relationship.  We fit on the normalised training data (stored in H)
        # via a closed-form ridge decoder: W = H^T (H H^T + alpha I)^{-1}.
        # For a clean implementation just reuse fit data stored internally.
        # Here we run a simple per-step linear prediction using H pinv.
        Q_inv_diag = 1.0 / np.diag(self.Q)          # (n_neurons,)
        HtQinv  = self.H.T * Q_inv_diag[np.newaxis, :]  # (2, n_neurons)
        W_dec   = np.linalg.solve(
            HtQinv @ self.H + np.eye(2),
            HtQinv
        )                                             # (2, n_neurons)

        pred_norm = (W_dec @ population_matrix.T).T  # (n_time, 2)
        pred_x = pred_norm[:, 0] * self.pos_x_std + self.pos_x_mean
        pred_y = pred_norm[:, 1] * self.pos_y_std + self.pos_y_mean
        return pred_x, pred_y


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(session: SessionData, bin_size_ms: float = 30.0,
                 x_col: str = 'nose_x', y_col: str = 'nose_y'):
    """
    Bin neural activity and interpolate position to match each bin.

    Parameters
    ----------
    session : SessionData
    bin_size_ms : float
        Bin width in milliseconds.  Default 30 ms to match ~33 fps video.
    x_col : str
        Column name in session.events for the X coordinate.  Default 'nose_x'.
    y_col : str
        Column name in session.events for the Y coordinate.  Default 'nose_y'.

    Returns
    -------
    population_matrix : np.ndarray, shape (n_valid_bins, n_neurons)
    positions_x       : np.ndarray, shape (n_valid_bins,)
    positions_y       : np.ndarray, shape (n_valid_bins,)
    time_bins         : np.ndarray, shape (n_valid_bins,)  [ms]
    """
    pop_matrix, time_bins, _ = session.create_population_raster(
        bin_size_ms=bin_size_ms, zscore_neurons=True
    )
    population_matrix = pop_matrix.T  # (n_time, n_neurons)

    available = list(session.events.columns)
    if x_col not in available:
        raise ValueError(
            f"Column '{x_col}' not found in session.events.\n"
            f"Available columns: {available}"
        )
    if y_col not in available:
        raise ValueError(
            f"Column '{y_col}' not found in session.events.\n"
            f"Available columns: {available}"
        )

    print(f"Using position columns: x='{x_col}', y='{y_col}'")
    positions_x = session.create_event_colormap(time_bins, x_col, aggregation='mean')
    positions_y = session.create_event_colormap(time_bins, y_col, aggregation='mean')

    # Drop bins where position is (0, 0) — typically tracking dropouts
    valid = ~((positions_x == 0) & (positions_y == 0))
    n_dropped = np.sum(~valid)
    if n_dropped:
        print(f"Dropped {n_dropped}/{len(valid)} bins with (0,0) positions")

    population_matrix = population_matrix[valid]
    positions_x      = positions_x[valid]
    positions_y      = positions_y[valid]
    time_bins        = time_bins[valid]

    print(f"Data prepared: {population_matrix.shape[0]} bins × "
          f"{population_matrix.shape[1]} neurons")
    print(f"Position range — X: [{positions_x.min():.0f}, {positions_x.max():.0f}]  "
          f"Y: [{positions_y.min():.0f}, {positions_y.max():.0f}]  (px)")

    return population_matrix, positions_x, positions_y, time_bins


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(true_x, true_y, pred_x, pred_y, time_s,
                 corr_x, corr_y, r2_x, r2_y, mean_err,
                 mouse_id, session_id):
    """Four-panel summary of Kalman filter decoding performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Kalman Filter Position Decoding — {mouse_id}_{session_id}",
        fontsize=13, fontweight='bold'
    )

    # 1. Spatial trajectory
    ax = axes[0, 0]
    ax.plot(true_x, true_y, lw=1.5, alpha=0.7, color='steelblue', label='True')
    ax.plot(pred_x, pred_y, lw=1.0, alpha=0.7, color='tomato',    label='Decoded')
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.set_title('Position trajectory')
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # 2. Time series
    ax = axes[0, 1]
    ax.plot(time_s, true_x, lw=1.5, alpha=0.8, color='steelblue',  label='True X')
    ax.plot(time_s, pred_x, lw=1.0, alpha=0.8, color='tomato',
            ls='--', label='Decoded X')
    ax.plot(time_s, true_y, lw=1.5, alpha=0.8, color='seagreen',   label='True Y')
    ax.plot(time_s, pred_y, lw=1.0, alpha=0.8, color='darkorange',
            ls='--', label='Decoded Y')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Position (px)')
    ax.set_title('Position vs time')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. True vs decoded scatter (X)
    ax = axes[1, 0]
    ax.scatter(true_x, pred_x, s=4, alpha=0.3, color='steelblue')
    lims = [min(true_x.min(), pred_x.min()), max(true_x.max(), pred_x.max())]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlabel('True X (px)'); ax.set_ylabel('Decoded X (px)')
    ax.set_title(f'X  r = {corr_x:.3f},  R² = {r2_x:.3f}')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # 4. Euclidean error distribution
    ax = axes[1, 1]
    errors = np.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
    ax.hist(errors, bins=40, color='slategray', edgecolor='white', alpha=0.85)
    ax.axvline(mean_err, color='red', ls='--',
               label=f'Mean = {mean_err:.1f} px')
    ax.set_xlabel('Euclidean position error (px)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error distribution  (Y: r = {corr_y:.3f}, R² = {r2_y:.3f})')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_kalman_filter(mouse_id: str, session_id: str, experiment: str,
                      base_path: str = "S:\\",
                      bin_size_ms: float = 30.0,
                      test_fraction: float = 0.2,
                      neural_smooth_sigma: float = 2.0,
                      pos_smooth_sigma: float = 2.0,
                      x_col: str = 'nose_x',
                      y_col: str = 'nose_y',
                      cluster_filter: str = None):
    """
    End-to-end Kalman filter position decoding.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier (e.g. '7004').
    session_id : str
        Session identifier (e.g. 'm4').
    experiment : str
        Experiment name (e.g. 'clickbait-motivate').
    base_path : str
        Root data directory.
    bin_size_ms : float
        Neural activity bin width in ms.  Default 30 ms ≈ 33 fps video.
    test_fraction : float
        Fraction of data reserved for testing (temporal split, no shuffle).
    neural_smooth_sigma : float
        Gaussian smoothing applied to neural activity (bins).  Reduces
        observation noise; equivalent to temporal integration.  Default 2.
    pos_smooth_sigma : float
        Gaussian smoothing applied to position labels during H fitting.
        Reduces SLEAP tracking jitter.  Default 2.
    x_col : str
        Column in session.events to use as the X coordinate.  Default 'nose_x'.
    y_col : str
        Column in session.events to use as the Y coordinate.  Default 'nose_y'.
    cluster_filter : str, optional
        Filter expression passed to session.filter_clusters()
        e.g. 'best_channel >= 16'.

    Returns
    -------
    decoder : KalmanFilterDecoder
    metrics : dict
        r2_x, r2_y, corr_x, corr_y, mean_position_error
    """
    print("=" * 60)
    print("KALMAN FILTER POSITION DECODING")
    print("=" * 60)

    # --- Load session ---
    print(f"\n1. Loading session {mouse_id}_{session_id}")
    session = SessionData(mouse_id, session_id, experiment,
                          base_path=base_path, verbose=True)

    if cluster_filter is not None:
        print(f"\n2. Filtering clusters: {cluster_filter}")
        n_before = session.n_clusters
        session = session.filter_clusters(cluster_filter)
        print(f"   {n_before} → {session.n_clusters} clusters")
        if session.n_clusters == 0:
            raise ValueError(f"No clusters remain after filter: {cluster_filter}")

    # --- Prepare data ---
    step = 3 if cluster_filter else 2
    print(f"\n{step}. Binning neural activity ({bin_size_ms} ms bins)")
    pop_matrix, pos_x, pos_y, time_bins = prepare_data(session, bin_size_ms,
                                                        x_col=x_col, y_col=y_col)

    # --- Temporal train / test split ---
    n = len(pop_matrix)
    split = int(n * (1 - test_fraction))
    X_train, X_test     = pop_matrix[:split], pop_matrix[split:]
    y_train_x, y_test_x = pos_x[:split], pos_x[split:]
    y_train_y, y_test_y = pos_y[:split], pos_y[split:]
    time_test = time_bins[split:]

    print(f"\n{step+1}. Train / test split")
    print(f"   Training bins: {split}  ({split * bin_size_ms / 1000:.1f} s)")
    print(f"   Test bins:     {n - split}  ({(n - split) * bin_size_ms / 1000:.1f} s)")

    # --- Fit Kalman filter ---
    print(f"\n{step+2}. Fitting Kalman filter  "
          f"(neural_smooth={neural_smooth_sigma}, pos_smooth={pos_smooth_sigma})")
    decoder = KalmanFilterDecoder(neural_smooth_sigma=neural_smooth_sigma,
                                  pos_smooth_sigma=pos_smooth_sigma)
    decoder.fit(X_train, y_train_x, y_train_y)

    # --- Linear baseline (diagnostic) ---
    lin_x, lin_y = decoder.decode_linear(X_test)
    lin_corr_x, _ = pearsonr(y_test_x, lin_x)
    lin_corr_y, _ = pearsonr(y_test_y, lin_y)
    print(f"\nLinear decode baseline (no Kalman smoothing):")
    print(f"  X  r = {lin_corr_x:.4f}   Y  r = {lin_corr_y:.4f}")
    print(f"  (If Kalman ≈ linear, the dynamics model is not helping.)")
    print(f"  (If both are weak, neural–position tuning may be limited.)")

    # --- Kalman decode ---
    print(f"\n{step+3}. Kalman decoding test set")
    pred_x, pred_y = decoder.decode(X_test)

    # --- Metrics ---
    corr_x, _ = pearsonr(y_test_x, pred_x)
    corr_y, _ = pearsonr(y_test_y, pred_y)
    r2_x      = r2_score(y_test_x, pred_x)
    r2_y      = r2_score(y_test_y, pred_y)
    mean_err  = float(np.mean(np.sqrt((y_test_x - pred_x) ** 2 +
                                       (y_test_y - pred_y) ** 2)))

    print("\nKalman decoding performance (test set):")
    print(f"  X  r = {corr_x:.4f}   R² = {r2_x:.4f}")
    print(f"  Y  r = {corr_y:.4f}   R² = {r2_y:.4f}")
    print(f"  Mean Euclidean error: {mean_err:.1f} px")

    metrics = {
        'corr_x': corr_x, 'corr_y': corr_y,
        'r2_x':   r2_x,   'r2_y':   r2_y,
        'mean_position_error': mean_err,
        'linear_corr_x': lin_corr_x,
        'linear_corr_y': lin_corr_y,
    }

    # --- Plot ---
    print(f"\n{step+4}. Plotting results")
    plot_results(
        y_test_x, y_test_y, pred_x, pred_y,
        time_test / 1000.0,          # ms → s
        corr_x, corr_y, r2_x, r2_y, mean_err,
        mouse_id, session_id
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return decoder, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_kalman_filter(
        mouse_id="7012",
        session_id="m10",
        experiment="clickbait-motivate",
        base_path="d:/data/",
        bin_size_ms=30.0,       # 30 ms ≈ video frame rate
        test_fraction=0.2,
        neural_smooth_sigma=1.0,  # Smooth neural activity over ~2 bins (60 ms)
        pos_smooth_sigma=1.0,     # Smooth position labels to reduce SLEAP jitter
        x_col='bonsai_centroid_x',
        y_col='bonsai_centroid_y',
    )

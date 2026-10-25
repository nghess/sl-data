"""
Kalman filter position decoding from neural population activity.

Implements a proper linear Kalman filter for decoding animal 2D position
from binned population activity, using 30 ms bins to match the temporal
resolution of the video tracking (≈33 fps).

State-space model
-----------------
    State:       z_t = [x, y]^T                  (position only)
    Transition:  z_t  = A * z_{t-1} + w_t,   w_t ~ N(0, W)
    Observation: n_t  = H * z_t   + q_t,     q_t ~ N(0, Q)

where n_t is the N-dimensional population firing-rate vector.

A is a diagonal AR(1) matrix diag(α_x, α_y) estimated from the lag-1
autocorrelation of each position coordinate.  α < 1 encodes the physical
constraint that animal position is a stationary (mean-reverting) process
in an enclosed arena, preventing drift accumulation.

H is estimated via Ridge regression; Q is diagonal (per-neuron obs noise).
The filter gain can optionally be fixed at its steady-state value K_inf by
solving the discrete algebraic Riccati equation (DARE), which eliminates
the high-gain transient at startup and reduces per-step computation.

References
----------
Wu W, et al. (2006) Bayesian Population Decoding of Motor Cortical Activity
Using a Kalman Filter. Neural Computation, 18(1), 80-118.

Code by Nate Gonzales-Hess.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2 as chi2_dist
from scipy.linalg import solve_discrete_are
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
        fitting H and before decoding.  Set to 0 to disable.  Default: 2.
    pos_smooth_sigma : float
        Std of Gaussian kernel (in bins) applied to position labels before
        fitting H.  Reduces SLEAP tracking jitter.  Set to 0 to disable.
        Default: 2.
    ar1_alpha : float or None
        AR(1) coefficient used for the state transition A = diag(α_x, α_y).
        If None (default), α is estimated per-axis from the lag-1
        autocorrelation of the training positions.  Set to a fixed value in
        (0, 1) to override (e.g. 0.98 for strong temporal smoothing).
    use_steady_state_gain : bool
        If True, pre-compute the optimal steady-state Kalman gain K_inf by
        solving the discrete algebraic Riccati equation (DARE) during fit().
        The recursive P update is then skipped in decode(), replacing the
        time-varying gain with a fixed K_inf.  Eliminates the high-gain
        transient at startup and speeds up decoding.  Default: False.
    bias_ema_lambda : float
        If > 0, subtract a running exponential moving average of the innovation
        before each Kalman update.  Removes slow drift caused by a shift in the
        test-set neural activity distribution relative to training.  τ ≈ 1/(1-λ)
        bins (e.g. λ=0.99 → τ≈100 bins≈3 s at 30 ms).  Default: 0.0 (off).

    Attributes
    ----------
    A : np.ndarray, shape (2, 2)
        Diagonal AR(1) state transition matrix diag(α_x, α_y).
    W : np.ndarray, shape (2, 2)
        Process noise covariance.
    H : np.ndarray, shape (n_neurons, 2)
        Observation matrix.
    Q : np.ndarray, shape (n_neurons, n_neurons)
        Observation noise covariance (diagonal approximation).
    K_inf : np.ndarray, shape (2, n_neurons) or None
        Steady-state Kalman gain (only set when use_steady_state_gain=True).
    P_inf : np.ndarray, shape (2, 2) or None
        Steady-state predicted covariance from DARE solution.
    """

    def __init__(self, neural_smooth_sigma: float = 2.0,
                 pos_smooth_sigma: float = 2.0,
                 ridge_alpha: float = 1.0,
                 obs_noise_scale: float = 1.0,
                 mahal_gate_percentile: float = 0.0,
                 ar1_alpha: float = None,
                 use_steady_state_gain: bool = False,
                 bias_ema_lambda: float = 0.0):
        self.neural_smooth_sigma   = neural_smooth_sigma
        self.pos_smooth_sigma      = pos_smooth_sigma
        self.ridge_alpha           = ridge_alpha
        self.obs_noise_scale       = obs_noise_scale
        # mahal_gate_percentile > 0 enables scalar Mahalanobis innovation gating.
        # The innovation v = n_t - H z_pred has squared Mahalanobis distance
        # chi2 = v^T S^{-1} v ~ chi2(n_neurons) under the model.  If chi2
        # exceeds the given percentile of that distribution, the innovation is
        # soft-scaled down so its effective Mahalanobis distance equals the
        # threshold.  This is more principled than per-neuron clipping because
        # it respects the full observation covariance S and targets whole-bin
        # outliers (noise bursts, tracking dropouts) rather than individual neurons.
        # Good starting value: 0.99.  Default 0.0 (off).
        self.mahal_gate_percentile = mahal_gate_percentile
        self.ar1_alpha             = ar1_alpha
        self.use_steady_state_gain = use_steady_state_gain
        # bias_ema_lambda > 0 enables online innovation bias correction.
        # A running EMA of the innovation vector tracks slow drift in the
        # neural-activity distribution and is subtracted each step before
        # the Kalman update.  λ controls the time constant: τ ≈ 1/(1-λ) bins
        # (e.g. λ=0.99 → τ≈100 bins≈3 s at 30 ms bins).  Default 0.0 (off).
        self.bias_ema_lambda       = bias_ema_lambda
        self.A     = None
        self.W     = None
        self.H     = None
        self.Q     = None
        self.K_inf = None   # steady-state gain (set when use_steady_state_gain=True)
        self.P_inf = None   # steady-state predicted covariance
        self.S_inf = None   # steady-state innovation covariance
        # Position normalisation stats (set during fit, used to denormalise decoded output)
        self.pos_x_mean = 0.0
        self.pos_x_std  = 1.0
        self.pos_y_mean = 0.0
        self.pos_y_std  = 1.0
        # Normalised position bounds for state clipping (set during fit)
        self.pos_x_min_norm = -np.inf
        self.pos_x_max_norm =  np.inf
        self.pos_y_min_norm = -np.inf
        self.pos_y_max_norm =  np.inf
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

        # Store arena bounds in normalised space for state clipping during decode
        self.pos_x_min_norm = float(np.min(px))
        self.pos_x_max_norm = float(np.max(px))
        self.pos_y_min_norm = float(np.min(py))
        self.pos_y_max_norm = float(np.max(py))

        # State matrix Z: rows are [x, y] — position only (no velocity).
        Z = np.column_stack([px, py])  # (n_time, 2)

        # ------------------------------------------------------------------
        # AR(1) state transition: z_t = A z_{t-1} + w_t,  A = diag(α_x, α_y)
        #
        # Animal position in an enclosed arena is a stationary (mean-reverting)
        # process — the animal cannot leave the arena.  Modelling this as an
        # AR(1) / Ornstein-Uhlenbeck process with α < 1 provides a restoring
        # force toward the arena centre (z = 0 in normalised coordinates) at
        # every step, preventing drift accumulation over long decode windows.
        #
        # A random walk (A = I) has no restoring force: any small systematic
        # bias in the Kalman gain accumulates without bound.  The AR(1) model
        # is the correct stationary prior for bounded exploration.
        #
        # α is estimated per-axis from the lag-1 autocorrelation of training
        # positions.  At 30 ms bins this is typically 0.95–0.99.
        # ------------------------------------------------------------------
        Z_curr = Z[1:]   # (n_time-1, 2)
        Z_prev = Z[:-1]  # (n_time-1, 2)

        if self.ar1_alpha is not None:
            alpha_x = alpha_y = float(self.ar1_alpha)
        else:
            alpha_x = float(np.corrcoef(px[:-1], px[1:])[0, 1])
            alpha_y = float(np.corrcoef(py[:-1], py[1:])[0, 1])
        # Clamp to (0, 1): negative values would be unusual/noisy; ≥ 1 unstable.
        alpha_x = float(np.clip(alpha_x, 0.0, 1.0 - 1e-4))
        alpha_y = float(np.clip(alpha_y, 0.0, 1.0 - 1e-4))
        self.A = np.diag([alpha_x, alpha_y])

        # Process noise W from AR(1) residuals
        residuals_state = Z_curr - Z_prev @ self.A.T   # (n_time-1, 2)
        self.W = np.cov(residuals_state.T) + np.eye(2) * 1e-6  # (2, 2)

        # ------------------------------------------------------------------
        # Estimate observation matrix H via Ridge regression
        # Model: n_t = H * z_t  =>  population_matrix ≈ Z @ H.T
        # Ridge avoids near-zero H entries for neurons with weak spatial tuning.
        # ------------------------------------------------------------------
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        ridge.fit(Z, population_matrix)      # fits all neurons jointly
        self.H = ridge.coef_                 # (n_neurons, 2)

        # Observation noise covariance Q (diagonal: independent neuron noise)
        residuals_obs = population_matrix - Z @ self.H.T   # (n_time, n_neurons)
        obs_var = np.var(residuals_obs, axis=0)        # (n_neurons,)
        obs_var = np.maximum(obs_var, 1e-6)            # avoid zero variance
        # obs_noise_scale multiplies Q directly, which reduces the Kalman gain
        # K = P H^T (H P H^T + Q)^{-1} without the ridge_alpha / Q coupling.
        # Values > 1 make the filter trust observations less → fewer overshoots.
        self.Q = np.diag(obs_var) * self.obs_noise_scale  # (n_neurons, n_neurons)

        # ------------------------------------------------------------------
        # Optional: steady-state Kalman gain via DARE
        #
        # Solve the discrete algebraic Riccati equation for the steady-state
        # predicted covariance P_inf.  scipy.linalg.solve_discrete_are solves:
        #   A^T X A - X - (A^T X B)(B^T X B + R)^{-1}(B^T X A) + Q = 0
        # Mapping to the Kalman DARE (predictor form):
        #   a = A^T,  b = H^T,  q = W,  r = Q_obs
        # P_inf is (2, 2) regardless of n_neurons — the DARE is in state space.
        # ------------------------------------------------------------------
        if self.use_steady_state_gain:
            P_inf = solve_discrete_are(self.A.T, self.H.T, self.W, self.Q)
            S_inf = self.H @ P_inf @ self.H.T + self.Q   # (n_neurons, n_neurons)
            self.K_inf = np.linalg.solve(S_inf, self.H @ P_inf).T  # (2, n_neurons)
            self.P_inf = P_inf
            self.S_inf = S_inf

        self.is_fitted = True
        src = "fixed" if self.ar1_alpha is not None else "estimated"
        print(f"Kalman filter fitted: {n_time} training bins, {n_neurons} neurons")
        print(f"  AR(1) α_x = {alpha_x:.4f}, α_y = {alpha_y:.4f}  ({src})")
        print(f"  State noise std (x,y): {np.sqrt(self.W[0,0]):.3f}, {np.sqrt(self.W[1,1]):.3f}")
        print(f"  Median obs noise std:  {np.median(np.sqrt(obs_var)):.3f}  "
              f"(×{self.obs_noise_scale} → {np.median(np.sqrt(obs_var)) * np.sqrt(self.obs_noise_scale):.3f} effective)")
        print(f"  H norm (Frobenius):    {np.linalg.norm(self.H):.4f}  "
              f"(ridge_alpha={self.ridge_alpha})")
        if self.use_steady_state_gain:
            print(f"  Steady-state gain:     K_inf norm = {np.linalg.norm(self.K_inf):.4f}  "
                  f"P_inf diag = [{self.P_inf[0,0]:.4f}, {self.P_inf[1,1]:.4f}]")

    # ------------------------------------------------------------------
    def decode(self, population_matrix: np.ndarray,
               initial_state: np.ndarray = None,
               initial_cov: np.ndarray = None,
               clip_to_bounds: bool = False):
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
        clip_to_bounds : bool
            If True, clip the state vector after each update to the normalised
            position bounds seen during training.  Prevents the filter from
            drifting outside the arena, but note this is a hard constraint that
            is inconsistent with the covariance P.  Default False (off).

        Note: mahal_gate_percentile is set at construction time and applied
        automatically here if > 0.

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

        n_time, n_neurons = population_matrix.shape
        A, W, H, Q = self.A, self.W, self.H, self.Q
        n_state = A.shape[0]
        I = np.eye(n_state)

        # Pre-compute chi2 threshold once (depends only on n_neurons and percentile)
        chi2_threshold = (chi2_dist.ppf(self.mahal_gate_percentile, df=n_neurons)
                          if self.mahal_gate_percentile > 0 else None)

        # Initialise state (in normalised position units)
        z = np.zeros(n_state) if initial_state is None else initial_state.copy()

        # Steady-state mode: K and S are fixed; no P recursion needed.
        # Recursive mode: initialise P and update it each step.
        if self.use_steady_state_gain:
            K_fixed = self.K_inf    # (2, n_neurons)
            S_fixed = self.S_inf    # (n_neurons, n_neurons)
        else:
            P = np.eye(n_state) * 10.0 if initial_cov is None else initial_cov.copy()

        pred_x_norm = np.empty(n_time)
        pred_y_norm = np.empty(n_time)

        # EMA bias state: tracks the running mean of the innovation vector.
        # Initialised to zero; warms up over the first ~τ = 1/(1-λ) bins.
        innovation_bias = np.zeros(n_neurons)

        for t in range(n_time):
            # --- Predict ---
            z_pred = A @ z

            # --- Gain and innovation covariance ---
            if self.use_steady_state_gain:
                K = K_fixed
                S = S_fixed
            else:
                P_pred = A @ P @ A.T + W
                HP = H @ P_pred              # (n_neurons, n_state)
                S  = HP @ H.T + Q            # (n_neurons, n_neurons)
                K  = np.linalg.solve(S, HP).T  # (n_state, n_neurons)

            # --- Innovation ---
            innovation = population_matrix[t] - H @ z_pred

            # Online innovation bias correction (EMA).
            # Update the bias estimate with the raw innovation, then subtract
            # it.  Removing the slow-varying mean before the Kalman update
            # prevents systematic distribution drift from accumulating into
            # the state.  The EMA is updated before gating so it tracks the
            # true running mean of innovations, not the clipped version.
            if self.bias_ema_lambda > 0:
                innovation_bias = (self.bias_ema_lambda * innovation_bias
                                   + (1 - self.bias_ema_lambda) * innovation)
                innovation = innovation - innovation_bias

            # Scalar Mahalanobis innovation gating.
            # Under the model, chi2 = v^T S^{-1} v ~ chi2(n_neurons).
            # If chi2 exceeds the threshold, the bin is a population-level
            # outlier (noise burst, tracking dropout).  We soft-scale the
            # innovation so its effective Mahalanobis distance equals the
            # threshold, preserving direction while limiting magnitude.
            if chi2_threshold is not None:
                S_inv_v = np.linalg.solve(S, innovation)
                mahal_sq = float(innovation @ S_inv_v)
                if mahal_sq > chi2_threshold:
                    innovation = innovation * np.sqrt(chi2_threshold / mahal_sq)

            # --- State update ---
            z = z_pred + K @ innovation

            # --- Covariance update (recursive mode only) ---
            # Joseph-stabilised form keeps P positive-definite under
            # floating-point errors: P = (I-KH) P (I-KH)^T + K Q K^T
            if not self.use_steady_state_gain:
                IKH = I - K @ H
                P = IKH @ P_pred @ IKH.T + K @ Q @ K.T

            # Clip state to training-data bounds to prevent arena overshoot.
            if clip_to_bounds:
                z[0] = np.clip(z[0], self.pos_x_min_norm, self.pos_x_max_norm)
                z[1] = np.clip(z[1], self.pos_y_min_norm, self.pos_y_max_norm)

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
# Statistical significance — circular-shift permutation test
# ---------------------------------------------------------------------------

def permutation_test(true_x: np.ndarray, true_y: np.ndarray,
                     pred_x: np.ndarray, pred_y: np.ndarray,
                     n_permutations: int = 1000,
                     seed: int = 42):
    """
    Circular-shift permutation test for above-chance decoding accuracy.

    The true position trace is circularly shifted by a random amount on each
    permutation, preserving its temporal autocorrelation (i.e. the animal
    still moves smoothly, just misaligned with the neural data).  This gives
    a conservative null distribution: if the decoder beats it, the correlation
    is not just due to slow position trends or auto-correlation artefacts.

    Parameters
    ----------
    true_x, true_y : np.ndarray, shape (n_time,)
        Ground-truth position on the test set.
    pred_x, pred_y : np.ndarray, shape (n_time,)
        Decoded position on the test set.
    n_permutations : int
        Number of shuffles.  Default 1000.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict with keys
        corr_x, corr_y           — true Pearson r for X and Y
        null_corr_x, null_corr_y — arrays of null-distribution correlations
        p_x, p_y                 — one-tailed p-values (fraction of null ≥ true r)
        pct_x, pct_y             — percentile of true r in null distribution
        true_euclid              — true mean Euclidean error (px)
        null_euclid              — array of null mean Euclidean errors
        p_euclid                 — one-tailed p-value (fraction of null ≤ true error)
        pct_euclid               — percentile of true error in null (lower = more significant)
    """
    rng = np.random.default_rng(seed)
    n = len(true_x)

    true_corr_x, _ = pearsonr(true_x, pred_x)
    true_corr_y, _ = pearsonr(true_y, pred_y)
    true_euclid = float(np.mean(np.sqrt((true_x - pred_x) ** 2 +
                                        (true_y - pred_y) ** 2)))

    null_corr_x  = np.empty(n_permutations)
    null_corr_y  = np.empty(n_permutations)
    null_euclid  = np.empty(n_permutations)

    for i in range(n_permutations):
        shift = rng.integers(1, n)          # never shift by 0
        sx = np.roll(true_x, shift)
        sy = np.roll(true_y, shift)
        null_corr_x[i], _ = pearsonr(sx, pred_x)
        null_corr_y[i], _ = pearsonr(sy, pred_y)
        null_euclid[i] = float(np.mean(np.sqrt((sx - pred_x) ** 2 +
                                               (sy - pred_y) ** 2)))

    p_x = float(np.mean(null_corr_x >= true_corr_x))
    p_y = float(np.mean(null_corr_y >= true_corr_y))
    # Lower Euclidean error is better, so p = fraction of null ≤ true error
    p_euclid = float(np.mean(null_euclid <= true_euclid))

    # Percentile of true value within null (higher = more significant for r;
    # lower = more significant for error)
    pct_x      = float(np.mean(null_corr_x < true_corr_x) * 100)
    pct_y      = float(np.mean(null_corr_y < true_corr_y) * 100)
    pct_euclid = float(np.mean(null_euclid  < true_euclid) * 100)

    return dict(
        corr_x=true_corr_x, corr_y=true_corr_y,
        null_corr_x=null_corr_x, null_corr_y=null_corr_y,
        p_x=p_x, p_y=p_y,
        pct_x=pct_x, pct_y=pct_y,
        true_euclid=true_euclid, null_euclid=null_euclid,
        p_euclid=p_euclid, pct_euclid=pct_euclid,
    )


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
                 mouse_id, session_id,
                 perm_results: dict = None,
                 save_path: str = None):
    """
    Summary figure for Kalman filter decoding performance.

    Produces a 2×2 grid (4 panels) when perm_results is None, or a 2×3 grid
    (6 panels) when permutation-test results are provided — replacing the error
    histogram with the Euclidean permutation test and adding X/Y null
    distribution histograms in the right column.
    """
    ncols = 3 if perm_results is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 9))
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

    # 4. Euclidean error histogram (fallback) or permutation test (when available)
    ax = axes[1, 1]
    if perm_results is not None:
        null_e = perm_results['null_euclid']
        true_e = perm_results['true_euclid']
        p_e    = perm_results['p_euclid']
        pct_e  = perm_results['pct_euclid']
        ax.hist(null_e, bins=30, color='slategray', edgecolor='white', alpha=0.8,
                label='Null (circular shifts)')
        ax.axvline(true_e, color='tomato', lw=2,
                   label=f'True = {true_e:.1f} px\np = {p_e:.4f}\n({pct_e:.1f}th pct)')
        ax.set_xlabel('Mean Euclidean error (px)')
        ax.set_ylabel('Count')
        ax.set_title('Permutation test — Euclidean error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        errors = np.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
        ax.hist(errors, bins=40, color='slategray', edgecolor='white', alpha=0.85)
        ax.axvline(mean_err, color='red', ls='--', label=f'Mean = {mean_err:.1f} px')
        ax.set_xlabel('Euclidean position error (px)')
        ax.set_ylabel('Count')
        ax.set_title(f'Error distribution  (Y: r = {corr_y:.3f}, R² = {r2_y:.3f})')
        ax.legend(); ax.grid(True, alpha=0.3)

    # 5–6. X and Y correlation permutation test panels (only when provided)
    if perm_results is not None:
        for row, (coord, key_null, key_r, key_p, key_pct) in enumerate([
            ('X', 'null_corr_x', 'corr_x', 'p_x', 'pct_x'),
            ('Y', 'null_corr_y', 'corr_y', 'p_y', 'pct_y'),
        ]):
            ax = axes[row, 2]
            null = perm_results[key_null]
            r    = perm_results[key_r]
            p    = perm_results[key_p]
            pct  = perm_results[key_pct]
            ax.hist(null, bins=30, color='slategray', edgecolor='white', alpha=0.8,
                    label='Null (circular shifts)')
            ax.axvline(r, color='tomato', lw=2,
                       label=f'True r = {r:.3f}\np = {p:.4f}\n({pct:.1f}th pct)')
            ax.set_xlabel(f'Pearson r ({coord})')
            ax.set_ylabel('Count')
            ax.set_title(f'Permutation test — {coord}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_kalman_filter(mouse_id: str, session_id: str, experiment: str,
                      base_path: str = "S:\\",
                      bin_size_ms: float = 30.0,
                      test_fraction: float = 0.2,
                      neural_smooth_sigma: float = 0,
                      pos_smooth_sigma: float = 1.0,
                      x_col: str = 'nose_x',
                      y_col: str = 'nose_y',
                      cluster_filter: str = None,
                      n_permutations: int = 1000,
                      ridge_alpha: float = 10.0,
                      obs_noise_scale: float = 1.0,
                      mahal_gate_percentile: float = 0.0,
                      ar1_alpha: float = None,
                      use_steady_state_gain: bool = False,
                      bias_ema_lambda: float = 0.0,
                      clip_to_bounds: bool = False,
                      save_plots: bool = False):
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

    n_permutations : int
        Number of circular-shift permutations for the significance test.
        Set to 0 to skip.  Default 1000.
    ridge_alpha : float
        Ridge regularisation for the observation matrix H.  Larger values
        shrink H, reducing the Kalman gain and dampening overshoots that
        result from strong (possibly spurious) neural-position correlations.
        Default 1.0.
    mahal_gate_percentile : float
        If > 0, enables scalar Mahalanobis innovation gating.  For each bin,
        the squared Mahalanobis distance chi2 = v^T S^{-1} v is compared to
        the given percentile of the chi2(n_neurons) distribution.  Bins where
        chi2 exceeds the threshold are population-level outliers (noise bursts,
        tracking dropouts); their innovations are soft-scaled down so the
        effective Mahalanobis distance equals the threshold.  This is more
        principled than per-neuron clipping because it respects the full
        observation covariance.  Good starting value: 0.99.  Default 0.0 (off).
    ar1_alpha : float or None
        AR(1) coefficient for the diagonal state transition A = diag(α_x, α_y).
        If None (default), α is estimated per-axis from the lag-1
        autocorrelation of training positions.  Set to a fixed value in (0, 1)
        to override.
    use_steady_state_gain : bool
        If True, solve the DARE during fit() to obtain the steady-state Kalman
        gain K_inf, and use it as a fixed gain in decode() (no recursive P
        update).  Eliminates the high-gain startup transient.  Default False.
    clip_to_bounds : bool
        If True, clip the decoded state to the arena bounds seen during
        training after each Kalman update.  Default False.
    save_plots : bool
        If True, save the summary figure as a PNG to
        ``{base_path}/{experiment}/kalman_decode/{mouse_id}/{session_id}/
        kalman_decode.png`` and suppress the interactive window.
        Default False (show interactively).

    Returns
    -------
    decoder : KalmanFilterDecoder
    metrics : dict
        r2_x, r2_y, corr_x, corr_y, mean_position_error, p_x, p_y
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
                                  pos_smooth_sigma=pos_smooth_sigma,
                                  ridge_alpha=ridge_alpha,
                                  obs_noise_scale=obs_noise_scale,
                                  mahal_gate_percentile=mahal_gate_percentile,
                                  ar1_alpha=ar1_alpha,
                                  use_steady_state_gain=use_steady_state_gain,
                                  bias_ema_lambda=bias_ema_lambda)
    decoder.fit(X_train, y_train_x, y_train_y)

    # --- Test-set mean subtraction ---
    # Firing rates can drift over the session (adaptation, behavioural state
    # changes), so the test-set per-neuron mean may differ from training.
    # Because H was fit to the training distribution, any DC offset in the
    # test set produces a non-zero mean innovation at every step, which the
    # Kalman gain integrates into a systematic position drift.  Removing the
    # test-set mean before decoding eliminates this bias without affecting the
    # shape of the decoded trajectory.
    test_mean = X_test.mean(axis=0)          # (n_neurons,)
    train_mean = X_train.mean(axis=0)
    print(f"\n  Test-set neural mean shift from training: "
          f"max={np.max(np.abs(test_mean - train_mean)):.3f}  "
          f"median={np.median(np.abs(test_mean - train_mean)):.3f}  (z-score units)")
    X_test_dec = X_test - test_mean

    # --- Linear baseline (diagnostic) ---
    lin_x, lin_y = decoder.decode_linear(X_test_dec)
    lin_corr_x, _ = pearsonr(y_test_x, lin_x)
    lin_corr_y, _ = pearsonr(y_test_y, lin_y)
    print(f"\nLinear decode baseline (no Kalman smoothing):")
    print(f"  X  r = {lin_corr_x:.4f}   Y  r = {lin_corr_y:.4f}")
    print(f"  (If Kalman ≈ linear, the dynamics model is not helping.)")
    print(f"  (If both are weak, neural–position tuning may be limited.)")

    # --- Kalman decode ---
    print(f"\n{step+3}. Kalman decoding test set")
    pred_x, pred_y = decoder.decode(X_test_dec, clip_to_bounds=clip_to_bounds)

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

    # --- Permutation test ---
    perm_results = None
    if n_permutations > 0:
        print(f"\n{step+4}. Permutation test ({n_permutations} circular shifts)")
        perm_results = permutation_test(
            y_test_x, y_test_y, pred_x, pred_y,
            n_permutations=n_permutations
        )
        print(f"  X  r = {perm_results['corr_x']:.4f}   p = {perm_results['p_x']:.4f}"
              f"   ({perm_results['pct_x']:.1f}th percentile of null)")
        print(f"  Y  r = {perm_results['corr_y']:.4f}   p = {perm_results['p_y']:.4f}"
              f"   ({perm_results['pct_y']:.1f}th percentile of null)")
        print(f"  Euclidean error = {perm_results['true_euclid']:.1f} px   "
              f"p = {perm_results['p_euclid']:.4f}"
              f"   ({perm_results['pct_euclid']:.1f}th percentile of null)")

    metrics = {
        'corr_x': corr_x, 'corr_y': corr_y,
        'r2_x':   r2_x,   'r2_y':   r2_y,
        'mean_position_error': mean_err,
        'linear_corr_x': lin_corr_x,
        'linear_corr_y': lin_corr_y,
        'p_x':      perm_results['p_x']      if perm_results else None,
        'p_y':      perm_results['p_y']      if perm_results else None,
        'p_euclid': perm_results['p_euclid'] if perm_results else None,
    }

    # --- Save metrics CSV ---
    if save_plots:
        import os, csv
        metrics_dir = os.path.join(
            base_path, experiment, "kalman_decode", mouse_id, session_id
        )
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "metrics.csv")
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)

    # --- Plot ---
    plot_step = step + 5 if n_permutations > 0 else step + 4
    save_path = None
    if save_plots:
        import os
        save_path = os.path.join(
            base_path, experiment, "kalman_decode",
            mouse_id, session_id, "kalman_decode.png"
        )
    print(f"\n{plot_step}. Plotting results"
          + (f"  → {save_path}" if save_path else ""))
    plot_results(
        y_test_x, y_test_y, pred_x, pred_y,
        time_test / 1000.0,          # ms → s
        corr_x, corr_y, r2_x, r2_y, mean_err,
        mouse_id, session_id,
        perm_results=perm_results,
        save_path=save_path,
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
        session_id="m5",
        experiment="clickbait-motivate",
        base_path="d:/data/",
        bin_size_ms=30.0,         # 30 ms ≈ video frame rate
        test_fraction=0.2,
        neural_smooth_sigma=0.0,  # Smooth neural activity over ~2 bins (60 ms)
        pos_smooth_sigma=0.0,     # Smooth position labels to reduce SLEAP jitter
        x_col='bonsai_centroid_x',
        y_col='bonsai_centroid_y',
        n_permutations=1000,      # Circular-shift permutation test
        ridge_alpha=1.0,            # Ridge regularisation for H; increase to dampen overshoots
        obs_noise_scale=1.0,        # Multiplier for Q; >1 reduces Kalman gain
        mahal_gate_percentile=0.99, # Mahalanobis gate: soft-scale outlier bins to this chi2 percentile
        ar1_alpha=None,             # AR(1) α; None = estimate from lag-1 autocorrelation
        use_steady_state_gain=True, # DARE steady-state gain: eliminates startup transient
        bias_ema_lambda=0.995,       # Innovation EMA bias correction; τ ≈ 1/(1-λ) bins
        clip_to_bounds=True,       # Clip decoded state to training arena bounds
        cluster_filter='best_channel <= 32'    # Only use neurons from given channels
    )

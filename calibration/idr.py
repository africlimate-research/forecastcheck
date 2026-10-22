# # Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
# Isotonic Distributional Regression (IDR) for rainfall forecast calibration.
#
# Pure NumPy reimplementation of evwalz/isodisreg (MIT licence).
# No C++ extensions, no pandas, no osqp, no scipy.optimize.
#
# Optimised version:
#   - _pava_antitonic: Numba JIT with pure-NumPy fallback
#   - IDRDistribution: fully vectorised quantile, pit, crps (no Python loops)
#   - IDRModel.predict: vectorised case-mask approach (no per-sample loop)
#   - _apply_zone_model: single batched predict+crps call across all timesteps
#   - idr_on_grid / idr_loyo_cv: three execution backends:
#       1. Dask apply_ufunc  (auto-detected when inputs are Dask-backed)
#       2. joblib parallel   (use_parallel=True on in-memory arrays)
#       3. Sequential        (default, no dependencies beyond numpy/xarray)
#   - run_zonal_loyo: optionally parallel across zones (use_parallel=True)
#
# Dependencies: numpy, xarray
# Optional: numba  (significant speedup on PAVA inner loop)
#            joblib (parallel grid-point / zone processing, use_parallel=True)
#
# References:
#   Henzi, Ziegel, Gneiting (2021) "Isotonic Distributional Regression"
#   J. R. Statist. Soc. B  https://doi.org/10.1111/rssb.12450
#
from __future__ import annotations

import functools
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

import numpy as np
import xarray as xr

NUM_DATA_PTS = 50_000
ZONE_SAMPLE_LOCS = 1_000
ZONE_MIN_VALID   = 100

# Number of parallel workers — default to all cores, override with env var
_N_JOBS = int(os.environ.get("IDR_N_JOBS", -1))

#: Base seed for the randomized PIT. Fixed (not None) so a re-run reproduces `pit`,
#: which used to be drawn from OS entropy and was therefore the one output that could
#: not be validated. Every other output was already deterministic. Pass
#: ``pit_seed=None`` to restore the old irreproducible behaviour.
_DEFAULT_PIT_SEED = 0


# ---------------------------------------------------------------------------
# Optional parallelism — seamless fallback to sequential
# ---------------------------------------------------------------------------

def _progress_bar(total, enabled):
    """tqdm bar over completed work units, or None when disabled/unavailable."""
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc="IDR grid cells", unit="cell")


def _consume(results_iter, bar):
    """Drain an iterator into a list, advancing the bar per item."""
    out = []
    for r in results_iter:
        out.append(r)
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return out


def _parallel_map(func, iterable, use_parallel, n_jobs, progress=False):
    """
    Apply *func* to every element of *iterable*, optionally in parallel,
    optionally showing a tqdm progress bar over completed elements.

    Parameters
    ----------
    func         : callable — signature func(*args)
    iterable     : sequence of tuples, each unpacked as positional args
    use_parallel : bool — if False, always run sequentially
    n_jobs       : int — number of workers (-1 = all cores); ignored when sequential
    progress     : bool — show a tqdm bar (no-op if tqdm is unavailable)
    """
    items = list(iterable)
    bar = _progress_bar(len(items), enabled=progress)

    if use_parallel:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            log.warning(
                "use_parallel=True but joblib is not installed — "
                "falling back to sequential.  Install: pip install joblib"
            )
            return _consume((func(*args) for args in items), bar)
        # return_as='generator' yields in submission order as cells finish, so
        # the bar advances during the run while result order is preserved.
        results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
            delayed(func)(*args) for args in items
        )
        return _consume(results, bar)

    return _consume((func(*args) for args in items), bar)


def _is_dask_backed(*arrays: xr.DataArray) -> bool:
    """Return True if any input array is backed by a Dask array."""
    import dask.array as da
    return any(isinstance(a.data, da.Array) for a in arrays)


# ---------------------------------------------------------------------------
# 1.  Weighted antitonic PAVA — Numba with pure-NumPy fallback
# ---------------------------------------------------------------------------

def _pava_antitonic_numpy(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pure-NumPy fallback for PAVA (used when Numba is unavailable)."""
    m  = len(y)
    x  = y.astype(float).copy()
    bw = w.astype(float).copy()

    stack_start = np.empty(m, dtype=int)
    top = -1

    for i in range(m):
        top += 1
        stack_start[top] = i

        while top > 0 and x[stack_start[top - 1]] < x[stack_start[top]]:
            s_prev     = stack_start[top - 1]
            s_cur      = stack_start[top]
            w_prev     = bw[s_prev]
            w_cur      = bw[s_cur]
            x[s_prev]  = (w_prev * x[s_prev] + w_cur * x[s_cur]) / (w_prev + w_cur)
            bw[s_prev] = w_prev + w_cur
            top -= 1

    result = np.empty(m)
    right  = m
    for k in range(top, -1, -1):
        left = stack_start[k]
        result[left:right] = x[stack_start[k]]
        right = left

    return result


try:
    import numba

    @numba.njit(cache=True)
    def _pava_antitonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Numba-JIT weighted antitonic PAVA — ~50-100x faster than pure Python."""
        m  = len(y)
        x  = y.astype(np.float64).copy()
        bw = w.astype(np.float64).copy()

        stack_start = np.empty(m, dtype=np.int64)
        top = -1

        for i in range(m):
            top += 1
            stack_start[top] = i

            while top > 0 and x[stack_start[top - 1]] < x[stack_start[top]]:
                s_prev     = stack_start[top - 1]
                s_cur      = stack_start[top]
                w_prev     = bw[s_prev]
                w_cur      = bw[s_cur]
                x[s_prev]  = (w_prev * x[s_prev] + w_cur * x[s_cur]) / (w_prev + w_cur)
                bw[s_prev] = w_prev + w_cur
                top -= 1

        result = np.empty(m, dtype=np.float64)
        right  = m
        for k in range(top, -1, -1):
            left = stack_start[k]
            for j in range(left, right):
                result[j] = x[stack_start[k]]
            right = left

        return result

    # Warm up JIT on first import
    _pava_antitonic(np.array([1.0, 0.5]), np.array([1.0, 1.0]))
    log.info("Numba JIT available — using accelerated PAVA")

except ImportError:
    _pava_antitonic = _pava_antitonic_numpy
    log.info("Numba not available — using pure-NumPy PAVA (install numba for ~50x speedup)")


# ---------------------------------------------------------------------------
# 2.  Core: distributional PAVA
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Legacy (original) _isocdf_seq saved for comparison / regression testing
# ---------------------------------------------------------------------------
def _isocdf_seq_legacy(
    group_counts: np.ndarray,
    Y_sorted: np.ndarray,
    posY: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Original implementation preserved as _isocdf_seq_legacy.
    """
    m = len(group_counts)
    T = len(thresholds)

    group_ends = np.cumsum(group_counts)
    group_starts = np.concatenate([[0], group_ends[:-1]])

    group_cdfs = np.zeros((m, T))
    for i in range(m):
        s, e = int(group_starts[i]), int(group_ends[i])
        if s == e:
            continue
        yi = Y_sorted[s:e]
        counts        = np.searchsorted(yi, thresholds, side="right")
        group_cdfs[i] = counts / (e - s)

    w   = group_counts.astype(float)
    cdf = np.empty((m, T))
    for t in range(T):
        cdf[:, t] = _pava_antitonic(group_cdfs[:, t], w)

    np.maximum.accumulate(cdf, axis=1, out=cdf)
    np.clip(cdf, 0.0, 1.0, out=cdf)
    return cdf


# Insert near the other functions in src/weather_post_processing/idr_xarray.py

# --- Numba-accelerated incremental PAVA implementation (exact) ---
# Numba incremental PAVA implementation (corrected)
# --- numba-accelerated exact (legacy) _isocdf_seq implementation ----------------
# Requires numba; safe fallback to pure-Python legacy exists below.

# Chunked counts computation (Numba-accelerated with NumPy fallback)

# --- Chunked counts + hash-cache _isocdf_seq (Option 1 implementation) ---
# Place this near other helpers in src/weather_post_processing/idr_xarray.py

# Try to detect numba for the fast binary-search counts block
# Parallel-per-column PAVA (Numba-parallel with joblib fallback)

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except Exception:
    _numba = None
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    import numpy as _np

    @_numba.njit(parallel=True, cache=True)
    def _pava_columns_numba(group_cdfs, w):
        """
        Compute PAVA on each column (threshold) in parallel.
        group_cdfs: (m, T) float64
        w: (m,) float64 weights
        Returns cdf: (m, T) float64
        """
        m, T = group_cdfs.shape
        cdf = _np.empty((m, T), dtype=_np.float64)

        # per-column PAVA implemented inline to avoid function call overhead
        for t in _numba.prange(T):
            # local copies
            x = group_cdfs[:, t].astype(_np.float64).copy()
            bw = w.copy()

            stack_start = _np.empty(m, dtype=_np.int64)
            top = -1

            for i in range(m):
                top += 1
                stack_start[top] = i
                # merge while previous block value < current block value (antitonic)
                while top > 0 and x[stack_start[top - 1]] < x[stack_start[top]]:
                    s_prev = stack_start[top - 1]
                    s_cur = stack_start[top]
                    w_prev = bw[s_prev]
                    w_cur = bw[s_cur]
                    x[s_prev] = (w_prev * x[s_prev] + w_cur * x[s_cur]) / (w_prev + w_cur)
                    bw[s_prev] = w_prev + w_cur
                    top -= 1

            # write result for column t
            right = m
            for k in range(top, -1, -1):
                left = stack_start[k]
                val = x[stack_start[k]]
                for j in range(left, right):
                    cdf[j, t] = val
                right = left

        return cdf

else:
    _pava_columns_numba = None  # fallback will be joblib-based



def _isocdf_seq(
    group_counts: np.ndarray,
    Y_sorted: np.ndarray,
    posY: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Non-chunked implementation that uses parallel PAVA across thresholds.

    - Computes group-level CDFs exactly as legacy: group_cdfs (m, T).
    - Runs PAVA per column in parallel: Numba-parallel if available, otherwise joblib threads.
    - Preserves legacy numeric results (same counts + same pooled PAVA).
    """
    m = len(group_counts)
    T = len(thresholds)

    group_ends = np.cumsum(group_counts)
    group_starts = np.concatenate([[0], group_ends[:-1]])

    # 1) Build group_cdfs exactly like legacy
    group_cdfs = np.zeros((m, T), dtype=float)
    for i in range(m):
        s, e = int(group_starts[i]), int(group_ends[i])
        if s == e:
            continue
        yi = Y_sorted[s:e]
        counts = np.searchsorted(yi, thresholds, side="right")
        group_cdfs[i] = counts / (e - s)

    w = group_counts.astype(float)

    # 2) Run parallel PAVA across columns
    if _pava_columns_numba is not None:
        # ensure types
        gcdf = np.asarray(group_cdfs, dtype=float, order="C")
        w_arr = w.astype(float)
        cdf = _pava_columns_numba(gcdf, w_arr)
    else:
       # pure Python fallback (legacy behaviour)
        cdf = np.empty((m, T), dtype=float)
        for t in range(T):
            cdf[:, t] = _pava_antitonic(group_cdfs[:, t], w)

    # 3) ensure monotonicity across thresholds and clip (legacy postprocessing)
    np.maximum.accumulate(cdf, axis=1, out=cdf)
    np.clip(cdf, 0.0, 1.0, out=cdf)
    return cdf
# ---------------------------------------------------------------------------

@dataclass
class IDRDistribution:
    """
    Calibrated predictive distributions for a batch of forecast values.

    Attributes
    ----------
    thresholds : (T,) array
    cdf        : (n, T) array
    lower, upper : (n, T) arrays or None
    """
    thresholds: np.ndarray
    cdf:        np.ndarray
    lower:      Optional[np.ndarray] = field(default=None)
    upper:      Optional[np.ndarray] = field(default=None)

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        """Quantile function Q(q) = inf{ t : F(t) >= q }.  Fully vectorised."""
        q = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((q < 0) | (q > 1)):
            raise ValueError("q must be in [0, 1]")
        n, T    = self.cdf.shape
        cdf_ext = np.hstack([self.cdf, np.ones((n, 1))])
        thr_ext = np.append(self.thresholds, self.thresholds[-1])

        exceeds = cdf_ext[:, :, None] >= q[None, None, :]
        idx     = np.argmax(exceeds, axis=1)
        out     = thr_ext[idx]
        return out.squeeze()

    def cdf_at(self, thresholds: float | np.ndarray) -> np.ndarray:
        """Evaluate step-function CDF at arbitrary threshold(s).  Vectorised."""
        thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
        n          = self.cdf.shape[0]
        thr_aug    = np.concatenate([[-np.inf], self.thresholds])
        cdf_aug    = np.hstack([np.zeros((n, 1)), self.cdf])

        idx = np.searchsorted(thr_aug, thresholds, side="right") - 1
        out = cdf_aug[:, idx]
        return out.squeeze()

    def crps(self, obs: np.ndarray, chunk_size: int = 100_000) -> np.ndarray:
        """CRPS — vectorised NumPy, chunked for large batches."""
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (self.cdf.shape[0],):
            raise ValueError(f"obs shape {obs.shape} != ({self.cdf.shape[0]},)")
        n = self.cdf.shape[0]
        if n <= chunk_size:
            return self._crps_batch(self.cdf, obs)
        result = np.empty(n)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            result[s:e] = self._crps_batch(self.cdf[s:e], obs[s:e])
        return result

    def _crps_batch(self, cdf: np.ndarray, obs: np.ndarray) -> np.ndarray:
        n       = cdf.shape[0]
        cdf_aug = np.hstack([np.zeros((n, 1)), cdf])
        delta_p = np.diff(cdf_aug, axis=1)
        T       = self.thresholds
        ind     = (obs[:, None] < T[None, :]).astype(float)
        return 2.0 * np.sum(
            delta_p * (ind - cdf + 0.5 * delta_p) * (T[None, :] - obs[:, None]),
            axis=1,
        )

    def brier_score(self, thresholds: float | np.ndarray, obs: np.ndarray) -> np.ndarray:
        obs        = np.asarray(obs, dtype=float)
        thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
        predicted  = np.atleast_2d(self.cdf_at(thresholds))
        observed   = (obs[:, None] <= thresholds[None, :]).astype(float)
        return ((predicted - observed) ** 2).squeeze()

    def pit(
        self,
        obs: np.ndarray,
        randomize: bool = True,
        seed: int | None = None,
    ) -> np.ndarray:
        """PIT values — vectorised, no Python for-loops."""
        obs   = np.asarray(obs, dtype=float)
        n     = len(obs)
        diffs = np.diff(self.thresholds)
        eps   = float(np.min(diffs)) * 0.5 if len(diffs) > 0 else 1e-10

        thr_aug = np.concatenate([[-np.inf], self.thresholds])
        cdf_aug = np.hstack([np.zeros((self.cdf.shape[0], 1)), self.cdf])

        idx_u = np.searchsorted(thr_aug, obs,       side="right") - 1
        idx_l = np.searchsorted(thr_aug, obs - eps,  side="right") - 1

        row_idx   = np.arange(n)
        pit_upper = cdf_aug[row_idx, idx_u]
        pit_lower = cdf_aug[row_idx, idx_l]

        if randomize:
            rng = np.random.default_rng(seed)
            u   = rng.uniform(size=n)
            return pit_lower + u * (pit_upper - pit_lower)
        return pit_upper


# ---------------------------------------------------------------------------
# 4.  IDR model — vectorised predict
# ---------------------------------------------------------------------------

class IDRModel:
    """Isotonic Distributional Regression (1-D covariate)."""

    def __init__(self, increasing: bool = True):
        self.increasing = increasing
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IDRModel":
        X = np.asarray(X, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X) < 2:
            raise ValueError("Need at least 2 training samples")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaN in training data — mask before calling fit()")

        unique_X, inv_idx, group_counts = np.unique(
            X, return_inverse=True, return_counts=True
        )

        if not self.increasing:
            inv_idx      = (len(unique_X) - 1) - inv_idx
            unique_X     = unique_X[::-1].copy()
            group_counts = group_counts[::-1].copy()

        sort_order = np.argsort(inv_idx, kind="stable")
        Y_sorted   = y[sort_order]
        posY       = inv_idx[sort_order]
        for i in range(len(unique_X)):
            mask             = posY == i
            Y_sorted[mask]   = np.sort(Y_sorted[mask])

        thresholds = np.sort(np.unique(y))
        if len(thresholds) < 2:
            raise ValueError("y must contain at least 2 distinct values")

        self._unique_X     = unique_X
        self._group_counts = group_counts
        self._thresholds   = thresholds
        self._cdf          = _isocdf_seq(group_counts, Y_sorted, posY, thresholds)
        self._y_train      = y
        self._is_fitted    = True
        return self

    def predict(self, X_new: np.ndarray, digits: int = 3) -> IDRDistribution:
        """Vectorised predict — case-mask approach, no per-sample loop."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        X_new = np.asarray(X_new, dtype=float).ravel()
        n     = len(X_new)
        T     = len(self._thresholds)
        Xtr   = self._unique_X
        m     = len(Xtr)

        cdf_out   = np.empty((n, T))
        lower_out = np.empty((n, T))
        upper_out = np.empty((n, T))

        idx_r = np.searchsorted(Xtr, X_new, side="left")

        at_left  = idx_r == 0
        at_right = idx_r == m
        safe_ir  = np.clip(idx_r, 0, m - 1)
        exact    = (~at_left) & (~at_right) & (Xtr[safe_ir] == X_new)
        interp   = ~(at_left | at_right | exact)

        if at_left.any():
            cdf_out[at_left]   = self._cdf[0]
            lower_out[at_left] = 0.0
            upper_out[at_left] = self._cdf[0]

        if at_right.any():
            cdf_out[at_right]   = self._cdf[-1]
            lower_out[at_right] = self._cdf[-1]
            upper_out[at_right] = 1.0

        if exact.any():
            ir_exact = idx_r[exact]
            cdf_out[exact]   = self._cdf[ir_exact]
            lower_out[exact] = self._cdf[ir_exact]
            upper_out[exact] = self._cdf[ir_exact]

        if interp.any():
            ir = idx_r[interp]
            il = ir - 1
            wg = (X_new[interp] - Xtr[il]) / (Xtr[ir] - Xtr[il])
            ws = 1.0 - wg
            cl = self._cdf[il]
            cr = self._cdf[ir]
            blended = np.round(cr * wg[:, None] + cl * ws[:, None], digits)
            np.maximum.accumulate(blended, axis=1, out=blended)
            np.clip(blended, 0.0, 1.0, out=blended)
            cdf_out[interp]   = blended
            lower_out[interp] = np.clip(cr, 0.0, 1.0)
            upper_out[interp] = np.clip(cl, 0.0, 1.0)

        return IDRDistribution(
            thresholds=self._thresholds,
            cdf=cdf_out,
            lower=lower_out,
            upper=upper_out,
        )

    def climatological_distribution(self) -> IDRDistribution:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        T        = self._thresholds
        clim_cdf = np.searchsorted(np.sort(self._y_train), T, side="right") / len(self._y_train)
        return IDRDistribution(thresholds=T, cdf=clim_cdf[None, :])


# ---------------------------------------------------------------------------
# 5.  Grid kernel — one grid point (shared by all backends)
# ---------------------------------------------------------------------------

# Per-cell scalar outputs (one value per test day), in fixed order. The kernel
# returns these as rows 0..N-1; any exceedance-probability thresholds add rows
# N..N+K-1. Names/attrs live here so both backends build identical Datasets.
# All rain-amount outputs are in mm/hr (the engine's internal standard); the
# pipeline divides forecast and obs by 24, so quantiles and CRPS are mm/hr.
_SCALAR_OUTPUTS = ("crps", "q5", "q50", "q95", "q99", "pit")
_N_SCALAR = len(_SCALAR_OUTPUTS)
_QUANTILE_LEVELS = np.array([0.05, 0.50, 0.95, 0.99])  # -> q5, q50, q95, q99
_SCALAR_ATTRS = {
    "crps": {"units": "mm/hr", "long_name": "IDR CRPS"},
    "q5":   {"units": "mm/hr", "long_name": "IDR 5th percentile"},
    "q50":  {"units": "mm/hr", "long_name": "IDR median"},
    "q95":  {"units": "mm/hr", "long_name": "IDR 95th percentile"},
    "q99":  {"units": "mm/hr", "long_name": "IDR 99th percentile"},
    "pit":  {"long_name": "IDR Probability Integral Transform"},  # dimensionless
}
_P_EXCEED_ATTRS = {
    "long_name": "probability daily precipitation exceeds threshold",
    "units": "1",
}


def _idr_grid_kernel(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    min_valid: int,
    min_wet_days: int,
    exceedance_thresholds: np.ndarray = (),
    seed: int | None = None,
) -> np.ndarray:
    """Fit IDR and evaluate for one grid point.

    ``seed`` makes the randomized PIT reproducible. It must be derived from the CELL
    INDEX by the caller, never drawn from one shared stream: joblib runs cells in
    separate processes, so a shared generator would hand out values that depend on
    worker count and scheduling. ``None`` keeps the historical seedless behaviour.

    Returns (N_SCALAR + K, n_test): rows 0..N_SCALAR-1 are crps/q10/q50/q90/pit;
    rows N_SCALAR.. are exceedance probabilities P(X > t) for each of the K
    `exceedance_thresholds` (in the same units as the data — mm/hr internally).
    """
    thr     = np.asarray(exceedance_thresholds, dtype=float)
    K       = thr.size
    n_test  = len(X_te)
    nan_out = np.full((_N_SCALAR + K, n_test), np.nan)

    valid_tr = np.isfinite(X_tr) & np.isfinite(y_tr)
    if valid_tr.sum() < min_valid:
        return nan_out
    if len(np.unique(y_tr[valid_tr])) < 2:
        return nan_out
    if np.sum(y_tr[valid_tr] > 0.1) < min_wet_days:
        return nan_out

    model    = IDRModel(increasing=True).fit(X_tr[valid_tr], y_tr[valid_tr])
    valid_te = np.isfinite(X_te) & np.isfinite(y_te)
    out      = nan_out.copy()
    if not valid_te.any():
        return out

    n_valid          = int(valid_te.sum())
    nq               = len(_QUANTILE_LEVELS)
    preds            = model.predict(X_te[valid_te])
    q                = np.asarray(preds.quantile(_QUANTILE_LEVELS)).reshape(n_valid, nq)
    out[0, valid_te] = preds.crps(y_te[valid_te])
    out[1:1 + nq, valid_te] = q.T                      # q5, q50, q95, q99
    out[_N_SCALAR - 1, valid_te] = preds.pit(y_te[valid_te], seed=seed)
    if K:
        # Exceedance probability P(X > t) = 1 - F(t); cdf_at -> (n_valid, K)
        # (reshape undoes its squeeze).
        cdf = np.asarray(preds.cdf_at(thr)).reshape(n_valid, K)
        out[_N_SCALAR:_N_SCALAR + K, valid_te] = (1.0 - cdf).T
    return out


# ---------------------------------------------------------------------------
# 6a.  idr_on_grid — Dask backend (apply_ufunc)
# ---------------------------------------------------------------------------

def _idr_on_grid_dask(
    forecast_train: xr.DataArray,
    obs_train: xr.DataArray,
    forecast_test: xr.DataArray,
    obs_test: xr.DataArray,
    spatial_dims: tuple[str, str],
    time_dim: str,
    train_time_dim: str,
    min_valid: int,
    min_wet_days: int,
    exceedance_thresholds: np.ndarray = (),
) -> xr.Dataset:
    """Dask-backed path: xr.apply_ufunc with dask='parallelized'."""
    n_test = forecast_test.sizes[time_dim]
    thr    = np.asarray(exceedance_thresholds, dtype=float)
    n_out  = _N_SCALAR + thr.size
    kernel = functools.partial(
        _idr_grid_kernel, min_valid=min_valid, min_wet_days=min_wet_days,
        exceedance_thresholds=thr,
    )

    result = xr.apply_ufunc(
        kernel,
        forecast_train, obs_train, forecast_test, obs_test,
        input_core_dims=[
            [train_time_dim], [train_time_dim], [time_dim], [time_dim],
        ],
        # train and test time axes are independent series of different lengths;
        # exempt them from apply_ufunc's exact-join alignment.
        exclude_dims={train_time_dim, time_dim},
        output_core_dims=[["output", time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {"output": n_out, time_dim: n_test},
            "allow_rechunk": True,
        },
    )

    ds = {
        name: result.isel(output=k).drop_vars("output", errors="ignore").assign_attrs(_SCALAR_ATTRS[name])
        for k, name in enumerate(_SCALAR_OUTPUTS)
    }
    if thr.size:
        ds["p_exceed"] = (
            result.isel(output=slice(_N_SCALAR, n_out))
                  .rename({"output": "threshold"})
                  .assign_coords(threshold=thr)
                  .assign_attrs(_P_EXCEED_ATTRS)
        )
    return xr.Dataset(ds)


# ---------------------------------------------------------------------------
# 6b.  idr_on_grid — in-memory backend (sequential or joblib)
# ---------------------------------------------------------------------------

def _cell_seed(base_seed, i, j):
    """A per-cell PIT seed, or None to stay seedless.

    Derived from (base, i, j) through SeedSequence so each cell gets an independent,
    well-separated stream that depends only on WHERE the cell is — not on how many
    workers ran, or in what order they finished. That is what makes ``pit`` identical
    between a sequential run and an ``n_jobs=-1`` run.
    """
    if base_seed is None:
        return None
    return np.random.SeedSequence(entropy=int(base_seed), spawn_key=(int(i), int(j)))


def _run_grid_point(i, j, Xtr_np, ytr_np, Xte_np, yte_np,
                    lat_ax, lon_ax, min_valid, min_wet_days,
                    exceedance_thresholds, pit_seed=None):
    """Worker for one grid point."""
    tr_sl = [slice(None)] * Xtr_np.ndim
    tr_sl[lat_ax] = i
    tr_sl[lon_ax] = j
    te_sl = [slice(None)] * Xte_np.ndim
    te_sl[lat_ax] = i
    te_sl[lon_ax] = j
    return _idr_grid_kernel(
        Xtr_np[tuple(tr_sl)], ytr_np[tuple(tr_sl)],
        Xte_np[tuple(te_sl)], yte_np[tuple(te_sl)],
        min_valid, min_wet_days, exceedance_thresholds,
        seed=_cell_seed(pit_seed, i, j),
    )


def _idr_on_grid_inmemory(
    forecast_train: xr.DataArray,
    obs_train: xr.DataArray,
    forecast_test: xr.DataArray,
    obs_test: xr.DataArray,
    spatial_dims: tuple[str, str],
    time_dim: str,
    min_valid: int,
    min_wet_days: int,
    use_parallel: bool,
    n_jobs: int,
    exceedance_thresholds: np.ndarray = (),
    pit_seed: int | None = None,
) -> xr.Dataset:
    """In-memory path: materialise to NumPy then loop (sequential or joblib)."""
    Xtr_np = forecast_train.values
    ytr_np = obs_train.values
    Xte_np = forecast_test.values
    yte_np = obs_test.values

    thr   = np.asarray(exceedance_thresholds, dtype=float)
    n_out = _N_SCALAR + thr.size

    lat_dim, lon_dim = spatial_dims
    nlat  = forecast_test.sizes[lat_dim]
    nlon  = forecast_test.sizes[lon_dim]
    ntime = forecast_test.sizes[time_dim]

    te_dims = list(forecast_test.dims)
    lat_ax  = te_dims.index(lat_dim)
    lon_ax  = te_dims.index(lon_dim)

    results = _parallel_map(
        func=_run_grid_point,
        iterable=(
            (i, j, Xtr_np, ytr_np, Xte_np, yte_np,
             lat_ax, lon_ax, min_valid, min_wet_days, thr, pit_seed)
            for i in range(nlat) for j in range(nlon)
        ),
        use_parallel=use_parallel,
        n_jobs=n_jobs,
        progress=True,
    )

    results_arr = np.stack(results)
    out = results_arr.reshape((nlat, nlon, n_out, ntime)).transpose(2, 3, 0, 1)

    coords = {
        time_dim:        forecast_test[time_dim],
        spatial_dims[0]: forecast_test[spatial_dims[0]],
        spatial_dims[1]: forecast_test[spatial_dims[1]],
    }
    ds = {
        name: xr.DataArray(
            out[k], dims=[time_dim, spatial_dims[0], spatial_dims[1]],
            coords=coords, attrs=_SCALAR_ATTRS[name],
        )
        for k, name in enumerate(_SCALAR_OUTPUTS)
    }
    if thr.size:
        ds["p_exceed"] = xr.DataArray(
            out[_N_SCALAR:n_out],
            dims=["threshold", time_dim, spatial_dims[0], spatial_dims[1]],
            coords={**coords, "threshold": thr},
            attrs=_P_EXCEED_ATTRS,
        )
    return xr.Dataset(ds)


# ---------------------------------------------------------------------------
# 6.  idr_on_grid — public API (auto-dispatches)
# ---------------------------------------------------------------------------

def idr_on_grid(
    forecast_train: xr.DataArray,
    obs_train: xr.DataArray,
    forecast_test: xr.DataArray,
    obs_test: xr.DataArray,
    spatial_dims: tuple[str, str] = ("lat", "lon"),
    time_dim: str = "time",
    train_time_dim: str | None = None,
    min_valid: int = 10,
    min_wet_days: int = 50,
    use_parallel: bool = False,
    n_jobs: int | None = None,
    force_dask: bool = False,
    exceedance_thresholds: tuple = (),
    pit_seed: int | None = _DEFAULT_PIT_SEED,
) -> xr.Dataset:
    """
    Apply IDR calibration at every grid point.

    Returns {crps, q5, q50, q95, q99, pit}(time, lat, lon) — all rain-amount
    outputs in mm/hr — plus, when `exceedance_thresholds` is non-empty,
    p_exceed(threshold, time, lat, lon), the calibrated exceedance probability
    P(X > t) at each threshold (in the data's units). The numbers do not depend
    on the execution backend or worker count.

    ``pit`` used to be the one exception — drawn with a seedless randomized PIT, so
    it could not be reproduced or validated. ``pit_seed`` now fixes it by default:
    each cell derives its own stream from ``(pit_seed, i, j)``, so the values depend
    only on WHERE a cell is, and a sequential run and an ``n_jobs=-1`` run agree
    exactly. Pass ``pit_seed=None`` for the historical behaviour.

    Caveat: the streaming-dask backend below has no cell index to derive from, so it
    stays seedless regardless of ``pit_seed``. Only ``pit`` is affected there; every
    other output is deterministic on every backend.

    Backend selection:

    1. **In-memory (default)** — runs `_idr_grid_kernel` per grid cell.
       Dask-backed inputs are first materialized into RAM (the common laptop
       case). With ``use_parallel=True`` cells are distributed across ``n_jobs``
       processes via joblib/loky (no GIL); otherwise sequential.

    2. **Streaming dask** (``force_dask=True``) — escape hatch for grids too
       large to materialize: ``xr.apply_ufunc`` with ``dask='parallelized'``.
       Single-threaded per cell (``use_parallel``/``n_jobs`` do not apply), so
       it is the slow path — use it only when RAM is the binding constraint.
       With ``force_dask=False`` the in-memory path calls ``.load()``; if the
       materialized grid exceeds RAM that will OOM — pass ``force_dask=True``.

    Parameters
    ----------
    forecast_train, obs_train : training arrays (train_time, lat, lon)
    forecast_test, obs_test   : test arrays (time, lat, lon)
    spatial_dims  : lat/lon dimension names
    time_dim      : time dimension name on test arrays
    train_time_dim: time dimension name on training arrays (if different)
    min_valid     : minimum valid training samples per grid point
    min_wet_days  : minimum wet days (> 0.1 mm/hr) in training data
    use_parallel  : enable joblib parallelism for the in-memory backend
    n_jobs        : number of joblib workers (-1 = all cores)
    force_dask    : keep the streaming apply_ufunc path for out-of-RAM grids
    exceedance_thresholds : thresholds (data units) for the exceedance
                    probability P(X > t); empty = skip, output then carries
                    no p_exceed variable
    """
    tr_time_dim = train_time_dim if train_time_dim is not None else time_dim
    workers     = n_jobs if n_jobs is not None else _N_JOBS

    try:
        dask_mode = _is_dask_backed(forecast_train, obs_train,
                                     forecast_test, obs_test)
    except ImportError:
        dask_mode = False

    if dask_mode and force_dask:
        # Escape hatch for grids too large to materialize in RAM: stream via
        # apply_ufunc. Single-threaded (use_parallel / n_jobs do not apply here).
        log.info("idr_on_grid: force_dask — streaming apply_ufunc path")
        return _idr_on_grid_dask(
            forecast_train, obs_train, forecast_test, obs_test,
            spatial_dims, time_dim, tr_time_dim,
            min_valid, min_wet_days, exceedance_thresholds,
        )

    if dask_mode:
        # Materialize so the in-memory backend (which honors use_parallel/n_jobs)
        # can run; see the docstring for why this beats the streaming path.
        log.info("idr_on_grid: materializing dask-backed inputs for in-memory backend")
        forecast_train = forecast_train.load()
        obs_train = obs_train.load()
        forecast_test = forecast_test.load()
        obs_test = obs_test.load()

    lat_dim, lon_dim = spatial_dims
    nlat = forecast_test.sizes[lat_dim]
    nlon = forecast_test.sizes[lon_dim]
    log.info(
        f"idr_on_grid: {nlat}x{nlon} grid, "
        f"use_parallel={use_parallel}, n_jobs={workers}"
    )
    return _idr_on_grid_inmemory(
        forecast_train, obs_train, forecast_test, obs_test,
        spatial_dims, time_dim,
        min_valid, min_wet_days,
        use_parallel, workers, exceedance_thresholds, pit_seed,
    )


# ---------------------------------------------------------------------------
# 7.  LOYO kernel (one grid point, shared by all backends)
# ---------------------------------------------------------------------------

def _loyo_kernel(
    X_all: np.ndarray,
    y_all: np.ndarray,
    years_all: np.ndarray,
    unique_years: np.ndarray,
    min_valid: int,
    min_wet_days: int,
) -> np.ndarray:
    """Leave-One-Year-Out IDR for one grid point.  Returns (5, N)."""
    N   = len(X_all)
    out = np.full((5, N), np.nan)

    for yr in unique_years:
        test_mask  = years_all == yr
        train_mask = ~test_mask

        X_tr = X_all[train_mask]
        y_tr = y_all[train_mask]
        X_te = X_all[test_mask]
        y_te = y_all[test_mask]

        valid_tr = np.isfinite(X_tr) & np.isfinite(y_tr)
        if valid_tr.sum() < min_valid:
            continue
        if len(np.unique(y_tr[valid_tr])) < 2:
            continue
        if np.sum(y_tr[valid_tr] > 0.1) < min_wet_days:
            continue

        valid_te = np.isfinite(X_te) & np.isfinite(y_te)
        if not valid_te.any():
            continue

        model        = IDRModel(increasing=True).fit(X_tr[valid_tr], y_tr[valid_tr])
        preds        = model.predict(X_te[valid_te])
        q            = preds.quantile(np.array([0.1, 0.5, 0.9]))
        te_positions = np.where(test_mask)[0][valid_te]

        out[0, te_positions] = preds.crps(y_te[valid_te])
        if q.ndim == 2:
            out[1, te_positions] = q[:, 0]
            out[2, te_positions] = q[:, 1]
            out[3, te_positions] = q[:, 2]
        else:
            out[1, te_positions] = q[0]
            out[2, te_positions] = q[1]
            out[3, te_positions] = q[2]
        out[4, te_positions] = preds.pit(y_te[valid_te])

    return out


# ---------------------------------------------------------------------------
# 8a.  idr_loyo_cv — Dask backend
# ---------------------------------------------------------------------------

def _idr_loyo_cv_dask(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    spatial_dims: tuple[str, str],
    time_dim: str,
    min_valid: int,
    min_wet_days: int,
) -> xr.Dataset:
    """Dask-backed LOYO: xr.apply_ufunc with dask='parallelized'."""
    years_np     = forecast[time_dim].dt.year.values.astype(int)
    unique_years = np.unique(years_np)
    N            = len(years_np)

    years_da = xr.DataArray(
        np.broadcast_to(
            years_np,
            (forecast.sizes[spatial_dims[0]],
             forecast.sizes[spatial_dims[1]], N),
        ).copy(),
        dims=[spatial_dims[0], spatial_dims[1], time_dim],
        coords={
            spatial_dims[0]: forecast[spatial_dims[0]],
            spatial_dims[1]: forecast[spatial_dims[1]],
            time_dim:        forecast[time_dim],
        },
    )

    kernel = functools.partial(
        _loyo_kernel,
        unique_years=unique_years,
        min_valid=min_valid,
        min_wet_days=min_wet_days,
    )

    result = xr.apply_ufunc(
        kernel,
        forecast, obs, years_da,
        input_core_dims=[[time_dim], [time_dim], [time_dim]],
        output_core_dims=[["output", time_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {"output": 5, time_dim: N},
            "allow_rechunk": True,
        },
    )

    output_names = ["crps", "q10", "q50", "q90", "pit"]
    attrs = {
        "crps": {"units": "mm/day", "long_name": "IDR CRPS (LOYO CV)"},
        "q10":  {"units": "mm/day", "long_name": "IDR 10th percentile (LOYO CV)"},
        "q50":  {"units": "mm/day", "long_name": "IDR median (LOYO CV)"},
        "q90":  {"units": "mm/day", "long_name": "IDR 90th percentile (LOYO CV)"},
        "pit":  {"long_name": "IDR PIT (LOYO CV)"},
    }
    return xr.Dataset({
        name: result.isel(output=k).drop_vars("output", errors="ignore").assign_attrs(attrs[name])
        for k, name in enumerate(output_names)
    })


# ---------------------------------------------------------------------------
# 8b.  idr_loyo_cv — in-memory backend
# ---------------------------------------------------------------------------

def _run_loyo_point(i, j, fcst_np, obs_np, years_np,
                    lat_ax, lon_ax, unique_years, min_valid, min_wet_days):
    """Worker for one LOYO grid point."""
    sl = [slice(None)] * fcst_np.ndim
    sl[lat_ax] = i
    sl[lon_ax] = j
    t = tuple(sl)
    return _loyo_kernel(fcst_np[t], obs_np[t], years_np,
                        unique_years, min_valid, min_wet_days)


def _idr_loyo_cv_inmemory(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    spatial_dims: tuple[str, str],
    time_dim: str,
    min_valid: int,
    min_wet_days: int,
    use_parallel: bool,
    n_jobs: int,
) -> xr.Dataset:
    """In-memory LOYO: materialise then loop (sequential or joblib)."""
    years_np     = forecast[time_dim].dt.year.values.astype(int)
    unique_years = np.unique(years_np)
    N            = len(years_np)

    lat_dim, lon_dim = spatial_dims
    nlat = forecast.sizes[lat_dim]
    nlon = forecast.sizes[lon_dim]

    fcst_np = forecast.values
    obs_np  = obs.values

    dims_list = list(forecast.dims)
    lat_ax = dims_list.index(lat_dim)
    lon_ax = dims_list.index(lon_dim)

    results = _parallel_map(
        func=_run_loyo_point,
        iterable=(
            (i, j, fcst_np, obs_np, years_np,
             lat_ax, lon_ax, unique_years, min_valid, min_wet_days)
            for i in range(nlat) for j in range(nlon)
        ),
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )

    out = np.full((5, N, nlat, nlon), np.nan)
    idx = 0
    for i in range(nlat):
        for j in range(nlon):
            out[:, :, i, j] = results[idx]
            idx += 1

    output_names = ["crps", "q10", "q50", "q90", "pit"]
    attrs = {
        "crps": {"units": "mm/day", "long_name": "IDR CRPS (LOYO CV)"},
        "q10":  {"units": "mm/day", "long_name": "IDR 10th percentile (LOYO CV)"},
        "q50":  {"units": "mm/day", "long_name": "IDR median (LOYO CV)"},
        "q90":  {"units": "mm/day", "long_name": "IDR 90th percentile (LOYO CV)"},
        "pit":  {"long_name": "IDR PIT (LOYO CV)"},
    }
    coords = {
        time_dim:        forecast[time_dim],
        spatial_dims[0]: forecast[spatial_dims[0]],
        spatial_dims[1]: forecast[spatial_dims[1]],
    }
    return xr.Dataset({
        name: xr.DataArray(
            out[k], dims=[time_dim, spatial_dims[0], spatial_dims[1]],
            coords=coords, attrs=attrs[name],
        )
        for k, name in enumerate(output_names)
    })


# ---------------------------------------------------------------------------
# 8.  idr_loyo_cv — public API (auto-dispatches)
# ---------------------------------------------------------------------------

def idr_loyo_cv(
    forecast: xr.DataArray,
    obs: xr.DataArray,
    spatial_dims: tuple[str, str] = ("lat", "lon"),
    time_dim: str = "time",
    min_valid: int = 10,
    min_wet_days: int = 50,
    use_parallel: bool = False,
    n_jobs: int | None = None,
) -> xr.Dataset:
    """
    Leave-One-Year-Out cross-validation across a grid.

    Execution backend is chosen automatically:

    1. **Dask** — if inputs are Dask-backed, uses ``xr.apply_ufunc``.
    2. **joblib** — if ``use_parallel=True`` and inputs are in-memory.
    3. **Sequential** — default.

    Parameters
    ----------
    forecast     : (time, lat, lon)
    obs          : (time, lat, lon)
    spatial_dims : lat/lon dimension names
    time_dim     : time dimension name
    min_valid    : minimum valid training samples per fold
    min_wet_days : minimum wet days (> 0.1 mm) in training fold
    use_parallel : enable joblib parallelism for in-memory arrays
    n_jobs       : number of joblib workers (-1 = all cores)
    """
    workers = n_jobs if n_jobs is not None else _N_JOBS

    try:
        dask_mode = _is_dask_backed(forecast, obs)
    except ImportError:
        dask_mode = False

    if dask_mode:
        log.info("idr_loyo_cv: Dask-backed inputs — using apply_ufunc")
        return _idr_loyo_cv_dask(
            forecast, obs, spatial_dims, time_dim,
            min_valid, min_wet_days,
        )

    lat_dim, lon_dim = spatial_dims
    nlat = forecast.sizes[lat_dim]
    nlon = forecast.sizes[lon_dim]
    log.info(
        f"idr_loyo_cv: {nlat}x{nlon} grid, "
        f"use_parallel={use_parallel}, n_jobs={workers}"
    )
    return _idr_loyo_cv_inmemory(
        forecast, obs, spatial_dims, time_dim,
        min_valid, min_wet_days,
        use_parallel, workers,
    )


# ---------------------------------------------------------------------------
# 9.  LOYO summary diagnostics
# ---------------------------------------------------------------------------

def loyo_summary(cv_ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    mean_crps    = cv_ds.crps.mean(time_dim)
    crps_by_year = cv_ds.crps.groupby(f"{time_dim}.year").mean(time_dim)

    pit_vals      = cv_ds.pit.values.ravel()
    pit_vals      = pit_vals[np.isfinite(pit_vals)]
    counts, edges = np.histogram(pit_vals, bins=20, range=(0, 1), density=True)
    pit_histogram = xr.DataArray(
        counts,
        dims=["pit_bin"],
        coords={"pit_bin": 0.5 * (edges[:-1] + edges[1:])},
        attrs={"long_name": "PIT histogram (uniform = calibrated)", "expected": 1.0},
    )

    iqr_proxy      = (cv_ds.q90 - cv_ds.q10).mean(time_dim)
    low_skill_mask = (mean_crps > 0.5 * iqr_proxy).assign_attrs({
        "long_name":    "Low-skill mask (CRPS > 0.5 * IQR proxy)",
        "flag_meaning": "1 = low skill / data-sparse,  0 = adequate calibration",
    })

    return xr.Dataset({
        "mean_crps":      mean_crps.assign_attrs({"units": "mm/day", "long_name": "Time-mean IDR CRPS (LOYO CV)"}),
        "crps_by_year":   crps_by_year.assign_attrs({"units": "mm/day", "long_name": "Annual mean IDR CRPS (LOYO CV)"}),
        "pit_histogram":  pit_histogram,
        "low_skill_mask": low_skill_mask,
    })


# ---------------------------------------------------------------------------
# Zonal strategy — batched + optionally parallel
# ---------------------------------------------------------------------------

def _fit_zone_model(
    X_tr: xr.DataArray,
    y_tr: xr.DataArray,
    zone_indices: np.ndarray,
    master_lat: xr.DataArray,
    master_lon: xr.DataArray,
    rng: np.random.Generator,
) -> Optional[IDRModel]:
    num_locs  = min(len(zone_indices), ZONE_SAMPLE_LOCS)
    sel       = zone_indices[rng.choice(len(zone_indices), size=num_locs, replace=False)]
    X_z       = X_tr.sel(lat=master_lat[sel[:, 0]], lon=master_lon[sel[:, 1]]).values.flatten()
    y_z       = y_tr.sel(lat=master_lat[sel[:, 0]], lon=master_lon[sel[:, 1]]).values.flatten()
    valid     = np.isfinite(X_z) & np.isfinite(y_z)
    n_fit     = min(int(valid.sum()), NUM_DATA_PTS)
    if n_fit < 2:
        return None
    train_idx = rng.choice(np.where(valid)[0], size=n_fit, replace=False)
    return IDRModel(increasing=True).fit(X_z[train_idx], y_z[train_idx])


def _apply_zone_worker(
    zid, zone_models, kg, X_te, y_te, crps_map_shape, t_positions
):
    mask      = kg == zid
    local_map = np.full(crps_map_shape, np.nan, dtype=np.float32)
    _apply_zone_model(
        zone_models[zid], mask, X_te, y_te, local_map, t_positions
    )
    return zid, local_map


def _apply_zone_model(
    model: IDRModel,
    mask: xr.DataArray,
    X_te: xr.DataArray,
    y_te: xr.DataArray,
    crps_map: np.ndarray,
    t_positions: np.ndarray,
    chunk_size: int = 100_000,
    use_parallel: bool = False,
    n_jobs: int | None = None,
    parallel_backend: str = "threading",  # "threading" or "loky"
) -> None:
    """
    Memory-safe, optionally parallel, chunked application of a zone model.

    - Builds X_zone_mat (n_time, n_space) once (contiguous float32).
    - If n_valid <= chunk_size: do single batched predict+crps.
    - Else: split indices into chunks and optionally process chunks in parallel.
    - Workers return (chunk_idx, crps_vals); main thread scatters into crps_map.
    """
    import math

    from joblib import Parallel, delayed

    # 1) Build compact zone arrays once (no repeated .compute() in inner loops)
    # If X_te / y_te are Dask-backed, compute() here will pull into memory once.
    X_zone = X_te.where(mask, drop=True).compute()
    y_zone = y_te.where(mask, drop=True).compute()

    mask_zone = mask.where(mask, drop=True).values == 1
    if mask_zone.sum() == 0:
        # nothing to do
        return

    # rows, cols map each local-space index -> global lat/lon index
    rows, cols = np.where(mask.values)
    n_space = int(mask_zone.sum())
    n_time = len(t_positions)

    # Create compact 2D array (n_time, n_space) contiguous, float32
    # X_zone.values shape usually (time, lat_masked?, lon_masked?) but .values[:, mask_zone] gives (time, n_space)
    # Using astype(np.float32, copy=False) ensures smaller dtype and contiguous memory
    X_zone_mat = X_zone.values.astype(np.float32, copy=False)[:, mask_zone]  # shape (n_time, n_space)
    y_zone_mat = y_zone.values.astype(np.float32, copy=False)[:, mask_zone]  # same shape

    # Flatten to 1D order (time major)
    # We'll keep them available for chunk workers as views (no copy if .ravel() returns view)
    X_all = X_zone_mat.ravel()
    y_all = y_zone_mat.ravel()

    # Valid indices
    valid_mask = np.isfinite(X_all) & np.isfinite(y_all)
    indices = np.flatnonzero(valid_mask)
    n_valid = indices.size
    if n_valid == 0:
        # free temporaries
        del X_zone, y_zone, X_zone_mat, y_zone_mat, X_all, y_all
        gc.collect()
        return

    # Helper scatter function (main thread)
    def _scatter(idx_array, crps_vals_chunk):
        # idx_array: 1d array of flattened indices into X_all (time-major)
        time_idx = (idx_array // n_space).astype(int)
        space_idx = (idx_array % n_space).astype(int)
        # vectorized assign into crps_map: t_positions[time_idx] may have repeat order
        crps_map[t_positions[time_idx], rows[space_idx], cols[space_idx]] = crps_vals_chunk

    # Fast path: small enough to do in one go
    if n_valid <= chunk_size:
        X_batch = X_all[indices].astype(np.float32, copy=False)
        y_batch = y_all[indices].astype(np.float32, copy=False)
        # Single predict+crps
        preds = model.predict(X_batch)
        crps_vals = preds.crps(y_batch)
        _scatter(indices, crps_vals)
        # free
        del X_zone, y_zone, X_zone_mat, y_zone_mat, X_all, y_all, X_batch, y_batch, preds, crps_vals
        gc.collect()
        return

    # Large path: chunk indices
    # Build list of index arrays (each small)
    chunk_idx_arrays = []
    for start in range(0, n_valid, chunk_size):
        stop = min(start + chunk_size, n_valid)
        chunk_idx_arrays.append(indices[start:stop])

    # Worker for a single chunk (runs in worker thread/process)
    def _process_chunk(idx_array):
        # idx_array is a numpy array of flattened positions
        X_chunk = X_all[idx_array].astype(np.float32, copy=False)
        y_chunk = y_all[idx_array].astype(np.float32, copy=False)
        preds = model.predict(X_chunk)
        crps_chunk = preds.crps(y_chunk)
        return idx_array, crps_chunk  # return indices and values (picklable)

    # Decide parallel execution method:
    if use_parallel and len(chunk_idx_arrays) > 1:
        # prefer threading backend to avoid model pickling costs if predict is numpy/numba heavy
        backend = parallel_backend if parallel_backend in ("threading", "loky") else "threading"
        workers = n_jobs if n_jobs is not None else -1
        if backend == "threading":
            # Joblib threading backend
            results = Parallel(n_jobs=workers, backend="threading")(
                delayed(_process_chunk)(ci) for ci in chunk_idx_arrays
            )
        else:
            # loky backend (processes) — model will be pickled
            results = Parallel(n_jobs=workers, backend="loky")(
                delayed(_process_chunk)(ci) for ci in chunk_idx_arrays
            )
    else:
        # sequential
        results = [ _process_chunk(ci) for ci in chunk_idx_arrays ]

    # Scatter results in main thread
    for idx_array, crps_vals in results:
        _scatter(idx_array, crps_vals)

    # cleanup
    del X_zone, y_zone, X_zone_mat, y_zone_mat, X_all, y_all, results
    gc.collect()


def run_zonal_final(
    X_train: xr.DataArray,
    y_train: xr.DataArray,
    X_test: xr.DataArray,
    y_test: xr.DataArray,
    kg: xr.DataArray,
    master_lat: xr.DataArray,
    master_lon: xr.DataArray,
    use_parallel: bool = False,
    n_jobs: int | None = None,
) -> np.ndarray:
    rng          = np.random.default_rng()
    unique_zones = np.unique(kg.values)
    crps_map     = np.full(
        (len(X_test.time), len(master_lat), len(master_lon)), np.nan, dtype=np.float32
    )
    for zone_id in unique_zones:
        if zone_id <= 0:
            continue
        mask         = kg == zone_id
        zone_indices = np.argwhere(mask.values)
        log.info(f"  Zone {zone_id:3d} | {len(zone_indices):6d} pixels")
        model = _fit_zone_model(X_train, y_train, zone_indices, master_lat, master_lon, rng)
        if model is not None:
            _apply_zone_model(model, mask, X_test, y_test, crps_map, np.arange(len(X_test.time)),use_parallel=use_parallel, n_jobs=n_jobs)
        del model
        gc.collect()
    return crps_map


def run_zonal_loyo(
    X_all: xr.DataArray,
    y_all: xr.DataArray,
    kg: xr.DataArray,
    master_lat: xr.DataArray,
    master_lon: xr.DataArray,
    train_years: list[int],
    use_parallel: bool = False,
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Zonal LOYO — fit zone models sequentially but apply them optionally in parallel per fold.
    """
    # _parallel_map is a module-level function (defined above) — referenced directly;
    # the original module-name self-import is unnecessary here.

    rng          = np.random.default_rng()
    unique_zones = np.unique(kg.values)
    unique_zones = unique_zones[unique_zones > 0]
    years_all    = X_all.time.dt.year.values
    crps_map     = np.full(
        (len(X_all.time), len(master_lat), len(master_lon)), np.nan, dtype=np.float32
    )

    for yr in train_years:
        log.info(f"  LOYO fold: held-out {yr}")
        test_mask   = years_all == yr
        t_positions = np.where(test_mask)[0]

        # Fit all models sequentially (shared rng)
        zone_models = {}
        for zone_id in unique_zones:
            mask         = kg == zone_id
            zone_indices = np.argwhere(mask.values)
            model = _fit_zone_model(
                X_all.isel(time=~test_mask),
                y_all.isel(time=~test_mask),
                zone_indices, master_lat, master_lon, rng,
            )
            if model is not None:
                zone_models[zone_id] = model

        # === Use _parallel_map for parallel/sequential application of zone models ===
        results = _parallel_map(
            func=_apply_zone_worker,
            iterable=(
                (zid, zone_models, kg, X_all.isel(time=test_mask),
                 y_all.isel(time=test_mask), crps_map.shape, t_positions)
                for zid in zone_models.keys()
            ),
            use_parallel=use_parallel,
            n_jobs=n_jobs if n_jobs is not None else os.cpu_count(),
        )
        # Assign each result block
        for zid, local_map in results:
            valid = np.isfinite(local_map)
            crps_map[valid] = local_map[valid]
    return crps_map

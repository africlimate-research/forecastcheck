"""Determinism guarantees of the IDR engine (``calibration.idr``).

The refactor's equivalence gate compares re-run output against a snapshot bit-for-bit,
which is only a fair test if the engine is actually deterministic. Every output except
``pit`` already was; ``pit`` drew from a seedless randomized PIT and so could never be
validated. These tests pin down what is now guaranteed:

  Q1  ``pit`` is identical between a sequential run and a parallel one — the seed is
      derived from the CELL INDEX, so it cannot depend on worker count or on the order
      in which workers happen to finish
  Q2  every other output is identical across worker counts (the property the gate
      already relied on, now asserted rather than assumed)
  Q3  ``pit_seed=None`` restores the historical seedless behaviour
  Q4  different cells get different PIT streams — a per-cell seed must not collapse
      into one shared stream, which would correlate neighbouring cells
  Q5  re-running with the same seed reproduces ``pit`` exactly

Runs standalone or under pytest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.idr import _cell_seed, idr_on_grid

_NLAT, _NLON, _NTRAIN, _NTEST = 2, 3, 120, 20


def _grid(n_time, seed, start="2016-01-01", dry_frac=0.45):
    """A (time, lat, lon) rainfall-shaped field with enough wet days to fit.

    Crucially this includes DRY DAYS (exact zeros). The randomized PIT only randomizes
    where the observation lands on a jump in the predictive CDF — with strictly
    continuous values ``pit_lower == pit_upper`` and the seed has no effect at all, so
    a tie-free fixture would silently pass every seeding assertion for the wrong
    reason. Real precipitation is zero-inflated, which is exactly where this matters.
    """
    rng = np.random.default_rng(seed)
    vals = rng.gamma(shape=0.6, scale=2.0, size=(n_time, _NLAT, _NLON))
    vals[rng.random(vals.shape) < dry_frac] = 0.0
    return xr.DataArray(
        vals, dims=("time", "lat", "lon"),
        coords={"time": pd.date_range(start, periods=n_time, freq="D"),
                "lat": np.arange(_NLAT, dtype=float),
                "lon": np.arange(_NLON, dtype=float)},
    )


def _inputs():
    """(fc_train, obs_train, fc_test, obs_test) — obs correlated with the forecast."""
    fc_tr = _grid(_NTRAIN, 1)
    fc_te = _grid(_NTEST, 3, start="2022-01-01")
    # Obs is its own zero-inflated draw rather than a scaled copy of the forecast: any
    # arithmetic on the forecast would smear the exact zeros away and remove the ties.
    obs_tr = _grid(_NTRAIN, 2)
    obs_te = _grid(_NTEST, 4, start="2022-01-01")
    return fc_tr, obs_tr, fc_te, obs_te


def _run(*, n_jobs, use_parallel, pit_seed=0):
    fc_tr, obs_tr, fc_te, obs_te = _inputs()
    return idr_on_grid(
        fc_tr, obs_tr, fc_te, obs_te,
        spatial_dims=("lat", "lon"), time_dim="time",
        min_valid=5, min_wet_days=5,
        use_parallel=use_parallel, n_jobs=n_jobs,
        exceedance_thresholds=(0.5, 2.0),
        pit_seed=pit_seed,
    )


def test_q1_pit_is_independent_of_worker_count():
    """Q1: seeded pit is identical sequential vs parallel — the whole point of per-cell seeds."""
    seq = _run(n_jobs=1, use_parallel=False)
    par = _run(n_jobs=-1, use_parallel=True)
    a, b = seq["pit"].values, par["pit"].values
    assert np.isfinite(a).any(), "no calibrated cells — the fixture failed the fit gates"
    np.testing.assert_array_equal(a, b)


def test_q2_every_other_output_is_worker_count_independent():
    """Q2: crps/quantiles/p_exceed identical across backends — what the gate assumes."""
    seq = _run(n_jobs=1, use_parallel=False)
    par = _run(n_jobs=-1, use_parallel=True)
    for name in ("crps", "q5", "q50", "q95", "q99", "p_exceed"):
        np.testing.assert_array_equal(
            seq[name].values, par[name].values,
            err_msg=f"{name} differs between sequential and parallel")


def test_q3_pit_seed_none_restores_seedless_behaviour():
    """Q3: pit_seed=None is still random per run; the other outputs stay stable."""
    a = _run(n_jobs=1, use_parallel=False, pit_seed=None)
    b = _run(n_jobs=1, use_parallel=False, pit_seed=None)
    pa, pb = a["pit"].values, b["pit"].values
    both = np.isfinite(pa) & np.isfinite(pb)
    assert both.any()
    # Randomized PIT can coincide on degenerate cells, so require only that the two
    # draws are not wholly identical.
    assert not np.array_equal(pa[both], pb[both]), "pit_seed=None should not be reproducible"
    np.testing.assert_array_equal(a["crps"].values, b["crps"].values)


def test_q4_cells_get_distinct_pit_streams():
    """Q4: a per-cell seed must not degenerate into one shared stream across cells."""
    seeds = [_cell_seed(0, i, j) for i in range(_NLAT) for j in range(_NLON)]
    draws = [np.random.default_rng(s).uniform(size=8) for s in seeds]
    for k in range(1, len(draws)):
        assert not np.array_equal(draws[0], draws[k]), "two cells share a PIT stream"
    assert _cell_seed(None, 0, 0) is None


def test_q5_same_seed_reproduces_pit():
    """Q5: re-running with the same seed gives bit-identical pit."""
    a = _run(n_jobs=1, use_parallel=False, pit_seed=7)
    b = _run(n_jobs=1, use_parallel=False, pit_seed=7)
    np.testing.assert_array_equal(a["pit"].values, b["pit"].values)
    # and a different seed really does change it
    c = _run(n_jobs=1, use_parallel=False, pit_seed=8)
    both = np.isfinite(a["pit"].values) & np.isfinite(c["pit"].values)
    assert not np.array_equal(a["pit"].values[both], c["pit"].values[both])


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

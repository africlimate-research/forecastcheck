"""Tests for the vendored engine shim (forecastcheck.viz / forecastcheck.paths).

Runs standalone (``python tests/test_shim.py``) or under pytest. Uses a tiny synthetic
results tree so it needs no real data.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.services.engine import (SummarySet, _classify_target_kind, kind_of,
                                 open_summary)


def _make_summary(root, forecast, var, target, metric, year, lead):
    """Write a minimal annual/monthly summary file at the engine's path layout.

    ``target`` is a truth-source token (``chirts-gridded``); on disk it is two segments.
    """
    from verification.store import split_truth_source

    product, kind = split_truth_source(target)
    d = (Path(root) / forecast / var / product / kind / metric / str(year)
         / "annual_monthly_averages")
    d.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"annual_average": (("year",), [0.5]),
         "annual_spatial_average": (("year", "lat", "lon"), np.ones((1, 2, 2))),
         "monthly_average": (("month",), np.arange(1, 13.0)),
         "monthly_spatial_average": (("month", "lat", "lon"), np.ones((12, 2, 2)))},
        coords={"year": [int(year)], "lat": [0.0, 1.0], "lon": [0.0, 1.0],
                "month": list(range(1, 13))},
    )
    path = d / f"{lead}h.nc"
    ds.to_netcdf(path)
    return path


def test_shim_discover_index():
    """discover finds a known-metric summary with the right tokens, leads, and target_kind."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "africa"
        _make_summary(root, "graphcast", "precipitation", "chirts-gridded", "crps", 2022, 120)
        df = SummarySet.discover(root).index
        assert len(df) == 1
        r = df.iloc[0]
        assert r["forecast"] == "graphcast" and r["target"] == "chirts-gridded" and r["metric"] == "crps"
        assert r["native_var"] == "precipitation"
        assert r["leads"] == [120]
        assert r["target_kind"] == "gridded"


def test_shim_excludes_curve_metrics():
    """discover ignores metrics outside KNOWN_METRICS (curve metrics live elsewhere)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "africa"
        _make_summary(root, "graphcast", "precipitation", "chirts-gridded", "crps", 2022, 120)
        _make_summary(root, "graphcast", "precipitation", "chirts-gridded", "roc", 2022, 120)
        df = SummarySet.discover(root).index
        assert set(df["metric"]) == {"crps"}


def test_shim_open_summary():
    """open_summary loads the four summary variables for a run + lead."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "africa"
        _make_summary(root, "graphcast", "precipitation", "chirts-gridded", "crps", 2022, 120)
        ds = open_summary("graphcast", "precipitation", "chirts-gridded", "crps", "2022", 120, root=root)
        assert {"annual_average", "annual_spatial_average",
                "monthly_average", "monthly_spatial_average"}.issubset(ds.data_vars)


def test_open_summary_missing_raises_cleanly():
    """A missing lead file raises FileNotFoundError (no HDF5 error-stack spam)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "africa"
        _make_summary(root, "graphcast", "precipitation", "chirts-gridded", "crps", 2022, 120)
        try:
            open_summary("graphcast", "precipitation", "chirts-gridded", "crps", "2022", 999, root=root)
            assert False, "expected FileNotFoundError for a missing lead"
        except FileNotFoundError:
            pass


def test_kind_of():
    """kind_of maps variable names to precip / Tmax / Tmin / unknown."""
    assert kind_of("precipitation") == "precip"
    assert kind_of("temperature_max") == "Tmax"
    assert kind_of("temperature_min") == "Tmin"
    assert kind_of("geopotential") == "unknown"


def test_classify_target_kind_gridded():
    """lat/lon dims classify as gridded (the engine only checked latitude/longitude)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_summary(Path(tmp) / "africa", "graphcast",
                             "precipitation", "chirts-gridded", "crps", 2022, 120)
        assert _classify_target_kind("chirts-gridded", path) == "gridded"


def test_ui_import_smoke():
    """The renamed UI package imports cleanly (catches the local_engine->forecastcheck rename)."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("SKIP test_ui_import_smoke (streamlit not installed in this env)")
        return
    import importlib
    importlib.import_module("ui.services.engine")
    importlib.import_module("ui.pages.layout")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)

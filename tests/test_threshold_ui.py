"""Tests for the threshold-tab data layer: curve discovery, loading, and plot-data helpers.

Builds a tiny real curve tree with verification.diagnostics, so it exercises the actual
stored schema. Runs standalone (``python tests/test_threshold_ui.py``) or under pytest.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import config, diagnostics
from ui.components.threshold_charts import cost_loss_xy, reliability_xy, roc_curve_xy
from ui.services.threshold_data import discover_curves, load_curve

_THRESHOLDS = [20, 40]
_BANDS = ["all", "0-1000"]
_LEAD = 120


def _tiny_samples():
    """Small synthetic Samples covering two thresholds and two altitude bands."""
    rng = np.random.default_rng(0)
    pairs = {}
    for thr in _THRESHOLDS:
        for band in _BANDS:
            p = rng.uniform(0, 1, 3000)
            o = (rng.uniform(0, 1, 3000) < 0.3).astype(int)
            pairs[(thr, band)] = (p, o)
    return diagnostics.Samples(lead_hours=_LEAD, pairs=pairs)


def _build_tree(outdir):
    """Write roc/cost_loss/reliability curve files (+ a scalar CRPS summary) under outdir.

    write_dataset nests everything under the deployment domain (config.domain(),
    driven by VERIFICATION_COUNTRY), so the tree — and the root returned here —
    must use the same segment or the tests fail on any per-country deployment.
    """
    s = _tiny_samples()
    roc = diagnostics.compute_roc(s)
    diagnostics.write_dataset(roc, "roc", _LEAD, outdir)
    diagnostics.write_dataset(diagnostics.compute_cost_loss(roc), "cost_loss", _LEAD, outdir)
    diagnostics.write_dataset(diagnostics.compute_reliability(s, n_bins=10),
                              "reliability", _LEAD, outdir)
    d = (Path(outdir) / "graphcast" / "precipitation" / "chirts-gridded" / "gridded"
         / "crps" / "2022" / "annual_monthly_averages")
    d.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"annual_average": (("year",), [0.5])},
               coords={"year": [2022]}).to_netcdf(d / f"{_LEAD}h.nc")
    return Path(outdir)


def test_discover_curves_lists_three_and_excludes_crps():
    """discover_curves finds the three curve metrics and ignores the scalar CRPS summary."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_tree(tmp)
        df = discover_curves(root)
        assert set(df["metric"]) == {"reliability", "roc", "cost_loss"}
        assert set(df["forecast"]) == {"graphcast"} and set(df["target"]) == {"chirts-gridded"}
        assert all(r == [_LEAD] for r in df["leads"])


def test_load_curve_opens_each():
    """load_curve opens each metric file with its expected variables."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_tree(tmp)
        roc = load_curve("graphcast", "precipitation", "chirts-gridded", "roc", "2022", _LEAD, root)
        rel = load_curve("graphcast", "precipitation", "chirts-gridded", "reliability", "2022", _LEAD, root)
        cl = load_curve("graphcast", "precipitation", "chirts-gridded", "cost_loss", "2022", _LEAD, root)
        assert "hits" in roc.data_vars
        assert "forecast_probability" in rel.data_vars
        assert "value" in cl.data_vars


def test_chart_data_helpers():
    """The plot-data helpers return arrays of the right shape and range."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_tree(tmp)
        roc = load_curve("graphcast", "precipitation", "chirts-gridded", "roc", "2022", _LEAD, root)
        rel = load_curve("graphcast", "precipitation", "chirts-gridded", "reliability", "2022", _LEAD, root)
        cl = load_curve("graphcast", "precipitation", "chirts-gridded", "cost_loss", "2022", _LEAD, root)

        fpr, tpr, auc = roc_curve_xy(roc, 20, "all")
        assert len(fpr) == len(tpr) and 0.0 <= auc <= 1.0
        assert fpr[0] == 0.0 and fpr[-1] == 1.0

        fp, of, lo, hi = reliability_xy(rel, 20, "all")
        assert len(fp) == len(of) == len(lo) == len(hi) == 10

        alphas, value, base_rate = cost_loss_xy(cl, 20, "all")
        assert len(alphas) == len(value) == len(diagnostics.COST_LOSS_RATIOS)
        assert 0.0 <= base_rate <= 1.0


def test_curve_for_a_season_with_no_file_on_disk():
    """A season the pipeline never wrote a file for must still render.

    This is the whole point of storing counts per region and per month rather than one
    curve file per combination: a season is a set of months summed at read time, so adding
    one — or drilling into an administrative level nobody pooled — costs no pipeline run.
    If this fails, the dashboard has quietly gone back to reading pre-pooled files and the
    flexibility exists only on paper.
    """
    import os
    import tempfile

    from ui.services.threshold_data import discover_curves, load_curve
    from verification import diagnostics as diag

    prev = os.environ.get("VERIFICATION_COUNTRY")
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        from verification import admin
        admin.clear_cache()
        # Two regions and a full calendar year of days, so both a sub-national selection
        # and a season have something to sum.
        samples = _tiny_samples()
        n_days = 365
        samples.blocks = {k: np.arange(v[0].size) % n_days for k, v in samples.pairs.items()}
        samples.regions = {k: np.arange(v[0].size) % 2 for k, v in samples.pairs.items()}
        samples.region_ids = ("alpha", "beta")
        samples.day_months = np.asarray([(i // 31) % 12 + 1 for i in range(n_days)],
                                        dtype=np.int8)

        with tempfile.TemporaryDirectory() as tmp:
            store = diag.compute_roc_by_region(samples)
            assert store is not None
            # ONLY the store is written — no per-season, no per-region curve files at all
            diag.write_dataset(store, "roc", _LEAD, tmp, subdir="by_region",
                               model="graphcast-idr")
            # An explicit outdir IS the diagnostics root: the domain sits above
            # diagnostics/ in the layout, so it is not appended to a caller's base.
            root = Path(tmp)

            written = {p.name for p in root.rglob("*h.nc")}
            assert written == {f"{_LEAD}h.nc"}, written

            df = discover_curves(root)
            assert "ndjf" in set(df["season"]), sorted(set(df["season"]))
            assert "alpha" not in set(df["region"])   # ids are internal; labels are offered

            # a season with no file, and the whole country, both render from the store
            for season in ("yearly", "ndjf"):
                ds = load_curve("graphcast-idr", "precipitation", "chirts-gridded", "roc",
                                config_year(), _LEAD, root=root, season=season)
                assert "hits" in ds.data_vars, season
                assert ds["hits"].dims == ("threshold", "altitude", "prob"), season
    finally:
        if prev is None:
            os.environ.pop("VERIFICATION_COUNTRY", None)
        else:
            os.environ["VERIFICATION_COUNTRY"] = prev


def config_year():
    """The year the tiny fixture is written under."""
    from verification.diagnostics import DEFAULT_YEAR

    return str(DEFAULT_YEAR)


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

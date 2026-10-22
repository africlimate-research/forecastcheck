"""Tests for per-admin-1-region threshold curves.

Covers the region path facet + discovery + loading (env-independent), the region-less
no-op that keeps the whole-country path byte-for-byte unchanged, and — under a
VERIFICATION_COUNTRY=Senegal deployment — that subsetting crops+masks the inputs to
exactly the department polygon before pooling. Runs standalone or under pytest.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import config, diagnostics as d
from verification.regions import default_region
from ui.pages.threshold.threshold import _load_series
from ui.services.threshold_data import discover_curves, load_curve


def _tiny_roc(bands=("all", "0-1000")):
    rng = np.random.default_rng(0)
    pairs = {(t, b): (rng.uniform(0, 1, 2000), (rng.uniform(0, 1, 2000) < 0.3).astype(int))
             for t in [20, 40] for b in bands}
    return d.compute_roc(d.Samples(lead_hours=120, pairs=pairs, bands=list(bands)))


def test_region_less_subset_is_identity():
    """region=None / an unknown region leaves the inputs untouched (whole-country path unchanged)."""
    lat = np.linspace(3, 15, 40); lon = np.linspace(33, 48, 50)
    fc = xr.Dataset({"v": (("lat", "lon"), np.random.rand(40, 50))}, coords={"lat": lat, "lon": lon})
    obs = xr.Dataset({"precip": (("time", "lat", "lon"), np.random.rand(2, 40, 50))},
                     coords={"time": [0, 1], "lat": lat, "lon": lon})
    assert d._subset_gridded_to_region(fc, obs, None) == (fc, obs)
    assert d._subset_gridded_to_region(fc, obs, "NotARegion") == (fc, obs)


def test_output_path_region_segment():
    """A subregion inserts a /{region}/ segment after the year; whole country stays region-less."""
    whole = str(d.output_path("roc", 120, "out", model="graphcast-idr", region=None))
    sub = str(d.output_path("roc", 120, "out", model="graphcast-idr", region="Pikine"))
    assert whole.endswith("/2022/120h.nc") and "Pikine" not in whole
    assert sub.endswith("/2022/Pikine/120h.nc")


def test_output_path_season_and_region_segments():
    """Season inserts a slug segment after the (optional) region one; yearly stays season-less."""
    mk = lambda **k: str(d.output_path("roc", 120, "out", model="graphcast-idr", **k))
    # 4 combos: (region, season) × (whole/admin1) × (yearly/jaso)
    assert mk(region=None, season=None).endswith("/2022/120h.nc")             # yearly + country
    assert mk(region=None, season="yearly").endswith("/2022/120h.nc")          # yearly is season-less
    assert mk(region=None, season="jaso").endswith("/2022/jaso/120h.nc")   # season + country
    assert mk(region="Podor", season=None).endswith("/2022/Podor/120h.nc")   # yearly + admin1
    assert mk(region="Podor", season="jaso").endswith("/2022/Podor/jaso/120h.nc")


def test_parse_curve_path_disambiguates_region_and_season():
    """_parse_curve_path classifies the 0–2 middle segments into (region, season) unambiguously."""
    from ui.services.threshold_data import _parse_curve_path
    root = "out"
    base = f"{root}/graphcast-idr/precipitation/chirts/gridded/roc/2022"

    def parse(path):
        return _parse_curve_path(Path(path), root)

    r = parse(f"{base}/120h.nc")                       # yearly + whole country
    assert r["region"] == default_region() and r["season"] == "yearly"
    r = parse(f"{base}/jaso/120h.nc")                # season + whole country
    assert r["region"] == default_region() and r["season"] == "jaso"
    r = parse(f"{base}/Podor/120h.nc")                # yearly + admin-1 (region, not a season slug)
    assert r["region"] == "Podor" and r["season"] == "yearly"
    r = parse(f"{base}/Podor/jaso/120h.nc")         # season + admin-1
    assert r["region"] == "Podor" and r["season"] == "jaso"


def test_build_samples_month_filter_shrinks_pool():
    """build_samples(months=...) pools strictly fewer pairs than the whole year (fewer days)."""
    import numpy as np
    import xarray as xr

    from verification import diagnostics as dd
    # Two forecast days: one in January (month 1), one in July (month 7).
    times = np.array(["2022-01-15", "2022-07-15"], dtype="datetime64[ns]")
    lat = np.array([9.0, 10.0])
    lon = np.array([38.0, 39.0])
    p_exceed = xr.DataArray(  # (threshold, time, lat, lon)
        np.full((1, 2, 2, 2), 0.4),
        dims=("threshold", "time", "lat", "lon"),
        coords={"threshold": [20], "time": times, "lat": lat, "lon": lon})
    fc = xr.Dataset({"p_exceed": p_exceed}, coords={"time": times, "lat": lat, "lon": lon})
    obs = xr.Dataset({"precip": (("time", "lat", "lon"), np.full((2, 2, 2), 30.0))},
                     coords={"time": times, "lat": lat, "lon": lon})
    # Yearly keeps both days; a single-month filter keeps one.
    _, all_days = dd._subset_to_months(fc, obs, None)
    _, jul_only = dd._subset_to_months(fc, obs, (6, 7, 8, 9))
    assert all_days.sizes["time"] == 2 and jul_only.sizes["time"] == 1
    # None / empty is a no-op (identity).
    assert dd._subset_to_months(fc, obs, ())[0].sizes["time"] == 2


def test_discover_and_load_whole_country_and_subregion():
    """Region-less files discover as the whole country; region files discover as themselves."""
    with tempfile.TemporaryDirectory() as tmp:
        roc = _tiny_roc()
        d.write_dataset(roc, "roc", 120, tmp, model="graphcast-idr", region=None)
        d.write_dataset(roc, "roc", 120, tmp, model="graphcast-idr", region="Pikine")
        # An explicit outdir IS the diagnostics root (the domain sits above
        # diagnostics/ in the layout, so it is not appended to a caller's base).
        root = Path(tmp)
        df = discover_curves(root)
        assert set(df["region"]) == {default_region(), "Pikine"}
        for region in (default_region(), "Pikine"):
            ds = load_curve("graphcast-idr", "precipitation", "chirts-gridded", "roc", "2022",
                            120, root, region=region)
            assert "hits" in ds.data_vars


def test_senegal_department_subset_crops_and_masks():
    """Under Senegal, subsetting keeps obs finite exactly at the department polygon's cells."""
    prev = os.environ.get("VERIFICATION_COUNTRY")
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        from verification.regions import _get_subregions, _subregion_keep
        if "Podor" not in _get_subregions():
            print("skip: Senegal shapefile not available")
            return
        lat = np.linspace(12, 17, 240); lon = np.linspace(-18, -11, 300)
        fc = xr.Dataset({"v": (("lat", "lon"), np.random.rand(240, 300))},
                        coords={"lat": lat, "lon": lon})
        obs = xr.Dataset({"precip": (("time", "lat", "lon"), np.random.rand(2, 240, 300))},
                         coords={"time": [0, 1], "lat": lat, "lon": lon})
        fcA, obsA = d._subset_gridded_to_region(fc, obs, "Podor")
        assert fcA.sizes["lat"] < 240 and fcA.sizes["lon"] < 300
        keep = _subregion_keep("Podor", fcA["lon"].values, fcA["lat"].values)
        assert keep.any()
        finite = np.isfinite(obsA["precip"].isel(time=0).values)
        assert np.array_equal(finite, keep)
    finally:
        if prev is None:
            os.environ.pop("VERIFICATION_COUNTRY", None)
        else:
            os.environ["VERIFICATION_COUNTRY"] = prev


def test_empty_pool_does_not_crash():
    """A band with no samples (common for a small subregion) yields NaN/zero curves, not a crash.

    Regression guard for the reliability np.quantile([]) / cost-loss 0/0 edge cases that only
    appear once the pool is subset to a small admin-1 region.
    """
    pairs = {(20, "all"): (np.array([]), np.array([], dtype=int))}
    s = d.Samples(lead_hours=120, pairs=pairs, thresholds=[20], bands=["all"],
                  blocks={(20, "all"): np.array([], dtype=int)})
    rel = d.compute_reliability(s, n_bins=10, n_bootstraps=5)
    assert np.isnan(rel["forecast_probability"].values).all()
    roc = d.compute_roc(s)
    cl = d.compute_cost_loss(roc)               # must not raise
    assert np.array_equal(cl["value"].values, np.zeros_like(cl["value"].values))


def test_load_series_skips_model_missing_band():
    """_load_series drops a model whose curve file lacks the selected altitude band.

    Safety net for models generated with different --altitude sets: a band absent from one
    model's file must yield the "No data" notice, not a KeyError from ds.sel(altitude=band).
    """
    with tempfile.TemporaryDirectory() as tmp:
        d.write_dataset(_tiny_roc(("all", "0-1000")), "roc", 120, tmp,
                        model="graphcast-idr", region=None)
        d.write_dataset(_tiny_roc(("all",)), "roc", 120, tmp,
                        model="ifs-idr", region=None)
        # An explicit outdir IS the diagnostics root (the domain sits above
        # diagnostics/ in the layout, so it is not appended to a caller's base).
        root = Path(tmp)
        sel = pd.DataFrame([
            {"forecast": "graphcast-idr", "metric": "roc", "leads": [120]},
            {"forecast": "ifs-idr", "metric": "roc", "leads": [120]},
        ])
        # band present in both → both models kept (yearly season = the season-less file)
        both = _load_series(["graphcast-idr", "ifs-idr"], "precipitation", "chirts-gridded",
                            "roc", "2022", 120, root, sel, default_region(), "yearly", "all")
        assert len(both) == 2
        # band only in graphcast-idr → ifs-idr skipped, no KeyError
        one = _load_series(["graphcast-idr", "ifs-idr"], "precipitation", "chirts-gridded",
                           "roc", "2022", 120, root, sel, default_region(), "yearly", "0-1000")
        assert [label for label, _ in one] and len(one) == 1


def test_threshold_pdf_report_builds():
    """generate_threshold_report assembles a valid PDF from the three curve series.

    Builds tiny reliability/ROC/cost-loss datasets in memory and asserts a real PDF comes back
    (``%PDF`` header, non-trivial size). Figure embedding needs kaleido; the assembly path and the
    graceful "figure unavailable" fallback are exercised regardless of whether it's installed.
    """
    from ui.components.threshold_pdf_report import generate_threshold_report
    rng = np.random.default_rng(0)
    pairs = {(20, "all"): (rng.uniform(0, 1, 2000), (rng.uniform(0, 1, 2000) < 0.3).astype(int))}
    s = d.Samples(lead_hours=24, pairs=pairs, thresholds=[20], bands=["all"],
                  blocks={(20, "all"): np.zeros(2000, dtype=int)})
    roc = d.compute_roc(s)
    rel = d.compute_reliability(s, n_bins=10, n_bootstraps=5)
    cl = d.compute_cost_loss(roc)
    series_by_metric = {"reliability": [("M", rel)], "roc": [("M", roc)], "cost_loss": [("M", cl)]}
    pdf = generate_threshold_report(
        year="2022", variable="precip", truth="chirts-gridded", region="Senegal", season="jaso",
        models_label="M", lead=24, threshold=20, band="all", series_by_metric=series_by_metric)
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000

    # An empty series for a metric must not crash — it renders a "no data" caption instead.
    pdf2 = generate_threshold_report(
        year="2022", variable="precip", truth="chirts-gridded", region="Senegal", season="yearly",
        models_label="M", lead=24, threshold=20, band="all",
        series_by_metric={"reliability": [], "roc": [], "cost_loss": []})
    assert pdf2[:4] == b"%PDF"


def _run():
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
    sys.exit(1 if _run() else 0)

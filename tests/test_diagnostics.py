"""Verification tests for the diagnostics foundation (step 1: build_samples).

Runs standalone (``python tests/test_diagnostics.py``) — no pytest required — and
also works under pytest if installed. Uses tiny synthetic data with known answers.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification.diagnostics import (ALTITUDE_BANDS, THRESHOLDS_MM_DAY,
                                       altitude_masks, annual_monthly_averages,
                                       band_masks_for, compute_cost_loss,
                                       compute_reliability, compute_roc,
                                       load_obs, pool_pairs, roc_counts, Samples,
                                       assign_bins, derive_reliability_bands,
                                       corr_annual_monthly_averages)


def _roc_ds(hits, false_alarms, misses, correct_negatives, probs):
    """A minimal single (threshold, altitude) ROC Dataset from count arrays."""
    shape = (1, 1, len(probs))
    return xr.Dataset(
        {"hits": (("threshold", "altitude", "prob"), np.reshape(hits, shape)),
         "false_alarms": (("threshold", "altitude", "prob"), np.reshape(false_alarms, shape)),
         "misses": (("threshold", "altitude", "prob"), np.reshape(misses, shape)),
         "correct_negatives": (("threshold", "altitude", "prob"),
                               np.reshape(correct_negatives, shape))},
        coords={"threshold": [20], "altitude": ["all"], "prob": probs},
    )


def _brute_roc(p, o, probs):
    """Reference contingency counts via an explicit per-cut-off boolean scan."""
    p = np.asarray(p)
    o = np.asarray(o).astype(bool)
    fc = p[None, :] >= probs[:, None]
    return {
        "hits": np.sum(fc & o[None, :], axis=1),
        "false_alarms": np.sum(fc & ~o[None, :], axis=1),
        "misses": np.sum(~fc & o[None, :], axis=1),
        "correct_negatives": np.sum(~fc & ~o[None, :], axis=1),
    }


def _auc(counts):
    """Area under the ROC curve from contingency counts, with (0,0)/(1,1) endpoints."""
    tpr = counts["hits"] / (counts["hits"] + counts["misses"])
    fpr = counts["false_alarms"] / (counts["false_alarms"] + counts["correct_negatives"])
    fpr = np.concatenate([[0.0], fpr[::-1], [1.0]])
    tpr = np.concatenate([[0.0], tpr[::-1], [1.0]])
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return trapezoid(tpr, fpr)


def test_a1_exceedance_and_rename():
    """A1: pooled o equals (precip > threshold) and gridded-obs Lat/Lon load as lat/lon."""
    rng = np.random.default_rng(0)
    precip_v = rng.uniform(0, 80, size=(4, 3, 3))
    p_v = rng.uniform(0, 1, size=(len(THRESHOLDS_MM_DAY), 4, 3, 3))
    p_exceed = xr.DataArray(p_v, dims=("threshold", "time", "lat", "lon"),
                            coords={"threshold": THRESHOLDS_MM_DAY})
    precip = xr.DataArray(precip_v, dims=("time", "lat", "lon"))
    masks = {"all": np.ones((3, 3), dtype=bool)}

    pairs, _, _ = pool_pairs(p_exceed, precip, masks)
    for i, thr in enumerate(THRESHOLDS_MM_DAY):
        p, o = pairs[(thr, "all")]
        np.testing.assert_array_equal(o.astype(bool), (precip_v > thr).ravel())
        np.testing.assert_allclose(np.sort(p), np.sort(p_v[i].ravel()))

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "obs.nc"
        xr.Dataset(
            {"precip": (("time", "Lat", "Lon"), precip_v)},
            coords={"time": np.arange(4), "Lat": [1.0, 2.0, 3.0], "Lon": [4.0, 5.0, 6.0]},
        ).to_netcdf(path)
        obs = load_obs(path)
    assert "lat" in obs.coords and "lon" in obs.coords
    assert "Lat" not in obs.coords and "Lon" not in obs.coords


def test_a2_altitude_bands():
    """A2: each band mask selects exactly the in-range cells; ocean (<0) excluded."""
    elev = xr.DataArray(np.array([[-5.0, 500.0], [1500.0, 3500.0]]), dims=("lat", "lon"))
    masks = altitude_masks(elev)
    np.testing.assert_array_equal(masks["0-1000"], [[False, True], [False, False]])
    np.testing.assert_array_equal(masks["1000-2000"], [[False, False], [True, False]])
    np.testing.assert_array_equal(masks["2000-3000"], [[False, False], [False, False]])
    np.testing.assert_array_equal(masks[">3000"], [[False, False], [False, True]])
    assert not masks["0-1000"][0, 0] and not masks["1000-2000"][0, 0]


def test_a3_threshold_and_band_labels():
    """A3: pooled keys carry mm/day thresholds and the five altitude-band labels."""
    rng = np.random.default_rng(1)
    p_exceed = xr.DataArray(rng.uniform(0, 1, (len(THRESHOLDS_MM_DAY), 2, 2, 2)),
                            dims=("threshold", "time", "lat", "lon"),
                            coords={"threshold": THRESHOLDS_MM_DAY})
    precip = xr.DataArray(rng.uniform(0, 80, (2, 2, 2)), dims=("time", "lat", "lon"))
    masks = {b: np.ones((2, 2), dtype=bool) for b in ALTITUDE_BANDS}

    pairs, _, _ = pool_pairs(p_exceed, precip, masks)
    assert {thr for thr, _ in pairs} == set(THRESHOLDS_MM_DAY)
    assert {band for _, band in pairs} == set(ALTITUDE_BANDS)
    assert list(ALTITUDE_BANDS) == ["all", "0-1000", "1000-2000", "2000-3000", ">3000"]


def test_b1_hand_computed_counts():
    """B1: at a chosen cut-off the four contingency cells match a hand count."""
    p = np.array([0.1, 0.4, 0.6, 0.9])
    o = np.array([0, 1, 0, 1])
    c = roc_counts(p, o, probs=np.array([0.5]))
    assert c["hits"][0] == 1
    assert c["false_alarms"][0] == 1
    assert c["misses"][0] == 1
    assert c["correct_negatives"][0] == 1


def test_b2_invariants():
    """B2: per cut-off counts sum to the totals; TPR and FPR are monotonic in p*."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 5000)
    o = (rng.uniform(0, 1, 5000) < 0.3).astype(int)
    probs = np.linspace(0.01, 1, 500)
    c = roc_counts(p, o, probs)
    n_event, n_non = int(o.sum()), int((1 - o).sum())
    assert np.all(c["hits"] + c["misses"] == n_event)
    assert np.all(c["false_alarms"] + c["correct_negatives"] == n_non)
    assert np.all(np.diff(c["hits"]) <= 0) and np.all(np.diff(c["false_alarms"]) <= 0)


def test_b3_extremes():
    """B3: a perfect forecast gives AUC == 1; a forecast independent of obs gives ~0.5."""
    rng = np.random.default_rng(3)
    probs = np.linspace(0.01, 1, 500)
    o = np.r_[np.ones(3000, int), np.zeros(3000, int)]
    p_perfect = np.r_[rng.uniform(0.5, 1, 3000), rng.uniform(0, 0.5, 3000)]
    assert _auc(roc_counts(p_perfect, o, probs)) > 0.99

    p_random = rng.uniform(0, 1, 6000)
    assert 0.45 < _auc(roc_counts(p_random, o, probs)) < 0.55


def test_b4_histogram_equals_brute_force():
    """B4: the searchsorted ROC matches an explicit boolean scan exactly (incl. ties)."""
    rng = np.random.default_rng(4)
    p = np.round(rng.uniform(0, 1, 2000), 2)
    o = (rng.uniform(0, 1, 2000) < 0.4).astype(int)
    probs = np.linspace(0.01, 1, 200)
    fast, slow = roc_counts(p, o, probs), _brute_roc(p, o, probs)
    for k in fast:
        np.testing.assert_array_equal(fast[k], slow[k])


def test_b5_roc_dataset_schema():
    """B5: compute_roc yields the expected dims, count vars, and mm/day threshold coord."""
    rng = np.random.default_rng(5)
    pairs = {(thr, band): (rng.uniform(0, 1, 500), (rng.uniform(0, 1, 500) < 0.3).astype(int))
             for thr in THRESHOLDS_MM_DAY for band in ALTITUDE_BANDS}
    ds = compute_roc(Samples(lead_hours=120, pairs=pairs))
    assert set(ds.data_vars) == {"hits", "false_alarms", "misses", "correct_negatives"}
    assert ds["hits"].dims == ("threshold", "altitude", "prob")
    assert list(ds["threshold"].values) == THRESHOLDS_MM_DAY
    assert ds.sizes["prob"] == 500 and ds.attrs["lead_hours"] == 120


def test_c1_worked_example():
    """C1: contingency h=.15,f=.10,m=.05,s=.20 gives V(0.10)=0.3125 and peaks at alpha=base_rate."""
    roc = _roc_ds([150], [100], [50], [700], probs=np.array([0.5]))
    cl = compute_cost_loss(roc, alphas=np.array([0.10]))
    assert abs(float(cl["value"].sel(threshold=20, altitude="all").values[0]) - 0.3125) < 1e-9
    assert abs(cl["base_rate"].values.item() - 0.20) < 1e-12

    alphas = np.linspace(0.01, 0.99, 99)
    cl2 = compute_cost_loss(roc, alphas=alphas)
    peak = alphas[int(np.argmax(cl2["value"].sel(threshold=20, altitude="all").values))]
    assert abs(peak - 0.20) < 0.02


def test_c2_envelope_is_max_over_cutoffs():
    """C2: the stored value equals the per-cut-off maximum and is >= any single cut-off's value."""
    probs = np.array([0.3, 0.5, 0.7])
    roc = _roc_ds([180, 150, 90], [200, 100, 30], [20, 50, 110], [600, 700, 770], probs)
    alphas = np.linspace(0.05, 0.95, 50)
    env = compute_cost_loss(roc, alphas=alphas)["value"].sel(threshold=20, altitude="all").values
    per_cutoff = np.stack([
        compute_cost_loss(_roc_ds([roc["hits"].values[0, 0, k]],
                                  [roc["false_alarms"].values[0, 0, k]],
                                  [roc["misses"].values[0, 0, k]],
                                  [roc["correct_negatives"].values[0, 0, k]],
                                  np.array([probs[k]])), alphas=alphas)
        ["value"].sel(threshold=20, altitude="all").values
        for k in range(len(probs))])
    np.testing.assert_allclose(env, per_cutoff.max(axis=0), atol=1e-12)
    assert np.all(env + 1e-12 >= per_cutoff.max(axis=0))


def test_c3_perfect_forecast_value_one():
    """C3: a perfect forecast (a cut-off with all hits, no false alarms/misses) gives V==1."""
    probs = np.array([0.2, 0.5, 0.8])
    roc = _roc_ds([200, 200, 120], [400, 0, 0], [0, 0, 80], [400, 800, 800], probs)
    cl = compute_cost_loss(roc, alphas=np.linspace(0.05, 0.95, 50))
    np.testing.assert_allclose(cl["value"].sel(threshold=20, altitude="all").values, 1.0, atol=1e-9)


def test_d1_well_calibrated_tracks_diagonal():
    """D1: when o ~ Bernoulli(p), each bin's observed frequency matches its forecast prob."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200000)
    o = (rng.uniform(0, 1, 200000) < p).astype(int)
    ds = compute_reliability(Samples(120, {(20, "all"): (p, o)}),
                             n_bins=10, thresholds=[20], bands=["all"])
    fp = ds["forecast_probability"].sel(threshold=20, altitude="all").values
    of = ds["observed_frequency"].sel(threshold=20, altitude="all").values
    assert np.nanmax(np.abs(fp - of)) < 0.03


def test_d2_binning_survives_the_point_mass_quantiles_cannot_split():
    """D2: the binning rule must not collapse on calibrated exceedance probabilities.

    Plain quantile binning is the obvious reading of "bin the predicted probabilities",
    and it is what this used to do. It cannot work here: IDR returns a probability of
    essentially zero whenever no training sample exceeded the threshold, so at 50-60 mm/day
    ~90% of cell-days sit on a single value. The 10th through 90th percentiles then all
    land on that value, every one of those samples falls into one bin, and the rest come
    out empty — the published files showed ONE populated bin out of ten at 50 and
    60 mm/day, a reliability diagram consisting of a single dot.

    The rule under test gives the point mass a band of its own and spends the remaining
    nine on quantiles of the probabilities above it, so the informative range is resolved.
    """
    rng = np.random.default_rng(0)
    p = np.concatenate([np.zeros(9000), rng.beta(1.0, 3.0, 1000)])   # 90% point mass
    o = (rng.random(p.size) < p).astype(int)

    edges = derive_reliability_bands(p, 10)
    occupied = np.bincount(assign_bins(p, edges), minlength=10)

    # what plain quantile binning would have produced on the same data
    q = np.quantile(p, np.linspace(0, 1, 11))
    q_occupied = np.bincount(np.clip(np.searchsorted(q[1:-1], p, side="right"), 0, 9),
                             minlength=10)

    assert (occupied > 0).all(), f"empty bands: {occupied}"
    assert (q_occupied > 0).sum() <= 2, (
        "the fixture must reproduce the collapse this rule exists to avoid; "
        f"plain quantile gave {(q_occupied > 0).sum()} populated bins")
    assert (occupied > 0).sum() > (q_occupied > 0).sum()

    # the point mass is isolated in the first band, not smeared across several
    assert occupied[0] == 9000, occupied[0]
    # and the rest are near-equally weighted, which is what quantile binning was for
    rest = occupied[1:]
    assert rest.max() <= 3 * rest.min(), rest


def test_d3_bootstrap_band_brackets_and_widens():
    """D3: the CI brackets the observed frequency and widens as the sample shrinks."""
    rng = np.random.default_rng(2)

    def mean_width(n):
        p = rng.uniform(0.2, 0.4, n)
        o = (rng.uniform(0, 1, n) < 0.3).astype(int)
        ds = compute_reliability(Samples(120, {(20, "all"): (p, o)}),
                                 n_bins=5, thresholds=[20], bands=["all"])
        of = ds["observed_frequency"].sel(threshold=20, altitude="all").values
        lo = ds["obs_freq_lower"].sel(threshold=20, altitude="all").values
        hi = ds["obs_freq_upper"].sel(threshold=20, altitude="all").values
        valid = ~np.isnan(of)
        assert np.all(lo[valid] <= of[valid] + 1e-9) and np.all(of[valid] <= hi[valid] + 1e-9)
        return np.nanmean(hi - lo)

    assert mean_width(50000) < mean_width(500)


def test_d4_reliability_schema():
    """D4: compute_reliability yields the expected vars/dims and bootstrap attrs."""
    rng = np.random.default_rng(3)
    pairs = {(thr, band): (rng.uniform(0, 1, 2000), (rng.uniform(0, 1, 2000) < 0.3).astype(int))
             for thr in THRESHOLDS_MM_DAY for band in ALTITUDE_BANDS}
    ds = compute_reliability(Samples(168, pairs), n_bins=10)
    assert set(ds.data_vars) == {"forecast_probability", "observed_frequency",
                                 "bin_count", "obs_freq_lower", "obs_freq_upper",
                                 "bin_lower", "bin_upper"}
    assert ds["observed_frequency"].dims == ("threshold", "altitude", "bin")
    # edges vary per (threshold, altitude): neither axis is ever summed over, so each
    # pool gets bands derived from itself
    assert ds["bin_lower"].dims == ("threshold", "altitude", "bin")
    assert ds.sizes["bin"] == 10 and ds.attrs["n_bootstraps"] == 50
    assert list(ds["threshold"].values) == THRESHOLDS_MM_DAY
    # the file says how it was binned, so a reader never has to guess
    assert "quantile" in ds.attrs["binning"]


def test_f1_annual_monthly_averages():
    """F1: the scalar reducer yields the 4 UI vars with correct monthly means and dims."""
    import pandas as pd
    time = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    months = time.month.values.astype(float)
    data = np.broadcast_to(months[:, None, None], (len(time), 2, 2)).copy()
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": [0, 1], "lon": [0, 1]})
    ds = annual_monthly_averages(da)
    assert set(ds.data_vars) == {"annual_average", "annual_spatial_average",
                                 "monthly_average", "monthly_spatial_average"}
    np.testing.assert_array_equal(
        np.atleast_1d(ds["monthly_average"].squeeze().values), np.arange(1, 13))
    assert ds["monthly_spatial_average"].dims == ("month", "lat", "lon")
    asa = ds["annual_spatial_average"].squeeze().values
    assert np.allclose(asa, asa.flat[0])


def test_f2_band_variables_and_index_map():
    """F2: band_masks adds per-band means + the int8 index layer; legacy vars unchanged."""
    import pandas as pd
    time = pd.date_range("2022-01-01", "2022-03-31", freq="D")
    rng = np.random.default_rng(0)
    da = xr.DataArray(rng.random((len(time), 2, 2)), dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": [0, 1], "lon": [0, 1]})
    masks = {"0-1000": np.array([[True, False], [False, False]]),
             "1000-2000": np.array([[False, True], [True, False]])}
    ds = annual_monthly_averages(da, band_masks=masks)

    base = annual_monthly_averages(da)          # legacy variables byte-identical
    for v in base.data_vars:
        np.testing.assert_array_equal(ds[v].values, base[v].values)

    assert list(ds["altitude"].values) == ["0-1000", "1000-2000"]
    annual = da.mean("time").values             # single year → time mean
    for band in masks:
        got = float(ds["annual_average_by_altitude"].sel(altitude=band).squeeze())
        assert np.isclose(got, annual[masks[band]].mean())
    assert ds["monthly_average_by_altitude"].sizes["altitude"] == 2
    np.testing.assert_array_equal(
        ds["altitude_band"].values, np.array([[0, 1], [1, -1]], dtype=np.int8))
    assert ds["altitude_band"].attrs["band_labels"] == "0-1000,1000-2000"


def test_f3_band_masks_for_sources():
    """F3: gridded bands come from the DEM, station bands from gauge elevation;
    a missing elevation source degrades to None (legacy-shaped files), never a crash."""
    fld = xr.DataArray(np.zeros((3, 4)), dims=("time", "station"),
                       coords={"station": np.arange(4),
                               "elevation": ("station", [500.0, 1500.0, 2500.0, 3500.0])})
    m = band_masks_for(fld)
    assert list(m) == [b for b, r in ALTITUDE_BANDS.items() if r is not None]
    np.testing.assert_array_equal(m["0-1000"], [True, False, False, False])
    np.testing.assert_array_equal(m[">3000"], [False, False, False, True])
    assert band_masks_for(fld.drop_vars("elevation")) is None

    grid = xr.DataArray(np.zeros((3, 2, 2)), dims=("time", "lat", "lon"),
                        coords={"lat": [9.0, 10.0], "lon": [38.0, 39.0]})
    assert band_masks_for(grid, dem_path="/nonexistent/dem.nc") is None
    with tempfile.TemporaryDirectory() as tmp:
        dem = xr.Dataset(
            {"z_bedrock": (("lat", "lon"),
                           np.array([[500.0, 500.0], [2500.0, 2500.0]]))},
            coords={"lat": [9.0, 10.0], "lon": [38.0, 39.0]})
        dem.to_netcdf(f"{tmp}/dem.nc")
        mg = band_masks_for(grid, dem_path=f"{tmp}/dem.nc")
        np.testing.assert_array_equal(mg["0-1000"], [[True, True], [False, False]])
        np.testing.assert_array_equal(mg["2000-3000"], [[False, False], [True, True]])


def test_g1_model_registry_and_paths():
    """G1: the model token selects the input folder and keys the output path."""
    from verification.diagnostics import MODELS, model_fc_root, output_path
    assert model_fc_root("graphcast").name == "graphcast"
    assert model_fc_root("ifs").name == "ifs"
    gc = output_path("roc", 120, "out", model="graphcast")
    ifs = output_path("roc", 120, "out", model="ifs")
    assert "/graphcast/" in str(gc) and "/ifs/" in str(ifs)
    assert set(MODELS) >= {"graphcast", "ifs"}


def test_j1_all_band_is_union_of_land_bands():
    """J1: the 'all' altitude mask equals the union of the numeric bands and excludes ocean."""
    elev = xr.DataArray(np.array([[-50.0, 500.0], [1500.0, 3500.0]]), dims=("lat", "lon"))
    m = altitude_masks(elev)
    union = m["0-1000"] | m["1000-2000"] | m["2000-3000"] | m[">3000"]
    np.testing.assert_array_equal(m["all"], union)
    assert not m["all"][0, 0]


def _populated_bin_ci(ds):
    """(lower, observed, upper) of the most-populated reliability bin for (20mm, all)."""
    sub = ds.sel(threshold=20, altitude="all")
    b = int(np.argmax(sub["bin_count"].values))
    return (float(sub["obs_freq_lower"].values[b]),
            float(sub["observed_frequency"].values[b]),
            float(sub["obs_freq_upper"].values[b]))


def test_i1_block_bootstrap_widens_ci_under_daily_correlation():
    """I1: day-blocked resampling yields a far wider CI than i.i.d. when obs correlate within days."""
    n_days, per = 20, 100
    wet = np.tile([1, 0], n_days // 2)
    p = np.full(n_days * per, 0.5)
    o = np.repeat(wet, per).astype(int)
    block = np.repeat(np.arange(n_days), per)

    ds_blk = compute_reliability(Samples(120, {(20, "all"): (p, o)}, blocks={(20, "all"): block}),
                                 n_bins=5, thresholds=[20], bands=["all"])
    ds_iid = compute_reliability(Samples(120, {(20, "all"): (p, o)}),
                                 n_bins=5, thresholds=[20], bands=["all"])

    lo_b, of_b, hi_b = _populated_bin_ci(ds_blk)
    lo_i, _, hi_i = _populated_bin_ci(ds_iid)
    assert (hi_b - lo_b) > 4 * (hi_i - lo_i)
    assert lo_b <= of_b <= hi_b


def test_raw_deterministic_gridded_precip_valid_date_pairing():
    """Raw gridded mae: the forecast's init date is advanced by its prediction_timedelta to the
    valid date before pairing with obs, so init day d scores against obs(d+1)."""
    import pandas as pd
    import verification.diagnostics as d

    tmp = Path(tempfile.mkdtemp())
    fc_root = tmp / "fc"; fc_root.mkdir()
    ftime = pd.date_range("2022-01-01", periods=5, freq="D")          # init days 01..05
    precip = np.arange(1, 6)[:, None, None] * np.ones((5, 2, 2))      # init-day value [1..5]
    fc = xr.Dataset(
        {"precipitation": (("prediction_timedelta", "time", "lat", "lon"), precip[None])},
        coords={"prediction_timedelta": [np.timedelta64(30, "h")],    # real lead-1 offset
                "time": ftime, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    fc["precipitation"].attrs["units"] = "mm/hr"
    fc.to_netcdf(fc_root / "24h.nc")
    # obs (mm/hr) [0..5] by date: init d's +30h valid date lands on d+1, where obs == the
    # forecast precip → mae 0. (Pairing init d with obs(d) instead would give mae 1.)
    otime = pd.date_range("2022-01-01", periods=6, freq="D")
    obs = xr.DataArray(np.arange(0, 6)[:, None, None] * np.ones((6, 2, 2)),
                       dims=("time", "lat", "lon"),
                       coords={"time": otime, "lat": [0.0, 1.0], "lon": [0.0, 1.0]})

    orig = d._load_gridded_obs
    d._load_gridded_obs = lambda *a, **k: obs
    try:
        raw = d.compute_deterministic(24, fc_root=fc_root, metric="mae", source="raw")
    finally:
        d._load_gridded_obs = orig

    assert float(np.nanmean(raw["annual_average"].values)) == 0.0
    assert raw["annual_average"].attrs.get("units") == "mm/hr"


def test_raw_deterministic_temperature_variable_aware():
    """Raw temperature_max mae: reads `temperature_2m` (not precipitation), obs is °C with NO
    ÷24, and a float-mismatched obs grid is snapped onto the forecast grid before subtracting."""
    import pandas as pd
    import verification.diagnostics as d

    # routing: the forecast tree is keyed by the TARGET VARIABLE, and max/min are
    # separate trees holding the same in-file name — so the path segment is the only
    # thing that says which reduction a file holds. It carries no product segment:
    # a forecast is scored against every product a country has, not stored per product.
    assert str(d.model_fc_root("graphcast", 2022, "temperature_max", raw=True)).endswith(
        "forecast/gridded/temperature_max/2022/graphcast")
    assert str(d.model_fc_root("graphcast", 2022, "temperature_min", raw=True)).endswith(
        "forecast/gridded/temperature_min/2022/graphcast")
    assert d._VARIABLE["temperature_max"]["raw_fc_var"] == "temperature_2m"
    assert d._VARIABLE["temperature_min"]["raw_fc_var"] == "temperature_2m"

    tmp = Path(tempfile.mkdtemp()); fc_root = tmp / "fc"; fc_root.mkdir()
    ftime = pd.date_range("2022-01-01", periods=5, freq="D")
    lat = np.array([0.0, 1.0]); lon = np.array([0.0, 1.0])
    t2m = np.arange(31, 36)[:, None, None] * np.ones((5, 2, 2), float)   # °C [31..35]
    fc = xr.Dataset(
        {"temperature_2m": (("prediction_timedelta", "time", "lat", "lon"), t2m[None])},
        coords={"prediction_timedelta": [np.timedelta64(30, "h")],
                "time": ftime, "lat": lat, "lon": lon},
    )
    fc["temperature_2m"].attrs["units"] = "°C"
    fc.to_netcdf(fc_root / "24h.nc")
    # obs in °C (NOT ÷24) = [30..35] by date, on a last-digit-perturbed grid to exercise snap.
    otime = pd.date_range("2022-01-01", periods=6, freq="D")
    obs = xr.DataArray(np.arange(30, 36)[:, None, None] * np.ones((6, 2, 2), float),
                       dims=("time", "lat", "lon"),
                       coords={"time": otime, "lat": lat + 1e-14, "lon": lon + 1e-14})

    orig = d._load_gridded_obs
    d._load_gridded_obs = lambda *a, **k: obs
    try:
        raw = d.compute_deterministic(24, fc_root=fc_root, variable="temperature_max",
                                      metric="mae", source="raw")
    finally:
        d._load_gridded_obs = orig

    # t2m(init d)=31+d valid d+1 == obs(d+1)=31+d → mae 0. Finite (not NaN) proves the snap
    # worked despite the 1e-14 mismatch; ~0 (not ~30) proves NO ÷24.
    assert float(np.nanmean(raw["annual_average"].values)) == 0.0
    assert raw["annual_average"].attrs.get("units") == "°C"


def test_raw_deterministic_station_reduces_over_station():
    """Raw station mae: forecast init date advanced to the valid date before pairing with the
    gauge (via the `name` coord), and the result reduced over `station` (not lat/lon)."""
    import pandas as pd
    import verification.diagnostics as d

    tmp = Path(tempfile.mkdtemp())
    fc_root = tmp / "fc"; fc_root.mkdir()
    obs_dir = tmp / "obs"; obs_dir.mkdir()
    ftime = pd.date_range("2022-01-01", periods=5, freq="D")
    st, names = np.arange(2), ["A", "B"]
    precip = np.arange(1, 6)[:, None] * np.ones((5, 2))     # (time, station) [1..5], flat in station
    fc = xr.Dataset(
        {"precipitation": (("prediction_timedelta", "time", "station"), precip[None])},
        coords={"prediction_timedelta": [np.timedelta64(30, "h")], "time": ftime,
                "station": st, "name": ("station", names)},
    )
    fc["precipitation"].attrs["units"] = "mm/hr"
    fc.to_netcdf(fc_root / "24h.nc")
    # gauge obs mm/day over 01..06 = [0..5]×24 (÷24 → [0..5] mm/hr); name coord drives the join.
    otime = pd.date_range("2022-01-01", periods=6, freq="D")
    xr.Dataset(
        {"total_precipitation_rate": (("time", "station"),
                                      np.arange(0, 6)[:, None] * np.ones((6, 2)) * 24.0)},
        coords={"time": otime, "station": st, "name": ("station", names)},
    ).to_netcdf(obs_dir / "stations_2022.nc")

    raw = d.compute_deterministic_station(24, fc_root=fc_root, obs_dir=obs_dir,
                                          metric="mae", source="raw")

    # precip(init d) valid on d+1 == gauge(d+1) → mae 0 (pairing init d with gauge(d) → mae 1).
    assert float(np.nanmean(raw["annual_average"].values)) == 0.0
    assert "station" in raw["annual_spatial_average"].dims


def test_m1_deterministic_metrics_are_raw_only():
    """M1: deterministic metrics (mae/bias/correlation) run for raw tokens only.

    Calibrated ('-idr') is scored by CRPS + threshold curves, so it gets no deterministic
    metrics at all. Stations further drop correlation (gauge series has one spatial dim).
    """
    import verification.run as r
    req = list(r._METRICS)
    assert r.select_deterministic_metrics(req, calibrated=False, is_station=False) == \
        ["mae", "spearmanr", "pearsonr", "bias"]                       # raw gridded: all four
    assert r.select_deterministic_metrics(req, calibrated=False, is_station=True) == \
        ["mae", "bias"]                                                # raw station: no corr
    assert r.select_deterministic_metrics(req, calibrated=True, is_station=False) == []
    assert r.select_deterministic_metrics(req, calibrated=True, is_station=True) == []


def test_k1_domain_from_country_env():
    """K1: VERIFICATION_COUNTRY drives the path domain; unset falls back to 'africa'."""
    import os
    from verification import config
    os.environ.pop("VERIFICATION_COUNTRY", None)
    assert config.country() is None and config.domain() == "africa"
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        assert config.country() == "Senegal"
        assert config.domain() == "senegal"
    finally:
        os.environ.pop("VERIFICATION_COUNTRY", None)


def test_k2_output_path_uses_domain():
    """K2: the default diagnostics root follows VERIFICATION_COUNTRY (else 'africa').

    The domain sits ABOVE ``diagnostics/`` in the layout, so it belongs to the root
    rather than to the facet path. An explicit ``--outdir`` therefore names the
    diagnostics root outright and gets no domain segment appended — which is exactly
    what lets a caller redirect one run's output without it landing in a second
    country-shaped tree.
    """
    import os
    from verification.diagnostics import output_path
    os.environ.pop("VERIFICATION_COUNTRY", None)
    assert "/africa/" in str(output_path("roc", 120))
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        p = str(output_path("roc", 120))
        assert "/senegal/" in p and "/africa/" not in p
        # an explicit outdir IS the root, verbatim
        assert str(output_path("roc", 120, "out")).startswith("out/")
    finally:
        os.environ.pop("VERIFICATION_COUNTRY", None)


def test_k3_region_filter_by_country():
    """K3: unscoped → the region table; a deployment country → it plus its regions."""
    import os
    from ui.utils import regions
    os.environ.pop("VERIFICATION_COUNTRY", None)
    assert regions.available_regions() == list(regions.REGION_COUNTRIES)
    os.environ["VERIFICATION_COUNTRY"] = "senegal"   # lower-case → matched case-insensitively
    try:
        regs = regions.available_regions()
        assert regs[0] == "Senegal"                            # whole country first
        # both levels follow; a name used at both is qualified by its level noun
        assert {"Pikine", "Podor", "Dakar (region)", "Dakar (department)"} <= set(regs)
        assert len(regs) == 1 + 14 + 45
        assert regions.default_region() == "Senegal"
    finally:
        os.environ.pop("VERIFICATION_COUNTRY", None)


def test_k5_deployment_defaults_seed_env_without_override():
    """K5: deployment.toml seeds country/product/data_root at launch; a real env var wins.

    Also pins that there is NO configurable diagnostics root. It used to seed
    ``VERIFICATION_DATA_DIR`` from a ``data_dir`` key, which was pinned to one country's
    tree and seeded into every process — so launching a second country with only
    VERIFICATION_COUNTRY set handed it the first country's path. The root is derived
    from the country now, which removes the hazard rather than documenting it.
    """
    import os
    from pathlib import Path
    from verification import config

    # The committed file pins THIS branch to Senegal.
    assert config.deployment_defaults().get("country") == "Senegal"
    assert "data_dir" not in config.deployment_defaults()

    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "deployment.toml"
        f.write_text('country = "Senegal"\ntruth_product = "era5"\n'
                     'data_root = "data"\n')
        keys = ("VERIFICATION_COUNTRY", "VERIFICATION_TRUTH_PRODUCT",
                "VERIFICATION_DATA_ROOT", "VERIFICATION_DATA_DIR")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            config.apply_deployment_defaults(f)                 # unset → filled from file
            assert os.environ["VERIFICATION_COUNTRY"] == "Senegal"
            assert os.environ["VERIFICATION_TRUTH_PRODUCT"] == "era5"
            dr = os.environ["VERIFICATION_DATA_ROOT"]
            assert Path(dr).is_absolute() and dr.endswith("data")
            # no results root is configurable, so none is seeded
            assert "VERIFICATION_DATA_DIR" not in os.environ

            os.environ["VERIFICATION_COUNTRY"] = "Elsewhere"    # already set → NOT overridden
            config.apply_deployment_defaults(f)
            assert os.environ["VERIFICATION_COUNTRY"] == "Elsewhere"
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _monotone_pair(n_days=59, ny=3, nx=4):          # 59 days = all of Jan + Feb
    """Forecast and an observation that is a monotone but non-linear function of it.

    obs = fc**3 keeps every ordering, so Spearman is exactly 1 along any axis while
    Pearson is strictly below 1: the two metrics cannot coincide by accident.
    """
    rng = np.random.default_rng(0)
    time = np.datetime64("2022-01-01") + np.arange(n_days).astype("timedelta64[D]")
    fc = rng.uniform(0.5, 3.0, size=(1, n_days, ny, nx))
    coords = {"time": time, "lat": np.arange(ny), "lon": np.arange(nx)}
    fc_da = xr.DataArray(fc, dims=("prediction_timedelta", "time", "lat", "lon"),
                         coords={**coords, "prediction_timedelta": [np.timedelta64(0, "D")]})
    obs_da = xr.DataArray(fc[0] ** 3, dims=("time", "lat", "lon"), coords=coords)
    return fc_da, obs_da


def test_c1_spearman_is_rank_correlation_in_every_summary():
    """Every summary of spearmanr is a rank correlation, and differs from pearsonr.

    Regression: the per-day (annual_average) and per-cell (annual_spatial_average)
    summaries once ignored corr_type and silently computed Pearson.
    """
    fc, obs = _monotone_pair()
    s = corr_annual_monthly_averages(fc, obs, corr_type="spearmanr")
    p = corr_annual_monthly_averages(fc, obs, corr_type="pearsonr")
    for v in ("annual_average", "annual_spatial_average", "monthly_average",
              "monthly_spatial_average"):
        assert np.allclose(s[v].values, 1.0), f"spearmanr {v} is not a rank correlation"
        assert np.all(p[v].values < 1.0 - 1e-6), f"pearsonr {v} should be below 1 here"


def test_c2_unknown_correlation_type_raises():
    fc, obs = _monotone_pair()
    try:
        corr_annual_monthly_averages(fc, obs, corr_type="kendall")
    except ValueError:
        return
    raise AssertionError("an unknown corr_type must raise, not fall back to Pearson")


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

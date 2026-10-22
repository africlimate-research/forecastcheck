"""UI-layer tests (Streamlit-dependent). Skip cleanly where streamlit isn't installed.

Runs standalone (``python tests/test_ui.py``) or under pytest. Focused on the pure,
crash-prone logic behind the spatial map + correlation charts — not Streamlit rendering.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_HAS_ST = importlib.util.find_spec("streamlit") is not None


def _station_da(lat_name="station_lat", lon_name="station_lon"):
    """A station-shaped DataArray with configurable lat/lon coord names."""
    n = 5
    return xr.DataArray(
        np.array([0.1, 0.2, np.nan, 0.3, 0.4]),
        dims=("station",),
        coords={"station": np.arange(n),
                lat_name: ("station", [9.0, 10.0, 11.0, 12.0, 13.0]),
                lon_name: ("station", [38.0, 39.0, 40.0, 41.0, 42.0])},
    )


def test_station_keep_mask_handles_station_lonlat():
    """H1: the keep-mask works on files that name coords station_lat/station_lon (not lat/lon)."""
    if not _HAS_ST:
        print("SKIP test_station_keep_mask_handles_station_lonlat (no streamlit)")
        return
    from ui.components.spatial_map import _station_keep_mask
    m = _station_keep_mask(_station_da(), "Africa")
    assert m.shape == (5,) and m.dtype == bool
    assert not m[2]  # the NaN station is excluded


def test_l1_common_leads_is_sorted_intersection():
    """L1: common_leads offers only leads present in EVERY selected series (sorted)."""
    from ui.utils.leads import common_leads
    assert common_leads([[24, 72, 120], [24, 120, 168]]) == [24, 120]
    assert common_leads([[72, 24], [24, 72]]) == [24, 72]        # sorted output
    assert common_leads([[24, 72], [120, 168]]) == []            # no overlap
    assert common_leads([[24, 72]]) == [24, 72]                  # single series
    assert common_leads([]) == []                                # nothing selected
    assert common_leads([[24, 72], []]) == []                    # one empty series → empty


def test_l2_threshold_leads_exclude_metric_gaps():
    """L2: the threshold tab's offered leads drop a lead one metric lacks (the crash scenario).

    reliability lacks 48h while cost_loss has it — 48h must NOT be offered, else the axes
    probe opens a non-existent reliability/48h.nc.
    """
    import pandas as pd
    from ui.utils.leads import common_leads
    sel = pd.DataFrame({
        "forecast": ["graphcast-idr", "graphcast-idr"],
        "metric":   ["reliability", "cost_loss"],
        "leads":    [[24, 72, 120], [24, 48, 72, 120]],
    })
    assert common_leads(sel["leads"]) == [24, 72, 120]  # 48 excluded


def test_h2_mask_region_filters_station_data_by_country():
    """H2: mask_region on a station field keeps only in-country stations (no KeyError 'lat').

    Regression: a country deployment (default region = its country) forces masking on the
    default 2024 station selection, which has station_lat/station_lon and no lat/lon grid.
    """
    if not _HAS_ST:
        print("SKIP test_h2_mask_region_filters_station_data_by_country (no streamlit)")
        return
    from ui.utils.regions import mask_region
    da = xr.DataArray(
        np.arange(3.0),
        dims=("station",),
        coords={"station": np.arange(3),
                "station_lat": ("station", [15.62, 13.77, 13.45]),    # Louga, Tambacounda,
                "station_lon": ("station", [-16.22, -13.67, -16.58])},  # Banjul (Gambia)
    )
    out = mask_region(da, "Senegal")
    assert "station" in out.dims and out.sizes["station"] == 2
    # 'Africa' (no member list) keeps every station unchanged.
    assert mask_region(da, "Africa").sizes["station"] == 3


def test_k4_results_root_follows_country_domain():
    """K4a: ui_results_root() nests under the deployment domain (country lowercased).

    The domain sits ABOVE ``diagnostics/`` in the layout, so the root ends in
    ``diagnostics`` and the domain is its parent.
    """
    if not _HAS_ST:
        print("SKIP test_k4_results_root_follows_country_domain (no streamlit)")
        return
    import os
    from ui.services.data_cache import ui_results_root
    os.environ.pop("VERIFICATION_COUNTRY", None)
    assert ui_results_root().parent.name == "africa"
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        assert ui_results_root().parent.name == "senegal"
    finally:
        os.environ.pop("VERIFICATION_COUNTRY", None)


def test_k4_settings_regions_narrow_to_country():
    """K4b: settings.REGIONS is the deployment country + its admin-1 subregions on (re)import."""
    if not _HAS_ST:
        print("SKIP test_k4_settings_regions_narrow_to_country (no streamlit)")
        return
    import importlib
    import os
    from ui.utils import regions
    from ui.app_config import settings
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        importlib.reload(regions)
        importlib.reload(settings)
        assert settings.REGIONS[0] == "Senegal"           # whole country first
        assert "Dakar (region)" in settings.REGIONS        # regions follow
        assert "Pikine" in settings.REGIONS                # ...and departments
        assert settings.DEFAULT_REGION == "Senegal"        # default = whole country
    finally:
        os.environ.pop("VERIFICATION_COUNTRY", None)
        importlib.reload(regions)
        importlib.reload(settings)


def test_k4c_settings_seasons_narrow_to_country():
    """K4c: the Season selector offers this deployment's calendar and nothing else.

    Offering every slug in the registry would show a forecaster seasons that mean
    nothing for their calendar, and send them looking for files never written.
    """
    if not _HAS_ST:
        print("SKIP test_k4c_settings_seasons_narrow_to_country (no streamlit)")
        return
    import importlib
    import os
    from ui.app_config import settings

    prev = os.environ.get("VERIFICATION_COUNTRY")
    try:
        os.environ["VERIFICATION_COUNTRY"] = "Senegal"
        importlib.reload(settings)
        assert list(settings.SEASONS) == ["Yearly", "NDJF (Nov–Feb)", "MAMJ (Mar–Jun)",
                                          "JASO (Jul–Oct)"], list(settings.SEASONS)
        assert settings.DEFAULT_SEASON == "Yearly"
        # Yearly is implicit everywhere and its months stay None (no filter)
        assert settings.season_months("Yearly") is None

        # a country with no calendar agreed yet gets Yearly only
        os.environ["VERIFICATION_COUNTRY"] = "Atlantis"
        importlib.reload(settings)
        assert list(settings.SEASONS) == ["Yearly"], list(settings.SEASONS)
    finally:
        if prev is None:
            os.environ.pop("VERIFICATION_COUNTRY", None)
        else:
            os.environ["VERIFICATION_COUNTRY"] = prev
        importlib.reload(settings)


def test_k4d_mask_region_refuses_an_unknown_region():
    """K4d: an unresolvable region name raises instead of returning unmasked data.

    Returning the array unmasked is the worst available answer — the page renders and
    shows whole-domain numbers under a regional heading, with nothing to show the mask was
    skipped. Same failure shape as the address-keyed caches that once served one file's
    values under another file's label.
    """
    if not _HAS_ST:
        print("SKIP test_k4d_mask_region_refuses_an_unknown_region (no streamlit)")
        return
    import importlib
    import os
    import numpy as np
    import xarray as xr
    from ui.utils import regions

    prev = os.environ.get("VERIFICATION_COUNTRY")
    try:
        os.environ["VERIFICATION_COUNTRY"] = "Senegal"
        importlib.reload(regions)
        da = xr.DataArray(
            np.ones((3, 3)),
            coords={"lat": [13.0, 14.0, 15.0], "lon": [-16.0, -15.0, -14.0]},
            dims=("lat", "lon"))

        # an id, a lower-cased label, and "Dakar" — ambiguous between region and department
        for unknown in ("Atlantis", "pikine", "dakar.pikine", "Dakar", ""):
            try:
                regions.mask_region(da, unknown)
            except KeyError as exc:
                assert unknown in str(exc) or "unknown region" in str(exc), exc
            else:
                raise AssertionError(
                    f"mask_region({unknown!r}) returned data instead of raising")

        # the full domain still passes through untouched — it genuinely needs no mask
        assert regions.mask_region(da, "Africa") is da
    finally:
        if prev is None:
            os.environ.pop("VERIFICATION_COUNTRY", None)
        else:
            os.environ["VERIFICATION_COUNTRY"] = prev
        importlib.reload(regions)


def _menu_index(rows):
    """A MenuIndex over synthetic discovery rows (list of (forecast, metric) pairs)."""
    import pandas as pd
    from ui.services.menu import MenuIndex
    return MenuIndex.from_index(pd.DataFrame([
        {"forecast": fc, "native_var": "precipitation", "target": "chirts-gridded",
         "metric": m, "year": "2022", "leads": [24, 72], "target_kind": "gridded"}
        for fc, m in rows
    ]))


def test_m1_resolver_raw_crps_falls_back_to_mae():
    """M1: a raw model under CRPS loads its mae files (deterministic CRPS = MAE)."""
    menu = _menu_index([("graphcast", "mae"), ("graphcast-idr", "crps")])
    r = menu.resolve_storage(forecast="graphcast", target="chirts-gridded",
                             metric="crps", year="2022")
    assert r is not None and r["storage_metric"] == "mae"
    assert r["native_var"] == "precipitation" and r["leads"] == [24, 72]


def test_m2_resolver_prefers_real_crps_run():
    """M2: the mae fallback only fires when no real crps run exists for the model."""
    menu = _menu_index([("graphcast", "mae"), ("graphcast", "crps")])
    r = menu.resolve_storage(forecast="graphcast", target="chirts-gridded",
                             metric="crps", year="2022")
    assert r is not None and r["storage_metric"] == "crps"


def test_m3_resolver_blocks_idr_deterministic():
    """M3: -idr tokens never load deterministic metrics, even if stale files exist on disk."""
    menu = _menu_index([("graphcast-idr", "mae"), ("graphcast-idr", "bias"),
                        ("graphcast-idr", "crps")])
    for metric in ("mae", "bias", "pearsonr", "spearmanr"):
        assert menu.resolve_storage(forecast="graphcast-idr", target="chirts-gridded",
                                    metric=metric, year="2022") is None
    # crps stays untouched by the policy
    r = menu.resolve_storage(forecast="graphcast-idr", target="chirts-gridded",
                             metric="crps", year="2022")
    assert r is not None and r["storage_metric"] == "crps"


def test_m4_resolver_passthrough_and_no_data():
    """M4: raw+mae passes through unchanged; raw+crps with no mae either → None."""
    menu = _menu_index([("graphcast", "mae")])
    r = menu.resolve_storage(forecast="graphcast", target="chirts-gridded",
                             metric="mae", year="2022")
    assert r is not None and r["storage_metric"] == "mae"
    assert menu.resolve_storage(forecast="ifs", target="chirts-gridded",
                                metric="crps", year="2022") is None


def _summary_ds(with_bands: bool):
    """A minimal summary Dataset in the on-disk schema (optionally band-aware)."""
    ds = xr.Dataset({
        "annual_average":          ((), 0.5),
        "annual_spatial_average":  (("lat", "lon"), np.full((2, 2), 0.5)),
        "monthly_average":         (("month",), np.array([0.4, 0.6])),
        "monthly_spatial_average": (("month", "lat", "lon"), np.full((2, 2, 2), 0.5)),
    }, coords={"lat": [9.0, 10.0], "lon": [38.0, 39.0], "month": [4, 5]})
    if with_bands:
        ds["annual_average_by_altitude"] = (("altitude",), np.array([0.3, 0.7]))
        ds["monthly_average_by_altitude"] = (
            ("altitude", "month"), np.array([[0.2, 0.4], [0.6, 0.8]]))
        ds["altitude_band"] = (("lat", "lon"),
                               np.array([[0, 1], [1, -1]], dtype=np.int8))
        ds["altitude_band"].attrs["band_labels"] = "0-1000,1000-2000"
        ds = ds.assign_coords(altitude=["0-1000", "1000-2000"])
    return ds


def test_n1_loader_extracts_band_variables():
    """N1: the loader carries the band variables when present and Nones them when not."""
    from ui.services.loader import _to_lead_time_data
    new = _to_lead_time_data(_summary_ds(with_bands=True))
    assert new.annual_average_by_altitude is not None
    assert list(new.annual_average_by_altitude["altitude"].values) == ["0-1000", "1000-2000"]
    assert new.altitude_band.dtype == np.int8
    old = _to_lead_time_data(_summary_ds(with_bands=False))
    assert old.annual_average_by_altitude is None
    assert old.monthly_average_by_altitude is None and old.altitude_band is None
    assert old.annual_average == 0.5          # legacy extraction untouched


def test_n2_band_accessors():
    """N2: band accessors read pipeline scalars, place months truly, NaN when absent."""
    from ui.services.loader import _to_lead_time_data
    from ui.utils.geo import band_annual_mean, band_monthly_means, has_band_data
    ltd = _to_lead_time_data(_summary_ds(with_bands=True))
    assert has_band_data(ltd)
    assert np.isclose(band_annual_mean(ltd, "1000-2000"), 0.7)
    m = band_monthly_means(ltd, "0-1000")
    assert np.isclose(m[3], 0.2) and np.isclose(m[4], 0.4)   # April, May
    assert np.isnan(m[0]) and np.isnan(m[11])                # months not in file
    assert np.isnan(band_annual_mean(ltd, ">3000"))          # unknown band → NaN
    legacy = _to_lead_time_data(_summary_ds(with_bands=False))
    assert not has_band_data(legacy)
    assert np.isnan(band_annual_mean(legacy, "0-1000"))
    assert np.all(np.isnan(band_monthly_means(legacy, "0-1000")))


def test_n3_apply_band_masks_by_index_layer():
    """N3: apply_band NaN-masks outside the band via the file's index layer;
    'all', legacy data, and unknown labels are no-ops."""
    from ui.services.loader import _to_lead_time_data
    from ui.utils.geo import apply_band
    ltd = _to_lead_time_data(_summary_ds(with_bands=True))
    da = ltd.annual_spatial_average
    out = apply_band(da, ltd, "0-1000")          # index layer: [[0, 1], [1, -1]]
    vals = np.asarray(out.values)
    assert np.isfinite(vals[0, 0])
    assert np.isnan(vals[0, 1]) and np.isnan(vals[1, 0]) and np.isnan(vals[1, 1])
    np.testing.assert_array_equal(apply_band(da, ltd, "all").values, da.values)
    np.testing.assert_array_equal(apply_band(da, ltd, ">3000").values, da.values)
    legacy = _to_lead_time_data(_summary_ds(with_bands=False))
    np.testing.assert_array_equal(
        apply_band(legacy.annual_spatial_average, legacy, "0-1000").values,
        legacy.annual_spatial_average.values)


def test_n4_deployment_country_follows_env():
    """N4: deployment_country() names the VERIFICATION_COUNTRY region, or None unscoped."""
    if not _HAS_ST:
        print("SKIP test_n4_deployment_country_follows_env (no streamlit)")
        return
    import os
    from ui.utils.regions import deployment_country
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        assert deployment_country() == "Senegal"
    finally:
        os.environ.pop("VERIFICATION_COUNTRY", None)
    assert deployment_country() is None

def test_season_months_maps_labels_to_calendar_months():
    """S1: season_months resolves each Senegal season to its 1-based months; Yearly → None."""
    from ui.app_config.settings import season_months
    assert season_months("Yearly") is None
    assert season_months("NDJF (Nov–Feb)") == (11, 12, 1, 2)
    assert season_months("MAMJ (Mar–Jun)") == (3, 4, 5, 6)
    assert season_months("JASO (Jul–Oct)") == (7, 8, 9, 10)
    assert season_months("nonexistent") is None  # unknown label degrades to Yearly


def test_season_mean_averages_selected_months():
    """S2: regional_season_mean = nanmean over the season's months; NaN when all missing; annual for None.

    Uses the "Africa" fast path in regional_monthly_means (returns ltd.monthly_average verbatim),
    so the season mean is a pure nanmean over the chosen month indices.
    """
    if not _HAS_ST:
        print("SKIP test_season_mean_averages_selected_months (no streamlit)")
        return
    from types import SimpleNamespace

    from ui.utils.geo import regional_season_mean
    # Jan..Dec = 1..12; monthly_spatial_average is None so the Africa branch uses monthly_average.
    monthly = np.arange(1.0, 13.0)
    ltd = SimpleNamespace(monthly_average=monthly, monthly_spatial_average=None,
                          annual_average=99.0, annual_spatial_average=None)
    assert regional_season_mean(ltd, "Africa", (6, 7, 8)) == 7.0        # mean(6,7,8)
    assert regional_season_mean(ltd, "Africa", (12, 1, 2)) == 5.0        # mean(12,1,2)
    assert regional_season_mean(ltd, "Africa", None) == 99.0             # Yearly → annual_average
    # All-NaN season → NaN, not a crash.
    monthly_gap = monthly.copy()
    monthly_gap[[5, 6, 7]] = np.nan                                     # Jun,Jul,Aug missing
    ltd_gap = SimpleNamespace(monthly_average=monthly_gap, monthly_spatial_average=None,
                              annual_average=99.0, annual_spatial_average=None)
    assert np.isnan(regional_season_mean(ltd_gap, "Africa", (6, 7, 8)))
    # Partially-missing season averages only the present months.
    assert regional_season_mean(ltd_gap, "Africa", (5, 6, 7)) == 5.0     # only May present


def test_station_pair_keeps_common_inregion_finite():
    """C1: _station_pair pairs two models on shared stations, keeping in-region & finite-in-both.

    A and B share stations 0-3; B is NaN at station 1; station 3 sits outside Senegal; station 4
    exists only in A. Only stations 0 and 2 should survive, and diff must equal vals_a - vals_b.
    """
    if not _HAS_ST:
        print("SKIP test_station_pair_keeps_common_inregion_finite (no streamlit)")
        return
    import os

    prev = os.environ.get("VERIFICATION_COUNTRY")
    os.environ["VERIFICATION_COUNTRY"] = "Senegal"
    try:
        import importlib

        from ui.utils import regions as _rg
        importlib.reload(_rg)
        from ui.components.comparison_tab import _station_pair
        # Senegalese coords for stations 0-2 & 4; station 3 at Banjul (in The Gambia).
        lat = [15.62, 13.77, 14.15, 13.45, 12.56]
        lon = [-16.22, -13.67, -16.07, -16.58, -12.18]

        def _da(vals, ids):
            n = len(ids)
            return xr.DataArray(
                np.array(vals, dtype=float), dims=("station",),
                coords={"station": ids,
                        "station_lat": ("station", [lat[i] for i in ids]),
                        "station_lon": ("station", [lon[i] for i in ids])})

        a = _da([1.0, 2.0, 3.0, 4.0, 9.0], [0, 1, 2, 3, 4])
        b = _da([0.5, np.nan, 1.0, 2.0], [0, 1, 2, 3])   # NaN@1, no station 4
        lons, lats, va, vb, diff, names = _station_pair(a, b, "Senegal", zero_best=False)
        # station 1 (NaN in B), 3 (out of region), 4 (not in B) all dropped → {0, 2}
        assert va.tolist() == [1.0, 3.0] and vb.tolist() == [0.5, 1.0]
        assert diff.tolist() == [0.5, 2.0]                          # va - vb
        assert len(lons) == 2 and len(names) == 2
    finally:
        if prev is None:
            os.environ.pop("VERIFICATION_COUNTRY", None)
        else:
            os.environ["VERIFICATION_COUNTRY"] = prev


def test_season_daily_series_shows_only_season_months():
    """C2: _season_daily_series keeps only the season's days and lays them out contiguously.

    A season restricts the correlation daily series to its months. For a wrap-around season
    (NDJF Nov-Feb) the Jan-Feb days are shifted +1 year so Nov-Feb form one contiguous
    block on the axis (the plot shows only the season, not the whole year with a gap).
    """
    if not _HAS_ST:
        print("SKIP test_season_daily_series_shows_only_season_months (no streamlit)")
        return
    from ui.components.time_serie_chart import _season_daily_series
    # One timestamp mid-month for every month of 2022; value == month number.
    times = np.array([f"2022-{m:02d}-15" for m in range(1, 13)], dtype="datetime64[ns]")
    vals = np.arange(1.0, 13.0)

    # Contiguous season (JASO Jul-Oct): only those 4 months, in order.
    t, v = _season_daily_series(times, vals, (7, 8, 9, 10))
    assert v.tolist() == [7.0, 8.0, 9.0, 10.0]
    assert (t == np.sort(t)).all()                             # chronological, no wrap needed

    # Wrap season (NDJF Nov-Feb): kept months {1,2,11,12}; Jan-Feb shifted after Dec so the
    # display order is Nov, Dec, Jan, Feb (values 11,12,1,2) and times are strictly increasing.
    t, v = _season_daily_series(times, vals, (11, 12, 1, 2))
    assert sorted(v.tolist()) == [1.0, 2.0, 11.0, 12.0]        # only NDJF months present
    assert v.tolist() == [11.0, 12.0, 1.0, 2.0]                # Jan-Feb last (next year)
    assert (np.diff(t) > np.timedelta64(0)).all()              # contiguous, strictly increasing

    # Empty when no season day is present (defensive).
    t, v = _season_daily_series(times[:1], vals[:1], (6, 7, 8, 9))
    assert t.size == 0 and v.size == 0


def test_station_detail_season_aware():
    """C3: station-detail helpers honour the season (season mean per lead + season-only months)."""
    if not _HAS_ST:
        print("SKIP test_station_detail_season_aware (no streamlit)")
        return
    from types import SimpleNamespace

    from ui.components.spatial_map import (_station_month_curve,
                                           _station_season_value)
    # Two stations so .squeeze() keeps the station dim; months are NDJF (11,12,1,2).
    msa = xr.DataArray(
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),
        dims=("month", "station"),
        coords={"month": [11, 12, 1, 2], "station": ["S1", "S2"]})
    asa = xr.DataArray([99.0, 88.0], dims=("station",), coords={"station": ["S1", "S2"]})
    ltd = SimpleNamespace(annual_spatial_average=asa, monthly_spatial_average=msa)

    # Season value: annual when no season; nanmean over the season's months otherwise.
    assert _station_season_value(ltd, "S1", None) == 99.0
    assert _station_season_value(ltd, "S1", (11, 12, 1, 2)) == 2.5      # mean(1,2,3,4)
    assert np.isnan(_station_season_value(ltd, "S1", (7, 8, 9, 10)))    # no in-season month

    # Month curve: season order (wrap-aware) when a season is active; calendar order otherwise.
    months, vals = _station_month_curve(ltd, "S1", (11, 12, 1, 2))
    assert months == [11, 12, 1, 2] and vals == [1.0, 2.0, 3.0, 4.0]
    months, vals = _station_month_curve(ltd, "S1", None)
    assert months == [1, 2, 11, 12] and vals == [3.0, 4.0, 1.0, 2.0]    # calendar-sorted


# ── Cache identity — the regional caches must key on the SOURCE FILE ───────
# Regression guard for the bug where ui/utils/geo.py keyed on id(ltd): the
# multipage pages load LeadTimeData per render and drop it, so CPython recycled
# the addresses and a chart was served another file's numbers (a rainfall value
# on a temperature axis, a negative "CRPS" that was really a bias).

def _summary_file(root, forecast, var, target, metric, year, lead, value):
    """Write a summary file at the engine's layout, filled with ``value``.

    ``target`` is a truth-source token (``chirts-gridded``); on disk it is two segments.
    """
    from verification.store import split_truth_source

    product, kind = split_truth_source(target)
    d = (Path(root) / forecast / var / product / kind / metric / str(year)
         / "annual_monthly_averages")
    d.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"annual_average":          ((), value),
         "annual_spatial_average":  (("lat", "lon"), np.full((2, 2), value)),
         "monthly_average":         (("month",), np.full(2, value)),
         "monthly_spatial_average": (("month", "lat", "lon"),
                                     np.full((2, 2, 2), value))},
        coords={"lat": [9.0, 10.0], "lon": [38.0, 39.0], "month": [4, 5]},
    ).to_netcdf(d / f"{lead}h.nc")


def test_o5_an_empty_region_is_nan_not_the_national_value():
    """A region containing no data must read NaN, never the whole-domain figure.

    ``regional_annual_mean`` used to fall back to ``ltd.annual_average`` when the masked
    selection came out empty, so a region with no gauge in it displayed the NATIONAL
    score under that region's name. A blank chart is recoverable; a plausible wrong
    number attributed to a place is not.
    """
    if not _HAS_ST:
        print("SKIP test_o5_an_empty_region_is_nan_not_the_national_value (no streamlit)")
        return
    import numpy as np
    import xarray as xr
    from ui.utils import geo

    class _Slice:
        """Minimal stand-in for LeadTimeData: a source path plus the two aggregates."""
        source = "/somewhere/graphcast-idr/crps/2022/24h.nc"
        annual_average = 42.0
        monthly_average = np.full(12, 42.0)
        annual_spatial_average = xr.DataArray(
            np.full((2, 2), 7.0),
            coords={"lat": [5.0, 6.0], "lon": [35.0, 36.0]}, dims=("lat", "lon"))

    ltd = _Slice()

    # a region whose mask leaves nothing behind
    def _empty(da, region):
        return da.where(False)

    real_mask = geo.mask_region
    geo.mask_region = _empty
    try:
        got = geo.regional_annual_mean(ltd, "Nowhere")
        assert np.isnan(got), f"empty region returned {got!r}, expected NaN"
        assert got != ltd.annual_average
    finally:
        geo.mask_region = real_mask
        geo._MEAN_CACHE.clear()

    # a mask that raises must also not fall back to the national figure
    def _boom(da, region):
        raise KeyError(region)

    geo.mask_region = _boom
    try:
        got = geo.regional_annual_mean(ltd, "Nowhere")
        assert np.isnan(got), f"unmaskable region returned {got!r}, expected NaN"
    finally:
        geo.mask_region = real_mask
        geo._MEAN_CACHE.clear()


def test_o1_cache_key_is_provenance_not_identity():
    """O1: the cache key comes from the source file — distinct files never share one."""
    from ui.services.loader import _to_lead_time_data
    from ui.utils.geo import _cache_key
    base = "/d/senegal/gc/{}/chirts/crps/2022/annual_monthly_averages/24h.nc"
    rain = _to_lead_time_data(_summary_ds(False), source=base.format("precipitation"))
    temp = _to_lead_time_data(_summary_ds(False), source=base.format("temperature_max"))
    same = _to_lead_time_data(_summary_ds(False), source=rain.source)
    # A different results root with the same run must not collide either.
    other_root = _to_lead_time_data(
        _summary_ds(False), source=rain.source.replace("/d/", "/other/"))
    assert _cache_key(rain, "Senegal") != _cache_key(other_root, "Senegal")
    # Different variable (or lead, model, metric, year) ⇒ different key…
    assert _cache_key(rain, "Senegal") != _cache_key(temp, "Senegal")
    # …and a different region never collides either.
    assert _cache_key(rain, "Senegal") != _cache_key(rain, "Pikine")
    # Same file ⇒ same key, even though they are distinct objects.
    assert _cache_key(rain, "Senegal") == _cache_key(same, "Senegal")
    assert rain is not same
    # No provenance ⇒ no key (so the caller computes instead of caching).
    assert _cache_key(_to_lead_time_data(_summary_ds(False)), "Senegal") is None


def test_o2_provenance_less_slice_is_never_cached():
    """O2: a slice with no source is computed fresh and leaves the caches untouched."""
    from ui.services.loader import _to_lead_time_data
    from ui.utils import geo
    ltd = _to_lead_time_data(_summary_ds(False))          # source=None
    before = (len(geo._MEAN_CACHE), len(geo._MONTHLY_CACHE), len(geo._GRID_CACHE))
    assert geo.regional_annual_mean(ltd, "Africa") == 0.5
    assert not np.isnan(geo.regional_monthly_means(ltd, "Africa")[3])
    geo.annual_spatial_grid(ltd, "Africa")
    assert (len(geo._MEAN_CACHE), len(geo._MONTHLY_CACHE),
            len(geo._GRID_CACHE)) == before
    assert not geo.is_region_warm(ltd, "Africa")          # never warm without a key


def test_o3_address_keyed_entries_are_never_read():
    """O3: an entry left under an id()-based key must never be served.

    This is the deterministic form of the bug. Relying on CPython to actually
    recycle an address is flaky (it may or may not within one test), so instead we
    plant the exact entry the old implementation would have written — keyed on the
    live object's address — and require the value to come from the file anyway.
    Under the old code that plant is a guaranteed cache hit.
    """
    from ui.services.loader import _to_lead_time_data
    from ui.utils import geo
    ltd = _to_lead_time_data(
        _summary_ds(False),                                    # every value 0.5
        source="/o3/sen/gc/temperature_max/chirts/crps/2022/"
               "annual_monthly_averages/24h.nc")

    geo._MEAN_CACHE[(id(ltd), "Africa")] = -2.9                # a bias value
    geo._MONTHLY_CACHE[(id(ltd), "Africa")] = np.full(12, -2.9)
    try:
        assert geo.regional_annual_mean(ltd, "Africa") == 0.5
        assert geo.regional_season_mean(ltd, "Africa", None) == 0.5
        # _summary_ds carries months 4-5 with values 0.4 / 0.6.
        monthly = geo.regional_monthly_means(ltd, "Africa")
        assert monthly[3] == 0.4 and monthly[4] == 0.6
        assert np.isnan(monthly[0])
    finally:
        geo._MEAN_CACHE.pop((id(ltd), "Africa"), None)
        geo._MONTHLY_CACHE.pop((id(ltd), "Africa"), None)


def test_o4_reloaded_slices_keep_their_own_values():
    """O4: end-to-end — load, drop every reference, reload; no value crosses over.

    Emulates successive Streamlit renders over a real results tree, which is how the
    corruption surfaced (the same selection plotting a different curve each render).
    """
    import gc
    import tempfile

    from ui.services.loader import load_run
    from ui.utils.geo import regional_annual_mean, regional_season_mean

    # Distinct value per (variable, lead) so any crosstalk is unmistakable.
    files = {("precipitation", 24): 0.061, ("precipitation", 168): 0.099,
             ("temperature_max", 24): 1.220, ("temperature_max", 168): 3.400}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "senegal"
        for (var, lead), val in files.items():
            _summary_file(root, "graphcast", var, "chirts-gridded", "crps", 2022, lead, val)

        for _render in range(6):
            for var in ("precipitation", "temperature_max"):
                lt_map = load_run("graphcast", var, "chirts-gridded", "crps", "2022",
                                  [24, 168], root=root)
                assert set(lt_map) == {24, 168}
                for lead, ltd in lt_map.items():
                    want = files[(var, lead)]
                    assert regional_annual_mean(ltd, "Africa") == want, (
                        f"{var} {lead}h: got another slice's value")
                    # The Yearly path the Miscalibration chart actually calls.
                    assert regional_season_mean(ltd, "Africa", None) == want
                del lt_map                   # the render ends; addresses freed
                gc.collect()


def test_u1_ui_root_equals_the_pipeline_write_root_for_every_product():
    """U1: the dashboard reads exactly where verification.run writes.

    ui_results_root() used to compose VERIFICATION_DATA_DIR / config.domain() by hand,
    which put the country in the wrong place relative to diagnostics/ — a deployment
    would read one tree while the pipeline wrote another. It fails SILENTLY: discovery
    returns an empty index and the app just offers nothing, so it needs a test rather
    than a code review. Now swept over country x product x target, since a product is
    one more segment both sides have to agree on.
    """
    if not _HAS_ST:
        print("SKIP test_u1 (streamlit not installed)")
        return
    import os

    from ui.services.data_cache import ui_results_root
    from verification.diagnostics import output_path

    keys = ("VERIFICATION_COUNTRY", "VERIFICATION_TRUTH_PRODUCT",
            "VERIFICATION_DATA_ROOT", "VERIFICATION_DATA_DIR")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for country, product in (("Senegal", "chirts"), ("Senegal", "era5")):
            os.environ.update({"VERIFICATION_COUNTRY": country,
                               "VERIFICATION_TRUTH_PRODUCT": product,
                               "VERIFICATION_DATA_ROOT": "/d",
                               "VERIFICATION_DATA_DIR": "/d/diagnostics"})
            for target in ("gridded", "station"):
                written = output_path("roc", 24, None, model="graphcast-idr",
                                      target=target, native_var="precipitation")
                root = ui_results_root()
                assert root in Path(written).parents, (
                    f"{country}/{product}/{target}: UI reads {root}, "
                    f"pipeline wrote {written}")
                assert f"/{product}/{target}/" in str(written), written
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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

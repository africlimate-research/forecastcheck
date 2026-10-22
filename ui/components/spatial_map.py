# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import threading
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from ui.app_config.settings import (CORRELATION_METRICS, MAP_COASTLINE,
                                    MAP_CORRELATION_SCALE, MAP_COUNTRY,
                                    MAP_LAND, MAP_OCEAN, MAP_SEQUENTIAL_SCALE,
                                    ZERO_IS_BEST_METRICS, higher_is_better,
                                    plotly_diverging_scale,
                                    plotly_signed_scale)
from ui.backend.data_loader.climatology_loader import ClimData
from ui.components import stat_blocks
from ui.services.loader import LeadTimeData
from ui.utils.geo import (annual_spatial_grid, apply_band,
                                       regional_monthly_means, subset_region)
from ui.utils.leads import common_leads
from ui.utils.regions import mask_region, region_boundary, stations_in_region

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _is_gridded(da: "xr.DataArray") -> bool:
    """True if ``da`` is a 2-D lat/lon field a heatmap can render.

    False for spatially-reduced metrics (acc / pearson_* / spearman_* collapse
    the map to a scalar) and station targets (a ``station`` dim, no lat/lon) —
    neither has a per-pixel field to draw.
    """
    has_lat = "latitude" in da.coords or "lat" in da.coords
    has_lon = "longitude" in da.coords or "lon" in da.coords
    return da.ndim >= 2 and has_lat and has_lon


def _is_station(da: "xr.DataArray") -> bool:
    """True if ``da`` is a per-station field (point map, not a grid)."""
    return "station" in da.dims


def _station_lonlat(da: "xr.DataArray") -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) arrays for a station field, tolerant of coord naming.

    The IDR path writes ``lat``/``lon``; the raw/deterministic path writes
    ``station_lat``/``station_lon`` — accept either so the map never crashes on one.
    """
    lat_name = "station_lat" if "station_lat" in da.coords else "lat"
    lon_name = "station_lon" if "station_lon" in da.coords else "lon"
    return (np.asarray(da[lon_name].values, dtype=float),
            np.asarray(da[lat_name].values, dtype=float))


def _station_keep_mask(da: "xr.DataArray", region: str) -> np.ndarray:
    """Boolean mask of stations that are in-region AND have a finite value.

    The single source of truth for "which stations are shown" — the point map
    and the detail selector use it, so a clicked marker's index lines up with
    the selector's options.
    """
    da = da.squeeze()
    lons, lats = _station_lonlat(da)
    vals = np.asarray(da.values, dtype=float)
    return stations_in_region(lons, lats, region) & np.isfinite(vals)


def _station_choices(da: "xr.DataArray", region: str) -> list[tuple]:
    """``(station_id, name)`` for each shown station, in marker/plot order."""
    da = da.squeeze()
    keep = _station_keep_mask(da, region)
    sids = np.asarray(da["station"].values)[keep]
    names = (np.asarray(da["name"].values)[keep]
             if "name" in da.coords else sids.astype(str))
    return [(sids[i].item() if hasattr(sids[i], "item") else sids[i], str(names[i]))
            for i in range(int(keep.sum()))]


def _station_season_value(ltd, station_id, season_months) -> float:
    """One station's metric at one lead: the season mean, or the annual value if no season.

    The single-station analogue of :func:`ui.utils.geo.regional_season_mean` — nanmean of the
    station's monthly values over the season's months. NaN-safe (a lead with no in-season data
    returns NaN, dropped by the caller's finite check).
    """
    if not season_months:
        return float(ltd.annual_spatial_average.squeeze().sel(station=station_id))
    msa = ltd.monthly_spatial_average
    if msa is None:
        return float("nan")
    sel = msa.squeeze().sel(station=station_id)
    months = np.asarray(sel["month"].values).astype(int)
    vals = np.asarray(sel.values, dtype=float)
    v = vals[np.isin(months, list(season_months))]
    return float(np.nanmean(v)) if v.size and not np.all(np.isnan(v)) else float("nan")


def _station_lead_curve(lt_map: dict, station_id, season_months=None) -> tuple[list, list]:
    """One station's metric across leads: (sorted leads, values), NaN-skipped.

    Uses the station's season mean per lead when ``season_months`` is given, else its annual value.
    """
    leads, vals = [], []
    for lead in sorted(lt_map):
        ltd = lt_map[lead]
        if ltd is None:
            continue
        try:
            v = _station_season_value(ltd, station_id, season_months)
        except Exception:
            continue
        if np.isfinite(v):
            leads.append(lead)
            vals.append(v)
    return leads, vals


def _station_month_curve(ltd: "LeadTimeData", station_id, season_months=None) -> tuple[list, list]:
    """One station's metric by calendar month, from a given lead's ltd.

    With ``season_months`` the curve is restricted to those months, in season order (wrap-aware,
    so NDJF → Nov, Dec, Jan, Feb); otherwise all available months are returned in calendar order.
    """
    msa = ltd.monthly_spatial_average
    if msa is None:
        return [], []
    sel = msa.squeeze().sel(station=station_id)
    mv = {int(m): float(x)
          for m, x in zip(np.asarray(sel["month"].values), np.asarray(sel.values, dtype=float))}
    order = list(season_months) if season_months else sorted(mv)
    months = [m for m in order if m in mv]
    return months, [mv[m] for m in months]


# ── Colorscales ────────────────────────────────────────────────────────────
# Read from ui.app_config.settings so a station map and the gridded map of the
# same field are coloured identically — the diverging scale in particular used
# to be PuOr here against the brand green/amber ramp on the cartopy maps.
_SEQUENTIAL_COLORSCALE  = MAP_SEQUENTIAL_SCALE    # error metrics: pale = low, red = high
_DIVERGING_COLORSCALE   = plotly_diverging_scale()  # Δ vs a baseline: green = better
_SIGNED_COLORSCALE      = plotly_signed_scale()     # bias: 0 ideal, neither side good
_CORRELATION_COLORSCALE = MAP_CORRELATION_SCALE   # red = low/bad, green = high/good

# ── Session-state keys ─────────────────────────────────────────────────────
_MODEL_IDX_KEY = "sm_model_idx"
_LT_IDX_KEY    = "sm_lt_idx"

# ── Module-level caches — keyed by scalar tuples, thread-safe reads ────────
_FIGURE_CACHE:      dict[tuple, go.Figure]          = {}
_STATS_CACHE:       dict[tuple, dict | None]        = {}
_CLIM_INTERP_CACHE: dict[tuple, np.ndarray | None]  = {}
_MATCHED_CLIM_CACHE: dict[tuple, "xr.DataArray | None"] = {}
_PREWARM_STARTED:   set[tuple]                      = set()


def _available_months(ltd: "LeadTimeData", region: str) -> tuple[int, ...]:
    """Return the calendar months (1-12) for which ltd actually has data."""
    try:
        monthly = regional_monthly_means(ltd, region)
    except Exception:
        return tuple(range(1, 13))
    if monthly is None:
        return tuple(range(1, 13))
    return tuple(
        i + 1 for i in range(min(12, len(monthly))) if not np.isnan(monthly[i])
    )


def _season_field(
    ltd:    "LeadTimeData",
    model:  str,
    lt:     int,
    months: "tuple[int, ...]",
    load_season=None,
) -> "xr.DataArray | None":
    """Season-mean spatial field (grid or station) for one model + lead, or None.

    Prefers the on-demand ``load_season`` reload (the gridded monthly field is
    freed after precaching); falls back to averaging the in-memory
    ``monthly_spatial_average`` over the season's *present* months (used for
    station targets, whose monthly field is kept resident).
    """
    raw = load_season(model, lt) if load_season is not None else None
    if raw is None and ltd.monthly_spatial_average is not None:
        da = ltd.monthly_spatial_average.squeeze()
        month_dim = next((d for d in ("month", "time") if d in da.dims), None)
        if month_dim is not None:
            present = [m for m in da[month_dim].values if int(m) in set(months)]
            if present:
                raw = da.sel({month_dim: present}).mean(month_dim, skipna=True)
    return raw


def _matched_clim_spatial(
    ltd:    "LeadTimeData",
    clim:   "ClimData | None",
    region: str,
    season_months: "tuple[int, ...] | None" = None,
) -> "xr.DataArray | None":
    """Climatology spatial mean restricted to ltd's available (∩ season) months.

    Partial-year models (e.g. IFS+IDR 2024 has months April-Dec only)
    have their `annual_spatial_average` aggregated over those months,
    but `clim.annual_spatial_average` aggregates all 12 months — so the
    out-of-the-box diff mixes different temporal subsets. This helper
    builds an apples-to-apples climatology by averaging
    `clim.monthly_spatial_average` over exactly the months ltd covers
    (further narrowed to ``season_months`` when a season is active),
    then masking it to the same region.

    Cached by (target_months, id(clim), region) so all lead times of
    one model (which share the same coverage) reuse a single matched
    array.
    """
    if clim is None:
        return None

    available_months = _available_months(ltd, region)
    if season_months:
        available_months = tuple(m for m in available_months if m in set(season_months))
    if not available_months:
        return None

    if len(available_months) == 12:
        # Full year — identical to clim.annual_spatial_average. Fast path.
        return subset_region(clim.annual_spatial_average, region)

    cache_key = (available_months, id(clim), region)
    if cache_key in _MATCHED_CLIM_CACHE:
        return _MATCHED_CLIM_CACHE[cache_key]

    clim_monthly_da = clim.monthly_spatial_average
    if clim_monthly_da is None:
        result: "xr.DataArray | None" = subset_region(clim.annual_spatial_average, region)
    else:
        try:
            month_dim = next(d for d in ("month", "time") if d in clim_monthly_da.dims)
            clim_months = list(clim_monthly_da[month_dim].values)
            keep = [int(m) for m in available_months if int(m) in (int(c) for c in clim_months)]
            if not keep:
                result = subset_region(clim.annual_spatial_average, region)
            else:
                matched = clim_monthly_da.sel({month_dim: keep})
                matched_mean = matched.mean(dim=month_dim)
                result = subset_region(matched_mean, region)
        except Exception as exc:
            print(f"[spatial_map] matched_clim fallback ({available_months}): {exc}")
            result = subset_region(clim.annual_spatial_average, region)

    _MATCHED_CLIM_CACHE[cache_key] = result
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════

@st.fragment
def render(
    model_results: dict[str, dict[int, LeadTimeData]],
    metric_key:    str,
    region:        str                     = "Africa",
    subtract_clim: bool                    = False,
    clim:          "ClimData | None"       = None,
    unit:          str                     = "",
    load_season=None,
    season_months: "tuple[int, ...] | None" = None,
    altitude_band: str                     = "all",
) -> None:
    """
    Renders the spatial map panel with model tabs and lead-time navigator.

    Parameters
    ----------
    model_results : {model_name: {lead_time: LeadTimeData}}
    metric_key    : internal metric key, e.g. "mae"
    region        : region name — used to subset the spatial grid at display time
    subtract_clim : whether to show improvement vs climatology
    clim          : full ClimData object. The fragment derives a
                    model-specific matched-month climatology from
                    `clim.monthly_spatial_average` so partial-year
                    models (e.g. months 4–12) are compared against
                    climatology averaged over the same months only.
    altitude_band : NaN-masks cells/stations outside this band (via the file's
                    altitude_band layer); "all" shows the full domain. The
                    caller disables subtract_clim when a band is selected.
    """
    models = [m for m, lt_map in model_results.items() if lt_map]

    if not models:
        _render_empty_state("No model data loaded yet")
        return

    # Navigator offers only leads every selected model has loaded, so switching model
    # tabs never lands on a lead that model lacks.
    leads = common_leads(model_results[m].keys() for m in models)
    if not leads:
        _render_empty_state("No lead time is available for every selected model")
        return

    # Stable tab / lead-time index
    st.session_state.setdefault(_MODEL_IDX_KEY, 0)
    st.session_state.setdefault(_LT_IDX_KEY,    0)
    model_idx = min(st.session_state[_MODEL_IDX_KEY], len(models) - 1)
    lt_idx    = min(st.session_state[_LT_IDX_KEY],    len(leads) - 1)

    _render_header(metric_key, subtract_clim, altitude_band)
    model_idx = _render_model_tabs(models, model_idx)
    model     = models[model_idx]
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-family:Inter,sans-serif;font-size:0.65rem;font-weight:700;"
        "letter-spacing:0.1em;text-transform:uppercase;color:#4B5563;margin-bottom:2px;'>"
        "⏱ Lead Time</div>",
        unsafe_allow_html=True,
    )
    lt_idx = _render_lead_navigator(leads, lt_idx)
    lt     = leads[lt_idx]
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    lt_map = model_results[model]
    ltd    = lt_map.get(lt)

    if ltd is None:
        _render_empty_state(f"{model} · {lt}h — not loaded")
        return

    # ── Two-pass spinner ──────────────────────────────────────────────────────
    # @st.fragment batches ALL deltas and flushes them when the run completes,
    # so a spinner that both starts and ends in the same run is never painted.
    # Fix: pass 1 renders a loading placeholder then calls st.rerun(scope=
    # "fragment"); the browser paints the placeholder while pass 2 computes.
    #
    # _skip_sm_spinner is set by programmatic st.rerun() calls (after PDF
    # generation or after download button click) so the extra rerun never
    # triggers the two-pass spinner and accidentally resets widget state.
    _skip_spinner   = st.session_state.pop("_skip_sm_spinner", False)
    _has_clim_flag  = subtract_clim and clim is not None
    _loading_target = (model, lt, metric_key, region, _has_clim_flag,
                       altitude_band, season_months)
    _compute_pass   = st.session_state.get("_sm_loading") == _loading_target
    _likely_cached  = any(
        k[0] == model and k[1] == lt and k[2] == metric_key
        and k[3] == region and k[4] == _has_clim_flag and k[7] == season_months
        for k in _FIGURE_CACHE
    )
    # Station targets render cheaply (no grid masking) and their map is a click-to-select
    # widget: routing them through the two-pass spinner would replace the chart with a
    # placeholder on pass 1, so Streamlit discards the click selection and the detail never
    # updates on click. Skip the spinner for stations — render directly so the click survives.
    _station_target = _is_station(ltd.annual_spatial_average)

    if (not _compute_pass and not _likely_cached and not _skip_spinner
            and not _station_target):
        # Pass 1: show loading placeholder, then trigger the compute pass.
        # Snapshot widget state for widgets that aren't rendered in this
        # pass (the test-choice radio sits after the spinner block in the
        # function body, so it's never reached in Pass 1). Streamlit
        # garbage-collects session-state keys for widgets that don't appear
        # in a run, so without this snapshot the radio silently falls back
        # to its default ("T-Test") in Pass 2 — visible as "Both selected
        # but only one card showing" on the first model switch.
        _preserved_keys = ("sm_test_choice",)
        _preserved = {
            k: st.session_state[k]
            for k in _preserved_keys
            if k in st.session_state
        }
        st.session_state["_sm_loading"] = _loading_target
        if _preserved:
            st.session_state["_sm_preserved"] = _preserved
        col_map, col_stats = st.columns([3, 1], gap="small")
        with col_map:
            _render_loading_placeholder(model, lt)
        try:
            st.rerun(scope="fragment")
        except Exception:
            st.rerun()

    if _compute_pass:
        # Re-seed widget state that Pass 1 didn't render, before any widget
        # below reads it. setdefault only fills if Streamlit cleared it.
        for k, v in st.session_state.pop("_sm_preserved", {}).items():
            st.session_state.setdefault(k, v)
        del st.session_state["_sm_loading"]

    # ── Computation (pass 2 or cache hit) ─────────────────────────────────────
    # Retrieve the masked annual grid — result is cached in _GRID_CACHE so the
    # da.where(keep).isel() chain only runs once per (ltd, region) combination.
    # Some selections have no per-pixel field: correlation metrics
    # (acc / pearson_* / spearman_*) reduce the map to a scalar, and station
    # targets carry a `station` dim, not lat/lon. Show a notice instead of
    # crashing the heatmap (the other tabs still work for these). Checked on
    # the raw field, before masking — region masking assumes lat/lon and would
    # itself raise on a non-gridded array.
    # Station targets are points, not a grid → a scatter point map, sized by
    # valid-day count and colored by the metric. (Bypasses annual_spatial_grid,
    # which assumes lat/lon dims.)
    if _is_station(ltd.annual_spatial_average):
        if season_months:
            asa = _season_field(ltd, model, lt, season_months, load_season)
            if asa is None:
                _render_empty_state(f"{model} · {lt}h — no station data for this season")
                return
            asa = asa.squeeze()
        else:
            asa = ltd.annual_spatial_average.squeeze()
        # Band filter: out-of-band stations become NaN, and the shared keep-mask
        # (in-region & finite) drops them from the map, the stats card, and the
        # detail selector alike.
        asa = apply_band(asa, ltd, altitude_band)
        choices = _station_choices(asa, region)
        fig = _build_station_figure(asa, model, metric_key, lt, region, unit)

        # The map's marker set changes with season/band, so the click-selection widget must
        # be distinct per (season, band) — a key that omitted them kept a stale selection and
        # the detail stopped updating on click when the season changed.
        _season_tag = "-".join(str(m) for m in season_months) if season_months else "all"

        col_map, col_stats = st.columns([3, 1], gap="small")
        event = None
        with col_map:
            if fig is not None:
                # on_select → clicking a marker returns its point index so the
                # detail below can drill into that exact station.
                event = st.plotly_chart(
                    fig, width="stretch", on_select="rerun", selection_mode="points",
                    key=f"sm_stationpts_{model}_{lt}_{region}_{altitude_band}_{_season_tag}",
                    config={"displayModeBar": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "scrollZoom": False})
            else:
                _render_empty_state(f"No stations inside {region}.")
        with col_stats:
            keep = _station_keep_mask(asa, region)
            stats = _compute_stats(np.asarray(asa.values, float)[keep], None)
            _render_stats_card(stats, metric_key, False)

        if choices:
            _render_station_detail(choices, model_results, lt, metric_key, unit, event,
                                   season_months)
        return

    if not _is_gridded(ltd.annual_spatial_average):
        _render_empty_state(f"{metric_key.upper()} has no spatial map for this selection.")
        return

    # Aggregation period is driven by the global Season selector: Yearly uses the
    # annual field; a season averages that season's months (reloaded on demand, as
    # the gridded monthly field is freed from memory after precaching). Both paths
    # apply the altitude-band mask, so season × band compose.
    if not season_months:
        grid = annual_spatial_grid(ltd, region, altitude_band)
    else:
        raw = _season_field(ltd, model, lt, season_months, load_season)
        if raw is None:
            _render_empty_state(f"{model} · {lt}h — no data for this season")
            return
        grid = mask_region(apply_band(raw.squeeze(), ltd, altitude_band), region)

    # Climatology diff — matched to the model's (∩ season) months, for both views.
    clim_interp: np.ndarray | None = None
    if subtract_clim and clim is not None:
        matched = _matched_clim_spatial(ltd, clim, region, season_months)
        if matched is not None:
            clim_interp = _interpolate_clim(grid, matched)

    has_clim  = subtract_clim and clim_interp is not None
    cache_key = _make_fig_key(model, lt, metric_key, region, has_clim, _data_hash(grid),
                              unit, season_months)

    fig = _FIGURE_CACHE.get(cache_key)
    if fig is None:
        fig = _build_heatmap_figure(grid, model, metric_key, lt,
                                    clim_interp if has_clim else None, unit, region)
        if fig is not None:
            _FIGURE_CACHE[cache_key] = fig

    stats_key = cache_key
    stats = _STATS_CACHE.get(stats_key)
    if stats is None and stats_key not in _STATS_CACHE:
        stats = _compute_stats(grid.values, clim_interp if has_clim else None)
        _STATS_CACHE[stats_key] = stats

    # Background pre-warm for all lead times × both clim states.
    # Include id(lt_map) so different years (different lt_map objects) each get warmed.
    pw_key = (model, metric_key, region, has_clim, id(lt_map))
    if pw_key not in _PREWARM_STARTED:
        _PREWARM_STARTED.add(pw_key)
        threading.Thread(
            target=_prewarm_model,
            args=(model, lt_map, metric_key, region, clim if subtract_clim else None, unit),
            daemon=True,
        ).start()

    col_map, col_stats = st.columns([3, 1], gap="small")
    with col_map:
        if fig is not None:
            st.plotly_chart(fig, width="stretch", config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "scrollZoom": False,
            })
        else:
            _render_empty_state(f"{model} · {lt}h — no grid data")
    with col_stats:
        _render_stats_card(stats, metric_key, has_clim)

    if has_clim and clim_interp is not None:
        stat_blocks.render_section_header(
            f"Statistical Insights · {model} vs Climatology · {lt}h lead"
        )
        col_sel, _ = st.columns([4, 3])
        with col_sel:
            _test_options = ["T-Test", "Wilcoxon + Sign", "Both"]
            # Same persistence pattern as the lead-time slider: a
            # non-widget `_last_*` backup survives Streamlit's widget
            # garbage collection (including the two-pass spinner's
            # interrupted runs), so "Both" doesn't silently revert to
            # "T-Test" and lose the Wilcoxon card.
            if st.session_state.get("sm_test_choice") not in _test_options:
                _last_choice = st.session_state.get("_last_sm_test_choice")
                if _last_choice in _test_options:
                    st.session_state["sm_test_choice"] = _last_choice
            test_choice = st.radio(
                "Statistical test",
                options=_test_options,
                horizontal=True,
                key="sm_test_choice",
                label_visibility="collapsed",
            )
            st.session_state["_last_sm_test_choice"] = test_choice
        _test = {"T-Test": "ttest", "Wilcoxon + Sign": "wilcoxon", "Both": "both"}[test_choice]
        stat_blocks.render_paired_ttest(
            grid.values, clim_interp,
            model, "Climatology",
            metric_key, unit, accent="#213b7d", test=_test,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-renderers (Streamlit calls)
# ═══════════════════════════════════════════════════════════════════════════

def _render_header(metric_key: str, subtract_clim: bool, altitude_band: str = "all") -> None:
    clim_badge = (
        "<span style='background:rgba(33,59,125,0.09);color:#213b7d;"
        "border:1px solid rgba(33,59,125,0.25);border-radius:99px;"
        "padding:2px 9px;font-family:Inter,sans-serif;font-size:0.62rem;"
        "font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'>"
        "vs Climatology</span>"
    ) if subtract_clim else ""
    if altitude_band != "all":
        clim_badge += (
            "<span style='background:rgba(242,176,41,0.12);color:#92580a;"
            "border:1px solid rgba(242,176,41,0.35);border-radius:99px;"
            "padding:2px 9px;font-family:Inter,sans-serif;font-size:0.62rem;"
            "font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'>"
            f"🏔️ {altitude_band} m</span>"
        )

    st.markdown(f"""
        <div style='border-left:4px solid #213b7d;padding:10px 16px 10px 14px;
            background:#ffffff;border-radius:0 6px 6px 0;margin-bottom:12px;
            box-shadow:0 1px 6px rgba(0,0,0,0.05);'>
          <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
            <span style='font-size:19px;'>🗺️</span>
            <span style='font-family:Inter,sans-serif;font-size:15px;
                font-weight:700;color:#171d1a;'>Spatial Distribution</span>
            <span style='background:rgba(33,59,125,0.09);color:#213b7d;
                border:1px solid rgba(33,59,125,0.25);border-radius:99px;
                padding:2px 9px;font-family:Inter,sans-serif;font-size:0.62rem;
                font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'>
                {metric_key.upper()}</span>
            {clim_badge}
          </div>
          <div style='font-family:Inter,sans-serif;font-size:11px;color:#6B7280;
              margin-top:3px;padding-left:27px;'>
              Geographic distribution of forecast error by region</div>
        </div>
    """, unsafe_allow_html=True)


def _render_model_tabs(models: list[str], current_idx: int) -> int:
    st.markdown("""
        <style>
        div[data-testid="stPills"] button[aria-selected="true"] {
            background-color: #213b7d !important;
            color:            #ffffff !important;
            border-color:     #213b7d !important;
        }
        </style>
    """, unsafe_allow_html=True)
    # Seed session state on first render (and re-seed if the selection has
    # become stale, e.g. the previously selected model is no longer in the
    # list). Once seeded, omit `default=` so Streamlit doesn't warn about
    # "default + Session State API" coexisting on the same widget.
    if st.session_state.get("sm_model_pills") not in models:
        st.session_state["sm_model_pills"] = models[current_idx]
    selected = st.pills(
        "Model",
        options=models,
        label_visibility="collapsed",
        key="sm_model_pills",
    )
    new_idx = models.index(selected) if selected in models else current_idx
    st.session_state[_MODEL_IDX_KEY] = new_idx
    return new_idx


def _render_lead_navigator(lead_times: list[int], current_idx: int) -> int:
    options  = [f"{lt}h" for lt in lead_times]
    if st.session_state.get("sm_lt_slider") not in options:
        st.session_state["sm_lt_slider"] = options[current_idx]
    selected = st.select_slider(
        "Lead Time",
        options=options,
        key="sm_lt_slider",
        label_visibility="collapsed",
    )
    new_idx = options.index(selected)
    st.session_state[_LT_IDX_KEY] = new_idx
    return new_idx


def _render_station_detail(
    choices: list, model_results: dict, lt: int,
    metric_key: str, unit: str, click_event, season_months=None,
) -> None:
    """Per-station drill-down: pick a station (dropdown OR click a marker) and
    show its metric-vs-lead and metric-by-month curves below the map.

    When several models are selected, every model's curves are overlaid (one line
    per model) so the station can be compared across models."""
    names = [c[1] for c in choices]
    key = "sm_station_sel"

    # Click-to-drill: a clicked marker's point_index maps 1:1 to `choices`
    # (both use the same keep-mask order).
    try:
        pts = (click_event or {}).get("selection", {}).get("points", [])
        idx = pts[0].get("point_index", pts[0].get("point_number")) if pts else None
    except Exception:
        idx = None
    if idx is not None and 0 <= idx < len(names):
        st.session_state[key] = names[idx]

    # Drop a stale stored value (e.g. region/model changed) before the widget.
    if st.session_state.get(key) not in names:
        st.session_state.pop(key, None)

    st.markdown(
        "<div style='border-top:1px solid #e5e7eb;margin:14px 0 6px;'></div>"
        "<div style='font-family:Inter,sans-serif;font-size:0.7rem;font-weight:700;"
        "letter-spacing:0.1em;text-transform:uppercase;color:#4B5563;'>"
        "📍 Station detail — pick a station or click a marker</div>",
        unsafe_allow_html=True,
    )
    sel_name = st.selectbox("Station", names, key=key, label_visibility="collapsed")
    sid = next(c[0] for c in choices if c[1] == sel_name)
    series = [(m, lt_map, lt_map.get(lt)) for m, lt_map in model_results.items() if lt_map]
    detail = _build_station_detail_figure(series, sid, sel_name, lt, metric_key, unit,
                                          season_months)
    if detail is not None:
        # Explicit key varying with station + lead + season so the chart always repaints
        # when any of them change (a keyless chart in this fragment can otherwise reuse a
        # stale render, so switching season looked like "no update").
        _season_tag = "-".join(str(m) for m in season_months) if season_months else "all"
        st.plotly_chart(detail, width="stretch", config={"displayModeBar": False},
                        key=f"sm_detail_{sel_name}_{lt}_{_season_tag}")
    else:
        st.caption("No per-station curves available for this selection.")


def _render_empty_state(message: str) -> None:
    st.markdown(
        f"""<div style="height:420px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;background:#F9FAFB;
            border-radius:10px;border:1px dashed #D1D5DB;color:#9CA3AF;
            font-family:'Inter',sans-serif;">
            <div style="font-size:32px;margin-bottom:12px;">🌐</div>
            <div style="font-size:13px;font-weight:600;">No data available</div>
            <div style="font-size:11px;margin-top:4px;">{message}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_loading_placeholder(model: str, lt: int) -> None:
    st.markdown(
        f"""<div style="height:500px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;background:#F9FAFB;
            border-radius:10px;border:1px dashed #D1D5DB;
            font-family:'Inter',sans-serif;gap:14px;">
            <div style="width:36px;height:36px;border:3px solid #E5E7EB;
                border-top-color:#213b7d;border-radius:50%;
                animation:sm-spin 0.75s linear infinite;"></div>
            <div style="font-size:13px;font-weight:600;color:#6B7280;">
                Rendering {model} &middot; {lt}h map&hellip;</div>
            <style>@keyframes sm-spin{{to{{transform:rotate(360deg);}}}}</style>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_stats_card(stats: dict | None, metric_key: str, has_clim: bool) -> None:
    def _row(label: str, value: str, accent: bool = False) -> str:
        bg    = "rgba(10,153,56,0.08)" if accent else "transparent"
        color = "#2e7d33" if accent else "#374151"
        return (
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;padding:6px 10px;border-radius:6px;"
            f"background:{bg};margin:2px 0;'>"
            f"<span style='font-size:11px;color:#6B7280;font-weight:500;"
            f"font-family:Inter,sans-serif;'>{label}</span>"
            f"<span style='font-size:12px;font-weight:700;color:{color};"
            f"font-family:Inter,sans-serif;'>{value}</span></div>"
        )

    if stats:
        rows  = _row("Mean",   f"{stats['mean']:.4f}",   accent=True)
        rows += _row("Median", f"{stats['median']:.4f}")
        rows += _row("Min",    f"{stats['min']:.4f}")
        rows += _row("Max",    f"{stats['max']:.4f}")
        rows += _row("P25",    f"{stats['p25']:.4f}")
        rows += _row("P75",    f"{stats['p75']:.4f}")
    else:
        rows = "<div style='color:#9CA3AF;font-size:12px;text-align:center;padding:20px 0;'>No data</div>"

    range_bar = ""
    if stats and stats["max"] > stats["min"]:
        pct = (stats["mean"] - stats["min"]) / (stats["max"] - stats["min"]) * 100
        range_bar = (
            f"<div style='margin:10px 10px 4px;'>"
            f"<div style='font-size:10px;color:#9CA3AF;margin-bottom:3px;"
            f"font-family:Inter,sans-serif;'>Range</div>"
            f"<div style='height:6px;background:#E5E7EB;border-radius:3px;position:relative;'>"
            f"<div style='position:absolute;left:0;top:0;height:6px;border-radius:3px;"
            f"background:linear-gradient(90deg,#0a9938,#213b7d);width:{pct:.1f}%;'></div></div>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:9px;color:#9CA3AF;margin-top:2px;font-family:Inter,sans-serif;'>"
            f"<span>{stats['min']:.3f}</span><span>{stats['max']:.3f}</span></div></div>"
        )

    title = f"{metric_key.upper()} (Δ vs Clim)" if has_clim else metric_key.upper()
    st.markdown(
        f"<div style='background:#FFFFFF;border:1px solid #E5E7EB;"
        f"border-radius:10px;padding:12px 0 12px;margin-top:44px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
        f"<div style='font-family:Inter,sans-serif;font-size:11px;"
        f"font-weight:700;color:#9CA3AF;letter-spacing:1px;"
        f"text-transform:uppercase;padding:0 10px 8px;"
        f"border-bottom:1px solid #F3F4F6;margin-bottom:6px;'>"
        f"Stats · {title}</div>"
        f"{rows}{range_bar}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Pure helpers — no Streamlit calls, safe to call from background threads
# ═══════════════════════════════════════════════════════════════════════════

def _add_region_boundary(fig: go.Figure, region: str) -> None:
    """Overlay the selected subregion's outline (like the coastline) on a geo figure.

    A no-op for whole-country / continent selections, whose borders the Natural Earth geo
    layer already draws. Shared by the heatmap, station, and diff maps so the outline reads
    identically everywhere.
    """
    boundary = region_boundary(region)
    if boundary is None:
        return
    blon, blat = boundary
    fig.add_trace(go.Scattergeo(
        lat=blat, lon=blon, mode="lines",
        line=dict(color="#1F2937", width=1.6),
        hoverinfo="skip", showlegend=False,
    ))


def _build_heatmap_figure(
    grid:        xr.DataArray,
    model_name:  str,
    metric_key:  str,
    lead_time:   int,
    clim_interp: np.ndarray | None,
    unit:        str = "",
    region:      str = "",
) -> go.Figure | None:
    has_clim = clim_interp is not None

    # No 2-D lat/lon field → no heatmap (the prewarm path reaches here too).
    if not _is_gridded(grid):
        return None

    lats = grid.latitude.values if "latitude" in grid.coords else grid.lat.values
    lons = grid.longitude.values if "longitude" in grid.coords else grid.lon.values

    z_vals = grid.values - clim_interp if has_clim else grid.values

    if has_clim:
        colorscale, reversescale = _DIVERGING_COLORSCALE, True
        abs_max    = float(np.nanpercentile(np.abs(z_vals[~np.isnan(z_vals)]), 98))
        zmin, zmax = -abs_max, abs_max
    elif metric_key.lower() in CORRELATION_METRICS:
        # Correlation is higher-is-better and can be negative: a diverging scale
        # centred at 0 (green = high/good, red = low/bad), symmetric so 0 reads as
        # neutral. Negatives are kept, not clipped like the error-metric branch.
        colorscale, reversescale = _CORRELATION_COLORSCALE, False
        finite = z_vals[~np.isnan(z_vals)]
        m = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        m = min(1.0, max(0.1, m))
        zmin, zmax = -m, m
    elif metric_key.lower() in ZERO_IS_BEST_METRICS:
        # Signed metric, best at 0 (bias): a diverging scale centred at 0 keeps
        # under- vs over-forecast distinguishable, with neutral at "no bias".
        # The value-NEUTRAL ramp, not the green/amber one — green there means
        # "better", and a negative bias is not better, just the other error.
        colorscale, reversescale = _SIGNED_COLORSCALE, False
        finite = z_vals[~np.isnan(z_vals)]
        m = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        zmin, zmax = -m, m
    else:
        # Error metrics (lower is better): YlOrRd read forward — pale yellow = low error,
        # deep red = high error (no reverse, so high/worse stands out).
        colorscale, reversescale = _SEQUENTIAL_COLORSCALE, False
        valid = z_vals[~np.isnan(z_vals) & (z_vals > 0)]
        if valid.size > 0:
            zmin = float(np.nanpercentile(valid, 2))
            zmax = float(np.nanpercentile(valid, 98))
        else:
            zmin, zmax = 0, 1

    lat_min, lat_max = float(lats.min()) - 0.5, float(lats.max()) + 0.5
    lon_min, lon_max = float(lons.min()) - 0.5, float(lons.max()) + 0.5
    _u = f" ({unit})" if unit else ""
    # The clim overlay plots model − clim (the same Δ the Lead-Time chart shows),
    # so say exactly that and which sign is good — "Improvement" would invert
    # the reading for error metrics, where negative Δ is the good direction.
    _delta_note = "positive = better" if higher_is_better(metric_key) else "negative = better"
    metric_label = (
        f"{metric_key.upper()} Δ vs Climatology (model − clim; {_delta_note}){_u}"
        if has_clim else f"{metric_key.upper()}{_u}"
    )

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_vals, x=lons, y=lats,
        colorscale=colorscale, reversescale=reversescale,
        zsmooth=False, connectgaps=False,
        zmin=zmin, zmax=zmax, opacity=0.88,
        colorbar=dict(
            title=dict(text=f"<b>{metric_label}</b>", side="right",
                       font=dict(size=11, color="#374151")),
            thickness=10, len=0.80,
            x=0.88, xanchor="left", xpad=4,
            tickfont=dict(size=9, color="#6B7280"),
            outlinewidth=0, bgcolor="rgba(255,255,255,0.75)",
        ),
        hovertemplate=(
            f"<b>{metric_label}</b>: %{{z:.4f}}<br>"
            "Lat: %{y:.2f}°  Lon: %{x:.2f}°<extra></extra>"
        ),
    ))
    # Invisible scattergeo to trigger Plotly's geo overlay (coastlines/borders)
    fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None], mode="markers",
        marker=dict(size=0, opacity=0), showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis=dict(range=[lon_min, lon_max], showgrid=False, zeroline=False,
                   showticklabels=False, domain=[0, 0.86]),
        yaxis=dict(range=[lat_min, lat_max], showgrid=False, zeroline=False,
                   showticklabels=False, domain=[0, 1], scaleanchor="x", scaleratio=1),
        geo=dict(
            domain=dict(x=[0, 0.86], y=[0, 1]),
            showocean=True,       oceancolor=MAP_OCEAN,
            showlakes=True,       lakecolor=MAP_OCEAN,
            showland=True,        landcolor="rgba(0,0,0,0)",
            showcoastlines=True,  coastlinecolor=MAP_COASTLINE, coastlinewidth=1.2,
            showcountries=True,   countrycolor=MAP_COUNTRY,  countrywidth=0.7,
            showframe=True,       framecolor="#9CA3AF",
            projection_type="equirectangular",
            lataxis_range=[lat_min, lat_max],
            lonaxis_range=[lon_min, lon_max],
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text=(
                f"<b>{model_name}</b>"
                f"  <span style='color:#6B7280;font-size:11px;'>{lead_time}h lead</span>"
            ),
            x=0.03, y=0.97,
            font=dict(size=13, color="#171d1a", family="Inter"),
        ),
        template="plotly_white", height=500,
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", bordercolor="#E5E7EB", font=dict(size=11)),
    )
    _add_region_boundary(fig, region)
    return fig


def _build_station_figure(
    da:         xr.DataArray,
    model_name: str,
    metric_key: str,
    lead_time:  int,
    region:     str,
    unit:       str = "",
) -> go.Figure | None:
    """Point map for a station target: one marker per station at its true
    lat/lon, colored by the metric, sized by valid-day count (``n_valid``).

    Returns None when no station falls inside ``region``. Marker size makes
    the data-density spread visible (the M1 caveat), and the hover surfaces
    nn_distance_km (the M2 representativeness caveat).
    """
    da = da.squeeze()  # drop a singleton year dim if present
    lons, lats = _station_lonlat(da)
    vals = np.asarray(da.values, dtype=float)

    # Shared keep-mask (in-region & finite) so markers line up with the detail
    # selector. Stations with no valid obs (n_valid==0 → NaN) are excluded.
    keep = _station_keep_mask(da, region)
    if not keep.any():
        return None
    lats, lons, vals = lats[keep], lons[keep], vals[keep]

    def _coord(name):
        return np.asarray(da[name].values)[keep] if name in da.coords else None
    names   = _coord("name")
    n_valid = _coord("n_valid")
    nn_dist = _coord("nn_distance_km")

    zero_best = metric_key.lower() in ZERO_IS_BEST_METRICS
    finite = vals[np.isfinite(vals)]
    if zero_best:
        # Signed metric, best at 0 (bias): diverging scale centred on 0, like the
        # gridded map — a sequential range from the value percentiles would hide
        # which stations under- vs over-forecast.
        m = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        cmin, cmax = -m, m
        marker_colorscale, marker_reversescale = _SIGNED_COLORSCALE, False
    else:
        cmin = float(np.percentile(finite, 2)) if finite.size else 0.0
        cmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        # Error metrics: YlOrRd forward — pale = low error, red = high error.
        marker_colorscale, marker_reversescale = _SEQUENTIAL_COLORSCALE, False

    # Marker size ∝ valid-day count, so sparse stations read as small dots.
    if n_valid is not None and np.nanmax(n_valid) > np.nanmin(n_valid):
        nv = np.nan_to_num(n_valid.astype(float), nan=float(np.nanmin(n_valid)))
        sizes = 6.0 + 12.0 * (nv - nv.min()) / (nv.max() - nv.min())
    else:
        sizes = np.full(vals.shape, 9.0)

    _u = f" ({unit})" if unit else ""
    metric_label = f"{metric_key.upper()}{_u}"
    custom = [
        [str(names[i]) if names is not None else "",
         float(n_valid[i]) if n_valid is not None else None,
         float(nn_dist[i]) if nn_dist is not None else None]
        for i in range(len(vals))
    ]
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        f"{metric_label}: %{{marker.color:.4f}}<br>"
        "Lat %{lat:.2f}°  Lon %{lon:.2f}°<br>"
        "valid days: %{customdata[1]:.0f}  ·  nearest cell: %{customdata[2]:.1f} km"
        "<extra></extra>"
    )

    lat_min, lat_max = float(lats.min()) - 0.5, float(lats.max()) + 0.5
    lon_min, lon_max = float(lons.min()) - 0.5, float(lons.max()) + 0.5

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, mode="markers", customdata=custom,
        marker=dict(
            color=vals, colorscale=marker_colorscale, reversescale=marker_reversescale,
            cmin=cmin, cmax=cmax, size=sizes, line=dict(width=0.5, color="#374151"),
            colorbar=dict(
                title=dict(text=f"<b>{metric_label}</b>", side="right",
                           font=dict(size=11, color="#374151")),
                thickness=10, len=0.80, tickfont=dict(size=9, color="#6B7280"),
                outlinewidth=0, bgcolor="rgba(255,255,255,0.75)",
            ),
        ),
        hovertemplate=hovertemplate,
    ))
    fig.update_layout(
        geo=dict(
            showocean=True,      oceancolor=MAP_OCEAN,
            showlakes=True,      lakecolor=MAP_OCEAN,
            showland=True,       landcolor=MAP_LAND,
            showcoastlines=True, coastlinecolor=MAP_COASTLINE, coastlinewidth=1.2,
            showcountries=True,  countrycolor=MAP_COUNTRY,  countrywidth=0.7,
            showframe=True,      framecolor="#9CA3AF",
            projection_type="equirectangular",
            lataxis_range=[lat_min, lat_max], lonaxis_range=[lon_min, lon_max],
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text=(f"<b>{model_name}</b>  <span style='color:#6B7280;font-size:11px;'>"
                  f"{lead_time}h lead · {len(lats)} stations</span>"),
            x=0.03, y=0.97, font=dict(size=13, color="#171d1a", family="Inter"),
        ),
        template="plotly_white", height=500,
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", bordercolor="#E5E7EB", font=dict(size=11)),
    )
    _add_region_boundary(fig, region)
    return fig


def _build_station_detail_figure(
    series:         list,
    station_id,
    station_name:   str,
    displayed_lead: int,
    metric_key:     str,
    unit:           str = "",
    season_months=None,
) -> "go.Figure | None":
    """One station's metric-vs-lead and metric-by-month curves, one line per model.

    ``series`` is ``[(model_name, lt_map, ltd_at_lead), ...]`` for every selected model.
    Both curves come from the already-loaded summaries (no extra I/O): the lead curve is
    the station's annual value across leads; the month curve is its monthly value at the
    displayed lead. Returns None if no model has data for this station.
    """
    from plotly.subplots import make_subplots

    from ui.app_config.settings import MODEL_PALETTE, forecast_label

    _MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _u = f" ({unit})" if unit else ""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{metric_key.upper()} vs lead",
                        f"{metric_key.upper()} by month · {displayed_lead}h"),
    )
    drawn = 0
    for i, (model_name, lt_map, ltd_at_lead) in enumerate(series):
        leads, lead_vals = _station_lead_curve(lt_map, station_id, season_months)
        months, month_vals = (_station_month_curve(ltd_at_lead, station_id, season_months)
                              if ltd_at_lead is not None else ([], []))
        if not leads and not months:
            continue
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        label = forecast_label(model_name)
        fig.add_trace(go.Scatter(
            x=leads, y=lead_vals, mode="lines+markers", name=label, legendgroup=model_name,
            line=dict(color=color, width=2), marker=dict(size=6),
            hovertemplate=f"{label}<br>lead %{{x}}h<br>%{{y:.4f}}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[_MN[m - 1] for m in months], y=month_vals, mode="lines+markers",
            name=label, legendgroup=model_name, showlegend=False,
            line=dict(color=color, width=2), marker=dict(size=6),
            hovertemplate=f"{label}<br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ), row=1, col=2)
        drawn += 1
    if drawn == 0:
        return None

    fig.update_yaxes(title_text=f"{metric_key.upper()}{_u}", row=1, col=1)
    fig.update_xaxes(title_text="lead time (h)", row=1, col=1)
    fig.update_layout(
        title=dict(text=f"<b>{station_name}</b>", x=0.03, y=0.97,
                   font=dict(size=13, color="#171d1a", family="Inter")),
        showlegend=drawn > 1, height=300, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        margin=dict(l=10, r=10, t=56, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


def _compute_stats(
    grid_values: np.ndarray,
    clim_values: np.ndarray | None,
) -> dict | None:
    vals = grid_values.ravel()
    if clim_values is not None:
        vals = vals - clim_values.ravel()
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return {
        "mean":   float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min":    float(np.min(vals)),
        "max":    float(np.max(vals)),
        "p25":    float(np.percentile(vals, 25)),
        "p75":    float(np.percentile(vals, 75)),
    }


def _interpolate_clim(
    model_grid:  xr.DataArray,
    clim_grid:   xr.DataArray,
) -> np.ndarray | None:
    """Bilinear regrid of clim_grid onto model_grid coordinates (cached by object ids)."""
    key = (id(model_grid), id(clim_grid))
    if key in _CLIM_INTERP_CACHE:
        return _CLIM_INTERP_CACHE[key]

    def _lat_name(da: xr.DataArray) -> str:
        return "latitude" if "latitude" in da.dims else "lat"

    def _lon_name(da: xr.DataArray) -> str:
        return "longitude" if "longitude" in da.dims else "lon"

    try:
        lat_m = _lat_name(model_grid)
        lon_m = _lon_name(model_grid)
        lat_c = _lat_name(clim_grid)
        lon_c = _lon_name(clim_grid)
        result: np.ndarray | None = clim_grid.interp(
            {lat_c: model_grid[lat_m], lon_c: model_grid[lon_m]},
            method="linear",
        ).values
    except Exception as exc:
        print(f"[spatial_map] clim interpolation failed: {exc}")
        result = None

    _CLIM_INTERP_CACHE[key] = result
    return result


def _make_fig_key(
    model:        str,
    lt:           int,
    metric_key:   str,
    region:       str,
    has_clim:     bool,
    data_hash:    int,
    unit:         str = "",
    season_months: "tuple[int, ...] | None" = None,
) -> tuple:
    return (model, lt, metric_key, region, has_clim, data_hash, unit, season_months)



def _data_hash(da: xr.DataArray) -> int:
    try:
        arr   = da.values.ravel()
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return 0
        return hash((
            round(float(np.mean(valid))  * 1e6),
            round(float(np.min(valid))   * 1e6),
            round(float(np.max(valid))   * 1e6),
            round(float(np.std(valid))   * 1e6),
            valid.size,
        ))
    except Exception:
        return 0


def _prewarm_model(
    model:        str,
    lt_map:       dict[int, LeadTimeData],
    metric_key:   str,
    region:       str,
    clim:         "ClimData | None",
    unit:         str = "",
) -> None:
    """Background thread: pre-build figures + stats for every lead time.

    Builds a matched-month climatology per ltd so the cached diff is the
    same comparison the foreground render will produce.
    """
    for lt, ltd in lt_map.items():
        if ltd is None:
            continue
        try:
            grid = annual_spatial_grid(ltd, region)
        except Exception:
            continue
        matched = _matched_clim_spatial(ltd, clim, region) if clim is not None else None
        clim_interp = _interpolate_clim(grid, matched) if matched is not None else None
        for has_clim in (False, True):
            ci  = clim_interp if (has_clim and clim_interp is not None) else None
            if has_clim and ci is None:
                continue
            key = _make_fig_key(model, lt, metric_key, region, has_clim, _data_hash(grid), unit)
            if key not in _FIGURE_CACHE:
                try:
                    _FIGURE_CACHE[key] = _build_heatmap_figure(grid, model, metric_key, lt,
                                                               ci, unit, region)
                except Exception as exc:
                    print(f"[spatial_map] prewarm {model}/{lt}/{has_clim}: {exc}")
            if key not in _STATS_CACHE:
                try:
                    _STATS_CACHE[key] = _compute_stats(grid.values, ci)
                except Exception as exc:
                    print(f"[spatial_map] prewarm stats {model}/{lt}/{has_clim}: {exc}")


def _prewarm_task(args: tuple) -> None:
    _prewarm_model(*args)


def prewarm_all_figures(
    entries: list[tuple],
    regions: list[str] | None = None,
) -> None:
    """Pre-build heatmap figures for all provided (model, metric, region) combos.

    Each entry is a 5-tuple: (model_label, lt_map, metric_key, clim, unit)
    where `clim` is the full ClimData (or None). The prewarm uses
    `clim.monthly_spatial_average` to build a per-model matched-month
    climatology so cached figures reflect the same fair comparison the
    foreground render produces.

    All tasks run through a bounded ThreadPoolExecutor (4 workers) started
    in a single daemon thread so startup is not blocked.
    """
    if regions is None:
        regions = ["Africa"]

    tasks: list[tuple] = []
    # Outer loop over regions so Africa is fully warmed before blocs, blocs before countries.
    for region in regions:
        for label, lt_map, metric_key, clim, unit in entries:
            if not lt_map:
                continue
            has_clim = clim is not None
            pw_key = (label, metric_key, region, has_clim, id(lt_map))
            if pw_key in _PREWARM_STARTED:
                continue
            _PREWARM_STARTED.add(pw_key)
            tasks.append((label, lt_map, metric_key, region, clim, unit))

    if not tasks:
        return

    def _run() -> None:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="fig-pw") as pool:
            pool.map(_prewarm_task, tasks)
        print(f"[spatial_map] figure prewarm complete — {len(tasks)} (model×region) tasks")

    threading.Thread(target=_run, daemon=True, name="fig-prewarm-pool").start()

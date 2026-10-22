# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""National Overview — ForecastCheck design, wired to the LOCAL engine.

A faithful rebuild of the redesign's National Overview page:

  1. Country & Region — chip picker over the deployment's regions
  2. Two section tabs, one Config panel each, only the selected one rendered:

     Calibrated Forecast Metrics — the CALIBRATED (+IDR) side, scored by CRPS:
       Country View — per-pixel season mean (or stations) at one lead
       Graph View   — a curve on either x-axis, chosen by the Month / Lead Time
                      switch: the season's monthly cycle at one lead, or its
                      skill against forecast horizon over every lead

     Raw Forecast Metrics — the UNCALIBRATED side, scored by mae / bias /
       correlation. Came from Detailed Metrics, where it was "Spatial
       Performance", so both halves of a model's story sit on one page:
       Country View — per-pixel field (or stations) at one lead
       Graph View   — skill against forecast horizon, one curve per selected
                      model, so models are compared directly

The lead-time curve replaces the separate Lead Time Performance section, which
carried a duplicate config panel and a per-pixel map nobody read. On the lead-time
x-axis the chart spans every lead, so the config panel drops its lead selector
rather than leaving a control that changes nothing.

The Conditioning group's YEAR selector picks a single verified year or ``All
years`` (the default), which averages across every year discovery finds — the
behaviour the pages had before the selector existed.

The layout, config-panel styling and view toggles mirror the design 1:1; the
DATA is read from local NetCDF via the discovery-driven engine (``menu`` →
``load_run`` → ``geo``). Every dropdown offers only what exists on disk, so the
menu can never request a file the pipeline hasn't produced. Climatology overlays
appear only when a climatology file exists for the selection (gracefully noted
otherwise).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.components.region_picker import render_region_picker

from ui.app_config.settings import (DEFAULT_REGION, DEFAULT_SEASON, REGIONS,
                                     SERIES_PALETTE, ZERO_IS_BEST_METRICS,
                                     map_scale_for,
                                     SEASONS, VARIABLE_UNITS, forecast_label,
                                     metric_label, sample_caveat, season_months,
                                     target_label, variable_label)
from ui.backend.data_loader.climatology_loader import (ClimatologyLoader,
                                                        climatology_available)
from ui.components.filters import _altitude_band_options
from ui.components.chrome import (AMBER, GREEN, GREEN_BG, GREEN_DARK, MUTED,
                                  NAVY, config_group, render_explore_cards,
                                  render_footer, render_header)
from ui.services.data_cache import get_menu_index, ui_results_root
from ui.services.loader import load_run, load_season_spatial
from ui.utils.geo import (annual_spatial_grid, apply_band, band_monthly_means,
                          band_season_mean, regional_monthly_means,
                          regional_season_mean)
from ui.utils.regions import mask_region

#: ALTITUDE selector sentinel — every band, i.e. no elevation filter.
ALL_BANDS = "All altitudes"

#: YEAR selector sentinel — the multi-year mean, which is what every page showed
#: before the selector existed and stays the default.
ALL_YEARS = "All years"

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Per-season line colours (design uses one hue per season).
_SEASON_COLORS = {
    "Yearly": NAVY, "JASO (Jul–Oct)": GREEN, "MAMJ (Mar–Jun)": "#d64545",
    "NDJF (Nov–Feb)": AMBER,
}
_clim_loader = ClimatologyLoader()


# ═════════════════════════════════════════════════════════════════════════
#  Page
# ═════════════════════════════════════════════════════════════════════════

def render() -> None:
    render_header(active="national_overview")
    _inject_page_css()

    st.markdown(
        f"""
        <div style="padding:26px 6px 0;max-width:700px;">
          <h1 style="margin:0 0 8px;">National Overview</h1>
          <p style="color:{MUTED};font-size:0.88rem;line-height:1.7;margin:0;">
            Explore forecast skill across seasons, lead times and regions.
            Calibrated Forecast Metrics scores the calibrated forecasts — calibrated
            with Isotonic Distributional Regression so that models of different
            kinds compare fairly — while Raw Forecast Metrics scores the
            uncalibrated runs by MAE, bias and correlation.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    region = _render_region_picker()
    menu = get_menu_index()
    root = ui_results_root()

    if _render_section_switch() == "Raw Forecast Metrics":
        _raw_metrics_section(menu, root, region)
    else:
        _calibrated_metrics_section(menu, root, region)

    render_explore_cards("score_decomposition", "detailed_metrics")
    render_footer()


def _render_section_switch() -> str:
    """Top-level tabs. Only the selected section renders.

    A segmented control rather than ``st.tabs``, which executes every tab body on
    each rerun — both sections read NetCDF, so tabs would load the hidden one too.
    """
    with st.container(key="no_section_switch"):
        picked = st.segmented_control(
            "section", ["Calibrated Forecast Metrics", "Raw Forecast Metrics"],
            default="Calibrated Forecast Metrics", key="no_section",
            label_visibility="collapsed")
    return picked or "Calibrated Forecast Metrics"


def _calibrated_metrics_section(menu, root, region: str) -> None:
    """Calibrated (+IDR) skill: Country View, or a graph on either x-axis."""
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.82rem;margin:2px 0 14px;'>"
        f"Seasonal skill of the selected calibrated model over <b>{region}</b></p>",
        unsafe_allow_html=True,
    )
    # Both toggles are chosen below but read here: on a lead-time x-axis the
    # chart spans every lead, so the config panel must not offer a single-lead
    # selector. The widgets have already written their new values to
    # session_state by this point in the rerun.
    sp_view = st.session_state.get("no_sp_view") or "Graph View"
    sp_x = st.session_state.get("no_sp_xaxis") or "Month"
    plots_one_lead = not (sp_view == "Graph View" and sp_x == "Lead Time")

    col_cfg, col_view = st.columns([1, 3], gap="medium")
    with col_cfg:
        # Graph View overlays one curve per model, so it takes a MODELS
        # multiselect; Country View is a per-pixel map and can only show one.
        sp_cfg = _render_config(menu, "sp", include_lead_time=plots_one_lead,
                                include_season=True, calibrated_only=True,
                                region=region,
                                multi_model=sp_view == "Graph View")
    with col_view:
        sp_view = st.segmented_control(
            "sp view", ["Country View", "Graph View"], default="Graph View",
            key="no_sp_view", label_visibility="collapsed")
        # The x-axis switch sits under the view toggle and only means anything
        # for the graph, so Country View hides it.
        if (sp_view or "Graph View") == "Graph View":
            st.markdown(
                f"<div style='font-size:0.6rem;font-weight:700;"
                f"letter-spacing:0.1em;color:{MUTED};margin:10px 0 2px 3px;'>"
                f"PLOT AGAINST</div>", unsafe_allow_html=True)
            sp_x = st.segmented_control(
                "sp x-axis", ["Month", "Lead Time"], default="Month",
                key="no_sp_xaxis", label_visibility="collapsed")
        _seasonal_section(menu, root, region, sp_cfg, sp_view or "Graph View",
                          sp_x or "Month")


def _raw_metrics_section(menu, root, region: str) -> None:
    """A RAW model's deterministic skill: per-pixel map, or skill vs lead time.

    Moved here from Detailed Metrics (where it was "Spatial Performance") so the
    two halves of a model's story sit on one page: Calibrated Forecast Metrics is the
    calibrated (+IDR) side under CRPS, this is the uncalibrated side under
    mae / bias / correlation.

    Country View is the per-pixel field at one lead; Graph View is the metric
    against forecast horizon over every lead, so the lead selector drops out
    there. Graph View is also the only view for a selection with no per-pixel
    field — a station truth, or a spatially-reduced metric.
    """
    # Read the view before the config renders — only the map pins a single lead.
    view = st.session_state.get("no_rm_view") or "Country View"

    st.markdown(
        f"<p style='color:{MUTED};font-size:0.82rem;margin:2px 0 14px;'>"
        f"Skill of the raw (uncalibrated) forecast over <b>{region}</b></p>",
        unsafe_allow_html=True,
    )
    col_cfg, col_view = st.columns([1, 3], gap="medium")
    with col_cfg:
        # Graph View overlays a curve per model; the map is one model at a time.
        # NOT gridded_only: station truth works in both views — the curve is a
        # regional mean over gauges, and Country View has a gauge map. The metric
        # facet narrows itself (stations carry mae/bias, no correlation).
        cfg = _render_config(menu, "rm", include_lead_time=view == "Country View",
                             include_season=True,
                             raw_only=True, region=region,
                             multi_model=view == "Graph View")
    with col_view:
        view = st.segmented_control(
            "rm view", ["Country View", "Graph View"], default="Country View",
            key="no_rm_view", label_visibility="collapsed") or "Country View"
        if cfg is None:
            _no_data("No raw models available for this variable.")
            return
        year_maps = _load_all_years(menu, cfg, root)
        if not year_maps:
            _no_data(f"No data for {forecast_label(cfg['forecast'])} · {region}.")
            return
        months = season_months(cfg["season"])
        band = cfg.get("band", "all")
        unit = VARIABLE_UNITS.get(cfg["variable"], "")
        season = cfg["season"]
        season_note = season if season != DEFAULT_SEASON else "all months"

        if view == "Graph View":
            fig = go.Figure()
            all_x: set = set()
            years_used: set = set()
            missing: list[str] = []
            # One curve per thing being compared — models when models vary, truth
            # sources when they do. This loop used to iterate models alone, so a
            # second truth source was selectable and silently never drawn.
            for i, (fc, tr, base) in enumerate(_series_keys(cfg)):
                ym = _load_all_years(menu, {**cfg, "forecast": fc, "truth": tr}, root)
                if not ym:
                    missing.append(base)
                    continue
                xs, ys = _season_leadtime(ym, region, months, band)
                if not xs:
                    missing.append(base)
                    continue
                years_used |= set(ym)
                all_x |= set(xs)
                name = f"{base} · {season}"
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=name,
                    line=dict(color=SERIES_PALETTE[i % len(SERIES_PALETTE)],
                              width=3),
                    marker=dict(size=7),
                    hovertemplate=f"{name} · %{{x}}h · %{{y:.4f}}<extra></extra>"))
            if not all_x:
                _no_data(f"No {season} data for this selection.")
                return
            # bias is signed with 0 as its ideal, so mark the target line.
            if cfg["metric"] in ZERO_IS_BEST_METRICS:
                fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
            _line_layout(fig, x_title="Lead Time (hours)", xs=sorted(all_x),
                         y_title=f"{metric_label(cfg['metric'])}"
                         f"{f' ({unit})' if unit else ''}", height=480)
            st.plotly_chart(fig, width="stretch")
            gap = f" · no data for {', '.join(missing)}" if missing else ""
            _agg_caption(sorted(years_used), False, extra=season_note + gap)
            return

        if _is_station_target(year_maps):
            # Gauge map + click-to-inspect, the same helper Calibrated Forecast Metrics
            # uses, so a station selection behaves identically on both tabs.
            _station_country_view(year_maps, region, months, cfg, [cfg["lead"]],
                                  band)
            _agg_caption(list(year_maps), False,
                         extra=f"{season_note} · {cfg['lead']}h")
            return
        grid = _agg_season_grid_lead(year_maps, region, months, cfg["lead"],
                                     cfg, root, band)
        if grid is None:
            _no_data("No per-pixel field for this selection — use Graph View.")
            return
        # bias -> neutral cool/warm, correlation -> RdYlGn, errors -> sequential.
        # Unit on the colorbar, exactly as the lead-time chart's y-axis carries it.
        # A map labelled only "CRPS" leaves the reader to guess mm/hr vs degC, and the
        # two are the same scale of number for this data.
        _draw_map(grid, cbar=f"{metric_label(cfg['metric'])}"
                       f"{f' ({unit})' if unit else ''}",
                  scale=map_scale_for(cfg["metric"]))
        _agg_caption(list(year_maps), False,
                     extra=f"{season_note} · {cfg['lead']}h")


# ═════════════════════════════════════════════════════════════════════════
#  Section 1 — Country & Region picker
# ═════════════════════════════════════════════════════════════════════════

def _render_region_picker() -> str:
    """The shared Country & Region selector (see ui.components.region_picker)."""
    return render_region_picker("no")


# ═════════════════════════════════════════════════════════════════════════
#  Config panel (one per section, mirroring the design)
# ═════════════════════════════════════════════════════════════════════════

def _render_config(menu, prefix: str, include_lead_time: bool,
                   include_season: bool, calibrated_only: bool = False,
                   gridded_only: bool = False,
                   raw_only: bool = False,
                   region: str | None = None,
                   multi_model: bool = False) -> dict | None:
    """The shared config panel. Returns the selection, or None if it is empty.

    ``multi_model`` turns MODEL into a MODELS multiselect for views that overlay
    one curve per model. Every downstream facet (truth, metric, year) is then the
    INTERSECTION across the selection, so a chart can never mix in a model that
    lacks one of them. ``cfg["forecast"]`` stays the first pick either way, so
    single-model paths need no change; ``cfg["forecasts"]`` is the full list.

    TRUTH SOURCE becomes a multiselect too, but **only while exactly one model is
    picked**, and the two multi-selects are deliberately mutually exclusive. They
    answer different questions and must not be mixed:

        several models, one truth   ->  which forecast is better?   lower IS better
        one model, several truths   ->  how much does the answer depend on the
                                        reference?                  lower is NOT better

    Two models against two truths would be four lines whose pairwise differences mean
    two different things, which nobody reads correctly. The second view is worth having
    because the gap between its curves is **truth uncertainty** — same forecast, same
    days, different reference — a property of the reference, not skill.
    """
    with st.container(key=f"{prefix}_config"):
        st.markdown(
            f"<div style='font-weight:700;font-size:0.78rem;letter-spacing:0.08em;"
            f"color:{GREEN_DARK};margin-bottom:6px;'>CONFIGURATION</div>",
            unsafe_allow_html=True,
        )

        with config_group(prefix, "ft", "Forecast &amp; Truth"):
            variables = menu.values("variable")
            if not variables:
                st.info("No verification summaries found on disk.")
                return None
            variable = st.selectbox("VARIABLE", variables,
                                    format_func=variable_label,
                                    key=f"{prefix}_variable")
            # calibrated_only (National Overview): show the +IDR variants only,
            # which are scored by CRPS so the metric group resolves to CRPS.
            # Otherwise (e.g. Detailed Metrics Spatial) show every model, with
            # its full discovered metric set.
            forecasts = menu.values("forecast", variable=variable)
            if calibrated_only:
                forecasts = [f for f in forecasts if f.endswith("-idr")]
                if not forecasts:
                    st.info("No calibrated (+IDR) models for this variable.")
                    return None
            elif raw_only:
                # Raw Forecast Metrics: the uncalibrated models, scored by the
                # deterministic metrics (mae / bias / correlation). Complements
                # Calibrated Forecast Metrics, which is the +IDR side under CRPS.
                forecasts = [f for f in forecasts if not f.endswith("-idr")]
                if not forecasts:
                    st.info("No raw models for this variable.")
                    return None
            if multi_model:
                picked = st.multiselect("MODELS", forecasts,
                                        default=forecasts[:2],
                                        format_func=forecast_label,
                                        key=f"{prefix}_models")
                if not picked:
                    st.info("Pick at least one model.")
                    return None
            else:
                picked = [st.selectbox("MODEL", forecasts,
                                       format_func=forecast_label,
                                       key=f"{prefix}_model")]
            # Say it where the selection is made, not on each view: a forecast
            # whose archive covers only part of the year is not scored on the
            # same sample as the others, and nothing in the numbers shows that.
            _render_sample_caveat(picked, variable)
            forecast = picked[0]
            truths = sorted(set.intersection(*(
                set(menu.values("target", forecast=f, variable=variable))
                for f in picked)))
            if gridded_only:  # per-pixel maps → gridded truth only
                truths = [t for t in truths if "station" not in menu.values(
                    "target_kind", forecast=forecast, target=t, variable=variable)]
            if not truths:
                st.info("The selected models share no truth source.")
                return None
            # One model -> compare truth sources; several -> compare models. Never
            # both, so each chart answers exactly one question (see the docstring).
            compare_truths = multi_model and len(picked) == 1 and len(truths) > 1
            if compare_truths:
                picked_truths = st.multiselect(
                    "TRUTH SOURCES", truths, default=truths,
                    format_func=target_label, key=f"{prefix}_truths")
                if not picked_truths:
                    st.info("Pick at least one truth source.")
                    return None
                truth = picked_truths[0]
            else:
                truth = st.selectbox("TRUTH SOURCE", truths,
                                     format_func=target_label,
                                     key=f"{prefix}_truth")
                picked_truths = [truth]

        with config_group(prefix, "sc", "Score"):
            metrics = sorted(set.intersection(*(
                set(menu.values("metric", forecast=f, target=tr, variable=variable))
                for f in picked for tr in picked_truths)))
            if not metrics:
                st.info("No metric available for this selection.")
                return None
            metric = st.selectbox("METRIC", metrics, format_func=metric_label,
                                  key=f"{prefix}_metric")

        season = DEFAULT_SEASON
        lead = None
        with config_group(prefix, "co", "Conditioning"):
            # Year: ALL_YEARS keeps the multi-year mean the pages have always
            # shown; picking one year restricts every chart to it. Newest first,
            # since a forecaster usually wants the most recent verified season.
            all_years = sorted(set.intersection(*(
                set(menu.values("year", forecast=f, target=truth,
                                variable=variable, metric=metric))
                for f in picked)))
            year_opts = [ALL_YEARS] + sorted(all_years, reverse=True)
            year_pick = st.selectbox("YEAR", year_opts, key=f"{prefix}_year")
            years = all_years if year_pick == ALL_YEARS else [year_pick]

            if include_season:
                season = st.selectbox("SEASON", list(SEASONS),
                                      index=_idx(list(SEASONS), DEFAULT_SEASON),
                                      key=f"{prefix}_season")
            if include_lead_time:
                # Leads offered are the selected year's, so a year that lacks a
                # lead cannot be asked for it.
                leads = _leads_for(menu, forecast, truth, metric, variable, years)
                if not leads:
                    st.info("No lead time on disk for this selection.")
                    return None
                mid = leads[len(leads) // 2]
                lead = st.selectbox("LEAD TIME", leads, index=leads.index(mid),
                                    format_func=lambda h: f"{h}h",
                                    key=f"{prefix}_lead")

            # Altitude bands ride on the crps/mae/bias summaries — from the DEM
            # for gridded truth, from each gauge's own elevation for stations.
            # _altitude_band_options hides the control unless the region IS the
            # deployment country (the stored band means are whole-domain, which
            # only equals the region then) and the files are band-aware.
            band = ALL_BANDS
            bands = _altitude_band_options(menu, region, variable, truth, metric,
                                           years[0] if years else None, [forecast])
            if bands:
                band = st.selectbox("ALTITUDE", [ALL_BANDS] + list(bands),
                                    key=f"{prefix}_band")
            vs_clim = st.toggle("Compare vs Climatology", key=f"{prefix}_clim")

    return {"variable": variable, "forecast": forecast, "forecasts": picked,
            # `truth` stays the first pick so every single-truth path is unchanged;
            # `truths` is the full list, and is longer than one only in the
            # one-model-several-truths view.
            "truth": truth, "truths": picked_truths,
            "metric": metric, "season": season, "lead": lead, "vs_clim": vs_clim,
            "year": None if year_pick == ALL_YEARS else year_pick,
            "band": "all" if band == ALL_BANDS else band}


def _leads_for(menu, forecast, truth, metric, variable, years) -> list[int]:
    leads: set[int] = set()
    for y in years:
        rs = menu.resolve_storage(forecast=forecast, target=truth, metric=metric,
                                  year=y, variable=variable)
        if rs:
            leads.update(rs["leads"])
    return sorted(leads)


# ═════════════════════════════════════════════════════════════════════════
#  Data loading (discovery-driven, aggregated over the test period)
# ═════════════════════════════════════════════════════════════════════════

def _load_all_years(menu, cfg, root):
    """{year: (lt_map, resolve_dict)} for the years in scope.

    ``cfg["year"]`` None (the ALL_YEARS pick, and the default for any caller that
    predates the selector) loads every year discovery finds and the page averages
    across them; a chosen year loads that one only.
    """
    years = menu.values("year", forecast=cfg["forecast"], target=cfg["truth"],
                        variable=cfg["variable"], metric=cfg["metric"])
    picked = cfg.get("year")
    if picked is not None:
        years = [y for y in years if y == picked]
    out: dict = {}
    for y in years:
        rs = menu.resolve_storage(forecast=cfg["forecast"], target=cfg["truth"],
                                  metric=cfg["metric"], year=y,
                                  variable=cfg["variable"])
        if rs is None:
            continue
        lt_map = load_run(cfg["forecast"], rs["native_var"], cfg["truth"],
                          rs["storage_metric"], y, rs["leads"], root=root)
        if lt_map:
            out[y] = (lt_map, rs)
    return out


def _multi_year_monthly(year_maps, lead, region, band="all") -> np.ndarray | None:
    arrs = []
    for lt_map, _ in year_maps.values():
        ltd = lt_map.get(lead) if lead is not None else next(iter(lt_map.values()), None)
        if ltd is None:
            continue
        # Band means are pipeline-computed per elevation band (DEM for gridded,
        # gauge elevation for stations); "all" falls back to the regional mean.
        arr = (band_monthly_means(ltd, band) if band != "all"
               else regional_monthly_means(ltd, region))
        if arr is not None and not np.all(np.isnan(arr)):
            arrs.append(np.asarray(arr, float)[:12])
    if not arrs:
        return None
    return np.nanmean(np.stack(arrs), axis=0)


def _season_leadtime(year_maps, region, months, band="all"):
    leads = sorted({lt for lt_map, _ in year_maps.values() for lt in lt_map})
    xs, ys = [], []
    for lt in leads:
        vals = []
        for lt_map, _ in year_maps.values():
            ltd = lt_map.get(lt)
            if ltd is None:
                continue
            v = (band_season_mean(ltd, band, months) if band != "all"
                 else regional_season_mean(ltd, region, months))
            if np.isfinite(v):
                vals.append(v)
        if vals:
            xs.append(lt)
            ys.append(float(np.mean(vals)))
    return xs, ys


# ── Climatology (best-effort; absent → gracefully noted) ──

def _clim(cfg, region):
    """A representative ClimData for the selection, or None if unavailable."""
    root = ui_results_root()
    menu = get_menu_index()
    years = menu.values("year", forecast=cfg["forecast"], target=cfg["truth"],
                        variable=cfg["variable"], metric=cfg["metric"])
    for y in years:
        if climatology_available(cfg["truth"], y, cfg["metric"], root=root):
            try:
                return _clim_loader.load(cfg["truth"], y, cfg["metric"], root=root)
            except Exception:
                return None
    return None


# ═════════════════════════════════════════════════════════════════════════
#  Section 2 — Calibrated Forecast Metrics
# ═════════════════════════════════════════════════════════════════════════

def _series_keys(cfg) -> list:
    """``[(forecast, truth, label)]`` — one entry per curve the chart should draw.

    Exactly one of the two axes is ever plural (the config panel enforces it), so this
    is the cross product of a list with a singleton: several models on one truth, or
    one model on several truths. Labelling follows suit — the model name when models
    vary, the truth name when truths do — so the legend names the thing being compared
    rather than repeating the constant.
    """
    forecasts = cfg.get("forecasts") or [cfg["forecast"]]
    truths = cfg.get("truths") or [cfg["truth"]]
    if len(truths) > 1:
        return [(forecasts[0], tr, target_label(tr)) for tr in truths]
    return [(fc, truths[0], forecast_label(fc)) for fc in forecasts]


def _seasonal_section(menu, root, region, cfg, view, xaxis="Month") -> None:
    """Country View, or Graph View on either x-axis.

    ``xaxis`` picks what the graph runs against: ``"Month"`` is the season's
    monthly cycle at the selected lead, ``"Lead Time"`` is the season's skill
    against forecast horizon over every lead (so the lead selector does not
    apply). Ignored by Country View.
    """
    if cfg is None:
        _no_data("No models available for this variable.")
        return
    year_maps = _load_all_years(menu, cfg, root)
    if not year_maps:
        _no_data(f"No data for {forecast_label(cfg['forecast'])} · {region}.")
        return

    unit = VARIABLE_UNITS.get(cfg["variable"], "")
    clim = _clim(cfg, region) if cfg["vs_clim"] else None
    season = cfg["season"]
    months = season_months(season)
    band = cfg.get("band", "all")

    graph = view == "Graph View"
    by_lead = graph and xaxis == "Lead Time"
    # Models the selection asked for that turned out to have nothing here. Only the
    # graph views overlay several models, but the caption is shared, so this is bound
    # for every path rather than conditionally.
    dropped: list[str] = []

    if graph and not by_lead:
        # Restrict the monthly cycle to the selected season's months (all 12
        # for Yearly), in season order.
        idx = [m - 1 for m in months] if months else list(range(12))
        xlab = [_MONTH_NAMES[i] for i in idx]
        fig = go.Figure()
        drew = False
        for i, (fc, tr, name) in enumerate(_series_keys(cfg)):
            ym = (year_maps if (fc, tr) == (cfg["forecast"], cfg["truth"])
                  else _load_all_years(menu, {**cfg, "forecast": fc, "truth": tr}, root))
            monthly = _multi_year_monthly(ym, cfg["lead"], region, band) if ym else None
            if monthly is None:
                dropped.append(name)
                continue
            drew = True
            colour = SERIES_PALETTE[i % len(SERIES_PALETTE)]
            fig.add_trace(go.Scatter(
                x=xlab, y=monthly[idx], mode="lines+markers", name=name,
                line=dict(color=colour, width=3), marker=dict(size=7, color=colour),
                hovertemplate=f"{name} · %{{x}}: %{{y:.4f}}<extra></extra>"))
        if not drew:
            _no_data("No monthly data for this selection.")
            return
        if clim is not None:
            cm = np.asarray(clim.monthly_average, float)[:12]
            fig.add_trace(go.Scatter(
                x=xlab, y=cm[idx], mode="lines", name="Climatology",
                line=dict(color=NAVY, width=2, dash="dash"),
                hovertemplate="Clim %{x}: %{y:.4f}<extra></extra>"))
        _line_layout(fig, y_title=f"{metric_label(cfg['metric'])}"
                     f"{f' ({unit})' if unit else ''}", height=500)
        st.plotly_chart(fig, width="stretch")
    elif by_lead:
        # Metric error against forecast horizon for the chosen season, over
        # every lead on disk — no single-lead selection applies here.
        fig = go.Figure()
        all_x: set = set()
        for i, (fc, tr, base) in enumerate(_series_keys(cfg)):
            ym = (year_maps if (fc, tr) == (cfg["forecast"], cfg["truth"])
                  else _load_all_years(menu, {**cfg, "forecast": fc, "truth": tr}, root))
            xs, ys = _season_leadtime(ym, region, months, band) if ym else ([], [])
            if not xs:
                dropped.append(base)
                continue
            all_x |= set(xs)
            name = f"{base} · {season}"
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=name,
                line=dict(color=SERIES_PALETTE[i % len(SERIES_PALETTE)], width=3),
                marker=dict(size=7),
                hovertemplate=f"{name} · %{{x}}h · %{{y:.4f}}<extra></extra>"))
        if not all_x:
            _no_data(f"No {season} data for this selection.")
            return
        xs = sorted(all_x)
        if clim is not None:
            base = float(np.nanmean(np.asarray(clim.monthly_average, float)
                                    [[m - 1 for m in (months or range(1, 13))]]))
            if np.isfinite(base):
                fig.add_trace(go.Scatter(
                    x=[min(xs), max(xs)], y=[base, base], mode="lines",
                    name=f"Climatology · {season}",
                    line=dict(color=NAVY, width=2, dash="dash"),
                    hovertemplate=f"Clim · {season}: {base:.4f}<extra></extra>"))
        _line_layout(fig, x_title="Lead Time (hours)", xs=xs,
                     y_title=f"{metric_label(cfg['metric'])}"
                     f"{f' ({unit})' if unit else ''}", height=480)
        st.plotly_chart(fig, width="stretch")
    elif _is_station_target(year_maps):  # Country View — station gauge map
        _station_country_view(year_maps, region, months, cfg, [cfg["lead"]], band)
    else:  # Country View — per-pixel season mean at the lead
        grid = _agg_season_grid_lead(year_maps, region, months, cfg["lead"],
                                     cfg, root, band)
        if grid is None:
            _no_data("Country View needs a gridded truth source (CHIRTS or ERA5). "
                     "Use Graph View for station data.")
            return
        # Unit on the colorbar, exactly as the lead-time chart's y-axis carries it.
        # A map labelled only "CRPS" leaves the reader to guess mm/hr vs degC, and the
        # two are the same scale of number for this data.
        # Ask the same one place the Raw section asks (`map_scale_for`) rather than
        # pinning sequential here. Today this section only ever shows crps, for which
        # the two agree — but a hardcoded family stops agreeing the moment a signed
        # metric reaches it, and it would do so silently, by dropping the sign.
        _draw_map(grid, cbar=f"{metric_label(cfg['metric'])}"
                       f"{f' ({unit})' if unit else ''}",
                  scale=map_scale_for(cfg["metric"]))

    season_note = season if season != DEFAULT_SEASON else "all months"
    # A model the selection asked for but that has nothing here is NAMED, not
    # silently absent: a missing curve otherwise reads as a model with no skill.
    gap = f" · no data for {', '.join(dropped)}" if dropped else ""
    _agg_caption(list(year_maps), cfg["vs_clim"] and clim is None,
                 extra=(season_note if by_lead
                        else f"{season_note} · {cfg['lead']}h") + gap)


# ═════════════════════════════════════════════════════════════════════════
#  Per-pixel grid aggregation + map rendering
# ═════════════════════════════════════════════════════════════════════════

def _agg_annual_grid(year_maps, region, lead, band="all"):
    grids, ref = [], None
    for lt_map, rs in year_maps.values():
        if rs.get("target_kind") == "station":
            return None
        ltd = lt_map.get(lead)
        if ltd is None:
            continue
        g = annual_spatial_grid(ltd, region, band)
        if g is None or g.ndim != 2:
            continue
        ref = g
        grids.append(g.values)
    if ref is None or not grids:
        return None
    return ref.copy(data=np.nanmean(np.stack(grids), axis=0))


def _agg_season_grid_lead(year_maps, region, months, lead, cfg, root, band="all"):
    """Per-pixel season-mean grid at ONE lead, averaged over years. ``months``
    None (Yearly) uses the resident annual grid; a season reloads the monthly
    fields for that season's months and averages them."""
    if not months:
        return _agg_annual_grid(year_maps, region, lead, band)
    mset = tuple(months)
    grids, ref = [], None
    for y, (lt_map, rs) in year_maps.items():
        if rs.get("target_kind") == "station" or lead not in lt_map:
            if rs.get("target_kind") == "station":
                return None
            continue
        da = load_season_spatial(cfg["forecast"], rs["native_var"], cfg["truth"],
                                 rs["storage_metric"], y, lead, mset, root=root)
        if da is None or da.ndim != 2:
            continue
        if band != "all":
            # Band layer shares the full-domain grid, so mask by band BEFORE
            # mask_region crops to the region's bounding box.
            ltd = lt_map.get(lead)
            if ltd is not None:
                da = apply_band(da, ltd, band)
        da = mask_region(da, region)
        if ref is not None and da.shape != ref.shape:
            continue
        ref = da
        grids.append(da.values)
    if ref is None or not grids:
        return None
    return ref.copy(data=np.nanmean(np.stack(grids), axis=0))



def _draw_map(da, cbar: str, diverging: bool = False,
              scale: str | None = None) -> None:
    """Render a 2-D field as a cartopy map (ocean, coastlines, borders) —
    the same publication style as the redesign. Falls back to a notice when
    the field has no finite pixels."""
    import matplotlib.pyplot as plt

    from ui.utils.map_plot import spatial_map_figure
    if not np.isfinite(da.values).any():
        _no_data("No finite values to map for this selection.")
        return
    fig = spatial_map_figure(da.values, da, title="", cbar_label=cbar,
                             diverging=diverging, scale=scale)
    if fig is None:
        _no_data("No finite values to map for this selection.")
        return
    st.pyplot(fig, width="content")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  Station (gauge) Country View — the old platform's station plots, in the
#  new design: a Scattergeo point map + click-to-inspect per-station curves.
#  Reuses ui.components.spatial_map station helpers so behaviour matches.
# ═════════════════════════════════════════════════════════════════════════

def _is_station_target(year_maps) -> bool:
    if not year_maps:
        return False
    return next(iter(year_maps.values()))[1].get("target_kind") == "station"


def _station_country_view(year_maps, region, months, cfg, leads, band="all") -> None:
    """Gauge map (per-station values averaged over `leads` × years) + drill-down.

    ``band`` NaN-masks stations outside the chosen altitude band (via the file's
    per-station altitude_band layer); "all" keeps every gauge.
    """
    from ui.components.spatial_map import (_build_station_figure,
                                           _station_choices,
                                           _station_season_value)
    ref_da, ref_ltd, acc, order = None, None, {}, None
    for _y, (lt_map, rs) in year_maps.items():
        for lead in leads:
            ltd = lt_map.get(lead)
            if ltd is None:
                continue
            da = ltd.annual_spatial_average.squeeze()
            if "station" not in da.dims:
                _no_data("No per-station field for this selection.")
                return
            if ref_da is None:
                ref_da, ref_ltd, order = da, ltd, list(da["station"].values)
            for sid in da["station"].values:
                v = _station_season_value(ltd, sid, months)
                if np.isfinite(v):
                    acc.setdefault(int(sid), []).append(v)
    if ref_da is None:
        _no_data("No station data for this selection.")
        return
    vals = np.array([float(np.mean(acc[int(s)])) if int(s) in acc else np.nan
                     for s in order])
    da2 = apply_band(ref_da.copy(data=vals), ref_ltd, band)
    unit = VARIABLE_UNITS.get(cfg["variable"], "")
    label_lead = leads[0] if len(leads) == 1 else max(leads)
    fig = _build_station_figure(da2, forecast_label(cfg["forecast"]),
                                cfg["metric"], label_lead, region, unit)
    if fig is None:
        _no_data(f"No stations inside {region} for this selection.")
        return
    choices = _station_choices(da2, region)
    event = st.plotly_chart(
        fig, width="stretch", on_select="rerun", selection_mode="points",
        key=f"no_stmap_{cfg['forecast']}_{cfg['truth']}_{'-'.join(map(str, leads))}",
        config={"displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    st.caption(f"{len(choices)} stations · mean across years · click a gauge to inspect it")
    pts = (event or {}).get("selection", {}).get("points", []) if event else []
    if pts:
        idx = pts[0].get("point_index")
        if idx is not None and 0 <= idx < len(choices):
            sid, name = choices[idx]
            _station_detail(year_maps, months, sid, name, cfg, leads)


def _station_detail(year_maps, months, sid, name, cfg, leads) -> None:
    """Per-station lead-time + monthly curves for the clicked gauge."""
    from ui.components.spatial_map import (_station_lead_curve,
                                           _station_month_curve)
    lt_map = next(iter(year_maps.values()))[0]
    xs, lvals = _station_lead_curve(lt_map, sid, months)
    month_lead = leads[0] if leads else next(iter(lt_map), None)
    ltd = lt_map.get(month_lead)
    mmonths, mvals = _station_month_curve(ltd, sid, months) if ltd else ([], [])
    unit = VARIABLE_UNITS.get(cfg["variable"], "")
    ylab = f"{metric_label(cfg['metric'])}{f' ({unit})' if unit else ''}"
    st.markdown(f"<div style='font-size:0.9rem;font-weight:700;color:{GREEN_DARK};"
                f"margin:6px 0 2px;'>📍 {name}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        if xs:
            f1 = go.Figure(go.Scatter(
                x=xs, y=lvals, mode="lines+markers",
                line=dict(color=GREEN_DARK, width=3), marker=dict(size=7)))
            f1.update_layout(template="plotly_white", height=300,
                             margin=dict(l=8, r=8, t=28, b=8),
                             title=dict(text="vs lead time", font=dict(size=12)),
                             xaxis=dict(title="Lead (h)"), yaxis=dict(title=ylab),
                             paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(f1, width="stretch")
        else:
            st.caption("No lead-time curve for this station.")
    with c2:
        if mmonths:
            f2 = go.Figure(go.Scatter(
                x=[_MONTH_NAMES[m - 1] for m in mmonths], y=mvals,
                mode="lines+markers",
                line=dict(color=AMBER, width=3), marker=dict(size=7)))
            f2.update_layout(template="plotly_white", height=300,
                             margin=dict(l=8, r=8, t=28, b=8),
                             title=dict(text="by month", font=dict(size=12)),
                             yaxis=dict(title=ylab),
                             paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(f2, width="stretch")
        else:
            st.caption("No monthly curve for this station.")


# ═════════════════════════════════════════════════════════════════════════
#  Shared bits
# ═════════════════════════════════════════════════════════════════════════

def _line_layout(fig, *, y_title, height, x_title=None, xs=None) -> None:
    xaxis = dict(gridcolor="#F3F4F6")
    if x_title:
        xaxis.update(title=f"<b>{x_title}</b>")
    if xs is not None:
        xaxis.update(tickmode="array", tickvals=xs, ticktext=[f"{h}h" for h in xs])
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=xaxis, yaxis=dict(title=f"<b>{y_title}</b>", gridcolor="#F3F4F6"),
        # Forced on: plotly hides the legend when there is one trace, which left a
        # single curve unlabelled — and with the truth-source comparison a lone curve
        # is exactly the case where you most need to know WHICH one it is.
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white", plot_bgcolor="white")


def _render_sample_caveat(forecasts, variable) -> None:
    """Warn when a selected forecast is not scored on a full-year sample.

    Imported by score_decomposition too, so both pages word this identically.
    Renders nothing when every selection is comparable, so it is safe to call
    unconditionally after any model picker.
    """
    note = sample_caveat(forecasts, variable)
    if not note:
        return
    st.markdown(
        f"<div style='font-size:0.62rem;line-height:1.45;color:{MUTED};"
        f"border-left:2px solid {AMBER};padding:2px 0 2px 7px;margin:2px 0 6px;'>"
        f"{note}.</div>",
        unsafe_allow_html=True,
    )


def _agg_caption(years, clim_missing: bool, extra: str = "") -> None:
    years = list(years)
    single = len(years) == 1
    note = (f"Year: {years[0]}" if single
            else f"Years included: {', '.join(map(str, years))}")
    if clim_missing:
        note += " · climatology unavailable — showing raw scores"
    # Only claim aggregation when more than one year actually went in — a
    # single-year view saying "aggregated over the test period" would misread.
    scope = (f"Scores for {years[0]}" if single
             else "Scores aggregated over the test period")
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;"
        f"font-size:0.68rem;color:{MUTED};margin-top:2px;'><span>{note}</span>"
        f"<span style='border:1px solid #e2e6e2;padding:3px 10px;border-radius:6px;'>"
        f"{scope}{f' · {extra}' if extra else ''}"
        f"</span></div>",
        unsafe_allow_html=True,
    )


def _no_data(msg: str) -> None:
    st.markdown(
        f"""<div style="height:420px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;background:#F9FAFB;
            border-radius:10px;border:1px dashed #D1D5DB;color:{MUTED};">
          <div style="font-size:30px;margin-bottom:10px;">🗺️</div>
          <div style="font-size:13px;font-weight:600;text-align:center;
              max-width:420px;">{msg}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _idx(options: list, value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _inject_page_css() -> None:
    st.html(f"""
<style>
/* Region chip picker */
.st-key-no_countries {{
  background: {GREEN_DARK};
  border-radius: 12px;
  padding: 14px 16px 10px;
}}
.st-key-no_countries [data-testid="stWidgetLabel"] p {{
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 0.8rem !important;
}}
.st-key-no_countries [data-testid="stPills"] button {{
  background: {GREEN_BG};
  border: 1px solid {GREEN_DARK};
  border-radius: 6px;
  color: {GREEN_DARK};
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 10px;
  min-height: 26px;
}}
.st-key-no_countries [data-testid="stPills"] button[data-selected="true"],
.st-key-no_countries [data-testid="stPills"] button[kind="pillsActive"],
.st-key-no_countries [data-testid="stBaseButton-pillsActive"] {{
  background: {AMBER} !important;
  border-color: {AMBER} !important;
  color: {GREEN_DARK} !important;
  font-weight: 700 !important;
}}
/* Config panels */
.st-key-sp_config, .st-key-rm_config {{
  background: {GREEN_BG};
  border: 1px solid {GREEN_DARK};
  border-radius: 12px;
  padding: 14px 14px 10px;
}}
.st-key-sp_config .fc-config-head, .st-key-rm_config .fc-config-head {{
  background: {GREEN_DARK};
  color: #ffffff;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border-radius: 6px;
  padding: 5px 10px;
  margin: 10px 0 6px;
}}
.st-key-sp_config [data-testid="stWidgetLabel"] p,
.st-key-rm_config [data-testid="stWidgetLabel"] p {{
  font-size: 0.62rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em;
  color: {GREEN_DARK} !important;
}}
/* ── Section tabs (Calibrated Forecast Metrics / Raw Forecast Metrics) ────────────
   Top-level: they swap the whole page body, so they outrank the view toggles
   nested inside a section — bigger type, seated tab shape, rule beneath. */
.st-key-no_section_switch {{
  margin: 26px 0 2px;
  border-bottom: 2px solid {GREEN_BG};
}}
.st-key-no_section_switch [data-testid="stSegmentedControl"] {{ gap: 4px; }}
.st-key-no_section_switch [data-testid="stSegmentedControl"] button {{
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em;
  padding: 9px 26px !important;
  color: {MUTED} !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 3px solid transparent !important;
  border-radius: 8px 8px 0 0 !important;
}}
.st-key-no_section_switch [data-testid="stSegmentedControl"] button:hover {{
  color: {GREEN_DARK} !important;
  background: {GREEN_BG} !important;
}}
.st-key-no_section_switch [data-testid="stSegmentedControl"]
  button[aria-checked="true"] {{
  color: {GREEN_DARK} !important;
  background: {GREEN_BG} !important;
  border-bottom: 3px solid {GREEN_DARK} !important;
}}

/* Primary view toggle — solid dark-green active pill */
.st-key-no_sp_view [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
  background: {GREEN_DARK} !important;
  color: #ffffff !important;
}}
/* Secondary x-axis switch — subordinate to the view toggle it sits under, so it
   is smaller and outlined rather than filled. Same family, lower rank: two
   equally-weighted pill rows would read as two independent choices. */
.st-key-no_sp_xaxis {{ margin-top: -2px; padding-left: 2px; }}
.st-key-no_sp_xaxis [data-testid="stSegmentedControl"] button {{
  font-size: 0.72rem !important;
  padding: 1px 13px !important;
  min-height: 26px !important;
}}
.st-key-no_sp_xaxis [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
  background: {GREEN_BG} !important;
  color: {GREEN_DARK} !important;
  border-color: {GREEN_DARK} !important;
  font-weight: 700 !important;
}}
</style>
    """)

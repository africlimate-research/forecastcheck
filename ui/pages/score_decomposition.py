# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Score Decomposition — Miscalibration & Discrimination (ForecastCheck design).

Both sections are computed LIVE from the local engine, over a single Country &
Region picker at the top:

- **Miscalibration** = skill(model + IDR) - skill(raw), plotted as ONE CURVE PER
  MODEL rather than a raw and a +IDR curve for a single model. Absolute skill
  differs between models, but the amount calibration removes is directly
  comparable, so several models can share the chart. Zero is drawn as the
  reference: below it, calibration helped.
  Subtracting in that order is deliberate — these are errors, so a NEGATIVE
  difference means the calibrated model erred less, and the diverging ramp puts
  negatives on green. Green therefore reads as "+IDR is better" everywhere, on
  the curves and the maps alike.
  The CRPS of a deterministic forecast equals its MAE, so the raw side reads the
  MAE files (the `MenuIndex.resolve_storage` policy) and the +IDR side reads CRPS.

- **Discrimination** = a target model compared against a REFERENCE forecast —
  climatology, another model, or another calibrated (+IDR) model — as lead-time
  curves (Graph View) or a per-pixel difference map (Country View). Climatology
  is used only when a climatology file exists for the selection.

Everything is discovery-driven — the menus offer only what exists on disk.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.components.region_picker import render_region_picker

from ui.app_config.settings import (
    DEFAULT_REGION,
    DEFAULT_SEASON,
    MAP_COASTLINE,
    MAP_COUNTRY,
    MAP_LAND,
    MAP_OCEAN,
    REFERENCE_PALETTE,
    REFERENCE_SYMBOLS,
    REFERENCE_WIDTH,
    REGIONS,
    SEASONS,
    SERIES_PALETTE,
    SERIES_WIDTH,
    VARIABLE_UNITS,
    forecast_label,
    metric_label,
    plotly_diverging_scale,
    season_months,
    target_label,
    variable_label,
)
from ui.backend.data_loader.climatology_loader import (
    ClimatologyLoader,
    climatology_available,
)
from ui.components.chrome import (
    AMBER,
    GREEN,
    GREEN_BG,
    GREEN_DARK,
    MUTED,
    config_group,
    render_explore_cards,
    render_footer,
    render_header,
)
from ui.components.filters import _altitude_band_options
from ui.pages.national_overview import (ALL_BANDS, ALL_YEARS, _draw_map,
                                        _render_sample_caveat,
                                        _season_leadtime)
from ui.services.data_cache import get_menu_index, ui_results_root
from ui.services.loader import load_run, load_season_spatial
from ui.utils.geo import apply_band, band_season_mean, regional_season_mean
from ui.utils.regions import mask_region

_CLIM = "__climatology__"      # sentinel reference option
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def render() -> None:
    render_header(active="score_decomposition")
    _inject_page_css()

    st.markdown(
        f"""
        <div style="padding:30px 6px 6px;max-width:760px;">
          <h1 style="margin:0 0 8px;">Score Decomposition</h1>
          <p style="color:{MUTED};font-size:0.88rem;line-height:1.7;margin:0;">
            Investigate the miscalibration error of your model and its improved
            discrimination in comparison to a reference forecast (either
            climatology or another calibrated forecast product).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    region = _render_region_picker()
    if _render_section_switch() == "Discrimination":
        _discrimination_section(region)
    else:
        _miscalibration_section(region)
    render_explore_cards("national_overview", "detailed_metrics")
    render_footer()


def _render_section_switch() -> str:
    """Top-level tabs for the page's two decompositions.

    A segmented control rather than ``st.tabs``: Streamlit executes the body of
    every tab on each rerun, and each of these sections loads NetCDF, so tabs
    would pay for the hidden section on every interaction. This renders only the
    section on screen — and stacking them, as the page used to, meant scrolling
    past a whole config panel and chart to reach the second one.
    """
    with st.container(key="sd_section_switch"):
        picked = st.segmented_control(
            "section", ["Miscalibration", "Discrimination"],
            default="Miscalibration", key="sd_section",
            label_visibility="collapsed")
    return picked or "Miscalibration"


# ═════════════════════════════════════════════════════════════════════════
#  Country & Region picker (top — shared by both sections)
# ═════════════════════════════════════════════════════════════════════════

def _render_region_picker() -> str:
    """The shared Country & Region selector (see ui.components.region_picker)."""
    return render_region_picker("sd")


# ═════════════════════════════════════════════════════════════════════════
#  Shared loading — years resolved via resolve_storage (raw crps → mae)
# ═════════════════════════════════════════════════════════════════════════

def _map_stepper(state_key: str, models: list) -> str:
    """Prev / Next through ``models`` for a pairwise map; returns the current one.

    A difference map shows two fields, so a multi-model selection cannot go on one
    map. Rather than silently mapping the first, step through them. The index is
    clamped on every run, so deselecting models cannot leave it out of range, and
    it wraps so Next is never a dead end.
    """
    if not models:
        return ""
    i = int(st.session_state.get(state_key, 0)) % len(models)
    st.session_state[state_key] = i
    if len(models) > 1:
        prev_c, next_c, label_c = st.columns([1, 1, 6], vertical_alignment="center")
        with prev_c:
            if st.button("‹ Prev", key=f"{state_key}_prev", type="tertiary"):
                st.session_state[state_key] = (i - 1) % len(models)
                st.rerun()
        with next_c:
            if st.button("Next ›", key=f"{state_key}_next", type="tertiary"):
                st.session_state[state_key] = (i + 1) % len(models)
                st.rerun()
        with label_c:
            st.markdown(
                f"<div style='font-size:0.68rem;color:{MUTED};'>"
                f"Map <b>{i + 1} of {len(models)}</b>: "
                f"<b>{forecast_label(models[i])}</b> — a difference map takes two "
                f"fields, so one model at a time. Graph View shows them all.</div>",
                unsafe_allow_html=True)
    return models[i]


def _scoped_mean(ltd, region, months, band="all") -> float:
    """Season mean for a slice: the band mean when a band is picked, else the
    regional mean. Bands are pipeline-computed (DEM for gridded, gauge
    elevation for stations), so a band is read straight off the file."""
    return (band_season_mean(ltd, band, months) if band != "all"
            else regional_season_mean(ltd, region, months))


def _band_ltd(forecast, rs, truth, y, lead, root):
    """One (run, lead) slice, loaded ONLY for its ``altitude_band`` layer.

    The gridded map helpers read their fields through ``load_season_spatial``,
    which returns a bare DataArray with no band layer attached — so to mask by
    elevation they need the summary slice as well.
    """
    lt = load_run(forecast, rs["native_var"], truth, rs["storage_metric"],
                  y, [lead], root=root)
    return lt.get(lead)


def _band_station_ids(ltd, band):
    """Station ids inside ``band``, or None when no band filtering applies.

    Derived by running the field through :func:`apply_band` — the same index-layer
    and label logic the rest of the app uses — and keeping the stations that survive.
    Re-deriving band membership here would be a second source of truth.
    """
    if band == "all" or ltd is None:
        return None
    da = ltd.annual_spatial_average.squeeze()
    if "station" not in da.dims:
        return None
    masked = apply_band(da, ltd, band)
    if masked is da:          # no layer, or an unknown label -> apply_band no-op
        return None
    vals = np.asarray(masked.values).ravel()
    sids = np.asarray(masked["station"].values).ravel()
    return {int(sid) for sid, v in zip(sids, vals) if np.isfinite(v)}


def _years_in_scope(menu, forecast, truth, variable, year=None):
    """Discovered years for a forecast, narrowed to ``year`` when one is picked.

    The YEAR selector's single choke point — every Discrimination helper walks
    years through here, so one selector governs the curves, the maps and the
    per-gauge drill-down alike. ``year`` None = every year (the ALL_YEARS mean).
    """
    years = menu.values("year", forecast=forecast, target=truth, variable=variable)
    return [y for y in years if year is None or y == year]


def _load_years(menu, forecast, truth, variable, metric, root, year=None):
    """{year: (lt_map, resolve_dict)} for a forecast, honouring the storage
    policy (raw CRPS falls back to MAE; -idr never scored by a point metric).
    Years are NOT pre-filtered by the metric facet so the fallback works."""
    out: dict = {}
    for y in _years_in_scope(menu, forecast, truth, variable, year):
        rs = menu.resolve_storage(forecast=forecast, target=truth, metric=metric,
                                  year=y, variable=variable)
        if rs is None:
            continue
        lt_map = load_run(forecast, rs["native_var"], truth, rs["storage_metric"],
                          y, rs["leads"], root=root)
        if lt_map:
            out[y] = (lt_map, rs)
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Miscalibration = skill(model + IDR) - skill(raw)   [negative = +IDR better]
# ═════════════════════════════════════════════════════════════════════════

def _paired_models(menu) -> list[str]:
    forecasts = set(menu.values("forecast"))
    return sorted(t for t in forecasts
                  if not t.endswith("-idr") and f"{t}-idr" in forecasts)


def _miscalibration_section(region: str) -> None:
    # No <h2> — the section tab above already names it; repeating it here just
    # pushed the chart further down the page.
    with st.expander("More on Miscalibration"):
        st.markdown(
            "Miscalibration measures the part of the forecast error that comes "
            "from systematic disagreement between predicted probabilities and "
            "observed frequencies. Each curve is one model's "
            "**skill(IDR) − skill(raw)** — the error its calibration removed — so "
            "**below zero means calibration helped**, and several models can be "
            "compared even when their absolute skill differs. On the maps the same "
            "sign convention holds: **green means the calibrated model erred "
            "less**, amber that it erred more. For a deterministic model (e.g. "
            "GraphCast) the CRPS equals the MAE, so the raw side uses the model's "
            "MAE files."
        )

    menu = get_menu_index()
    bases = _paired_models(menu)
    if not bases:
        _no_data_card("Requires a base model and its IDR-calibrated counterpart "
                      "on disk for at least one common lead time.")
        return

    root = ui_results_root()
    # Read the view before the config renders: the lead-time curve spans every
    # lead, so only Country View needs a lead. The widget has already written
    # its new value to session_state by this point in the rerun.
    view = st.session_state.get("sd_mc_view") or "Graph View"

    col_cfg, col_view = st.columns([1, 3], gap="medium")
    with col_cfg:
        with st.container(key="sd_mc_config"):
            _config_title()
            with config_group("sd_mc", "ft", "Forecast &amp; Truth"):
                # Several models at once: the chart plots each one's
                # (model+IDR) − model difference, which is comparable across
                # models even when their absolute skill levels are not.
                models = st.multiselect("MODELS", bases, default=bases[:1],
                                        format_func=forecast_label,
                                        key="sd_mc_models")
                if not models:
                    st.info("Pick at least one model.")
                    return
                # Facets must hold for EVERY selected model, else a chart could
                # mix a variable one model lacks.
                variables = sorted(set.intersection(
                    *(set(menu.values("variable", forecast=m)) for m in models)))
                if not variables:
                    st.info("The selected models share no variable.")
                    return
                variable = st.selectbox("VARIABLE", variables,
                                        format_func=variable_label, key="sd_mc_var")
                _render_sample_caveat(models, variable)
                truths = sorted(set.intersection(
                    *(set(menu.values("target", forecast=m, variable=variable))
                      for m in models)))
                if not truths:
                    st.info("The selected models share no truth source.")
                    return
                # Same rule as the other pages: several models on one truth compares
                # forecasts; one model on several truths compares references. Never
                # both, so each chart answers exactly one question.
                #
                # GRAPH VIEW ONLY. Country View draws a per-pixel difference map,
                # which can show exactly one field — offering a multiselect there
                # would silently ignore every truth source but the first, which is
                # worse than not offering it. `view` was read from session_state
                # above, before this panel renders, for exactly this reason.
                if view == "Graph View" and len(models) == 1 and len(truths) > 1:
                    picked_truths = st.multiselect(
                        "TRUTH SOURCES", truths, default=truths,
                        format_func=target_label, key="sd_mc_truths")
                    if not picked_truths:
                        st.info("Pick at least one truth source.")
                        return
                    truth = picked_truths[0]
                else:
                    truth = st.selectbox("TRUTH SOURCE", truths,
                                         format_func=target_label, key="sd_mc_truth")
                    picked_truths = [truth]
            with config_group("sd_mc", "sc", "Score"):
                # Fixed to CRPS: the +IDR side is scored by CRPS and the raw
                # side by MAE (== CRPS for a deterministic forecast), so both
                # curves live on one CRPS axis. The legend flags the raw = MAE.
                st.selectbox("METRIC", ["CRPS"], key="sd_mc_metric")
            with config_group("sd_mc", "co", "Conditioning"):
                # Paired years only — a year needs BOTH the raw and the +IDR run
                # for a miscalibration gap to exist at all.
                yr_sets = [set(_mcb_years(menu, m, f"{m}-idr", variable, truth))
                           for m in models]
                all_years = sorted(set.intersection(*yr_sets), reverse=True) \
                    if yr_sets else []
                year_pick = st.selectbox("YEAR", [ALL_YEARS] + all_years,
                                         key="sd_mc_year")
                year = None if year_pick == ALL_YEARS else year_pick
                season = st.selectbox("SEASON", list(SEASONS),
                                      index=_idx(list(SEASONS), DEFAULT_SEASON),
                                      key="sd_mc_season")
                # Bands come from the DEM (gridded) or each gauge's elevation
                # (stations); the control hides itself unless the region is the
                # deployment country and the files are band-aware.
                band = ALL_BANDS
                mc_bands = _altitude_band_options(
                    menu, region, variable, truth, "crps",
                    (all_years or [None])[0], [f"{m}-idr" for m in models])
                if mc_bands:
                    band = st.selectbox("ALTITUDE", [ALL_BANDS] + list(mc_bands),
                                        key="sd_mc_band")
                band = "all" if band == ALL_BANDS else band
                lead = None
                if view == "Country View":
                    # Only the map pins a single lead; the curve spans them all,
                    # so Graph View shows no lead selector at all.
                    leads = sorted({lt for m in models for lt in
                                    _mcb_leads(menu, m, f"{m}-idr", variable,
                                               truth, year)})
                    if not leads:
                        st.info("No paired raw/+IDR lead times on disk for this "
                                "selection.")
                        return
                    lead = st.selectbox("LEAD TIME", leads,
                                        index=len(leads) // 2,
                                        format_func=lambda h: f"{h}h",
                                        key="sd_mc_lead")

    with col_view:
        view = st.segmented_control(
            "mc view", ["Graph View", "Country View"], default="Graph View",
            key="sd_mc_view", label_visibility="collapsed") or "Graph View"
        months = season_months(season)
        if view == "Graph View":
            fig, missing, no_band = _mcb_figure(menu, models, variable, truth,
                                                region, months, root, year, band,
                                                truths=picked_truths)
            if fig is None:
                _no_data_card(f"No paired raw/+IDR data for {region} · {season} "
                              f"at any common lead time.")
            else:
                st.plotly_chart(fig, width="stretch")
                gap = ""
                if missing:
                    gap += ("; no paired data for "
                            + ", ".join(forecast_label(m) for m in missing))
                if no_band:
                    gap += ("; " + ", ".join(forecast_label(m) for m in no_band)
                            + f" carries no {band} m layer, so it is not plotted"
                            " (regenerate its diagnostics to include altitude"
                            " bands)")
                st.markdown(
                    f"<div style='font-size:0.68rem;color:{MUTED};margin-top:2px;'>"
                    f"Each curve is skill(IDR) − skill(raw · MAE) for one model — the "
                    f"calibration removed. <b>Below zero means calibration "
                    f"helped.</b> Regional mean over {region} · {season}, "
                    f"{_year_note(year)}{gap}.</div>",
                    unsafe_allow_html=True)
        else:  # Country View — (IDR − raw) map at the selected lead
            # One map per model, stepped through rather than truncated to the
            # first: a difference map takes exactly two fields, so the several
            # models Graph View overlays cannot share one map.
            base = _map_stepper("sd_mc_map_i", models)
            idr = f"{base}-idr"
            grid = _mcb_country_grid(menu, base, idr, variable, truth, region,
                                     months, root, lead, year, band)
            unit = VARIABLE_UNITS.get(variable, "")
            note = (f"skill(IDR · CRPS) − skill(raw · MAE) · {region} · "
                    f"{season} · {lead}h, {_year_note(year)}")
            if grid is not None:  # gridded truth → per-pixel map
                _draw_map(grid, cbar=f"Miscalibration (IDR − raw)"
                               f"{f' ({unit})' if unit else ''}",
                          diverging=True)
                st.markdown(
                    f"<div style='font-size:0.68rem;color:{MUTED};margin-top:2px;'>"
                    f"Per-pixel miscalibration: {note}. Green = IDR erred less."
                    f"</div>",
                    unsafe_allow_html=True)
            else:  # station truth → per-gauge miscalibration map + drill-down
                fig, da2 = _mcb_station_map(menu, base, idr, variable, truth,
                                            region, months, root, lead, year,
                                            band)
                if fig is None:
                    _no_data_card(f"No miscalibration map for {forecast_label(base)} · "
                                  f"{region} · {season} · {lead}h.")
                else:
                    from ui.components.spatial_map import _station_choices
                    choices = _station_choices(da2, region)
                    event = st.plotly_chart(
                        fig, width="stretch", on_select="rerun",
                        selection_mode="points", key="sd_mc_stmap",
                        config={"displayModeBar": True,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
                    st.markdown(
                        f"<div style='font-size:0.68rem;color:{MUTED};margin-top:2px;'>"
                        f"Per-station miscalibration: {note} · green = IDR erred "
                        f"less · click a gauge to "
                        f"inspect its raw-vs-IDR curves.</div>",
                        unsafe_allow_html=True)
                    pts = ((event or {}).get("selection", {}).get("points", [])
                           if event else [])
                    if pts:
                        i = pts[0].get("point_index")
                        if i is not None and 0 <= i < len(choices):
                            s_id, s_name = choices[i]
                            _mcb_station_detail(menu, base, idr, variable, truth,
                                                region, months, root, s_id,
                                                s_name, lead, year)


def _year_note(year) -> str:
    """Caption fragment — never claim an average when one year is shown."""
    return f"for {year}" if year else "averaged over all years"


def _mcb_pairs(menu, base, idr, variable, truth, year=None):
    """Yield (year, resolve_raw, resolve_cal, common_leads) per paired year.

    A year only qualifies when the base model AND its +IDR twin are both on disk
    for it — the miscalibration gap is the difference between the two, so one
    without the other is not a partial answer, it is no answer. ``year`` None
    walks every paired year (the ALL_YEARS mean); a value restricts to that one.
    """
    for y in menu.values("year", forecast=base, target=truth, variable=variable):
        if year is not None and y != year:
            continue
        raw = menu.resolve_storage(forecast=base, target=truth, metric="crps",
                                   year=y, variable=variable)
        cal = menu.resolve_storage(forecast=idr, target=truth, metric="crps",
                                   year=y, variable=variable)
        if raw is None or cal is None:
            continue
        common = sorted(set(raw["leads"]) & set(cal["leads"]))
        if common:
            yield y, raw, cal, common


def _mcb_years(menu, base, idr, variable, truth) -> list[str]:
    """Years offered by the YEAR selector — paired years only, newest first."""
    return sorted({y for y, _r, _c, _l
                   in _mcb_pairs(menu, base, idr, variable, truth)}, reverse=True)


def _mcb_leads(menu, base, idr, variable, truth, year=None):
    """Union of common (raw & +IDR) lead times across the years in scope."""
    leads: set[int] = set()
    for _y, _raw, _cal, common in _mcb_pairs(menu, base, idr, variable, truth, year):
        leads.update(common)
    return sorted(leads)


def _mcb_delta_series(menu, base, variable, truth, region, months, root,
                      year=None, band="all"):
    """``(leads, deltas)`` for one model: skill(IDR) − skill(raw) per lead.

    Both sides are scored on one axis — the +IDR run by CRPS, the raw run by MAE,
    which IS its CRPS because a deterministic forecast's CRPS equals its absolute
    error. The difference is therefore what calibration removed, in the
    variable's own units. Negative = the calibrated run erred less.

    Averaged over the years in scope, and the difference is formed per (year,
    lead) BEFORE averaging, so a year present for one side only cannot skew it.
    """
    idr = f"{base}-idr"
    by_lead: dict[int, list] = {}
    for y, raw, cal, common in _mcb_pairs(menu, base, idr, variable, truth, year):
        rmap = load_run(base, raw["native_var"], truth, raw["storage_metric"],
                        y, common, root=root)
        cmap = load_run(idr, cal["native_var"], truth, cal["storage_metric"],
                        y, common, root=root)
        for lt in common:
            if lt not in rmap or lt not in cmap:
                continue
            rv = _scoped_mean(rmap[lt], region, months, band)
            cv = _scoped_mean(cmap[lt], region, months, band)
            if np.isfinite(rv) and np.isfinite(cv):
                by_lead.setdefault(lt, []).append(cv - rv)
    leads = sorted(by_lead)
    return leads, [float(np.mean(by_lead[lt])) for lt in leads]


def _mcb_figure(menu, bases, variable, truth, region, months, root, year=None,
                band="all", truths=None):
    """One curve per model — or per TRUTH SOURCE — of (model+IDR) − model vs lead.

    ``truths`` longer than one switches the comparison axis: one model scored
    against several references, rather than several models against one. Exactly one
    of the two is ever plural (the config panel enforces it), so this stays one curve
    per thing being compared, and the legend names that thing.

    Plotting the difference rather than the two raw/+IDR curves is what lets
    several models share the chart: their absolute skill levels differ, but the
    amount calibration removes is directly comparable. Zero is drawn as the
    reference — below it, calibration helped.
    """
    unit = VARIABLE_UNITS.get(variable, "")
    fig = go.Figure()
    all_leads: set = set()
    plotted, missing, no_band = [], [], []
    # (model, truth, legend name) per curve — the cross product of a list with a
    # singleton, so the legend names whichever axis actually varies.
    truths = list(truths or [truth])
    if len(truths) > 1:
        pairs = [(bases[0], tr, f"{target_label(tr)} (IDR − raw)") for tr in truths]
    else:
        pairs = [(b, truths[0], f"{forecast_label(b)} (IDR − raw)") for b in bases]
    for i, (base, truth, name) in enumerate(pairs):
        leads, deltas = _mcb_delta_series(menu, base, variable, truth, region,
                                          months, root, year, band)
        if not leads:
            # Two different failure modes, and a curve must never vanish without
            # saying which: no paired raw/+IDR files at all, or files that carry
            # no layer for THIS band (diagnostics written before altitude
            # stratification land here — regenerate that model/variable/year).
            if band != "all" and _mcb_delta_series(
                    menu, base, variable, truth, region, months, root,
                    year, "all")[0]:
                no_band.append(base)
            else:
                missing.append(base)
            continue
        all_leads |= set(leads)
        plotted.append(base)
        # "GraphCast + IDR − raw" parsed as GraphCast plus (IDR − raw).
        # `name` comes from `pairs` above: the model when models vary, the truth
        # source when truths do. The difference stays bracketed either way, so the
        # curve names the quantity it plots.
        fig.add_trace(go.Scatter(
            x=leads, y=deltas, mode="lines+markers", name=name,
            line=dict(color=SERIES_PALETTE[i % len(SERIES_PALETTE)], width=3),
            marker=dict(size=7),
            hovertemplate=f"{name} · %{{x}}h · %{{y:+.4f}}<extra></extra>"))
    if not plotted:
        return None, missing, no_band
    # Zero = calibration changed nothing; the sign of the curve is the whole read.
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
    # Unit after a comma, not in a second pair of parens: "(IDR − raw) (mm/hr)"
    # read as one expression, which is the confusion this naming just fixed.
    _lt_layout(fig, sorted(all_leads),
               f"Δ CRPS (IDR − raw){f', {unit}' if unit else ''}")
    return fig, missing, no_band


def _mcb_country_grid(menu, base, idr, variable, truth, region, months, root, lead,
                      year=None, band="all"):
    """Per-pixel miscalibration map (IDR − raw) at one lead, averaged over years."""
    mset = tuple(months) if months else tuple(range(1, 13))
    diffs, ref = [], None
    for y, raw, cal, common in _mcb_pairs(menu, base, idr, variable, truth, year):
        if lead not in common:
            continue
        rg = load_season_spatial(base, raw["native_var"], truth,
                                 raw["storage_metric"], y, lead, mset, root=root)
        cg = load_season_spatial(idr, cal["native_var"], truth,
                                 cal["storage_metric"], y, lead, mset, root=root)
        if rg is None or cg is None or rg.ndim != 2 or cg.ndim != 2:
            continue
        if band != "all":
            # The band layer shares the FULL-domain grid, so mask by elevation
            # before mask_region crops to the region's bounding box.
            rl = _band_ltd(base, raw, truth, y, lead, root)
            cl = _band_ltd(idr, cal, truth, y, lead, root)
            if rl is not None:
                rg = apply_band(rg, rl, band)
            if cl is not None:
                cg = apply_band(cg, cl, band)
        rg = mask_region(rg, region)
        cg = mask_region(cg, region)
        if rg.shape != cg.shape or (ref is not None and cg.shape != ref.shape):
            continue
        ref = cg
        # +IDR minus raw, NOT raw minus +IDR: the diverging ramp runs green
        # (low) -> amber (high), so a negative difference — IDR's error below
        # the raw model's — reads green for "calibration helped".
        diffs.append(cg.values - rg.values)
    if ref is None or not diffs:
        return None
    return ref.copy(data=np.nanmean(np.stack(diffs), axis=0))


def _mcb_station_map(menu, base, idr, variable, truth, region, months, root, lead,
                     year=None, band="all"):
    """Per-GAUGE miscalibration map (raw MAE − IDR CRPS) at one lead, averaged
    over years — the station counterpart of the gridded map."""
    from ui.components.spatial_map import _station_season_value
    ref_da, acc, order = None, {}, None
    for y, raw, cal, common in _mcb_pairs(menu, base, idr, variable, truth, year):
        if lead not in common:
            continue
        rmap = load_run(base, raw["native_var"], truth, raw["storage_metric"],
                        y, [lead], root=root)
        cmap = load_run(idr, cal["native_var"], truth, cal["storage_metric"],
                        y, [lead], root=root)
        rltd, cltd = rmap.get(lead), cmap.get(lead)
        if rltd is None or cltd is None:
            continue
        rda = rltd.annual_spatial_average.squeeze()
        if "station" not in rda.dims:
            return None  # not a station target — caller uses the gridded map
        if ref_da is None:
            ref_da, order = rda, list(rda["station"].values)
        in_band = _band_station_ids(rltd, band)
        for sid in order:
            if in_band is not None and int(sid) not in in_band:
                continue
            rv = _station_season_value(rltd, sid, months)
            cv = _station_season_value(cltd, sid, months)
            if np.isfinite(rv) and np.isfinite(cv):
                # (+IDR - raw), matching the gridded map above.
                acc.setdefault(int(sid), []).append(cv - rv)
    if ref_da is None or not acc:
        return None, None
    vals = np.array([float(np.mean(acc[int(s)])) if int(s) in acc else np.nan
                     for s in order])
    da2 = ref_da.copy(data=vals)
    unit = VARIABLE_UNITS.get(variable, "")
    fig = _station_diverging_figure(
        da2, region, f"Miscalibration (IDR − raw){f' ({unit})' if unit else ''}")
    return fig, da2


def _mcb_station_detail(menu, base, idr, variable, truth, region, months, root,
                        sid, name, lead, year=None) -> None:
    """One gauge's raw-MAE vs +IDR-CRPS curves — the station-level miscalibration."""
    from ui.components.spatial_map import _station_month_curve, _station_season_value
    raw_by, cal_by = {}, {}
    rltd_lead = cltd_lead = None
    for y, raw, cal, common in _mcb_pairs(menu, base, idr, variable, truth, year):
        rmap = load_run(base, raw["native_var"], truth, raw["storage_metric"],
                        y, common, root=root)
        cmap = load_run(idr, cal["native_var"], truth, cal["storage_metric"],
                        y, common, root=root)
        for lt in common:
            rl, cl = rmap.get(lt), cmap.get(lt)
            if rl is not None:
                v = _station_season_value(rl, sid, months)
                if np.isfinite(v):
                    raw_by.setdefault(lt, []).append(v)
            if cl is not None:
                v = _station_season_value(cl, sid, months)
                if np.isfinite(v):
                    cal_by.setdefault(lt, []).append(v)
        if lead in rmap and rltd_lead is None:
            rltd_lead, cltd_lead = rmap.get(lead), cmap.get(lead)
    leads = sorted(set(raw_by) & set(cal_by))
    unit = VARIABLE_UNITS.get(variable, "")
    ylab = f"CRPS{f' ({unit})' if unit else ''}"
    st.markdown(f"<div style='font-size:0.9rem;font-weight:700;color:{GREEN_DARK};"
                f"margin:6px 0 2px;'>📍 {name}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        if leads:
            f1 = go.Figure()
            f1.add_trace(go.Scatter(
                x=leads, y=[float(np.mean(raw_by[lt])) for lt in leads],
                mode="lines+markers", name="raw · MAE",
                line=dict(color=AMBER, width=3), marker=dict(size=6)))
            f1.add_trace(go.Scatter(
                x=leads, y=[float(np.mean(cal_by[lt])) for lt in leads],
                mode="lines+markers", name="IDR · CRPS",
                line=dict(color=GREEN, width=3), marker=dict(size=6)))
            f1.update_layout(template="plotly_white", height=300,
                             margin=dict(l=8, r=8, t=28, b=8),
                             title=dict(text="vs lead time", font=dict(size=12)),
                             xaxis=dict(title="Lead (h)"), yaxis=dict(title=ylab),
                             legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                             paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(f1, width="stretch")
        else:
            st.caption("No lead-time curves for this station.")
    with c2:
        rm = _station_month_curve(rltd_lead, sid, months) if rltd_lead else ([], [])
        cm = _station_month_curve(cltd_lead, sid, months) if cltd_lead else ([], [])
        if rm[0] or cm[0]:
            f2 = go.Figure()
            if rm[0]:
                f2.add_trace(go.Scatter(
                    x=[_MONTH_NAMES[m - 1] for m in rm[0]], y=rm[1],
                    mode="lines+markers", name="raw · MAE",
                    line=dict(color=AMBER, width=3), marker=dict(size=6)))
            if cm[0]:
                f2.add_trace(go.Scatter(
                    x=[_MONTH_NAMES[m - 1] for m in cm[0]], y=cm[1],
                    mode="lines+markers", name="IDR · CRPS",
                    line=dict(color=GREEN, width=3), marker=dict(size=6)))
            f2.update_layout(template="plotly_white", height=300,
                             margin=dict(l=8, r=8, t=28, b=8),
                             title=dict(text=f"by month · {lead}h", font=dict(size=12)),
                             yaxis=dict(title=ylab),
                             legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                             paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(f2, width="stretch")
        else:
            st.caption("No monthly curves for this station.")


def _station_diverging_figure(da, region, cbar_title):
    """A per-gauge Scattergeo on a diverging scale centred on 0 (for signed
    fields like miscalibration), matching the station-map look."""
    from ui.components.spatial_map import (
        _add_region_boundary,
        _station_keep_mask,
        _station_lonlat,
    )
    da = da.squeeze()
    lons, lats = _station_lonlat(da)
    vals = np.asarray(da.values, dtype=float)
    keep = _station_keep_mask(da, region)
    if not keep.any():
        return None
    lats, lons, vals = lats[keep], lons[keep], vals[keep]
    names = np.asarray(da["name"].values)[keep] if "name" in da.coords else None
    finite = vals[np.isfinite(vals)]
    m = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    custom = [[str(names[i]) if names is not None else ""] for i in range(len(vals))]
    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, mode="markers", customdata=custom,
        marker=dict(color=vals, colorscale=plotly_diverging_scale(),
                    cmin=-m, cmax=m, size=9,
                    line=dict(width=0.5, color="#374151"),
                    colorbar=dict(title=dict(text=f"<b>{cbar_title}</b>", side="right",
                                             font=dict(size=10, color="#374151")),
                                  thickness=10, len=0.8,
                                  tickfont=dict(size=9, color="#6B7280"))),
        hovertemplate=("<b>%{customdata[0]}</b><br>" + cbar_title +
                       ": %{marker.color:.4f}<br>Lat %{lat:.2f}°  Lon %{lon:.2f}°"
                       "<extra></extra>")))
    lat_min, lat_max = float(lats.min()) - 0.5, float(lats.max()) + 0.5
    lon_min, lon_max = float(lons.min()) - 0.5, float(lons.max()) + 0.5
    fig.update_layout(
        geo=dict(showocean=True, oceancolor=MAP_OCEAN, showland=True,
                 landcolor=MAP_LAND, showcoastlines=True, coastlinecolor=MAP_COASTLINE,
                 coastlinewidth=1.2, showcountries=True, countrycolor=MAP_COUNTRY,
                 countrywidth=0.7, showframe=True, framecolor="#9CA3AF",
                 projection_type="equirectangular",
                 lataxis_range=[lat_min, lat_max], lonaxis_range=[lon_min, lon_max],
                 bgcolor="rgba(0,0,0,0)"),
        template="plotly_white", height=500, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="white", plot_bgcolor="white")
    try:
        _add_region_boundary(fig, region)
    except Exception:  # noqa: BLE001
        pass
    return fig


# ═════════════════════════════════════════════════════════════════════════
#  Discrimination — target vs reference (climatology / another model)
# ═════════════════════════════════════════════════════════════════════════

def _discrimination_section(region: str) -> None:
    # See _miscalibration_section: the tab carries the name.
    with st.expander("More on Discrimination"):
        st.markdown(
            "Discrimination measures a forecast's ability to distinguish between "
            "different outcomes — how much better it separates events from "
            "non-events than a reference forecast does. Pick a **reference**: "
            "climatology, another model, or another calibrated (IDR) model. "
            "Graph View overlays the two lead-time curves; Country View maps their "
            "per-pixel difference (target − reference)."
        )

    menu = get_menu_index()
    root = ui_results_root()
    # Read the view before the config renders — only the map pins a single lead.
    view = st.session_state.get("sd_dc_view") or "Graph View"

    col_cfg, col_view = st.columns([1, 3], gap="medium")
    with col_cfg:
        with st.container(key="sd_dc_config"):
            _config_title()
            with config_group("sd_dc", "ft", "Forecast &amp; Truth"):
                variables = menu.values("variable")
                if not variables:
                    st.info("No summaries on disk.")
                    return
                variable = st.selectbox("VARIABLE", variables,
                                        format_func=variable_label, key="sd_dc_var")
                truths = menu.values("target", variable=variable)
                truth = st.selectbox("TRUTH SOURCE", truths,
                                     format_func=target_label, key="sd_dc_truth")
                all_forecasts = menu.values("forecast", variable=variable,
                                            target=truth)
                # Open model set, like the old dashboard's Models Comparison:
                # ANY discovered forecast can be plotted, raw or calibrated. The
                # storage policy keeps the axis honest — a raw model has no CRPS
                # files, so it is read from mae, which IS its CRPS. The legend
                # tags each curve with the metric it actually came from.
                if not all_forecasts:
                    st.info("No models on disk for this selection.")
                    return
                model = st.selectbox("MODEL", all_forecasts,
                                     format_func=forecast_label,
                                     key="sd_dc_model")
                # Reference = climatology, or any OTHER discovered forecast (raw
                # or calibrated). One at a time: both views compare exactly two
                # forecasts, so a single pair keeps the chart and the map on the
                # same comparison.
                ref_opts = [_CLIM] + [f for f in all_forecasts if f != model]
                reference = st.selectbox(
                    "REFERENCE", ref_opts,
                    format_func=lambda t: "Climatology" if t == _CLIM
                    else forecast_label(t),
                    key="sd_dc_ref")
                _render_sample_caveat([model, reference], variable)

            with config_group("sd_dc", "sc", "Score"):
                # CRPS only. Raw models have no CRPS files, so they are scored by
                # MAE (== CRPS for a deterministic forecast) — flagged per curve.
                st.selectbox("METRIC", ["CRPS"], key="sd_dc_metric")
                metric = "crps"

            with config_group("sd_dc", "co", "Conditioning"):
                common_years = sorted(_years_in_scope(menu, model, truth, variable),
                                      reverse=True)
                year_pick = st.selectbox("YEAR", [ALL_YEARS] + common_years,
                                         key="sd_dc_year")
                year = None if year_pick == ALL_YEARS else year_pick
                season = st.selectbox("SEASON", list(SEASONS),
                                      index=_idx(list(SEASONS), DEFAULT_SEASON),
                                      key="sd_dc_season")
                band = ALL_BANDS
                dc_bands = _altitude_band_options(menu, region, variable, truth,
                                                  metric,
                                                  (common_years or [None])[0],
                                                  [model])
                if dc_bands:
                    band = st.selectbox("ALTITUDE", [ALL_BANDS] + list(dc_bands),
                                        key="sd_dc_band")
                band = "all" if band == ALL_BANDS else band
                lead = None
                if view == "Country View":
                    # Only the map pins a lead; the curves span every lead.
                    dc_leads = _forecast_leads(menu, model, variable, truth,
                                               metric, year)
                    lead = st.selectbox(
                        "LEAD TIME", dc_leads, index=len(dc_leads) // 2,
                        format_func=lambda h: f"{h}h",
                        key="sd_dc_lead") if dc_leads else None

    with col_view:
        view = st.segmented_control(
            "dc view", ["Graph View", "Country View"], default="Graph View",
            key="sd_dc_view", label_visibility="collapsed") or "Graph View"
        _render_discrimination(menu, root, region, variable, model, truth, metric,
                               reference, season, view, lead, year, band)


def _render_discrimination(menu, root, region, variable, model, truth, metric,
                           reference, season, view, lead=None, year=None,
                           band="all") -> None:
    """One model against one reference — a curve pair, or their difference map.

    Both views compare exactly two forecasts, so the section takes a single pair.
    The model may be raw or calibrated: a raw run has no CRPS files and is read
    from mae, which IS its CRPS for a deterministic forecast, so each curve is
    tagged with the metric it actually came from.
    """
    months = season_months(season)
    unit = VARIABLE_UNITS.get(variable, "")
    ym_t = _load_years(menu, model, truth, variable, metric, root, year)
    if not ym_t:
        _no_data_card(f"No {metric_label(metric)} data for "
                      f"{forecast_label(model)} · {region}.")
        return

    if view == "Graph View":
        xs, ys = _season_leadtime(ym_t, region, months, band)
        if not xs:
            _no_data_card(f"No {season} data for {forecast_label(model)}.")
            return
        t_name = f"{forecast_label(model)} · {_score_tag(model)}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=t_name,
            line=dict(color=SERIES_PALETTE[0], width=SERIES_WIDTH),
            marker=dict(size=7, symbol="circle"),
            hovertemplate=f"{t_name} · %{{x}}h · %{{y:.4f}}<extra></extra>"))
        # The reference is warm and SOLID — a first-class series, not an
        # annotation. Amber sits only ΔE 6.9 from the model green under
        # protan/deutan, so shape carries the distinction: a diamond marker and a
        # heavier stroke against the model's circles.
        note = ""
        if reference == _CLIM:
            base = _clim_baseline(truth, metric, months, root, list(ym_t))
            if base is not None:
                fig.add_trace(go.Scatter(
                    x=[min(xs), max(xs)], y=[base, base], mode="lines",
                    name="Climatology",
                    line=dict(color=REFERENCE_PALETTE[0], width=REFERENCE_WIDTH),
                    hovertemplate=f"Climatology: {base:.4f}<extra></extra>"))
                if band != "all":
                    # The climatology files carry no altitude layer, so this
                    # line is the whole-domain value while the model curve is
                    # banded. Flagged rather than hidden: the comparison is
                    # still useful, but it is not like-for-like.
                    note = (f"climatology is not altitude-stratified — its line "
                            f"is the whole-domain value, not {band} m")
            else:
                note = "climatology unavailable for this selection — target only"
        else:
            ym_r = _load_years(menu, reference, truth, variable, metric, root, year)
            rxs, rys = (_season_leadtime(ym_r, region, months, band)
                        if ym_r else ([], []))
            r_name = f"{forecast_label(reference)} · {_score_tag(reference)}"
            if rxs:
                fig.add_trace(go.Scatter(
                    x=rxs, y=rys, mode="lines+markers", name=r_name,
                    line=dict(color=REFERENCE_PALETTE[0], width=REFERENCE_WIDTH),
                    marker=dict(size=8, symbol=REFERENCE_SYMBOLS[0]),
                    hovertemplate=f"{r_name} · %{{x}}h · %{{y:.4f}}<extra></extra>"))
            else:
                # Same two failure modes as Miscalibration: no comparable files
                # at all, or none carrying this band's layer.
                banded_out = bool(ym_r) and band != "all" and bool(
                    _season_leadtime(ym_r, region, months, "all")[0])
                note = (f"{forecast_label(reference)} carries no {band} m layer "
                        f"— target only (regenerate its diagnostics to include "
                        f"altitude bands)" if banded_out else
                        f"{forecast_label(reference)} has no comparable data — "
                        f"target only")
        _lt_layout(fig, xs, f"CRPS{f' ({unit})' if unit else ''}")
        st.plotly_chart(fig, width="stretch")
        _caption(list(ym_t), f"{season}" + (f" · {note}" if note else ""))
        return

    # ── Country View — difference target − reference at one lead ──
    if lead is None:
        _no_data_card("No lead time on disk for this selection.")
        return
    if reference == _CLIM:
        _no_data_card("Spatial difference vs climatology needs gridded climatology "
                      "files, which are not published for this deployment yet. "
                      "Use Graph View, or pick another model as the reference.")
        return
    cap = (f"{forecast_label(model)} ({_score_tag(model)}) − "
           f"{forecast_label(reference)} ({_score_tag(reference)}) · "
           f"{season} · {lead}h · {_year_note(year)}")
    tgrid = _lead_grid(ym_t, model, truth, region, months, lead, root, band)
    if tgrid is not None:  # gridded truth → per-pixel difference
        ym_r = _load_years(menu, reference, truth, variable, metric, root, year)
        rgrid = _lead_grid(ym_r, reference, truth, region, months, lead, root,
                           band)
        if rgrid is None or rgrid.shape != tgrid.shape:
            _no_data_card(f"No comparable gridded data for {forecast_label(reference)}.")
            return
        diff = tgrid.copy(data=tgrid.values - rgrid.values)
        _draw_map(diff, cbar=f"Δ CRPS{f' ({unit})' if unit else ''}", diverging=True)
        _caption(list(ym_t), cap)
        return
    # station truth → per-gauge difference map + drill-down
    fig, da2 = _disc_station_map(menu, model, reference, variable, truth, region,
                                 months, root, lead, year, band)
    if fig is None:
        _no_data_card(f"No comparable station data for {forecast_label(reference)}.")
        return
    from ui.components.spatial_map import _station_choices
    choices = _station_choices(da2, region)
    event = st.plotly_chart(
        fig, width="stretch", on_select="rerun", selection_mode="points",
        key="sd_dc_stmap", config={"displayModeBar": True,
                                   "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    st.markdown(
        f"<div style='font-size:0.68rem;color:{MUTED};margin-top:2px;'>Per-station "
        f"difference: {cap} · click a gauge to inspect both models.</div>",
        unsafe_allow_html=True)
    pts = (event or {}).get("selection", {}).get("points", []) if event else []
    if pts:
        i = pts[0].get("point_index")
        if i is not None and 0 <= i < len(choices):
            s_id, s_name = choices[i]
            _disc_station_detail(menu, model, reference, variable, truth, region,
                                 months, root, s_id, s_name, lead, year)


def _forecast_leads(menu, forecast, variable, truth, metric, year=None):
    """Lead times available on disk for one forecast, across all years."""
    leads: set[int] = set()
    for y in _years_in_scope(menu, forecast, truth, variable, year):
        rs = menu.resolve_storage(forecast=forecast, target=truth, metric=metric,
                                  year=y, variable=variable)
        if rs:
            leads.update(rs["leads"])
    return sorted(leads)


def _lead_grid(ym, forecast, truth, region, months, lead, root, band="all"):
    """Per-pixel season grid for one forecast at one lead, averaged over years."""
    mset = tuple(months) if months else tuple(range(1, 13))
    grids, ref = [], None
    for y, (lt_map, rs) in ym.items():
        if rs.get("target_kind") == "station" or lead not in lt_map:
            if rs.get("target_kind") == "station":
                return None
            continue
        da = load_season_spatial(forecast, rs["native_var"], truth,
                                 rs["storage_metric"], y, lead, mset, root=root)
        if da is None or da.ndim != 2:
            continue
        if band != "all":
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


def _score_tag(forecast: str) -> str:
    """The score family a forecast is shown under: raw → MAE, +IDR → CRPS."""
    return "CRPS" if forecast.endswith("-idr") else "MAE"


def _station_vals_at_lead(menu, forecast, variable, truth, months, root, lead,
                          year=None, band="all"):
    """(template station DataArray, {sid: season-mean over years}) for one
    forecast at one lead, or (None, None) if not a station target / no data."""
    from ui.components.spatial_map import _station_season_value
    ref_da, acc, order = None, {}, None
    for y in _years_in_scope(menu, forecast, truth, variable, year):
        rs = menu.resolve_storage(forecast=forecast, target=truth, metric="crps",
                                  year=y, variable=variable)
        if rs is None or lead not in rs["leads"]:
            continue
        lt_map = load_run(forecast, rs["native_var"], truth, rs["storage_metric"],
                          y, [lead], root=root)
        ltd = lt_map.get(lead)
        if ltd is None:
            continue
        da = ltd.annual_spatial_average.squeeze()
        if "station" not in da.dims:
            return None, None
        if ref_da is None:
            ref_da, order = da, list(da["station"].values)
        in_band = _band_station_ids(ltd, band)
        for sid in order:
            if in_band is not None and int(sid) not in in_band:
                continue
            v = _station_season_value(ltd, sid, months)
            if np.isfinite(v):
                acc.setdefault(int(sid), []).append(v)
    return (ref_da, acc) if ref_da is not None else (None, None)


def _disc_station_map(menu, model, reference, variable, truth, region, months,
                      root, lead, year=None, band="all"):
    """Per-gauge (target − reference) map at one lead, averaged over years."""
    tda, tacc = _station_vals_at_lead(menu, model, variable, truth, months, root,
                                     lead, year, band)
    rda, racc = _station_vals_at_lead(menu, reference, variable, truth, months,
                                      root, lead, year, band)
    if tda is None or rda is None or not tacc or not racc:
        return None, None
    order = list(tda["station"].values)
    vals = np.array([
        (float(np.mean(tacc[int(s)])) - float(np.mean(racc[int(s)])))
        if int(s) in tacc and int(s) in racc else np.nan for s in order])
    da2 = tda.copy(data=vals)
    unit = VARIABLE_UNITS.get(variable, "")
    return _station_diverging_figure(
        da2, region, f"Δ CRPS{f' ({unit})' if unit else ''}"), da2


def _station_lead_pair(menu, forecast, variable, truth, region, months, root, sid,
                       year=None):
    """One gauge's season-mean vs lead time for a forecast (over years)."""
    from ui.components.spatial_map import _station_season_value
    by: dict[int, list] = {}
    for y in _years_in_scope(menu, forecast, truth, variable, year):
        rs = menu.resolve_storage(forecast=forecast, target=truth, metric="crps",
                                  year=y, variable=variable)
        if rs is None:
            continue
        lt_map = load_run(forecast, rs["native_var"], truth, rs["storage_metric"],
                          y, rs["leads"], root=root)
        for lt, ltd in lt_map.items():
            v = _station_season_value(ltd, sid, months)
            if np.isfinite(v):
                by.setdefault(lt, []).append(v)
    leads = sorted(by)
    return leads, [float(np.mean(by[lt])) for lt in leads]


def _station_month_pair(menu, forecast, variable, truth, months, root, sid, lead,
                        year=None):
    """One gauge's monthly curve for a forecast at a lead (first year with data)."""
    from ui.components.spatial_map import _station_month_curve
    for y in _years_in_scope(menu, forecast, truth, variable, year):
        rs = menu.resolve_storage(forecast=forecast, target=truth, metric="crps",
                                  year=y, variable=variable)
        if rs is None or lead not in rs["leads"]:
            continue
        lt_map = load_run(forecast, rs["native_var"], truth, rs["storage_metric"],
                          y, [lead], root=root)
        ltd = lt_map.get(lead)
        if ltd is not None:
            return _station_month_curve(ltd, sid, months)
    return [], []


def _disc_station_detail(menu, model, reference, variable, truth, region, months,
                         root, sid, name, lead, year=None) -> None:
    """One gauge's target vs reference curves (lead-time + monthly)."""
    tx, ty = _station_lead_pair(menu, model, variable, truth, region, months, root,
                                sid, year)
    rx, ry = _station_lead_pair(menu, reference, variable, truth, region, months,
                                root, sid, year)
    tm = _station_month_pair(menu, model, variable, truth, months, root, sid, lead,
                             year)
    rm = _station_month_pair(menu, reference, variable, truth, months, root, sid,
                             lead, year)
    unit = VARIABLE_UNITS.get(variable, "")
    ylab = f"CRPS{f' ({unit})' if unit else ''}"
    t_name = f"{forecast_label(model)} · {_score_tag(model)}"
    r_name = f"{forecast_label(reference)} · {_score_tag(reference)}"
    st.markdown(f"<div style='font-size:0.9rem;font-weight:700;color:{GREEN_DARK};"
                f"margin:6px 0 2px;'>📍 {name}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        if tx or rx:
            f1 = go.Figure()
            if tx:
                f1.add_trace(go.Scatter(x=tx, y=ty, mode="lines+markers",
                             name=t_name, line=dict(color=GREEN, width=3),
                             marker=dict(size=6)))
            if rx:
                f1.add_trace(go.Scatter(x=rx, y=ry, mode="lines+markers",
                             name=r_name,
                             line=dict(color=REFERENCE_PALETTE[0],
                                       width=REFERENCE_WIDTH),
                             marker=dict(size=6)))
            f1.update_layout(template="plotly_white", height=300,
                             margin=dict(l=8, r=8, t=28, b=8),
                             title=dict(text="vs lead time", font=dict(size=12)),
                             xaxis=dict(title="Lead (h)"), yaxis=dict(title=ylab),
                             legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                             paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(f1, width="stretch")
        else:
            st.caption("No lead-time curves for this station.")
    with c2:
        if tm[0] or rm[0]:
            f2 = go.Figure()
            if tm[0]:
                f2.add_trace(go.Scatter(x=[_MONTH_NAMES[m - 1] for m in tm[0]],
                             y=tm[1], mode="lines+markers", name=t_name,
                             line=dict(color=GREEN, width=3), marker=dict(size=6)))
            if rm[0]:
                f2.add_trace(go.Scatter(x=[_MONTH_NAMES[m - 1] for m in rm[0]],
                             y=rm[1], mode="lines+markers", name=r_name,
                             line=dict(color=REFERENCE_PALETTE[0],
                                       width=REFERENCE_WIDTH),
                             marker=dict(size=6)))
            f2.update_layout(template="plotly_white", height=300,
                             margin=dict(l=8, r=8, t=28, b=8),
                             title=dict(text=f"by month · {lead}h", font=dict(size=12)),
                             yaxis=dict(title=ylab),
                             legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                             paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(f2, width="stretch")
        else:
            st.caption("No monthly curves for this station.")


def _clim_baseline(truth, metric, months, root, years):
    """Flat climatology baseline (season mean of the monthly climatology), or
    None when no climatology file exists for the selection."""
    idx = [m - 1 for m in (months or range(1, 13))]
    for y in years:
        if not climatology_available(truth, y, metric, root=root):
            continue
        try:
            cd = ClimatologyLoader().load(truth, y, metric, root=root)
        except Exception:
            cd = None
        if cd is not None:
            v = float(np.nanmean(np.asarray(cd.monthly_average, float)[idx]))
            if np.isfinite(v):
                return v
    return None


# ═════════════════════════════════════════════════════════════════════════
#  Shared UI bits
# ═════════════════════════════════════════════════════════════════════════

def _lt_layout(fig, xs, y_title) -> None:
    fig.update_layout(
        template="plotly_white", height=470, margin=dict(l=10, r=10, t=18, b=10),
        xaxis=dict(title="<b>Lead Time (hours)</b>", tickmode="array",
                   tickvals=xs, ticktext=[f"{h}h" for h in xs], gridcolor="#F3F4F6"),
        yaxis=dict(title=f"<b>{y_title}</b>", gridcolor="#F3F4F6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white", plot_bgcolor="white")


def _config_title() -> None:
    st.markdown(
        f"<div style='font-weight:700;font-size:0.78rem;letter-spacing:0.08em;"
        f"color:{GREEN_DARK};margin-bottom:6px;'>CONFIGURATION</div>",
        unsafe_allow_html=True,
    )


def _caption(years, extra: str) -> None:
    left = f"Years included: {', '.join(map(str, years))}" if years else ""
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;"
        f"font-size:0.68rem;color:{MUTED};margin-top:2px;'><span>{left}</span>"
        f"<span style='border:1px solid #e2e6e2;padding:3px 10px;border-radius:6px;'>"
        f"Scores aggregated over the test period · {extra}</span></div>",
        unsafe_allow_html=True,
    )


def _idx(options: list, value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _no_data_card(msg: str) -> None:
    st.markdown(
        f"""<div style="height:420px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;background:#F9FAFB;
            border-radius:10px;border:1px dashed #D1D5DB;color:{MUTED};">
          <div style="font-size:30px;margin-bottom:10px;">📉</div>
          <div style="font-size:13px;font-weight:600;text-align:center;
              max-width:440px;">{msg}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _inject_page_css() -> None:
    st.html(f"""
<style>
/* Region chip picker */
.st-key-sd_countries {{
  background: {GREEN_DARK}; border-radius: 12px; padding: 14px 16px 10px;
}}
.st-key-sd_countries [data-testid="stWidgetLabel"] p {{
  color: #ffffff !important; font-weight: 700 !important; font-size: 0.8rem !important;
}}
.st-key-sd_countries [data-testid="stPills"] button {{
  background: {GREEN_BG}; border: 1px solid {GREEN_DARK}; border-radius: 6px;
  color: {GREEN_DARK}; font-size: 0.7rem; font-weight: 600; padding: 2px 10px;
  min-height: 26px;
}}
.st-key-sd_countries [data-testid="stPills"] button[data-selected="true"],
.st-key-sd_countries [data-testid="stPills"] button[kind="pillsActive"],
.st-key-sd_countries [data-testid="stBaseButton-pillsActive"] {{
  background: {AMBER} !important; border-color: {AMBER} !important;
  color: {GREEN_DARK} !important; font-weight: 700 !important;
}}
/* Config panels */
.st-key-sd_mc_config, .st-key-sd_dc_config {{
  background: {GREEN_BG}; border: 1px solid {GREEN_DARK}; border-radius: 12px;
  padding: 14px 14px 10px;
}}
.st-key-sd_mc_config .fc-config-head, .st-key-sd_dc_config .fc-config-head {{
  background: {GREEN_DARK}; color: #ffffff; font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.04em; border-radius: 6px; padding: 5px 10px; margin: 10px 0 6px;
}}
.st-key-sd_mc_config [data-testid="stWidgetLabel"] p,
.st-key-sd_dc_config [data-testid="stWidgetLabel"] p {{
  font-size: 0.62rem !important; font-weight: 700 !important;
  letter-spacing: 0.08em; color: {GREEN_DARK} !important;
}}
/* View toggles — dark-green active pill */
.st-key-sd_dc_view [data-testid="stSegmentedControl"] button[aria-checked="true"],
.st-key-sd_mc_view [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
  background: {GREEN_DARK} !important; color: #ffffff !important;
}}

/* ── Section tabs (Miscalibration / Discrimination) ────────────────────────
   Top-level: these switch the whole page body, so they outrank the view
   toggles nested inside a section and are sized to say so — bigger type, a
   seated tab shape, and a rule beneath that the active tab sits on. */
.st-key-sd_section_switch {{
  margin: 26px 0 2px;
  border-bottom: 2px solid {GREEN_BG};
}}
.st-key-sd_section_switch [data-testid="stSegmentedControl"] {{ gap: 4px; }}
.st-key-sd_section_switch [data-testid="stSegmentedControl"] button {{
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
.st-key-sd_section_switch [data-testid="stSegmentedControl"] button:hover {{
  color: {GREEN_DARK} !important;
  background: {GREEN_BG} !important;
}}
.st-key-sd_section_switch [data-testid="stSegmentedControl"]
  button[aria-checked="true"] {{
  color: {GREEN_DARK} !important;
  background: {GREEN_BG} !important;
  border-bottom: 3px solid {GREEN_DARK} !important;
}}
</style>
    """)

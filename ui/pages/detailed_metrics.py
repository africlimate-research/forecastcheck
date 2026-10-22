# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Threshold Diagnostics page — ForecastCheck design, wired to the LOCAL engine.

The curve metrics: reliability, ROC and cost-loss over exceedance thresholds,
under a single Country & Region picker.

This page was "Detailed Metrics" and carried three sections. The other two moved
out: *Spatial Performance* became National Overview's **Raw Forecast Metrics**
tab (next to Calibrated Forecast Metrics, where the comparison belongs),
and *Geographical / Station performance* was dropped once the altitude-band
selector on the analysis pages covered the same subsetting. Only the threshold
curves are left, so the page is named for them and shows no section heading of
its own. The module file keeps its ``detailed_metrics`` name, as do the nav slug
and icon asset — renaming those would touch every ``render_explore_cards`` call
for no user-visible gain (the URL *is* ``threshold-diagnostics``).

Everything is DISCOVERY-DRIVEN: the selectors offer only the model / metric /
region / season / lead / threshold combinations `verification.run` has produced.
Threshold curves pool over time, so each slice has to be pooled separately by the
pipeline — which is why a slice missing from disk simply is not offered.
"""
from __future__ import annotations

import streamlit as st

from ui.components.region_picker import render_region_picker

from ui.app_config.settings import ALTITUDE_UNIT
from ui.app_config.settings import (
    DEFAULT_REGION,
    REGIONS,
    forecast_label,
    target_label,
    variable_label,
)
from ui.components import threshold_figures
from ui.components.event_box import counting_series, render_event_box
from ui.components.chrome import (
    AMBER,
    GREEN_BG,
    GREEN_DARK,
    MUTED,
    config_group,
    render_explore_cards,
    render_footer,
    render_header,
)
from ui.pages.threshold.threshold import _axes_metric, _draw, _load_series
from ui.services.data_cache import ui_results_root
from ui.services.engine import kind_of
from ui.services.threshold_data import discover_curves, load_curve
from ui.utils.leads import common_leads
from verification.seasons import DEFAULT_SEASON_SLUG, season_label

# Segmented-control labels → curve metric key.
_THRESHOLD_VIEWS = {
    "Reliability diagram": "reliability",
    "ROC curve": "roc",
    "Cost-loss curves": "cost_loss",
}


def render() -> None:
    render_header(active="detailed_metrics")
    _inject_page_css()

    st.markdown(
        f"""
        <div style="padding:26px 6px 0;max-width:720px;">
          <h1 style="margin:0 0 8px;">Threshold Diagnostics</h1>
          <p style="color:{MUTED};font-size:0.88rem;line-height:1.7;margin:0;">
            Reliability, ROC and cost-loss curves for exceedance thresholds.
            Because these curves pool over time, each threshold, altitude,
            region and season slice is produced separately by the pipeline — the
            selectors below offer only the slices that exist on disk.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    region = _render_region_picker()
    _threshold_section(region)
    render_explore_cards("national_overview", "score_decomposition")
    render_footer()


# ═════════════════════════════════════════════════════════════════════════
#  Country & Region picker (top — applied to every section)
# ═════════════════════════════════════════════════════════════════════════

def _render_region_picker() -> str:
    """The shared Country & Region selector (see ui.components.region_picker)."""
    return render_region_picker("dm")




# ═════════════════════════════════════════════════════════════════════════
#  Section 3 — Threshold-based performance (curve diagnostics)
# ═════════════════════════════════════════════════════════════════════════

def _threshold_section(region: str) -> None:
    # No <h2>: this is the only section on the page, so the page title names it.
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.82rem;margin:6px 0 14px;'>"
        f"Curves pooled over <b>{region}</b></p>",
        unsafe_allow_html=True,
    )

    root = ui_results_root()
    df = discover_curves(root)
    if df.empty:
        st.info(
            "No threshold diagnostics found under the data root. Generate them "
            "with e.g. `python -m verification.run --model graphcast-idr ifs-idr "
            "--metric roc reliability cost_loss --lead-day 1 3 5 7 --threshold "
            "10 20 40 50 --altitude all` (see `threshold_commands.txt`)."
        )
        return
    df = df.copy()
    df["variable"] = df["native_var"].map(kind_of)
    df = df[df["region"] == region]
    if df.empty:
        st.info(f"No threshold curves have been generated for {region}. "
                "Run `verification.run --region {region} …` to add them.")
        return

    col_cfg, col_view = st.columns([1, 3], gap="medium")

    with col_cfg:
        with st.container(key="dm_thr_config"):
            _config_title()
            with config_group("dm_thr", "ft", "Forecast &amp; Truth"):
                year = st.selectbox("YEAR", sorted(df["year"].unique()),
                                    key="dm_thr_year")
                d = df[df["year"] == year]
                variable = st.selectbox("VARIABLE", sorted(d["variable"].unique()),
                                        format_func=variable_label, key="dm_thr_var")
                d = d[d["variable"] == variable]
                truths_available = sorted(d["target"].unique())
                # Deferred: the multiselect only appears once exactly one model is
                # picked, and MODELS is chosen below. Read the previous run's model
                # selection from session_state to decide — the same trick the view
                # toggles use, since Streamlit has already written it by this point.
                _prev_models = st.session_state.get("dm_thr_models") or []
                compare_truths = len(_prev_models) == 1 and len(truths_available) > 1
                if compare_truths:
                    picked_truths = st.multiselect(
                        "TRUTH SOURCES", truths_available, default=truths_available,
                        format_func=target_label, key="dm_thr_truths")
                    if not picked_truths:
                        st.info("Pick at least one truth source.")
                        return
                    truth = picked_truths[0]
                else:
                    truth = st.selectbox("TRUTH SOURCE", truths_available,
                                         format_func=target_label, key="dm_thr_truth")
                    picked_truths = [truth]
                # Facets below stay the INTERSECTION across whatever is being compared,
                # so a curve can never be drawn for a combination one side lacks.
                d = d[d["target"].isin(picked_truths)]

                # MODELS belongs here — a model IS the forecast. It is therefore
                # offered before SEASON is known, so the options are every model
                # with curves for this year/variable/truth; a model that lacks the
                # season picked below is reported rather than silently dropped.
                models_available = sorted(d["forecast"].unique())
                models = st.multiselect("MODELS", models_available,
                                        default=models_available,
                                        format_func=forecast_label,
                                        key="dm_thr_models")
                if not models:
                    st.info("Select at least one model.")
                    return
                d = d[d["forecast"].isin(models)]

            with config_group("dm_thr", "co", "Conditioning"):
                seasons_available = sorted(d["season"].unique())
                if DEFAULT_SEASON_SLUG in seasons_available:
                    seasons_available = [DEFAULT_SEASON_SLUG] + [
                        s for s in seasons_available if s != DEFAULT_SEASON_SLUG]
                season = st.selectbox("SEASON", seasons_available,
                                      format_func=season_label, key="dm_thr_season")
                sel = d[d["season"] == season]

                # Curves pool per (region, season) slice, so a selected model can
                # simply have none for this season — say which, don't silently drop.
                missing = sorted(set(models) - set(sel["forecast"].unique()))
                if missing:
                    st.info("No " + season_label(season) + " curves for "
                            + ", ".join(forecast_label(m) for m in missing) + ".")
                if sel.empty:
                    st.info("No curves for this season and model selection.")
                    return
                models = [m for m in models if m in set(sel["forecast"].unique())]
                native_var = sel.iloc[0]["native_var"]
                leads = common_leads(sel["leads"])
                if not leads:
                    st.info("No lead time is available for every selected model.")
                    return
                if len(leads) == 1:
                    # A select_slider cannot draw a single option: the browser raises
                    # "RangeError: min (0) is equal/bigger than max (0)".
                    lead = leads[0]
                    st.selectbox("LEAD TIME (h)", leads, disabled=True,
                                 key="dm_thr_lead_only")
                else:
                    lead = st.select_slider("LEAD TIME (h)", options=leads,
                                            value=leads[len(leads) // 2],
                                            key="dm_thr_lead")

                # Axes are the UNION over the selected models, not model[0]'s.
                # Models run with different --threshold / --altitude sets carry
                # different coordinates (AIFS has all 13 thresholds; graphcast and
                # ifs were run with 5), and reading only the first model's axes
                # offered values the others lack — selecting one then raised a
                # KeyError from ds.sel(threshold=…) deep in the drawing layer.
                # Union rather than intersection so a value only one model has is
                # still offerable: _load_series drops the models that lack it and
                # plots the rest.
                thr_set: set[int] = set()
                band_set: set[str] = set()
                thr_unit_set: set[str] = set()
                for m in models:
                    ax = load_curve(m, native_var, truth, _axes_metric(sel, m),
                                    year, lead, root, region=region, season=season)
                    thr_set |= {int(t) for t in ax["threshold"].values}
                    thr_unit_set.add(ax["threshold"].attrs.get("units", ""))
                    band_set |= {str(b) for b in ax["altitude"].values}
                thr_options = sorted(thr_set)
                # Units come from the FILE: thresholds are per variable (mm/day for
                # precipitation, degC for daily-max temperature) and the writer stamps
                # them on the coord. 20 is precipitation's default and is not an option
                # for temperature, so the lowest threshold is the fallback.
                thr_units = thr_unit_set.pop() if len(thr_unit_set) == 1 else ""
                default_thr = 20 if 20 in thr_options else thr_options[0]
                if st.session_state.get("dm_thr_thr") not in thr_options:
                    st.session_state["dm_thr_thr"] = default_thr
                threshold = st.selectbox(
                    f"THRESHOLD ({thr_units})" if thr_units else "THRESHOLD",
                    thr_options, key="dm_thr_thr")
                # "all" first, then ascending bands — sorted() would put ">3000"
                # before the numeric ranges.
                band_options = [b for b in ("all", "0-1000", "1000-2000",
                                            "2000-3000", ">3000") if b in band_set]
                band_options += sorted(band_set - set(band_options))
                if st.session_state.get("dm_thr_band") not in band_options:
                    st.session_state["dm_thr_band"] = (
                        "all" if "all" in band_options else band_options[0])
                band = st.selectbox(f"ALTITUDE BAND ({ALTITUDE_UNIT})", band_options,
                                    key="dm_thr_band")

    with col_view:
        view_label = st.segmented_control(
            "threshold view", list(_THRESHOLD_VIEWS),
            default="Reliability diagram", key="dm_thr_view",
            label_visibility="collapsed",
        ) or "Reliability diagram"
        metric = _THRESHOLD_VIEWS[view_label]

        logx = False
        if metric == "cost_loss":
            logx = st.radio("Cost-loss ratio axis", ["linear", "log"],
                            horizontal=True, key="dm_thr_logx") == "log"

        def _load(m):
            if len(picked_truths) > 1:
                # One model across several truths: label by TRUTH, since that is what
                # varies. `_load_series` labels by model, so relabel its single entry.
                out = []
                for tr in picked_truths:
                    got = _load_series(models, native_var, tr, m, year, lead,
                                       root, sel, region, season, band,
                                       threshold=threshold)
                    out += [(target_label(tr), ds) for _, ds in got]
                return out
            return _load_series(models, native_var, truth, m, year, lead,
                                root, sel, region, season, band, threshold=threshold)

        series = _load(metric)
        # Name the models this threshold/band excluded rather than dropping them
        # silently — the same courtesy the season mismatch above gets.
        plotted = {label for label, _ in series}
        dropped = ([] if len(picked_truths) > 1
                   else [m for m in models if forecast_label(m) not in plotted])
        if dropped and series:
            st.info(f"No {threshold} {thr_units} curve for "
                    + ", ".join(forecast_label(m) for m in dropped)
                    + " — plotting the models that have it.")
        elif dropped:
            st.info("No model in this selection has a curve at "
                    f"{threshold} {thr_units} · {band} · {lead}h. "
                    "Pick another threshold, or generate it with "
                    "`verification.run --metric "
                    f"{metric} --threshold {threshold}`.")
            return
        if metric == "reliability":
            _draw(series, lambda s: threshold_figures.reliability_figure(s, threshold, band, thr_units))
        elif metric == "roc":
            _draw(series, lambda s: threshold_figures.roc_figure(s, threshold, band, thr_units))
        else:
            _draw(series, lambda s: threshold_figures.cost_loss_figure(s, threshold, band, logx=logx, units=thr_units))
        # How much data the curve above rests on, per series (models do not share an
        # evaluation set). Same slice as the figure; on the cost-loss tab the counts are
        # read from the matching ROC/reliability file, so all three tabs show the same box.
        render_event_box(counting_series(series, metric, _load),
                         threshold=threshold, units=thr_units, band=band,
                         lead=lead, region=region, years=[year], season=season)


# ═════════════════════════════════════════════════════════════════════════
#  Shared bits
# ═════════════════════════════════════════════════════════════════════════

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


def _inject_page_css() -> None:
    st.html(f"""
<style>
/* Region chip picker */
.st-key-dm_countries {{
  background: {GREEN_DARK};
  border-radius: 12px;
  padding: 14px 16px 10px;
}}
.st-key-dm_countries [data-testid="stWidgetLabel"] p {{
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 0.8rem !important;
}}
.st-key-dm_countries [data-testid="stPills"] button {{
  background: {GREEN_BG};
  border: 1px solid {GREEN_DARK};
  border-radius: 6px;
  color: {GREEN_DARK};
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 10px;
  min-height: 26px;
}}
.st-key-dm_countries [data-testid="stPills"] button[data-selected="true"],
.st-key-dm_countries [data-testid="stPills"] button[kind="pillsActive"],
.st-key-dm_countries [data-testid="stBaseButton-pillsActive"] {{
  background: {AMBER} !important;
  border-color: {AMBER} !important;
  color: {GREEN_DARK} !important;
  font-weight: 700 !important;
}}
/* Config panels (Spatial / Station / Threshold share the look) */
.st-key-dm_sp_config, .st-key-dm_geo_config, .st-key-dm_thr_config {{
  background: {GREEN_BG};
  border: 1px solid {GREEN_DARK};
  border-radius: 12px;
  padding: 14px 14px 10px;
}}
.st-key-dm_sp_config .fc-config-head,
.st-key-dm_geo_config .fc-config-head,
.st-key-dm_thr_config .fc-config-head {{
  background: {GREEN_DARK};
  color: #ffffff;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border-radius: 6px;
  padding: 5px 10px;
  margin: 10px 0 6px;
}}
.st-key-dm_sp_config [data-testid="stWidgetLabel"] p,
.st-key-dm_geo_config [data-testid="stWidgetLabel"] p,
.st-key-dm_thr_config [data-testid="stWidgetLabel"] p {{
  font-size: 0.62rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em;
  color: {GREEN_DARK} !important;
}}
/* Segmented control — dark-green active pill */
.st-key-dm_thr_view [data-testid="stSegmentedControl"] button[aria-checked="true"],
.st-key-dm_geo_view [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
  background: {GREEN_DARK} !important;
  color: #ffffff !important;
}}
</style>
    """)

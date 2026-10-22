# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from ui.app_config.settings import ALTITUDE_UNIT
from ui.app_config.settings import (DEFAULT_REGION, DEFAULT_SEASON, REGIONS,
                                                 SEASONS, forecast_label,
                                                 metric_label, target_label,
                                                 variable_label)
from ui.services.data_cache import get_menu_index
from ui.utils.styling import inject_css

# ── Filter panel CSS — scoped to the inline panel ─────────────────────────
_PANEL_CSS = """
<style>
/* Expanders inside the filter panel */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
    box-shadow: none !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary {
    font-family: Inter, sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #374151 !important;
}
[data-testid="stExpander"] label {
    font-family: Inter, sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}
/* Compact vertical rhythm so every filter section fits one screen */
[data-testid="stExpander"] summary {
    padding: 0.45rem 0.8rem !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0.35rem 0.8rem 0.55rem !important;
}
[data-testid="stExpander"] [data-testid="stVerticalBlock"] {
    gap: 0.45rem !important;
}
/* Hide the sidebar collapse control entirely */
[data-testid="collapsedControl"] { display: none !important; }
</style>
"""


# ── FilterState ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FilterState:
    region:            str
    variable:          str
    truth:             str
    year:              str
    selected_variants: tuple[str, ...]  # forecast tokens
    metric_key:        str
    # Altitude band ("all" = whole domain). Only offered when the region is
    # the deployment country and the data carries band values; see
    # _altitude_band_options for the gate.
    altitude_band:     str = "all"
    season:            str = DEFAULT_SEASON  # display label from settings.SEASONS


def variant_display_name(forecast: str) -> str:
    """Display label for a selected forecast token."""
    return forecast_label(forecast)


# ── Atomic widgets ────────────────────────────────────────────────────────
#
# Options are now discovery-driven, so they change as parent facets change.
# Streamlit raises if a widget's stored session value is no longer in its
# options, so each helper drops a stale stored value before rendering — the
# widget then falls back to the computed default.

def _selectbox(label, options, default=None, key=None, format_func=str):
    if not options:
        return None
    if key and key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    index = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=index, key=key, format_func=format_func)


def _multiselect(label, options, default=None, key=None, format_func=str):
    if key and key in st.session_state:
        # Session state already holds the selection — sanitize it against the
        # current options and let it drive (passing `default` too would warn).
        st.session_state[key] = [v for v in st.session_state[key] if v in options]
        return st.multiselect(label, options=options, key=key, format_func=format_func)
    return st.multiselect(label, options=options, default=default or [],
                          key=key, format_func=format_func)


# Band labels per probed run — one summary file is opened per run, then reused.
_BAND_PROBE_CACHE: dict[tuple, list[str]] = {}


def _altitude_band_options(menu, region, variable, truth, metric, year, variants) -> list[str]:
    """Band labels for the altitude filter, or [] when it must stay hidden.

    Hidden unless the selected region IS the deployment country — the
    whole-domain band means the pipeline stored are that country's band means
    only then — and the data was produced by the band-aware pipeline (legacy
    files keep the filter off until a year is regenerated). Probes one summary
    file per run, cached for the process lifetime.
    """
    from ui.services.data_cache import ui_results_root
    from ui.services.engine import open_summary
    from ui.utils.regions import deployment_country

    if not variants or region != deployment_country():
        return []
    root = ui_results_root()
    for fc in variants:
        r = menu.resolve_storage(forecast=fc, target=truth, metric=metric,
                                 year=year, variable=variable)
        if not r or not r["leads"]:
            continue
        key = (fc, truth, r["storage_metric"], year, str(root))
        if key not in _BAND_PROBE_CACHE:
            labels: list[str] = []
            try:
                with open_summary(fc, r["native_var"], truth, r["storage_metric"],
                                  year, r["leads"][0], root=root) as ds:
                    if "annual_average_by_altitude" in ds:
                        labels = [str(b) for b in ds["altitude"].values]
            except Exception:
                pass
            _BAND_PROBE_CACHE[key] = labels
        if _BAND_PROBE_CACHE[key]:
            return _BAND_PROBE_CACHE[key]
    return []


# ── Main render ───────────────────────────────────────────────────────────

def render_filters() -> FilterState:
    inject_css(Path(__file__).resolve().parent / "styles" / "filters.css")
    st.markdown(_PANEL_CSS, unsafe_allow_html=True)

    menu = get_menu_index()
    if not menu.values("year"):
        st.warning(
            "No verification results found for this country. Run "
            "`python -m verification.run` to generate summaries, or check "
            "`VERIFICATION_COUNTRY` / `VERIFICATION_DATA_ROOT`."
        )
        st.stop()

    st.markdown("""
        <div style='display:flex;align-items:center;gap:9px;padding:6px 2px 6px;'>
          <div style='width:30px;height:30px;border-radius:8px;
              background:rgba(10,153,56,0.09);border:1px solid rgba(10,153,56,0.2);
              display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
            <svg width='15' height='15' viewBox='0 0 24 24' fill='none'
                stroke='#0a9938' stroke-width='2.2' stroke-linecap='round'
                stroke-linejoin='round'>
              <circle cx='12' cy='12' r='3'/>
              <path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83
                2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21
                a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0
                0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0
                4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0
                0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1
                2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0
                1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0
                0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0
                0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65
                1.65 0 0 0-1.51 1z'/>
            </svg>
          </div>
          <div>
            <div style='font-family:Inter,sans-serif;font-size:0.65rem;font-weight:700;
                letter-spacing:0.12em;text-transform:uppercase;color:#0a9938;
                line-height:1;margin-bottom:2px;'>Configuration</div>
            <div style='font-family:Inter,sans-serif;font-size:0.82rem;font-weight:700;
                color:#171d1a;line-height:1.2;'>Analysis Settings</div>
          </div>
        </div>
        <div style='border-top:1px solid #e5e7eb;margin:0 0 6px;'></div>
    """, unsafe_allow_html=True)

    with st.expander("🌍 Spatial & Temporal", expanded=True):
        region = _selectbox("Region / Country", REGIONS, default=DEFAULT_REGION, key="f_region")
        years  = menu.values("year")
        year   = _selectbox("Year", years,
                            default="2024" if "2024" in years else None, key="f_year")
        # Fixed options (not discovery-driven): Yearly + the deployment's seasons.
        # Drives the season aggregate on the Lead-Time, Spatial, and Comparison
        # sub-tabs; the Seasonal Evaluation tab keeps its own month-range control.
        season = _selectbox("Season", list(SEASONS), default=DEFAULT_SEASON, key="f_season")

    with st.expander("🤖 Model & Variable", expanded=True):
        # Cascading facets — each list is narrowed to what exists on disk
        # given the choices above it.
        var_kinds = menu.values("variable", year=year)
        variable  = _selectbox("Variable", var_kinds,
                               format_func=variable_label, key="f_variable")

        targets = menu.values("target", year=year, variable=variable)
        truth   = _selectbox("Truth Source", targets,
                             format_func=target_label, key="f_truth")

        forecasts = menu.values("forecast", year=year, variable=variable, target=truth)
        selected  = _multiselect("Models", options=forecasts,
                                 default=forecasts[:1], format_func=forecast_label,
                                 key="f_models")
        selected_variants = tuple(selected)

    with st.expander("📊 Metric", expanded=True):
        # Metric is faceted on (year, variable, truth) but NOT on the chosen
        # models — so the metric list stays stable as you toggle models. A
        # model lacking the chosen metric just loads empty and is reported in
        # the "No data found for…" notice rather than vanishing the option.
        metrics    = menu.values("metric", year=year, variable=variable, target=truth)
        metric_key = _selectbox("Metric", metrics, default="mae" if "mae" in metrics else None,
                                format_func=metric_label, key="f_metric")

        # Altitude band — appears only for the deployment country on
        # band-aware data (see _altitude_band_options).
        band_labels = _altitude_band_options(menu, region, variable, truth,
                                             metric_key, year, selected_variants)
        if band_labels:
            band_opts = ["All altitudes"] + [f"{b} m" for b in band_labels]
            sel = _selectbox(f"Altitude ({ALTITUDE_UNIT})", band_opts, key="f_altband")
            altitude_band = "all" if sel == "All altitudes" else sel.removesuffix(" m")
        else:
            altitude_band = "all"

    return FilterState(
        region=region,
        variable=variable,
        truth=truth,
        year=year,
        selected_variants=selected_variants,
        metric_key=metric_key,
        altitude_band=altitude_band,
        season=season,
    )

# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from ui.app_config.settings import (CORRELATION_METRICS, LEAD_TIMES,
                                                 MODEL_PALETTE, forecast_label)
from ui.services.loader import LeadTimeData
from ui.utils.geo import (annual_spatial_grid, band_annual_mean,
                          band_season_mean, regional_annual_mean,
                          regional_season_mean)
from ui.utils.leads import common_leads

_FIGURE_CACHE:        dict[tuple, go.Figure]       = {}

_LOADING_CARD = """
<div style="height:460px;display:flex;flex-direction:column;align-items:center;
    justify-content:center;background:#F9FAFB;border-radius:10px;
    border:1px dashed #D1D5DB;gap:14px;font-family:'Inter',sans-serif;">
  <div style="width:38px;height:38px;border:3px solid #E5E7EB;
      border-top-color:#0a9938;border-radius:50%;
      animation:ts-spin 0.75s linear infinite;"></div>
  <div style="font-size:13px;font-weight:600;color:#6B7280;">{label}</div>
  <style>@keyframes ts-spin{{to{{transform:rotate(360deg);}}}}</style>
</div>
"""
# Clim regridded onto each model's grid — keyed per (model, region, clim).
# All lead times of one model share the same lat/lon grid, so we only
# interpolate climatology once per model and reuse it.
_MATCHED_CLIM_CACHE:  dict[tuple, np.ndarray | None] = {}
# Mean of (model_pixel − clim_interp_pixel) on the shared-finite mask;
# this is the value the Lead-Time chart plots when subtract_clim is on.
_DIFF_MEAN_CACHE:     dict[tuple, float | None]   = {}

_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "star", "hexagram", "x"]


def _matched_clim_for_model(
    model_name:   str,
    lt_map:       dict[int, LeadTimeData],
    region:       str,
    clim:         "object | None",
) -> "np.ndarray | None":
    """Build a month-matched climatology spatially aligned to this model.

    For partial-year models (e.g. months 4-12), `clim.annual_spatial_average`
    averages all 12 months and produces an unfair baseline. This helper
    averages `clim.monthly_spatial_average` over only the months the
    model actually has data for, then interpolates onto the model's grid.
    Result is cached per model so all of its lead times reuse a single
    interp.
    """
    if clim is None:
        return None
    key = (model_name, region, id(clim))
    if key in _MATCHED_CLIM_CACHE:
        return _MATCHED_CLIM_CACHE[key]
    first_ltd = next((ltd for ltd in lt_map.values() if ltd is not None), None)
    if first_ltd is None:
        _MATCHED_CLIM_CACHE[key] = None
        return None
    from ui.components.spatial_map import (_interpolate_clim,
                                                        _matched_clim_spatial)
    matched = _matched_clim_spatial(first_ltd, clim, region)
    if matched is None:
        _MATCHED_CLIM_CACHE[key] = None
        return None
    grid = annual_spatial_grid(first_ltd, region)
    result = _interpolate_clim(grid, matched)
    _MATCHED_CLIM_CACHE[key] = result
    return result


def _diff_mean(
    ltd:              "LeadTimeData",
    clim_interp:      "np.ndarray",
    region:           str,
) -> float | None:
    """`nanmean(model − clim_interp)` over the shared-finite mask.

    Matches the Spatial Distribution panel's mean exactly, so the chart
    and the map can never disagree on sign for IDR-style restricted-
    coverage models.
    """
    key = (id(ltd), id(clim_interp), region)
    if key in _DIFF_MEAN_CACHE:
        return _DIFF_MEAN_CACHE[key]
    try:
        grid_values = annual_spatial_grid(ltd, region).values
        diff = grid_values - clim_interp
        result: float | None = float(np.nanmean(diff))
        if not np.isfinite(result):
            result = None
    except Exception:
        result = None
    _DIFF_MEAN_CACHE[key] = result
    return result


def _fingerprint(
    model_results:     dict[str, dict[int, LeadTimeData]],
    metric_key:        str,
    region:            str,
    climatology_value: float | None,
    clim:              "object | None" = None,
    altitude_band:     str = "all",
    season_months:     "tuple[int, ...] | None" = None,
) -> tuple:
    model_means = tuple(
        (
            name,
            round(float(np.nanmean(
                [regional_season_mean(ltd, region, season_months) for ltd in lt_map.values()])), 8),
        )
        for name, lt_map in sorted(model_results.items())
        if lt_map
    )
    clim = round(float(climatology_value), 8) if isinstance(climatology_value, (int, float)) else None
    clim_id = id(clim) if clim is not None else None
    return (metric_key, region, model_means, clim, clim_id, altitude_band, season_months)


def render(
    model_results:     dict[str, dict[int, LeadTimeData]],
    metric_key:        str,
    region:            str                   = "Africa",
    climatology_value: float | None          = None,
    unit:              str                   = "",
    clim:              "object | None" = None,
    load_daily=None,
    altitude_band:     str                   = "all",
    season_months:     "tuple[int, ...] | None" = None,
) -> None:
    # Correlation is a per-day quantity (spatial-pattern correlation), not a single
    # value per lead — so this tab shows its daily series (Flavor B) rather than a
    # metric-vs-lead line. The map tab shows the per-cell temporal correlation (Flavor A).
    if metric_key.lower() in CORRELATION_METRICS and load_daily is not None:
        _render_correlation_daily(model_results, metric_key, unit, load_daily, season_months)
        return

    # Paint the loading card IMMEDIATELY, before any data processing
    # (fingerprint computation, climatology interpolation, per-(model, lt)
    # nanmean). Streamlit's progressive renderer flushes this delta to
    # the browser right away, so the user gets feedback the moment they
    # change a sidebar option. On warm caches everything below finishes
    # in tens of milliseconds and the placeholder is replaced before the
    # browser can paint it.
    chart_slot = st.empty()
    chart_slot.markdown(
        _LOADING_CARD.format(label=f"Building {metric_key.upper()} lead-time chart…"),
        unsafe_allow_html=True,
    )

    fp  = _fingerprint(model_results, metric_key, region, climatology_value,
                       clim, altitude_band, season_months) + (unit,)
    fig = _FIGURE_CACHE.get(fp)

    if fig is None:
        fig = _build_figure(model_results, metric_key, region,
                            climatology_value, unit, clim, altitude_band, season_months)
        if fig is not None:
            _FIGURE_CACHE[fp] = fig

    if fig is None:
        chart_slot.empty()
        st.markdown(
            f"""<div style="height:460px;display:flex;flex-direction:column;
                align-items:center;justify-content:center;background:#F9FAFB;
                border-radius:10px;border:1px dashed #D1D5DB;color:#9CA3AF;
                font-family:'Inter',sans-serif;">
                <div style="font-size:32px;margin-bottom:12px;">📈</div>
                <div style="font-size:13px;font-weight:600;">No data available</div>
                <div style="font-size:11px;margin-top:4px;">
                    {metric_key.upper()} · no results for this configuration</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Replace the loading placeholder with the chart in the same slot.
    with chart_slot.container():
        st.plotly_chart(fig, width="stretch")


def _season_day_mask(times, season_months) -> "np.ndarray":
    """Boolean mask of ``times`` whose calendar month is in ``season_months`` (1-based)."""
    months = np.asarray(times, dtype="datetime64[M]").astype(int) % 12 + 1
    return np.isin(months, list(season_months))


def _season_daily_series(times, vals, season_months):
    """Restrict a daily ``(times, vals)`` series to a season's days, contiguous on the axis.

    Keeps only the days whose calendar month is in ``season_months`` (the ordered season tuple),
    so the plot shows **only** the season's months. For a season that wraps the year end
    (e.g. NDJF Nov→Feb), the Jan–Feb days are shifted forward one year so Nov–Feb form
    one contiguous block on the time axis instead of leaving the whole year on screen with a gap.
    Returns ``(display_times, vals)`` sorted by display time (empty if no season day is present).
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    vals = np.asarray(vals, dtype=float)
    keep = _season_day_mask(times, season_months)
    t, v = times[keep], vals[keep]
    if t.size == 0:
        return t, v
    # Wrap tail = season months numerically smaller than the season's first (calendar) month;
    # shift those days +1 year so the kept days are chronologically contiguous.
    ordered = list(season_months)
    wrap = {m for m in ordered if m < ordered[0]}
    if wrap:
        m = t.astype("datetime64[M]").astype(int) % 12 + 1
        t = t.copy()
        t[np.isin(m, list(wrap))] += np.timedelta64(365, "D")
    order = np.argsort(t)
    return t[order], v[order]


def _render_correlation_daily(model_results, metric_key, unit, load_daily,
                              season_months=None) -> None:
    """Daily spatial-pattern correlation over the year, one line per model, at a chosen lead.

    Each lead has its own daily series, so a lead selector picks which to show; the series
    is loaded on demand (it isn't kept on ``LeadTimeData``). The correlation is over the
    whole domain each day — the region filter does not subset it. When a season is active
    the series is restricted to that season's calendar months (a valid subset of days).
    """
    # Intersection so the one selected lead has a daily series for every selected model.
    leads = common_leads(lm.keys() for lm in model_results.values() if lm)
    if not leads:
        st.info("No lead time is available for every selected model.")
        return
    lead = st.select_slider("Lead time (h)", options=leads,
                            value=leads[len(leads) // 2], key="ts_corr_lead")

    fig = go.Figure()
    drawn = 0
    for i, (model, lm) in enumerate(model_results.items()):
        if not lm:
            continue
        s = load_daily(model, lead)
        if s is None:
            continue
        times, vals = s
        if season_months:
            # Show ONLY the season's months: drop other days and lay the kept ones out
            # contiguously (NDJF's Jan–Feb side shifted +1 year), so the axis spans just the season.
            times, vals = _season_daily_series(times, vals, season_months)
            if times.size == 0:
                continue
        fig.add_trace(go.Scatter(
            x=times, y=vals, mode="lines", name=forecast_label(model),
            line=dict(color=MODEL_PALETTE[i % len(MODEL_PALETTE)], width=1.5),
            hovertemplate=f"{forecast_label(model)}<br>%{{x|%d %b}}<br>%{{y:.3f}}<extra></extra>"))
        drawn += 1
    if drawn == 0:
        st.info(f"No daily {metric_key.upper()} series for this selection.")
        return

    fig.add_hline(y=0, line=dict(color="#9CA3AF", width=1))
    fig.update_layout(
        template="plotly_white", height=460,
        title=dict(text=f"<b>{metric_key.upper()} — daily spatial-pattern correlation · {lead}h</b>",
                   x=0.01, font=dict(size=13, color="#171d1a")),
        xaxis_title="<b>Date</b>", yaxis_title=f"<b>{metric_key.upper()}</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        margin=dict(l=20, r=20, t=44, b=20),
    )
    fig.update_yaxes(range=[-1, 1])
    st.plotly_chart(fig, width="stretch")


def _build_figure(
    model_results:     dict[str, dict[int, LeadTimeData]],
    metric_key:        str,
    region:            str,
    climatology_value: float | None,
    unit:              str = "",
    clim:              "object | None" = None,
    altitude_band:     str = "all",
    season_months:     "tuple[int, ...] | None" = None,
) -> go.Figure | None:
    fig        = go.Figure()
    data_found = False
    has_clim   = isinstance(climatology_value, (int, float))
    # A season restricts the per-lead value to a subset of months; the per-pixel
    # matched-clim diff (below) is an annual-only refinement, so under a season we
    # fall back to scalar subtraction against the season's climatology mean.
    season_clim = (
        regional_season_mean(clim, region, season_months)
        if season_months and clim is not None else None
    )

    for i, (model_name, lt_map) in enumerate(model_results.items()):
        if not lt_map:
            continue

        # When subtract_clim is on (Yearly only), prefer the per-pixel diff mean
        # over the shared-finite mask — same aggregation the Spatial map uses, so
        # the chart and the map never disagree on sign (the chart's old
        # `mean(model) − mean(clim)` formula flipped sign for IDR variants
        # because clim's full-grid mean ≠ clim's mean over the model's coverage).
        clim_interp = (
            _matched_clim_for_model(model_name, lt_map, region, clim)
            if has_clim and clim is not None and not season_months else None
        )

        pairs: list[tuple[int, float]] = []
        for lt, ltd in lt_map.items():
            if season_months:
                # Season × band compose: the band's own season mean when a band
                # is selected (clim is disabled upstream in that case).
                val = (band_season_mean(ltd, altitude_band, season_months)
                       if altitude_band != "all"
                       else regional_season_mean(ltd, region, season_months))
                if has_clim:
                    val = val - (season_clim if season_clim is not None else climatology_value)
            elif clim_interp is not None:
                val = _diff_mean(ltd, clim_interp, region)
                if val is None:
                    # Fall back to scalar subtraction if per-pixel failed.
                    val = regional_annual_mean(ltd, region) - climatology_value
            elif has_clim:
                val = regional_annual_mean(ltd, region) - climatology_value
            elif altitude_band != "all":
                # Band view — pipeline-computed band mean (clim is disabled
                # upstream when a band is selected, so no clim branches here).
                val = band_annual_mean(ltd, altitude_band)
            else:
                val = regional_annual_mean(ltd, region)
            pairs.append((lt, val))
        pairs.sort()
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        color      = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        symbol     = _SYMBOLS[i % len(_SYMBOLS)]
        data_found = True

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color, symbol=symbol,
                        line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "Lead: <b>%{x}h</b><br>"
                f"{metric_key.upper()}: <b>%{{y:.4f}}</b><extra></extra>"
            ),
        ))

    if not data_found:
        return None

    if has_clim:
        fig.add_trace(go.Scatter(
            x=[min(LEAD_TIMES), max(LEAD_TIMES)],
            y=[0, 0],
            mode="lines",
            name="Climatology baseline",
            line=dict(color="#213b7d", dash="dot", width=2),
            showlegend=True,
        ))

    _u = f" ({unit})" if unit else ""
    y_title = (
        f"<b>{metric_key.upper()} Δ vs Climatology{_u}</b>"
        if has_clim else
        f"<b>{metric_key.upper()}{_u}</b>"
    )

    fig.update_layout(
        template="plotly_white",
        height=460,
        showlegend=True,
        xaxis=dict(
            title="<b>Lead Time (hours)</b>",
            title_font=dict(size=12, color="#374151"),
            tickmode="array",
            tickvals=LEAD_TIMES,
            ticktext=[f"{h}h" for h in LEAD_TIMES],
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="#F3F4F6", linecolor="#E5E7EB",
            showline=True, zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            title_font=dict(size=12, color="#374151"),
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="#F3F4F6", linecolor="#E5E7EB",
            showline=True,
            zerolinecolor="#213b7d", zerolinewidth=1,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11, color="#374151"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E5E7EB", borderwidth=1,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=16, b=20),
        hoverlabel=dict(bgcolor="white", bordercolor="#E5E7EB",
                        font=dict(size=13, color="#111827")),
    )
    return fig

# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from ui.app_config.settings import direction_note, higher_is_better, zero_is_best
from ui.components import stat_blocks
from ui.services.loader import LeadTimeData
from ui.components.spatial_map import (_add_region_boundary, _is_station,
                                       _season_field, _station_lonlat)
from ui.utils.geo import annual_spatial_grid, apply_band
from ui.utils.regions import mask_region, stations_in_region

_FIGURE_CACHE: dict[tuple, go.Figure] = {}
# Cached (grid_b_aligned, diff) per (id(grid_a), id(grid_b)). The interp
# call is the slow part of this tab (~150 ms each), so caching it makes
# slider/test-choice changes instant once a pair has been computed once.
_ALIGN_CACHE:  dict[tuple, tuple]     = {}
# Cached |grid| per source grid (bias is compared by magnitude). Each entry
# keeps a reference to its source grid so the id() key can never be recycled.
_ABS_CACHE:    dict[int, tuple]       = {}

_LOADING_CARD = """
<div style="height:420px;display:flex;flex-direction:column;align-items:center;
    justify-content:center;background:#F9FAFB;border-radius:10px;
    border:1px dashed #D1D5DB;gap:14px;font-family:'Inter',sans-serif;">
  <div style="width:38px;height:38px;border:3px solid #E5E7EB;
      border-top-color:#213b7d;border-radius:50%;
      animation:cmp-spin 0.75s linear infinite;"></div>
  <div style="font-size:13px;font-weight:600;color:#6B7280;">{label}</div>
  <style>@keyframes cmp-spin{{to{{transform:rotate(360deg);}}}}</style>
</div>
"""

# ── Diverging palette (A better = green, B better = blue) ─────────────────
# The map plots A − B, so which end of the scale means "A better" depends on
# the metric's direction: for error metrics negative favours A (green at the
# low end); for correlation metrics positive favours A (green at the high end).
_DIFF_COLORSCALE_LOWER_BETTER = [
    [0.0,  "#0a9938"],   # A much better  (Jungle Green)
    [0.35, "#6ee89a"],
    [0.5,  "#f9fafb"],   # neutral
    [0.65, "#7a9fd4"],
    [1.0,  "#213b7d"],   # B much better  (Blue)
]
_DIFF_COLORSCALE_HIGHER_BETTER = [
    [0.0,  "#213b7d"],   # B much better  (Blue)
    [0.35, "#7a9fd4"],
    [0.5,  "#f9fafb"],   # neutral
    [0.65, "#6ee89a"],
    [1.0,  "#0a9938"],   # A much better  (Jungle Green)
]


def _abs_grid(grid):
    """|grid|, cached per source grid object (used for zero-is-best metrics)."""
    entry = _ABS_CACHE.get(id(grid))
    if entry is None or entry[0] is not grid:
        entry = (grid, abs(grid))
        _ABS_CACHE[id(grid)] = entry
    return entry[1]


def render(
    model_results: dict[str, dict[int, LeadTimeData]],
    metric_key:    str,
    region:        str,
    unit:          str = "",
    altitude_band: str = "all",
    load_season=None,
    season_months: "tuple[int, ...] | None" = None,
) -> None:
    models = list(model_results.keys())

    # Direction of goodness for this metric. Bias (zero-is-best) is compared by
    # magnitude — |A| − |B| — after which "lower is better" applies.
    zb = zero_is_best(metric_key)
    hb = higher_is_better(metric_key)
    metric_disp = f"|{metric_key.upper()}|" if zb else metric_key.upper()
    note = ("compared by magnitude |bias| — closer to 0 is better"
            if zb else direction_note(metric_key))
    if altitude_band != "all":
        note += f" · altitude {altitude_band} m"

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
        <div style='border-left:4px solid #213b7d;padding:10px 16px 10px 14px;
            background:#ffffff;border-radius:0 6px 6px 0;margin-bottom:12px;
            box-shadow:0 1px 6px rgba(0,0,0,0.05);'>
          <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
            <span style='font-size:19px;'>⚖️</span>
            <span style='font-family:Inter,sans-serif;font-size:15px;
                font-weight:700;color:#171d1a;'>Models Comparison</span>
            <span style='background:rgba(33,59,125,0.09);color:#213b7d;
                border:1px solid rgba(33,59,125,0.25);border-radius:99px;
                padding:2px 9px;font-family:Inter,sans-serif;font-size:0.62rem;
                font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'>
                {metric_disp} difference</span>
          </div>
          <div style='font-family:Inter,sans-serif;font-size:11px;color:#6B7280;
              margin-top:3px;padding-left:27px;'>
              Spatial map of Model A − Model B · {note} · green = A better · blue = B better</div>
        </div>
    """, unsafe_allow_html=True)

    if len(models) < 2:
        st.info("Select at least two model variants in the sidebar to use the comparison view.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    _label = (
        "font-family:Inter,sans-serif;font-size:0.65rem;font-weight:700;"
        "letter-spacing:0.1em;text-transform:uppercase;color:#4B5563;margin-bottom:2px;"
    )
    col_a, col_b, col_lt = st.columns([2, 2, 1])
    with col_a:
        st.markdown(f"<div style='{_label}'>🟢 Model A</div>", unsafe_allow_html=True)
        model_a = st.selectbox(
            "Model A", models, index=0,
            key="cmp_model_a", label_visibility="collapsed",
        )
    with col_b:
        st.markdown(f"<div style='{_label}'>🟣 Model B</div>", unsafe_allow_html=True)
        model_b = st.selectbox(
            "Model B", models, index=min(1, len(models) - 1),
            key="cmp_model_b", label_visibility="collapsed",
        )
    with col_lt:
        st.markdown(f"<div style='{_label}'>⏱ Lead Time</div>", unsafe_allow_html=True)
        available_lts = sorted({lt for lt_map in model_results.values() for lt in lt_map})
        lt_options    = [f"{lt}h" for lt in available_lts]
        default_lt    = lt_options[len(lt_options) // 2]
        # Only seed widget state when it's missing or no longer valid —
        # NEVER when the user has just picked a new value (otherwise the
        # stale backup would overwrite the fresh selection). The
        # `_last_*` key is our private memory that survives sidebar
        # reruns; the widget key carries the live interaction.
        current = st.session_state.get("cmp_lt")
        if current not in lt_options:
            _last = st.session_state.get("_last_cmp_lt")
            if _last in lt_options:
                st.session_state["cmp_lt"] = _last
            elif _last:
                try:
                    _target = int(_last.rstrip("h"))
                    st.session_state["cmp_lt"] = min(
                        lt_options,
                        key=lambda o: abs(int(o.rstrip("h")) - _target),
                    )
                except (AttributeError, ValueError):
                    st.session_state["cmp_lt"] = default_lt
            else:
                st.session_state["cmp_lt"] = default_lt
        lt_str = st.select_slider(
            "Lead Time", options=lt_options,
            key="cmp_lt", label_visibility="collapsed",
        )
        st.session_state["_last_cmp_lt"] = lt_str
        lt = int(lt_str.rstrip("h"))

    if model_a == model_b:
        st.warning("Model A and Model B are the same — select two different models.")
        return

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Reserve a slot and paint the loading card IMMEDIATELY, before any
    # data processing (grid masking, alignment, figure build). Streamlit's
    # progressive renderer flushes this delta to the browser right away,
    # so the user sees a spinner the moment they change A/B/lead time.
    # On warm caches everything below finishes in tens of milliseconds and
    # the placeholder is replaced before the browser can paint it.
    chart_slot = st.empty()
    chart_slot.markdown(
        _LOADING_CARD.format(
            label=f"Computing {model_a} − {model_b} difference map at {lt}h…"
        ),
        unsafe_allow_html=True,
    )

    # ── Load grids (cached masked DataArrays — no on-demand recomputation) ────
    ltd_a = model_results.get(model_a, {}).get(lt)
    ltd_b = model_results.get(model_b, {}).get(lt)

    if ltd_a is None or ltd_b is None:
        chart_slot.empty()
        missing = model_a if ltd_a is None else model_b
        st.warning(f"No data for **{missing}** at {lt}h lead time.")
        return

    # Warn when the two models have different month coverage — their
    # `annual_spatial_average`s are aggregated over different temporal
    # subsets, so the per-pixel diff below mixes months and the spatial
    # map is approximate. `monthly_spatial_average` is freed for models
    # after precaching to save memory, so we can't re-aggregate to a
    # common-months subset at the per-pixel level.
    from ui.utils.geo import regional_monthly_means
    try:
        _months_a = {i + 1 for i in range(12)
                     if not np.isnan(regional_monthly_means(ltd_a, region)[i])}
        _months_b = {i + 1 for i in range(12)
                     if not np.isnan(regional_monthly_means(ltd_b, region)[i])}
    except Exception:
        _months_a, _months_b = set(), set()
    if _months_a and _months_b and _months_a != _months_b:
        _MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        _only_a = ", ".join(_MONTHS[m - 1] for m in sorted(_months_a - _months_b))
        _only_b = ", ".join(_MONTHS[m - 1] for m in sorted(_months_b - _months_a))
        _common = ", ".join(_MONTHS[m - 1] for m in sorted(_months_a & _months_b)) or "none"
        _bits = []
        if _only_a:
            _bits.append(f"**{model_a}** has extra months ({_only_a})")
        if _only_b:
            _bits.append(f"**{model_b}** has extra months ({_only_b})")
        st.warning(
            "Coverage mismatch — "
            + "; ".join(_bits)
            + f". Common months: {_common}. The spatial difference map below "
            "is computed from each model's own annual mean, so it mixes "
            "different temporal subsets across the two models."
        )

    # Station targets have a single `station` dim (no lat/lon grid), so the gridded
    # interp-align + heatmap path can't run. Pair the two models on their shared stations
    # instead and show a per-station difference scatter + the same paired tests.
    if _is_station(ltd_a.annual_spatial_average):
        _render_station_comparison(ltd_a, ltd_b, model_a, model_b, lt, metric_key,
                                   region, unit, zb, hb, season_months, load_season,
                                   chart_slot, altitude_band)
        return

    # Grids: annual (from _GRID_CACHE, built once, band-masked) or a season mean
    # (averaged over the season's months, reloaded on demand since the monthly field
    # is freed), with the same altitude-band mask applied so season × band compose.
    if not season_months:
        grid_a = annual_spatial_grid(ltd_a, region, altitude_band)
        grid_b = annual_spatial_grid(ltd_b, region, altitude_band)
    else:
        raw_a = _season_field(ltd_a, model_a, lt, season_months, load_season)
        raw_b = _season_field(ltd_b, model_b, lt, season_months, load_season)
        if raw_a is None or raw_b is None:
            chart_slot.empty()
            missing = model_a if raw_a is None else model_b
            st.warning(f"No data for **{missing}** in this season at {lt}h.")
            return
        grid_a = mask_region(apply_band(raw_a.squeeze(), ltd_a, altitude_band), region)
        grid_b = mask_region(apply_band(raw_b.squeeze(), ltd_b, altitude_band), region)
    if zb:
        grid_a, grid_b = _abs_grid(grid_a), _abs_grid(grid_b)

    align_key = (id(grid_a), id(grid_b))
    fig_key   = (model_a, model_b, lt, metric_key, region, unit,
                 altitude_band, season_months, id(ltd_a), id(ltd_b))

    # Align grids: interpolate B onto A's coordinates if needed (cached).
    if align_key in _ALIGN_CACHE:
        grid_b_aligned, diff = _ALIGN_CACHE[align_key]
    else:
        try:
            lat_name = "latitude" if "latitude" in grid_a.dims else "lat"
            lon_name = "longitude" if "longitude" in grid_a.dims else "lon"
            lat_b    = "latitude" if "latitude" in grid_b.dims else "lat"
            lon_b    = "longitude" if "longitude" in grid_b.dims else "lon"
            grid_b_aligned = grid_b.interp(
                {lat_b: grid_a[lat_name], lon_b: grid_a[lon_name]},
                method="linear",
            )
            diff = grid_a.values - grid_b_aligned.values
            _ALIGN_CACHE[align_key] = (grid_b_aligned, diff)
        except Exception as exc:
            chart_slot.empty()
            st.error(f"Could not align grids for comparison: {exc}")
            return

    fig = _FIGURE_CACHE.get(fig_key)
    if fig is None:
        fig = _build_diff_figure(grid_a, grid_b_aligned, diff, model_a, model_b,
                                 lt, metric_disp, unit, higher_better=hb, region=region)
        if fig is not None:
            _FIGURE_CACHE[fig_key] = fig

    if fig is None:
        chart_slot.empty()
        st.warning(f"Could not build difference map for {model_a} vs {model_b} at {lt}h.")
        return

    # Replace the loading card with the chart in the same slot.
    with chart_slot.container():
        st.plotly_chart(fig, width="stretch",
                        key=f"cmp_diff_{model_a}_{model_b}_{lt}")

    # ── Insight blocks ────────────────────────────────────────────────────────
    _render_stat_tests(diff, grid_a.values, grid_b_aligned.values,
                       model_a, model_b, lt, metric_key, unit)


def _render_stat_tests(diff, vals_a, vals_b, model_a, model_b, lt, metric_key, unit):
    """The section header + test-choice radio + descriptive/test cards.

    Shared by the gridded (per-pixel) and station (per-station) comparison paths so both
    targets run exactly the same paired tests — only how ``diff``/``vals_a``/``vals_b`` are
    built differs upstream.
    """
    stat_blocks.render_section_header(
        f"Statistical Insights · {model_a} vs {model_b} · {lt}h lead"
    )
    col_sel, _ = st.columns([4, 3])
    with col_sel:
        _test_options = ["T-Test", "Wilcoxon + Sign", "Both"]
        # Same persistence pattern as the lead-time slider: a non-widget
        # `_last_*` key survives Streamlit's widget garbage collection
        # across sidebar reruns, so "Both" doesn't silently revert to
        # "T-Test" (which would make render_desc_and_ttest_row draw two
        # cards instead of three).
        if st.session_state.get("cmp_test_choice") not in _test_options:
            _last_choice = st.session_state.get("_last_cmp_test_choice")
            if _last_choice in _test_options:
                st.session_state["cmp_test_choice"] = _last_choice
        test_choice = st.radio(
            "Statistical test",
            options=_test_options,
            horizontal=True,
            key="cmp_test_choice",
            label_visibility="collapsed",
        )
        st.session_state["_last_cmp_test_choice"] = test_choice
    _test = {"T-Test": "ttest", "Wilcoxon + Sign": "wilcoxon", "Both": "both"}[test_choice]
    stat_blocks.render_desc_and_ttest_row(
        vals=diff,
        name_desc=f"{model_a} − {model_b}",
        vals_a=vals_a,
        vals_b=vals_b,
        name_a=model_a,
        name_b=model_b,
        metric_key=metric_key,
        unit=unit,
        accent_desc="#213b7d",
        accent_test="#213b7d",
        test=_test,
    )


# ── Station comparison (point targets) ────────────────────────────────────

def _station_pair(a, b, region, zero_best):
    """Pair two station fields on their shared ``station`` coord; keep in-region & finite-in-both.

    Returns ``(lons, lats, vals_a, vals_b, diff, names)`` over the kept stations (possibly empty).
    ``xr.align(join="inner")`` intersects the two station indices and yields identical ordering,
    so ``vals_a``/``vals_b`` are element-wise paired (what ``stat_blocks._paired_t`` expects).
    For a zero-is-best metric (bias) both are compared by magnitude, matching the gridded path.
    """
    a = a.squeeze()
    b = b.squeeze()
    if zero_best:
        a, b = np.abs(a), np.abs(b)
    a, b = xr.align(a, b, join="inner")
    lons, lats = _station_lonlat(a)
    av = np.asarray(a.values, dtype=float)
    bv = np.asarray(b.values, dtype=float)
    keep = stations_in_region(lons, lats, region) & np.isfinite(av) & np.isfinite(bv)
    names = (np.asarray(a["name"].values) if "name" in a.coords
             else np.asarray(a["station"].values).astype(str))
    return lons[keep], lats[keep], av[keep], bv[keep], av[keep] - bv[keep], names[keep]


def _render_station_comparison(ltd_a, ltd_b, model_a, model_b, lt, metric_key,
                               region, unit, zero_best, higher_better,
                               season_months, load_season, chart_slot,
                               altitude_band="all"):
    """Model-A − Model-B comparison for a station target: per-station diff scatter + paired tests."""
    # Season-aware fields, mirroring the gridded branch: annual, or the season mean.
    if not season_months:
        a, b = ltd_a.annual_spatial_average, ltd_b.annual_spatial_average
    else:
        a = _season_field(ltd_a, model_a, lt, season_months, load_season)
        b = _season_field(ltd_b, model_b, lt, season_months, load_season)
        if a is None or b is None:
            chart_slot.empty()
            missing = model_a if a is None else model_b
            st.warning(f"No data for **{missing}** in this season at {lt}h.")
            return

    # Altitude filter: out-of-band stations become NaN and drop out of the pairing.
    a = apply_band(a, ltd_a, altitude_band)
    b = apply_band(b, ltd_b, altitude_band)

    lons, lats, vals_a, vals_b, diff, names = _station_pair(a, b, region, zero_best)
    if diff.size == 0:
        chart_slot.empty()
        st.warning(f"No common in-region stations for **{model_a}** vs **{model_b}** at {lt}h.")
        return

    metric_disp = f"|{metric_key.upper()}|" if zero_best else metric_key.upper()
    fig = _build_station_diff_figure(lons, lats, diff, names, model_a, model_b, lt,
                                     metric_disp, unit, higher_better, region)
    with chart_slot.container():
        st.plotly_chart(fig, width="stretch", key=f"cmp_stdiff_{model_a}_{model_b}_{lt}")

    _render_stat_tests(diff, vals_a, vals_b, model_a, model_b, lt, metric_key, unit)


def _build_station_diff_figure(lons, lats, diff, names, model_a, model_b, lead_time,
                               metric_disp, unit, higher_better, region):
    """Diverging per-station A−B scatter (green = A better, blue = B better)."""
    valid = diff[np.isfinite(diff)]
    m = float(np.nanpercentile(np.abs(valid), 98)) if valid.size else 1.0
    if not np.isfinite(m) or m <= 0:
        m = 1.0
    _u = f" ({unit})" if unit else ""
    fig = go.Figure(go.Scattergeo(
        lon=lons, lat=lats, mode="markers",
        text=[str(n) for n in names],
        marker=dict(
            size=10, color=diff,
            colorscale=(_DIFF_COLORSCALE_HIGHER_BETTER if higher_better
                        else _DIFF_COLORSCALE_LOWER_BETTER),
            cmin=-m, cmax=m, line=dict(color="#374151", width=0.5),
            colorbar=dict(
                title=dict(text=f"<b>{metric_disp}{_u}</b><br>{model_a} − {model_b}",
                           side="right", font=dict(size=11, color="#374151")),
                thickness=10, len=0.8, tickfont=dict(size=9, color="#6B7280")),
        ),
        hovertemplate=(f"<b>%{{text}}</b><br>{model_a} − {model_b}: %{{marker.color:.4f}}{_u}"
                       "<br>Lat %{lat:.2f}°  Lon %{lon:.2f}°<extra></extra>"),
    ))
    lat_min, lat_max = float(np.min(lats)) - 0.5, float(np.max(lats)) + 0.5
    lon_min, lon_max = float(np.min(lons)) - 0.5, float(np.max(lons)) + 0.5
    fig.update_layout(
        template="plotly_white", height=500,
        title=dict(text=(f"<b>{model_a}</b><span style='color:#6B7280;'> − </span>"
                         f"<b>{model_b}</b>  <span style='color:#6B7280;font-size:11px;'>"
                         f"{lead_time}h lead · per station</span>"),
                   x=0.03, y=0.97, font=dict(size=13, color="#171d1a", family="Inter")),
        geo=dict(
            showocean=True, oceancolor="#D6ECFA", showlakes=True, lakecolor="#D6ECFA",
            showland=True, landcolor="#F3F4F6", showcoastlines=True, coastlinecolor="#1F2937",
            coastlinewidth=1.0, showcountries=True, countrycolor="#6B7280", countrywidth=0.7,
            projection_type="equirectangular",
            lataxis_range=[lat_min, lat_max], lonaxis_range=[lon_min, lon_max],
            bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=44, b=0), paper_bgcolor="white",
    )
    _add_region_boundary(fig, region)
    return fig


# ── Pure figure builder — no Streamlit calls ──────────────────────────────

def _build_diff_figure(
    grid_a:       "xr.DataArray",
    grid_b_aligned: "xr.DataArray | np.ndarray",
    diff:         np.ndarray,
    model_a:      str,
    model_b:      str,
    lead_time:    int,
    metric_disp:  str,
    unit:         str = "",
    higher_better: bool = False,
    region:       str = "",
) -> "go.Figure | None":
    try:
        lats = grid_a.latitude.values
        lons = grid_a.longitude.values
    except AttributeError:
        lats = grid_a.lat.values if "lat" in grid_a.coords else np.arange(grid_a.shape[0])
        lons = grid_a.lon.values if "lon" in grid_a.coords else np.arange(grid_a.shape[1])

    valid   = diff[~np.isnan(diff)]
    if valid.size == 0:
        return None
    abs_max = float(np.nanpercentile(np.abs(valid), 98))

    lat_min, lat_max = float(lats.min()) - 0.5, float(lats.max()) + 0.5
    lon_min, lon_max = float(lons.min()) - 0.5, float(lons.max()) + 0.5
    _u = f" ({unit})" if unit else ""

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=diff, x=lons, y=lats,
        colorscale=(_DIFF_COLORSCALE_HIGHER_BETTER if higher_better
                    else _DIFF_COLORSCALE_LOWER_BETTER),
        reversescale=False,
        zmin=-abs_max, zmax=abs_max,
        zsmooth=False, connectgaps=False, opacity=0.90,
        colorbar=dict(
            title=dict(
                text=f"<b>{metric_disp}{_u}</b><br>{model_a} − {model_b}",
                side="right", font=dict(size=11, color="#374151"),
            ),
            thickness=10, len=0.80,
            x=0.88, xanchor="left", xpad=4,
            tickfont=dict(size=9, color="#6B7280"),
            outlinewidth=0, bgcolor="rgba(255,255,255,0.75)",
        ),
        hovertemplate=(
            f"<b>{model_a} − {model_b}</b>: %{{z:.4f}}{f' {unit}' if unit else ''}<br>"
            "Lat: %{y:.2f}°  Lon: %{x:.2f}°<extra></extra>"
        ),
    ))
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
            showocean=True,      oceancolor="#D6ECFA",
            showlakes=True,      lakecolor="#D6ECFA",
            showland=True,       landcolor="rgba(0,0,0,0)",
            showcoastlines=True, coastlinecolor="#1F2937", coastlinewidth=1.2,
            showcountries=True,  countrycolor="#6B7280",  countrywidth=0.7,
            showframe=True,      framecolor="#9CA3AF",
            projection_type="equirectangular",
            lataxis_range=[lat_min, lat_max],
            lonaxis_range=[lon_min, lon_max],
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text=(
                f"<b>{model_a}</b>"
                f"<span style='color:#6B7280;'> − </span>"
                f"<b>{model_b}</b>"
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

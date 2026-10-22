# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ui.app_config.settings import LEAD_TIMES, MODEL_PALETTE
from ui.services.loader import LeadTimeData
from ui.utils.geo import band_monthly_means, regional_monthly_means

_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_FIGURE_CACHE: dict[tuple, go.Figure] = {}

_LOADING_CARD = """
<div style="height:460px;display:flex;flex-direction:column;align-items:center;
    justify-content:center;background:#F9FAFB;border-radius:10px;
    border:1px dashed #D1D5DB;gap:14px;font-family:'Inter',sans-serif;">
  <div style="width:38px;height:38px;border:3px solid #E5E7EB;
      border-top-color:#0a9938;border-radius:50%;
      animation:mc-spin 0.75s linear infinite;"></div>
  <div style="font-size:13px;font-weight:600;color:#6B7280;">{label}</div>
  <style>@keyframes mc-spin{{to{{transform:rotate(360deg);}}}}</style>
</div>
"""


def render_header(metric_key: str, clim_monthly: "np.ndarray | None" = None) -> None:
    """Render just the section header card.

    Exposed so callers can place additional controls (like the
    "Average across years" multiselect) between the header and the
    chart's own Lead-Time / Month-Range controls.
    """
    st.markdown(f"""
        <div style='border-left:4px solid #f2b029;padding:10px 16px 10px 14px;
            background:#ffffff;border-radius:0 6px 6px 0;margin-bottom:12px;
            box-shadow:0 1px 6px rgba(0,0,0,0.05);'>
          <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
            <span style='font-size:19px;'>📅</span>
            <span style='font-family:Inter,sans-serif;font-size:15px;
                font-weight:700;color:#171d1a;'>Seasonal Evaluation</span>
            <span style='background:rgba(242,176,41,0.12);color:#b45309;
                border:1px solid rgba(242,176,41,0.4);border-radius:99px;
                padding:2px 9px;font-family:Inter,sans-serif;font-size:0.62rem;
                font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'>
                {metric_key.upper()}</span>
            {"<span style='background:rgba(33,59,125,0.09);color:#213b7d;border:1px solid rgba(33,59,125,0.25);border-radius:99px;padding:2px 9px;font-family:Inter,sans-serif;font-size:0.62rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;'>vs Climatology</span>" if clim_monthly is not None else ""}
          </div>
          <div style='font-family:Inter,sans-serif;font-size:11px;color:#6B7280;
              margin-top:3px;padding-left:27px;'>
              Monthly breakdown of forecast error at the selected lead time</div>
        </div>
    """, unsafe_allow_html=True)


def render(
    model_results:          dict[str, dict[int, LeadTimeData]],
    metric_key:             str,
    region:                 str                              = "Africa",
    clim_monthly:           np.ndarray | None               = None,
    model_monthly_overrides: dict[str, dict[int, np.ndarray]] | None = None,
    unit:                   str                              = "",
    incomplete_info:        dict[str, dict[str, list[str]]] | None = None,
    include_header:         bool                              = True,
    altitude_band:          str                               = "all",
) -> None:
    if include_header:
        render_header(metric_key, clim_monthly)

    available_lts = sorted({
        lt for lt_map in model_results.values() for lt in lt_map
    })
    if not available_lts:
        _render_empty_state("No lead-time data available")
        return

    col_lt, col_months = st.columns([1, 2], gap="large")

    _label_style = (
        "font-family:Inter,sans-serif;font-size:0.65rem;font-weight:700;"
        "letter-spacing:0.1em;text-transform:uppercase;color:#4B5563;"
        "margin-bottom:2px;"
    )

    with col_lt:
        st.markdown(f"<div style='{_label_style}'>⏱ Lead Time</div>", unsafe_allow_html=True)
        lt_options  = [f"{lt}h" for lt in available_lts]
        default_lt  = lt_options[len(lt_options) // 2]
        # Only seed widget state when it's missing or no longer valid —
        # NEVER when the user has just picked a new value (otherwise the
        # stale backup would overwrite the fresh selection). The
        # `_last_*` key is our private memory that survives sidebar
        # reruns; the widget key carries the live interaction.
        current = st.session_state.get("monthly_lt_slider")
        if current not in lt_options:
            _last = st.session_state.get("_last_monthly_lt")
            if _last in lt_options:
                st.session_state["monthly_lt_slider"] = _last
            elif _last:
                try:
                    _target = int(_last.rstrip("h"))
                    st.session_state["monthly_lt_slider"] = min(
                        lt_options,
                        key=lambda o: abs(int(o.rstrip("h")) - _target),
                    )
                except (AttributeError, ValueError):
                    st.session_state["monthly_lt_slider"] = default_lt
            else:
                st.session_state["monthly_lt_slider"] = default_lt
        selected_lt_str = st.select_slider(
            "Lead Time",
            options=lt_options,
            key="monthly_lt_slider",
            label_visibility="collapsed",
        )
        st.session_state["_last_monthly_lt"] = selected_lt_str
        selected_lt = int(selected_lt_str.rstrip("h"))

    with col_months:
        st.markdown(f"<div style='{_label_style}'>📅 Month Range</div>", unsafe_allow_html=True)
        c_from, c_to = st.columns(2)
        with c_from:
            month_start = st.selectbox(
                "From", options=_MONTH_LABELS, index=0,
                key="monthly_start", label_visibility="visible",
            )
        with c_to:
            month_end = st.selectbox(
                "To", options=_MONTH_LABELS, index=11,
                key="monthly_end", label_visibility="visible",
            )

    i_start = _MONTH_LABELS.index(month_start)
    i_end   = _MONTH_LABELS.index(month_end)

    # Support wrap-around seasons like DJF (Dec → Feb)
    if i_start <= i_end:
        month_indices = list(range(i_start, i_end + 1))
    else:
        month_indices = list(range(i_start, 12)) + list(range(0, i_end + 1))

    labels_sel = [_MONTH_LABELS[i] for i in month_indices]

    # Paint the loading card IMMEDIATELY, before fingerprint computation
    # or the figure build runs. Same pattern as the Lead-Time and
    # Comparison tabs: gives the user a spinner the moment they change a
    # sidebar option, and is replaced by the chart in the same slot.
    chart_slot = st.empty()
    chart_slot.markdown(
        _LOADING_CARD.format(label=f"Building seasonal chart at {selected_lt}h…"),
        unsafe_allow_html=True,
    )

    clim_key = round(float(np.mean(clim_monthly)), 10) if clim_monthly is not None else None
    fp  = _fingerprint(model_results, metric_key, region, selected_lt, clim_key,
                       tuple(month_indices), model_monthly_overrides) + (unit, altitude_band)
    fig = _FIGURE_CACHE.get(fp)
    if fig is None:
        fig = _build_figure(
            model_results, metric_key, region, selected_lt,
            clim_monthly, labels_sel, month_indices, model_monthly_overrides, unit,
            altitude_band,
        )
        if fig is not None:
            _FIGURE_CACHE[fp] = fig

    # Warn about models that have no data at the selected lead time
    no_data = [
        name for name, lt_map in model_results.items()
        if model_monthly_overrides and name in model_monthly_overrides
            and model_monthly_overrides[name].get(selected_lt) is None
        or not (model_monthly_overrides and name in model_monthly_overrides)
            and selected_lt not in lt_map
    ]
    if no_data:
        st.warning(
            f"No data at **{selected_lt}h** lead time for: "
            + ", ".join(f"**{m}**" for m in no_data)
        )

    # Show missing months only for models that DO have data at the selected lead time
    if incomplete_info:
        no_data_set = set(no_data)
        parts = "; ".join(
            "**{model}** — {yr_parts}".format(
                model=model,
                yr_parts=", ".join(
                    f"{yr}: missing {', '.join(months)}"
                    for yr, months in sorted(yr_map.items())
                ),
            )
            for model, yr_map in sorted(incomplete_info.items())
            if model not in no_data_set
        )
        if parts:
            st.info(f"Missing months — {parts}. Those months are excluded from the average.")

    if fig is None:
        chart_slot.empty()
        _render_empty_state(f"No monthly data at {selected_lt}h lead time")
        return

    # Replace the loading placeholder with the chart in the same slot.
    with chart_slot.container():
        st.plotly_chart(
            fig, width="stretch",
            key=f"monthly_chart_{selected_lt}_{'_'.join(map(str, month_indices))}",
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _fingerprint(
    model_results:           dict[str, dict[int, LeadTimeData]],
    metric_key:              str,
    region:                  str,
    lead_time:               int,
    clim_key:                float | None,
    month_indices:           tuple[int, ...],
    model_monthly_overrides: dict[str, dict[int, np.ndarray]] | None,
) -> tuple:
    if model_monthly_overrides is not None:
        override_key = tuple(
            (name, round(float(np.mean(lt_map.get(lead_time, np.zeros(12)))), 8))
            for name, lt_map in sorted(model_monthly_overrides.items())
        )
    else:
        override_key = tuple(
            (
                name,
                round(float(np.mean(regional_monthly_means(lt_map[lead_time], region))), 8)
                if lead_time in lt_map else None,
            )
            for name, lt_map in sorted(model_results.items())
        )
    return (metric_key, region, lead_time, override_key, clim_key, month_indices)


def _build_figure(
    model_results:           dict[str, dict[int, LeadTimeData]],
    metric_key:              str,
    region:                  str,
    lead_time:               int,
    clim_monthly:            np.ndarray | None                    = None,
    labels_sel:              list[str]                            = _MONTH_LABELS,
    month_indices:           list[int]                            = list(range(12)),
    model_monthly_overrides: dict[str, dict[int, np.ndarray]] | None = None,
    unit:                    str                                  = "",
    altitude_band:           str                                  = "all",
) -> go.Figure | None:
    fig        = go.Figure()
    data_found = False
    has_clim   = clim_monthly is not None and len(clim_monthly) == 12

    for i, (model_name, lt_map) in enumerate(model_results.items()):
        if model_monthly_overrides is not None:
            # Override mode (caller supplied multi-year aggregations) —
            # only plot models that survived the multi-year aggregation.
            # NEVER fall back to single-year data here, otherwise a model
            # with no data for the selected "Average across years" would
            # be silently drawn using the sidebar year instead.
            if model_name not in model_monthly_overrides:
                continue
            monthly = model_monthly_overrides[model_name].get(lead_time)
            if monthly is None:
                continue
        else:
            ltd = lt_map.get(lead_time)
            if ltd is None:
                continue
            monthly = (band_monthly_means(ltd, altitude_band)
                       if altitude_band != "all"
                       else regional_monthly_means(ltd, region))

        monthly = np.asarray(monthly).squeeze()
        if monthly.ndim != 1:
            continue
        if monthly.shape[0] < 12:
            # Pad missing months with NaN so the model still appears in the chart
            padded = np.full(12, np.nan)
            padded[:monthly.shape[0]] = monthly
            monthly = padded
        elif monthly.shape[0] > 12:
            monthly = monthly[:12]

        color      = MODEL_PALETTE[i % len(MODEL_PALETTE)]
        data_found = True

        fig.add_trace(go.Scatter(
            x=labels_sel,
            y=monthly[month_indices].tolist(),
            mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color, line=dict(color="white", width=1.5)),
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "Month: <b>%{x}</b><br>"
                f"{metric_key.upper()}: <b>%{{y:.4f}}</b><extra></extra>"
            ),
        ))

    if not data_found:
        return None

    if has_clim:
        fig.add_trace(go.Scatter(
            x=labels_sel,
            y=clim_monthly[month_indices].tolist(),
            mode="lines+markers",
            name="Climatology",
            line=dict(color="#213b7d", dash="dot", width=2),
            marker=dict(size=6, color="#213b7d", symbol="diamond",
                        line=dict(color="white", width=1)),
            hovertemplate=(
                "<b>Climatology</b><br>"
                "Month: <b>%{x}</b><br>"
                f"{metric_key.upper()}: <b>%{{y:.4f}}</b><extra></extra>"
            ),
        ))

    _u = f" ({unit})" if unit else ""
    y_title = f"<b>{metric_key.upper()}{_u}</b>"

    fig.update_layout(
        template="plotly_white",
        height=460,
        showlegend=True,
        title=dict(
            text=(
                f"<b>{lead_time}h</b>"
                f"  <span style='color:#6B7280;font-size:11px;'>lead time</span>"
            ),
            x=0.01, y=0.98,
            font=dict(size=13, color="#171d1a", family="Inter"),
        ),
        xaxis=dict(
            title="<b>Month</b>",
            title_font=dict(size=12, color="#374151"),
            tickfont=dict(size=11, color="#6B7280"),
            gridcolor="#F3F4F6", linecolor="#E5E7EB",
            showline=True, zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            title_font=dict(size=12, color="#374151"),
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="#F3F4F6", linecolor="#E5E7EB",
            showline=True, zeroline=False,
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


def _render_empty_state(message: str) -> None:
    st.markdown(
        f"""<div style="height:420px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;background:#F9FAFB;
            border-radius:10px;border:1px dashed #D1D5DB;color:#9CA3AF;
            font-family:'Inter',sans-serif;">
            <div style="font-size:32px;margin-bottom:12px;">📅</div>
            <div style="font-size:13px;font-weight:600;">No data available</div>
            <div style="font-size:11px;margin-top:4px;">{message}</div>
        </div>""",
        unsafe_allow_html=True,
    )

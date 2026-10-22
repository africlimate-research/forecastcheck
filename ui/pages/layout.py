# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
from ui.pages.header import render as render_header
from ui.pages.home.home import render as render_home
from ui.pages.threshold.threshold import render as render_threshold
from ui.pages.verification.verification import \
    render as render_verification
from ui.utils.styling import inject_css

#: Package directories resolved relative to this file so assets/CSS load
#: regardless of the working directory Streamlit is launched from.
_UI     = Path(__file__).resolve().parent.parent
_STYLES = _UI / "components" / "styles"
_ASSETS = _UI / "assets"


def render() -> None:
    inject_css(_STYLES / "verification.css")
    inject_css(_STYLES / "tabs.css")
    inject_css(_STYLES / "control_center.css")
    inject_css(_STYLES / "header.css")
    _inject_favicon()

    render_header()

    tab_home, tab_medium, tab_threshold, tab_seasonal = st.tabs([
        "🏠  Home",
        "📈  Short & Medium-Range",
        "🎯  Threshold Diagnostics",
        "🗓️  Seasonal Forecast",
    ])

    with tab_home:
        render_home()

    with tab_medium:
        render_verification()

    with tab_threshold:
        render_threshold()

    with tab_seasonal:
        _render_seasonal_placeholder()

    st.markdown(_footer_html(), unsafe_allow_html=True)


def _inject_favicon() -> None:
    st.markdown(
        "<link rel='shortcut icon' type='image/svg+xml' href='/app/static/favicon.svg'>",
        unsafe_allow_html=True,
    )


def _svg_img(path: Path, style: str) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f"<img src='data:image/svg+xml;base64,{data}' style='{style}' alt=''>"


@st.cache_resource
def _footer_html() -> str:
    """Build footer HTML once at startup — SVG encoding is expensive."""
    logo_acai = _svg_img(
        _ASSETS / "AfriClimate Symbol.svg",
        "width:44px;height:44px;border-radius:10px;object-fit:contain;"
        "background:rgba(255,255,255,0.12);padding:6px;flex-shrink:0;",
    ) or "<span style='font-size:2rem;'>🌍</span>"

    logo_f4a = _svg_img(
        _ASSETS / "F4A (Horizontal Condensed).svg",
        "height:44px;width:auto;max-width:130px;border-radius:8px;object-fit:contain;"
        "background:rgba(255,255,255,0.10);padding:6px 10px;flex-shrink:0;"
        "border:1px solid rgba(242,176,41,0.30);",
    )

    return f"""
        <div style='
            position:relative;
            left:50%;
            transform:translateX(-50%);
            width:100vw;
            margin-top:48px;
            background:#1a2e6e;
            padding:32px clamp(2rem, 6vw, 6rem);
            display:flex;
            align-items:center;
            gap:24px;
            box-shadow:0 -4px 24px rgba(26,46,110,0.25);'>
          {logo_acai}
          <div style='flex:1;'>
            <div style='font-family:Inter,sans-serif;font-size:0.85rem;
                color:rgba(255,255,255,0.95);line-height:1.7;'>
              <b>AfriClimate AI Verification Platform</b> is developed and
              maintained by <b>AfriClimate AI</b> as part of the
              <b>Forecast4Africa</b> initiative.
            </div>
            <div style='font-family:Inter,sans-serif;font-size:0.72rem;
                color:rgba(255,255,255,0.55);margin-top:5px;line-height:1.6;'>
              Data and results are released under the
              <span style='color:rgba(255,255,255,0.75);'>Creative Commons
              Attribution 4.0 International (CC BY 4.0)</span> licence.
              Please cite the companion manuscript when using this resource.
            </div>
            <div style='font-family:Inter,sans-serif;font-size:0.72rem;
                color:rgba(255,255,255,0.55);margin-top:3px;'>
              Queries, collaborations, or data contributions:
              <a href='mailto:research@africlimate.ai'
                style='color:#6ee89a;text-decoration:none;font-weight:600;'>
                research@africlimate.ai</a>
            </div>
          </div>
          {logo_f4a}
        </div>
    """


def _render_seasonal_placeholder() -> None:
    st.markdown("""
        <div style='max-width:640px;margin:80px auto 0;text-align:center;
            font-family:Inter,sans-serif;'>
          <div style='font-size:48px;margin-bottom:20px;'>🗓️</div>
          <div style='font-size:22px;font-weight:800;color:#171d1a;
              margin-bottom:10px;'>Seasonal Forecast Verification</div>
          <div style='font-size:14px;color:#6B7280;line-height:1.7;
              margin-bottom:32px;'>
            Verification of monthly-to-seasonal lead-time forecasts
            (1 – 6 month horizons).<br>
            This section is under active development.
          </div>
          <div style='display:inline-flex;align-items:center;gap:8px;
              background:rgba(10,153,56,0.08);border:1px solid rgba(10,153,56,0.25);
              border-radius:99px;padding:6px 18px;'>
            <span style='width:8px;height:8px;border-radius:50%;
                background:#0a9938;display:inline-block;
                animation:pulse 1.8s ease-in-out infinite;'></span>
            <span style='font-size:12px;font-weight:700;color:#0a9938;
                letter-spacing:0.08em;text-transform:uppercase;'>Coming soon</span>
          </div>
        </div>
        <style>
          @keyframes pulse {
            0%,100% { opacity:1; } 50% { opacity:.3; }
          }
        </style>
    """, unsafe_allow_html=True)

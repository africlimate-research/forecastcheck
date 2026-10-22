# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

_ASSETS   = Path(__file__).resolve().parent.parent / "assets"
_LOGO     = _ASSETS / "AfriClimate Symbol.svg"
_LOGO_F4A = _ASSETS / "F4A (Horizontal Condensed).svg"


def _img_b64(path: Path, css_class: str) -> str:
    if path.exists():
        data = base64.b64encode(path.read_bytes()).decode()
        mime = "svg+xml" if path.suffix == ".svg" else path.suffix.lstrip(".")
        return f"<img class='{css_class}' src='data:image/{mime};base64,{data}' alt=''>"
    return "<div class='hdr-logo-fallback'>🌍</div>"


@st.cache_resource
def _header_html() -> str:
    logo     = _img_b64(_LOGO,     "hdr-logo")
    logo_f4a = _img_b64(_LOGO_F4A, "hdr-logo-f4a")
    return f"""
    <div class="hdr-wrapper">
      <div class="hdr-left">
        {logo}
        <div class="hdr-titles">
          <p class="hdr-org">AfriClimate AI</p>
          <p class="hdr-title">AI Forecast <span>Verification</span> Platform</p>
          <p class="hdr-subtitle">
            Comprehensive analysis of atmospheric model performance vs. ground truth
          </p>
        </div>
      </div>
      <div class="hdr-right">
        <div class="hdr-right-inner">
          <div class="hdr-badges">
            <span class="hdr-badge">
              <span class="hdr-dot"></span>Verification
            </span>
          </div>
          {logo_f4a}
        </div>
      </div>
    </div>
    """


def render() -> None:
    st.markdown(_header_html(), unsafe_allow_html=True)

# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import streamlit as st


def inject_css(css_path: Path) -> None:
    if not css_path.exists():
        return
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

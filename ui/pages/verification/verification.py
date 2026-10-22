# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import streamlit as st
from ui.components import scores_tab
from ui.components.filters import render_filters


def render() -> None:
    col_filters, col_charts = st.columns([1, 3.5], gap="large")

    with col_filters:
        filters = render_filters()

    with col_charts:
        scores_tab.show(filters)

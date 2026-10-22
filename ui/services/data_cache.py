# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Shared, cached access to discovery — the dashboard's data entry point."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from ui.services.menu import MenuIndex


def ui_results_root() -> Path:
    """The single results root the menu and loader both read from.

    Delegates to the deployment's :class:`verification.store.CountryStore`, which is the
    same object ``verification.run`` writes through — so the dashboard reads exactly
    where the pipeline wrote, by construction rather than by convention. A mismatch
    would be silent: discovery would return an empty index and the app offer nothing.
    """
    from verification.store import CountryStore
    return CountryStore.current().diagnostics_root()


@st.cache_resource(show_spinner=False)
def get_menu_index() -> MenuIndex:
    """The discovery-backed menu source, built once per process.

    Scans :func:`ui_results_root` — the same root the loader reads — so the
    menu can only offer runs the loader can actually open. New runs surface
    on the next app restart / cache clear.
    """
    return MenuIndex.from_discovery(root=ui_results_root())

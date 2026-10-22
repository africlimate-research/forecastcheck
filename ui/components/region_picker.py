# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""The Country & Region selector, shared by every page that has one.

One widget, shared by every page. ``st.pills`` renders every option as a button, so the
catalogue's hierarchy splits the list:

* **pills** — the country and its top level (Senegal's 14 regions).
* **a searchable dropdown** — every finer level (the 45 departments), collapsed by
  default. When the finest level *is* the top one it is not rendered at all.

Regions with no data are left out, and what is left out is stated in a caption rather
than silently dropped.
"""
from __future__ import annotations

from functools import lru_cache

import streamlit as st

from ui.app_config.settings import DEFAULT_REGION, REGIONS

#: Placeholder for "nothing chosen in the dropdown". A sentinel rather than an
#: empty string so it cannot collide with a real region name.
_NONE = "—"


def render_region_picker(key_prefix: str) -> str:
    """Draw the Country & Region selector and return the chosen region name.

    ``key_prefix`` namespaces the widget state so several pages can each hold their own
    selection (Streamlit keys are global). Returns a display name from
    :data:`ui.app_config.settings.REGIONS`, always resolvable by the masking layer.
    """
    top, rest = _split_by_level()
    pills_key, more_key = f"{key_prefix}_region_pick", f"{key_prefix}_region_more"

    # Seed the initial selection in session state rather than passing ``default=`` /
    # ``index=`` to the widgets. Streamlit warns when a widget carries both a default and
    # a session-state value, because the state silently wins and the default is dead
    # code — and these widgets must write their state, to clear one another. Seeding once
    # gives the same first render with one owner of the value.
    st.session_state.setdefault(pills_key, DEFAULT_REGION)
    st.session_state.setdefault(more_key, _NONE)

    head_l, head_r = st.columns([6, 1], vertical_alignment="bottom")
    with head_l:
        st.markdown("<h2 style='margin:26px 0 8px;'>Country &amp; Region</h2>",
                    unsafe_allow_html=True)
    with head_r:
        # Safe to assign here: the widgets below have not been created yet this run.
        if st.button("Reset ›", key=f"{key_prefix}_reset", type="tertiary"):
            st.session_state[pills_key] = DEFAULT_REGION
            st.session_state[more_key] = _NONE
            st.rerun()

    # The two widgets must not disagree about what is selected, and only one of them can
    # own the answer. Clearing the other is done in an on_change CALLBACK rather than
    # inline: Streamlit refuses to let a widget's state be written once that widget has
    # been instantiated this run, so doing it inline raises as soon as a user picks
    # anything. Callbacks run before the next run builds its widgets, where it is legal.
    def _clear_more():
        st.session_state[more_key] = _NONE

    def _clear_pills():
        st.session_state[pills_key] = None

    with st.container(key=f"{key_prefix}_countries"):
        picked = st.pills("Region ›", options=top, selection_mode="single",
                          key=pills_key, on_change=_clear_more if rest else None)

    if not rest:
        return picked or DEFAULT_REGION

    with st.expander(f"More regions ({len(rest)}) ›", expanded=False):
        finer = st.selectbox(
            "Search a district or municipality", options=[_NONE, *rest],
            key=more_key, on_change=_clear_pills,
            help="Type to filter. Only regions containing at least one station are listed.")
        caption = _coverage_caption()
        if caption:
            st.caption(caption)

    # Whichever widget currently holds a value wins; the callbacks guarantee at most one
    # does. Falling back to the country keeps a region always selected.
    if finer and finer != _NONE:
        return finer
    return picked or DEFAULT_REGION


def _split_by_level() -> tuple[list, list]:
    """``(pills, dropdown)`` — the country plus its top level, then everything finer.

    Falls back to putting everything in the pills when there is no catalogue to consult
    (the pan-African build, or a country with no boundaries onboarded), which is the
    behaviour these pages had before this widget existed.
    """
    catalogue = _catalogue()
    if catalogue is None or catalogue.pool_level == 0:
        return list(REGIONS), []

    levels = catalogue.levels()
    top_names = {catalogue.display_label(r.id) for r in catalogue.at_level(levels[0])}
    offered = set(_regions_with_data())

    top = [r for r in REGIONS if r == DEFAULT_REGION or (r in top_names and r in offered)]
    rest = [r for r in REGIONS
            if r != DEFAULT_REGION and r not in top_names and r in offered]
    return top, rest


def _catalogue():
    """This deployment's region catalogue, or None if it cannot be built."""
    try:
        from verification import admin

        return admin.catalogue_for()
    except Exception:  # noqa: BLE001 — a picker must never take the page down
        return None


@lru_cache(maxsize=4)
def _with_data(country: str | None) -> frozenset:
    """Display names of regions holding at least one station, at every level.

    Computed from the gauge coordinates rather than from the scored output: it is one
    cheap point-in-polygon pass, it does not change as the user switches model or year,
    and a picker that reshuffled itself per selection would be worse than one that
    occasionally offers a region with no data for *this* year — which the page already
    reports as "no data" rather than drawing something wrong.

    An empty result means "no opinion": every region is offered. That is the right answer
    for a gridded country, where every region contains cells anyway.
    """
    from verification import admin
    from verification.store import CountryStore

    catalogue = admin.catalogue_for(country)
    if catalogue is None or catalogue.pool_level == 0:
        return frozenset()

    store = CountryStore.current()
    # A GRIDDED truth covers the whole country, so every region has observations and
    # narrowing by where the gauges happen to be would hide regions that do have data.
    if "gridded" in store.targets():
        return frozenset()

    coords = _station_coords(store)
    if coords is None:
        return frozenset()

    import numpy as np

    lons, lats = coords
    labels = admin.assign_points(catalogue, lons, lats)
    index = admin.leaf_index(catalogue)
    holding = {r.id for r in catalogue.leaves()
               if r.id != admin.UNASSIGNED_ID and bool((labels == index[r.id]).any())}
    if not holding:
        return frozenset()
    return frozenset(
        catalogue.display_label(r.id)
        for level in catalogue.levels()
        for r in catalogue.at_level(level)
        if r.id != admin.UNASSIGNED_ID and set(catalogue.leaves_under(r.id)) & holding)


def _station_coords(store):
    """``(lons, lats)`` of this country's gauges, or None if it has no station tree.

    Sweeps every product the country HOLDS rather than the one it is configured to
    write to. The dashboard is product-agnostic — it shows whatever is on disk — so
    binding this to the configured product would let a deployment whose product has no
    gauge tree lose its station coordinates silently, and the failure is invisible: the
    picker just stops narrowing regions to the ones containing gauges.
    """
    import xarray as xr

    try:
        if "station" not in store.targets():
            return None
        products = sorted(store.products()) or [store.product]
        for variable in sorted(store.variables("station")):
            for year in _candidate_years(store, variable):
                path = next(
                    (p for p in (store.truth(target="station", variable=variable,
                                             year=year, product=prod)
                                 for prod in products)
                     if p is not None and p.exists()), None)
                if path is not None:
                    with xr.open_dataset(path) as ds:
                        lat = "station_lat" if "station_lat" in ds.coords else "lat"
                        lon = "station_lon" if "station_lon" in ds.coords else "lon"
                        if lat in ds.coords and lon in ds.coords:
                            import numpy as np

                            return (np.asarray(ds[lon].values, float),
                                    np.asarray(ds[lat].values, float))
    except Exception:  # noqa: BLE001 — no coordinates just means "offer everything"
        return None
    return None


def _candidate_years(store, variable):
    """Years to look for a truth file in, newest first."""
    import datetime

    this_year = datetime.date.today().year
    return range(this_year, this_year - 12, -1)


def _regions_with_data() -> frozenset:
    """Regions worth offering for this deployment, or every region if unknown."""
    from verification import config

    found = _with_data(config.country())
    return found if found else frozenset(REGIONS)


def _coverage_caption() -> str:
    """One line saying what was left out, so an absence is explained rather than puzzling."""
    catalogue = _catalogue()
    if catalogue is None or catalogue.pool_level == 0:
        return ""
    offered = _regions_with_data()
    finest = [r for r in catalogue.at_level(catalogue.pool_level)
              if r.id != "unassigned"]
    shown = [r for r in finest if catalogue.display_label(r.id) in offered]
    if len(shown) == len(finest):
        return ""
    return (f"{len(shown)} of {len(finest)} hold at least one station; "
            f"the rest are not listed because they have nothing to plot.")

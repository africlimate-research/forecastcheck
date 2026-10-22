# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Region definitions and spatial geometry (shared by the pipeline and the UI).

The region→countries table, the admin-1 subregion shapefiles, and the pure masking
geometry (keep-grids for gridded fields, point-in-polygon for stations) live here — in
the ``verification`` layer — so **both** the diagnostics pipeline (which subsets the IDR
to a region before pooling) and the Streamlit UI (which masks a field at display time)
can use them. The UI's ``ui.utils.regions`` re-exports these and adds display-time
concerns (mask caching, prewarming).

``regionmask`` and the Natural Earth country set load lazily on first mask, so reading
``REGION_COUNTRIES`` (e.g. to populate a dropdown) does not pull the heavy geodata
dependency. Country names must match the Natural Earth v5 50m dataset exactly. ``None``
means the full loaded domain (no mask).

A deployment country (``VERIFICATION_COUNTRY``) additionally exposes its **subregions**,
at every administrative level it has onboarded, from :mod:`verification.admin`. Those are
masked with the country's own boundary polygons rather than Natural Earth, which only
knows whole countries.
"""
from __future__ import annotations

import threading as _threading
import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr

from verification import config

#: Region name -> Natural Earth member countries (None = the full domain, no mask).
#: Regions and departments come from :mod:`verification.admin`, not from this table.
REGION_COUNTRIES: dict[str, list[str] | None] = {
    "Africa": None,
    "Senegal": ["Senegal"],
    "Sierra Leone": ["Sierra Leone"],
}


def _country_match() -> str | None:
    """The REGION_COUNTRIES key matching the deployment country (case-insensitive), or None."""
    c = config.country()
    if not c:
        return None
    target = c.strip().lower()
    for key in REGION_COUNTRIES:
        if key.lower() == target:
            return key
    return None


def available_regions() -> list[str]:
    """Regions to offer in the UI.

    Under a country deployment: the country itself plus every region of its catalogue, at
    every administrative level, so a user can subset from the whole country down to the
    finest unit onboarded. Unscoped: the REGION_COUNTRIES list.
    """
    match = _country_match()
    if match:
        return [match] + _subregion_names(match)
    return list(REGION_COUNTRIES)


def _subregion_names(country: str) -> list[str]:
    """Display names of ``country``'s subregions, every level, sorted.

    Names are qualified where the bare one is ambiguous, so this list is usable both as a
    CLI choice set and as the UI picker.
    """
    return sorted(_get_subregions())


def default_region() -> str:
    """The region selected by default: the deployment country, or 'Africa' when unscoped."""
    return _country_match() or "Africa"


def regions_for_domain() -> list[str]:
    """The region names the verification CLI may target for the current deployment.

    Same set as :func:`available_regions` — the deployment country + every region of its
    catalogue (or the REGION_COUNTRIES list when unscoped). This is the ``--region`` choice
    list; the whole-country entry (``default_region()``) is the region-less default.
    """
    return available_regions()


# Natural Earth country set, built once on first mask (kept out of import
# time so the dict above stays cheap to read).
_COUNTRIES = None


# GEOS geometry queries are NOT thread-safe the way plain shapely calls look:
# prepared geometries (used internally by regionmask masking and contains_xy)
# build their spatial index lazily on the FIRST query, and two threads racing
# that build segfault the process (observed: GEOS PreparedPolygon::contains
# crashing in the dashboard's precache pool once admin-1 masks multiplied the
# concurrent queries). The lazy shapefile/Natural-Earth loads race the same
# way. Every GEOS-touching entry point below therefore serializes on this
# reentrant lock; callers cache the resulting masks, so the serialization
# costs one pass per (region, grid) — not per render.
_GEOS_LOCK = _threading.RLock()


def _get_countries():
    global _COUNTRIES
    with _GEOS_LOCK:
        if _COUNTRIES is None:
            import regionmask  # type: ignore
            _COUNTRIES = regionmask.defined_regions.natural_earth_v5_1_2.countries_50
        return _COUNTRIES


# {subregion name -> shapely geometry}, cached per deployment country (_SUBREGIONS_FOR).
_SUBREGIONS: dict | None = None
_SUBREGIONS_FOR: str | None = None


class _Subregions(Mapping):
    """``{display label: geometry}`` for one country, resolving geometry on demand.

    A Mapping rather than a plain dict because a coarse region's geometry is the union of
    its children's; only the geometry a caller actually asks for is built (and the
    catalogue caches it).
    """

    def __init__(self, catalogue, labels: dict):
        self._catalogue = catalogue
        self._labels = labels            # display label -> region id

    def __getitem__(self, label):
        return self._catalogue.geometry(self._labels[label])

    def __iter__(self):
        return iter(self._labels)

    def __len__(self):
        return len(self._labels)


def _get_subregions():
    """Subregions of the deployment country as ``{display label: geometry}``, or ``{}``.

    Sourced from :mod:`verification.admin`, so every administrative level the country
    onboarded is maskable.

    Empty when no boundaries are onboarded (the country is then its own only region).
    Cached, but rebuilt if the deployment country changes, so tests that flip
    ``VERIFICATION_COUNTRY`` see the right set.
    """
    global _SUBREGIONS, _SUBREGIONS_FOR
    match = _country_match()
    with _GEOS_LOCK:
        if _SUBREGIONS is None or _SUBREGIONS_FOR != match:
            _SUBREGIONS_FOR = match
            _SUBREGIONS = _build_subregions(match)
        return _SUBREGIONS


def _build_subregions(country):
    """Every named region of ``country``, at every level, as a lazy label -> geometry map."""
    if not country:
        return {}
    try:
        from verification import admin

        catalogue = admin.catalogue_for(country)
        if catalogue.pool_level == 0:
            return {}
        labels = {catalogue.display_label(r.id): r.id
                  for level in catalogue.levels()
                  for r in catalogue.at_level(level)
                  if r.id != admin.UNASSIGNED_ID}
        return _Subregions(catalogue, labels)
    except Exception as exc:  # noqa: BLE001 — a bad manifest must not brick masking
        print(f"[regions] region catalogue unavailable ({exc}); no subregions")
        return {}


def _frac_mask(subset, lon: np.ndarray, lat: np.ndarray) -> np.ndarray | None:
    """Try mask_3D_frac_approx with regularised coords; return None if grid is irregular."""
    def _regularise(arr: np.ndarray) -> np.ndarray | None:
        if arr.ndim != 1 or len(arr) < 2:
            return None
        diffs = np.diff(arr.astype(float))
        if np.max(np.abs(diffs - diffs[0])) > 1e-4 * abs(diffs[0]):
            return None  # genuinely unequal spacing
        return np.linspace(float(arr[0]), float(arr[-1]), len(arr))

    reg_lon = _regularise(lon)
    reg_lat = _regularise(lat)
    if reg_lon is None or reg_lat is None:
        return None
    try:
        with _GEOS_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            frac = subset.mask_3D_frac_approx(reg_lon, reg_lat)
            return (frac > 0).any(dim="region").values
    except Exception:
        return None


def _country_keep(region, lon, lat, lat_name, lon_name):
    """Boolean keep grid for a REGION_COUNTRIES region, or None if no polygons match.

    Uses the fast fractional-overlap mask on a regular grid, falling back to the Natural
    Earth center-point mask on a genuinely irregular one.
    """
    countries = _get_countries()
    ids = set()
    for name in REGION_COUNTRIES[region]:
        try:
            ids.add(countries.map_keys(name))
        except KeyError:
            print(f"[mask_region] country '{name}' not found in Natural Earth — skipped")
    if not ids:
        return None

    # Select only the relevant countries so masking is fast
    positions = [i for i, n in enumerate(countries.numbers) if n in ids]
    keep_raw = _frac_mask(countries[positions], lon, lat)
    if keep_raw is None:
        # Truly irregular grid — fall back to center-point mask
        with _GEOS_LOCK:
            cm = countries.mask(lon, lat)
        rename = {}
        if lat_name != "lat" and "lat" in cm.dims:
            rename["lat"] = lat_name
        if lon_name != "lon" and "lon" in cm.dims:
            rename["lon"] = lon_name
        if rename:
            cm = cm.rename(rename)
        keep_raw = cm.isin(list(ids)).values
    return keep_raw


def _subregion_keep(region, lon, lat):
    """Boolean keep grid for an admin-1 subregion, by point-in-polygon on centers."""
    import shapely

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    with _GEOS_LOCK:
        return shapely.contains_xy(_get_subregions()[region], lon2d, lat2d)


def stations_in_region(lons, lats, region: str) -> np.ndarray:
    """Boolean mask: which station points (lons, lats) fall inside ``region``.

    The point analogue of a grid mask. An admin-1 subregion uses containment against its
    shapefile polygon; a REGION_COUNTRIES key uses the union of its member-country
    polygons; a ``None`` member list ("Africa" / full domain) keeps every station.
    """
    import shapely

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    subregions = _get_subregions()
    if region in subregions:
        with _GEOS_LOCK:
            return shapely.contains_xy(subregions[region], lons, lats)

    members = REGION_COUNTRIES.get(region)
    if members is None:
        return np.ones(lons.shape, dtype=bool)

    countries = _get_countries()
    ids = set()
    for name in members:
        try:
            ids.add(countries.map_keys(name))
        except KeyError:
            print(f"[stations_in_region] country '{name}' not in Natural Earth — skipped")
    if not ids:
        return np.ones(lons.shape, dtype=bool)

    with _GEOS_LOCK:
        geom = shapely.union_all(
            [p for p, n in zip(countries.polygons, countries.numbers) if n in ids]
        )
        return shapely.contains_xy(geom, lons, lats)


def region_boundary(region: str):
    """(lon, lat) polyline of a subregion's boundary for map overlay, or None.

    Flat arrays with ``np.nan`` gaps between rings, so one line trace draws each ring
    separately (exterior + any holes, across a MultiPolygon's parts). None when ``region``
    is not a shapefile subregion — whole-country and continent borders are already drawn by
    the map's Natural Earth geo layer.
    """
    geom = _get_subregions().get(region)
    if geom is None:
        return None

    lons: list[float] = []
    lats: list[float] = []

    def _add(ring):
        x, y = ring.coords.xy
        lons.extend(x); lons.append(np.nan)
        lats.extend(y); lats.append(np.nan)

    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        _add(poly.exterior)
        for hole in poly.interiors:
            _add(hole)
    return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)

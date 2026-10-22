# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Forecast verification diagnostics — shared sample foundation.

Builds the pooled (forecast probability, observed binary event) pairs that the
reliability, ROC, and cost-loss diagnostics all consume. This module is the only
place that knows the on-disk shapes: the calibrated GraphCast ``p_exceed``
forecasts, the gridded observations (whose ``Lat``/``Lon`` are renamed to
the forecast's ``lat``/``lon``), and the DEM used for altitude stratification.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import rankdata

from verification import config
from verification import provenance as _provenance
from verification import settings as _settings
from verification.store import CountryStore as _CountryStore
from verification.store import VARIABLES as _VARIABLE_SPECS
from verification.store import truth_field as _truth_field

#: Precipitation's decision thresholds, kept under the historical name because it is
#: the default for the station path (gauges are precipitation-only). Thresholds are a
#: PER-VARIABLE fact now — see :func:`thresholds_for` — because temperature_max defines
#: its own hot-day set (37-40 degC) and a global list cannot describe both.
THRESHOLDS_MM_DAY = list(_VARIABLE_SPECS["precipitation"].exceedance)


#: Default elevation strata, in metres. A band is half-open ``[lo, hi)`` and ``all`` is
#: the union of the numeric ones (land only). ``config/verification.toml`` overrides it.
ALTITUDE_BANDS = {
    "all": None,
    "0-1000": (0.0, 1000.0),
    "1000-2000": (1000.0, 2000.0),
    "2000-3000": (2000.0, 3000.0),
    ">3000": (3000.0, float("inf")),
}


def altitude_bands(country: str | None = None) -> dict:
    """Elevation strata for a country (default: this deployment's), metres.

    Read at call time, like :func:`verification.config.country`, so a test or a second
    deployment in the same process sees the right terrain rather than whatever was
    current at import.

    ``config/verification.toml`` wins when it names this country (see
    :mod:`verification.settings`), else :data:`ALTITUDE_BANDS`.
    """
    if country is None:
        country = config.country()
    configured = _settings.altitude_bands(country)
    if configured is not None:
        return configured
    return ALTITUDE_BANDS

YEAR = 2022

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Every data location now comes from the per-country store (verification.store), which
# is the single owner of the on-disk scheme. Resolved at CALL time, never at import:
# the deployment seeds its environment in each entry point, and a root frozen at import
# would both ignore that and break tests/test_shim.py::test_ui_import_smoke.
def store():
    """The deployment's CountryStore — the single owner of the on-disk path scheme.

    Public so the CLI can reach the same store this module uses. Cheap to build and
    holds no open files, so callers construct one per use rather than caching it.
    """
    return _CountryStore.current()



DEFAULT_YEAR = 2022
DEFAULT_VARIABLE = "precipitation"

# Per-variable facts live in verification.store.VARIABLES — one registry shared with
# the calibration side, which used to keep its own copy of the same obs subdir/prefix
# table. This dict view keeps the historical mapping-style access (`_VARIABLE[v]["obs_var"]`)
# used throughout this module while the single source of truth stays in the store.
_VARIABLE = {
    name: {"obs_var": spec.obs_var, "to_rate": spec.to_rate,
           "raw_fc_var": spec.raw_fc_var,
           "exceedance": list(spec.exceedance),
           "exceedance_units": spec.exceedance_units}
    for name, spec in _VARIABLE_SPECS.items()
}

def thresholds_for(variable=DEFAULT_VARIABLE, country=None):
    """Decision thresholds for ``variable``, in ITS display units; [] if it has none.

    Resolved per COUNTRY at call time. The calibration emits ``p_exceed`` over exactly these
    values and the curves below select them back by label, so both sides read this one
    resolution — a mismatch yields an empty selection rather than an error.
    """
    from verification.store import exceedance_thresholds

    return list(exceedance_thresholds(variable, country))


def threshold_units(variable=DEFAULT_VARIABLE):
    """Units of :func:`thresholds_for` — ``mm/day`` for precip, ``degC`` for Tmax.

    Precipitation is the awkward one: the thresholds are mm/**day** while the
    calibrated field is mm/hr, because a decision threshold is about a daily total.
    """
    return _VARIABLE_SPECS[variable].exceedance_units


#: The onboarded variables, in registry order — the CLI's ``--variable`` choices.
#: Adding a row to ``verification.store.VARIABLES`` is the only edit a new variable
#: needs; nothing here restates the list.
VARIABLES = tuple(_VARIABLE)

# Model token (used in output paths and the UI) -> input folder under the year subtree.
# AIFS ships as TWO models from one archive family: aifs_single (deterministic) and
# aifs_ens (26-member mean). They are listed separately because they are separate
# forecasts — pooling their scores would average a deterministic run with an ensemble
# mean. A bare "aifs" token cannot say which is which, so there isn't one.
MODELS = {"graphcast": "graphcast", "ifs": "ifs",
          "aifs_single": "aifs_single", "aifs_ens": "aifs_ens"}

# The four builders below are thin, deliberate pass-throughs to the store: they exist
# because the whole module (and the CLI) already speaks in terms of "the forecast dir
# for this run", and collapsing them into their ~10 call sites would spread the store
# lookup rather than concentrate it. They add the model-token registry check that the
# store, which knows nothing about model tokens, does not perform.


def model_fc_root(model, year=DEFAULT_YEAR, variable=DEFAULT_VARIABLE, raw=False):
    """Gridded forecast dir for a (model, year, variable).

    raw=True → the uncalibrated ``forecast/`` tree, which carries no product segment
    because a forecast is truth-agnostic. Otherwise the calibrated IDR output tree,
    which does: an IDR fit belongs to the product it was trained against.
    """
    s, m = store(), MODELS[model]
    if raw:
        return s.forecast_dir(target="gridded", variable=variable, year=year, model=m)
    return s.calibrated_dir(target="gridded", variable=variable, year=year, model=m)


def station_fc_root(model, year=DEFAULT_YEAR):
    """Station forecast dir for a (model, year) — under the station IDR tree."""
    return store().calibrated_dir(target="station", variable=DEFAULT_VARIABLE,
                                   year=year, model=MODELS[model])


def station_raw_fc_root(model, year=DEFAULT_YEAR):
    """Raw (uncalibrated) forecast-at-station dir for a (model, year) — the calibration input."""
    return store().forecast_dir(target="station", variable=DEFAULT_VARIABLE,
                                year=year, model=MODELS[model])


def gridded_obs_path(year=DEFAULT_YEAR, variable="precipitation"):
    """Gridded obs file for a (year, variable), from the configured truth product.

    e.g. Senegal/CHIRTS maximum temperature →
    ``data/senegal/truth/chirts/gridded/temperature_max/{year}.nc``.
    """
    return store().truth(target="gridded", variable=variable, year=year)


def station_obs_path(year=DEFAULT_YEAR):
    """Gauge obs file for one year (the engine tier: single-variable, QC applied)."""
    return store().truth(target="station", variable=DEFAULT_VARIABLE, year=year)


def ui_model_token(model, calibrated):
    """UI Model-facet token: calibrated (IDR) forecasts get a ``-idr`` suffix; raw stay plain.

    The data trees stay keyed by the real model; this is only the label the diagnostics
    path (and thus the dashboard's Model selector) shows, so ``graphcast`` (raw) and
    ``graphcast-idr`` (calibrated) appear side by side. The calibrated data still lives,
    self-documenting, under ``idr/`` — it is not renamed or re-published.
    """
    return f"{model}-idr" if calibrated else model


def parse_model_token(token):
    """Inverse of :func:`ui_model_token`: split a UI model token into (base_model, calibrated).

    ``'graphcast-idr' -> ('graphcast', True)``; ``'graphcast' -> ('graphcast', False)``.
    The CLI uses this so the token a user types (``graphcast`` vs ``graphcast-idr``) selects
    the source — raw forecast vs IDR-calibrated — and is also the token written out.
    """
    if token.endswith("-idr"):
        return token[:-4], True
    return token, False


@dataclass
class Samples:
    """Pooled (p, o) pairs for one lead, keyed by (threshold_mm_day, altitude_band).

    ``pairs[(thr, band)]`` is a tuple ``(p, o)`` of equal-length 1-D arrays: ``p``
    the forecast exceedance probabilities and ``o`` the binary observed events
    (1 if the observation exceeded ``thr``, in :attr:`variable`'s threshold units —
    mm/day for precipitation, degC for daily-max temperature), pooled over time and
    space.
    ``thresholds`` and ``bands`` record the axes (and their order) this set covers.
    ``blocks`` (optional, same keys as ``pairs``) gives each pooled sample's day index,
    enabling the day-block bootstrap used for reliability confidence intervals.

    ``regions`` (same keys again) gives each sample's **region axis position**, and
    ``region_ids`` names that axis. Together with ``blocks`` they are what let a single
    pass be re-cut afterwards into any region and any season: every sample carries where
    and when it happened, so counts can be accumulated per (region, day) once and summed
    back into whatever grouping a reader asks for. ``regions`` is None for a country with
    no boundaries onboarded — there is nothing to cut by, and the national pool is the
    only answer.

    ``day_months`` maps a day index to its 1-based calendar month, so a season is a
    selection on the month axis rather than a filter applied before pooling.
    """
    lead_hours: int
    pairs: dict
    thresholds: list = None
    bands: list = None
    blocks: dict = None
    regions: dict = None
    region_ids: tuple = ()
    day_months: np.ndarray = None
    target: str = "gridded"
    #: The variable these pairs were pooled for. Carried because the curve writers
    #: stamp the threshold coord's UNITS, and those are per variable (mm/day for
    #: precipitation, degC for Tmax) — a hardcoded "mm/day" on a temperature curve
    #: mislabels the axis everywhere it is read, with nothing to catch it.
    variable: str = "precipitation"


def load_forecast(lead_hours, fc_root=None):
    """Open the calibrated forecast file for one lead (``{lead_hours}h.nc``).

    ``fc_root=None`` resolves the default run lazily; it used to be a module-level
    constant evaluated at import, which froze the data root before the deployment had
    a chance to seed its environment.
    """
    root = fc_root if fc_root is not None else model_fc_root("graphcast")
    return xr.open_dataset(store().lead_file(root, lead_hours))


def load_obs(obs_path=None):
    """Open gridded observations, renaming ``Lat``/``Lon`` to the forecast's ``lat``/``lon``."""
    obs = xr.open_dataset(obs_path or gridded_obs_path())
    rename = {old: old.lower() for old in ("Lat", "Lon")
              if old in obs.dims or old in obs.coords}
    return obs.rename(rename)


def interp_dem(like, dem_path=None):
    """Interpolate the DEM elevation onto the grid of ``like``, as a (lat, lon) DataArray."""
    path = dem_path if dem_path is not None else store().dem()
    dem = xr.open_dataset(path)["z_bedrock"]
    return dem.interp(lat=like["lat"], lon=like["lon"], method="linear")


def altitude_masks(elevation, country: str | None = None):
    """Boolean mask per altitude band, all land-only (elevation >= 0).

    Every band — including 'all' — excludes ocean/no-data cells, so 'all' equals the
    union of the numeric bands (a clean aggregate of the strata it summarises).
    ``country`` selects the strata (see :func:`altitude_bands`); None means this
    deployment's.
    """
    elev = np.asarray(elevation.values)
    land = np.isfinite(elev) & (elev >= 0)
    masks = {}
    for band, bounds in altitude_bands(country).items():
        if bounds is None:
            masks[band] = land
        else:
            lo, hi = bounds
            masks[band] = land & (elev >= lo) & (elev < hi)
    return masks


def pool_pairs(p_exceed, precip, masks, thresholds=THRESHOLDS_MM_DAY, region_labels=None):
    """Pool (p, o), day indices and region labels per (threshold, altitude band).

    ``p_exceed`` has dims (threshold, time, *space); ``precip`` has dims (time, *space)
    in mm/day, where *space* is one or more spatial dims — (lat, lon) for gridded
    targets, (station,) for station targets. All spatial dims are flattened to a
    single axis so the pooling is the same for both; the mask is flattened the same
    way. The event ``o`` is ``precip > threshold``; ``p`` is the forecast exceedance
    probability.

    Returns ``(pairs, blocks, regions)``: ``pairs[(thr, band)] = (p, o)``,
    ``blocks[(thr, band)]`` each sample's day index (the time coordinate position, the
    unit resampled by the block bootstrap), and ``regions[(thr, band)]`` each sample's
    region-axis position — or None throughout when ``region_labels`` is not supplied.

    ``region_labels`` is a spatial array (same shape as one time slice) of region axis
    positions from :func:`verification.admin.assign_grid`. It rides the *same* boolean
    selection as ``p`` and ``o``, which is what guarantees the three arrays stay aligned
    element-for-element — the alternative, masking the region separately, is how a
    partition silently stops matching the samples it is supposed to describe.

    **Supplying labels also scopes the pool to the country.** Cells the partition marks
    :data:`verification.admin.OUTSIDE` are dropped here, once, so every consumer sees the
    same samples — an obs grid is a bounding box and generally reaches beyond the country
    it is named for (see :mod:`verification.admin`).

    One threshold slice is materialised at a time to keep the memory footprint bounded.
    """
    precip_v = np.asarray(precip.values)
    n_time = precip_v.shape[0]
    # C-order flatten reproduces the old nested (lat, lon) pooling exactly, so the
    # gridded result is byte-identical; for stations the single dim is a no-op.
    precip_v = precip_v.reshape(n_time, -1)            # (time, space)
    finite_obs = np.isfinite(precip_v)
    day = np.broadcast_to(np.arange(n_time)[:, None], precip_v.shape)
    if region_labels is None:
        region_grid = None
    else:
        region_grid = np.broadcast_to(
            np.asarray(region_labels).reshape(-1)[np.newaxis, :], precip_v.shape)
    pairs = {}
    blocks = {}
    regions = {} if region_grid is not None else None
    for thr in thresholds:
        # Select by label, not position — trust the threshold coord the producer wrote.
        p_full = np.asarray(p_exceed.sel(threshold=thr).values).reshape(n_time, -1)
        o_full = precip_v > thr
        finite = finite_obs & np.isfinite(p_full)
        for band, mask in masks.items():
            sel = finite & np.asarray(mask).reshape(-1)[np.newaxis, :]
            if region_grid is not None:
                sel = sel & (region_grid >= 0)      # inside the country, see the docstring
            pairs[(thr, band)] = (p_full[sel], o_full[sel].astype(np.int8))
            blocks[(thr, band)] = day[sel]
            if region_grid is not None:
                regions[(thr, band)] = region_grid[sel]
    return pairs, blocks, regions


def _subset_gridded_to_region(fc, obs, region):
    """Crop ``fc``/``obs`` to an admin-1 region's bbox and NaN cells outside its polygon.

    Only admin-1 subregions of the deployment country (``VERIFICATION_COUNTRY``) subset;
    ``region`` None / the whole country / any non-subregion key returns the pair unchanged
    (whole-domain behaviour). The obs is masked (not the forecast) because pooling drops a
    cell whenever the obs is NaN, so out-of-polygon cells fall out on their own. Cropping to
    the bbox first keeps memory bounded — a single region is a small slice of the grid.
    """
    from verification.regions import _get_subregions, _subregion_keep

    if not region or region not in _get_subregions():
        return fc, obs
    lon = fc["lon"].values
    lat = fc["lat"].values
    keep = _subregion_keep(region, lon, lat)          # (nlat, nlon) bool
    lat_idx = np.where(keep.any(axis=1))[0]
    lon_idx = np.where(keep.any(axis=0))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        return fc.isel(lat=slice(0, 0), lon=slice(0, 0)), obs.isel(lat=slice(0, 0), lon=slice(0, 0))
    lat_sl = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    lon_sl = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    fc = fc.isel(lat=lat_sl, lon=lon_sl)
    obs = obs.isel(lat=lat_sl, lon=lon_sl)
    keep_da = xr.DataArray(keep[lat_sl, lon_sl], dims=("lat", "lon"),
                           coords={"lat": fc["lat"], "lon": fc["lon"]})
    return fc, obs.where(keep_da)


def _subset_to_months(fc, obs, months):
    """Restrict ``fc``/``obs`` to rows whose ``time`` falls in ``months`` (1-based).

    ``months`` None / empty returns the pair unchanged (Yearly = the whole year). A season
    crossing the year boundary (e.g. NDJF Nov–Feb → months 11,12,1,2) selects those calendar
    months of the single verification year. The two arrays share one ``time`` dim, so the same
    boolean mask applies to both.
    """
    if not months:
        return fc, obs
    keep = np.isin(fc["time"].dt.month.values, list(months))
    return fc.isel(time=keep), obs.isel(time=keep)


def _day_months(like):
    """1-based calendar month of every position on ``like``'s time axis.

    A season is a set of months, so this is the only thing a reader needs in order to
    slice one out of a pooled result — which is why it is carried on the Samples rather
    than applied as a filter before pooling.
    """
    if "time" not in like.coords and "time" not in like.dims:
        return None
    return np.asarray(like["time"].dt.month.values, dtype=np.int8)


def _region_labels_gridded(fc):
    """``(labels, region_ids)`` for a gridded forecast, or ``(None, ())`` if not possible.

    None means this country has no boundaries onboarded, so there is no partition to
    make and the national pool is the only answer — not an error. The label array is a
    read-only cached partition of the *unchanged* country mask; see
    :mod:`verification.admin`.
    """
    from verification import admin

    if "lat" not in fc.coords or "lon" not in fc.coords:
        return None, ()
    catalogue = admin.catalogue_for()
    labels = admin.assign_grid(catalogue, fc["lon"].values, fc["lat"].values)
    if (labels == admin.OUTSIDE).all():
        return None, ()
    return labels, tuple(r.id for r in catalogue.leaves())


def _region_labels_station(fc):
    """``(labels, region_ids)`` for a station forecast — the point analogue of above.

    Uses the gauge coordinates (``station_lat``/``station_lon``, falling back to
    ``lat``/``lon`` on the station dim), the same key the region subsetting has always
    used, so a gauge lands in the same region however it is reached.
    """
    from verification import admin

    lat_name = "station_lat" if "station_lat" in fc.coords else "lat"
    lon_name = "station_lon" if "station_lon" in fc.coords else "lon"
    if lat_name not in fc.coords or lon_name not in fc.coords:
        return None, ()
    catalogue = admin.catalogue_for()
    labels = admin.assign_points(catalogue, fc[lon_name].values, fc[lat_name].values)
    if (labels == admin.OUTSIDE).all():
        return None, ()
    return labels, tuple(r.id for r in catalogue.leaves())


def build_samples(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                  thresholds=None, bands=None,
                  fc_root=None, obs_path=None, dem_path=None, region=None, months=None,
                  variable=DEFAULT_VARIABLE):
    """Build pooled (p, o) :class:`Samples` for one lead + model + year, restricted to the given axes.

    Every sample is tagged with the region it falls in and the day it happened, so one
    pass supports **every** region and season afterwards. ``region`` and ``months`` are
    the older pre-filtering route: they crop before pooling and are kept for callers that
    want a single pre-cut pool (and for the tests that pin that behaviour), but the
    pipeline no longer needs them — it pools once and re-cuts at read time.
    """
    bands = bands if bands is not None else list(altitude_bands())
    thresholds = thresholds if thresholds is not None else thresholds_for(variable)
    fc = load_forecast(lead_hours, fc_root or model_fc_root(model, year, variable))
    obs = load_obs(obs_path or gridded_obs_path(year, variable))
    fc, obs = xr.align(fc, obs, join="inner")
    fc, obs = _subset_gridded_to_region(fc, obs, region)
    fc, obs = _subset_to_months(fc, obs, months)
    elevation = interp_dem(fc, dem_path)
    masks = {b: m for b, m in altitude_masks(elevation).items() if b in bands}
    labels, region_ids = _region_labels_gridded(fc)
    # The observation field, in the SAME units as the thresholds. Precipitation obs are
    # mm/day as stored (load_obs does not convert) and its thresholds are mm/day;
    # temperature is degC on both sides. truth_field tolerates a product naming the
    # field differently.
    obs_da = obs[_truth_field(obs, variable)]
    pairs, blocks, regions = pool_pairs(fc["p_exceed"], obs_da, masks,
                                        thresholds, region_labels=labels)
    return Samples(lead_hours=lead_hours, pairs=pairs, blocks=blocks, regions=regions,
                   region_ids=region_ids, day_months=_day_months(fc), target="gridded",
                   thresholds=list(thresholds), bands=list(bands), variable=variable)


def load_obs_station(year, obs_dir=None):
    """Open the gauge obs for a year (``total_precipitation_rate``, mm/day).

    ``obs_dir`` overrides the location (the injection seam the station tests use);
    None asks the store, which knows the engine-tier truth path for this country.
    """
    path = (Path(obs_dir) / f"stations_{year}.nc" if obs_dir is not None
            else station_obs_path(year))
    return xr.open_dataset(path)


def _subset_station_to_region(fc, obs_precip, region):
    """Keep only stations whose gauge coordinate falls inside an admin-1 region.

    The station analogue of :func:`_subset_gridded_to_region`: only admin-1 subregions
    filter; None / the whole country returns the pair unchanged. Uses the gauge coords
    (``station_lat``/``station_lon``, falling back to ``lat``/``lon`` on the station dim).
    """
    from verification.regions import _get_subregions, stations_in_region

    if not region or region not in _get_subregions():
        return fc, obs_precip
    lat_name = "station_lat" if "station_lat" in fc.coords else "lat"
    lon_name = "station_lon" if "station_lon" in fc.coords else "lon"
    keep = stations_in_region(fc[lon_name].values, fc[lat_name].values, region)
    return fc.isel(station=keep), obs_precip.isel(station=keep)


def build_samples_station(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                          thresholds=THRESHOLDS_MM_DAY, bands=None,
                          fc_root=None, obs_dir=None, region=None, months=None):
    """Build pooled (p, o) :class:`Samples` for one lead at stations.

    The station counterpart of :func:`build_samples`, on the (time, station) grid:
    the calibrated station forecast supplies ``p_exceed``; the EMA gauge supplies the
    event truth (gauge mm/day > threshold). The two are paired by the same
    ``(name, gauge-lat, gauge-lon)`` key + calendar date used in production
    (``align_stations``), and stations are stratified into altitude bands by their own
    ``elevation`` coord — no DEM interpolation, since each gauge carries its altitude.
    """
    from calibration.run_station import align_stations  # pure name+date pairing helper

    bands = bands if bands is not None else list(altitude_bands())
    fc = load_forecast(lead_hours, fc_root or station_fc_root(model, year))
    obs = load_obs_station(year, obs_dir)
    fc, obs_precip = align_stations(fc, obs["total_precipitation_rate"])
    fc, obs_precip = _subset_station_to_region(fc, obs_precip, region)
    fc, obs_precip = _subset_to_months(fc, obs_precip, months)
    if "elevation" not in fc.coords:
        # A gauge archive without elevation cannot be stratified. Pool everything into
        # the "all" band rather than raising: the alternative is a KeyError deep in a
        # run, and altitude bands are a refinement, not a precondition for curves.
        #
        # Every requested band still gets a key, because compute_roc indexes pairs for
        # the full declared axes: emitting only "all" here used to KeyError as soon as a
        # caller asked for a numeric band on a gauge archive without elevations. The
        # numeric bands are empty rather than absent, which the curve code already
        # handles as an empty pool.
        print("  note: no station `elevation` coord; pooling the 'all' band only")
        n_station = fc.sizes["station"]
        masks = {b: (np.ones(n_station, dtype=bool) if b == "all"
                     else np.zeros(n_station, dtype=bool)) for b in bands}
    else:
        masks = {b: m for b, m in altitude_masks(fc["elevation"]).items() if b in bands}
    labels, region_ids = _region_labels_station(fc)
    pairs, blocks, regions = pool_pairs(fc["p_exceed"], obs_precip, masks,
                                        thresholds, region_labels=labels)
    return Samples(lead_hours=lead_hours, pairs=pairs, blocks=blocks, regions=regions,
                   region_ids=region_ids, day_months=_day_months(fc), target="station",
                   thresholds=list(thresholds), bands=list(bands))


def _axes(samples):
    """Resolve the (thresholds, bands) axes of a Samples, in canonical order."""
    keys = set(samples.pairs)
    thresholds = samples.thresholds or [t for t in thresholds_for(samples.variable)
                                        if any(k[0] == t for k in keys)]
    bands = samples.bands or [b for b in altitude_bands() if any(k[1] == b for k in keys)]
    return thresholds, bands


ROC_PROBS = np.linspace(0.01, 1.0, 500)


def roc_counts(p, o, probs=ROC_PROBS):
    """Contingency counts (hits, false_alarms, misses, correct_negatives) at every cut-off.

    Declares an event where ``p >= p*``. Evaluates all cut-offs in one pass by sorting
    the event and non-event probabilities and counting with ``searchsorted`` — the cheap
    equivalent of the histogram/cumsum sweep, avoiding a 500-fold boolean scan. Returns a
    dict of int64 arrays, each aligned to ``probs``.
    """
    p = np.asarray(p)
    o = np.asarray(o).astype(bool)
    pe = np.sort(p[o])
    pn = np.sort(p[~o])
    ne, nn = pe.size, pn.size
    hits = ne - np.searchsorted(pe, probs, side="left")
    false_alarms = nn - np.searchsorted(pn, probs, side="left")
    return {
        "hits": hits.astype(np.int64),
        "false_alarms": false_alarms.astype(np.int64),
        "misses": (ne - hits).astype(np.int64),
        "correct_negatives": (nn - false_alarms).astype(np.int64),
    }


def compute_roc(samples, probs=ROC_PROBS, thresholds=None, bands=None):
    """Build the ROC contingency-table Dataset over (threshold, altitude, prob) for one lead."""
    axt, axb = _axes(samples)
    thresholds = thresholds if thresholds is not None else axt
    bands = bands if bands is not None else axb
    shape = (len(thresholds), len(bands), len(probs))
    fields = ("hits", "false_alarms", "misses", "correct_negatives")
    arrs = {k: np.zeros(shape, dtype=np.int64) for k in fields}
    for i, thr in enumerate(thresholds):
        for j, band in enumerate(bands):
            p, o = samples.pairs[(thr, band)]
            counts = roc_counts(p, o, probs)
            for k in fields:
                arrs[k][i, j, :] = counts[k]
    ds = xr.Dataset(
        {k: (("threshold", "altitude", "prob"), v) for k, v in arrs.items()},
        coords={"threshold": thresholds, "altitude": bands, "prob": probs},
    )
    ds["threshold"].attrs["units"] = threshold_units(samples.variable)
    ds["prob"].attrs["long_name"] = "probability cut-off p*"
    ds.attrs["lead_hours"] = samples.lead_hours
    return ds


ROC_COUNT_FIELDS = ("hits", "false_alarms", "misses", "correct_negatives")


def _compress(ds, *, ints=("bin_total", "bin_events", *ROC_COUNT_FIELDS)):
    """Stamp on-disk compression for a by-region store, in place.

    These stores are far larger than the whole-country files they replace — a region axis
    and a month or day axis on top of (threshold, altitude, prob) — but they are also
    mostly zeros, because a region holds a small share of the domain and many hold none of
    it at all (most regions have no gauge). zlib turns that
    sparsity into the storage saving it should be.

    Counts are narrowed to int32 on disk: the largest per-(region, month) cell is the
    national pool of one month, far inside int32, and the sums that rebuild a curve are
    widened back to int64 by :mod:`verification.curves` before any arithmetic.
    """
    for name, array in ds.data_vars.items():
        array.encoding.update(zlib=True, complevel=4, shuffle=True)
        if name in ints:
            array.encoding["dtype"] = "int32"
        # One chunk per region, whole otherwise. The dashboard's dominant read is "one
        # region, every threshold and band", and without region-aligned chunks that
        # selection still decompresses its neighbours — the arrays expand to well over a
        # gigabyte for a country with a few hundred regions, so it is the difference
        # between a responsive selector and a two-second wait per click.
        if "region" in array.dims:
            array.encoding["chunksizes"] = tuple(
                1 if d == "region" else ds.sizes[d] for d in array.dims)
    return ds

#: Month axis of the by-region ROC store. Fixed and complete, so a month with no data is
#: an explicit zero rather than an absent label — that is what makes summing a season a
#: plain selection instead of a join.
MONTHS = tuple(range(1, 13))


def _grouped_roc_counts(p, o, group, n_groups, probs):
    """Contingency counts at every cut-off, for every group, in one pass.

    ``group`` is a per-sample integer in ``[0, n_groups)``. Returns
    ``(n_groups, len(probs))`` int64 arrays.

    Sorting once by (group, p) is what makes this affordable: within a group the
    probabilities are then already ordered, so each group's event and non-event arrays
    are sorted slices and ``searchsorted`` can sweep all 500 cut-offs without a second
    sort. Pooling the whole country and pooling every department therefore cost
    essentially the same — the work scales with the number of samples, not with the
    number of regions.
    """
    counts = {k: np.zeros((n_groups, probs.size), dtype=np.int64)
              for k in ROC_COUNT_FIELDS}
    if p.size == 0:
        return counts

    order = np.lexsort((p, group))
    gs = group[order]
    ps = p[order]
    os_ = o[order].astype(bool)
    bounds = np.searchsorted(gs, np.arange(n_groups + 1))
    for g in range(n_groups):
        lo, hi = int(bounds[g]), int(bounds[g + 1])
        if lo == hi:
            continue
        pg, og = ps[lo:hi], os_[lo:hi]
        pe = pg[og]                 # already sorted: the slice was sorted by p
        pn = pg[~og]
        n_event, n_non = pe.size, pn.size
        hits = n_event - np.searchsorted(pe, probs, side="left")
        false_alarms = n_non - np.searchsorted(pn, probs, side="left")
        counts["hits"][g] = hits
        counts["false_alarms"][g] = false_alarms
        counts["misses"][g] = n_event - hits
        counts["correct_negatives"][g] = n_non - false_alarms
    return counts


def compute_roc_by_region(samples, probs=ROC_PROBS, thresholds=None, bands=None):
    """ROC counts over (region, month, threshold, altitude, prob) — the additive store.

    This is the file every curve is later derived from. Because contingency counts are
    additive over disjoint pools and the ``prob`` axis is the same in every file, any
    region (a province is the sum of its municipalities, the country the sum of them all)
    and any season (a set of months) is recovered by **integer addition** at read time —
    exactly, not approximately. See :mod:`verification.curves`.

    Returns None when the country has no region partition; the national ROC is then the
    only meaningful answer and :func:`compute_roc` already provides it.
    """
    if samples.regions is None or samples.day_months is None:
        return None
    axt, axb = _axes(samples)
    thresholds = thresholds if thresholds is not None else axt
    bands = bands if bands is not None else axb

    region_ids = list(samples.region_ids)
    n_region, n_month = len(region_ids), len(MONTHS)
    n_groups = n_region * n_month
    shape = (n_region, n_month, len(thresholds), len(bands), probs.size)
    arrs = {k: np.zeros(shape, dtype=np.int64) for k in ROC_COUNT_FIELDS}

    month_of_day = np.asarray(samples.day_months, dtype=np.int64) - 1
    for i, thr in enumerate(thresholds):
        for j, band in enumerate(bands):
            p, o = samples.pairs[(thr, band)]
            region = np.asarray(samples.regions[(thr, band)], dtype=np.int64)
            month = month_of_day[np.asarray(samples.blocks[(thr, band)], dtype=np.int64)]
            counts = _grouped_roc_counts(p, o, region * n_month + month, n_groups, probs)
            for k in ROC_COUNT_FIELDS:
                arrs[k][:, :, i, j, :] = counts[k].reshape(n_region, n_month, probs.size)

    dims = ("region", "month", "threshold", "altitude", "prob")
    ds = xr.Dataset(
        {k: (dims, v) for k, v in arrs.items()},
        coords={"region": region_ids, "month": list(MONTHS),
                "threshold": thresholds, "altitude": bands, "prob": probs},
    )
    ds["threshold"].attrs["units"] = threshold_units(samples.variable)
    ds["prob"].attrs["long_name"] = "probability cut-off p*"
    ds["region"].attrs["long_name"] = "administrative region id"
    ds.attrs["lead_hours"] = samples.lead_hours
    ds.attrs["rollup"] = "counts are additive over region and month"
    return _compress(ds)


def output_path(metric, lead_hours, outdir=None, year=DEFAULT_YEAR, subdir=None, model="graphcast",
                target="gridded", native_var=DEFAULT_VARIABLE, region=None, season=None,
                product=None):
    """The on-disk path for a diagnostic file, mirroring the engine's facet order.

    ``subdir`` inserts an extra segment before the file (e.g. ``annual_monthly_averages``
    for scalar metrics the dashboard discovers); curve metrics leave it None. ``model`` is
    the forecast token (the UI keys its Model facet off this segment).

    ``target`` is the SHAPE of the observations (``gridded`` or ``station``) and
    ``product`` is WHICH dataset they came from (``chirts``, ``era5``), defaulting to the
    deployment's configured product.

    ``native_var`` is the variable segment the UI's ``kind_of`` reads (``precipitation``
    → precip, ``temperature_max`` → Tmax), so variables never collide. ``region`` inserts
    an admin-1 segment after the year for per-region threshold curves; None writes the
    region-less whole-domain path. ``season`` inserts a season-slug segment *after* the
    region one; None / ``"yearly"`` writes the season-less path. The slug is a controlled
    vocabulary (see :mod:`verification.seasons`) so discovery can tell it apart from a
    region segment.
    """
    # The facet segment order lives in verification.store — one definition, shared with
    # the UI's discovery — so this function never spells the order out itself.
    s = store()
    rel = s.diagnostic_relative(
        model=model, variable=native_var, target=target, product=product,
        metric=metric, year=year, lead=lead_hours,
        subdir=subdir, region=region, season=season)
    # Where the domain segment sits relative to `outdir` is a LAYOUT decision, not this
    # function's. Placing it here would make the writer disagree with discovery.
    root = s.diagnostics_root() if outdir is None else s.diagnostics_root_under(outdir)
    return root / rel


def write_dataset(ds, metric, lead_hours, outdir=None, year=DEFAULT_YEAR, subdir=None, model="graphcast",
                  target="gridded", native_var=DEFAULT_VARIABLE, region=None, season=None,
                  product=None):
    """Write a diagnostic Dataset to its :func:`output_path`, creating parent dirs."""
    path = output_path(metric, lead_hours, outdir, year, subdir, model=model, target=target,
                       native_var=native_var, region=region, season=season, product=product)
    # write_dataset is the single funnel for every diagnostic file, so one stamp here
    # covers all three metric families. Global attrs only: discovery reads dims and
    # path segments, never attributes, so this cannot change what the dashboard offers.
    _provenance.stamp(
        ds, stage="verification", metric=metric, model=model, variable=native_var,
        target=target, year=year, lead_hours=lead_hours,
        region=region, season=season, subdir=subdir, domain=config.domain(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def roc_rates(ds, threshold, band):
    """FPR and TPR arrays (ascending FPR, with (0,0)/(1,1) endpoints) for one ROC curve."""
    sub = ds.sel(threshold=threshold, altitude=band)
    h, m = sub["hits"].values, sub["misses"].values
    fa, cn = sub["false_alarms"].values, sub["correct_negatives"].values
    tpr = h / np.maximum(h + m, 1)
    fpr = fa / np.maximum(fa + cn, 1)
    order = np.argsort(fpr)
    fpr = np.concatenate([[0.0], fpr[order], [1.0]])
    tpr = np.concatenate([[0.0], tpr[order], [1.0]])
    return fpr, tpr



def _thr_units(ds):
    """Threshold units read off a curve Dataset, for labelling.

    The Dataset carries them (stamped by whichever writer produced it), so a figure
    never has to know which variable it is drawing. A hardcoded "mm/day" here put that
    label on every temperature plot.
    """
    try:
        return ds["threshold"].attrs.get("units", "")
    except Exception:  # noqa: BLE001 — a label must never break a plot
        return ""


def plot_roc(ds, out_png, band="all"):
    """Save a ROC plot (one curve per threshold, with AUC in the legend) for one altitude band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    fig, ax = plt.subplots(figsize=(6, 6))
    for thr in ds["threshold"].values:
        fpr, tpr = roc_rates(ds, thr, band)
        ax.plot(fpr, tpr, marker="", lw=2,
                label=f"{int(thr)} {_thr_units(ds)}".rstrip()
                      + f"  (AUC={trapezoid(tpr, fpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="no skill")
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Hit rate")
    ax.set_title(f"ROC — GraphCast vs observations — {ds.attrs.get('lead_hours')}h — altitude: {band}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def _band_slug(band):
    """Filename-safe form of an altitude-band label."""
    return str(band).replace(">", "gt").replace(" ", "")


def plot_by_altitude(ds, out_dir, plot_fn, prefix):
    """Save one PNG per altitude band using ``plot_fn``; returns the written paths."""
    lead = ds.attrs.get("lead_hours")
    paths = []
    for band in (str(b) for b in ds["altitude"].values):
        png = Path(out_dir) / f"{prefix}_{lead}h_alt_{_band_slug(band)}.png"
        plot_fn(ds, png, band=band)
        paths.append(png)
    return paths


COST_LOSS_RATIOS = np.linspace(0.01, 0.99, 100)


def _value_envelope(hits, false_alarms, misses, correct_negatives, alphas):
    """Relative economic value V(alpha), maximised over cut-offs, from one curve's counts.

    For a user with cost-loss ratio ``alpha = C/L`` and event base rate ``s``, the value of
    acting on the forecast is compared against climatology and a perfect forecast:
    ``V = (E_clim - E_forecast) / (E_clim - E_perfect)``. The envelope takes the best V over
    all probability cut-offs. Returns ``(V_envelope, base_rate)``; V is zeros if no events.
    """
    n = hits[0] + misses[0] + false_alarms[0] + correct_negatives[0]
    if n == 0:                       # empty pool (e.g. a band absent from a small subregion)
        return np.zeros_like(alphas), 0.0
    s = (hits[0] + misses[0]) / n
    if s <= 0 or s >= 1:
        return np.zeros_like(alphas), float(s)
    hit_fa = (hits + false_alarms) / n
    miss = misses / n
    e_clim = np.minimum(alphas, s)
    e_perfect = alphas * s
    denom = e_clim - e_perfect
    e_forecast = alphas[:, None] * hit_fa[None, :] + miss[None, :]
    value = (e_clim[:, None] - e_forecast) / denom[:, None]
    return value.max(axis=1), float(s)


def compute_cost_loss(roc, alphas=COST_LOSS_RATIOS):
    """Build the cost-loss value-envelope Dataset from a ROC contingency Dataset (issue #4)."""
    thresholds = roc["threshold"].values
    bands = [str(b) for b in roc["altitude"].values]
    value = np.zeros((len(thresholds), len(bands), len(alphas)))
    base_rate = np.zeros((len(thresholds), len(bands)))
    for i, thr in enumerate(thresholds):
        for j, band in enumerate(roc["altitude"].values):
            sub = roc.sel(threshold=thr, altitude=band)
            value[i, j, :], base_rate[i, j] = _value_envelope(
                sub["hits"].values, sub["false_alarms"].values,
                sub["misses"].values, sub["correct_negatives"].values, alphas)
    ds = xr.Dataset(
        {"value": (("threshold", "altitude", "cost_loss_ratio"), value),
         "base_rate": (("threshold", "altitude"), base_rate),
         "max_value": (("threshold", "altitude"), value.max(axis=2))},
        coords={"threshold": thresholds, "altitude": bands, "cost_loss_ratio": alphas},
    )
    # Inherit the input ROC's units rather than assuming: cost-loss is derived
    # from those counts, so its axis is whatever theirs was.
    ds["threshold"].attrs["units"] = roc["threshold"].attrs.get("units", "")
    ds["cost_loss_ratio"].attrs["long_name"] = "cost-loss ratio alpha = C/L"
    ds.attrs["lead_hours"] = roc.attrs.get("lead_hours")
    return ds


def plot_cost_loss(ds, out_png, band="all"):
    """Save a cost-loss value curve (one curve per threshold) for one altitude band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = ds["cost_loss_ratio"].values
    fig, ax = plt.subplots(figsize=(6, 5))
    for thr in ds["threshold"].values:
        v = ds["value"].sel(threshold=thr, altitude=band).values
        ax.plot(alphas, v, lw=2,
                label=f"{int(thr)} {_thr_units(ds)}".rstrip())
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Cost-loss ratio  α = C/L")
    ax.set_ylabel("Relative economic value V")
    ax.set_title(f"Cost-loss value — {ds.attrs.get('lead_hours')}h — altitude: {band}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


RELIABILITY_N_BINS = 10
RELIABILITY_N_BOOTSTRAPS = 50

#: Below this, a calibrated forecast is saying "no chance". IDR returns a probability of
#: essentially zero whenever no training sample exceeded the threshold, which at 50-60
#: mm/day is ~90% of cell-days — a point mass, not a tail.
RELIABILITY_ZERO_BAND = 1e-4


def derive_reliability_bands(p, n_bins=RELIABILITY_N_BINS, floor=RELIABILITY_ZERO_BAND):
    """Probability bands for one pool: a band for the point mass, then quantiles of the rest.

    Plain quantile binning cannot work on calibrated exceedance probabilities. IDR returns
    a probability of essentially zero whenever no training sample exceeded the threshold —
    ~90% of cell-days at the high thresholds — so the inner quantiles all land on that one
    value, every such sample falls into a single bin, and the rest come out empty. Observed
    on real output: one populated bin out of ten at 50 and 60 mm/day.

    So the point mass gets one band of its own, ``[0, floor)``, and the remaining bands are
    quantiles of the probabilities above it. That keeps the intent of quantile binning —
    every point carries comparable weight — while being robust to a distribution that is
    part discrete and part continuous.

    Bands are derived from the pool passed in, which is the whole pool for a
    (threshold, altitude) pair: all regions, all months. Every region and month therefore
    shares one set of edges, which is what lets their counts be summed. ``threshold`` and
    ``altitude`` are never summed over, so each may have its own bands.
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return tuple(np.linspace(0.0, 1.0, n_bins + 1))

    informative = p[p >= floor]
    if informative.size < n_bins:
        # Almost nothing above the floor. Spread the upper bands geometrically rather
        # than emitting edges the data cannot justify.
        return (0.0, floor, *np.geomspace(floor * 10, 1.0, n_bins - 1))

    inner = np.quantile(informative, np.linspace(0, 1, n_bins))
    edges = np.round(np.concatenate(([0.0], inner)), 5)
    edges[1] = floor
    edges[-1] = 1.0
    return _spread_ties(edges, n_bins + 1)


def _spread_ties(edges, n_edges):
    """Drop duplicate edges and re-insert the missing ones in the widest log gap.

    Quantiles tie at the rare thresholds even above the floor. Nudging each repeat up by a
    fixed epsilon leaves bands ~1e-5 wide that no sample lands in, costing a point for
    nothing; splitting the widest gap in log space puts the recovered edges where the
    samples actually are. Probability bands span orders of magnitude, so log is the scale.
    """
    kept = sorted({float(v) for v in edges})
    while len(kept) < n_edges:
        positive = [v for v in kept if v > 0]
        gaps = [(np.log(b) - np.log(a), i) for i, (a, b)
                in enumerate(zip(positive[:-1], positive[1:]))]
        if not gaps:
            break
        _, i = max(gaps)
        mid = float(np.sqrt(positive[i] * positive[i + 1]))
        if not positive[i] < mid < positive[i + 1]:
            break                      # numerically indistinguishable — stop honestly
        kept = sorted(set(kept) | {mid})
    kept[0], kept[-1] = 0.0, 1.0
    return tuple(kept)


def bootstrap_seed(seed, threshold, band) -> int:
    """A reproducible RNG seed for one (threshold, band) bootstrap.

    Derived from the values rather than from position in a loop, so the confidence
    interval for a given pool is the same whichever run produced it — a whole-country
    run, a per-region run, or a read-time derivation from stored counts. Threading one
    generator through the loop made the draws depend on how many axes were requested,
    which is why changing ``--threshold`` or ``--altitude`` used to change the intervals.
    """
    payload = f"{seed}|{float(threshold):.6g}|{band}".encode()
    import hashlib

    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def assign_bins(p, edges):
    """Band index in ``[0, len(edges) - 2]`` for each probability in ``p``.

    Half-open bands ``[lo, hi)``, with the top band closed so ``p == 1`` lands in it
    rather than out of range.
    """
    inner = np.asarray(edges[1:-1], dtype=float)
    return np.clip(np.searchsorted(inner, np.asarray(p, dtype=float), side="right"),
                   0, len(edges) - 2).astype(np.int64)


def compute_reliability_counts(samples, n_bins=RELIABILITY_N_BINS,
                               thresholds=None, bands=None):
    """Per-day reliability counts over (region, day, bin, threshold, altitude).

    Stores three summable quantities per cell: ``bin_total`` (samples), ``bin_events``
    (observed exceedances) and ``prob_sum`` (sum of forecast probabilities). Every number
    on a reliability diagram is recovered from those by addition — observed frequency is
    ``bin_events / bin_total``, mean forecast probability is ``prob_sum / bin_total`` —
    so any region and any season is exact rather than approximate.

    **Day resolution is what makes the confidence interval exact too.** The interval comes
    from a block bootstrap that resamples whole days; keeping counts per day means the
    same resample can be replayed at read time over whatever days a season selects, giving
    precisely the interval a directly-pooled run would have produced. Month resolution
    would have forced a month-stratified bootstrap instead — valid, but not the same
    number.

    Returns None when the country has no region partition.
    """
    if samples.regions is None or samples.day_months is None:
        return None
    axt, axb = _axes(samples)
    thresholds = thresholds if thresholds is not None else axt
    bands = bands if bands is not None else axb

    region_ids = list(samples.region_ids)
    n_region = len(region_ids)
    n_day = int(len(samples.day_months))
    shape = (n_region, n_day, n_bins, len(thresholds), len(bands))
    total = np.zeros(shape, dtype=np.int64)
    events = np.zeros(shape, dtype=np.int64)
    prob_sum = np.zeros(shape, dtype=np.float64)
    cell_shape = (n_region, n_day, n_bins)
    n_cells = n_region * n_day * n_bins

    lower = np.zeros((len(thresholds), len(bands), n_bins))
    upper = np.zeros((len(thresholds), len(bands), n_bins))
    for i, thr in enumerate(thresholds):
        for j, band in enumerate(bands):
            p, o = samples.pairs[(thr, band)]
            # Derived from THIS (threshold, altitude) pool — all regions, all months — so
            # every region and month shares one set of edges and their counts can be
            # summed. Neither axis is ever summed over, so each may differ.
            edges = derive_reliability_bands(p, n_bins)
            lower[i, j] = np.asarray(edges[:-1])
            upper[i, j] = np.asarray(edges[1:])
            if p.size == 0:
                continue
            region = np.asarray(samples.regions[(thr, band)], dtype=np.int64)
            day = np.asarray(samples.blocks[(thr, band)], dtype=np.int64)
            cell = (region * n_day + day) * n_bins + assign_bins(p, edges)
            total[:, :, :, i, j] = np.bincount(
                cell, minlength=n_cells).reshape(cell_shape)
            events[:, :, :, i, j] = np.bincount(
                cell, weights=o.astype(np.float64),
                minlength=n_cells).reshape(cell_shape).astype(np.int64)
            prob_sum[:, :, :, i, j] = np.bincount(
                cell, weights=np.asarray(p, dtype=np.float64),
                minlength=n_cells).reshape(cell_shape)

    dims = ("region", "day", "bin", "threshold", "altitude")
    ds = xr.Dataset(
        {"bin_total": (dims, total),
         "bin_events": (dims, events),
         "prob_sum": (dims, prob_sum),
         "bin_lower": (("threshold", "altitude", "bin"), lower),
         "bin_upper": (("threshold", "altitude", "bin"), upper)},
        coords={"region": region_ids, "day": np.arange(n_day),
                "bin": np.arange(n_bins), "threshold": thresholds, "altitude": bands,
                "month": ("day", np.asarray(samples.day_months, dtype=np.int8))},
    )
    ds["threshold"].attrs["units"] = threshold_units(samples.variable)
    ds["region"].attrs["long_name"] = "administrative region id"
    ds["day"].attrs["long_name"] = "position on the verification year's time axis"
    ds["month"].attrs["long_name"] = "1-based calendar month of each day"
    ds.attrs["lead_hours"] = samples.lead_hours
    ds.attrs["binning"] = "zero-band + quantiles of the informative tail"
    ds.attrs["binning_target"] = samples.target
    ds.attrs["rollup"] = "counts are additive over region and day"
    return _compress(ds)


def bootstrap_ci_from_day_counts(tot, pos, n_bootstraps, rng):
    """95% CI of each band's observed frequency, resampling whole days with replacement.

    ``tot`` and ``pos`` are ``(day, bin)`` counts — samples and observed events per day
    per probability band. Each replicate redraws the set of days, so the interval reflects
    between-day (and within-day spatial) dependence rather than assuming i.i.d. cells.

    This is the **only** implementation of the interval, called both by the pooling path
    (which aggregates raw samples into these counts first) and by the read-time derivation
    in :mod:`verification.curves` (which reads them straight off disk). That is what makes
    a confidence interval computed from stored counts identical to one computed from the
    raw samples, rather than merely similar — the arithmetic is the same arithmetic.
    """
    tot = np.asarray(tot, dtype=float)
    pos = np.asarray(pos, dtype=float)
    n_blocks, n_bins = tot.shape
    boot = np.full((n_bootstraps, n_bins), np.nan)
    for r in range(n_bootstraps):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        bt = tot[chosen].sum(0)
        bp = pos[chosen].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[r] = np.where(bt > 0, bp / bt, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    nonempty = tot.sum(0) > 0
    if nonempty.any():
        lo[nonempty], hi[nonempty] = np.nanpercentile(boot[:, nonempty], [2.5, 97.5],
                                                      axis=0)
    return lo, hi


def _block_bootstrap_ci(idx, o, block, n_bins, n_bootstraps, rng):
    """Aggregate samples into per-day band counts, then take the block-bootstrap CI.

    The aggregation is the only thing here; the resampling lives in
    :func:`bootstrap_ci_from_day_counts` so the stored-counts path and the raw-sample path
    cannot diverge.
    """
    _, compact = np.unique(block, return_inverse=True)
    n_blocks = int(compact.max()) + 1
    tot = np.zeros((n_blocks, n_bins))
    pos = np.zeros((n_blocks, n_bins))
    np.add.at(tot, (compact, idx), 1.0)
    np.add.at(pos, (compact, idx), o.astype(float))
    return bootstrap_ci_from_day_counts(tot, pos, n_bootstraps, rng)


def _reliability_curve(p, o, n_bins, n_bootstraps, rng, block=None, edges=None):
    """Per probability band: mean forecast prob, observed frequency, count, and 95% CI.

    Bands are the **fixed** edges from :func:`reliability_bin_edges` — shared by every
    region, month, lead and model. They used to be quantile edges taken from ``p`` itself,
    which made two pools' bands incomparable and so unsummable; the derived fixed edges
    keep the roughly-equal-occupancy that motivated quantile binning while letting a
    parent region be recovered from its children. Under the point mass at p≈0 for rare
    thresholds a low band can still come out empty and stay NaN.

    When ``block`` (a day index per sample) is given, the CI comes from a day-block
    bootstrap (:func:`_block_bootstrap_ci`), which respects spatial+temporal
    autocorrelation; otherwise it falls back to an i.i.d. binomial draw, which is only
    valid for genuinely independent samples.
    """
    # An empty pool (e.g. an altitude band with no cells inside a small subregion) still
    # has bands — they are fixed — but no samples to put in them, so every one is NaN.
    if np.asarray(p).size == 0:
        nan = np.full(n_bins, np.nan)
        return nan, nan.copy(), np.zeros(n_bins, dtype=np.int64), nan.copy(), nan.copy()
    idx = assign_bins(p, edges)
    fp = np.full(n_bins, np.nan)
    of = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    if block is not None:
        lo, hi = _block_bootstrap_ci(idx, o, block, n_bins, n_bootstraps, rng)
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        cnt[b] = n
        if n == 0:
            continue
        fp[b] = p[sel].mean()
        k = int(o[sel].sum())
        of[b] = k / n
        if block is None:
            boot = rng.binomial(n, k / n, size=n_bootstraps) / n
            lo[b], hi[b] = np.percentile(boot, [2.5, 97.5])
    return fp, of, cnt, lo, hi


def compute_reliability(samples, n_bins=RELIABILITY_N_BINS,
                        n_bootstraps=RELIABILITY_N_BOOTSTRAPS,
                        thresholds=None, bands=None, seed=0, bin_edges=None):
    """Build the reliability-diagram Dataset over (threshold, altitude, bin) for one lead (issue #2)."""
    axt, axb = _axes(samples)
    thresholds = thresholds if thresholds is not None else axt
    bands = bands if bands is not None else axb
    shape = (len(thresholds), len(bands), n_bins)
    fp = np.full(shape, np.nan)
    of = np.full(shape, np.nan)
    lo = np.full(shape, np.nan)
    hi = np.full(shape, np.nan)
    cnt = np.zeros(shape, dtype=np.int64)
    lower = np.zeros((len(thresholds), len(bands), n_bins))
    upper = np.zeros((len(thresholds), len(bands), n_bins))
    for i, thr in enumerate(thresholds):
        for j, band in enumerate(bands):
            # Seeded per (threshold, band) rather than threading one generator through
            # the double loop: the draws then do not depend on how many axes a run
            # happened to request, so a whole-country run and a per-region run produce
            # the same interval for the same pool.
            rng = np.random.default_rng(bootstrap_seed(seed, thr, band))
            p, o = samples.pairs[(thr, band)]
            # Same pool, same rule, so these edges match the by-region store's exactly.
            # ``bin_edges`` overrides that with a caller-supplied {(thr, band): edges}
            # map, which is how a *subset* of a pool can be scored on the bands of the
            # whole: bands are pool-derived, so a region pooled directly would otherwise
            # get its own, and the two would not be comparable bin for bin.
            edges = (bin_edges or {}).get((thr, band)) or derive_reliability_bands(p, n_bins)
            lower[i, j], upper[i, j] = np.asarray(edges[:-1]), np.asarray(edges[1:])
            block = samples.blocks.get((thr, band)) if samples.blocks else None
            fp[i, j], of[i, j], cnt[i, j], lo[i, j], hi[i, j] = _reliability_curve(
                p, o, n_bins, n_bootstraps, rng, block=block, edges=edges)
    dims = ("threshold", "altitude", "bin")
    ds = xr.Dataset(
        {"forecast_probability": (dims, fp),
         "observed_frequency": (dims, of),
         "bin_count": (dims, cnt),
         "obs_freq_lower": (dims, lo),
         "obs_freq_upper": (dims, hi),
         "bin_lower": (("threshold", "altitude", "bin"), lower),
         "bin_upper": (("threshold", "altitude", "bin"), upper)},
        coords={"threshold": thresholds, "altitude": bands, "bin": np.arange(n_bins)},
    )
    ds["threshold"].attrs["units"] = threshold_units(samples.variable)
    ds.attrs["lead_hours"] = samples.lead_hours
    ds.attrs["n_bootstraps"] = n_bootstraps
    ds.attrs["binning"] = "zero-band + quantiles of the informative tail"
    return ds


def plot_reliability(ds, out_png, band="all"):
    """Save a reliability diagram (one curve per threshold, with bootstrap CI) for one band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_ds = ds.sel(altitude=band)
    extent = np.concatenate([band_ds["forecast_probability"].values.ravel(),
                             band_ds["observed_frequency"].values.ravel(),
                             band_ds["obs_freq_upper"].values.ravel()])
    top = np.nanmax(extent)
    lim = 1.0 if not np.isfinite(top) else min(1.0, top * 1.08)

    fig, ax = plt.subplots(figsize=(6, 6))
    for thr in ds["threshold"].values:
        sub = ds.sel(threshold=thr, altitude=band)
        fp = sub["forecast_probability"].values
        of = sub["observed_frequency"].values
        lo = sub["obs_freq_lower"].values
        hi = sub["obs_freq_upper"].values
        ax.errorbar(fp, of, yerr=[of - lo, hi - of], marker="o", ms=4, lw=1.5,
                    capsize=2,
                    label=f"{int(thr)} {_thr_units(ds)}".rstrip())
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability — {ds.attrs.get('lead_hours')}h — altitude: {band}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Scalar-field metrics (CRPS, and the same shape as MAE / SEEPS) for the
# existing dashboard. These reduce a per-cell field to the four annual/monthly
# summaries the UI loader reads; they share nothing with the threshold family.
# ---------------------------------------------------------------------------

def annual_monthly_averages(da, time_dim="time", band_masks=None):
    """Reduce a (time, *space) field to the dashboard's summary variables.

    Mirrors the upstream weather-postprocessing transform: group by year and by month and
    average over time, then over the spatial dims (plain mean, skipna — no area weighting,
    to match the existing CRPS/MAE/SEEPS summaries). Returns a Dataset with annual_average,
    annual_spatial_average, monthly_average, and monthly_spatial_average.

    ``band_masks`` (label → boolean mask over the spatial dims; numeric altitude
    bands only — "all" stays the legacy unmasked scalars) appends the variables the
    dashboard's altitude selector reads:

    - ``annual_average_by_altitude(year, altitude)`` / ``monthly_average_by_altitude
      (month, altitude)`` — per-band spatial means, ``altitude`` coord = band labels;
    - ``altitude_band(*space)`` — int8 band index per cell/station (−1 = no band),
      index order matching the ``altitude`` coord, so the UI can mask maps and
      station markers without knowing the DEM.

    Without ``band_masks`` the file keeps its legacy shape (old data stays valid).
    """
    spatial = [d for d in da.dims if d != time_dim]
    annual_spatial = da.groupby(f"{time_dim}.year").mean(time_dim, skipna=True)
    monthly_spatial = da.groupby(f"{time_dim}.month").mean(time_dim, skipna=True)
    ds = xr.Dataset(
        {"annual_average": annual_spatial.mean(spatial, skipna=True),
         "annual_spatial_average": annual_spatial,
         "monthly_average": monthly_spatial.mean(spatial, skipna=True),
         "monthly_spatial_average": monthly_spatial},
    )
    if band_masks:
        labels = list(band_masks)
        coords = {d: da[d] for d in spatial if d in da.coords}
        masks_da = [xr.DataArray(np.asarray(m), dims=spatial, coords=coords)
                    for m in band_masks.values()]
        ds["annual_average_by_altitude"] = xr.concat(
            [annual_spatial.where(m).mean(spatial, skipna=True) for m in masks_da],
            dim="altitude").assign_coords(altitude=labels)
        ds["monthly_average_by_altitude"] = xr.concat(
            [monthly_spatial.where(m).mean(spatial, skipna=True) for m in masks_da],
            dim="altitude").assign_coords(altitude=labels)
        band_idx = np.full([da.sizes[d] for d in spatial], -1, dtype=np.int8)
        for i, m in enumerate(band_masks.values()):
            band_idx[np.asarray(m)] = i
        ds["altitude_band"] = xr.DataArray(
            band_idx, dims=spatial, coords=coords,
            attrs={"long_name": "altitude band index (-1 = no band)",
                   "band_labels": ",".join(labels)})
        ds["altitude"].attrs["units"] = "m"
    return ds


def band_masks_for(field, dem_path=None):
    """Per-band boolean masks for a summary field's spatial layout, or None.

    Gridded (lat, lon) fields stratify by the DEM interpolated onto the field's
    grid; station fields use each gauge's own ``elevation`` coord (no DEM).
    Numeric bands only — "all" is the legacy unmasked scalars, not a band.
    Returns None (→ legacy-shaped summary file) when the field has no usable
    elevation source, e.g. the DEM file is absent on this deployment.
    """
    if "lat" in field.dims and "lon" in field.dims:
        path = dem_path if dem_path is not None else store().dem()
        if path is None or not Path(path).exists():
            print(f"  note: no DEM available ({path}); "
                  "writing summaries without altitude bands")
            return None
        elev = interp_dem(field, path)
    elif "station" in field.dims and "elevation" in field.coords:
        elev = field["elevation"]
    else:
        return None
    masks = altitude_masks(elev)
    return {b: masks[b] for b, rng in altitude_bands().items() if rng is not None}


def compute_crps(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                 variable=DEFAULT_VARIABLE, fc_root=None):
    """Aggregate a model's per-cell CRPS into the UI annual/monthly summaries.

    Works for gridded or station, precip or temperature — `crps` reduces over
    whatever spatial dim it has, in whatever units the calibration wrote. `fc_root`
    (if given) wins; otherwise the gridded IDR root for (model, year, variable).
    """
    fc = load_forecast(lead_hours, fc_root or model_fc_root(model, year, variable))
    ds = annual_monthly_averages(fc["crps"], band_masks=band_masks_for(fc["crps"]))
    units = fc["crps"].attrs.get("units", "")
    for v in ds.data_vars:
        if v != "altitude_band":          # int8 index layer, not a physical quantity
            ds[v].attrs["units"] = units
    ds.attrs["lead_hours"] = lead_hours
    return ds


CORRELATIONS = ("pearsonr", "spearmanr")


def _safe_correlation(x, y, axis, corr_type):
    """Compute correlation while silencing constant-input warnings.

    ``corr_type`` is ``"pearsonr"`` or ``"spearmanr"`` (Pearson of the ranks) and has no
    default: a default once let two call sites silently score "spearmanr" as Pearson.
    Returns NaN for slices where either input is constant or has fewer than two
    finite paired samples.
    """
    if corr_type not in CORRELATIONS:
        raise ValueError(f"corr_type must be one of {CORRELATIONS}, got {corr_type!r}")
    result = np.full(x.shape[1 - axis], np.nan, dtype=float)
    for i in range(x.shape[1 - axis]):
        mask = np.isfinite(x.take(i, axis=1 - axis)) & np.isfinite(y.take(i, axis=1 - axis))
        if mask.sum()>=2:
            x_vals = x.take(i, axis=1 - axis)[mask]
            y_vals = y.take(i, axis=1 - axis)[mask]
            if np.nanmin(x_vals)==np.nanmax(x_vals) or\
            np.nanmin(y_vals)==np.nanmax(y_vals):
                continue
            if corr_type=='spearmanr':
                x_vals = rankdata(x_vals,nan_policy='omit')
                y_vals = rankdata(y_vals,nan_policy='omit')
            result[i] = np.corrcoef(x_vals[None,:],y_vals[None,:])[0,1]
    return result

def corr_annual_monthly_averages(fc_raw, obs, corr_type, time_dim="time"):
    # For simplicity we replace the fcst init time with valid time
    # and drop prediction_timedelta, since we only care about the valid time for correlation

    fc = fc_raw.copy()
    fc[time_dim] = fc[time_dim] + fc["prediction_timedelta"]
    fc = fc.squeeze("prediction_timedelta")

    spatial_dims = [d for d in fc.dims if d != time_dim]

    fc_shape = fc.values.shape
    obs_shape = obs.values.shape
    if fc_shape != obs_shape:
        raise ValueError(f"Forecast and observation arrays must have the same shape. "
                         f"Got {fc_shape} and {obs_shape}.")
    # Compute annual and monthly averages along the time dimension
    # Two different correlations share this file:
    #   *_spatial_average: per grid cell, across days (does the cell's day-to-day signal follow?)
    #   annual_average / monthly_average: per day, across grid cells (does the day's map
    #     match?), then averaged over the days of the year or month.
    annual_spatial_average = _safe_correlation(fc.values.reshape(fc_shape[0], -1), obs.values.reshape(obs_shape[0], -1), axis=0, corr_type=corr_type).reshape(fc_shape[1:])
    annual_average = _safe_correlation(fc.values.reshape(fc_shape[0], -1), obs.values.reshape(obs_shape[0], -1), axis=1, corr_type=corr_type)

    months = np.unique(fc[time_dim].dt.month.values)
    monthly_spatial_average = np.zeros((len(months), *fc_shape[1:]))
    monthly_average = np.zeros(len(months))
    for i, month in enumerate(months):
        fc_month = fc.sel({time_dim: fc[time_dim].dt.month == month})
        obs_month = obs.sel({time_dim: obs[time_dim].dt.month == month})
        monthly_spatial_average[i] = _safe_correlation(
            fc_month.values.reshape(fc_month.values.shape[0], -1),
            obs_month.values.reshape(obs_month.values.shape[0], -1),
            axis=0,corr_type=corr_type
        ).reshape(fc_shape[1:])
        monthly_average[i] = np.nanmean(_safe_correlation(
            fc_month.values.reshape(fc_month.values.shape[0], -1),
            obs_month.values.reshape(obs_month.values.shape[0], -1),
            axis=1,corr_type=corr_type,
        ), axis=0)

    return xr.Dataset(
        {"annual_average": (["time"], annual_average),
         "annual_spatial_average": (["latitude", "longitude"], annual_spatial_average),
         "monthly_average": (["month"], monthly_average),
         "monthly_spatial_average": (["month", "latitude", "longitude"], monthly_spatial_average)},
         coords={"time": np.unique(fc[time_dim].values),
                 "month": months,
                 "latitude": fc[spatial_dims[0]].values,
                 "longitude": fc[spatial_dims[1]].values},
    )

def _load_gridded_obs(variable, year):
    """Gridded obs for ``variable`` in the metric's standard unit — precipitation
    ÷24 → mm/hr, temperature left in °C (not a flux) — as a (time, lat, lon) DataArray."""
    spec = _VARIABLE[variable]
    obs = xr.open_dataset(gridded_obs_path(year, variable))
    rename = {o: o.lower() for o in ("Lat", "Lon") if o in obs.dims or o in obs.coords}
    obs = obs.rename(rename)
    da = obs[_truth_field(obs, variable)]
    return da / 24.0 if spec["to_rate"] else da


def compute_deterministic(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                          variable=DEFAULT_VARIABLE, fc_root=None, metric='mae', source='raw'):
    """Deterministic skill (mae / bias / spearmanr / pearsonr) of a single-valued forecast
    against the gridded truth, summarised into the UI annual/monthly file. Works for precipitation
    (mm/hr) and temperature_max (°C); the per-variable spec picks the obs/forecast vars.

    ``source='raw'``        scores the harmonized raw forecast (``raw_fc_var``). The valid
                            date is ``time + prediction_timedelta`` — the day the forecast's
                            daily window covers, the offset that maximises forecast/obs
                            correlation at every lead.
    ``source='calibrated'`` scores the IDR calibrated median (``q50``) on the q50's own
                            ``time`` axis (the calibration puts that on the valid date; no
                            verification-side shift is applied).
    """
    spec = _VARIABLE[variable]
    if source == 'raw':
        fc = load_forecast(lead_hours, fc_root or model_fc_root(model, year, variable, raw=True))
        fc_da = fc[spec["raw_fc_var"]]
        # Valid date = init + prediction_timedelta (the day the daily window covers).
        ptd = fc_da["prediction_timedelta"].values.reshape(-1)
        if ptd.size == 0:                  # e.g. day-9: window runs past the horizon
            raise FileNotFoundError(f"empty prediction_timedelta at {lead_hours}h")
        fc_dates = (fc_da["time"].values + ptd[0]).astype('datetime64[D]')
    elif source == 'calibrated':
        fc = load_forecast(lead_hours, fc_root or model_fc_root(model, year, variable))
        fc_da = fc["q50"]
        fc_dates = fc_da["time"].values.astype('datetime64[D]')
    else:
        raise ValueError(f"unknown source {source!r}; expected 'raw' or 'calibrated'")
    if "prediction_timedelta" in fc_da.dims:
        fc_da = fc_da.squeeze("prediction_timedelta", drop=True)

    obs = _load_gridded_obs(variable, year)
    # Snap obs lat/lon onto the forecast grid: the same grid can differ in the last
    # float digit, which would misalign the
    # subtraction and silently NaN most cells.
    if (obs.sizes.get("lat"), obs.sizes.get("lon")) == (fc_da.sizes.get("lat"), fc_da.sizes.get("lon")):
        obs = obs.assign_coords(lat=fc_da["lat"], lon=fc_da["lon"])
    _, io, ifc = np.intersect1d(obs["time"].values.astype('datetime64[D]'),
                                fc_dates, return_indices=True)
    # Pair positionally, then put the forecast on the obs (valid) time axis so the
    # arithmetic aligns 1:1 and the monthly grouping uses the valid date.
    fc_da = fc_da.isel(time=ifc).assign_coords(time=obs["time"].values[io])
    obs_da = obs.isel(time=io)

    if metric in ('mae', 'bias'):
        field = np.abs(fc_da - obs_da) if metric == 'mae' else fc_da - obs_da
        ds = annual_monthly_averages(field, band_masks=band_masks_for(field))
    elif metric in ('spearmanr', 'pearsonr'):
        # The pair is already on a shared valid-date axis; corr_annual_monthly_averages
        # adds prediction_timedelta then drops it, so a zero shift leaves it in place.
        # (No altitude bands — the correlation reducer is not a spatial mean.)
        fc_ptd0 = fc_da.expand_dims(prediction_timedelta=[np.timedelta64(0, "ns")])
        ds = corr_annual_monthly_averages(fc_ptd0, obs_da, corr_type=metric)
    else:
        raise ValueError(f"unsupported deterministic metric {metric!r}")

    units = fc_da.attrs.get("units", "")
    for v in ds.data_vars:
        if v != "altitude_band":          # int8 index layer, not a physical quantity
            ds[v].attrs["units"] = units
    ds.attrs["lead_hours"] = lead_hours
    return ds


def compute_deterministic_station(lead_hours, model="graphcast", year=DEFAULT_YEAR,
                                  fc_root=None, obs_dir=None, metric='mae', source='raw'):
    """Deterministic precip skill (mae / bias) at gauge stations — the station counterpart of
    :func:`compute_deterministic`, on the (time, station) grid.

    ``source='raw'``        scores the forecast-at-station; its valid date is
                            ``time + prediction_timedelta`` (``align_stations`` then floors
                            it to the day the gauge's window covers).
    ``source='calibrated'`` scores the IDR ``q50`` from the station tree (already valid-date).

    Forecast and gauge are paired by ``(name, gauge-lat, gauge-lon)`` + date via
    :func:`align_stations` and reduced over the ``station`` axis. Both are mm/hr (gauge ÷24).
    Correlation is not supported at stations (the gridded corr reducer assumes two spatial
    dims); use mae / bias.
    """
    from calibration.run_station import align_stations   # (name, lat, lon) + date pairing

    if source == 'raw':
        fc = load_forecast(lead_hours, fc_root or station_raw_fc_root(model, year))
        da = fc["precipitation"]
        ptd = da["prediction_timedelta"].values.reshape(-1)
        if ptd.size == 0:                  # e.g. day-9: window runs past the horizon
            raise FileNotFoundError(f"empty prediction_timedelta at {lead_hours}h")
        if "prediction_timedelta" in da.dims:
            da = da.squeeze("prediction_timedelta", drop=True)
        da = da.assign_coords(time=da["time"] + ptd[0])   # init -> valid; align_stations floors
    elif source == 'calibrated':
        fc = load_forecast(lead_hours, fc_root or station_fc_root(model, year))
        da = fc["q50"]
    else:
        raise ValueError(f"unknown source {source!r}; expected 'raw' or 'calibrated'")
    if metric not in ('mae', 'bias'):
        raise ValueError(f"station deterministic supports mae/bias, not {metric!r}")

    units = da.attrs.get("units", "")
    obs = load_obs_station(year, obs_dir)["total_precipitation_rate"] / 24.0   # mm/day -> mm/hr
    fc_al, obs_al = align_stations(da, obs)

    diff = fc_al - obs_al if metric == 'bias' else np.abs(fc_al - obs_al)
    # Band masks need a per-station elevation, taken from whichever side of the
    # aligned pair carries it: the calibrated forecast has its own `elevation`
    # coord, the raw forecast-at-station does not — there the gauge obs supplies
    # it (same station order as diff). Arithmetic can drop the coord, so don't
    # rely on diff carrying it.
    elev_src = fc_al if "elevation" in fc_al.coords else obs_al
    ds = annual_monthly_averages(diff, band_masks=band_masks_for(elev_src))
    for v in ds.data_vars:
        if v != "altitude_band":          # int8 index layer, not a physical quantity
            ds[v].attrs["units"] = units
    ds.attrs["lead_hours"] = lead_hours
    return ds


def plot_spatial(ds, out_png, metric="CRPS"):
    """Save a seasonal metric preview (domain-mean CRPS per month)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monthly = np.atleast_1d(ds["monthly_average"].values.squeeze())
    months = ds["month"].values if "month" in ds.coords else np.arange(1, len(monthly) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(months, monthly, marker="o", lw=2, color="#0a9938")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{metric} ({ds['monthly_average'].attrs.get('units', '')})")
    ax.set_title(f"{metric} seasonal cycle — {ds.attrs.get('lead_hours')}h")
    ax.set_xticks(range(1, 13))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png

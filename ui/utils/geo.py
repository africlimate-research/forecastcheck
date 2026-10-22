# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import xarray as xr
from ui.utils.regions import mask_region

# Regional result caches, keyed by the slice's PROVENANCE (its source file) plus
# the region — never by id(ltd). An address-keyed cache is only safe while every
# LeadTimeData is retained for the process lifetime, which is true of the legacy
# tab layout (ui.components.scores_tab._DATA_CACHE) but NOT of the multipage app:
# its pages call load_run() fresh on every rerun and keep nothing, so CPython
# recycles the addresses and the cache serves another file's numbers — observed
# as the same selection plotting a different curve on each render, and as a
# rainfall value appearing on a temperature axis. See _cache_key.
_MEAN_CACHE:    dict[tuple, float]         = {}
_MONTHLY_CACHE: dict[tuple, np.ndarray]    = {}
_GRID_CACHE:    dict[tuple, xr.DataArray]  = {}
_CLIM_GRID_CACHE: dict[tuple, xr.DataArray] = {}  # keyed by (id(clim), region)

#: Cap on _GRID_CACHE. Its entries are masked full-domain grids (~2 MB each) and,
#: now that the keys are stable, they accumulate across every
#: (run × lead × region × band) the user visits instead of being overwritten.
#: Oldest-first eviction keeps a long session bounded; the scalar/monthly caches
#: hold a float and a (12,) array so they need no cap.
_GRID_CACHE_MAX = 256


def _cache_key(obj, *rest) -> tuple | None:
    """Cache key from the file ``obj`` was loaded from, or None if unknown.

    ``LeadTimeData.source`` / ``ClimData.source`` is the *path* the slice was read
    from — a value identity that stays correct across reruns, unlike ``id(obj)``:
    these objects are rebuilt per render and freed, so their addresses are handed
    back to later slices of other variables, metrics and lead times. The path (not
    just the run tokens) is the identity, so two results roots holding the same run
    can never share an entry. None means the object carries no provenance
    (hand-built in tests, or a type that has none), and callers then compute
    without caching rather than risk answering from a stale entry.
    """
    src = getattr(obj, "source", None)
    return None if src is None else (src, *rest)


def is_region_warm(ltd: "LeadTimeData", region: str) -> bool:  # type: ignore[name-defined]
    """Return True if the scalar/monthly caches are populated for this (ltd, region).

    Spatial grids are computed on demand by the map tab and are not required
    for time-series or seasonal charts, so _GRID_CACHE is not checked here.
    A slice with no provenance is never cached, so it is never warm.
    """
    key = _cache_key(ltd, region)
    return key is not None and key in _MEAN_CACHE and key in _MONTHLY_CACHE


def ensure_region_warm(
    all_data: "dict[str, dict[int, LeadTimeData]]",  # type: ignore[name-defined]
    region: str,
    on_progress: "Callable[[int, int], None] | None" = None,  # type: ignore[name-defined]
) -> None:
    """Populate scalar and monthly caches for every LeadTimeData in all_data.

    Spatial grids (annual_spatial_grid) are NOT warmed here — they are large
    (2 MB each × all lead times × all regions = hundreds of GB) and are built
    on demand by the spatial map tab only for the lead time being viewed.

    on_progress(done, total) is called after each cache operation so callers
    can drive a progress bar with fine-grained percentage updates.
    """
    items = [
        ltd
        for lt_map in all_data.values()
        for ltd in lt_map.values()
        if ltd is not None
    ]
    total = len(items) * 2  # mean + monthly per LeadTimeData
    done = 0
    for ltd in items:
        regional_annual_mean(ltd, region)
        done += 1
        if on_progress:
            on_progress(done, total)
        regional_monthly_means(ltd, region)
        done += 1
        if on_progress:
            on_progress(done, total)


def subset_region(da: xr.DataArray, region: str) -> xr.DataArray:
    """Mask and crop a DataArray to the given region using exact country borders (cached)."""
    key = (id(da), region)
    if key not in _CLIM_GRID_CACHE:
        _CLIM_GRID_CACHE[key] = mask_region(da, region)
    return _CLIM_GRID_CACHE[key]


def apply_band(da: xr.DataArray, ltd: "LeadTimeData", altitude_band: str = "all") -> xr.DataArray:  # type: ignore[name-defined]
    """NaN-mask a spatial/station field outside the chosen altitude band.

    Uses the pipeline's ``altitude_band`` index layer (same file, so same
    cell/station order as the field) — the UI never touches the DEM.
    ``"all"``, legacy data without the layer, or an unknown label are no-ops.
    """
    if altitude_band in (None, "all"):
        return da
    layer = getattr(ltd, "altitude_band", None)
    if layer is None:
        return da
    # Band index order: the layer's band_labels attr, falling back to the
    # altitude coord of the band means (same order by construction) in case
    # an intermediary stripped the attrs.
    labels = [s for s in str(layer.attrs.get("band_labels", "")).split(",") if s]
    if not labels and getattr(ltd, "annual_average_by_altitude", None) is not None:
        labels = [str(b) for b in ltd.annual_average_by_altitude["altitude"].values]
    if altitude_band not in labels:
        return da
    return da.where(layer.squeeze() == labels.index(altitude_band))


def annual_spatial_grid(ltd: "LeadTimeData", region: str, altitude_band: str = "all") -> xr.DataArray:  # type: ignore[name-defined]
    """Return the masked annual spatial grid for a LeadTimeData object (cached).

    Caches the result of mask_region(ltd.annual_spatial_average.squeeze(), region)
    so spatial map renders don't recompute the da.where(keep).isel() chain on every
    frame — the masked numpy array is built once and reused. ``altitude_band``
    additionally NaN-masks cells outside the band (band first, then region —
    the band layer shares the full-domain grid, mask_region crops it).
    """
    key = _cache_key(ltd, region, altitude_band)
    if key is not None and key in _GRID_CACHE:
        return _GRID_CACHE[key]

    base = apply_band(ltd.annual_spatial_average.squeeze(), ltd, altitude_band)
    grid = mask_region(base, region)

    if key is not None:
        if len(_GRID_CACHE) >= _GRID_CACHE_MAX:
            # Oldest first (dicts keep insertion order) — these are ~2 MB each.
            del _GRID_CACHE[next(iter(_GRID_CACHE))]
        _GRID_CACHE[key] = grid
    return grid


def regional_annual_mean(ltd: "LeadTimeData", region: str) -> float:  # type: ignore[name-defined]
    """Spatial mean of annual_spatial_average within the region, or NaN (cached).

    **NaN when the region holds no data**, never the whole-domain value. A region with no
    gauge in it is the common case, not an edge case, and substituting the national figure
    would label it as that region's score. A blank chart is recoverable; a plausible wrong number is not.
    The same goes for a masking failure: it is surfaced as NaN rather than papered over.
    """
    key = _cache_key(ltd, region)
    if key is not None and key in _MEAN_CACHE:
        return _MEAN_CACHE[key]

    if region == "Africa":
        result = float(ltd.annual_average)
    else:
        try:
            grid = mask_region(ltd.annual_spatial_average, region)
            result = float(grid.mean().values)      # NaN when nothing is inside
        except Exception as exc:  # noqa: BLE001 — an unmaskable region has no value
            print(f"[geo] regional mean unavailable for {region!r}: {exc}")
            result = float("nan")

    if key is not None:
        _MEAN_CACHE[key] = result
    return result


def regional_monthly_means(ltd: "LeadTimeData", region: str) -> np.ndarray:  # type: ignore[name-defined]
    """(12,) array of regional spatial means indexed Jan→Dec.

    Values are placed at their TRUE calendar-month positions using the
    NetCDF's `month` coordinate. Months absent from the data are NaN —
    not silently filled with the next available month, which previously
    made April's value appear under January for 2024 IFS+IDR (months 4-12).
    """
    key = _cache_key(ltd, region)
    if key is not None and key in _MONTHLY_CACHE:
        return _MONTHLY_CACHE[key]

    da = ltd.monthly_spatial_average

    # Read the real month numbers from the NetCDF — these are the
    # calendar months each value actually represents (e.g. [4, 5, ..., 12]
    # for an April-onwards file).
    months: list[int] | None = None
    if da is not None:
        try:
            month_dim = next(d for d in ("month", "time") if d in da.dims)
            months = [int(m) for m in da[month_dim].values]
        except (StopIteration, Exception):
            months = None

    if region == "Africa":
        arr = np.asarray(ltd.monthly_average)
    elif da is None:
        # monthly_spatial_average was freed after precaching; all regional
        # monthly caches should have been populated before freeing.  If we
        # reach here it means this (ltd, region) pair was never precached
        # (e.g. a new region added after startup) — return NaN so charts
        # degrade gracefully rather than silently showing Africa values.
        result = np.full(12, np.nan)
        if key is not None:
            _MONTHLY_CACHE[key] = result
        return result
    else:
        try:
            da_sub  = mask_region(da, region)
            sp_dims = [d for d in da_sub.dims if d not in ("month", "time")]
            vals    = da_sub.mean(dim=sp_dims).values
            # All-NaN means nothing lies inside the region. Keep it NaN — the branch
            # above already refuses to fall back to whole-domain values, and doing it
            # here would relabel the national series as this region's.
            arr = vals.astype(float)
        except Exception as exc:  # noqa: BLE001
            print(f"[geo] regional monthly means unavailable for {region!r}: {exc}")
            arr = np.full(12, np.nan)

    # A single-month file can arrive as a 0-d scalar; normalise to 1-d so
    # the month-length checks below never index an empty shape.
    arr = np.atleast_1d(arr)
    result = np.full(12, np.nan)
    if months is not None and len(months) == arr.shape[0]:
        # Honour the file's month labels — value at index 0 of `arr` may
        # correspond to any calendar month (e.g. April for a year that
        # only has data from April onwards).
        for m, v in zip(months, arr):
            if 1 <= m <= 12:
                result[m - 1] = v
    else:
        # Fallback when month coords are unavailable (e.g. cache is read
        # back after `monthly_spatial_average` was freed and we never saw
        # the coordinate). Assume the array is already Jan-indexed.
        n = min(arr.shape[0], 12)
        result[:n] = arr[:n]

    if key is not None:
        _MONTHLY_CACHE[key] = result
    return result


# ── Altitude bands — pipeline-computed scalars, read as-is ──────────────────
# The altitude selector is gated to the deployment country, whose data domain
# IS that country, so the whole-domain band means the pipeline stored are the
# region's band means by construction — no client-side masking of means.

def has_band_data(ltd: "LeadTimeData") -> bool:  # type: ignore[name-defined]
    """True when this slice was produced by the band-aware pipeline."""
    return getattr(ltd, "annual_average_by_altitude", None) is not None


def band_annual_mean(ltd: "LeadTimeData", band: str) -> float:  # type: ignore[name-defined]
    """One altitude band's annual mean (pipeline-computed); NaN when absent."""
    da = getattr(ltd, "annual_average_by_altitude", None)
    if da is None:
        return float("nan")
    try:
        return float(np.asarray(da.sel(altitude=band).values).squeeze())
    except Exception:
        return float("nan")


def band_monthly_means(ltd: "LeadTimeData", band: str) -> np.ndarray:  # type: ignore[name-defined]
    """(12,) Jan→Dec array of one band's monthly means, NaN where absent.

    Same true-calendar-month placement contract as :func:`regional_monthly_means`:
    values land at their month positions; missing months stay NaN.
    """
    result = np.full(12, np.nan)
    da = getattr(ltd, "monthly_average_by_altitude", None)
    if da is None:
        return result
    try:
        sel = da.sel(altitude=band)
        months = [int(m) for m in np.atleast_1d(sel["month"].values)]
        vals = np.atleast_1d(np.asarray(sel.values, dtype=float)).ravel()
        for m, v in zip(months, vals):
            if 1 <= m <= 12:
                result[m - 1] = v
    except Exception:
        pass
    return result


def regional_season_mean(
    ltd: "LeadTimeData",  # type: ignore[name-defined]
    region: str,
    months: "tuple[int, ...] | list[int] | None",
) -> float:
    """Regional season mean for a LeadTimeData/ClimData object.

    ``months`` is a tuple/list of 1-based calendar months (e.g. (7, 8, 9, 10) for
    JASO); ``None`` (or empty) means Yearly, delegating to
    :func:`regional_annual_mean`. Otherwise the value is the unweighted nanmean
    of the cached regional monthly means over those months — a season with no
    data (all months NaN) returns NaN so charts degrade gracefully.
    """
    if not months:
        return regional_annual_mean(ltd, region)
    monthly = regional_monthly_means(ltd, region)
    vals = monthly[[m - 1 for m in months]]
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")


def band_season_mean(
    ltd: "LeadTimeData",  # type: ignore[name-defined]
    band: str,
    months: "tuple[int, ...] | list[int] | None",
) -> float:
    """One altitude band's season mean — the band counterpart of
    :func:`regional_season_mean`: nanmean of the band's monthly means over the
    season's months; Yearly (None/empty) delegates to :func:`band_annual_mean`.
    """
    if not months:
        return band_annual_mean(ltd, band)
    monthly = band_monthly_means(ltd, band)
    vals = monthly[[m - 1 for m in months]]
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")


def prewarm_regional_cache(
    data_cache: dict,
    regions:    list[str],
) -> None:
    """Pre-compute regional scalar and monthly means for every (LeadTimeData, region) pair.

    Spatial grids are intentionally excluded: storing a full Africa-sized masked
    DataArray for every (ltd, region) combination would require hundreds of GB.
    They are computed on demand by the spatial map tab and cached lazily.
    """
    total = 0
    for region in regions:                  # Africa first → blocs → individual countries
        for lt_map in data_cache.values():
            if not isinstance(lt_map, dict):
                continue
            for ltd in lt_map.values():
                if ltd is None:
                    continue
                regional_annual_mean(ltd, region)
                regional_monthly_means(ltd, region)
                total += 1
    print(f"[prewarm_regional_cache] {total} (region × ltd) scalar/monthly entries cached")

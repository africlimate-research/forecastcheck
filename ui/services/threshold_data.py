"""Discovery + loading of threshold (curve) diagnostics for the dashboard.

The curve metrics (reliability / ROC / cost-loss) are stored flat at
``{root}/{forecast}/{native_var}/{target}/{metric}/{year}/{lead}h.nc`` — i.e. the file
sits directly under the year, with no ``annual_monthly_averages`` subdir — so this
discovery never collides with the scalar-summary discovery in :mod:`ui.services.engine`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional, Union

import pandas as pd
import xarray as xr

from ui.backend.nc_io import read_dataset
from ui.services.engine import RUN_KEYS
from verification import curves as _curves
from verification.regions import default_region
from verification.seasons import DEFAULT_SEASON_SLUG, season_months
from verification.store import (BY_REGION_SUBDIR, SUBDIR_SEGMENTS, facet_relpath,
                                parse_facet_relpath, split_truth_source,
                                truth_source)

CURVE_METRICS = ("reliability", "roc", "cost_loss")
# Facet columns of the discovered curve index: the run keys, plus the region and season a
# file belongs to (whole-country files are region-less and yearly files are season-less on
# disk, labelled default_region() / DEFAULT_SEASON_SLUG respectively).
CURVE_KEYS = (*RUN_KEYS, "region", "season")


def _parse_curve_path(path, root) -> Optional[dict]:
    """Parse a curve-file path under ``root`` into run tokens + region + season + lead, or None.

    The structural decoding (segment order, and telling an admin-1 region segment apart
    from a season slug by controlled vocabulary) lives in
    :func:`verification.store.parse_facet_relpath`, shared with the pipeline that writes
    these files and with the scalar-summary reader. What stays here is this family's own
    business: excluding the scalar summaries, restricting to curve metrics, and turning
    "no segment" into the DISPLAY defaults the selectors expect — whole country and
    yearly — which is why the two readers disagree about None on purpose.
    """
    got = parse_facet_relpath(path, root)
    # Any output-family subdir is a different product, not a curve: the scalar
    # summaries and the additive by-region count store both live under one.
    if got is None or got["subdir"] in SUBDIR_SEGMENTS:
        return None
    if got["metric"] not in CURVE_METRICS:
        return None
    return {"forecast": got["model"], "native_var": got["variable"],
            "target": truth_source(got["product"], got["target"]),
            "metric": got["metric"],
            "year": got["year"],
            "region": got["region"] or default_region(),
            "season": got["season"] or DEFAULT_SEASON_SLUG,
            "lead": got["lead"]}


def _store_entries(path, root) -> list:
    """Index a by-region count store as the (region, season) combinations it can serve.

    A store is not a curve — it is counts — but it is what every curve for that run is
    derived from, so it advertises the whole offered grid rather than the handful of
    combinations the pipeline happened to write files for. This is what puts a season on
    the selector that no file exists for, which is the entire point of storing counts.

    Cost-loss has no store of its own; it is derived from ROC's, so a ROC store advertises
    both.
    """
    got = parse_facet_relpath(path, root)
    if got is None or got["subdir"] != BY_REGION_SUBDIR:
        return []
    metrics = [m for m, src in _curves.SOURCE_METRIC.items() if src == got["metric"]]
    if not metrics:
        return []
    return [{"forecast": got["model"], "native_var": got["variable"],
             "target": truth_source(got["product"], got["target"]),
             "metric": metric, "year": got["year"],
             "region": region, "season": season, "lead": got["lead"]}
            for metric in metrics
            for region in offered_regions()
            for season in offered_seasons()]


def offered_regions() -> list:
    """Every region this deployment can show, from the catalogue — not from the files."""
    from verification.regions import available_regions

    return list(available_regions())


def offered_seasons() -> list:
    """Every season slug this deployment offers, from its calendar — not from the files."""
    from verification import config
    from verification.seasons import seasons_for_country

    return list(seasons_for_country(config.country()))


def _index_entries(path, root) -> list:
    """Index rows for one file: a legacy curve file, or a store advertising its grid."""
    entry = _parse_curve_path(path, root)
    if entry is not None:
        return [entry]
    return _store_entries(path, root)


def discover_curves(root: Union[Path, str, None] = None) -> pd.DataFrame:
    """Index the curve diagnostics under ``root`` as a faceted DataFrame (one row per run+region)."""
    columns = [*CURVE_KEYS, "leads"]
    root_path = Path(root) if root is not None else Path(".")
    if not root_path.exists():
        return pd.DataFrame(columns=columns)
    by_run: dict = {}
    for nc in sorted(root_path.rglob("*h.nc")):
        for idx in _index_entries(nc, root_path):
            by_run.setdefault(tuple(idx[k] for k in CURVE_KEYS), set()).add(idx["lead"])
    rows = [{**dict(zip(CURVE_KEYS, key)), "leads": sorted(leads)}
            for key, leads in by_run.items()]
    rows.sort(key=lambda r: tuple(r[k] for k in CURVE_KEYS))
    return pd.DataFrame(rows, columns=columns)


def _leaf_ids(region) -> Optional[list]:
    """Leaf region ids for a picked region name, or None for the whole country.

    None means "every leaf", which is what the national total is. A name the catalogue
    cannot resolve raises rather than quietly widening to the whole country — showing
    national numbers under a regional heading is the failure mode worth refusing.
    """
    if not region or region == default_region():
        return None
    from verification import admin

    catalogue = admin.catalogue_for()
    region_id = catalogue.id_for(region)
    if region_id is None:
        raise KeyError(f"unknown region {region!r} for this deployment")
    return list(catalogue.leaves_under(region_id))


def _store_path(base, forecast, native_var, target, metric, year, lead) -> Path:
    """Path of the additive count store a curve metric is derived from.

    cost-loss has no store of its own — it is a pure function of the ROC counts — so it
    reads ROC's. :data:`verification.curves.SOURCE_METRIC` is the one place that mapping
    lives.
    """
    product, target_kind = split_truth_source(target)
    return base / facet_relpath(
        model=forecast, variable=native_var, product=product, target=target_kind,
        metric=_curves.SOURCE_METRIC.get(metric, metric), year=year, lead=lead,
        subdir=BY_REGION_SUBDIR)


def load_curve(forecast, native_var, target, metric, year, lead,
               root: Union[Path, str, None] = None, region=None,
               season=None) -> xr.Dataset:
    """Open the stored curve Dataset for one run + lead (+ optional admin-1 region + season).

    Checks existence first and raises FileNotFoundError before touching the netCDF backend,
    so a lead a run lacks fails cleanly (no C-level error stack) instead of killing the page —
    the selector already restricts choices to existing leads; this is the safety net.
    ``region`` None / the whole country (``default_region()``) reads the region-less path;
    ``season`` None / ``"yearly"`` reads the season-less path. A subregion and/or season insert
    their segments after the year: ``.../{year}/[{region}/][{season}/]{lead}h.nc``.
    """
    base = Path(str(root) if root is not None else ".")

    # A precomputed file, when the pipeline wrote one for this exact combination. It is
    # read in full and is small; deriving the same numbers from the count store means
    # decompressing the whole store (~1.5 GB expanded for a 235-region country) and
    # reducing it, which costs seconds per model. Deriving is the FALLBACK, not the
    # default — the two agree bit for bit, so preferring the cheap one is free.
    product, target_kind = split_truth_source(target)
    path = base / facet_relpath(
        model=forecast, variable=native_var, product=product, target=target_kind,
        metric=metric, year=year, lead=lead,
        region=None if (not region or region == default_region()) else region,
        season=None if (not season or season == DEFAULT_SEASON_SLUG) else season)
    if path.exists():
        return read_dataset(path)

    # No file for this combination: sum it out of the additive count store. This is what
    # makes a region or season nobody pooled renderable at all, and is cached because the
    # reduction is expensive and a dashboard asks for the same slice on every rerun.
    store = _store_path(base, forecast, native_var, target, metric, year, lead)
    if not store.exists():
        raise FileNotFoundError(str(path))
    return _derive_cached(str(store), metric, tuple(_leaf_ids(region) or ()),
                          tuple(season_months(season or DEFAULT_SEASON_SLUG) or ()))


@lru_cache(maxsize=64)
def _derive_cached(store_path: str, metric: str, regions: tuple, months: tuple):
    """Sum a curve out of the count store, memoised on the selection.

    Keyed on the store path plus the exact selection, so two pages asking for the same
    region and season share one reduction. Tuples because the key must be hashable; empty
    means "everything", which is how ``verification.curves`` reads None.

    The store is opened **lazily** rather than through the UI's usual read-everything
    helper: a selection of one province out of 235 otherwise pays to decompress all of
    them, and these arrays expand to well over a gigabyte. Selecting first means the
    backend only reads the slices asked for. The reduction runs inside the lock so the
    small result is fully materialised before the file is released.
    """
    import xarray as xr

    from ui.backend.nc_io import _HDF5_LOCK

    derive = _curves.DERIVERS[metric]
    with _HDF5_LOCK:
        with xr.open_dataset(store_path) as counts:
            out = derive(counts, regions=list(regions) or None,
                         months=list(months) or None)
            return out.load()


# ═════════════════════════════════════════════════════════════════════════
#  How much data is behind one curve
# ═════════════════════════════════════════════════════════════════════════

class EventCount(NamedTuple):
    """Observations behind one (threshold, band) curve.

    ``events`` is the number of times the observation exceeded the threshold and
    ``total`` the size of the evaluation set it was drawn from — both pooled exactly
    as the curve is, so they describe *that* curve rather than the country at large.
    ``fraction`` is the base rate, ``events / total``.

    ``events``/``total`` are None for cost-loss, which stores only the rate.
    """
    events: Optional[float]
    total: Optional[float]
    fraction: Optional[float]


def event_counts(ds, threshold, band: str = "all") -> Optional[EventCount]:
    """Sample behind one curve, or None when the file cannot say.

    Read off the SAME dataset the curve is drawn from, so the numbers cannot describe
    a different pooling than the figure — the region, season, altitude band and date
    subset are already baked into ``ds``. Which is also why this is per-series: models
    do not share an evaluation set. AIFS carries ~8 initialisations a month against
    GraphCast's daily run, so at one threshold its curve rests on roughly a quarter of
    the observations, and a single count printed for the chart would be wrong for one
    of them.

    Each metric stores a different shape, so the counts come from whatever that file
    actually holds:

    * **roc** — the four contingency counts. ``hits + misses`` is the observed-event
      total and is constant across probability bins by construction, so the max over
      bins recovers it without assuming which bin is populated.
    * **reliability** — ``bin_count`` weighted by ``observed_frequency``.
    * **cost_loss** — only ``base_rate``; the absolute counts are not stored, so the
      fraction comes back alone rather than being invented from another file (which
      would silently describe a different pooling).
    """
    import numpy as np

    try:
        sub = ds.sel(threshold=threshold, altitude=band)
    except (KeyError, ValueError):
        return None

    have = set(sub.data_vars)
    if {"hits", "misses", "false_alarms", "correct_negatives"} <= have:
        h = np.asarray(sub["hits"].values, dtype=float)
        m = np.asarray(sub["misses"].values, dtype=float)
        fa = np.asarray(sub["false_alarms"].values, dtype=float)
        cn = np.asarray(sub["correct_negatives"].values, dtype=float)
        events = float(np.nanmax(h + m)) if h.size else 0.0
        total = float(np.nanmax(h + m + fa + cn)) if h.size else 0.0
    elif {"bin_count", "observed_frequency"} <= have:
        c = np.asarray(sub["bin_count"].values, dtype=float)
        o = np.asarray(sub["observed_frequency"].values, dtype=float)
        total = float(np.nansum(c))
        events = float(np.nansum(np.where(np.isfinite(o), c * o, 0.0)))
    elif "base_rate" in have:
        br = np.asarray(sub["base_rate"].values, dtype=float)
        br = br[np.isfinite(br)]
        return EventCount(None, None, float(br.mean()) if br.size else None)
    else:
        return None

    if not np.isfinite(total) or total <= 0:
        return None
    return EventCount(events, total, events / total)

# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Turn stored counts into curves — for any region, any season, at read time.

The pipeline pools **once** and stores contingency counts per (region, month) for ROC and
per (region, day) for reliability. Everything a reader asks for afterwards is a sum over
those axes:

=========================  ====================================================
the user picks             this module does
=========================  ====================================================
whole country              sum every region, every month
one province               sum the leaves under it (:meth:`RegionCatalogue.leaves_under`)
a season                   select its months and sum
a season in one province   both
=========================  ====================================================

Because contingency counts are additive over disjoint pools and every file shares the same
``prob`` axis and the same probability bands, those sums are **exact** — they reproduce,
bit for bit, what pooling that selection directly would have produced. That is what makes
a season a configuration line rather than a pipeline run, and an extra admin level free.

Cost-loss is not stored at all: it is a pure function of the ROC counts, so it is derived
here. Reliability's confidence interval is likewise derived rather than stored, by
replaying the same day-block bootstrap over the days the selection covers —
:func:`verification.diagnostics.bootstrap_ci_from_day_counts` is shared with the pooling
path so the two cannot drift.

Everything here is read-only: no function in this module writes to disk.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from verification import diagnostics as d


def _select(ds: xr.Dataset, regions=None, months=None, *, day_axis: bool) -> xr.Dataset:
    """Restrict a by-region store to the chosen regions and months.

    ``regions`` None means every region — including ``unassigned``, which is what makes
    the national total match the country mask exactly rather than losing its border
    cells. ``months`` None means the whole year.

    Reliability stores a ``day`` axis carrying a ``month`` coordinate; ROC stores ``month``
    directly. ``day_axis`` says which, so a season selects the right thing.
    """
    if regions is not None:
        wanted = [r for r in list(regions) if r in set(ds["region"].values.tolist())]
        ds = ds.sel(region=wanted)
    if months is not None:
        wanted_months = list(months)
        if day_axis:
            keep = np.isin(np.asarray(ds["month"].values), wanted_months)
            ds = ds.isel(day=np.flatnonzero(keep))
        else:
            ds = ds.sel(month=[m for m in ds["month"].values.tolist()
                               if m in wanted_months])
    return ds


def roc_for(ds: xr.Dataset, regions=None, months=None) -> xr.Dataset:
    """ROC contingency counts for one selection, summed from the by-region store.

    Returns the same schema :func:`verification.diagnostics.compute_roc` writes —
    ``(threshold, altitude, prob)`` int64 counts — so every existing reader, plot and
    cost-loss call works unchanged on the result.
    """
    chosen = _select(ds, regions, months, day_axis=False)
    out = xr.Dataset(
        {name: chosen[name].sum(dim=("region", "month")).astype(np.int64)
         for name in d.ROC_COUNT_FIELDS},
        coords={"threshold": chosen["threshold"], "altitude": chosen["altitude"],
                "prob": chosen["prob"]},
    )
    out = out.transpose("threshold", "altitude", "prob")
    # Inherit the STORE's units rather than restating them. These curves are a pure
    # summation of the by-region counts, so the threshold axis is whatever was pooled —
    # mm/day for precipitation, degC for daily-max temperature. Hardcoding "mm/day" put
    # that label on every temperature curve, and a units string is exactly the kind of
    # wrong that nothing downstream can detect.
    out["threshold"].attrs["units"] = ds["threshold"].attrs.get("units", "")
    out["prob"].attrs["long_name"] = "probability cut-off p*"
    out.attrs["lead_hours"] = ds.attrs.get("lead_hours")
    return out


def cost_loss_for(ds: xr.Dataset, regions=None, months=None,
                  alphas=d.COST_LOSS_RATIOS) -> xr.Dataset:
    """Cost-loss value envelope for one selection.

    Derived, never stored: it is a pure function of the ROC counts, so storing it for every
    region would be redundant data that could fall out of step with the counts it
    came from.
    """
    return d.compute_cost_loss(roc_for(ds, regions, months), alphas=alphas)


def reliability_for(ds: xr.Dataset, regions=None, months=None, *,
                    n_bootstraps=d.RELIABILITY_N_BOOTSTRAPS, seed=0) -> xr.Dataset:
    """Reliability diagram for one selection, with an exact bootstrap interval.

    Point estimates come from summing the stored counts: observed frequency is
    ``bin_events / bin_total`` and mean forecast probability is ``prob_sum / bin_total``,
    both over the selected regions and days. Those are exact.

    So is the interval. The bootstrap resamples whole **days**, and days are shared across
    regions, so replaying it over the per-day counts of a selection gives precisely the
    interval that pooling that selection directly would have produced. Days carrying no
    samples for a given (threshold, band) are dropped first, which is what the pooling
    path's ``np.unique`` over observed day indices does implicitly — resampling 365 days
    when only 300 carry data would draw a different, narrower interval.

    Returns the schema :func:`verification.diagnostics.compute_reliability` writes.
    """
    chosen = _select(ds, regions, months, day_axis=True)
    # (day, bin, threshold, altitude) — regions summed away, days kept for the bootstrap.
    total = chosen["bin_total"].sum(dim="region").transpose(
        "day", "bin", "threshold", "altitude").values.astype(float)
    events = chosen["bin_events"].sum(dim="region").transpose(
        "day", "bin", "threshold", "altitude").values.astype(float)
    prob_sum = chosen["prob_sum"].sum(dim="region").transpose(
        "day", "bin", "threshold", "altitude").values

    thresholds = chosen["threshold"].values
    bands = chosen["altitude"].values
    n_bins = chosen.sizes["bin"]
    shape = (len(thresholds), len(bands), n_bins)
    fp = np.full(shape, np.nan)
    of = np.full(shape, np.nan)
    lo = np.full(shape, np.nan)
    hi = np.full(shape, np.nan)
    cnt = np.zeros(shape, dtype=np.int64)

    for i, thr in enumerate(thresholds):
        for j, band in enumerate(bands):
            tot_day = total[:, :, i, j]
            pos_day = events[:, :, i, j]
            psum_day = prob_sum[:, :, i, j]

            n = tot_day.sum(axis=0)
            cnt[i, j] = n.astype(np.int64)
            with np.errstate(invalid="ignore", divide="ignore"):
                of[i, j] = np.where(n > 0, pos_day.sum(axis=0) / n, np.nan)
                fp[i, j] = np.where(n > 0, psum_day.sum(axis=0) / n, np.nan)

            present = tot_day.sum(axis=1) > 0      # days this pool actually saw
            if not present.any():
                continue
            rng = np.random.default_rng(d.bootstrap_seed(seed, thr, band))
            lo[i, j], hi[i, j] = d.bootstrap_ci_from_day_counts(
                tot_day[present], pos_day[present], n_bootstraps, rng)

    dims = ("threshold", "altitude", "bin")
    out = xr.Dataset(
        {"forecast_probability": (dims, fp),
         "observed_frequency": (dims, of),
         "bin_count": (dims, cnt),
         "obs_freq_lower": (dims, lo),
         "obs_freq_upper": (dims, hi),
         "bin_lower": (("threshold", "altitude", "bin"),
                       chosen["bin_lower"].transpose("threshold", "altitude", "bin").values),
         "bin_upper": (("threshold", "altitude", "bin"),
                       chosen["bin_upper"].transpose("threshold", "altitude", "bin").values)},
        coords={"threshold": thresholds, "altitude": bands,
                "bin": np.arange(n_bins)},
    )
    # Inherit the STORE's units rather than restating them. These curves are a pure
    # summation of the by-region counts, so the threshold axis is whatever was pooled —
    # mm/day for precipitation, degC for daily-max temperature. Hardcoding "mm/day" put
    # that label on every temperature curve, and a units string is exactly the kind of
    # wrong that nothing downstream can detect.
    out["threshold"].attrs["units"] = ds["threshold"].attrs.get("units", "")
    out.attrs["lead_hours"] = ds.attrs.get("lead_hours")
    out.attrs["n_bootstraps"] = n_bootstraps
    out.attrs["binning"] = ds.attrs.get("binning", "")
    return out


#: Read-time derivation per curve metric, so a caller can dispatch on the metric name
#: rather than branching. ``cost_loss`` and ``reliability`` both consume a *different*
#: store than they are named for — cost_loss reads the ROC counts — which is exactly the
#: kind of thing a caller should not have to know.
DERIVERS = {"roc": roc_for, "cost_loss": cost_loss_for, "reliability": reliability_for}

#: Which stored file each curve metric is derived from.
SOURCE_METRIC = {"roc": "roc", "cost_loss": "roc", "reliability": "reliability"}

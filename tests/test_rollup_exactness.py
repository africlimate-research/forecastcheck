# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Roll-up exactness: summing children must reproduce a direct pool, not approximate it.

Runs standalone (``python tests/test_rollup_exactness.py``) or under pytest.

The whole design rests on one claim: because contingency counts are additive over disjoint
sample pools, a region's curves can be recovered by *summing* the counts of the regions
below it, and a season's by summing its months — giving exactly what pooling that
selection directly would have produced. These tests are that claim, checked against a
brute-force direct pool on synthetic data where every sample's region and day is known.

Everything here is integer-exact except one quantity, and the difference is stated
explicitly where it arises: mean forecast probability is a sum of floats, so the two paths
add the same numbers in a different order and agree to floating-point rounding rather than
bit for bit. Counts, observed frequencies and confidence intervals are exact.

No real data needed — these run anywhere.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import curves
from verification import diagnostics as d

_THRESHOLDS = [20, 40]
_BANDS = ["all", "0-1000"]
_LEAD = 120


def _synthetic(n_days=90, n_cells=40, n_regions=4, seed=7, empty_region=False):
    """A Samples whose every entry has a known region and day.

    Deliberately lumpy: probabilities are drawn per day so events cluster in time (which
    is what the day-block bootstrap exists to respect), and region sizes are uneven so a
    roll-up that quietly averaged instead of summing would show up.
    """
    rng = np.random.default_rng(seed)
    n = n_days * n_cells
    day = np.repeat(np.arange(n_days), n_cells)
    # uneven region sizes; optionally leave the last region with no samples at all
    usable = n_regions - 1 if empty_region else n_regions
    region = np.array([i % usable for i in range(n_cells)], dtype=np.int64)
    region = np.tile(region, n_days)

    daily = rng.beta(1.2, 4.0, size=n_days)          # a wet/dry signal per day
    pairs, blocks, regions = {}, {}, {}
    for thr in _THRESHOLDS:
        scale = 1.0 if thr == 20 else 0.4            # rarer events at the high threshold
        for band in _BANDS:
            take = np.ones(n, dtype=bool) if band == "all" else (region % 2 == 0)
            p = np.clip(daily[day] * scale + rng.normal(0, 0.05, size=n), 0.0, 1.0)
            o = (rng.random(n) < p).astype(np.int8)
            pairs[(thr, band)] = (p[take], o[take])
            blocks[(thr, band)] = day[take]
            regions[(thr, band)] = region[take]

    return d.Samples(
        lead_hours=_LEAD, pairs=pairs, blocks=blocks, regions=regions,
        region_ids=tuple(f"r{i}" for i in range(n_regions)),
        day_months=((np.arange(n_days) // 8) % 12 + 1).astype(np.int8),
        thresholds=list(_THRESHOLDS), bands=list(_BANDS))


def _subset(samples, keep_regions=None, keep_months=None):
    """A Samples holding only the entries in the given regions / months.

    The brute-force reference: what the pipeline would have produced had it been asked to
    pool exactly this selection, the expensive way.
    """
    pairs, blocks, regions = {}, {}, {}
    month_of_day = np.asarray(samples.day_months)
    for key, (p, o) in samples.pairs.items():
        reg = np.asarray(samples.regions[key])
        day = np.asarray(samples.blocks[key])
        take = np.ones(p.shape, dtype=bool)
        if keep_regions is not None:
            idx = [samples.region_ids.index(r) for r in keep_regions]
            take &= np.isin(reg, idx)
        if keep_months is not None:
            take &= np.isin(month_of_day[day], list(keep_months))
        pairs[key] = (p[take], o[take])
        blocks[key] = day[take]
        regions[key] = reg[take]
    return d.Samples(lead_hours=samples.lead_hours, pairs=pairs, blocks=blocks,
                     regions=regions, region_ids=samples.region_ids,
                     day_months=samples.day_months,
                     thresholds=samples.thresholds, bands=samples.bands)


def test_e1_roc_counts_sum_to_the_direct_pool():
    """National ROC summed from every region == ROC pooled over everything, bit for bit.

    Integer addition on a shared probability axis, so 'bit for bit' is the right standard
    here — anything less would mean the axes or the partition disagree.
    """
    samples = _synthetic()
    store = d.compute_roc_by_region(samples)
    rolled = curves.roc_for(store)
    direct = d.compute_roc(samples)

    for field in d.ROC_COUNT_FIELDS:
        assert np.array_equal(rolled[field].values, direct[field].values), field
        assert rolled[field].dtype == direct[field].dtype, field
    assert list(rolled["prob"].values) == list(direct["prob"].values)
    assert list(rolled["threshold"].values) == list(direct["threshold"].values)
    assert list(rolled["altitude"].values) == list(direct["altitude"].values)


def test_e2_reliability_counts_sum_to_the_direct_pool():
    """Per-day reliability counts summed over regions and days == the direct pool.

    Checked at the count level, before any rate is formed, because that is the quantity
    the additivity claim is actually about.
    """
    samples = _synthetic()
    store = d.compute_reliability_counts(samples)
    direct = d.compute_reliability(samples)

    total = store["bin_total"].sum(dim=("region", "day")).transpose(
        "threshold", "altitude", "bin").values
    assert np.array_equal(total, direct["bin_count"].values)

    # every sample is accounted for exactly once, in exactly one (region, day, bin)
    for (thr, band), (p, _) in samples.pairs.items():
        i = samples.thresholds.index(thr)
        j = samples.bands.index(band)
        assert int(total[i, j].sum()) == p.size, (thr, band)


def test_e3_cost_loss_derived_from_summed_counts_is_exact():
    """Cost-loss from the summed ROC == cost-loss from the direct pool, bit for bit.

    It is never stored, so this is what justifies deriving it: the same function applied
    to identical counts gives identical floats.
    """
    samples = _synthetic()
    store = d.compute_roc_by_region(samples)
    rolled = curves.cost_loss_for(store)
    direct = d.compute_cost_loss(d.compute_roc(samples))

    for name in ("value", "base_rate", "max_value"):
        assert np.array_equal(rolled[name].values, direct[name].values,
                              equal_nan=True), name


def test_e4_a_season_is_the_sum_of_its_months():
    """Seasons are month-sums: yearly == all months, and a season == its months alone.

    This is what makes a season configuration rather than a pipeline run — including a
    cross-year one like NDJF, which is still just a set of calendar months.
    """
    samples = _synthetic()
    store = d.compute_roc_by_region(samples)

    everything = curves.roc_for(store)
    all_months = curves.roc_for(store, months=list(range(1, 13)))
    for field in d.ROC_COUNT_FIELDS:
        assert np.array_equal(everything[field].values, all_months[field].values), field

    # a cross-year season, against a direct pool of exactly those months
    ndjf = [11, 12, 1, 2]
    rolled = curves.roc_for(store, months=ndjf)
    direct = d.compute_roc(_subset(samples, keep_months=ndjf))
    for field in d.ROC_COUNT_FIELDS:
        assert np.array_equal(rolled[field].values, direct[field].values), field

    # and a season is the sum of its individual months
    summed = sum(curves.roc_for(store, months=[m])["hits"].values for m in ndjf)
    assert np.array_equal(summed, rolled["hits"].values)


def test_e5_confidence_intervals_from_stored_counts_are_exact():
    """The CI derived from per-day counts equals the CI computed from raw samples.

    The bootstrap resamples whole days and days are shared across regions, so replaying it
    over stored per-day counts reproduces the interval exactly — this is the reason the
    store keeps day resolution rather than month.

    Counts, observed frequencies and both interval bounds are bit-identical. Mean forecast
    probability is the one exception: it is a sum of floats, and the two paths add the same
    values in a different order, so they agree to rounding (asserted below at 1e-12) rather
    than bit for bit.
    """
    samples = _synthetic()
    store = d.compute_reliability_counts(samples)
    rolled = curves.reliability_for(store)
    direct = d.compute_reliability(samples)

    assert np.array_equal(rolled["bin_count"].values, direct["bin_count"].values)
    assert np.array_equal(rolled["observed_frequency"].values,
                          direct["observed_frequency"].values, equal_nan=True)
    for bound in ("obs_freq_lower", "obs_freq_upper"):
        assert np.array_equal(rolled[bound].values, direct[bound].values,
                              equal_nan=True), bound

    a = rolled["forecast_probability"].values
    b = direct["forecast_probability"].values
    assert np.array_equal(np.isnan(a), np.isnan(b))
    finite = ~np.isnan(a)
    assert np.allclose(a[finite], b[finite], rtol=1e-12, atol=0), \
        float(np.nanmax(np.abs(a - b)))


def test_e6_a_parent_region_is_the_sum_of_its_children():
    """Selecting a subset of regions == pooling only those regions' samples.

    The level roll-up in miniature: a parent region is a set of child regions, so this is the
    same arithmetic that makes admin level 2 free once level 3 is stored.
    """
    samples = _synthetic()
    roc_store = d.compute_reliability_counts(samples), d.compute_roc_by_region(samples)
    rel_store, roc = roc_store

    for group in (["r0"], ["r0", "r2"], ["r1", "r2", "r3"]):
        rolled = curves.roc_for(roc, regions=group)
        direct = d.compute_roc(_subset(samples, keep_regions=group))
        for field in d.ROC_COUNT_FIELDS:
            assert np.array_equal(rolled[field].values, direct[field].values), (group, field)

        # Bands are derived from the pool being scored, so a region pooled directly
        # would derive its OWN bands while the store applies the whole pool's. Hold the
        # bands fixed to compare like with like — that is exactly the condition the
        # roll-up relies on, and the thing worth asserting.
        whole_edges = {k: d.derive_reliability_bands(v[0]) for k, v in samples.pairs.items()}
        rolled_rel = curves.reliability_for(rel_store, regions=group)
        direct_rel = d.compute_reliability(_subset(samples, keep_regions=group),
                                           bin_edges=whole_edges)
        assert np.array_equal(rolled_rel["bin_count"].values,
                              direct_rel["bin_count"].values), group
        assert np.array_equal(rolled_rel["obs_freq_lower"].values,
                              direct_rel["obs_freq_lower"].values, equal_nan=True), group

    # the children partition the whole: every region together is the national pool
    everything = curves.roc_for(roc, regions=list(samples.region_ids))
    assert np.array_equal(everything["hits"].values,
                          curves.roc_for(roc)["hits"].values)


def test_e7_an_empty_region_contributes_nothing():
    """A region with no samples is zeros, and leaves its parent's totals untouched.

    Many sub-regions hold no gauge, so this is a common case, not an edge case — and an
    empty pool must not turn into a NaN that poisons the sum it belongs to.
    """
    samples = _synthetic(empty_region=True)
    roc = d.compute_roc_by_region(samples)
    empty = "r3"

    counts = roc["hits"].sel(region=empty).values
    assert counts.sum() == 0, counts.sum()

    with_empty = curves.roc_for(roc, regions=list(samples.region_ids))
    without = curves.roc_for(roc, regions=[r for r in samples.region_ids if r != empty])
    for field in d.ROC_COUNT_FIELDS:
        assert np.array_equal(with_empty[field].values, without[field].values), field

    # and the curve code tolerates being asked for the empty region itself
    only_empty = curves.cost_loss_for(roc, regions=[empty])
    assert np.array_equal(only_empty["value"].values,
                          np.zeros_like(only_empty["value"].values))
    rel = curves.reliability_for(d.compute_reliability_counts(samples), regions=[empty])
    assert int(rel["bin_count"].values.sum()) == 0
    assert np.isnan(rel["observed_frequency"].values).all()


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)

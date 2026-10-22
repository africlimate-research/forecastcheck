# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Reliability binning: summable across regions, and robust to the point mass.

Runs standalone (``python tests/test_reliability_bins.py``) or under pytest.

Bands used to be plain quantile edges of the pool being scored. That had two problems, and
only one of them was about roll-up.

The roll-up problem: quantile edges differ per pool, so bin 3 of Podor covered a different
probability interval than bin 3 of Bakel and the two could not be added to get their
parent. Deriving one set of edges per (threshold, altitude) — from the whole pool, shared
by every region and month — fixes that, because those are the only axes ever summed over.

The worse problem: quantile binning **cannot** split a point mass. IDR returns a
probability of essentially zero whenever no training sample exceeded the threshold, so at
50-60 mm/day ~90% of cell-days sit on one value, all nine inner quantiles land on it, and
every sample falls into a single bin. Real curve files showed exactly that —
one populated bin out of ten at 50 and 60 mm/day, a diagram consisting of a single dot, at
the thresholds a forecaster issues warnings on. Giving the point mass its own band and
spending the remaining nine on quantiles of the rest is what turns that back into ten
usable points.

So the tests here cover both: that one set of bands is shared by everything that gets
summed, that the bands tile [0, 1] with no degenerate widths on any distribution shape,
and — the part that matters scientifically — that a diagram drawn on them still tracks the
diagonal when a forecast is calibrated and departs from it detectably when it is not.

No real data needed — these run anywhere.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import curves
from verification import diagnostics as d

_THR = 20
_BAND = "all"


def _samples(n_days=120, n_cells=60, calibration=1.0, seed=5, n_regions=3):
    """Samples whose events are drawn at ``calibration`` times the forecast probability.

    ``calibration=1.0`` is a perfectly reliable forecast; 0.5 systematically over-forecasts
    (events happen half as often as claimed), which is what a reliability diagram exists to
    reveal.
    """
    rng = np.random.default_rng(seed)
    n = n_days * n_cells
    day = np.repeat(np.arange(n_days), n_cells)
    region = np.tile(np.arange(n_cells) % n_regions, n_days).astype(np.int64)
    p = rng.beta(0.8, 3.0, size=n)
    o = (rng.random(n) < np.clip(p * calibration, 0, 1)).astype(np.int8)
    return d.Samples(
        lead_hours=24, pairs={(_THR, _BAND): (p, o)},
        blocks={(_THR, _BAND): day}, regions={(_THR, _BAND): region},
        region_ids=tuple(f"r{i}" for i in range(n_regions)),
        day_months=((np.arange(n_days) // 10) % 12 + 1).astype(np.int8),
        thresholds=[_THR], bands=[_BAND], target="station")


def test_b1_one_set_of_bands_per_pool_shared_by_every_region_and_month():
    """B1: the invariant additivity actually needs — and no more than that.

    Bands are derived from the pool being scored, not looked up from a table, so a new
    country gets correct bands from its own data on its first run with nothing to
    configure. What additivity requires is narrower than "globally fixed": the edges must
    be identical across everything that gets **summed**, which is regions and months.
    ``threshold`` and ``altitude`` are never summed over, so each may differ — and does,
    because each is a genuinely different distribution.
    """
    samples = _samples()
    store = d.compute_reliability_counts(samples)

    # one set of edges per (threshold, altitude), carried in the file itself
    assert store["bin_lower"].dims == ("threshold", "altitude", "bin")
    assert store["bin_upper"].dims == ("threshold", "altitude", "bin")

    # every region and every month is scored on those same edges, which is what lets
    # their counts be added — the file has no per-region or per-month edge axis at all
    assert "region" not in store["bin_lower"].dims
    assert "day" not in store["bin_lower"].dims

    # deriving twice from the same pool gives the same answer (no hidden randomness)
    again = d.compute_reliability_counts(samples)
    assert np.array_equal(store["bin_lower"].values, again["bin_lower"].values)

    # and the national path derives the SAME edges from the same pool, so the
    # whole-country file and the by-region store cannot disagree about the x-axis
    direct = d.compute_reliability(samples)
    assert np.array_equal(direct["bin_lower"].values, store["bin_lower"].values)
    assert np.array_equal(direct["bin_upper"].values, store["bin_upper"].values)


def test_b2_bands_tile_the_unit_interval_with_positive_width():
    """B2: every probability lands in exactly one band, and no band is degenerate.

    Bands are half-open, so a zero-width one could never be selected and its samples would
    pile silently into a neighbour. Ties are expected — quantiles repeat at the rare
    thresholds — and are resolved by dropping the duplicates and re-splitting the widest
    log gap, rather than by nudging each repeat up by an epsilon that leaves bands too
    narrow to hold anything.
    """
    rng = np.random.default_rng(3)
    pools = {
        "continuous": rng.beta(1.0, 3.0, 20000),
        "point mass": np.concatenate([np.zeros(18000), rng.beta(1.0, 3.0, 2000)]),
        "extreme": np.concatenate([np.zeros(19900), rng.beta(1.0, 3.0, 100)]),
        "all zero": np.zeros(5000),
    }
    for label, p in pools.items():
        edges = np.asarray(d.derive_reliability_bands(p, 10))
        assert edges[0] == 0.0 and edges[-1] == 1.0, (label, edges)
        assert (np.diff(edges) > 0).all(), (label, np.diff(edges).min())
        assert len(edges) == 11, (label, len(edges))

        idx = d.assign_bins(p, tuple(edges))
        assert idx.min() >= 0 and idx.max() <= 9, label
        assert np.bincount(idx, minlength=10).sum() == p.size, label

    # boundaries behave: 0 in the first band, 1.0 in the last, monotone in between
    edges = d.derive_reliability_bands(pools["continuous"], 10)
    probe = np.array([0.0, 1e-9, 1e-4, 0.5, 0.999, 1.0])
    idx = d.assign_bins(probe, edges)
    assert idx[0] == 0 and idx[-1] == 9
    assert (np.diff(idx) >= 0).all()


def test_b3_point_estimates_are_recovered_from_children_by_summing():
    """Observed frequency and mean forecast probability survive the roll-up.

    Both are ratios of stored sums, so a parent is recovered from its children exactly —
    this is the property that fixing the bands bought.
    """
    samples = _samples()
    store = d.compute_reliability_counts(samples)

    rolled = curves.reliability_for(store)
    direct = d.compute_reliability(samples)

    assert np.array_equal(rolled["bin_count"].values, direct["bin_count"].values)
    assert np.array_equal(rolled["observed_frequency"].values,
                          direct["observed_frequency"].values, equal_nan=True)

    # summing a subset of regions equals pooling that subset: children -> parent
    per_region = [curves.reliability_for(store, regions=[r])["bin_count"].values
                  for r in samples.region_ids]
    assert np.array_equal(sum(per_region), rolled["bin_count"].values)


#: A reliability point can only be judged where the band expects enough events for the
#: interval to have power. Measured on this fixture: at fewer than ~2 expected events the
#: bootstrap interval collapses to [0, 0] (observing zero is ordinary, so every replicate
#: agrees) and would "detect" miscalibration in a perfectly calibrated forecast; between 2
#: and 16 a genuine 2x over-forecast is flagged in only one band of four; from ~30 upwards
#: detection is reliable. 30 is therefore where a diagram starts carrying evidence.
_INFORMATIVE_EVENTS = 30


def _curve(ds):
    """The (fp, of, lo, hi, count) arrays of the one threshold/band this fixture scores."""
    g = lambda name: ds[name].sel(threshold=_THR, altitude=_BAND).values  # noqa: E731
    return (g("forecast_probability"), g("observed_frequency"),
            g("obs_freq_lower"), g("obs_freq_upper"), g("bin_count"))


def test_b4_binning_still_judges_calibration_correctly():
    """The scientific gate: the diagram must track the diagonal only when it should.

    Bands that sum beautifully but can no longer tell a reliable forecast from an
    over-forecasting one would be a regression however neat the arithmetic. So both
    directions are checked, over the bands with enough expected events to carry evidence
    (see :data:`_INFORMATIVE_EVENTS`):

    * a perfectly calibrated forecast sits on the diagonal, inside its own interval;
    * one that over-forecasts two-fold sits below it, outside the interval — i.e. the
      diagram would actually show a forecaster the problem.

    Low-occupancy bands are excluded deliberately, not for convenience: their intervals
    are degenerate and would report miscalibration in the calibrated case too.
    """
    good = d.compute_reliability(_samples(calibration=1.0), n_bootstraps=200)
    fp, of, lo, hi, cnt = _curve(good)
    informative = cnt * fp >= _INFORMATIVE_EVENTS
    assert informative.sum() >= 4, f"only {informative.sum()} bands carry evidence"

    inside = (lo[informative] <= fp[informative]) & (fp[informative] <= hi[informative])
    assert inside.all(), \
        "a calibrated forecast must sit on the diagonal: " + str(
            list(zip(fp[informative], lo[informative], hi[informative])))

    bad = d.compute_reliability(_samples(calibration=0.5), n_bootstraps=200)
    fp_b, of_b, _, hi_b, cnt_b = _curve(bad)
    informative_b = cnt_b * fp_b >= _INFORMATIVE_EVENTS
    assert informative_b.sum() >= 4, informative_b.sum()

    # direction first: events happen less often than forecast, in every band that has any
    measurable = cnt_b * fp_b >= 2
    assert (of_b[measurable] < fp_b[measurable]).all(), \
        list(zip(fp_b[measurable], of_b[measurable]))

    # then significance: the departure clears the interval wherever there is power to see it
    detected = fp_b[informative_b] > hi_b[informative_b]
    assert detected.all(), \
        f"miscalibration flagged in only {detected.sum()} of {informative_b.sum()} bands"

    # the two runs differ because of calibration, not because the bands moved
    assert np.array_equal(good["bin_lower"].values, bad["bin_lower"].values)
    assert abs(float(np.nanmean(of)) - float(np.nanmean(of_b))) > 0.02


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

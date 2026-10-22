# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""The sample stated behind a threshold curve.

A curve looks equally authoritative at every rung, and the base rate is the fact that
says otherwise. These gates protect the two ways the number could quietly become wrong:
by being read off a different pooling than the figure, and by being formatted into
nothing ("0.0%" for 901 real events).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ui.components.event_box import _fmt_fraction, combine, counting_series
from ui.services.threshold_data import EventCount, event_counts

THRESHOLDS = [37, 45]
BANDS = ["all", "0-25"]
PROBS = np.linspace(0, 1, 11)


def _roc_ds(events: int, non_events: int) -> xr.Dataset:
    """A ROC store whose contingency counts describe a known sample.

    hits + misses is constant across probability bins by construction, which is the
    property ``event_counts`` relies on; this builds it that way deliberately.
    """
    shape = (len(THRESHOLDS), len(BANDS), len(PROBS))
    hits = np.zeros(shape)
    misses = np.zeros(shape)
    fa = np.zeros(shape)
    cn = np.zeros(shape)
    for k, p in enumerate(PROBS):
        hits[:, :, k] = round(events * (1 - p))
        misses[:, :, k] = events - round(events * (1 - p))
        fa[:, :, k] = round(non_events * (1 - p) * 0.3)
        cn[:, :, k] = non_events - round(non_events * (1 - p) * 0.3)
    return xr.Dataset(
        {"hits": (("threshold", "altitude", "prob"), hits),
         "misses": (("threshold", "altitude", "prob"), misses),
         "false_alarms": (("threshold", "altitude", "prob"), fa),
         "correct_negatives": (("threshold", "altitude", "prob"), cn)},
        coords={"threshold": THRESHOLDS, "altitude": BANDS, "prob": PROBS},
    )


def _reliability_ds(events: int, non_events: int) -> xr.Dataset:
    """A reliability store carrying the same sample, split across bins."""
    nbin = 5
    total = events + non_events
    counts = np.full(nbin, total / nbin)
    freqs = np.full(nbin, events / total)
    shape = (len(THRESHOLDS), len(BANDS), nbin)
    return xr.Dataset(
        {"bin_count": (("threshold", "altitude", "bin"),
                       np.broadcast_to(counts, shape).copy()),
         "observed_frequency": (("threshold", "altitude", "bin"),
                                np.broadcast_to(freqs, shape).copy())},
        coords={"threshold": THRESHOLDS, "altitude": BANDS, "bin": range(nbin)},
    )


# --------------------------------------------------------------------------- e1
def test_e1_roc_and_reliability_agree_on_the_same_sample():
    """Two file shapes, one sample — they must not disagree.

    The page draws whichever metric is selected, so if these two disagreed the same
    curve would report a different base rate depending on which tab you were on.
    """
    roc = event_counts(_roc_ds(901, 2_277_714), 45)
    rel = event_counts(_reliability_ds(901, 2_277_714), 45)
    assert roc.events == pytest.approx(901)
    assert rel.events == pytest.approx(901, rel=1e-6)
    assert roc.total == pytest.approx(rel.total, rel=1e-9)
    assert roc.fraction == pytest.approx(rel.fraction, rel=1e-6)


# --------------------------------------------------------------------------- e2
def test_e2_counts_follow_the_threshold_and_band_selected():
    """The numbers describe the slice drawn, not the file.

    Built as different samples per rung so a helper that ignored its arguments and
    summed the whole file would be caught.
    """
    ds = _roc_ds(901, 2_277_714).copy(deep=True)
    ds["hits"][0] *= 10          # threshold 37 holds ten times the events
    ds["misses"][0] *= 10
    low = event_counts(ds, 37)
    high = event_counts(ds, 45)
    assert low.events == pytest.approx(10 * high.events)

    # a band the file does not have is a miss, not a traceback on the page
    assert event_counts(ds, 45, band="9999-10000") is None
    assert event_counts(ds, 999) is None


# --------------------------------------------------------------------------- e3
def test_e3_cost_loss_reports_a_rate_and_admits_it_has_no_counts():
    """cost-loss stores only base_rate; inventing a count would be a fabrication.

    The count it could be given (from the ROC file) describes a pooling this dataset
    does not state, so it would be right by luck and wrong silently when it was not.
    """
    ds = xr.Dataset(
        {"base_rate": (("threshold", "altitude"),
                       np.array([[0.4252, 0.4252], [0.000395, 0.000395]]))},
        coords={"threshold": THRESHOLDS, "altitude": BANDS},
    )
    got = event_counts(ds, 45)
    assert got.events is None and got.total is None
    assert got.fraction == pytest.approx(0.000395)


# --------------------------------------------------------------------------- e4
def test_e4_years_pool_by_adding_counts_not_averaging_rates():
    """A sparse year must not swing the base rate as far as a full one.

    Averaging the two fractions below gives 0.2525; the honest pooled rate is 0.0204,
    because the sparse year contributes 100 of the 4,900 evaluated cases, not half.
    """
    full = EventCount(100.0, 4900.0, 100 / 4900)
    sparse = EventCount(50.0, 100.0, 0.5)
    pooled = combine([full, sparse])
    assert pooled.events == 150
    assert pooled.total == 5000
    assert pooled.fraction == pytest.approx(150 / 5000)
    assert pooled.fraction != pytest.approx((full.fraction + sparse.fraction) / 2)


# --------------------------------------------------------------------------- e5
def test_e5_pooling_degrades_gracefully_when_a_slice_has_no_counts():
    """Mixing a cost-loss slice in cannot yield a made-up total."""
    pooled = combine([EventCount(100.0, 1000.0, 0.1), EventCount(None, None, 0.3)])
    assert pooled.events is None and pooled.total is None
    assert pooled.fraction == pytest.approx(0.2)
    assert combine([None, None]) is None
    assert combine([]) is None


# --------------------------------------------------------------------------- e6
def test_e6_a_rare_rate_survives_formatting():
    """``.1%`` prints a rare rung as "0.0%" — which reads as none.

    901 events is a real sample; the format has to carry four orders of magnitude.
    """
    assert _fmt_fraction(0.4252) == "43%"
    assert _fmt_fraction(0.0549) == "5.5%"
    assert _fmt_fraction(0.00395) == "0.40%"
    assert _fmt_fraction(0.000395) == "0.040%"
    assert float(_fmt_fraction(0.000090).rstrip("%")) > 0




# --------------------------------------------------------------------------- e8
def test_e8_real_curve_files_agree_across_all_three_metrics():
    """Against the tree, if it is present: roc, reliability and cost-loss must concur.

    They are three views of one pooled sample, so a discrepancy means one of the three
    readers is looking at the wrong array.
    """
    from verification import config
    base = (config.data_root() / "senegal/diagnostics/graphcast-idr/temperature_max/chirts"
            "/gridded/{m}/2022/24h.nc")
    paths = {m: Path(str(base).format(m=m)) for m in ("roc", "reliability", "cost_loss")}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("Senegal curve tree not present")
    got = {}
    for m, p in paths.items():
        with xr.open_dataset(p) as ds:
            got[m] = event_counts(ds, 45)
    assert got["roc"].events == pytest.approx(got["reliability"].events, rel=1e-6)
    assert got["roc"].fraction == pytest.approx(got["cost_loss"].fraction, rel=1e-6)


# --------------------------------------------------------------------------- e9
def test_e9_the_three_numbers_shown_are_arithmetically_consistent():
    """Events / Evaluated must equal the percentage, for every rung.

    The card prints all three, so a reader will divide them. If the denominator were
    ever a different total than the one behind the fraction — a whole-country count
    under a regional curve, say — the row would read as a typo and be believed anyway.
    """
    ds = _roc_ds(901, 2_277_714)
    c = event_counts(ds, 45)
    assert c.fraction == pytest.approx(c.events / c.total, rel=1e-15)


# --------------------------------------------------------------------------- e10
def test_e10_pooled_years_keep_that_consistency():
    """The same invariant after `combine`, which recomputes rather than averages."""
    pooled = combine([EventCount(901.0, 2_278_615.0, 901 / 2_278_615),
                      EventCount(1_240.0, 2_318_151.0, 1240 / 2_318_151)])
    assert pooled.fraction == pytest.approx(pooled.events / pooled.total, rel=1e-15)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------- e11
def test_e11_cost_loss_tab_counts_from_the_same_slice_roc():
    """All three tabs show the same box: cost-loss borrows the slice's ROC counts.

    A series with no ROC or reliability file keeps its own dataset (rate only), and the
    other tabs are passed through untouched.
    """
    cost = xr.Dataset({"base_rate": (("threshold", "altitude"), np.full((2, 2), 0.1))},
                      coords={"threshold": THRESHOLDS, "altitude": BANDS})
    roc = _roc_ds(901, 2_277_714)
    rel = _reliability_ds(50, 950)
    files = {"roc": [("GraphCast", roc)], "reliability": [("IFS", rel)]}
    series = [("GraphCast", cost), ("IFS", cost), ("AIFS", cost)]

    got = dict(counting_series(series, "cost_loss", lambda m: files.get(m, [])))
    assert got["GraphCast"] is roc                      # ROC first
    assert got["IFS"] is rel                            # then reliability
    assert got["AIFS"] is cost                          # nothing to borrow
    assert event_counts(got["GraphCast"], 40) == event_counts(roc, 40)

    for metric in ("roc", "reliability"):
        assert counting_series(series, metric, lambda m: 1 / 0) is series

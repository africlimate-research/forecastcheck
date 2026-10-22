"""Tests for the equivalence gate (``tools.compare_runs``).

The gate is what decides whether the CountryStore refactor preserved the science,
so it must fail for the right reasons and pass only when the numbers really match.
These tests give it trees it should accept and trees it must reject:

  R1  identical trees are EQUIVALENT
  R2  a single changed value is caught, with the max abs difference reported
  R3  ``--exclude pit`` ignores exactly that variable and nothing else
  R4  a file present on one side only is a failure (a skipped lead must not pass)
  R5  NaN in the same position is equal; NaN moving position is not
  R6  differing global attributes do NOT fail (provenance is meant to differ)

Runs standalone or under pytest.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_runs import compare_datasets, compare_trees, format_report

_REL = "idr/station/precipitation/2022/graphcast"


def _idr_ds(*, crps_bump=0.0, pit_seed=0, nan_at=None, attrs=None):
    """A miniature station IDR output: the variables the calibration actually writes."""
    n_time, n_station = 5, 3
    rng = np.random.default_rng(pit_seed)
    crps = np.arange(n_time * n_station, dtype="float64").reshape(n_time, n_station)
    crps = crps + crps_bump
    if nan_at is not None:
        crps = crps.copy()
        crps[nan_at] = np.nan
    ds = xr.Dataset(
        {
            "crps": (("time", "station"), crps),
            "q50": (("time", "station"), crps * 0.5),
            "pit": (("time", "station"), rng.uniform(size=(n_time, n_station))),
            # int64 counts exist in the real trees (ROC tables); dtype must not break
            "n_used": (("station",), np.array([10, 20, 30], dtype="int64")),
        },
        coords={"time": np.arange(n_time), "station": np.arange(n_station),
                "name": ("station", ["a", "b", "c"])},
        attrs=attrs or {},
    )
    return ds


def _tree(root, ds, *, leads=(24, 72)):
    """Write ``ds`` at the real relative path shape for each lead."""
    d = Path(root) / _REL
    d.mkdir(parents=True, exist_ok=True)
    for lead in leads:
        ds.to_netcdf(d / f"{lead}h.nc")
    return Path(root)


def test_r1_identical_trees_are_equivalent():
    """R1: byte-identical trees compare equal and report every file."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _tree(a, _idr_ds())
        _tree(b, _idr_ds())
        rep = compare_trees(a, b)
        assert rep.compared == 2, rep.compared
        assert rep.identical == 2
        assert rep.ok, format_report(rep)
        assert "EQUIVALENT" in format_report(rep)


def test_r2_a_changed_value_is_caught_with_its_magnitude():
    """R2: a 0.25 bump in crps fails and the reported max abs diff is 0.25."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _tree(a, _idr_ds())
        _tree(b, _idr_ds(crps_bump=0.25))
        rep = compare_trees(a, b)
        assert not rep.ok
        assert len(rep.mismatches) == 2
        assert "crps" in rep.mismatches[0].detail
        worst = rep.worst[rep.mismatches[0].rel]
        assert abs(worst["crps"] - 0.25) < 1e-12
        # q50 is crps*0.5, so it moves too — both must be reported, not just the first
        assert abs(worst["q50"] - 0.125) < 1e-12
        assert "NOT EQUIVALENT" in format_report(rep)


def test_r3_exclude_ignores_only_the_named_variable():
    """R3: pit differing is tolerated under --exclude pit; crps still is not."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _tree(a, _idr_ds(pit_seed=1))
        _tree(b, _idr_ds(pit_seed=2))
        # pit differs by construction (different RNG streams)
        assert not compare_trees(a, b).ok
        assert compare_trees(a, b, exclude=frozenset({"pit"})).ok
    # and excluding pit must NOT mask a real crps regression
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _tree(a, _idr_ds(pit_seed=1))
        _tree(b, _idr_ds(pit_seed=2, crps_bump=1e-9))
        rep = compare_trees(a, b, exclude=frozenset({"pit"}))
        assert not rep.ok
        assert "crps" in rep.mismatches[0].detail


def test_r4_a_missing_or_extra_file_fails():
    """R4: a lead written on one side only is a failure, not a silent pass."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _tree(a, _idr_ds(), leads=(24, 72))
        _tree(b, _idr_ds(), leads=(24,))
        rep = compare_trees(a, b)
        assert not rep.ok
        assert rep.only_baseline == [f"{_REL}/72h.nc"]
        assert rep.identical == 1
        assert "MISSING from candidate" in format_report(rep)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _tree(a, _idr_ds(), leads=(24,))
        _tree(b, _idr_ds(), leads=(24, 72))
        rep = compare_trees(a, b)
        assert not rep.ok and rep.only_candidate == [f"{_REL}/72h.nc"]


def test_r5_nan_position_matters():
    """R5: NaN equals NaN in the same cell; a moved NaN is a difference."""
    same_a = _idr_ds(nan_at=(0, 0))
    same_b = _idr_ds(nan_at=(0, 0))
    assert compare_datasets(same_a, same_b) == []
    moved = _idr_ds(nan_at=(1, 1))
    problems = compare_datasets(same_a, moved)
    assert any("crps" in p for p in problems), problems


def test_r6_global_attrs_are_ignored():
    """R6: provenance attrs differ between runs by design and must not fail the gate."""
    a = _idr_ds(attrs={"train_years": "2016,2017", "code_commit": "abc123"})
    b = _idr_ds(attrs={"train_years": "2016,2017", "code_commit": "def456",
                       "created": "2026-09-07T12:00:00Z"})
    assert compare_datasets(a, b) == []


def test_r7_empty_comparison_is_reported_not_passed():
    """R7: comparing two empty roots compares nothing — the caller must not read that as OK."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        rep = compare_trees(a, b)
        assert rep.compared == 0
        # `ok` is vacuously true, which is exactly why main() treats compared==0 as an
        # error; assert the signal a caller needs is available.
        assert rep.identical == 0 and not rep.mismatches


def test_r8_allow_extra_tolerates_a_superset_candidate():
    """R8: --allow-extra passes a candidate superset but still fails on a MISSING file."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _tree(a, _idr_ds(), leads=(24,))
        _tree(b, _idr_ds(), leads=(24, 72, 168))
        strict = compare_trees(a, b)
        assert not strict.ok and len(strict.only_candidate) == 2
        relaxed = compare_trees(a, b, allow_extra=True)
        assert relaxed.ok, format_report(relaxed)
        assert "(allowed)" in format_report(relaxed)
    # a missing baseline file must still fail even with --allow-extra
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _tree(a, _idr_ds(), leads=(24, 72))
        _tree(b, _idr_ds(), leads=(24,))
        assert not compare_trees(a, b, allow_extra=True).ok


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

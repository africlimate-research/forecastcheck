# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Every curve file must be filed under the variable it was actually computed for.

This is a regression test for a bug that was **silent for as long as it could be**.
``diagnostics.write_dataset`` defaults ``native_var`` to ``DEFAULT_VARIABLE``
("precipitation"), and the threshold-curve writers in ``verification.run`` did not pass
it. That was invisibly correct for years because threshold curves only ever ran for
precipitation — the wrong default happened to be the right answer.

The moment ``temperature_max`` gained exceedance thresholds, every temperature curve was
written to ``.../precipitation/...``: 20 files, under a variable the country may have no
data for at all, with no error and nothing in the output to suggest it. It was caught by
reading a path.

The scalar metrics never had the bug (they always passed ``native_var``), which is
exactly why the gap survived: the two families are written by different code, and only
one of them was ever exercised with a second variable.

Runs standalone (``python tests/test_curve_variable_segment.py``) or under pytest.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verification import diagnostics as d
from verification import run as vrun
from verification.regions import default_region


def _samples(variable, thresholds):
    """A minimal pooled Samples for one variable: two thresholds, one band."""
    rng = np.random.default_rng(0)
    n = 64
    pairs, blocks = {}, {}
    for thr in thresholds:
        p = rng.random(n)
        o = (rng.random(n) < p).astype(float)
        pairs[(thr, "all")] = (p, o)
        blocks[(thr, "all")] = np.arange(n) // 4
    return d.Samples(lead_hours=24, pairs=pairs, blocks=blocks,
                     day_months=np.full(n, 7, dtype=np.int8),
                     thresholds=list(thresholds), bands=["all"],
                     target="gridded", variable=variable)


def _record_writes(monkeypatch=None):
    """Replace write_dataset with a recorder; returns the list it appends kwargs to."""
    seen = []

    def fake(ds, metric, lead_hours, outdir=None, **kw):
        seen.append({"metric": metric, **kw})
        return Path(outdir or ".") / f"{metric}_{lead_hours}h.nc"

    return seen, fake


def test_legacy_curve_writer_files_under_the_verified_variable():
    """_write_legacy_curves must stamp the variable it was given, not the default.

    Asserted on the WRITE CALL rather than on a path string, so it holds regardless of
    how the path scheme is arranged underneath — the point is that the writer is told
    which variable it is writing, not where that lands.
    """
    for variable in ("temperature_max", "precipitation"):
        thresholds = d.thresholds_for(variable)
        samples = _samples(variable, thresholds[:2])
        seen, fake = _record_writes()
        real = d.write_dataset
        d.write_dataset = fake
        try:
            # the whole country + yearly, so the writer takes the no-region,
            # no-season path and needs no boundary assets to be onboarded
            args = SimpleNamespace(
                metric=["roc", "cost_loss", "reliability"],
                region=[default_region()], season=["yearly"], outdir="/tmp/x",
                n_bootstraps=4, n_bins=5, plot=False, per_altitude=False)
            vrun._write_legacy_curves(args, samples, None, None, 24, 1, 2022,
                                      "graphcast-idr", "gridded", variable)
        finally:
            d.write_dataset = real

        assert seen, f"{variable}: no curve was written at all"
        for call in seen:
            assert call.get("native_var") == variable, (
                f"{variable}: {call['metric']} was filed under "
                f"{call.get('native_var')!r} — a curve written under the wrong "
                f"variable is invisible to that variable's menus and pollutes another's")


def test_every_write_in_the_cli_names_its_variable():
    """No write_dataset call in verification.run may rely on the native_var default.

    A source-level check, deliberately. The default exists for tests and cannot simply
    be removed, so nothing at runtime distinguishes "meant precipitation" from "forgot
    to say" — the only place that distinction survives is the call site.
    """
    source = Path(vrun.__file__).read_text()
    # every write_dataset call, flattened onto one line so a wrapped call still matches
    flat = " ".join(source.split())
    calls = []
    i = 0
    while (i := flat.find("write_dataset(", i)) != -1:
        depth, j = 0, i + len("write_dataset")
        for j in range(j, len(flat)):
            if flat[j] == "(":
                depth += 1
            elif flat[j] == ")":
                depth -= 1
                if depth == 0:
                    break
        calls.append(flat[i:j + 1])
        i = j + 1

    assert calls, "no write_dataset calls found — did the CLI move?"
    for call in calls:
        # `**common` counts: the dict it expands is asserted to carry native_var above.
        assert "native_var" in call or "**common" in call, (
            f"write_dataset call omits native_var and would default to "
            f"{d.DEFAULT_VARIABLE!r}: {call[:110]}")


def test_derived_curves_keep_the_stores_threshold_units():
    """curves.roc_for / reliability_for must inherit units, not restate them.

    These build a curve by SUMMING the by-region count store, so the threshold axis is
    whatever was pooled — mm/day for precipitation, degC for daily-max temperature.
    Both hardcoded "mm/day", which put that label on every temperature curve.

    This is the sibling of the native_var bug and hid in the same blind spot: the
    by-region STORE was stamped correctly (it comes from diagnostics.py), so checking
    the store — or re-deriving a curve through freshly fixed code — looked fine while
    the whole-country files on disk were wrong. Only reading a written file shows it.
    """
    import numpy as np
    import xarray as xr

    from verification import curves

    for units in ("degC", "mm/day"):
        thresholds = [37, 38] if units == "degC" else [1, 5]
        n_reg, n_month, n_prob, n_bin = 2, 12, 3, 4
        roc_coords = {"threshold": thresholds, "altitude": ["all"],
                      "region": ["a", "b"], "month": list(range(1, 13))}

        roc_store = xr.Dataset(
            {f: (("threshold", "altitude", "prob", "region", "month"),
                 np.ones((len(thresholds), 1, n_prob, n_reg, n_month), dtype=np.int64))
             for f in d.ROC_COUNT_FIELDS},
            coords={**roc_coords, "prob": np.linspace(0.1, 0.9, n_prob)},
        )
        roc_store["threshold"].attrs["units"] = units
        got = curves.roc_for(roc_store)
        assert got["threshold"].attrs.get("units") == units, (
            f"roc_for dropped {units!r} -> {got['threshold'].attrs.get('units')!r}; "
            f"a wrong units string is invisible to everything downstream")

        # the reliability store is per-DAY (the bootstrap resamples whole days) with
        # bin_total / bin_events / prob_sum, not per-month like the ROC store
        n_day = 8
        rel_store = xr.Dataset(
            {"bin_total": (("threshold", "altitude", "bin", "region", "day"),
                           np.ones((len(thresholds), 1, n_bin, n_reg, n_day),
                                   dtype=np.int64)),
             "bin_events": (("threshold", "altitude", "bin", "region", "day"),
                            np.zeros((len(thresholds), 1, n_bin, n_reg, n_day),
                                     dtype=np.int64)),
             "prob_sum": (("threshold", "altitude", "bin", "region", "day"),
                          np.full((len(thresholds), 1, n_bin, n_reg, n_day), 0.5)),
             "bin_lower": (("threshold", "altitude", "bin"),
                           np.tile(np.linspace(0, 0.75, n_bin), (len(thresholds), 1, 1))),
             "bin_upper": (("threshold", "altitude", "bin"),
                           np.tile(np.linspace(0.25, 1.0, n_bin), (len(thresholds), 1, 1)))},
            coords={"threshold": thresholds, "altitude": ["all"],
                    "region": ["a", "b"], "day": np.arange(n_day),
                    "bin": np.arange(n_bin)},
        )
        rel_store["threshold"].attrs["units"] = units
        got = curves.reliability_for(rel_store, n_bootstraps=2)
        assert got["threshold"].attrs.get("units") == units, (
            f"reliability_for dropped {units!r} -> "
            f"{got['threshold'].attrs.get('units')!r}")


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()

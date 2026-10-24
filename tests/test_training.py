# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Training years: a missing year is left out, never the whole lead.

What these gates protect: one training rule (GraphCast 2016–2021, IFS 2018–2021) works
for every dataset, including one that starts later (IFS in 2018, a gauge network later
still). Before this, one missing year made calibration skip the whole lead and write
nothing, while still exiting 0.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.training import describe, stack_training_years  # noqa: E402


def _year(y, value):
    """A tiny daily series for one year: three days, one cell."""
    t = pd.date_range(f"{y}-01-01", periods=3, freq="D")
    return xr.DataArray(np.full(3, float(value)), dims="time", coords={"time": t})


def _loader(have):
    """A loader that has data only for the years in ``have``."""
    def load(y):
        if y not in have:
            raise FileNotFoundError(f"no file for {y}")
        return _year(y, y)
    return load


def test_t1_a_missing_year_is_dropped_and_reported():
    """IFS has no 2016-2017: train on the rest, and say which years were used."""
    x, o, used = stack_training_years(
        range(2016, 2022), _loader({2018, 2019, 2020, 2021}), _loader(set(range(2016, 2022))))
    assert used == [2018, 2019, 2020, 2021]
    assert x.sizes["time"] == o.sizes["time"] == 12
    assert describe(range(2016, 2022), used) == \
        "training on 2018, 2019, 2020, 2021 (no data for 2016, 2017)"


def test_t2_a_year_needs_both_forecast_and_observations():
    """A year with forecasts but no observations (or the reverse) is not used."""
    _, _, used = stack_training_years(
        [2018, 2019, 2020], _loader({2018, 2019, 2020}), _loader({2018, 2020}))
    assert used == [2018, 2020]


def test_t3_all_years_present_is_unchanged():
    """With every year there, the result is the plain stack, and nothing is reported."""
    years = [2018, 2019]
    x, _, used = stack_training_years(years, _loader(set(years)), _loader(set(years)))
    expected = xr.concat([_year(y, y) for y in years], dim="time")
    assert used == years and describe(years, used) is None
    xr.testing.assert_identical(x, expected)


def test_t4_no_year_left_skips_the_lead():
    """Nothing to train on raises FileNotFoundError, which the caller turns into a SKIP."""
    with pytest.raises(FileNotFoundError, match="no training year"):
        stack_training_years([2010, 2011], _loader(set()), _loader(set()))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    print(f"Running {len(fns)} tests in {__file__}")
    sys.exit(pytest.main([__file__, "-q"]))

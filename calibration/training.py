# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Which training years a calibration actually uses.

A training window is a rule — GraphCast 2016–2021, IFS 2018–2021 — and a dataset need not
cover all of it: IFS starts in 2018, a national gauge network may start later still. A
year whose forecasts or observations are missing is left out, rather than failing the
whole lead; the years actually used come back to the caller, which prints them and
records them in the output's provenance. Shared by the gridded and the gauge calibration
(calibration.run, calibration.run_station) so the rule lives in one place.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import xarray as xr


def stack_training_years(
    years: Iterable[int],
    load_forecast: Callable[[int], xr.DataArray],
    load_obs: Callable[[int], xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray, list[int]]:
    """The training forecasts and observations, each stacked along time, and the years used.

    A year is used only when BOTH of its loaders succeed; a ``FileNotFoundError`` from
    either (no file, or an empty one such as a lead past the model's horizon) drops that
    year. With every year present the stacking is exactly the former all-or-nothing one,
    so results do not change. Raises ``FileNotFoundError`` naming the years tried when
    none is left: the caller then skips the lead, as for any missing file.
    """
    years = list(years)
    forecasts, observations, used = [], [], []
    for year in years:
        try:
            forecast, obs = load_forecast(year), load_obs(year)
        except FileNotFoundError:
            continue
        forecasts.append(forecast)
        observations.append(obs)
        used.append(year)
    if not used:
        raise FileNotFoundError(
            f"no training year has both forecasts and observations: {years}")
    return _stack(forecasts), _stack(observations), used


def _stack(arrays: list[xr.DataArray]) -> xr.DataArray:
    """Concatenate per-year arrays along time (sorted, de-duplicated)."""
    return xr.concat(arrays, dim="time").sortby("time").drop_duplicates("time")


def describe(requested: Iterable[int], used: list[int]) -> str | None:
    """A one-line note when some requested years had no data, else None."""
    missing = [y for y in requested if y not in used]
    if not missing:
        return None
    return (f"training on {', '.join(map(str, used))} "
            f"(no data for {', '.join(map(str, missing))})")

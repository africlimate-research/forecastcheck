# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""The climatology result path.

Model result paths are no longer composed here — discovery + the engine's
``ui.services.engine`` own the result-path scheme. Only the (distinct,
``{metric}.nc``-shaped) climatology path remains, used by the climatology
loader until the climatology toggle is reworked (issue #44).

This module used to export a ``DATA_DIR`` constant read from
``VERIFICATION_DATA_DIR`` at IMPORT time, which is why ``ui/app.py`` has to seed the
environment before importing anything. It is gone: every root is now derived by
:class:`verification.store.CountryStore` from the country, so there is no hand-written
results root left to disagree with the pipeline, and no second import-time trap.
"""
from pathlib import Path


def _normalise(**kwargs) -> dict:
    return {k: str(v).lower().strip() for k, v in kwargs.items()}


def get_climatology_data_path(
    forecast_variable: str,
    truth_source: str,
    metric: str,
    year: str,
    *,
    root: Path,
) -> Path:
    """Returns the climatology NetCDF path under ``root`` (the results root).

    ``root`` is the UI's results root (``ui_results_root()``) so the
    ``africa`` segment lives in exactly one place. Layout:
      {root}/climatology/{variable}/{truth}/{metric}/{year}/annual_monthly_averages/{metric}.nc
    """
    n = _normalise(
        forecast_variable=forecast_variable,
        truth_source=truth_source,
        metric=metric,
        year=year,
    )
    return (
        Path(root) / "climatology"
        / n["forecast_variable"] / n["truth_source"]
        / n["metric"] / n["year"] / "annual_monthly_averages"
        / f"{n['metric']}.nc"
    )

# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Climatology baseline reader for the dashboard.

Climatology results have a distinct on-disk shape ({metric}.nc, not
{lead}h.nc), so they're read here rather than through the engine's
open_summary. ``climatology_available`` lets the UI offer the
"vs Climatology" toggle only when a baseline actually exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from ui.backend.config.data_path import get_climatology_data_path
from ui.backend.nc_io import read_dataset

_FORECAST_VARIABLE = "total_precipitation_rate"


def _clim_metric(metric: str) -> str:
    """The metric whose climatology file backs ``metric``.

    MAE has no dedicated climatology file — it reuses the CRPS climatology.
    """
    return "crps" if metric in ("mae", "crps") else metric


def _clim_path(truth: str, year: str, metric: str, root: Union[Path, str]) -> Path:
    return get_climatology_data_path(
        forecast_variable=_FORECAST_VARIABLE, truth_source=truth,
        metric=_clim_metric(metric), year=year, root=Path(root),
    )


def climatology_available(truth: str, year: str, metric: str, *,
                          root: Union[Path, str]) -> bool:
    """True iff a climatology baseline exists for this (truth, year, metric)."""
    return _clim_path(truth, year, metric, root).exists()


@dataclass
class ClimData:
    """Precomputed climatology statistics for a single configuration."""
    annual_average:          float        # scalar mean — baseline for time-series
    annual_spatial_average:  xr.DataArray # (lat, lon)  — spatial difference maps
    monthly_average:         np.ndarray   # (12,)       — baseline for monthly chart
    monthly_spatial_average: xr.DataArray # (month, lat, lon) — monthly spatial
    # Provenance — the path this baseline was read from. Same contract as
    # ``LeadTimeData.source``: it lets the ``ui.utils.geo`` caches key on the
    # data's identity rather than on this object's memory address, which is
    # recycled as soon as a render drops it.
    source:                  str | None   = None


class ClimatologyLoader:
    """Loads climatology benchmark statistics from local disk.

    Returns None if the file for the requested (truth, year, metric) is not
    present — callers gate on :func:`climatology_available` first.
    """

    def load(self, truth: str, year: str, metric: str, *,
             root: Union[Path, str]) -> ClimData | None:
        path = _clim_path(truth, year, metric, root)
        if not path.exists():
            return None
        try:
            print(f"[ClimatologyLoader] reading {path}")
            ds = read_dataset(path, engine="h5netcdf")

            ann_spatial = ds["annual_spatial_average"].squeeze()
            for dim in ("year", "lead_time"):
                if dim in ann_spatial.dims and ann_spatial.sizes[dim] == 1:
                    ann_spatial = ann_spatial.squeeze(dim)

            mon_spatial = ds["monthly_spatial_average"]
            for dim in ("year", "lead_time"):
                if dim in mon_spatial.dims and mon_spatial.sizes[dim] == 1:
                    mon_spatial = mon_spatial.squeeze(dim)

            return ClimData(
                annual_average          = float(ds["annual_average"].values.flat[0]),
                annual_spatial_average  = ann_spatial,
                monthly_average         = np.atleast_1d(ds["monthly_average"].values.squeeze()),
                monthly_spatial_average = mon_spatial,
                source                  = str(path),
            )
        except Exception as exc:
            print(f"[ClimatologyLoader] failed: {exc}")
            return None

"""IDR calibration — produces the calibrated p_exceed / crps / quantile forecasts
that the verification engine (verification.run) consumes.

The engine (`idr.py`) is a verbatim lift of the AfriClimate IDR implementation
(Isotonic Distributional Regression); `run.py` is the thin driver that feeds it
the locally harmonised forecasts + gridded obs and writes the per-lead outputs.
"""
from .idr import IDRDistribution, IDRModel, idr_on_grid

__all__ = ["idr_on_grid", "IDRModel", "IDRDistribution"]

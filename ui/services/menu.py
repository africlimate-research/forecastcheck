"""MenuIndex — the discovery-backed faceted option source for the sidebar.

The dashboard's filters used to read a hardcoded list of models / variables /
truths / metrics that drifted from what the engine actually produced. This
module replaces that with a faceted view over the *discovered* summaries
(:class:`ui.services.engine.SummarySet`): the menu can only ever
offer combinations that exist on disk.

MenuIndex deals in engine tokens (forecast / target / metric / year) plus a
derived variable *kind* (``precip`` / ``Tmax`` / ``Tmin``). Pretty labels,
units, and colors stay in the frontend, keyed off these tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from ui.services.engine import kind_of


# Deterministic point metrics (mirrors the verification CLI's set). Part of the
# metric-source policy in :meth:`MenuIndex.resolve_storage`: calibrated (-idr)
# forecasts are never scored by these.
_DETERMINISTIC_METRICS = {"mae", "bias", "pearsonr", "spearmanr"}


@dataclass(frozen=True)
class MenuIndex:
    """Faceted option lists + selection resolver over discovered summaries."""

    table: pd.DataFrame

    @classmethod
    def from_index(cls, index: pd.DataFrame) -> "MenuIndex":
        """Build from a SummarySet-style index, adding a ``variable`` column.

        ``variable`` is the display-kind of ``native_var`` (precip / Tmax /
        Tmin / unknown) — the axis the sidebar groups by.
        """
        df = index.copy()
        if df.empty:
            df = pd.DataFrame(columns=["forecast", "native_var", "target",
                                       "metric", "year", "leads", "target_kind",
                                       "variable"])
        else:
            df["variable"] = df["native_var"].map(kind_of)
        return cls(table=df)

    @classmethod
    def from_discovery(cls, root: Union[Path, str, None] = None) -> "MenuIndex":
        """Discover summaries under ``root`` and build a MenuIndex."""
        from ui.services.engine import SummarySet
        return cls.from_index(SummarySet.discover(root).index)

    def values(self, field: str, **filters: Optional[str]) -> list:
        """Sorted unique values of ``field`` among rows matching ``filters``.

        ``filters`` are ``column=value`` constraints; a ``None`` value is
        ignored (no constraint). This powers cascading dropdowns: pass the
        already-chosen facets to narrow the next field's options.
        """
        rows = self._matching(filters)
        return sorted(rows[field].dropna().unique().tolist())

    def resolve(
        self, *, forecast: str, target: str, metric: str, year: str,
        variable: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve a full selection to the tokens the loader needs.

        Returns ``{native_var, leads, target_kind}`` for the matching run,
        or ``None`` when the selection has no run on disk. ``native_var`` is
        what the loader reads under (replacing the old hardcoded
        forecast→variable map); ``target_kind`` drives the spatial renderer.

        Assumes ``(forecast, target, metric, year[, variable])`` identifies
        at most one ``native_var``; if a forecast ever emits two variables of
        the same kind for one target, the first match wins (pass ``variable``
        to disambiguate).
        """
        rows = self._matching(dict(forecast=forecast, target=target,
                                   metric=metric, year=year, variable=variable))
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {
            "native_var":  row["native_var"],
            "leads":       list(row["leads"]),
            "target_kind": row["target_kind"],
        }

    def resolve_storage(
        self, *, forecast: str, target: str, metric: str, year: str,
        variable: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve a selection to loadable tokens, applying the metric-source policy.

        Like :meth:`resolve`, plus a ``storage_metric`` key naming the metric
        segment actually read on disk. The two policy rules live here — and
        only here, so the menu, loader, and caches can never disagree:

        - A **calibrated (-idr) forecast is never scored by a deterministic
          point metric** (mae / bias / correlation): it is judged by CRPS and
          the threshold curves. Stale q50 deterministic files left on disk by
          older pipeline versions are ignored, so the MAE facet shows raw
          models only.
        - A **raw forecast selected under ``crps`` falls back to its ``mae``
          files**: a raw forecast is deterministic, and the CRPS of a
          deterministic forecast equals its absolute error — the mae summary
          *is* the raw model's CRPS. This is what lets raw and IDR variants
          sit side by side under the CRPS metric.
        """
        if forecast.endswith("-idr") and metric in _DETERMINISTIC_METRICS:
            return None
        r = self.resolve(forecast=forecast, target=target, metric=metric,
                         year=year, variable=variable)
        if r is not None:
            return {**r, "storage_metric": metric}
        if metric == "crps" and not forecast.endswith("-idr"):
            r = self.resolve(forecast=forecast, target=target, metric="mae",
                             year=year, variable=variable)
            if r is not None:
                return {**r, "storage_metric": "mae"}
        return None

    def _matching(self, filters: dict) -> pd.DataFrame:
        rows = self.table
        for col, val in filters.items():
            if val is not None:
                rows = rows[rows[col] == val]
        return rows

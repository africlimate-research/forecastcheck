# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Stamp how a NetCDF output was produced onto the file itself.

Until now the only record of a run's settings was the path it was written to, which
carries the facets that *select* a file (domain, model, variable, target, metric, year,
region, season, lead) and nothing about how it was computed. The gap that mattered most:
**the calibration's training years were unrecoverable from its output** — the one input
that most changes the numbers, invisible in the artefact. Threshold and altitude subsets,
the wet-day gate, and the code version were equally absent.

The stamp is deliberately *global attributes only*, for two reasons:

* Discovery cannot see it. ``SummarySet.discover`` infers ``target_kind`` from dims and
  ``discover_curves`` parses path segments; neither reads attributes, so adding them
  cannot change which files the dashboard offers. A new path *segment* would.
* The equivalence gate ignores it. ``tools.compare_runs`` compares data variables and
  coordinates only, because provenance is *meant* to differ between two runs of the
  same computation — a timestamp that changed would otherwise fail every comparison.

Values are coerced to NetCDF-safe scalars (strings and numbers); sequences become
comma-separated strings, so ``thresholds_mm_day`` reads back as ``"1,5,10,…"``.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Prefix for every attribute this module writes, so provenance is greppable and can
#: be stripped without guessing which keys were ours.
PREFIX = "provenance_"

_commit_cache: str | None = None


def code_version() -> str:
    """Short git commit for the working tree, with a ``+dirty`` marker, or ``"unknown"``.

    Resolved once per process. Never raises: a missing git, a tarball checkout, or a
    slow filesystem must not take down a calibration run that is otherwise fine.
    """
    global _commit_cache
    if _commit_cache is not None:
        return _commit_cache
    try:
        rev = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        _commit_cache = f"{rev}+dirty" if dirty else rev
    except Exception:  # noqa: BLE001 — provenance must never break a run
        _commit_cache = "unknown"
    return _commit_cache


def _scalar(value):
    """Coerce one fact to something NetCDF can store."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return ",".join(str(v) for v in sorted(value) if v is not None) or "none"
    return str(value)


def stamp(ds, **facts):
    """Return ``ds`` with provenance global attributes added.

    ``facts`` are whatever the caller knows that the path does not record — training
    years, fit gates, threshold subsets. Keys are prefixed and values coerced; a fact
    whose value is ``None`` is written as ``"none"`` rather than dropped, so "this run
    had no region filter" is distinguishable from "this file predates provenance".

    Mutates and returns the same object, matching how the callers already build their
    Datasets immediately before writing.
    """
    ds.attrs[f"{PREFIX}created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(
        timespec="seconds")
    ds.attrs[f"{PREFIX}code_version"] = code_version()
    for key, value in facts.items():
        ds.attrs[f"{PREFIX}{key}"] = _scalar(value)
    return ds


def read(ds) -> dict:
    """The provenance facts on a Dataset, unprefixed ({} for a file written before this)."""
    return {k[len(PREFIX):]: v for k, v in ds.attrs.items() if k.startswith(PREFIX)}

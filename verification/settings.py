# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""The per-agency study parameters, from one editable file — ``config/verification.toml``.

Three things an agency chooses for itself, which used to be Python tables only a
developer could reach:

* **exceedance thresholds** — which events the threshold curves describe
  (default: ``store.VARIABLES``)
* **altitude bands** — how the country is stratified by elevation
  (default: ``diagnostics.ALTITUDE_BANDS``)
* **seasons** — which windows the curves are pooled over (``seasons.SEASONS_BY_COUNTRY``)

The code tables remain, and remain the DEFAULT. This file only overrides: a country the
config does not mention, or a section it leaves out, behaves exactly as before. That is
what lets the file be optional and lets an agency change one thing without restating the
rest.

**A section REPLACES rather than merges.** Writing a ``[Senegal.thresholds]`` table gives
Senegal exactly the thresholds listed there — the code defaults are not appended. Merging
would make "remove a threshold I do not care about" impossible to express, which is half
of what the file is for.

WHY THIS RAISES INSTEAD OF WARNING
----------------------------------
``config.deployment_defaults`` deliberately swallows a broken ``deployment.toml``: a bad
country fails *visibly* (the dashboard shows no data), so limping on beats bricking the
launch. The opposite is true here. A threshold list silently falling back to the default
produces a complete, plausible, WRONG set of curves — scored against events nobody asked
about, with nothing on screen to say so. So a malformed file stops the run, naming the
country and key at fault.

WHAT CHANGING THIS FILE COSTS
-----------------------------
All three are **stored axes**, not display-time selections, so editing them invalidates
existing output rather than re-slicing it:

* thresholds  -> re-run ``calibration.run`` *and* ``verification.run``; the calibration
  emits ``p_exceed`` over exactly these values and the curves select them back by label.
* altitude bands -> re-run ``verification.run`` (they ride on the crps/mae/bias summaries).
* seasons -> re-run ``verification.run`` for the curve metrics, which pool over time and
  so cannot be sliced by season after the fact.

Stale files are not deleted and not overwritten: they simply stop being selected, because
a curve is found by the label in its path. Old thresholds linger in the dashboard until
removed by hand.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The one file. ``VERIFICATION_SETTINGS_FILE`` redirects it (tests, or an agency keeping
#: its parameters outside the checkout).
DEFAULT_SETTINGS_FILE = _REPO_ROOT / "config" / "verification.toml"

#: Section names a country table may hold. Anything else is a typo, and a typo that was
#: ignored would look exactly like a setting that had no effect.
_SECTIONS = frozenset({"thresholds", "altitude_bands", "seasons"})

#: Season slugs become PATH SEGMENTS and are matched against admin region names during
#: curve discovery, so they are restricted to a shape that cannot be mistaken for one.
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: ``yearly`` is implicit for every country and is not a window anyone defines.
_RESERVED_SLUGS = frozenset({"yearly"})

#: Auto-inserted into every band set, first: the union of the numeric bands (land only).
_ALL_BAND = "all"


class SettingsError(RuntimeError):
    """A malformed ``config/verification.toml``. Raised, never swallowed — see module doc."""


def settings_path() -> Path:
    """The settings file this process reads. Read at call time, like every other config."""
    value = os.environ.get("VERIFICATION_SETTINGS_FILE", "").strip()
    return Path(value) if value else DEFAULT_SETTINGS_FILE


def _fail(what: str) -> None:
    raise SettingsError(f"{settings_path()}: {what}")


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where}: expected a number, got {value!r}")
    return float(value)


def _scalar(value, where: str) -> float | int:
    """Validate a number but keep ``int`` as ``int``.

    The threshold list becomes a coordinate, and curves are selected back BY LABEL. The
    code tables spell these as ints, so coercing ``37`` to ``37.0`` here would change the
    dtype of every newly written file against a tree full of int-keyed ones — the kind of
    mismatch that yields an empty selection rather than an error.
    """
    _number(value, where)
    return value


def _parse_thresholds(country: str, table) -> dict[str, tuple[float, ...]]:
    if not isinstance(table, dict):
        _fail(f"[{country}.thresholds] must be a table of variable = [values]")
    out: dict[str, tuple[float, ...]] = {}
    for variable, values in table.items():
        where = f"[{country}.thresholds] {variable}"
        if not isinstance(values, list):
            _fail(f"{where}: expected a list, got {values!r}")
        # An EMPTY list is meaningful and allowed: it turns the threshold metrics off for
        # that variable, which is how temperature_min is configured out of a heat study.
        nums = tuple(_scalar(v, where) for v in values)
        if len(set(nums)) != len(nums):
            _fail(f"{where}: duplicate values in {list(nums)}")
        if list(nums) != sorted(nums):
            _fail(f"{where}: values must ascend, got {list(nums)}")
        out[str(variable)] = nums
    return out


def _parse_altitude_bands(country: str, table) -> dict[str, tuple[float, float] | None]:
    if not isinstance(table, dict):
        _fail(f"[{country}.altitude_bands] must be a table of name = [low, high]")
    if not table:
        _fail(f"[{country}.altitude_bands] is empty; omit the section to keep the default")
    out: dict[str, tuple[float, float] | None] = {_ALL_BAND: None}
    for name, span in table.items():
        where = f"[{country}.altitude_bands] {name!r}"
        if str(name) == _ALL_BAND:
            _fail(f"{where}: '{_ALL_BAND}' is added automatically; list only numeric bands")
        if not isinstance(span, list) or not 1 <= len(span) <= 2:
            _fail(f"{where}: expected [low, high] or [low] for open-ended, got {span!r}")
        low = _number(span[0], where)
        high = float("inf")
        if len(span) == 2:
            # "inf" spelt as a string, since TOML has no infinity literal.
            high = (float("inf") if isinstance(span[1], str) and span[1].strip().lower()
                    in {"inf", "+inf", "infinity"} else _number(span[1], where))
        if not high > low:
            _fail(f"{where}: high must exceed low, got {low} to {high}")
        out[str(name)] = (low, high)
    return out


def _parse_seasons(country: str, table) -> dict[str, tuple[str, tuple[int, ...]]]:
    if not isinstance(table, dict):
        _fail(f"[{country}.seasons] must hold one [{country}.seasons.<slug>] per season")
    out: dict[str, tuple[str, tuple[int, ...]]] = {}
    for slug, body in table.items():
        where = f"[{country}.seasons.{slug}]"
        if not _SLUG_RE.match(str(slug)):
            _fail(f"{where}: a slug is lowercase letters, digits and underscores, "
                  f"starting with a letter — it becomes a path segment")
        if str(slug) in _RESERVED_SLUGS:
            _fail(f"{where}: '{slug}' is reserved; it is offered to every country already")
        if not isinstance(body, dict):
            _fail(f"{where}: expected a table with `months` and an optional `label`")
        months = body.get("months")
        if not isinstance(months, list) or not months:
            _fail(f"{where}: `months` must be a non-empty list of 1-12")
        nums = []
        for m in months:
            if isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= 12:
                _fail(f"{where}: `months` entries are 1-12, got {m!r}")
            nums.append(int(m))
        if len(set(nums)) != len(nums):
            _fail(f"{where}: duplicate month in {nums}")
        label = body.get("label") or str(slug).upper()
        if not isinstance(label, str):
            _fail(f"{where}: `label` must be text, got {label!r}")
        unknown = set(body) - {"months", "label"}
        if unknown:
            _fail(f"{where}: unknown key(s) {sorted(unknown)}; expected `months`, `label`")
        # Months are stored in the order written. A cross-year window is written in
        # calendar order of the window, not of the year (11, 12, 1, 2), and the pipeline
        # treats all of them as months of the SAME verification year.
        out[str(slug)] = (label, tuple(nums))
    return out


@lru_cache(maxsize=8)
def _load(path: str, _stamp: int) -> dict:
    """Parse and validate the file. Cached on (path, mtime) so an edit is picked up."""
    import tomllib
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except OSError as exc:
        raise SettingsError(f"{path}: cannot be read: {exc}") from exc
    except Exception as exc:  # tomllib.TOMLDecodeError and anything it wraps
        raise SettingsError(f"{path}: is not valid TOML: {exc}") from exc

    parsed: dict[str, dict] = {}
    for country, table in raw.items():
        if not isinstance(table, dict):
            _fail(f"top-level `{country}` must be a country table, e.g. [{country}.thresholds]")
        unknown = set(table) - _SECTIONS
        if unknown:
            _fail(f"[{country}]: unknown section(s) {sorted(unknown)}; "
                  f"expected {sorted(_SECTIONS)}")
        # The name AS WRITTEN is kept alongside the folded lookup key: the seasons merge
        # writes it back into SEASONS_BY_COUNTRY, which reads better keyed by a display
        # name than by a lower-cased one.
        entry: dict = {"name": str(country).strip()}
        if "thresholds" in table:
            entry["thresholds"] = _parse_thresholds(country, table["thresholds"])
        if "altitude_bands" in table:
            entry["altitude_bands"] = _parse_altitude_bands(country, table["altitude_bands"])
        if "seasons" in table:
            entry["seasons"] = _parse_seasons(country, table["seasons"])
        # Countries are matched case-insensitively: an agency should not have to know
        # how the code capitalises a name.
        parsed[str(country).strip().lower()] = entry
    return parsed


def _settings() -> dict:
    """The parsed file, or ``{}`` when there is none (the pure-code-defaults build)."""
    p = settings_path()
    try:
        stamp = p.stat().st_mtime_ns
    except OSError:
        return {}
    return _load(str(p), stamp)


def _for(country: str | None) -> dict:
    return _settings().get((country or "").strip().lower(), {})


def thresholds(variable: str, country: str | None) -> tuple[float, ...] | None:
    """Configured exceedance thresholds, or None to use the code default.

    ``()`` is a real answer meaning "no threshold metrics for this variable", which is
    why the miss is None rather than an empty tuple.
    """
    table = _for(country).get("thresholds")
    if table is None or variable not in table:
        return None
    return table[variable]


def altitude_bands(country: str | None) -> dict | None:
    """Configured elevation strata (``all`` already inserted), or None for the default."""
    return _for(country).get("altitude_bands")


def seasons(country: str | None) -> dict[str, tuple[str, tuple[int, ...]]] | None:
    """Configured seasons as ``slug -> (label, months)``, or None for the default."""
    return _for(country).get("seasons")


def all_seasons() -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    """Every configured country's seasons, keyed by the country name as written.

    Used once, at ``verification.seasons`` import, to widen the global slug vocabulary —
    a slug has to be known to curve DISCOVERY even when the deployment is a different
    country, or the reader would take that path segment for a region name.
    """
    return {entry["name"]: entry["seasons"]
            for entry in _settings().values() if "seasons" in entry}


def configured_countries() -> tuple[str, ...]:
    """Lower-cased names of the countries the file mentions. For diagnostics and tests."""
    return tuple(sorted(_settings()))

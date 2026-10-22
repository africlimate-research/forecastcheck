# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Compare two pipeline output trees variable-by-variable.

The equivalence gate for a behaviour-preserving refactor: re-run the pipeline, then
prove the new tree holds the *same numbers* as a snapshot of the old one. A path
refactor's failure mode is silently wrong values, not a crash, so "the run finished"
proves nothing on its own.

What it compares, and why that shape:

* **Data variables and coordinates**, by exact equality with NaN treated as equal to
  NaN in the same position. Bit-identical is the right bar here because the IDR
  engine is deterministic for every output except ``pit`` — fixed cell order, joblib
  preserving submission order, no cross-cell accumulation.
* **Global attributes are ignored.** Provenance (train years, code commit, timestamp)
  is *meant* to differ between runs; comparing it would fail every time.
* Files are keyed by their **path relative to each root**, which for this validation
  *is* the facet tuple, so a behaviour-preserving refactor must write to
  byte-identical paths. A file present on one side only is a
  failure in its own right — a silently-skipped lead is exactly what this catches.

``--exclude pit`` is expected for the CountryStore validation: the baseline was
written with a seedless RNG (``IDRDistribution.pit`` drew from OS entropy), so no
seed can reproduce it. Every other variable must match exactly.

Usage:
    python -m tools.compare_runs --baseline <dir> --candidate <dir> [--exclude pit ...]
    python -m tools.compare_runs --baseline b --candidate c --subtree idr/station
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass
class Mismatch:
    """One concrete difference, phrased so the reader can act on it."""

    rel: str
    detail: str


@dataclass
class Report:
    """Outcome of comparing two trees."""

    allow_extra: bool = False
    compared: int = 0
    identical: int = 0
    skipped_vars: set[str] = field(default_factory=set)
    only_baseline: list[str] = field(default_factory=list)
    only_candidate: list[str] = field(default_factory=list)
    mismatches: list[Mismatch] = field(default_factory=list)
    #: rel -> {var: max abs difference}, populated only for mismatching variables.
    worst: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every baseline file matched.

        Files present only in the *candidate* fail by default, which is what a
        like-for-like rerun wants. ``allow_extra`` relaxes that for the case where
        the candidate is a whole live tree and the baseline is a deliberate subset of
        it — there, unrelated files are expected and are not evidence of anything.
        A file missing from the candidate always fails: a silently skipped lead is
        precisely what this gate exists to catch.
        """
        extra_is_fatal = bool(self.only_candidate) and not self.allow_extra
        return not (self.mismatches or self.only_baseline or extra_is_fatal)


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality, treating NaN in the same position as equal.

    ``np.array_equal(..., equal_nan=True)`` rejects non-float dtypes, and these trees
    hold int64 counts (ROC contingency tables) alongside float fields, so the dtype
    is checked first.
    """
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _numeric(a: np.ndarray) -> bool:
    """True if this array supports arithmetic differencing."""
    return a.dtype.kind in "fciub"


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference over positions finite in both, or nan.

    Positions where either side is NaN are excluded, so this number says nothing
    about a change in the NaN *pattern* — use :func:`_nan_pattern_diff` for that.
    """
    if a.shape != b.shape or not (_numeric(a) and _numeric(b)):
        return float("nan")
    try:
        af, bf = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    except (TypeError, ValueError):
        return float("nan")
    both = np.isfinite(af) & np.isfinite(bf)
    if not both.any():
        return 0.0
    return float(np.max(np.abs(af[both] - bf[both])))


def _nan_pattern_diff(a: np.ndarray, b: np.ndarray) -> int:
    """How many cells changed between missing and present.

    A station that was NaN (uncalibrated — it failed the wet-day gate) becoming a
    number, or vice versa, is a substantive scientific change even though the max
    abs difference over commonly-finite cells can be exactly 0. Reporting only the
    magnitude would print "differs (max abs diff 0)", which reads like a non-finding.
    """
    if a.shape != b.shape or not (_numeric(a) and _numeric(b)):
        return 0
    try:
        af, bf = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    except (TypeError, ValueError):
        return 0
    return int(np.count_nonzero(np.isfinite(af) != np.isfinite(bf)))


def describe_difference(a: np.ndarray, b: np.ndarray) -> str:
    """Why two arrays differ, in terms a reviewer can weigh.

    Deliberately never returns a bare "max abs diff 0" or "max abs diff nan": both
    read as "nothing happened" next to a real regression.
    """
    if not (_numeric(a) and _numeric(b)):
        n = int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
        return f"{n} non-numeric value(s) differ"
    parts = []
    moved = _nan_pattern_diff(a, b)
    if moved:
        parts.append(f"missing/present changed at {moved} cell(s)")
    mad = _max_abs_diff(a, b)
    if mad != 0.0 and not np.isnan(mad):
        parts.append(f"max abs diff {mad:.6g}")
    return "; ".join(parts) if parts else "values differ"


def compare_datasets(base: xr.Dataset, cand: xr.Dataset, *,
                     exclude: frozenset[str] = frozenset()) -> list[str]:
    """Differences between two Datasets, as human-readable strings ([] if equivalent).

    Compares data variables and coordinates; ignores global attributes.
    """
    problems: list[str] = []
    for kind, bnames, cnames in (
        ("var", set(base.data_vars), set(cand.data_vars)),
        ("coord", set(base.coords), set(cand.coords)),
    ):
        bnames, cnames = bnames - exclude, cnames - exclude
        for missing in sorted(bnames - cnames):
            problems.append(f"{kind} {missing!r} missing from candidate")
        for extra in sorted(cnames - bnames):
            problems.append(f"{kind} {extra!r} only in candidate")
        for name in sorted(bnames & cnames):
            a = np.asarray(base[name].values)
            b = np.asarray(cand[name].values)
            if a.shape != b.shape:
                problems.append(f"{kind} {name!r} shape {a.shape} != {b.shape}")
            elif not _arrays_equal(a, b):
                problems.append(f"{kind} {name!r} differs "
                                f"({describe_difference(a, b)})")
    return problems


def compare_trees(baseline, candidate, *, exclude=frozenset(),
                  subtree: str | None = None, allow_extra: bool = False) -> Report:
    """Compare every ``*.nc`` under two roots, keyed by relative path."""
    broot, croot = Path(baseline), Path(candidate)
    rep = Report(skipped_vars=set(exclude), allow_extra=allow_extra)

    def rels(root: Path) -> dict[str, Path]:
        base = root / subtree if subtree else root
        if not base.is_dir():
            return {}
        return {str(p.relative_to(root)): p for p in sorted(base.rglob("*.nc"))}

    bfiles, cfiles = rels(broot), rels(croot)
    rep.only_baseline = sorted(set(bfiles) - set(cfiles))
    rep.only_candidate = sorted(set(cfiles) - set(bfiles))

    for rel in sorted(set(bfiles) & set(cfiles)):
        rep.compared += 1
        try:
            with xr.open_dataset(bfiles[rel]) as b, xr.open_dataset(cfiles[rel]) as c:
                problems = compare_datasets(b, c, exclude=exclude)
                if problems:
                    worst = {}
                    for name in set(b.data_vars) & set(c.data_vars) - set(exclude):
                        bv, cv = np.asarray(b[name].values), np.asarray(c[name].values)
                        if _arrays_equal(bv, cv):
                            continue
                        d = _max_abs_diff(bv, cv)
                        # A NaN-pattern-only change has magnitude 0 but is still a
                        # difference; record it rather than dropping it silently.
                        worst[name] = 0.0 if np.isnan(d) else d
                    if worst:
                        rep.worst[rel] = worst
        except Exception as exc:  # noqa: BLE001 — a bad file is a finding, not a crash
            problems = [f"could not open: {exc!r}"]
        if problems:
            rep.mismatches.append(Mismatch(rel=rel, detail="; ".join(problems)))
        else:
            rep.identical += 1
    return rep


def format_report(rep: Report, *, limit: int = 20) -> str:
    """A verdict a reviewer can read without opening the trees."""
    lines = [
        f"compared {rep.compared} file(s): {rep.identical} identical, "
        f"{len(rep.mismatches)} differing",
    ]
    if rep.skipped_vars:
        lines.append(f"excluded variables: {sorted(rep.skipped_vars)}")
    if rep.only_baseline:
        lines.append(f"MISSING from candidate ({len(rep.only_baseline)}):")
        lines += [f"  - {r}" for r in rep.only_baseline[:limit]]
    if rep.only_candidate:
        note = " (allowed)" if rep.allow_extra else ""
        lines.append(f"EXTRA in candidate ({len(rep.only_candidate)}){note}:")
        shown = 3 if rep.allow_extra else limit
        lines += [f"  + {r}" for r in rep.only_candidate[:shown]]
    if rep.mismatches:
        lines.append(f"DIFFERING ({len(rep.mismatches)}):")
        for m in rep.mismatches[:limit]:
            lines.append(f"  ! {m.rel}")
            lines.append(f"      {m.detail}")
        if len(rep.mismatches) > limit:
            lines.append(f"  … and {len(rep.mismatches) - limit} more")
    lines.append("VERDICT: EQUIVALENT" if rep.ok else "VERDICT: NOT EQUIVALENT")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, help="snapshot of the trusted output")
    ap.add_argument("--candidate", required=True, help="tree produced by the new code")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="variable names to ignore (use 'pit' for the IDR refactor)")
    ap.add_argument("--subtree", default=None,
                    help="compare only this path under both roots, e.g. idr/station")
    ap.add_argument("--allow-extra", action="store_true",
                    help="do not fail on files present only in the candidate "
                         "(for when the candidate is a full live tree)")
    ap.add_argument("--limit", type=int, default=20, help="max findings to print")
    args = ap.parse_args(argv)

    rep = compare_trees(args.baseline, args.candidate,
                        exclude=frozenset(args.exclude), subtree=args.subtree,
                        allow_extra=args.allow_extra)
    print(format_report(rep, limit=args.limit))
    if rep.compared == 0:
        print("ERROR: nothing was compared — check the roots and --subtree")
        return 2
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())

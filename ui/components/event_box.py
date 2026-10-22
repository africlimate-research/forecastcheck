# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""How much data is behind a threshold curve, stated under it.

A threshold curve looks equally authoritative at every rung and is not: a rare rung may
rest on a few hundred events. Two numbers — how many events, and what fraction of the
evaluation set that is — let a reader tell it apart from a rung that rests on a million.

Counts are per series, not per chart. Models do not share an evaluation set: AIFS
publishes ~8 initialisations a month against GraphCast's daily run, so at one threshold
its curve rests on roughly a quarter of the observations.

The card is built from ``stat_blocks``' own ``_card`` / ``_label`` rather than a shell
of its own, so it is the same object as the Descriptive Stats and T-Test cards — white,
1px grey border, 3px accent top rule, micro-caps title — and it stays the same object
when that design changes. Reaching for a sibling component's private helpers is the
same bargain ``threshold_pdf_report`` already makes with ``pdf_report``: one look, one
place to change it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import streamlit as st

from ui.components.stat_blocks import _F, _label
from ui.services.threshold_data import EventCount, event_counts

#: The green the other insight cards use for their accent rule.
_ACCENT = "#0a9938"

_HAIRLINE = "#F3F4F6"
_VALUE = "#171d1a"
_NAME = "#374151"


def combine(parts: Sequence[Optional[EventCount]]) -> Optional[EventCount]:
    """Pool several slices of one series into one count — e.g. several years.

    Events and evaluation sets ADD; the fraction is recomputed from the totals rather
    than averaged, so a sparse year cannot pull the base rate around by the same amount
    as a full one.

    Cost-loss slices carry a rate and no counts, so a mixture falls back to the plain
    mean of the rates and reports no counts: weighting them correctly needs the very
    totals that file does not store, and a count invented from elsewhere would describe
    a different pooling than the curve.
    """
    real = [p for p in parts if p is not None]
    if not real:
        return None
    if len(real) == 1:
        return real[0]
    if any(p.events is None or p.total is None for p in real):
        fracs = [p.fraction for p in real if p.fraction is not None]
        return EventCount(None, None, sum(fracs) / len(fracs) if fracs else None)
    events = sum(p.events for p in real)
    total = sum(p.total for p in real)
    return EventCount(events, total, (events / total) if total > 0 else None)


def counting_series(series, metric: str, load):
    """The series to count from, so every threshold tab shows the same event box.

    Cost-loss files store only the base rate, so for that tab each series is swapped for
    the same slice's ROC (else reliability) dataset, which carries the counts. All three
    are pooled from one sample, so the numbers describe the same curve. ``load(metric)``
    returns ``(label, dataset)`` pairs for the current selection; a label with neither
    file keeps its own dataset (rate only).
    """
    if metric != "cost_loss":
        return series
    counted: dict = {}
    for other in ("roc", "reliability"):
        for label, ds in load(other):
            counted.setdefault(label, ds)
    return [(label, counted.get(label, ds)) for label, ds in series]


def _fmt_count(n: float) -> str:
    return f"{n:,.0f}"


def _fmt_fraction(frac: float) -> str:
    """Percent at a precision that survives four orders of magnitude.

    A fixed ``.1%`` prints a rare rung as "0.0%", which reads as *none* when it is
    hundreds of real events.
    """
    if frac >= 0.10:
        return f"{frac:.0%}"
    if frac >= 0.01:
        return f"{frac:.1%}"
    if frac >= 0.001:
        return f"{frac:.2%}"
    return f"{frac:.3%}"


def _shell(title: str, body: str) -> str:
    """``stat_blocks._card`` without its 3px accent top rule.

    Everything else is that card, value for value — white fill, 1px #E5E7EB border,
    8px radius, the same soft shadow and the same micro-caps title in the accent
    green. Only the coloured rule goes: this card sits directly beneath a chart
    rather than heading a column of cards, and the rule read as a second axis line
    under the plot.

    Spelt out here rather than passed as a flag to ``_card`` so the shared card keeps
    exactly one appearance; if that shell changes, this is the one place to match.
    """
    return (
        f"<div style='background:#FFFFFF;border:1px solid #E5E7EB;border-radius:8px;"
        f"overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.06);'>"
        f"<div style='{_F}font-size:0.6rem;font-weight:700;letter-spacing:0.12em;"
        f"text-transform:uppercase;color:{_ACCENT};padding:8px 14px 6px;"
        f"border-bottom:1px solid {_HAIRLINE};'>{title}</div>"
        f"{body}</div>"
    )


def _hero(caption: str, value: str) -> str:
    return (f"<div style='flex:1;'>{_label(caption)}"
            f"<div style='{_F}font-size:22px;font-weight:800;color:{_VALUE};"
            f"line-height:1;'>{value}</div></div>")


def _series_block(name: str, c: Optional[EventCount], *, named: bool,
                  last: bool) -> str:
    events = "—" if c is None or c.events is None else _fmt_count(c.events)
    total = "—" if c is None or c.total is None else _fmt_count(c.total)
    frac = "—" if c is None or c.fraction is None else _fmt_fraction(c.fraction)
    # One series needs no name — the card sits under its own chart. Several do, and
    # then the name is what says which line each row of numbers belongs to.
    heading = ("" if not named else
               f"<div style='{_F}font-size:11px;font-weight:700;color:{_NAME};"
               f"margin-bottom:5px;'>{name}</div>")
    edge = "" if last else f"border-bottom:1px solid {_HAIRLINE};"
    rule = f"<div style='width:1px;background:{_HAIRLINE};margin:0 14px;'></div>"
    # Left to right the row reads as the sentence it is: 901 out of 2,278,615, 0.040%.
    return (
        f"<div style='padding:9px 14px;{edge}'>{heading}"
        f"<div style='display:flex;'>"
        + _hero("Events", events) + rule
        + _hero("Evaluated", total) + rule
        + _hero("% of evaluation set", frac)
        + "</div></div>"
    )


def render_event_box(series, *, threshold, units: str = "", band: str = "all",
                     lead=None, region: str | None = None, years=None,
                     season: str | None = None) -> None:
    """The sample behind each drawn curve, as an insight card under the figure.

    ``series`` is the ``(label, dataset)`` list the figure itself is built from, so the
    numbers describe exactly what is plotted — same region, season, band and dates —
    rather than a separate query that could drift from it. Nothing here is pinned to a
    year: whatever the page selected is what was loaded, and a label appearing more than
    once (one entry per year, say) is pooled by :func:`combine`.
    """
    if not series:
        return

    # Group by label first, so a series delivered as several slices — one per year —
    # reports one pooled count rather than several rows for the same line on the chart.
    order: list[str] = []
    parts: dict[str, list] = {}
    for label, ds in series:
        if label not in parts:
            parts[label] = []
            order.append(label)
        parts[label].append(event_counts(ds, threshold, band))
    rows = [(label, combine(parts[label])) for label in order]
    if all(c is None for _, c in rows):
        return

    unit_txt = f" {units}" if units else ""
    if years is None:
        years_txt = None
    else:
        ys = sorted({str(y) for y in (years if isinstance(years, (list, tuple, set))
                                      else [years])})
        years_txt = ys[0] if len(ys) == 1 else f"{ys[0]}–{ys[-1]}"
    title = " · ".join(x for x in [
        f"Sample &gt; {threshold}{unit_txt}",
        region if region else None,
        None if band == "all" else f"{band} m",
        f"{lead}h" if lead is not None else None,
        season if season and season != "yearly" else None,
        years_txt,
    ] if x)

    body = "".join(
        _series_block(label, c, named=len(rows) > 1, last=(i == len(rows) - 1))
        for i, (label, c) in enumerate(rows)
    )
    st.markdown(_shell(title, body), unsafe_allow_html=True)

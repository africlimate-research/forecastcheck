# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Lead-time availability — the rule every lead selector shares.

A lead selector must only offer choices that resolve to a real file, and a single
choice that drives several loads (many models, many metrics) must be valid for *all*
of them. So the offered set is the **intersection** of the per-series discovered leads:
a lead is shown only when every selected series has it, guaranteeing each one renders.
The static ``LEAD_TIMES`` axis is kept elsewhere only for canonical ordering, never to
decide which leads exist.
"""
from __future__ import annotations

from typing import Iterable


def common_leads(lead_lists: Iterable[Iterable[int]]) -> list[int]:
    """Sorted leads present in every provided series; empty if none given or any is empty."""
    sets = [set(leads) for leads in lead_lists]
    if not sets:
        return []
    return sorted(set.intersection(*sets))

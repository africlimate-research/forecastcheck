# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Deployment configuration — the single knob that makes this a per-agency platform.

One env var, ``VERIFICATION_COUNTRY`` (``"Senegal"`` on this branch), scopes a local
deployment to one met agency. It drives both the on-disk **domain** path segment
(``data/senegal/...``) and the UI's Region dropdown (the country + its regions). Unset ⇒
domain ``africa`` and the REGION_COUNTRIES list.

``country()`` reads at call time (not import) so a deployment sets it before launch and
tests can vary it. To spare the agency from setting env vars by hand, a per-agency branch
commits a ``deployment.toml`` and each launch entry point calls
:func:`apply_deployment_defaults` once before anything reads the config — see that function.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOYMENT_FILE = _REPO_ROOT / "deployment.toml"


def country() -> str | None:
    """The deployment country display name (a REGION_COUNTRIES key), or None if unset."""
    value = os.environ.get("VERIFICATION_COUNTRY", "").strip()
    return value or None


def slug(name: str | None) -> str:
    """Filesystem slug for a country name; ``"africa"`` when unscoped.

    Lowercased with internal whitespace collapsed to underscores, so a multi-word
    country becomes one path segment (``"New Country"`` -> ``new_country``) rather
    than a segment containing a space. ``"Senegal"`` -> ``senegal``.
    """
    if not name or not str(name).strip():
        return "africa"
    return "_".join(str(name).strip().lower().split())


def domain() -> str:
    """The path-domain segment for the deployment country, or 'africa' when unscoped."""
    return slug(country())


def truth_product() -> str | None:
    """The observational product a run writes to, or None to derive it from the disk.

    Returns only what is EXPLICITLY configured — ``VERIFICATION_TRUTH_PRODUCT``, else
    ``deployment.toml``'s ``truth_product``. None means nothing was configured, and
    :meth:`verification.store.CountryStore.current` then reads the product off the
    country's own ``truth/`` tree.

    That deliberate absence of a default is what makes **the country the only thing a
    deployment has to specify**: a constant default would silently write a phantom
    product tree next to the real one.

    This names only the product a run **writes to**. Which products a country *holds*
    is probed from disk (``CountryStore.products``), so the dashboard shows them all.
    """
    value = os.environ.get("VERIFICATION_TRUTH_PRODUCT", "").strip()
    if value:
        return value
    configured = deployment_defaults().get("truth_product")
    return str(configured) if configured else None


def data_root() -> Path:
    """Root directory holding every country's data tree.

    ``VERIFICATION_DATA_ROOT`` (seeded from ``deployment.toml``'s ``data_root``) wins;
    otherwise the repo's own ``data/``. Read at call time, like :func:`country`, so a
    deployment or test can set it late.
    """
    value = os.environ.get("VERIFICATION_DATA_ROOT", "").strip()
    return Path(value) if value else _REPO_ROOT / "data"


def deployment_defaults(path: Path | None = None) -> dict:
    """Committed per-deployment defaults from ``deployment.toml`` at the repo root.

    This file is how a per-agency branch pins its country (and data dir) so the agency
    doesn't set environment variables by hand: a plain checkout launches pre-configured.
    Returns ``{}`` when the file is absent (the pan-African / dev build) or unreadable.
    """
    f = path or _DEPLOYMENT_FILE
    if not f.exists():
        return {}
    import tomllib
    try:
        with open(f, "rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:  # noqa: BLE001 — a broken file must not brick the launch
        print(f"[config] ignoring unreadable {f.name}: {exc}")
        return {}


def apply_deployment_defaults(path: Path | None = None) -> None:
    """Seed the deployment env vars from ``deployment.toml`` without overriding real ones.

    Uses ``os.environ.setdefault``, so a value already in the environment always wins;
    this only fills the gap for an agency that launched with nothing set. Call once at
    each launch entry point (dashboard + pipeline CLIs) *before* anything reads the
    config, so the whole process sees the pinned values.

    Note there is no ``data_dir``: the diagnostics root is DERIVED from the country by
    :class:`verification.store.CountryStore`, not configured. A hand-written results root
    could disagree with where the pipeline wrote — silently, since discovery would just
    return nothing — and being pinned to one country it could not survive a second one.
    ``data_root`` (the parent of every country tree) is still configurable; a relative
    value resolves against the repo root, so it holds wherever the launch happened from.
    """
    d = deployment_defaults(path)
    country_default = d.get("country")
    if country_default:
        os.environ.setdefault("VERIFICATION_COUNTRY", str(country_default))
    value = d.get("data_root")
    if value:
        p = Path(value)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        os.environ.setdefault("VERIFICATION_DATA_ROOT", str(p))
    product = d.get("truth_product")
    if product:
        os.environ.setdefault("VERIFICATION_TRUTH_PRODUCT", str(product))

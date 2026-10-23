# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Deployment configuration — the single knob that makes this a per-agency platform.

One env var, ``VERIFICATION_COUNTRY`` (e.g. ``"Senegal"``), scopes a local deployment to
one met agency. It drives both the on-disk **domain** path segment (``data/senegal/...``)
and the UI's Region dropdown (the country + its regions). Unset ⇒ domain ``africa`` and
the REGION_COUNTRIES list.

``country()`` reads at call time (not import) so a deployment sets it before launch and
tests can vary it.

**Countries are files, not branches.** Every country's settings live in one committed
file, ``deployments/{name}.toml`` (e.g. ``senegal``), all on
``main``, so a change to the platform reaches every country at once. A machine runs as
ONE of them, named by :func:`deployment_name` (the git-ignored ``.deployment`` file, or
``VERIFICATION_DEPLOYMENT``). Each launch entry point calls
:func:`apply_deployment_defaults` once, before anything reads the config, to seed the env
vars from that file — see that function. No selection is the pan-African / development
build.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: One file per country, named by the country's id: ``deployments/senegal.toml``.
DEPLOYMENTS_DIR = _REPO_ROOT / "deployments"

#: This machine's country: one line naming a file in :data:`DEPLOYMENTS_DIR`
#: (``senegal``). Written once when the machine is set up, and git-ignored, so a
#: ``git pull`` never changes which country a machine runs as.
_SELECTION_FILE = _REPO_ROOT / ".deployment"

#: The ``VERIFICATION_DEPLOYMENT`` value that means "no country", overriding the
#: machine's ``.deployment`` (the pan-African build; ``tests/conftest.py`` pins the suite
#: to it). An empty value already means "unset", so "no country" needs a name of its own.
NO_DEPLOYMENT = "none"


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
    the deployment file's ``truth_product``. None means nothing was configured, and
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

    ``VERIFICATION_DATA_ROOT`` (seeded from the deployment file's ``data_root``) wins;
    otherwise the repo's own ``data/``. Read at call time, like :func:`country`, so a
    deployment or test can set it late.
    """
    value = os.environ.get("VERIFICATION_DATA_ROOT", "").strip()
    return Path(value) if value else _REPO_ROOT / "data"


def available_deployments() -> tuple[str, ...]:
    """Every country that has a deployment file, by name, sorted."""
    return tuple(sorted(f.stem for f in DEPLOYMENTS_DIR.glob("*.toml")))


def deployment_name() -> str | None:
    """The country this process runs as, or None for the pan-African build.

    ``VERIFICATION_DEPLOYMENT`` wins (a shell, a test, a second country on one
    machine); otherwise the machine's ``.deployment`` file; ``none`` in either means no
    country. Read at call time, like :func:`country`.
    """
    value = os.environ.get("VERIFICATION_DEPLOYMENT", "").strip()
    if not value and _SELECTION_FILE.exists():
        lines = (ln.strip() for ln in _SELECTION_FILE.read_text().splitlines())
        value = next((ln for ln in lines if ln and not ln.startswith("#")), "")
    return None if value in ("", NO_DEPLOYMENT) else value


def deployment_path(name: str) -> Path:
    """The file of the country called ``name`` (``senegal`` → ``deployments/senegal.toml``).

    Raises ``ValueError``, listing the countries that exist, when there is none: a
    misspelt name must stop the launch rather than run the platform unconfigured — or,
    worse, show another country's data.
    """
    f = DEPLOYMENTS_DIR / f"{name}.toml"
    if not f.exists():
        raise ValueError(f"no deployment {name!r}: expected {f.relative_to(_REPO_ROOT)}; "
                         f"available: {', '.join(available_deployments())}")
    return f


def deployment_file() -> Path | None:
    """The selected country's file, or None when no country is selected."""
    name = deployment_name()
    return None if name is None else deployment_path(name)


def deployment_defaults(path: Path | None = None) -> dict:
    """The selected country's settings (:func:`deployment_file`), or those in ``path``.

    This is how a machine set up for one country launches pre-configured, without anyone
    setting environment variables by hand. Returns ``{}`` only when no country is
    selected (the pan-African / dev build). A selected country whose file is missing or
    does not parse raises ``ValueError`` — at launch, from
    :func:`apply_deployment_defaults` — because a country silently running unconfigured
    is worse than one that does not start.
    """
    f = path or deployment_file()
    if f is None:
        return {}
    import tomllib
    try:
        with open(f, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"deployment file {f}: {exc}") from exc


def apply_deployment_defaults(path: Path | None = None) -> None:
    """Seed the deployment env vars from the country's file without overriding real ones.

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

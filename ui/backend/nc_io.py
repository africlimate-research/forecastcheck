# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""Serialized NetCDF reads — the bundled libhdf5 is not thread-safe.

The dashboard opens NetCDF files from several worker threads at once: the
Scores-tab precache pool (``netcdf4`` engine) and climatology pool
(``h5netcdf`` engine), the spatial-map figure-prewarm pool, per-render
daemon threads, and the Streamlit script-runner thread(s) themselves. The
libhdf5 shipped inside the ``netCDF4``/``h5netcdf`` wheels is built without
``--enable-threadsafe`` and keeps a *process-global* metadata cache with no
internal locking, so two concurrent calls into it corrupt that cache and
segfault (``H5C_unprotect`` writing to a null-ish address).

Every entry into libhdf5 must therefore be serialized behind one
process-wide lock — and that means not only the open but the *lazy array
reads* that follow it, since ``xr.open_dataset`` defers I/O to element
access. :func:`read_dataset` closes that gap: it opens, reads the whole
file into memory, and closes the backend, all while holding the lock, and
returns a Dataset fully detached from the file. Later ``.sel``/``.load``/
``.values`` on the result never re-enter libhdf5, so they are safe to run
on any thread.

GEOS/shapely and NumPy are thread-safe and deliberately stay outside this
lock — only the HDF5 portion is serialized, so the pools keep overlapping
network, masking, and compute work.
"""
from __future__ import annotations

import threading
from typing import Any

import xarray as xr

# Guards every call into libhdf5 across all engines (netcdf4 + h5netcdf) and
# all threads in this process. Reentrant so a lock-holding read can nest.
_HDF5_LOCK = threading.RLock()


def read_dataset(path: Any, **open_kwargs: Any) -> xr.Dataset:
    """Read ``path`` fully into memory and close the backend, under the HDF5 lock.

    A drop-in replacement for ``xr.open_dataset`` at any read site the UI
    reaches from a worker thread. The returned Dataset is detached from the
    file — safe to ``.sel``/``.load``/``.squeeze`` afterwards without holding
    the lock. ``open_kwargs`` are forwarded verbatim (e.g. ``engine=``).
    """
    with _HDF5_LOCK:
        with xr.open_dataset(path, **open_kwargs) as ds:
            return ds.load()

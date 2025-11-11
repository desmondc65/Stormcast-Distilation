#!/usr/bin/env python3
"""CLI entry point for the nc_to_zarr pipeline."""

from __future__ import annotations

if __package__ in {None, ""}:
    # Allow execution as a standalone script: python nc_to_zarr.py
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from nc_to_zarr_pipeline import NCToZarrPipeline, main
else:
    from .nc_to_zarr_pipeline import NCToZarrPipeline, main

__all__ = ["NCToZarrPipeline", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

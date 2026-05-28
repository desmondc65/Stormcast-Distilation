#!/usr/bin/env python3
"""Add JSON metadata for Zarr folders produced by nc_to_zarr.

This script inspects one Zarr store or a directory containing multiple
StormCast/CorrDiff Zarr stores and writes a single JSON metadata file with
schema, channel, time, grid, validity-mask, and stats summaries.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("xarray is required for zarr metadata extraction") from exc


def _as_python(value: Any) -> Any:
    """Convert numpy scalars to plain Python scalars for JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_float(value: Any) -> Optional[float]:
    """Convert numeric value to float, returning None for NaN/Inf/invalid values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _to_string_list(values: Iterable[Any]) -> List[str]:
    return [str(_as_python(v)) for v in values]


def _best_effort_time_summary(time_values: np.ndarray) -> Dict[str, Any]:
    """Build a compact and robust summary for a time coordinate."""
    summary: Dict[str, Any] = {
        "count": int(time_values.size),
        "dtype": str(time_values.dtype),
    }

    if time_values.size == 0:
        return summary

    if np.issubdtype(time_values.dtype, np.datetime64):
        start = np.datetime_as_string(time_values[0], unit="s")
        end = np.datetime_as_string(time_values[-1], unit="s")
        summary["start"] = start
        summary["end"] = end

        if time_values.size > 1:
            deltas = np.diff(time_values.astype("datetime64[ns]"))
            delta_hours = deltas.astype("timedelta64[s]").astype(np.int64) / 3600.0
            unique_steps = np.unique(delta_hours)
            summary["step_hours"] = _safe_float(unique_steps[0]) if unique_steps.size == 1 else None
            summary["is_regular"] = bool(unique_steps.size == 1)
        return summary

    # Numeric/object times (for example "hours since ...")
    raw_start = _as_python(time_values[0])
    raw_end = _as_python(time_values[-1])
    summary["start_raw"] = raw_start
    summary["end_raw"] = raw_end

    if time_values.size > 1:
        try:
            step = _as_python(time_values[1] - time_values[0])
            summary["step_raw"] = step
        except Exception:
            summary["step_raw"] = None
    return summary


def _grid_summary(array: np.ndarray) -> Dict[str, Any]:
    """Return shape/min/max for a latitude/longitude coordinate grid."""
    arr = np.asarray(array)
    result: Dict[str, Any] = {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
    }

    finite = np.isfinite(arr)
    if finite.any():
        result["min"] = _safe_float(np.nanmin(arr))
        result["max"] = _safe_float(np.nanmax(arr))
    else:
        result["min"] = None
        result["max"] = None
    return result


def _validity_summary(ds: xr.Dataset) -> Optional[Dict[str, Any]]:
    """Summarize validity flags if the dataset has a 'valid' variable."""
    if "valid" not in ds.data_vars:
        return None

    valid = np.asarray(ds["valid"].to_numpy(), dtype=bool)
    if valid.ndim > 1:
        valid = valid.reshape(valid.shape[0], -1)[:, 0]

    total = int(valid.size)
    valid_count = int(valid.sum())
    invalid_count = total - valid_count

    return {
        "total_timesteps": total,
        "valid_timesteps": valid_count,
        "invalid_timesteps": invalid_count,
        "valid_ratio": _safe_float(valid_count / total) if total > 0 else None,
    }


def _collect_variable_summary(ds: xr.Dataset) -> Dict[str, Dict[str, Any]]:
    """Collect variable-level metadata without expensive full-data scans."""
    variables: Dict[str, Dict[str, Any]] = {}

    for name, var in ds.data_vars.items():
        chunks: Optional[List[List[int]]]
        if var.chunks is None:
            chunks = None
        else:
            chunks = [[int(c) for c in chunk_dim] for chunk_dim in var.chunks]

        var_info: Dict[str, Any] = {
            "dims": list(var.dims),
            "shape": [int(x) for x in var.shape],
            "dtype": str(var.dtype),
            "chunks": chunks,
        }
        if var.attrs:
            var_info["attrs"] = {
                str(k): _as_python(v)
                for k, v in var.attrs.items()
            }
        variables[name] = var_info

    return variables


def _open_zarr(path: pathlib.Path) -> xr.Dataset:
    """Open Zarr with consolidated metadata when possible."""
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def _collect_store_metadata(store_path: pathlib.Path, root: pathlib.Path) -> Dict[str, Any]:
    """Read one Zarr store and build metadata summary."""
    relative_path = str(store_path.relative_to(root)) if store_path.is_relative_to(root) else str(store_path)
    store_meta: Dict[str, Any] = {
        "store_path": relative_path,
    }

    parts = store_path.parts
    if "LowRes" in parts:
        store_meta["section"] = "LowRes"
    elif "HighRes" in parts:
        store_meta["section"] = "HighRes"
    elif "invariants" in parts:
        store_meta["section"] = "invariants"
    else:
        store_meta["section"] = "other"

    ds = _open_zarr(store_path)
    try:
        store_meta["sizes"] = {str(k): int(v) for k, v in ds.sizes.items()}
        if ds.attrs:
            store_meta["attrs"] = {
                str(k): _as_python(v)
                for k, v in ds.attrs.items()
            }

        if "channel" in ds.coords:
            channels = _to_string_list(ds.coords["channel"].to_numpy().tolist())
            store_meta["channels"] = {
                "count": len(channels),
                "names": channels,
            }

        if "time" in ds.coords:
            time_values = np.asarray(ds.coords["time"].to_numpy())
            store_meta["time"] = _best_effort_time_summary(time_values)
            units = ds.coords["time"].attrs.get("units")
            if units is not None:
                store_meta["time"]["units"] = str(units)

        if "latitude" in ds.coords:
            store_meta["latitude"] = _grid_summary(ds.coords["latitude"].to_numpy())
        if "longitude" in ds.coords:
            store_meta["longitude"] = _grid_summary(ds.coords["longitude"].to_numpy())

        validity = _validity_summary(ds)
        if validity is not None:
            store_meta["validity"] = validity

        store_meta["variables"] = _collect_variable_summary(ds)
    finally:
        ds.close()

    return store_meta


def _load_optional_npy(path: pathlib.Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def _collect_stats_metadata(root: pathlib.Path) -> List[Dict[str, Any]]:
    """Collect stats artifacts under LowRes/stats and HighRes/stats when available."""
    sections = ["LowRes", "HighRes"]
    output: List[Dict[str, Any]] = []

    for section in sections:
        stats_dir = root / section / "stats"
        if not stats_dir.exists():
            continue

        means = _load_optional_npy(stats_dir / "means.npy")
        stds = _load_optional_npy(stats_dir / "stds.npy")

        channels_path = stats_dir / "channels.txt"
        channels: List[str] = []
        if channels_path.exists():
            with channels_path.open("r", encoding="utf-8") as fh:
                channels = [line.strip() for line in fh if line.strip()]

        section_info: Dict[str, Any] = {
            "section": section,
            "stats_dir": str(stats_dir.relative_to(root)),
            "channels": channels,
        }

        if means is not None:
            section_info["means"] = [float(x) for x in means.tolist()]
        if stds is not None:
            section_info["stds"] = [float(x) for x in stds.tolist()]

        if channels and means is not None and stds is not None and len(channels) == len(means) == len(stds):
            section_info["channel_stats"] = {
                ch: {
                    "mean": float(means[idx]),
                    "std": float(stds[idx]),
                }
                for idx, ch in enumerate(channels)
            }
        output.append(section_info)

    return output


def _find_zarr_stores(root: pathlib.Path) -> List[pathlib.Path]:
    """Return sorted list of Zarr stores under a root path."""
    if root.suffix == ".zarr" and root.is_dir():
        return [root]
    return sorted(path for path in root.rglob("*.zarr") if path.is_dir())


def _build_root_summary(stores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create aggregate counts across all discovered stores."""
    summary: Dict[str, Any] = {
        "store_count": len(stores),
        "sections": {},
    }
    section_counts: Dict[str, int] = {}
    for item in stores:
        section = str(item.get("section", "other"))
        section_counts[section] = section_counts.get(section, 0) + 1
    summary["sections"] = section_counts
    return summary


def build_metadata(root: pathlib.Path) -> Dict[str, Any]:
    """Build complete metadata document for the provided root directory."""
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    zarr_stores = _find_zarr_stores(root)
    stores_meta: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for store in zarr_stores:
        try:
            stores_meta.append(_collect_store_metadata(store, root))
        except Exception as exc:  # pragma: no cover - best effort metadata collection
            errors.append({
                "store_path": str(store.relative_to(root)) if store.is_relative_to(root) else str(store),
                "error": str(exc),
            })

    metadata: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "zarr_root": str(root),
        "summary": _build_root_summary(stores_meta),
        "stores": stores_meta,
        "stats": _collect_stats_metadata(root),
    }

    if errors:
        metadata["errors"] = errors

    return metadata


def write_metadata(metadata: Dict[str, Any], output_path: pathlib.Path, overwrite: bool) -> None:
    """Write metadata dictionary to JSON file."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a JSON metadata file for nc_to_zarr-produced Zarr folders."
    )
    parser.add_argument(
        "zarr_root",
        type=pathlib.Path,
        help="Path to a Zarr folder or a directory containing Zarr stores",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output metadata JSON path (default: <zarr_root>/zarr_metadata.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output JSON if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.zarr_root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else root / "zarr_metadata.json"

    metadata = build_metadata(root)
    write_metadata(metadata, output, overwrite=args.overwrite)

    print(f"Metadata written: {output}")
    print(f"Discovered stores: {metadata['summary']['store_count']}")
    print(f"Sections: {metadata['summary']['sections']}")
    if "errors" in metadata:
        print(f"Stores with read errors: {len(metadata['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

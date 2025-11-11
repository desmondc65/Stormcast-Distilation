from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import netCDF4 as nc4
import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)


def init_worker_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name, logging.DEBUG),
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
    )


@dataclass
class Task:
    path: Path
    output_dir: Path
    variable: str
    time_variable: str
    dt_hours: int
    pressure_level: int | None = None  # None for surface variables
    source_variable: str | None = None  # Base variable name in the file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split monthly NetCDF files into hourly NetCDF files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("split_month_to_hour_nc.yaml"),
        help="Path to configuration YAML.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of worker processes to use.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> MutableMapping[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, MutableMapping):
        raise ValueError("Configuration file must contain a mapping.")
    return cfg


def _format_time_for_filename(tobj: object) -> str:
    """Format a time object (datetime or cftime) to YYYYMMDDTHH for filenames."""
    # Support datetime.datetime and cftime datetime-like objects
    try:
        return f"{tobj.year:04d}{tobj.month:02d}{tobj.day:02d}T{tobj.hour:02d}"
    except Exception:
        # Fallback for numpy datetime64: convert to string safely
        try:
            dt_str = np.datetime_as_string(np.datetime64(tobj), unit="h")
            return dt_str.replace(":", "")
        except Exception:
            raise TypeError(f"Unsupported time object for filename formatting: {type(tobj)!r}")


def determine_hour_file_name(variable: str, base_time: object) -> str:
    dt = _format_time_for_filename(base_time)
    return f"{variable}_{dt}.nc"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_time_axis(dataset: nc4.Dataset, time_variable: str):
    """Return decoded time values as Python/cftime datetime objects.

    Keeping native datetime objects preserves calendar information for correct
    encoding when writing out.
    """
    time_var = dataset.variables[time_variable]
    return nc4.num2date(
        time_var[:],
        units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
    )


def write_hour_file(
    dest: Path,
    variable: str,
    time_value: object,
    data_slice: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    global_attrs: Mapping[str, object],
    var_attrs: Mapping[str, object],
    time_var_name: str,
    time_units: str,
    time_calendar: str | None,
    time_attrs: Mapping[str, object] | None,
) -> None:
    ensure_directory(dest.parent)

    fill_value = var_attrs.get("_FillValue")
    try:
        with nc4.Dataset(dest, "w") as dst:
            # Preserve dimension names from source for time axis
            time_dim = time_var_name
            dst.createDimension("latitude", latitudes.shape[0])
            dst.createDimension("longitude", longitudes.shape[0])
            dst.createDimension(time_dim, 1)

            lat_var = dst.createVariable("latitude", latitudes.dtype, ("latitude",))
            lon_var = dst.createVariable("longitude", longitudes.dtype, ("longitude",))
            time_var = dst.createVariable(time_var_name, "f8", (time_dim,))
            create_kwargs = {}
            if fill_value is not None:
                create_kwargs["fill_value"] = fill_value
            data_var = dst.createVariable(variable, data_slice.dtype, (time_dim, "latitude", "longitude"), **create_kwargs)

            lat_var[:] = latitudes
            lon_var[:] = longitudes
            # Encode time using HOURS since epoch (uniform output units) and preserve calendar
            # Use a canonical hours-since epoch format so downstream code sees hourly units
            out_time_units = "hours since 1970-01-01 00:00:00"
            time_var[:] = nc4.date2num([time_value], units=out_time_units, calendar=(time_calendar or "standard"))
            # Set time variable attributes to the output encoding
            time_var.units = out_time_units
            if time_calendar is not None:
                time_var.calendar = time_calendar
            if time_attrs:
                for k, v in time_attrs.items():
                    # Avoid overwriting essential encoding we just set
                    if k in {"units", "calendar"}:
                        continue
                    setattr(time_var, k, v)
            data_var[0, :, :] = data_slice

            for key, value in global_attrs.items():
                setattr(dst, key, value)
            for key, value in var_attrs.items():
                if key == "_FillValue":
                    continue
                setattr(data_var, key, value)
    except PermissionError as exc:
        LOGGER.error("Unable to write %s (%s). Check directory ownership and permissions.", dest, exc)
        return

    LOGGER.info("Wrote %s", dest)


def process_variable_file(task: Task) -> None:
    path = task.path
    variable = task.variable
    source_var = task.source_variable or variable
    
    LOGGER.info("Processing %s (output variable: %s, source variable: %s, pressure level: %s)", path, variable, source_var, task.pressure_level)
    with nc4.Dataset(path) as src:
        if source_var not in src.variables:
            LOGGER.warning("Variable %s not found in %s; skipping", source_var, path)
            LOGGER.info("Variable not found; skipping. Available variables in %s: %s", path, list(src.variables.keys()))
            return
        if task.time_variable not in src.variables:
            raise KeyError(f"Time variable {task.time_variable} not found in {path}")

        data_var = src.variables[source_var]
        times = read_time_axis(src, task.time_variable)
        latitudes = np.asarray(src.variables["latitude"][:])
        longitudes = np.asarray(src.variables["longitude"][:])


        global_attrs = {attr: getattr(src, attr) for attr in src.ncattrs()}
        var_attrs = {attr: getattr(data_var, attr) for attr in data_var.ncattrs()}
        time_var_src = src.variables[task.time_variable]
        time_units = getattr(time_var_src, "units", "hours since 1970-01-01")
        time_calendar = getattr(time_var_src, "calendar", None)
        time_attrs = {attr: getattr(time_var_src, attr) for attr in time_var_src.ncattrs()}

        step = np.timedelta64(task.dt_hours, "h")
        variable_out_dir = task.output_dir / variable

        for idx, time_value in enumerate(times):
            # time_value is a datetime-like object (datetime or cftime)
            try:
                hour = int(getattr(time_value, "hour"))
            except Exception:
                # Robust fallback for unusual types
                hour = int(str(time_value)[11:13])
            if hour % task.dt_hours != 0:
                LOGGER.debug("Skipping time %s for variable %s (not aligned with dt=%s)", time_value, variable, task.dt_hours)
                continue
            
            LOGGER.info("Processing time %s for variable %s", time_value, variable)
            
            slice_data = data_var[idx, ...]
            
            if isinstance(slice_data, np.ma.MaskedArray):
                slice_data = slice_data.filled(np.nan)
            dest = variable_out_dir / determine_hour_file_name(variable, time_value)
            write_hour_file(
                dest,
                variable,
                time_value,
                np.asarray(slice_data),
                latitudes,
                longitudes,
                global_attrs,
                var_attrs,
                task.time_variable,
                time_units,
                time_calendar,
                time_attrs,
            )
            LOGGER.info("Processed time %s for variable %s", time_value, variable)


def gather_tasks(root_dir: Path, output_dir: Path, variables: Sequence[str], variable_mapping: Mapping[str, str], time_variable: str, dt_hours: int) -> List[Task]:
    tasks: List[Task] = []
    
    # Process regular surface variables
    for variable in variables:
        pattern = f"**/{variable}*.nc"
        matches = sorted(root_dir.rglob(pattern))
        if not matches:
            LOGGER.warning("No files found for variable %s using pattern %s", variable, pattern)
            continue
        
        # Get the actual variable name in the NetCDF file (may differ from filename)
        nc_variable_name = variable_mapping.get(variable, variable)
        
        for path in matches:
            stem = path.stem.split("_")[0]
            if stem != variable:
                LOGGER.debug("Skipping %s because stem %s != variable %s", path, stem, variable)
                continue
            tasks.append(Task(
                path=path,
                output_dir=output_dir,
                variable=variable,  # Output variable name (from filename)
                time_variable=time_variable,
                dt_hours=dt_hours,
                pressure_level=None,
                source_variable=nc_variable_name  # Actual variable name in NetCDF
            ))
    
    return tasks


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.DEBUG),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = load_config(args.config)
    root_dir = Path(config["root_dir"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    time_variable = str(config.get("time_variable", "time"))
    dt_hours = int(config.get("dt_hours", 1))
    variables: Sequence[str] = list(config.get("variables", []))
    variable_mapping: Mapping[str, str] = dict(config.get("variable_mapping", {}))
    workers = int(config.get("workers", args.workers))
    # log variable list
    LOGGER.info("Surface variables to process: %s", variables)
    LOGGER.info("Variable name mapping: %s", variable_mapping)

    if dt_hours <= 0:
        raise ValueError("dt_hours must be positive.")
    if not variables:
        raise ValueError("No variables specified in configuration.")

    ensure_directory(output_dir)

    tasks = gather_tasks(root_dir, output_dir, variables, variable_mapping, time_variable, dt_hours)
    if not tasks:
        LOGGER.warning("No tasks found; exiting.")
        return

    worker_count = max(1, workers)
    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=worker_count, initializer=init_worker_logging, initargs=(args.log_level,)) as pool:
        pool.map(process_variable_file, tasks)


if __name__ == "__main__":
    main()

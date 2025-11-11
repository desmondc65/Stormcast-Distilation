from __future__ import annotations

import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from .nc_to_zarr_pipeline import NCToZarrPipeline, load_config


CONFIG_PATH = Path(__file__).with_name("config.yaml")
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "Stormcast_test" / "Zarr_test"
LOWRES_TRAIN_STORE = DATA_ROOT / "LowRes" / "train_era5.zarr"
HIGHRES_TRAIN_STORE = DATA_ROOT / "HighRes" / "train_rwrf.zarr"


def _prepare_config(tmp_path: Path):
    """Load repo config and adapt paths for isolated tests."""
    cfg = load_config(str(CONFIG_PATH))
    # Point data paths to temporary directories so pipeline init succeeds.
    for key in ("era5-path", "rwrf-path", "qpepre-path", "output-path"):
        target = tmp_path / key.replace("-", "_")
        target.mkdir(parents=True, exist_ok=True)
        cfg[key] = str(target)
    # Keep the repo settings but trim variable lists to first two entries for speed.
    cfg["era5-variables"] = list(cfg.get("era5-variables", []))[:2]
    cfg["rwrf-variables"] = list(cfg.get("rwrf-variables", []))[:2]
    cfg["invariant-variables"] = []
    cfg["max-workers"] = 1
    cfg["compute-stats"] = False
    cfg["overwrite"] = True
    cfg["skip-era5"] = False
    cfg["skip-rwrf"] = False
    return cfg


def _build_pipeline(tmp_path: Path):
    cfg = _prepare_config(tmp_path)
    pipeline = NCToZarrPipeline(cfg)
    pipeline.era5_variables = cfg["era5-variables"]
    pipeline.rwrf_variables = cfg["rwrf-variables"]
    pipeline.invariant_variables = []
    pipeline.max_workers = 1  # force sequential path for deterministic tests
    pipeline.compute_stats = False
    return pipeline, cfg


def test_config_settings_match_repo_defaults(tmp_path):
    """Ensure NCToZarrPipeline honors the geometry settings from config.yaml."""
    pipeline, cfg = _build_pipeline(tmp_path)
    expected_domain = tuple(cfg["domain-size"])
    expected_lon = tuple(cfg["lon-bounds"])
    expected_lat = tuple(cfg["lat-bounds"])

    assert pipeline.domain_size == expected_domain
    assert (pipeline.lon_min, pipeline.lon_max) == expected_lon
    assert (pipeline.lat_min, pipeline.lat_max) == expected_lat

    # Helpful context in CI logs.
    print(f"Domain size from config: {pipeline.domain_size}")
    print(f"Longitude bounds: {(pipeline.lon_min, pipeline.lon_max)}")
    print(f"Latitude bounds: {(pipeline.lat_min, pipeline.lat_max)}")


def test_valid_masks_flag_missing_timesteps(tmp_path):
    """Validate that LowRes/HighRes writers emit the (time, 1) valid flag."""
    pipeline, _ = _build_pipeline(tmp_path)
    datetimes = [
        datetime(2019, 8, 1, 0),
        datetime(2019, 8, 1, 1),
    ]
    captured = {}

    def fake_write(
        self,
        store,
        data_var,
        data_cube,
        time_coord,
        channels,
        lon_grid,
        lat_grid,
        valid_mask=None,
    ):
        captured[data_var] = {
            "shape": data_cube.shape,
            "valid": None if valid_mask is None else valid_mask.copy(),
            "time": time_coord.copy(),
        }
        print(f"{data_var} data cube shape: {data_cube.shape}")
        if valid_mask is not None:
            print(f"{data_var} valid mask shape: {valid_mask.shape}")

    pipeline._write_zarr = types.MethodType(fake_write, pipeline)

    def fake_era5_single(self, dt, var):
        if dt.hour == 1 and var == self.era5_variables[0]:
            return None
        data = np.full(self.domain_size, fill_value=float(dt.hour + 1), dtype=np.float32)
        lon = np.zeros(self.domain_size, dtype=np.float32)
        lat = np.zeros(self.domain_size, dtype=np.float32)
        return data, lon, lat

    pipeline._process_era5_single = types.MethodType(fake_era5_single, pipeline)

    def fake_rwrf_single(self, dt):
        lon = np.zeros(self.domain_size, dtype=np.float32)
        lat = np.zeros(self.domain_size, dtype=np.float32)
        variables = {}
        for idx, name in enumerate(self.rwrf_variables):
            if dt.hour == 1 and idx == len(self.rwrf_variables) - 1:
                # Drop the last channel to force an invalid timestep.
                continue
            variables[name] = np.full(self.domain_size, fill_value=float(dt.hour + idx), dtype=np.float32)
        if not variables:
            return None
        return variables, lon, lat

    pipeline._process_rwrf_single = types.MethodType(fake_rwrf_single, pipeline)

    lowres_store = Path(pipeline.output_base) / "LowRes" / "test_lowres.zarr"
    highres_store = Path(pipeline.output_base) / "HighRes" / "test_highres.zarr"

    pipeline.process_lowres("test", datetimes, lowres_store)
    pipeline.process_highres("test", datetimes, highres_store)

    assert "LowRes" in captured and "HighRes" in captured

    low_shape = captured["LowRes"]["shape"]
    high_shape = captured["HighRes"]["shape"]
    assert low_shape[0] == len(datetimes)
    assert high_shape[0] == len(datetimes)

    low_valid = captured["LowRes"]["valid"]
    high_valid = captured["HighRes"]["valid"]
    assert low_valid.shape == (len(datetimes),)
    assert high_valid.shape == (len(datetimes),)
    assert low_valid.dtype == np.bool_
    assert high_valid.dtype == np.bool_
    assert low_valid.tolist() == [True, False]
    assert high_valid.tolist() == [True, False]


@pytest.mark.skipif(
    not (LOWRES_TRAIN_STORE.exists() and HIGHRES_TRAIN_STORE.exists()),
    reason="StormCast sample Zarr stores missing",
)
def test_load_valid_metadata_matches_real_store(tmp_path):
    pipeline, _ = _build_pipeline(tmp_path)
    low_meta = pipeline._load_valid_metadata(LOWRES_TRAIN_STORE)
    high_meta = pipeline._load_valid_metadata(HIGHRES_TRAIN_STORE)

    assert low_meta is not None
    assert high_meta is not None
    assert low_meta["mask"].shape[0] == high_meta["mask"].shape[0] == 48
    assert low_meta["mask"].sum() == 8
    assert high_meta["mask"].sum() == 48

    shared, overlap = pipeline._compute_valid_overlap(low_meta, high_meta)
    assert shared == 48
    assert overlap == 8

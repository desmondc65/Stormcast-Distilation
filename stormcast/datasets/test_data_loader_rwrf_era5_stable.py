from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import xarray as xr

from .data_loader_rwrf_era5_stable import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Stormcast_test" / "Zarr_test"
LOWRES_TRAIN = DATA_ROOT / "LowRes" / "train_era5.zarr"
LOWRES_VALID = DATA_ROOT / "LowRes" / "valid_era5.zarr"
HIGHRES_TRAIN = DATA_ROOT / "HighRes" / "train_rwrf.zarr"
HIGHRES_VALID = DATA_ROOT / "HighRes" / "valid_rwrf.zarr"

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (LOWRES_TRAIN, LOWRES_VALID, HIGHRES_TRAIN, HIGHRES_VALID)),
    reason="StormCast sample Zarr stores missing",
)

TRAIN_PAIR = "stormcast_train_pair"
VALID_PAIR = "stormcast_valid_pair"
DT_HOURS = 6


@pytest.fixture(autouse=True)
def _init_distributed_manager(monkeypatch):
    """Bypass DistributedManager singleton initialization requirements in tests."""
    from physicsnemo.distributed import DistributedManager

    monkeypatch.setattr(DistributedManager, "_is_initialized", True, raising=False)
    yield
    monkeypatch.setattr(DistributedManager, "_is_initialized", False, raising=False)


@dataclass
class DummyParams:
    location: str
    dt: int = DT_HOURS
    train_dates: Tuple[str, str] = ("2019/08/01", "2019/08/02")
    valid_dates: Tuple[str, str] = ("2019/08/03", "2019/08/04")
    kept_LowRes_channels: str | Sequence[str] = "all"
    kept_HighRes_channels: str | Sequence[str] = "all"
    invariants: List[str] | None = None
    HighRes_img_size: Tuple[int, int] = (224, 128)
    exp_train_zarrs: List[str] | None = None
    exp_valid_zarrs: List[str] | None = None

    def __post_init__(self) -> None:
        if self.invariants is None:
            self.invariants = []
        if self.exp_train_zarrs is None:
            self.exp_train_zarrs = []
        if self.exp_valid_zarrs is None:
            self.exp_valid_zarrs = []


def _symlink_tree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"Destination {dst} already exists and is not a symlink")
        dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _prepare_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "StormCast_fixture"
    lowres_dir = fixture / "LowRes"
    highres_dir = fixture / "HighRes"
    invariant_dir = fixture / "invariants"
    for directory in (lowres_dir, highres_dir, invariant_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _symlink_tree(LOWRES_TRAIN, lowres_dir / f"{TRAIN_PAIR}.zarr")
    _symlink_tree(HIGHRES_TRAIN, highres_dir / f"{TRAIN_PAIR}.zarr")
    _symlink_tree(LOWRES_VALID, lowres_dir / f"{VALID_PAIR}.zarr")
    _symlink_tree(HIGHRES_VALID, highres_dir / f"{VALID_PAIR}.zarr")
    _symlink_tree(DATA_ROOT / "LowRes" / "stats", lowres_dir / "stats")
    _symlink_tree(DATA_ROOT / "HighRes" / "stats", highres_dir / "stats")
    _symlink_tree(DATA_ROOT / "invariants" / "invariants.zarr", invariant_dir / "invariants.zarr")

    return fixture


def _to_datetime(ts: np.datetime64) -> datetime:
    seconds = (ts - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).replace(tzinfo=None)


def _expected_valid_samples(
    low_store: Path,
    high_store: Path,
    date_range: Tuple[str, str],
    dt_hours: int,
) -> List[datetime]:
    start = datetime.strptime(date_range[0], "%Y/%m/%d").replace(hour=0, minute=0, second=0)
    end = datetime.strptime(date_range[1], "%Y/%m/%d").replace(hour=23, minute=0, second=0)
    with xr.open_zarr(low_store, consolidated=True) as low, xr.open_zarr(high_store, consolidated=True) as high:
        low_mask = np.asarray(low["valid"].values).reshape(low.dims["time"], -1)[:, 0]
        high_mask = np.asarray(high["valid"].values).reshape(high.dims["time"], -1)[:, 0]
        overlap = np.logical_and(low_mask, high_mask)
        times = np.asarray(low.time.values)[overlap]
    overlap_py: List[datetime] = []
    for ts in times:
        dt_ts = _to_datetime(ts)
        if start <= dt_ts <= end:
            overlap_py.append(dt_ts)
    available = set(overlap_py)
    result: List[datetime] = []
    for ts in overlap_py:
        target = ts + timedelta(hours=dt_hours)
        if target > end:
            continue
        if target in available:
            result.append(ts)
    return result


@pytest.mark.parametrize(
    "train_flag, low_store, high_store, date_range",
    [
        (True, LOWRES_TRAIN, HIGHRES_TRAIN, ("2019/08/01", "2019/08/02")),
        (False, LOWRES_VALID, HIGHRES_VALID, ("2019/08/03", "2019/08/04")),
    ],
)
def test_dataset_valid_samples_follow_real_overlap(tmp_path, train_flag, low_store, high_store, date_range):
    fixture = _prepare_fixture(tmp_path)
    params = DummyParams(
        location=str(fixture),
        exp_train_zarrs=[TRAIN_PAIR],
        exp_valid_zarrs=[VALID_PAIR],
    )
    dataset = Dataset(params, train=train_flag)
    expected = _expected_valid_samples(low_store, high_store, date_range, params.dt)
    print("Expected valid samples:")
    for dt in expected:
        print(f"  {dt.isoformat()}")
    print("Dataset valid samples:")
    for dt in dataset.valid_samples:
        print(f"  {dt.isoformat()}")
        
    assert dataset.valid_samples == expected
    assert len(dataset) == len(expected)

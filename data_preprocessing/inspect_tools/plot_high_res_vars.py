#!/usr/bin/env python3
"""
Plot HighRes variable distributions from StormCast Zarr stores without OOM.

The script processes each channel in time chunks so it never loads the full
time-axis into memory. For each selected HighRes variable, it creates:
1. Histogram with Gaussian overlay (using empirical mean/std)
2. QQ plot against the fitted Gaussian
3. One high-resolution image per channel
4. CSV summary with Gaussian diagnostics

Examples:
	python plot_high_res_vars.py /path/to/zarr_output_dir
	python plot_high_res_vars.py /path/to/HighRes/stormcast_test_train.zarr
	python plot_high_res_vars.py /path/to/zarr_output_dir --channels t2m,u10,v10,qpepre --chunk-time 24
"""

import argparse
import csv
import math
import pathlib
import sys
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

try:
	import xarray as xr

	HAS_XARRAY = True
except ImportError:
	HAS_XARRAY = False
	xr = None


GAUSSIAN_COVERAGE_1SIGMA = 0.682689492137
GAUSSIAN_COVERAGE_2SIGMA = 0.954499736104
GAUSSIAN_COVERAGE_3SIGMA = 0.997300203937


@dataclass
class FirstPassStats:
	channel_name: str
	valid_count: int
	missing_count: int
	min_value: float
	max_value: float
	mean: float
	std: float
	skewness: float
	excess_kurtosis: float


@dataclass
class ChannelDiagnostics:
	channel_name: str
	valid_count: int
	missing_count: int
	min_value: float
	max_value: float
	mean: float
	std: float
	skewness: float
	excess_kurtosis: float
	within_1sigma: float
	within_2sigma: float
	within_3sigma: float
	out_of_hist_range_fraction: float
	hist_edges: np.ndarray
	hist_counts: np.ndarray
	qq_sample: np.ndarray


def _filename_safe(name: str) -> str:
	"""Create a filesystem-safe token for output filenames."""
	safe = []
	for ch in name:
		if ch.isalnum() or ch in {"-", "_"}:
			safe.append(ch)
		else:
			safe.append("_")
	return "".join(safe).strip("_") or "channel"


def _open_zarr_dataset(zarr_path: pathlib.Path) -> Any:
	"""Open Zarr with consolidated metadata if possible, fallback otherwise."""
	try:
		return xr.open_zarr(zarr_path, consolidated=True)
	except Exception:
		return xr.open_zarr(zarr_path, consolidated=False)


def _iter_channel_chunks(
	da_channel: Any,
	*,
	time_dim: str,
	chunk_time: int,
) -> Iterable[tuple[np.ndarray, int]]:
	"""Yield finite values for one channel in time chunks.

	Returns tuples of (finite_values, total_values_in_chunk).
	"""
	n_time = int(da_channel.sizes[time_dim])
	for start in range(0, n_time, chunk_time):
		stop = min(start + chunk_time, n_time)
		block = da_channel.isel({time_dim: slice(start, stop)}).values
		flat = np.asarray(block, dtype=np.float64).ravel()
		if flat.size == 0:
			continue

		finite_mask = np.isfinite(flat)
		if np.any(finite_mask):
			finite_vals = flat[finite_mask]
		else:
			finite_vals = np.empty((0,), dtype=np.float64)

		yield finite_vals, int(flat.size)


def _compute_first_pass_stats(
	da_channel: Any,
	*,
	channel_name: str,
	time_dim: str,
	chunk_time: int,
) -> FirstPassStats:
	"""Streaming first pass: compute moments, min/max, and counts."""
	valid_count = 0
	total_count = 0

	sum_1 = 0.0
	sum_2 = 0.0
	sum_3 = 0.0
	sum_4 = 0.0

	min_value = math.inf
	max_value = -math.inf

	for finite_vals, chunk_total in _iter_channel_chunks(
		da_channel,
		time_dim=time_dim,
		chunk_time=chunk_time,
	):
		total_count += chunk_total
		if finite_vals.size == 0:
			continue

		valid_count += int(finite_vals.size)
		min_value = min(min_value, float(np.min(finite_vals)))
		max_value = max(max_value, float(np.max(finite_vals)))

		sum_1 += float(np.sum(finite_vals))
		sq = finite_vals * finite_vals
		sum_2 += float(np.sum(sq))
		cube = sq * finite_vals
		sum_3 += float(np.sum(cube))
		sum_4 += float(np.sum(cube * finite_vals))

	if valid_count == 0:
		raise ValueError(f"Channel '{channel_name}' has no finite values")

	mean = sum_1 / valid_count
	var = max(sum_2 / valid_count - mean * mean, 0.0)
	std = math.sqrt(var)

	if var > 0.0:
		m3 = sum_3 / valid_count - 3.0 * mean * (sum_2 / valid_count) + 2.0 * mean**3
		m4 = (
			sum_4 / valid_count
			- 4.0 * mean * (sum_3 / valid_count)
			+ 6.0 * (mean**2) * (sum_2 / valid_count)
			- 3.0 * mean**4
		)
		skewness = m3 / (var ** 1.5)
		excess_kurtosis = m4 / (var * var) - 3.0
	else:
		skewness = 0.0
		excess_kurtosis = 0.0

	return FirstPassStats(
		channel_name=channel_name,
		valid_count=valid_count,
		missing_count=total_count - valid_count,
		min_value=min_value,
		max_value=max_value,
		mean=mean,
		std=std,
		skewness=skewness,
		excess_kurtosis=excess_kurtosis,
	)


def _compute_second_pass_diagnostics(
	da_channel: Any,
	*,
	stats: FirstPassStats,
	time_dim: str,
	chunk_time: int,
	bins: int,
	sample_size: int,
	range_mode: str,
	hist_sigma: float,
) -> ChannelDiagnostics:
	"""Streaming second pass: histogram, sigma coverage, and QQ sample."""
	if range_mode == "minmax" or stats.std <= 0.0:
		hist_min = stats.min_value
		hist_max = stats.max_value
	else:
		hist_min = stats.mean - hist_sigma * stats.std
		hist_max = stats.mean + hist_sigma * stats.std

	if not np.isfinite(hist_min) or not np.isfinite(hist_max) or hist_min == hist_max:
		center = stats.mean
		pad = max(abs(center) * 1e-6, 1e-6)
		hist_min = center - pad
		hist_max = center + pad

	hist_edges = np.linspace(hist_min, hist_max, bins + 1)
	hist_counts = np.zeros((bins,), dtype=np.int64)

	within_1_count = 0
	within_2_count = 0
	within_3_count = 0
	out_left_count = 0
	out_right_count = 0

	sample_parts: list[np.ndarray] = []
	sample_collected = 0
	stride = max(1, stats.valid_count // max(sample_size, 1))
	global_valid_index = 0

	for finite_vals, _chunk_total in _iter_channel_chunks(
		da_channel,
		time_dim=time_dim,
		chunk_time=chunk_time,
	):
		if finite_vals.size == 0:
			continue

		if stats.std > 0.0:
			abs_dev = np.abs(finite_vals - stats.mean)
			within_1_count += int(np.count_nonzero(abs_dev <= stats.std))
			within_2_count += int(np.count_nonzero(abs_dev <= 2.0 * stats.std))
			within_3_count += int(np.count_nonzero(abs_dev <= 3.0 * stats.std))
		else:
			within_1_count += int(finite_vals.size)
			within_2_count += int(finite_vals.size)
			within_3_count += int(finite_vals.size)

		out_left_count += int(np.count_nonzero(finite_vals < hist_min))
		out_right_count += int(np.count_nonzero(finite_vals > hist_max))

		in_range_mask = (finite_vals >= hist_min) & (finite_vals <= hist_max)
		in_range_vals = finite_vals[in_range_mask]
		if in_range_vals.size > 0:
			hist_counts += np.histogram(in_range_vals, bins=bins, range=(hist_min, hist_max))[0]

		if sample_collected < sample_size:
			if stride <= 1:
				sampled = finite_vals
			else:
				start = (stride - (global_valid_index % stride)) % stride
				sampled = finite_vals[start::stride]

			global_valid_index += int(finite_vals.size)

			if sampled.size > 0:
				remaining = sample_size - sample_collected
				if sampled.size > remaining:
					sampled = sampled[:remaining]
				if sampled.size > 0:
					sample_parts.append(sampled.copy())
					sample_collected += int(sampled.size)
		else:
			global_valid_index += int(finite_vals.size)

	qq_sample = (
		np.concatenate(sample_parts).astype(np.float64, copy=False)
		if sample_parts
		else np.empty((0,), dtype=np.float64)
	)

	valid_count = max(stats.valid_count, 1)
	return ChannelDiagnostics(
		channel_name=stats.channel_name,
		valid_count=stats.valid_count,
		missing_count=stats.missing_count,
		min_value=stats.min_value,
		max_value=stats.max_value,
		mean=stats.mean,
		std=stats.std,
		skewness=stats.skewness,
		excess_kurtosis=stats.excess_kurtosis,
		within_1sigma=within_1_count / valid_count,
		within_2sigma=within_2_count / valid_count,
		within_3sigma=within_3_count / valid_count,
		out_of_hist_range_fraction=(out_left_count + out_right_count) / valid_count,
		hist_edges=hist_edges,
		hist_counts=hist_counts,
		qq_sample=qq_sample,
	)


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
	"""Compute Gaussian PDF on x."""
	if std <= 0.0:
		return np.zeros_like(x)
	inv = 1.0 / (std * math.sqrt(2.0 * math.pi))
	z = (x - mean) / std
	return inv * np.exp(-0.5 * z * z)


def _plot_channel_row(ax_hist: Any, ax_qq: Any, diag: ChannelDiagnostics) -> None:
	"""Plot one row: histogram+Gaussian and QQ plot."""
	centers = 0.5 * (diag.hist_edges[:-1] + diag.hist_edges[1:])
	widths = np.diff(diag.hist_edges)
	bin_width = float(widths[0]) if widths.size > 0 else 1.0

	if diag.std > 0.0 and diag.valid_count > 0:
		density = diag.hist_counts / (diag.valid_count * bin_width)
		ax_hist.bar(centers, density, width=widths, color="#5B8FF9", alpha=0.65, edgecolor="none")
		gaussian = _normal_pdf(centers, diag.mean, diag.std)
		ax_hist.plot(centers, gaussian, color="#D62728", linewidth=2.0)
	else:
		ax_hist.axvline(diag.mean, color="#D62728", linewidth=2.0)

	ax_hist.set_title(f"{diag.channel_name} distribution", fontsize=11)
	ax_hist.set_xlabel("Value")
	ax_hist.set_ylabel("Density")
	ax_hist.grid(alpha=0.25)

	hist_text = (
		f"n={diag.valid_count:,}, missing={diag.missing_count:,}\n"
		f"mean={diag.mean:.5g}, std={diag.std:.5g}\n"
		f"skew={diag.skewness:.4g}, excess kurt={diag.excess_kurtosis:.4g}\n"
		f"|x-mu|<=1sigma: {diag.within_1sigma:.4f} (N(0,1): {GAUSSIAN_COVERAGE_1SIGMA:.4f})\n"
		f"|x-mu|<=2sigma: {diag.within_2sigma:.4f} (N(0,1): {GAUSSIAN_COVERAGE_2SIGMA:.4f})\n"
		f"|x-mu|<=3sigma: {diag.within_3sigma:.4f} (N(0,1): {GAUSSIAN_COVERAGE_3SIGMA:.4f})\n"
		f"outside plotted hist range: {diag.out_of_hist_range_fraction:.4f}"
	)
	ax_hist.text(
		0.01,
		0.99,
		hist_text,
		transform=ax_hist.transAxes,
		va="top",
		ha="left",
		fontsize=8,
		bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
	)

	ax_qq.set_title(f"{diag.channel_name} QQ vs Gaussian", fontsize=11)
	ax_qq.set_xlabel("Theoretical quantile")
	ax_qq.set_ylabel("Empirical quantile")
	ax_qq.grid(alpha=0.25)

	if diag.qq_sample.size >= 50 and diag.std > 0.0:
		probs = np.linspace(0.01, 0.99, 200)
		empirical_q = np.quantile(diag.qq_sample, probs)
		nd = NormalDist(mu=diag.mean, sigma=diag.std)
		theoretical_q = np.array([nd.inv_cdf(float(p)) for p in probs], dtype=np.float64)

		ax_qq.scatter(theoretical_q, empirical_q, s=8, alpha=0.45, color="#00A878")

		min_lim = float(min(np.min(theoretical_q), np.min(empirical_q)))
		max_lim = float(max(np.max(theoretical_q), np.max(empirical_q)))
		ax_qq.plot([min_lim, max_lim], [min_lim, max_lim], color="#D62728", linestyle="--", linewidth=1.5)
	else:
		ax_qq.text(
			0.5,
			0.5,
			"QQ sample too small or std=0",
			transform=ax_qq.transAxes,
			ha="center",
			va="center",
			fontsize=10,
		)


def _save_csv_summary(csv_path: pathlib.Path, diagnostics: list[ChannelDiagnostics]) -> None:
	"""Write diagnostics summary as CSV."""
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with open(csv_path, "w", newline="", encoding="utf-8") as fp:
		writer = csv.writer(fp)
		writer.writerow(
			[
				"channel",
				"valid_count",
				"missing_count",
				"min",
				"max",
				"mean",
				"std",
				"skewness",
				"excess_kurtosis",
				"within_1sigma",
				"within_2sigma",
				"within_3sigma",
				"gaussian_1sigma",
				"gaussian_2sigma",
				"gaussian_3sigma",
				"out_of_hist_range_fraction",
			]
		)
		for diag in diagnostics:
			writer.writerow(
				[
					diag.channel_name,
					diag.valid_count,
					diag.missing_count,
					diag.min_value,
					diag.max_value,
					diag.mean,
					diag.std,
					diag.skewness,
					diag.excess_kurtosis,
					diag.within_1sigma,
					diag.within_2sigma,
					diag.within_3sigma,
					GAUSSIAN_COVERAGE_1SIGMA,
					GAUSSIAN_COVERAGE_2SIGMA,
					GAUSSIAN_COVERAGE_3SIGMA,
					diag.out_of_hist_range_fraction,
				]
			)


def _resolve_input_stores(path: pathlib.Path) -> list[pathlib.Path]:
	"""Resolve input path into one or more .zarr stores."""
	if path.suffix == ".zarr":
		return [path]

	if not path.is_dir():
		return []

	if path.name == "HighRes":
		return sorted(path.glob("*.zarr"))

	highres_dir = path / "HighRes"
	if highres_dir.exists() and highres_dir.is_dir():
		return sorted(highres_dir.glob("*.zarr"))

	return sorted(path.glob("*.zarr"))


def _resolve_channel_indices(
	channel_values: np.ndarray,
	selected_channels: list[str] | None,
) -> list[int]:
	"""Resolve optional channel selection (names or numeric indices)."""
	if selected_channels is None or len(selected_channels) == 0:
		return list(range(len(channel_values)))

	name_to_idx = {str(v): idx for idx, v in enumerate(channel_values)}
	lower_to_idx = {str(v).lower(): idx for idx, v in enumerate(channel_values)}

	indices: list[int] = []
	for token in selected_channels:
		tok = token.strip()
		if not tok:
			continue

		if tok.isdigit():
			idx = int(tok)
			if idx < 0 or idx >= len(channel_values):
				raise ValueError(f"Channel index out of range: {idx}")
			indices.append(idx)
			continue

		if tok in name_to_idx:
			indices.append(name_to_idx[tok])
			continue

		tok_lower = tok.lower()
		if tok_lower in lower_to_idx:
			indices.append(lower_to_idx[tok_lower])
			continue

		raise ValueError(f"Unknown channel: {tok}")

	seen = set()
	deduped: list[int] = []
	for idx in indices:
		if idx not in seen:
			deduped.append(idx)
			seen.add(idx)
	return deduped


def analyze_store(
	zarr_path: pathlib.Path,
	*,
	output_dir: pathlib.Path,
	data_var: str,
	channel_dim: str,
	time_dim: str,
	selected_channels: list[str] | None,
	log1p_channels: set[str] | None,
	chunk_time: int,
	bins: int,
	sample_size: int,
	range_mode: str,
	hist_sigma: float,
	dpi: int,
) -> None:
	"""Analyze one HighRes store and save outputs."""
	print(f"\nOpening: {zarr_path}")
	ds = _open_zarr_dataset(zarr_path)

	if data_var not in ds.data_vars:
		available = ", ".join(sorted(ds.data_vars))
		ds.close()
		raise ValueError(f"Data variable '{data_var}' not found. Available: {available}")

	da = ds[data_var]

	if channel_dim not in da.dims:
		ds.close()
		raise ValueError(f"Channel dimension '{channel_dim}' not in {da.dims}")
	if time_dim not in da.dims:
		ds.close()
		raise ValueError(f"Time dimension '{time_dim}' not in {da.dims}")

	if channel_dim in ds.coords:
		channel_values = np.asarray(ds.coords[channel_dim].values)
	else:
		channel_values = np.arange(da.sizes[channel_dim])

	indices = _resolve_channel_indices(channel_values, selected_channels)
	if not indices:
		ds.close()
		raise ValueError("No channels selected")

	print(f"Analyzing channels: {[str(channel_values[i]) for i in indices]}")
	diagnostics: list[ChannelDiagnostics] = []

	log1p_applied = False
	for idx in indices:
		channel_name = str(channel_values[idx])
		da_channel = da.isel({channel_dim: idx})
		if log1p_channels and channel_name.lower() in log1p_channels:
			# Same transform as the cleaning pipeline: log1p(max(0, x)).
			da_channel = np.log1p(da_channel.clip(min=0.0))
			channel_name = f"log1p_{channel_name}"
			log1p_applied = True
		print(f"  - First pass stats for channel '{channel_name}'")
		first = _compute_first_pass_stats(
			da_channel,
			channel_name=channel_name,
			time_dim=time_dim,
			chunk_time=chunk_time,
		)

		print(f"  - Second pass histogram/QQ for channel '{channel_name}'")
		diag = _compute_second_pass_diagnostics(
			da_channel,
			stats=first,
			time_dim=time_dim,
			chunk_time=chunk_time,
			bins=bins,
			sample_size=sample_size,
			range_mode=range_mode,
			hist_sigma=hist_sigma,
		)
		diagnostics.append(diag)

	ds.close()

	output_dir.mkdir(parents=True, exist_ok=True)
	saved_images: list[pathlib.Path] = []

	for diag in diagnostics:
		fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), squeeze=False)
		_plot_channel_row(axes[0, 0], axes[0, 1], diag)

		fig.suptitle(
			f"Gaussian diagnostics for {data_var}:{diag.channel_name} - {zarr_path.name}",
			fontsize=14,
			fontweight="bold",
		)
		fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

		channel_safe = _filename_safe(diag.channel_name)
		png_path = output_dir / f"{zarr_path.stem}_{data_var}_{channel_safe}_gaussian_check.png"
		fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
		plt.close(fig)
		saved_images.append(png_path)

	csv_tag = "_log1p" if log1p_applied else ""
	csv_path = output_dir / f"{zarr_path.stem}_{data_var}_gaussian_stats{csv_tag}.csv"
	_save_csv_summary(csv_path, diagnostics)

	for img in saved_images:
		print(f"Saved figure: {img}")
	print(f"Saved stats : {csv_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot HighRes variable distributions and Gaussian diagnostics in a memory-safe way"
	)
	parser.add_argument(
		"path",
		type=str,
		help="Path to output directory, HighRes directory, or a specific .zarr store",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Directory to save outputs (default: <input_parent>/distribution_plots)",
	)
	parser.add_argument(
		"--data-var",
		type=str,
		default="HighRes",
		help="Data variable to analyze (default: HighRes)",
	)
	parser.add_argument(
		"--channel-dim",
		type=str,
		default="channel",
		help="Channel dimension name (default: channel)",
	)
	parser.add_argument(
		"--time-dim",
		type=str,
		default="time",
		help="Time dimension name (default: time)",
	)
	parser.add_argument(
		"--channels",
		type=str,
		default=None,
		help="Comma-separated channels to analyze (names or indices). Default: all channels",
	)
	parser.add_argument(
		"--log1p-channels",
		type=str,
		default=None,
		help=(
			"Comma-separated channel names to transform with log1p(max(0, x)) before "
			"analysis (matches the cleaning pipeline). Outputs get a 'log1p_' prefix"
		),
	)
	parser.add_argument(
		"--chunk-time",
		type=int,
		default=48,
		help="Number of timesteps to load per chunk (default: 48)",
	)
	parser.add_argument(
		"--bins",
		type=int,
		default=160,
		help="Histogram bins (default: 160)",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=250000,
		help="Max sample size for QQ plot (default: 250000)",
	)
	parser.add_argument(
		"--range-mode",
		type=str,
		choices=["sigma", "minmax"],
		default="sigma",
		help="Histogram range mode: sigma (mu +/- n*std) or minmax (default: sigma)",
	)
	parser.add_argument(
		"--hist-sigma",
		type=float,
		default=6.0,
		help="Number of std for histogram range when --range-mode sigma (default: 6.0)",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=600,
		help="Saved figure DPI (default: 600)",
	)
	return parser.parse_args()


def main() -> None:
	if not HAS_XARRAY:
		print("ERROR: xarray is required. Please install xarray first.")
		sys.exit(1)

	args = parse_args()
	input_path = pathlib.Path(args.path).expanduser().resolve()

	if not input_path.exists():
		print(f"ERROR: Path does not exist: {input_path}")
		sys.exit(1)

	if args.chunk_time <= 0:
		print("ERROR: --chunk-time must be > 0")
		sys.exit(1)
	if args.bins <= 5:
		print("ERROR: --bins must be > 5")
		sys.exit(1)
	if args.sample_size <= 0:
		print("ERROR: --sample-size must be > 0")
		sys.exit(1)
	if args.hist_sigma <= 0:
		print("ERROR: --hist-sigma must be > 0")
		sys.exit(1)

	zarr_stores = _resolve_input_stores(input_path)
	if not zarr_stores:
		print(f"ERROR: No .zarr stores found under: {input_path}")
		sys.exit(1)

	selected_channels = None
	if args.channels:
		selected_channels = [item.strip() for item in args.channels.split(",") if item.strip()]

	log1p_channels = None
	if args.log1p_channels:
		log1p_channels = {
			item.strip().lower() for item in args.log1p_channels.split(",") if item.strip()
		}

	if args.output_dir is not None:
		out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
	else:
		if input_path.suffix == ".zarr":
			out_dir = input_path.parent / "distribution_plots"
		else:
			out_dir = input_path / "distribution_plots"

	print("=" * 80)
	print("Memory-safe HighRes Gaussian diagnostics")
	print("=" * 80)
	print(f"Input: {input_path}")
	print(f"Stores: {[p.name for p in zarr_stores]}")
	print(f"Output: {out_dir}")
	print(f"Chunk time: {args.chunk_time}, bins: {args.bins}, QQ sample size: {args.sample_size}")

	for store in zarr_stores:
		analyze_store(
			store,
			output_dir=out_dir,
			data_var=args.data_var,
			channel_dim=args.channel_dim,
			time_dim=args.time_dim,
			selected_channels=selected_channels,
			log1p_channels=log1p_channels,
			chunk_time=args.chunk_time,
			bins=args.bins,
			sample_size=args.sample_size,
			range_mode=args.range_mode,
			hist_sigma=args.hist_sigma,
			dpi=args.dpi,
		)

	print("=" * 80)
	print("Done.")


if __name__ == "__main__":
	main()

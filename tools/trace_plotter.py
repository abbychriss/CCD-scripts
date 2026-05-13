#!/usr/bin/env python3
"""
trace_plotter.py — Convert RJ45ADCDATA binary to CSV and plot CCD readout traces.

Pipeline:
    binary → CSV (binary_to_csv)
    CSV    → plot (plot_trace)

Row structure: rows are detected from V2 flag transitions in the ADC stream.
Each V2-high block marks a vertical transfer (between rows); samples between
two such blocks form one CCD row.

Usage examples:
    # convert and plot full trace (metafile auto-detected from binary path)
    python trace_plotter.py file.RJ45ADCDATA0

    # also save CSV
    python trace_plotter.py file.RJ45ADCDATA0 --save_csv

    # plot only the 3rd row (0-indexed) with a 1000 µs window on each side
    python trace_plotter.py file.RJ45ADCDATA0 --row 3 --window 1000

    # plot from a pre-converted CSV
    python trace_plotter.py file.csv --row 3

    # override the auto-detected metafile
    python trace_plotter.py file.RJ45ADCDATA0 --meta other.meta
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ADC_MHZ = 15  # ADC sampling frequency


# ---------------------------------------------------------------------------
# Binary → DataFrame
# ---------------------------------------------------------------------------

def _get_decoder():
    """Return a Decoder class from whichever acm package is installed.

    Prefers the new `acmdecoder` (no metafile required); falls back to the
    older `acmpy` which needs a JSON metafile alongside the binary.
    """
    try:
        from acmdecoder import Decoder
        return Decoder, "acmdecoder"
    except ImportError:
        pass
    try:
        from acmpy import Decoder
        return Decoder, "acmpy"
    except ImportError:
        sys.exit(
            "Neither `acmdecoder` nor `acmpy` is installed in this Python.\n"
            "Activate an environment that has one of them, e.g.:\n"
            "  conda activate acm-env"
        )


def load_binary(binfile: str, metafile: str = None) -> pd.DataFrame:
    """Decode an RJ45ADCDATA binary file to a DataFrame.

    With `acmdecoder` the metafile is optional and ignored.
    With the older `acmpy` it is required; if not given, look for a `.meta`
    file next to the binary.
    """
    Decoder, backend = _get_decoder()

    if backend == "acmdecoder":
        dec = Decoder(binfile, debug=0)
    else:
        if metafile is None:
            metafile = os.path.splitext(binfile)[0] + ".meta"
        if not os.path.exists(metafile):
            sys.exit(f"Metafile not found: {metafile}\n"
                     f"Pass it explicitly with --meta if it lives elsewhere.")
        dec = Decoder(binfile, metafile, debug=0)

    df = dec.df0_view.copy()      # columns: clk, V2, H2, TG, OG, SW, DG, RU, RD, val
    df["t"] = df.index / ADC_MHZ  # time axis in µs
    return df


def binary_to_csv(binfile: str, metafile: str = None,
                  outfile: str = None) -> str:
    """Decode binary file and write CSV. Returns the CSV path."""
    if outfile is None:
        outfile = binfile.rsplit(".", 1)[0] + ".csv"
    df = load_binary(binfile, metafile)
    df.to_csv(outfile, index=True)
    print(f"Saved CSV → {outfile}  ({len(df):,} rows)")
    return outfile


def load_csv(csvfile: str) -> pd.DataFrame:
    df = pd.read_csv(csvfile, index_col=0)
    if "t" not in df.columns:
        df["t"] = df.index / ADC_MHZ
    return df


# ---------------------------------------------------------------------------
# NROW resolution
# ---------------------------------------------------------------------------

def _nrow_from_metafile(binfile: str) -> int | None:
    """Look for a .meta JSON next to the binary and return its NROW field."""
    meta = os.path.splitext(binfile)[0] + ".meta"
    if not os.path.exists(meta):
        return None
    try:
        with open(meta) as f:
            return int(json.load(f)["NROW"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _nrow_from_filename(binfile: str) -> int | None:
    """Parse NROW out of names like trace_ch0_20x20x40_1x1_ch0_... → 20."""
    m = re.search(r"_ch\d+_(\d+)x\d+x\d+_", os.path.basename(binfile))
    return int(m.group(1)) if m else None


def resolve_nrow(binfile: str, cli_value: int | None) -> int:
    """Resolve NROW from --nrow, then metafile, then filename, else prompt."""
    if cli_value is not None:
        return cli_value

    n = _nrow_from_metafile(binfile)
    if n is not None:
        return n

    n = _nrow_from_filename(binfile)
    if n is not None:
        print(f"NROW={n} parsed from filename (no metafile found).",
              file=sys.stderr)
        return n

    if not sys.stdin.isatty():
        sys.exit("Could not determine NROW from metafile or filename. "
                 "Pass it explicitly with --nrow N.")
    try:
        return int(input("Enter NROW (number of CCD rows in this file): ").strip())
    except (ValueError, EOFError):
        sys.exit("Invalid NROW input.")


# ---------------------------------------------------------------------------
# Row detection
# ---------------------------------------------------------------------------

def detect_row_boundaries(df: pd.DataFrame, n_rows: int) -> np.ndarray:
    """
    Return the integer positions (in df) where each CCD row begins.

    A "row" is one contiguous burst of pedestal/signal (RD/RU) activity.
    We pick the top `n_rows - 1` inactivity gaps as row boundaries — these
    are always the largest gaps in the trace (vertical transfer + exposure
    wait between rows), regardless of binning factor.

    If a leading pre-activity region exists it is reported as an extra
    "row 0" (so the returned array has up to `n_rows + 1` entries).
    """
    active = ((df["RU"].to_numpy() == 1) | (df["RD"].to_numpy() == 1)).astype(np.int8)

    if active.sum() == 0:
        return np.array([0], dtype=np.int64)

    d = np.diff(active)
    starts = np.where(d ==  1)[0] + 1
    ends   = np.where(d == -1)[0] + 1
    if active[0] == 1:
        starts = np.concatenate([[0], starts])
    if active[-1] == 1:
        ends = np.concatenate([ends, [len(df)]])

    gaps = starts[1:] - ends[:-1]  # in samples

    n_splits = max(0, n_rows - 1)
    if n_splits >= len(gaps):
        # not enough gaps → split at every gap
        long_gap = np.ones(len(gaps), dtype=bool)
    else:
        # take the top n_splits largest gaps
        threshold = np.partition(gaps, -n_splits)[-n_splits] if n_splits > 0 \
                    else gaps.max() + 1
        long_gap = gaps >= threshold
        # tie-break: if too many gaps meet the threshold, keep only the top n
        if long_gap.sum() > n_splits:
            top_idx = np.argpartition(gaps, -n_splits)[-n_splits:]
            mask = np.zeros(len(gaps), dtype=bool)
            mask[top_idx] = True
            long_gap = mask

    row_starts = np.concatenate([[starts[0]], starts[1:][long_gap]])

    if row_starts[0] > 0:
        row_starts = np.concatenate([[0], row_starts])

    return row_starts.astype(np.int64)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_colors():
    return dict(
        ped="peru",
        sig="forestgreen",
        adc="dimgrey",
    )


def plot_full_trace(df: pd.DataFrame, title: str = "ADC trace",
                    max_trace_points: int = 500_000):
    """Reproduce the first-style plot: full trace with pedestal/signal overlays.

    The raw trace is downsampled to `max_trace_points` for rendering speed while
    keeping every pedestal/signal sample at full resolution.
    """
    c = _plot_colors()
    fig, ax = plt.subplots(figsize=(20, 5))

    t = df["t"].to_numpy()
    val = df["val"].to_numpy()
    ped_mask = df["RD"].to_numpy() == 1
    sig_mask = df["RU"].to_numpy() == 1

    # thin the background trace so matplotlib stays snappy with huge files
    stride = max(1, len(df) // max_trace_points)
    ax.plot(t[::stride], val[::stride], color=c["adc"], lw=0.4,
            label=os.path.basename(title), rasterized=True)

    ax.plot(t[ped_mask], val[ped_mask], ".", ms=1.5, color=c["ped"],
            label="pedestal", zorder=3, rasterized=True)
    ax.plot(t[sig_mask], val[sig_mask], ".", ms=1.5, color=c["sig"],
            label="signal", zorder=3, rasterized=True)

    ax.set_xlabel(r"time ($\mu$s)")
    ax.set_ylabel("ADC value (ADU)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_row(df: pd.DataFrame, row_idx: int, n_rows: int,
             window_us: float = 1000.0, title: str = "ADC trace"):
    """
    Plot a single CCD row, centred on its pedestal/signal integration region.

    The integration region (where RD or RU == 1) is typically much shorter than
    the full row (which includes exposure/wait time after readout).  The plot
    shows [first_ped_t - window_us, last_sig_t + window_us] so the actual
    measurement region is always visible.  If the row has no RD/RU samples the
    full row extent is used instead.

    Parameters
    ----------
    df         : full DataFrame (output of load_binary / load_csv)
    row_idx    : 0-indexed row number
    n_rows     : expected number of CCD rows (from metafile / filename)
    window_us  : extra context (µs) to add before and after the integration region
    title      : figure title (usually the filename)
    """
    boundaries = detect_row_boundaries(df, n_rows=n_rows)
    n_rows = len(boundaries)

    if row_idx < 0 or row_idx >= n_rows:
        sys.exit(f"Row index {row_idx} out of range [0, {n_rows - 1}].")

    row_start_pos = boundaries[row_idx]
    row_end_pos   = boundaries[row_idx + 1] if row_idx + 1 < n_rows else len(df)

    row_df = df.iloc[row_start_pos:row_end_pos]
    row_start_t = df["t"].iloc[row_start_pos]
    row_end_t   = df["t"].iloc[row_end_pos - 1]

    # find the actual integration region (where RD or RU are active)
    active = row_df[(row_df["RD"] == 1) | (row_df["RU"] == 1)]
    if len(active) > 0:
        t_int_start = active["t"].iloc[0]
        t_int_end   = active["t"].iloc[-1]
    else:
        t_int_start = row_start_t
        t_int_end   = row_end_t

    t_lo = t_int_start - window_us
    t_hi = t_int_end   + window_us

    df_view = df[(df["t"] >= t_lo) & (df["t"] <= t_hi)]

    c = _plot_colors()
    fig, ax = plt.subplots(figsize=(20, 5))

    t = df_view["t"].to_numpy()
    val = df_view["val"].to_numpy()
    ped_mask = df_view["RD"].to_numpy() == 1
    sig_mask = df_view["RU"].to_numpy() == 1

    ax.plot(t, val, color=c["adc"], lw=0.4,
            label=os.path.basename(title), rasterized=True)
    ax.plot(t[ped_mask], val[ped_mask], ".", ms=1.5, color=c["ped"],
            label="pedestal", zorder=3, rasterized=True)
    ax.plot(t[sig_mask], val[sig_mask], ".", ms=1.5, color=c["sig"],
            label="signal", zorder=3, rasterized=True)

    # shade the full row region (clipped to the view)
    ax.axvspan(max(row_start_t, t_lo), min(row_end_t, t_hi),
               alpha=0.08, color="grey", label=f"row {row_idx}")

    ax.set_xlabel(r"time ($\mu$s)")
    ax.set_ylabel("ADC value (ADU)")
    ax.set_title(f"Row {row_idx}  (integration: {t_int_start:.1f} – {t_int_end:.1f} µs)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_time_range(df: pd.DataFrame, t_min: float, t_max: float,
                    title: str = "ADC trace"):
    """Plot the trace between t_min and t_max (µs), with ped/signal overlays."""
    if t_min is None:
        t_min = df["t"].min()
    if t_max is None:
        t_max = df["t"].max()
    if t_min >= t_max:
        sys.exit(f"--tmin ({t_min}) must be < --tmax ({t_max}).")

    df_view = df[(df["t"] >= t_min) & (df["t"] <= t_max)]
    if len(df_view) == 0:
        sys.exit(f"No samples in [{t_min}, {t_max}] µs "
                 f"(file spans 0 – {df['t'].max():.1f} µs).")

    c = _plot_colors()
    fig, ax = plt.subplots(figsize=(20, 5))

    t = df_view["t"].to_numpy()
    val = df_view["val"].to_numpy()
    ped_mask = df_view["RD"].to_numpy() == 1
    sig_mask = df_view["RU"].to_numpy() == 1

    ax.plot(t, val, color=c["adc"], lw=0.4,
            label=os.path.basename(title), rasterized=True)
    ax.plot(t[ped_mask], val[ped_mask], ".", ms=1.5, color=c["ped"],
            label="pedestal", zorder=3, rasterized=True)
    ax.plot(t[sig_mask], val[sig_mask], ".", ms=1.5, color=c["sig"],
            label="signal", zorder=3, rasterized=True)

    ax.set_xlabel(r"time ($\mu$s)")
    ax.set_ylabel("ADC value (ADU)")
    ax.set_title(f"{t_min:.1f} – {t_max:.1f} µs")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def print_row_summary(df: pd.DataFrame, n_rows: int):
    """Print a table of detected rows with their time spans."""
    boundaries = detect_row_boundaries(df, n_rows=n_rows)
    n_rows = len(boundaries)
    print(f"\nDetected {n_rows} row(s):\n")
    print(f"{'row':>5}  {'t_start (µs)':>14}  {'t_end (µs)':>12}  {'duration (µs)':>14}  {'n_samples':>10}")
    print("-" * 65)
    for i, pos in enumerate(boundaries):
        end_pos = boundaries[i + 1] if i + 1 < n_rows else len(df)
        t_start = df["t"].iloc[pos]
        t_end   = df["t"].iloc[end_pos - 1]
        n_samp  = end_pos - pos
        print(f"{i:>5}  {t_start:>14.1f}  {t_end:>12.1f}  {t_end - t_start:>14.1f}  {n_samp:>10,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Binary → CSV → trace plot for CCD ADC data."
    )
    parser.add_argument("file", metavar="FILE",
                        help="Input file: .RJ45ADCDATA* binary or pre-converted .csv")

    parser.add_argument("--meta",     metavar="FILE", default=None,
                        help="JSON metadata file (only needed by the older acmpy "
                             "backend; auto-detected from FILE if omitted)")
    parser.add_argument("--save_csv", action="store_true",
                        help="Save decoded DataFrame to CSV alongside the binary")
    parser.add_argument("--csv_out",  metavar="FILE", default=None,
                        help="Explicit CSV output path (implies --save_csv)")

    parser.add_argument("--row",    type=int, default=None,
                        help="Plot only this row (0-indexed). Omit for full trace.")
    parser.add_argument("--nrow",   type=int, default=None,
                        help="Number of CCD rows in the trace. Read from .meta or "
                             "the filename if not given.")
    parser.add_argument("--window", type=float, default=1000.0,
                        help="Extra context (µs) on each side of the integration region [1000]")
    parser.add_argument("--tmin",   type=float, default=None,
                        help="Lower edge (µs) of plot window. Overrides --row if given.")
    parser.add_argument("--tmax",   type=float, default=None,
                        help="Upper edge (µs) of plot window. Overrides --row if given.")

    parser.add_argument("--list_rows", action="store_true",
                        help="Print detected row boundaries and exit")
    parser.add_argument("--save_plot", action="store_true",
                        help="Save plot to JPEG next to the source file")
    parser.add_argument("--no_show",   action="store_true",
                        help="Do not open interactive plot window")

    args = parser.parse_args()

    # ---- load data ---------------------------------------------------------
    title = args.file
    is_csv = args.file.lower().endswith(".csv")

    if is_csv:
        df = load_csv(args.file)
    elif args.save_csv or args.csv_out:
        outpath = binary_to_csv(args.file, args.meta, args.csv_out)
        df = load_csv(outpath)
    else:
        df = load_binary(args.file, args.meta)

    print(f"Loaded {len(df):,} samples  (~{df['t'].max():.1f} µs)")

    # ---- resolve NROW (needed for any row-based operation) -----------------
    needs_rows = args.list_rows or args.row is not None
    n_rows = resolve_nrow(args.file, args.nrow) if needs_rows else None

    # ---- row listing -------------------------------------------------------
    if args.list_rows:
        print_row_summary(df, n_rows)
        return

    # ---- plot --------------------------------------------------------------
    use_trange = args.tmin is not None or args.tmax is not None

    if use_trange and args.row is not None:
        print(f"WARNING: --tmin/--tmax given, ignoring --row {args.row}.",
              file=sys.stderr)

    if use_trange:
        fig, ax = plot_time_range(df, args.tmin, args.tmax, title)
        lo = args.tmin if args.tmin is not None else df['t'].min()
        hi = args.tmax if args.tmax is not None else df['t'].max()
        plot_label = f"_t{lo:.0f}-{hi:.0f}"
    elif args.row is not None:
        fig, ax = plot_row(df, args.row, n_rows, args.window, title)
        plot_label = f"_row{args.row}"
    else:
        fig, ax = plot_full_trace(df, title)
        plot_label = "_full"

    if args.save_plot:
        base = title.rsplit(".", 1)[0]
        jpg_path = base + plot_label + ".jpg"
        fig.savefig(jpg_path, dpi=350, bbox_inches="tight")
        print(f"Saved plot → {jpg_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

"""Plot every scalar chart from all wandb *offline* runs under a folder.

Reads each ``offline-run-*/run-*.wandb`` file directly via the wandb SDK's
LevelDB-log datastore (no cloud sync required) and dumps one matplotlib
figure per run (subplot per metric), plus an optional combined figure that
overlays all runs.

Style mirrors ``hw3/src/utils/visualization.py``: clean light-grey axes,
dashed grid, no top/right spines, bold axis labels, 300 DPI export. Titles
show only the metric name (no run id).

Usage:
    /home/longpm/miniconda3/envs/hw3/bin/python plot_wandb.py \
        --wandb-dir wandb --output-dir wandb_plots --combined --smooth 5
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

# --------------------------------------------------------------------------
#  Matplotlib style (same look-and-feel as hw3/src/utils/visualization.py)
# --------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "#F9F9F9",
})

_TAB_COLORS = list(plt.get_cmap("tab10").colors)


# --------------------------------------------------------------------------
#  wandb offline-log parsing
# --------------------------------------------------------------------------
# Suffixes that mark image / media metadata sub-fields, not training scalars.
_MEDIA_SUFFIXES = ("/_type", "/sha256", "/path", "/format", "/size",
                   "/width", "/height", "/caption", "/digest")


def _is_metric(key):
    """True if ``key`` looks like a user-logged scalar metric."""
    if not key or key.startswith("_"):
        return False
    if any(key.endswith(suf) for suf in _MEDIA_SUFFIXES):
        return False
    return True


def _item_key(item):
    """HistoryItem keys live in either ``key`` or ``nested_key`` (joined with '/')."""
    if item.key:
        return item.key
    if item.nested_key:
        return "/".join(item.nested_key)
    return ""


def read_history(wandb_file):
    """Return ``(run_name, {metric: [(step, value), ...]})`` from a .wandb file."""
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))
    run_name = None
    history = defaultdict(list)
    try:
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = pb.Record()
            rec.ParseFromString(data)
            which = rec.WhichOneof("record_type")
            if which == "run":
                run_name = rec.run.display_name or rec.run.run_id
            elif which == "history":
                step = None
                scalars = {}
                for item in rec.history.item:
                    key = _item_key(item)
                    try:
                        val = json.loads(item.value_json)
                    except Exception:
                        continue
                    if key == "_step":
                        step = val
                    elif isinstance(val, bool):
                        continue
                    elif isinstance(val, (int, float)) and _is_metric(key):
                        scalars[key] = val
                for k, v in scalars.items():
                    history[k].append((step, v))
    finally:
        ds.close()
    return run_name, history


def discover_runs(wandb_dir):
    """Find every ``*-run-*/<run>.wandb`` under ``wandb_dir``."""
    out = []
    for d in sorted(glob.glob(os.path.join(wandb_dir, "*run-*"))):
        if not os.path.isdir(d) or os.path.islink(d):
            continue
        wfs = glob.glob(os.path.join(d, "*.wandb"))
        if wfs:
            out.append((os.path.basename(d), wfs[0]))
    return out


# --------------------------------------------------------------------------
#  Plot helpers
# --------------------------------------------------------------------------
def smooth(values, window):
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)
    arr = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _grid(n):
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    return rows, cols


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pretty_name(key):
    """``train_loss_step`` -> ``Train Loss``."""
    name = key.replace("_step", "").replace("_epoch", "").replace("/", " / ")
    return name.replace("_", " ").strip().title()


def _y_label(key):
    """Guess a nice y-axis label from the metric name."""
    k = key.lower()
    if "psnr" in k:
        return "PSNR (dB)"
    if "ssim" in k:
        return "SSIM"
    if "loss" in k:
        return "Loss"
    if any(s in k for s in ("acc", "ap", "map", "score", "iou")):
        return "Score"
    if k == "lr" or k.endswith("/lr") or "learning_rate" in k:
        return "Learning Rate"
    return "Value"


def _plot_series(ax, xs, ys, color, label=None, smooth_window=1):
    """Plot a single (xs, ys) line with the project's house style."""
    use_markers = len(xs) <= 200
    raw_alpha = 0.35 if smooth_window > 1 else 1.0
    raw_kw = dict(linewidth=2.0, color=color, alpha=raw_alpha)
    if use_markers:
        raw_kw.update(marker="o", markersize=3)
    ax.plot(xs, ys, **raw_kw,
            label=None if smooth_window > 1 else label)
    if smooth_window > 1:
        ys_smooth = smooth(ys, smooth_window)
        ax.plot(xs, ys_smooth, linewidth=2.0, color=color, label=label)


# --------------------------------------------------------------------------
#  Per-run figure (one subplot per metric)
# --------------------------------------------------------------------------
def plot_run(run_name, run_dir, history, out_dir, smooth_window=1,
             suptitle=True):
    metrics = sorted(history.keys())
    if not metrics:
        print(f"  {run_dir}: no scalar metrics")
        return
    rows, cols = _grid(len(metrics))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(6 * cols, 4.2 * rows), squeeze=False)
    for i, metric in enumerate(metrics):
        ax = axes[i // cols][i % cols]
        pts = sorted((s, v) for s, v in history[metric] if s is not None)
        if not pts:
            ax.set_visible(False)
            continue
        xs, ys = zip(*pts)
        color = _TAB_COLORS[i % len(_TAB_COLORS)]
        _plot_series(ax, xs, ys, color, label=_pretty_name(metric),
                     smooth_window=smooth_window)
        ax.set_title(_pretty_name(metric), fontweight="bold")
        ax.set_xlabel("Step", fontweight="bold")
        ax.set_ylabel(_y_label(metric), fontweight="bold")
        ax.legend(loc="best")

    for i in range(len(metrics), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    if run_name and suptitle:
        fig.suptitle(run_name, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    safe_run_name = (run_name or "run").replace("/", "_")
    out = os.path.join(out_dir, f"{safe_run_name}__{run_dir}.png")
    _save(fig, out)
    print(f"  wrote {out}  ({len(metrics)} metrics)")


# --------------------------------------------------------------------------
#  Combined figure (one subplot per metric, all runs overlaid)
# --------------------------------------------------------------------------
def plot_combined(runs, out_dir, smooth_window=1, comparison_tag="All Runs",
                  suptitle=True):
    """``runs`` is a list of ``(legend_label, history)``."""
    all_metrics = sorted({m for _, h in runs for m in h.keys()})
    if not all_metrics:
        return
    rows, cols = _grid(len(all_metrics))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(6 * cols, 4.2 * rows), squeeze=False)
    for i, metric in enumerate(all_metrics):
        ax = axes[i // cols][i % cols]
        any_plotted = False
        for j, (label, h) in enumerate(runs):
            pts = sorted((s, v) for s, v in h.get(metric, []) if s is not None)
            if not pts:
                continue
            xs, ys = zip(*pts)
            color = _TAB_COLORS[j % len(_TAB_COLORS)]
            _plot_series(ax, xs, ys, color, label=label,
                         smooth_window=smooth_window)
            any_plotted = True
        if not any_plotted:
            ax.set_visible(False)
            continue
        ax.set_title(_pretty_name(metric), fontweight="bold")
        ax.set_xlabel("Step", fontweight="bold")
        ax.set_ylabel(_y_label(metric), fontweight="bold")
        # only the first axis carries the legend to avoid clutter
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    for i in range(len(all_metrics), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    if comparison_tag and suptitle:
        fig.suptitle(comparison_tag, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(out_dir, "_all_runs.png")
    _save(fig, out)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
#  CLI
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Plot all charts from offline wandb runs.")
    p.add_argument("--wandb-dir", default="wandb",
                   help="Folder containing offline-run-* subfolders.")
    p.add_argument("--output-dir", default="wandb_plots",
                   help="Where to write the PNG figures.")
    p.add_argument("--smooth", type=int, default=1,
                   help="Moving-average window over y values (1 = no smoothing).")
    p.add_argument("--combined", action="store_true",
                   help="Also produce a single figure overlaying all runs.")
    p.add_argument("--runs", nargs="*",
                   help="Optional list of run folder names to include (default: all).")
    p.add_argument("--comparison-tag", default="All Runs",
                   help="Suptitle for the combined comparison figure.")
    p.add_argument("--no-suptitle", action="store_true",
                   help="Drop the big figure title; keep only per-metric titles.")
    p.add_argument("--name", default=None,
                   help="Override run display name (used for titles and the "
                        "filename prefix) instead of the path-like wandb name.")
    args = p.parse_args()
    suptitle = not args.no_suptitle

    os.makedirs(args.output_dir, exist_ok=True)
    runs = discover_runs(args.wandb_dir)
    if args.runs:
        runs = [(d, f) for d, f in runs if d in args.runs]
    if not runs:
        raise SystemExit(f"no .wandb files found under {args.wandb_dir}")

    print(f"found {len(runs)} run(s) in {args.wandb_dir}")
    parsed = []
    for run_dir, wfile in runs:
        print(f"reading {run_dir}")
        try:
            run_name, history = read_history(wfile)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        display_name = args.name or run_name
        plot_run(display_name, run_dir, history, args.output_dir, args.smooth,
                 suptitle=suptitle)
        parsed.append((display_name or run_dir, history))

    if args.combined and parsed:
        plot_combined(parsed, args.output_dir, args.smooth,
                      comparison_tag=args.comparison_tag, suptitle=suptitle)


if __name__ == "__main__":
    main()

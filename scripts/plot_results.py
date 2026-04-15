#!/usr/bin/env python3
"""
plot_results.py — Generate benchmark plots from benchmark_results.csv

Produces:
  results/plots/speedup.png             Speedup vs sequential at each grid size
  results/plots/absolute_time.png       Absolute ms per simulation at each grid size
  results/plots/version_progression.png Bar chart: time reduction version-over-version

Usage:  python3 scripts/plot_results.py [path/to/benchmark_results.csv]
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/metrics/benchmark_results.csv"
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

VERSION_ORDER  = ["seq", "v1", "v2", "v3", "v4"]
VERSION_LABELS = {
    "seq": "V0: Sequential",
    "v1":  "V1: Naive CUDA (AoS)",
    "v2":  "V2: SoA Layout",
    "v3":  "V3: Shared Mem Tiling",
    "v4":  "V4: Optimized (Fused)",
}
VERSION_COLORS = {
    "seq": "#888888",
    "v1":  "#e74c3c",
    "v2":  "#e67e22",
    "v3":  "#2ecc71",
    "v4":  "#3498db",
}

df = pd.read_csv(CSV_PATH)
df["size_label"] = df["grid_w"].astype(str) + "x" + df["grid_h"].astype(str)

medians = (df.groupby(["version", "grid_w", "grid_h", "size_label"])["elapsed_ms"]
             .median()
             .reset_index()
             .rename(columns={"elapsed_ms": "median_ms"}))

sizes = sorted(medians["size_label"].unique(),
               key=lambda s: int(s.split("x")[0]))

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Speedup vs Sequential
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

seq_times = medians[medians["version"] == "seq"].set_index("size_label")["median_ms"]

for ver in VERSION_ORDER:
    if ver == "seq":
        continue
    sub = medians[medians["version"] == ver].set_index("size_label")
    speedups = []
    x_labels = []
    for s in sizes:
        if s in sub.index and s in seq_times.index:
            speedups.append(seq_times[s] / sub.loc[s, "median_ms"])
            x_labels.append(s)
    if speedups:
        ax.plot(x_labels, speedups, "o-", color=VERSION_COLORS[ver],
                label=VERSION_LABELS[ver], linewidth=2, markersize=8)

ax.set_xlabel("Grid Size (particles)", fontsize=13)
ax.set_ylabel("Speedup over Sequential", fontsize=13)
ax.set_title("GPU Speedup vs CPU Sequential Baseline", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "speedup.png"), dpi=150)
plt.close(fig)
print(f"  -> {PLOT_DIR}/speedup.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Absolute Time (log scale)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

for ver in VERSION_ORDER:
    sub = medians[medians["version"] == ver].set_index("size_label")
    times = []
    x_labels = []
    for s in sizes:
        if s in sub.index:
            times.append(sub.loc[s, "median_ms"])
            x_labels.append(s)
    if times:
        ax.plot(x_labels, times, "o-", color=VERSION_COLORS[ver],
                label=VERSION_LABELS[ver], linewidth=2, markersize=8)

ax.set_yscale("log")
ax.set_xlabel("Grid Size", fontsize=13)
ax.set_ylabel("Total Simulation Time (ms, log scale)", fontsize=13)
ax.set_title("Absolute Runtime — All Versions", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which="both")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "absolute_time.png"), dpi=150)
plt.close(fig)
print(f"  -> {PLOT_DIR}/absolute_time.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Version Progression (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

versions_present = [v for v in VERSION_ORDER if v in medians["version"].values]
n_ver = len(versions_present)
n_sizes = len(sizes)
bar_width = 0.8 / n_ver
x_pos = np.arange(n_sizes)

for j, ver in enumerate(versions_present):
    sub = medians[medians["version"] == ver].set_index("size_label")
    vals = [sub.loc[s, "median_ms"] if s in sub.index else 0 for s in sizes]
    offset = (j - n_ver / 2 + 0.5) * bar_width
    bars = ax.bar(x_pos + offset, vals, bar_width,
                  color=VERSION_COLORS[ver], label=VERSION_LABELS[ver], alpha=0.85)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x_pos)
ax.set_xticklabels(sizes, fontsize=12)
ax.set_xlabel("Grid Size", fontsize=13)
ax.set_ylabel("Total Simulation Time (ms)", fontsize=13)
ax.set_title("Version Progression — Runtime at Each Grid Size", fontsize=15)
ax.legend(fontsize=10, ncol=2)
ax.set_yscale("log")
ax.grid(True, alpha=0.3, axis="y", which="both")

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "version_progression.png"), dpi=150)
plt.close(fig)
print(f"  -> {PLOT_DIR}/version_progression.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Print summary table to stdout
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Benchmark Summary (median ms) ===")
pivot = medians.pivot_table(values="median_ms", index="version", columns="size_label")
pivot = pivot.reindex(index=VERSION_ORDER, columns=sizes)
print(pivot.to_string(float_format="{:.2f}".format))

if "seq" in pivot.index:
    print("\n=== Speedup over Sequential ===")
    seq_row = pivot.loc["seq"]
    for ver in VERSION_ORDER:
        if ver == "seq" or ver not in pivot.index:
            continue
        speedup = seq_row / pivot.loc[ver]
        print(f"  {VERSION_LABELS[ver]:30s}  ", end="")
        for s in sizes:
            if s in speedup.index and not np.isnan(speedup[s]):
                print(f"{s}: {speedup[s]:.2f}x   ", end="")
        print()

print(f"\nPlots saved to {PLOT_DIR}/")

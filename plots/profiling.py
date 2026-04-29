import matplotlib.pyplot as plt
import numpy as np

tiles            = ["8×8", "16×16", "32×8", "32×16"]
shared_mem_bytes = [1728, 4800, 5184, 8640]

force_avg      = [43_589.7, 41_856.5, 40_871.6, 41_890.6]
integrate_avg  = [38_989.9, 39_103.4, 39_046.5, 39_088.0]
force_total    = [52_351_281, 50_269_667, 49_086_817, 50_310_648]
integrate_total = [46_826_842, 46_963_165, 46_894_852, 46_944_628]

cuda_api = {
    "cudaMemcpyToSymbol": [105_423_777, 61_700_056, 56_904_954, 57_073_496],
    "cudaLaunchKernel":   [ 57_585_640, 56_412_829, 55_642_424, 56_314_240],
    "cudaEventSynchronize": [42_795_134, 41_988_536, 41_583_412, 42_102_495],
    "cudaMemcpy (HtoD/DtoH)": [2_489_746, 2_529_868, 2_451_976, 2_440_754],
}

api_colors    = ["#e07b54", "#5b8db8", "#6bbf6a", "#b07fc7"]
kernel_colors = ["#5b8db8", "#e07b54"]
x     = np.arange(len(tiles))
bar_w = 0.35

# ── 1. CUDA API time breakdown (raw ms, stacked bar) ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
bottoms = np.zeros(len(tiles))
for (label, vals), color in zip(cuda_api.items(), api_colors):
    vals_ms = [v / 1e6 for v in vals]
    bars = ax.bar(tiles, vals_ms, bottom=bottoms, label=label, color=color, width=0.5)
    for j, (b, v) in enumerate(zip(bars, vals_ms)):
        if v > 2:
            ax.text(b.get_x() + b.get_width() / 2,
                    bottoms[j] + v / 2,
                    f"{v:.1f} ms", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
    bottoms += np.array(vals_ms)

ax.set_title("CUDA API Time Breakdown — Raw Runtime (ms)", fontsize=13, fontweight="bold")
ax.set_ylabel("Time (ms)")
ax.set_xlabel("Tile Configuration")
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/nsys_cuda_api_breakdown.png", dpi=200)
plt.close()
print("Saved nsys_cuda_api_breakdown.png")

# ── 2. Shared memory size vs kernel latency ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(shared_mem_bytes, [v / 1e3 for v in force_avg],
        marker="o", linewidth=2, color=kernel_colors[0], label="fused_force_kernel")
ax.plot(shared_mem_bytes, [v / 1e3 for v in integrate_avg],
        marker="s", linewidth=2, color=kernel_colors[1], label="fused_integrate_collision_kernel")

for sm, f, it, t in zip(shared_mem_bytes, force_avg, integrate_avg, tiles):
    ax.annotate(t, (sm, f / 1e3),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, color=kernel_colors[0])
    ax.annotate(t, (sm, it / 1e3),
                textcoords="offset points", xytext=(6, -10),
                fontsize=9, color=kernel_colors[1])

ax.set_title("Shared Memory Size vs Avg Kernel Latency", fontsize=13, fontweight="bold")
ax.set_xlabel("Shared Memory per Block (bytes)")
ax.set_ylabel("Avg Latency per Call (µs)")
ax.legend(fontsize=9)
ax.grid(linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/nsys_shared_mem_vs_latency.png", dpi=200)
plt.close()
print("Saved nsys_shared_mem_vs_latency.png")

# ── 3. GPU kernel total time split (stacked bar) ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ft_ms = [v / 1e6 for v in force_total]
it_ms = [v / 1e6 for v in integrate_total]
b1 = ax.bar(tiles, ft_ms, label="fused_force_kernel",               color=kernel_colors[0], width=0.5)
b2 = ax.bar(tiles, it_ms, label="fused_integrate_collision_kernel",  color=kernel_colors[1], width=0.5, bottom=ft_ms)

for i, (f, it) in enumerate(zip(ft_ms, it_ms)):
    ax.text(i, f / 2,      f"{f:.1f} ms", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    ax.text(i, f + it / 2, f"{it:.1f} ms", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    total = f + it
    pct_f  = f  / total * 100
    pct_it = it / total * 100
    ax.text(i, total + 0.5, f"{pct_f:.1f}% / {pct_it:.1f}%",
            ha="center", fontsize=8, color="#333333")

ax.set_title("GPU Kernel Total Time Split (ms)\n1,201 calls each", fontsize=13, fontweight="bold")
ax.set_ylabel("Total GPU Time (ms)")
ax.set_xlabel("Tile Configuration")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/nsys_kernel_time_split.png", dpi=200)
plt.close()
print("Saved nsys_kernel_time_split.png")

# ── 4. Avg kernel latency per call (grouped bar) ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - bar_w / 2, [v / 1e3 for v in force_avg],
       bar_w, label="fused_force_kernel",               color=kernel_colors[0])
ax.bar(x + bar_w / 2, [v / 1e3 for v in integrate_avg],
       bar_w, label="fused_integrate_collision_kernel",  color=kernel_colors[1])

for i, (f, it) in enumerate(zip(force_avg, integrate_avg)):
    ax.text(i - bar_w / 2, f / 1e3 + 0.15, f"{f/1e3:.2f}", ha="center", fontsize=8.5)
    ax.text(i + bar_w / 2, it / 1e3 + 0.15, f"{it/1e3:.2f}", ha="center", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(tiles)
ax.set_title("Avg Kernel Latency per Call (µs)", fontsize=13, fontweight="bold")
ax.set_ylabel("Avg Latency (µs)")
ax.set_xlabel("Tile Configuration")
ax.set_ylim(0, max(force_avg) / 1e3 * 1.2)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/nsys_kernel_avg_latency.png", dpi=200)
plt.close()
print("Saved nsys_kernel_avg_latency.png")
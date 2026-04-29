import matplotlib.pyplot as plt

drop_heights = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
times_ms     = [362.275, 361.218, 359.612, 359.203, 358.114, 357.215, 354.573]

plt.figure(figsize=(9, 5))

plt.plot(drop_heights, times_ms, marker="o", linewidth=2, color="steelblue")

for x, y in zip(drop_heights, times_ms):
    plt.annotate(
        f"{y:.3f} ms",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        color="steelblue",
    )

plt.xlabel("Drop Height (m)")
plt.ylabel("Runtime (ms)")
plt.title("V4 Runtime Sensitivity to Drop Height")

plt.xticks(drop_heights)
ypad = (max(times_ms) - min(times_ms)) * 2
plt.ylim(min(times_ms) - ypad, max(times_ms) + ypad * 3)

plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig("plots/v4_sensitivity_drop_height.png", dpi=300)
plt.close()

print("Saved to plots/v4_sensitivity_drop_height.png")
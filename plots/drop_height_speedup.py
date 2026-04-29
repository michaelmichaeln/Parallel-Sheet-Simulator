import matplotlib.pyplot as plt

drop_heights = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
speedups = [128.80, 129.84, 130.67, 176.59, 203.74, 206.10, 206.30]

plt.figure(figsize=(10, 6))

plt.plot(drop_heights, speedups, marker="o", linewidth=2, color="steelblue", label="V4 vs Sequential")

for x, y in zip(drop_heights, speedups):
    plt.annotate(
        f"{y:.2f}x",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        color="steelblue",
    )

plt.xlabel("Drop Height (m)")
plt.ylabel("Speedup")
plt.title("V4 Speedup vs Sequential by Drop Height")

plt.xticks(drop_heights)
plt.ylim(0, max(speedups) * 1.15)

plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig("plots/speedup_vs_drop_height.png", dpi=300)
plt.close()

print("Saved to plots/speedup_vs_drop_height.png")
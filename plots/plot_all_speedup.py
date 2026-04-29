import csv
import matplotlib.pyplot as plt

sizes = []
speedups = {
    "V1": [],
    "V2": [],
    "V3": [],
    "V4": [],
}

with open("results/speedup_vs_size.csv") as f:
    reader = csv.DictReader(f)

    for row in reader:
        try:
            size = int(row["size"])
            v0 = float(row["v0_ms"])
            times = {
                "V1": float(row["v1_ms"]),
                "V2": float(row["v2_ms"]),
                "V3": float(row["v3_ms"]),
                "V4": float(row["v4_ms"]),
            }
        except:
            continue

        sizes.append(size)
        for version in speedups:
            speedups[version].append(v0 / times[version])

colors = {"V1": "steelblue", "V2": "tomato", "V3": "seagreen", "V4": "mediumpurple"}

plt.figure(figsize=(10, 6))

for version, vals in speedups.items():
    plt.plot(sizes, vals, marker="o", linewidth=2, color=colors[version], label=version)

    for x, y in zip(sizes, vals):
        plt.annotate(
            f"{y:.2f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=colors[version],
        )

plt.xlabel("Cloth Size (N x N)")
plt.ylabel("Speedup vs Sequential")
plt.title("Speedup vs Cloth Size (All Versions)")

plt.xticks(sizes)
plt.ylim(0, max(v for vals in speedups.values() for v in vals) * 1.15)

plt.legend(title="Version")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig("plots/speedup_vs_size_all.png", dpi=300)
plt.close()

print("Saved combined plot to plots/speedup_vs_size_all.png")
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

for version, vals in speedups.items():
    plt.figure(figsize=(8, 5))

    plt.plot(sizes, vals, marker="o", linewidth=2)

    # label each point
    for x, y in zip(sizes, vals):
        plt.annotate(
            f"{x}x{x}\n{y:.2f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    plt.xlabel("Cloth Size (N x N)")
    plt.ylabel("Speedup vs Sequential")
    plt.title(f"{version} Speedup vs Cloth Size")

    plt.xticks(sizes)

    # ✅ FORCE y-axis to start at 0
    plt.ylim(0, max(vals) * 1.15)

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    outpath = f"results/plots/speedup_vs_size_{version.lower()}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()

print("Overwritten plots in results/plots/")
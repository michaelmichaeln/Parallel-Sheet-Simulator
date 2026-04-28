import csv
import matplotlib.pyplot as plt

sizes = []
runtimes = {
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
            t1 = float(row["v1_ms"])
            t2 = float(row["v2_ms"])
            t3 = float(row["v3_ms"])
            t4 = float(row["v4_ms"])
        except:
            continue

        sizes.append(size)
        runtimes["V1"].append(t1)
        runtimes["V2"].append(t2)
        runtimes["V3"].append(t3)
        runtimes["V4"].append(t4)

for version, vals in runtimes.items():
    plt.figure(figsize=(8, 5))

    plt.plot(sizes, vals, marker="o", linewidth=2)

    # label each point
    for x, y in zip(sizes, vals):
        plt.annotate(
            f"{x}x{x}\n{y:.1f} ms",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    plt.xlabel("Cloth Size (N x N)")
    plt.ylabel("Runtime (ms)")
    plt.title(f"{version} Runtime vs Cloth Size")

    plt.xticks(sizes)

    # force y-axis from 0
    plt.ylim(0, max(vals) * 1.15)

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    outpath = f"results/plots/runtime_vs_size_{version.lower()}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()

print("Runtime plots saved to results/plots/")
import matplotlib.pyplot as plt

sphere_radii = [0.2, 0.4, 0.6, 0.8, 1.0]
speedups     = [140.44, 138.56, 139.43, 139.82, 139.66]

plt.figure(figsize=(9, 5))

plt.plot(sphere_radii, speedups, marker="o", linewidth=2, color="steelblue", label="V4 vs Sequential")

for x, y in zip(sphere_radii, speedups):
    plt.annotate(
        f"{y:.2f}x",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        color="steelblue",
    )

plt.xlabel("Sphere Radius (m)")
plt.ylabel("Speedup")
plt.title("V4 Speedup vs Sequential by Sphere Radius")

plt.xticks(sphere_radii)
plt.ylim(0, max(speedups) * 1.15)

plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig("plots/speedup_vs_sphere_radius.png", dpi=300)
plt.close()

print("Saved to plots/speedup_vs_sphere_radius.png")
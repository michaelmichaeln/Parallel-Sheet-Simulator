import matplotlib.pyplot as plt

sphere_radii = [0.2, 0.4, 0.6, 0.8, 1.0]
times_ms     = [98.513, 98.573, 98.783, 99.306, 99.640]

plt.figure(figsize=(9, 5))

plt.plot(sphere_radii, times_ms, marker="o", linewidth=2, color="steelblue")

for x, y in zip(sphere_radii, times_ms):
    plt.annotate(
        f"{y:.3f} ms",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=9,
        color="steelblue",
    )

plt.xlabel("Sphere Radius (m)")
plt.ylabel("Runtime (ms)")
plt.title("V4 Runtime Sensitivity to Sphere Radius")

plt.xticks(sphere_radii)
ypad = (max(times_ms) - min(times_ms)) * 2
plt.ylim(min(times_ms) - ypad, max(times_ms) + ypad * 3)

plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig("plots/v4_sensitivity_sphere_radius.png", dpi=300)
plt.close()

print("Saved to plots/v4_sensitivity_sphere_radius.png")
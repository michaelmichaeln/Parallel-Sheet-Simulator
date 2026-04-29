import matplotlib.pyplot as plt
import numpy as np

data = [
    (8,  8,  353.661),
    (8,  16, 345.589),
    (8,  32, 351.607),
    (16, 8,  343.421),
    (16, 16, 347.356),
    (16, 32, 356.928),
    (32, 8,  339.881),
    (32, 16, 346.166),
    (32, 32, 381.962),
]

tile_ws = sorted(set(r[0] for r in data))
tile_hs = sorted(set(r[1] for r in data))

times = {(w, h): t for w, h, t in data}
worst = max(times.values())

speedup_grid = np.array([
    [worst / times[(w, h)] for w in tile_ws]
    for h in tile_hs
])

fig, ax = plt.subplots(figsize=(7, 5))

im = ax.imshow(speedup_grid, cmap="YlGnBu", aspect="auto")

ax.set_xticks(range(len(tile_ws)))
ax.set_xticklabels(tile_ws)
ax.set_yticks(range(len(tile_hs)))
ax.set_yticklabels(tile_hs)
ax.set_xlabel("Tile Width")
ax.set_ylabel("Tile Height")
ax.set_title("V4 Speedup vs Tile Size\n(relative to slowest config: 32×32)")

for i, h in enumerate(tile_hs):
    for j, w in enumerate(tile_ws):
        t = times[(w, h)]
        s = worst / t
        ax.text(j, i, f"{s:.3f}x\n({t:.1f} ms)", ha="center", va="center", fontsize=9)

plt.colorbar(im, ax=ax, label="Relative Speedup")
plt.tight_layout()

plt.savefig("plots/speedup_vs_tile_size.png", dpi=300)
plt.close()

print("Saved to plots/speedup_vs_tile_size.png")
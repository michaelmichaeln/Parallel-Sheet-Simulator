"""
visualize_cloth.py
Reads cloth_frames.csv produced by cloth_sim_sequential.cpp
and renders an animated 3-D wireframe of the cloth.

Requirements:  pip install matplotlib pandas numpy
Run:           python visualize_cloth.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("cloth_frames.csv")
steps     = sorted(df["step"].unique())
GRID_H    = df["row"].max() + 1
GRID_W    = df["col"].max() + 1

def get_grid(step_num):
    """Return (X, Y, Z) meshgrids for a given step."""
    frame = df[df["step"] == step_num].sort_values(["row", "col"])
    X = frame["x"].values.reshape(GRID_H, GRID_W)
    Y = frame["y"].values.reshape(GRID_H, GRID_W)
    Z = frame["z"].values.reshape(GRID_H, GRID_W)
    return X, Y, Z

# ── Plot setup ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection="3d")

X0, Y0, Z0 = get_grid(steps[0])
wire = [ax.plot_wireframe(X0, Y0, Z0, rstride=1, cstride=1,
                          color="steelblue", linewidth=0.6, alpha=0.8)]

# Ground plane (semi-transparent)
GROUND_Y = -1.0
gx = np.linspace(-1.2, 1.2, 2)
gz = np.linspace(-1.2, 1.2, 2)
GX, GZ = np.meshgrid(gx, gz)
GY      = np.full_like(GX, GROUND_Y)
ax.plot_surface(GX, GY, GZ, alpha=0.15, color="sandybrown")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 2.0); ax.set_zlim(-1.2, 1.2)
title = ax.set_title(f"Step {steps[0]}")

# ── Animation update ─────────────────────────────────────────────────────────
def update(frame_idx):
    wire[0].remove()
    X, Y, Z   = get_grid(steps[frame_idx])
    wire[0]   = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1,
                                  color="steelblue", linewidth=0.6, alpha=0.8)
    title.set_text(f"Step {steps[frame_idx]}")
    return wire[0], title

ani = animation.FuncAnimation(fig, update, frames=len(steps),
                               interval=60, blit=False)

plt.tight_layout()
ani.save("cloth_sim.gif", writer="pillow", fps=20)
print("Saved cloth_sim.gif")

# Optionally save as GIF:
# ani.save("cloth_sim.gif", writer="pillow", fps=20)
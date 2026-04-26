"""
Visualize cloth simulation frames as a GIF.

Requirements: pip install matplotlib pandas numpy pillow
Run:
  python scripts/visualize.py
  python scripts/visualize.py --frames results/outputs/cloth_frames.csv --gif results/outputs/cloth_sim.gif
"""

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Scene constants — must match simulation parameters
GROUND_Y   = -1.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render cloth simulation frames to GIF.")
    parser.add_argument("--frames", default="results/outputs/cloth_frames.csv")
    parser.add_argument("--gif",    default="results/outputs/cloth_sim.gif")
    parser.add_argument("--fps",    type=int,   default=20)
    parser.add_argument("--dpi",    type=int,   default=100)
    parser.add_argument("--show", action="store_true",
                    help="Display animation instead of saving GIF")
    parser.add_argument("--every",  type=int,   default=1,
                        help="Only render every Nth saved step (use 2+ to speed up large grids)")
    parser.add_argument("--sphere_r", type=float, default=0.6)
    return parser.parse_args()


def draw_sphere(ax, SPHERE_R, SPHERE_CX, SPHERE_CY, SPHERE_CZ):
    """Draw collision sphere using visualization axes:
    X = horizontal, Y = north-south/depth, Z = height.
    Simulation coordinates are x, y(height), z(depth),
    so we plot as x, z, y.
    """
    u = np.linspace(0, np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)

    sim_x = SPHERE_R * np.outer(np.sin(u), np.cos(v)) + SPHERE_CX
    sim_y = SPHERE_R * np.outer(np.cos(u), np.ones_like(v)) + SPHERE_CY
    sim_z = SPHERE_R * np.outer(np.sin(u), np.sin(v)) + SPHERE_CZ

    ax.plot_surface(
        sim_x,
        sim_z,
        sim_y,
        color="coral",
        alpha=0.55,
        linewidth=0,
        zorder=1,
    )


def draw_ground(ax):
    """Draw ground plane using Z as height."""
    gx = np.linspace(-1.6, 1.6, 2)
    gy = np.linspace(-1.6, 1.6, 2)
    gx_m, gy_m = np.meshgrid(gx, gy)
    gz_m = np.full_like(gx_m, GROUND_Y)

    ax.plot_surface(
        gx_m,
        gy_m,
        gz_m,
        alpha=0.12,
        color="sandybrown",
        linewidth=0,
    )


def main() -> int:
    args = parse_args()
    GROUND_Y = args.ground_y if hasattr(args, "ground_y") else -1.0

    SPHERE_R = args.sphere_r
    SPHERE_CX = 0.0
    SPHERE_CZ = 0.0
    SPHERE_CY = GROUND_Y + SPHERE_R

    frames_path = Path(args.frames)
    gif_path    = Path(args.gif)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames_path.exists():
        print(f"Error: Frames file not found: {frames_path}")
        return 1

    print(f"Loading {frames_path} ...")
    df = pd.read_csv(frames_path)

    steps  = sorted(df["step"].unique())[::args.every]
    grid_h = int(df["row"].max()) + 1
    grid_w = int(df["col"].max()) + 1
    print(f"Grid {grid_w}x{grid_h}  |  {len(steps)} frames to render")

    # Adaptive wire stride: target ~20 wires per axis so cloth looks like cloth
    stride = max(1, min(grid_w, grid_h) // 20)
    print(f"Wire stride: {stride}  (drawing every {stride}th grid line)")

    # Pre-load all frames into RAM — much faster than filtering in the update loop
    print("Pre-loading frames ...")
    all_x, all_y, all_z = [], [], []
    for step in steps:
        frame = df[df["step"] == step].sort_values(["row", "col"])
        all_x.append(frame["x"].values.reshape(grid_h, grid_w))
        all_y.append(frame["y"].values.reshape(grid_h, grid_w))
        all_z.append(frame["z"].values.reshape(grid_h, grid_w))

    # Height-based colour for cloth (low = cooler blue, high = warm gold)
    y_min_global = min(y.min() for y in all_y)
    y_max_global = max(y.max() for y in all_y)
    cmap = plt.cm.coolwarm_r

    def cloth_color(y_grid):
        norm = (y_grid - y_min_global) / max(y_max_global - y_min_global, 1e-6)
        return cmap(norm)

    # --- Figure setup ---
    fig = plt.figure(figsize=(9, 7), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")

    ax.set_xlim(-1.6, 1.6)
    ax.set_zlim(GROUND_Y - 0.1, 3.2)
    ax.set_ylim(-1.6, 1.6)
    ax.set_box_aspect((3.2, 3.2, 4.3))

    ax.set_xlabel("X", color="white", labelpad=4)
    ax.set_ylabel("Y / north-south", color="white", labelpad=4)
    ax.set_zlabel("Z / height", color="white", labelpad=4)
    ax.tick_params(colors="white", labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333355")

    ax.view_init(elev=30, azim=-60)

    draw_ground(ax)
    draw_sphere(ax, SPHERE_R, SPHERE_CX, SPHERE_CY, SPHERE_CZ)

    # Initial cloth surface (height-coloured) + wireframe overlay
    surf  = [ax.plot_surface(all_x[0], all_z[0], all_y[0],
                              facecolors=cloth_color(all_y[0]),
                              rstride=stride, cstride=stride,
                              linewidth=0, antialiased=False, alpha=0.85, zorder=2)]
    wire  = [ax.plot_wireframe(all_x[0], all_z[0], all_y[0],
                                rstride=stride, cstride=stride,
                                color="white", linewidth=0.25, alpha=0.25, zorder=3)]

    sim_dt  = 0.005
    title   = ax.set_title(
        f"V4 CUDA  |  {grid_w}×{grid_h}  |  t = {steps[0] * sim_dt:.2f} s",
        color="white", fontsize=11, pad=8)

    plt.tight_layout()

    def update(i):
        surf[0].remove()
        wire[0].remove()
        surf[0] = ax.plot_surface(all_x[i], all_z[i], all_y[i],
                                   facecolors=cloth_color(all_y[i]),
                                   rstride=stride, cstride=stride,
                                   linewidth=0, antialiased=False, alpha=0.85, zorder=2)
        wire[0] = ax.plot_wireframe(all_x[i], all_z[i], all_y[i],
                                     rstride=stride, cstride=stride,
                                     color="white", linewidth=0.25, alpha=0.25, zorder=3)
        title.set_text(
            f"V4 CUDA  |  {grid_w}×{grid_h}  |  t = {steps[i] * sim_dt:.2f} s")
        return surf[0], wire[0], title

    ani = animation.FuncAnimation(fig, update, frames=len(steps),
                                  interval=1000 // args.fps, blit=False)

    if args.show:
        plt.show()
    else:
        print(f"Rendering GIF -> {gif_path} ...")
        ani.save(gif_path, writer="pillow", fps=args.fps, dpi=args.dpi)
        print(f"Saved {gif_path}  ({gif_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

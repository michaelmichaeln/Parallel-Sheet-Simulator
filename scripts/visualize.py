"""
Visualize cloth simulation frames as a GIF.

Requirements: pip install matplotlib pandas numpy
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render cloth simulation frames to GIF.")
    parser.add_argument(
        "--frames",
        default="results/outputs/cloth_frames.csv",
        help="Path to cloth frames CSV.",
    )
    parser.add_argument(
        "--gif",
        default="results/outputs/cloth_sim.gif",
        help="Output GIF path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames_path = Path(args.frames)
    gif_path = Path(args.gif)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames_path.exists():
        print(f"Error: Frames file not found: {frames_path}")
        return 1

    df = pd.read_csv(frames_path)
    steps = sorted(df["step"].unique())
    grid_h = int(df["row"].max()) + 1
    grid_w = int(df["col"].max()) + 1

    def get_grid(step_num):
        frame = df[df["step"] == step_num].sort_values(["row", "col"])
        x = frame["x"].values.reshape(grid_h, grid_w)
        y = frame["y"].values.reshape(grid_h, grid_w)
        z = frame["z"].values.reshape(grid_h, grid_w)
        return x, y, z

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    x0, y0, z0 = get_grid(steps[0])
    wire = [ax.plot_wireframe(x0, y0, z0, rstride=1, cstride=1, color="steelblue", linewidth=0.6, alpha=0.8)]

    ground_y = -1.0
    gx = np.linspace(-1.2, 1.2, 2)
    gz = np.linspace(-1.2, 1.2, 2)
    gx_mesh, gz_mesh = np.meshgrid(gx, gz)
    gy_mesh = np.full_like(gx_mesh, ground_y)
    ax.plot_surface(gx_mesh, gy_mesh, gz_mesh, alpha=0.15, color="sandybrown")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 2.0)
    ax.set_zlim(-1.2, 1.2)
    title = ax.set_title(f"Step {steps[0]}")

    def update(frame_idx):
        wire[0].remove()
        x, y, z = get_grid(steps[frame_idx])
        wire[0] = ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color="steelblue", linewidth=0.6, alpha=0.8)
        title.set_text(f"Step {steps[frame_idx]}")
        return wire[0], title

    ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=60, blit=False)

    plt.tight_layout()
    ani.save(gif_path, writer="pillow", fps=20)
    print(f"Saved {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
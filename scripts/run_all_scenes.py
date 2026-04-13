#!/usr/bin/env python3
"""
Run all experiment scenes and optionally render per-scene GIFs.

Usage:
  python scripts/run_all_scenes.py
  python scripts/run_all_scenes.py -v
  python scripts/run_all_scenes.py -v --gif-prefix demo_ --gif-suffix _seq
  python scripts/run_all_scenes.py -t
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
SIMULATOR_PATH = PROJECT_ROOT / "simCode" / "cloth_sim"
EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"
VISUALIZER_PATH = PROJECT_ROOT / "scripts" / "visualize.py"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
SCENES_OUT_DIR = PROJECT_ROOT / "results" / "outputs" / "scenes"
GIF_OUT_DIR = PROJECT_ROOT / "results" / "outputs" / "gif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all scene configs and save per-scene outputs."
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Render a GIF for each scene into results/outputs/gif.",
    )
    parser.add_argument(
        "-t",
        "--timing",
        action="store_true",
        help="Print timing data for each scene in the terminal.",
    )
    parser.add_argument(
        "--gif-prefix",
        default="",
        help="Optional prefix for generated GIF file names.",
    )
    parser.add_argument(
        "--gif-suffix",
        default="",
        help="Optional suffix for generated GIF file names.",
    )
    return parser.parse_args()


def ensure_required_paths() -> bool:
    if not SIMULATOR_PATH.exists():
        print(f"Error: Simulator not found at {SIMULATOR_PATH}")
        print("Compile first:")
        print("  g++ -O2 -std=c++17 -I simCode/include -o simCode/cloth_sim simCode/cloth_sim_seq.cpp")
        return False
    SCENES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    GIF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return True


def latest_summary_for_scene(scene_name: str) -> Path | None:
    summaries = sorted(METRICS_DIR.glob(f"{scene_name}_*_summary.json"), reverse=True)
    return summaries[0] if summaries else None


def build_temp_config(config_path: Path, run_tag: str) -> tuple[Path, str, Path, Path]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    scene_name = config.get("scene", config_path.stem)
    output_cfg = config.setdefault("output", {})
    frames_path = SCENES_OUT_DIR / f"{scene_name}_{run_tag}_frames.csv"
    meta_path = SCENES_OUT_DIR / f"{scene_name}_{run_tag}_meta.csv"
    output_cfg["frames_path"] = str(frames_path)
    output_cfg["meta_path"] = str(meta_path)

    # Keep metrics under results/metrics unless caller config overrides it.
    output_cfg.setdefault("metrics_path", "results/metrics/")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    with tmp:
        json.dump(config, tmp, indent=2)
    return Path(tmp.name), scene_name, frames_path, meta_path


def run_visualizer(frames_path: Path, gif_path: Path) -> bool:
    cmd = [
        sys.executable,
        str(VISUALIZER_PATH),
        "--frames",
        str(frames_path),
        "--gif",
        str(gif_path),
    ]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  GIF render failed: {gif_path.name}")
        if res.stderr.strip():
            print(f"    {res.stderr.strip()}")
        return False
    print(f"  GIF saved: {gif_path}")
    return True


def run_scene(config_path: Path, args: argparse.Namespace, run_tag: str) -> bool:
    temp_cfg, scene_name, frames_path, _ = build_temp_config(config_path, run_tag)
    start = time.perf_counter()
    sim_ok = False
    try:
        res = subprocess.run(
            [str(SIMULATOR_PATH), str(temp_cfg)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        sim_ok = res.returncode == 0
        if not sim_ok:
            print(f"FAILED: {scene_name}")
            if res.stderr.strip():
                print(f"  {res.stderr.strip()}")
            return False

        print(f"PASSED: {scene_name}")

        summary_path = latest_summary_for_scene(scene_name)
        if args.timing:
            wall_s = time.perf_counter() - start
            if summary_path and summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                results = summary.get("results", {})
                total_time_s = results.get("total_time_s", "n/a")
                sps = results.get("steps_per_second", "n/a")
                avg_ms = results.get("avg_step_ms", "n/a")
                print(f"  timing(sim): total={total_time_s}s, steps/s={sps}, avg_step_ms={avg_ms}")
            print(f"  timing(wall): {wall_s:.3f}s")

        if args.visualize and frames_path.exists():
            gif_name = f"{args.gif_prefix}{scene_name}{args.gif_suffix}_{run_tag}.gif"
            gif_path = GIF_OUT_DIR / gif_name
            run_visualizer(frames_path, gif_path)
        elif args.visualize:
            print(f"  Skipping GIF: frames file missing at {frames_path}")

        return True
    finally:
        temp_cfg.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if not ensure_required_paths():
        return 1

    configs = sorted(EXPERIMENTS_DIR.glob("*.json"))
    if not configs:
        print(f"No scene configs found in {EXPERIMENTS_DIR}")
        return 1

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    print(f"Run tag: {run_tag}")
    print(f"Scenes output: {SCENES_OUT_DIR}")
    if args.visualize:
        print(f"GIF output: {GIF_OUT_DIR}")
    print("-" * 60)

    failed = []
    for config in configs:
        ok = run_scene(config, args, run_tag)
        if not ok:
            failed.append(config.stem)

    print("\n" + "=" * 60)
    print("RUN ALL SCENES SUMMARY")
    print("=" * 60)
    print(f"Total scenes: {len(configs)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed scenes: {', '.join(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

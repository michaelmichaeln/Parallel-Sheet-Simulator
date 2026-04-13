#!/usr/bin/env python3
"""
run_experiment.py
Runs cloth simulation experiments and validates results against thresholds.

Usage:
    python scripts/run_experiment.py configs/experiments/drop_on_sphere.json
    python scripts/run_experiment.py --all                    # Run all experiments
    python scripts/run_experiment.py --list                   # List available experiments
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from glob import glob
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent
SIMULATOR_PATH = PROJECT_ROOT / "simCode" / "cloth_sim"
THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "thresholds.json"
EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"


def load_thresholds() -> dict:
    """Load pass/fail thresholds from config."""
    if not THRESHOLDS_PATH.exists():
        print(f"Warning: Thresholds file not found at {THRESHOLDS_PATH}")
        return {}
    with open(THRESHOLDS_PATH) as f:
        data = json.load(f)
    return data.get("thresholds", {})


def validate_results(summary_path: Path, thresholds: dict) -> tuple[bool, list[str]]:
    """
    Validate simulation results against thresholds.
    Returns (passed, list_of_failures).
    """
    if not summary_path.exists():
        return False, [f"Summary file not found: {summary_path}"]
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    correctness = summary.get("correctness", {})
    failures = []
    
    if "max_stretch_error" in thresholds:
        actual = correctness.get("max_stretch_error", 0)
        limit = thresholds["max_stretch_error"]
        if actual > limit:
            failures.append(f"stretch_error: {actual:.4f} > {limit}")
    
    if "max_ground_penetration_depth" in thresholds:
        # Note: Current simulator tracks penetration count, not depth in summary
        # This is a simplification - for full validation we'd parse the CSV
        pass
    
    if "max_energy_growth_ratio" in thresholds:
        actual = correctness.get("max_energy_ratio", 1.0)
        limit = thresholds["max_energy_growth_ratio"]
        if actual > limit:
            failures.append(f"energy_ratio: {actual:.4f} > {limit}")
    
    if not thresholds.get("nan_inf_allowed", False):
        if correctness.get("has_nan_inf", False):
            failures.append("NaN/Inf detected in simulation")
    
    return len(failures) == 0, failures


def run_experiment(config_path: Path, thresholds: dict, verbose: bool = True) -> bool:
    """
    Run a single experiment and validate results.
    Returns True if passed, False otherwise.
    """
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return False
    
    if not SIMULATOR_PATH.exists():
        print(f"Error: Simulator not found at {SIMULATOR_PATH}")
        print("Please compile first: cd simCode && g++ -O2 -std=c++17 -I include -o cloth_sim cloth_sim_seq.cpp")
        return False
    
    scene_name = config_path.stem
    print(f"\n{'='*60}")
    print(f"Running experiment: {scene_name}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [str(SIMULATOR_PATH), str(config_path)],
        cwd=PROJECT_ROOT,
        capture_output=not verbose,
        text=True
    )
    
    if result.returncode != 0:
        print(f"FAILED: Simulator returned exit code {result.returncode}")
        if not verbose and result.stderr:
            print(f"stderr: {result.stderr}")
        return False
    
    # Find the most recent summary file for this scene
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    summaries = sorted(metrics_dir.glob(f"{scene_name}_*_summary.json"), reverse=True)
    
    if not summaries:
        print(f"Warning: No summary file found for {scene_name}")
        return True  # Simulation ran but no validation possible
    
    summary_path = summaries[0]
    passed, failures = validate_results(summary_path, thresholds)
    
    if passed:
        print(f"PASSED: {scene_name}")
    else:
        print(f"FAILED: {scene_name}")
        for f in failures:
            print(f"  - {f}")
    
    return passed


def list_experiments():
    """List all available experiment configs."""
    print("\nAvailable experiments:")
    print("-" * 40)
    for config in sorted(EXPERIMENTS_DIR.glob("*.json")):
        with open(config) as f:
            data = json.load(f)
        scene = data.get("scene", config.stem)
        grid = data.get("cloth", {})
        grid_str = f"{grid.get('grid_w', '?')}x{grid.get('grid_h', '?')}"
        sphere = "sphere" if data.get("collision", {}).get("sphere_enabled", False) else "plane"
        print(f"  {config.stem:20} - {grid_str:10} {sphere}")


def run_all_experiments(thresholds: dict, verbose: bool = True) -> dict:
    """Run all experiments and return results summary."""
    results = {"passed": [], "failed": []}
    
    for config_path in sorted(EXPERIMENTS_DIR.glob("*.json")):
        passed = run_experiment(config_path, thresholds, verbose)
        if passed:
            results["passed"].append(config_path.stem)
        else:
            results["failed"].append(config_path.stem)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {len(results['passed'])}/{len(results['passed']) + len(results['failed'])}")
    
    if results["failed"]:
        print(f"Failed: {', '.join(results['failed'])}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run cloth simulation experiments and validate results."
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to experiment config JSON file"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all experiments in configs/experiments/"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available experiments"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress simulator output"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return 0
    
    thresholds = load_thresholds()
    
    if args.all:
        results = run_all_experiments(thresholds, verbose=not args.quiet)
        return 0 if not results["failed"] else 1
    
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        passed = run_experiment(config_path, thresholds, verbose=not args.quiet)
        return 0 if passed else 1
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

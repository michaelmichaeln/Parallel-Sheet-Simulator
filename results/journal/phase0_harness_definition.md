# Phase 0: Test Harness and Experiment Infrastructure

**Date**: 2026-04-13  
**Authors**: Ankita Kundu, Michael Nguyen

## Summary

Implemented a rigorous experiment harness for the cloth simulator with config-driven experiments, correctness validation, and structured metrics output.

## Components Implemented

### 1. Config-Driven Experiments

- Added nlohmann/json single-header library for JSON parsing
- Refactored `cloth_sim_seq.cpp` to load all parameters from JSON config files
- CLI argument: `./cloth_sim path/to/config.json`

### 2. Predefined Test Scenes

Created 6 standard experiment configurations in `configs/experiments/`:

| Scene | Grid | Key Feature |
|-------|------|-------------|
| `drop_on_sphere` | 25x25 | Baseline with sphere collision |
| `flat_drop_plane` | 25x25 | No sphere, ground-only collision |
| `large_grid` | 50x50 | Scalability test |
| `stiff_cloth` | 25x25 | High stiffness (k=2000) |
| `soft_cloth` | 25x25 | Low stiffness (k=100) |
| `high_drop` | 25x25 | Drop from y=6.0, 2000 steps |

### 3. Correctness Metrics

Per-frame metrics computed and logged:
- **Spring stretch error**: `max(|len - rest| / rest)`
- **Ground penetration**: count and max depth
- **Sphere penetration**: count and max depth  
- **Energy tracking**: kinetic, potential, spring, total
- **NaN/Inf detection**: immediate flagging

### 4. Pass/Fail Thresholds

Defined in `configs/thresholds.json`:
- `max_stretch_error`: 1.5 (150% for basic mass-spring)
- `max_energy_growth_ratio`: 2.0
- `nan_inf_allowed`: false

Stricter thresholds documented for future PBD implementation.

### 5. Python Runner Script

`scripts/run_experiment.py` provides:
- Single experiment: `python run_experiment.py config.json`
- Batch execution: `python run_experiment.py --all`
- Listing: `python run_experiment.py --list`
- Validation against thresholds with pass/fail reporting

## Output Structure

```
results/
  metrics/
    {scene}_{timestamp}.csv       # Per-frame metrics
    {scene}_{timestamp}_summary.json  # Run summary
  journal/
    phase0_harness_definition.md  # This document
```

## Validation Results

Initial validation run (2026-04-13):
- **Passed**: drop_on_sphere, flat_drop_plane, high_drop, stiff_cloth
- **Failed**: large_grid (stretch 2.49), soft_cloth (stretch 3.57)

Failures are expected - they document the stretch behavior of mass-spring systems without position constraints.

## Next Steps

1. **Phase 1 completion**: Add irregular collision surfaces (triangle mesh or SDF)
2. **Phase 2**: Port to CUDA with node-parallel and edge-parallel kernels
3. **Phase 3**: Design space exploration (AoS vs SoA, sync strategies)

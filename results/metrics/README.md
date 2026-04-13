# Metrics Output Directory

This directory contains per-experiment metrics output from cloth simulation runs.

## File Naming Convention

- `{scene}_{timestamp}.csv` - Per-frame metrics data
- `{scene}_{timestamp}_summary.json` - Run summary with config and aggregate correctness metrics

## CSV Columns

| Column | Description |
|--------|-------------|
| `step` | Simulation step number |
| `max_stretch_error` | Maximum spring stretch ratio `\|len - rest\| / rest` |
| `ground_penetrations` | Number of particles below ground plane |
| `max_ground_pen` | Maximum penetration depth below ground |
| `sphere_penetrations` | Number of particles inside collision sphere |
| `max_sphere_pen` | Maximum penetration depth into sphere |
| `kinetic_energy` | Total kinetic energy (0.5 * m * v^2) |
| `potential_energy` | Gravitational potential energy (m * g * h) |
| `spring_energy` | Elastic potential energy in springs (0.5 * k * x^2) |
| `total_energy` | Sum of all energy components |
| `has_nan_inf` | 1 if NaN/Inf detected, 0 otherwise |
| `step_time_ms` | Wall-clock time for this step (ms) |
| `cumulative_time_ms` | Total elapsed simulation time (ms) |

## Summary JSON Structure

```json
{
  "scene": "scene_name",
  "timestamp": "YYYYMMDD_HHMMSS",
  "config": { ... },
  "results": {
    "particles": 625,
    "springs": 3502,
    "total_time_s": 0.05,
    "steps_per_second": 24000,
    "avg_step_ms": 0.012
  },
  "correctness": {
    "max_stretch_error": 1.08,
    "max_ground_penetrations": 0,
    "max_sphere_penetrations": 31,
    "max_energy_ratio": 1.0,
    "has_nan_inf": false
  }
}
```

## Validation

Run validation against thresholds:

```bash
python scripts/run_experiment.py configs/experiments/drop_on_sphere.json
python scripts/run_experiment.py --all  # Run all experiments
```

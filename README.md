# Parallel-Sheet-Simulator

File layout:

- `simCode/` - simulator code (`cloth_sim_seq.cpp`, `cloth_sim_parallel.cpp`) and `include/json.hpp`.
- `configs/` - experiment presets (`experiments/*.json`) and validation thresholds (`thresholds.json`).
- `scripts/` - tools:
  - `run_experiment.py`: run one/all experiments and validate results.
  - `visualize.py`: render `results/outputs/cloth_frames.csv` to `results/outputs/cloth_sim.gif`.
- `results/` - generated outputs only:
  - `metrics/` for per-step CSV + summary JSON.
  - `outputs/` for frames/meta/GIF artifacts.

Quick usage:

```bash
g++ -O2 -std=c++17 -I simCode/include -o simCode/cloth_sim simCode/cloth_sim_seq.cpp
./simCode/cloth_sim configs/experiments/drop_on_sphere.json
python scripts/run_experiment.py --all
python scripts/visualize.py
```

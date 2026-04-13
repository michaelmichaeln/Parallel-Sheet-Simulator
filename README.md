# Parallel-Sheet-Simulator

File layout:

- `simCode/` - simulator code (`cloth_sim_seq.cpp`, `cloth_sim_parallel.cpp`) and `include/json.hpp`.
- `configs/` - experiment presets (`experiments/*.json`) and validation thresholds (`thresholds.json`).
- `scripts/` - tools:
  - `run_experiment.py`: run one/all experiments and validate results.
  - `visualize.py`: render `results/outputs/cloth_frames.csv` to `results/outputs/cloth_sim.gif`.
  - `run_all_scenes.py`: run all scenes and save per-scene outputs/GIFs.  
      -v(visualize), -c(correctness), -t(timing), -b(bench sweep: small/med/large)
- `results/` - generated outputs only:
  - `metrics/` for per-step CSV + summary JSON.
  - `outputs/scenes/` for per-scene frames/meta files.
  - `outputs/gif/` for per-scene GIF files.

Quick usage:

```bash
g++ -O2 -std=c++17 -I simCode/include -o simCode/cloth_sim simCode/cloth_sim_seq.cpp
./simCode/cloth_sim configs/experiments/drop_on_sphere.json
python scripts/run_experiment.py --all
python scripts/visualize.py
python scripts/run_all_scenes.py -v -t
python scripts/run_all_scenes.py -v -c -t -b
```

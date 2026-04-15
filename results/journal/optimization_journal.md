# Optimization Journal — Parallel Sheet Simulator
## Ankita Kundu & Michael Nguyen — 15-418/618 Final Project

**Hardware**: Intel i7-9700 (8C), NVIDIA RTX 2080 (Turing, SM 7.5, 2944 CUDA cores, 8GB GDDR6)  
**Grid sizes tested**: 100x100 (10K), 250x250 (62.5K), 500x500 (250K), 1000x1000 (1M particles)  
**Simulation**: 1200 timesteps, dt=0.005s, mass-spring cloth drop with sphere collision

---

## V0: Sequential CPU Baseline

**File**: `cloth_sim_seq.cpp`

### Design
Single-threaded C++ with `-O2 -march=native`. Semi-implicit Euler integration.  
Iterates over all springs to compute Hooke's law forces, then integrates and handles collisions.

### Timing (fill after running)

| Grid Size | Particles | Springs | Time (ms) | Steps/s |
|-----------|-----------|---------|-----------|---------|
| 100x100   | 10,000    | 59,002  | _____     | _____   |
| 250x250   | 62,500    | 371,252 | _____     | _____   |
| 500x500   | 250,000   | 1,489,002 | _____   | _____   |
| 1000x1000 | 1,000,000 | 5,970,002 | _____   | _____   |

### Observations
- Sequential time should scale roughly linearly with spring count.
- This is the reference for correctness validation and speedup computation.

---

## V0 → V1: Naive CUDA Port

**File**: `simCode/cloth_sim_v1.cu`

### Hypothesis
A direct port with one thread per particle and AoS layout will show speedup at large sizes
but will be limited by poor memory coalescing (stride = `sizeof(Particle)` = 40+ bytes between
consecutive threads' accesses).

### What Changed
- Ported all simulation phases to 5 CUDA kernels: `clear_forces`, `apply_external`, `compute_spring_forces`, `integrate`, `collision`
- Used node-parallel strategy for spring forces: each thread iterates its connected springs via a precomputed CSR adjacency list
- AoS layout: same `Particle` struct as sequential, stored on device
- Block size: 256 threads (1D)

### Why Node-Parallel (Not Spring-Parallel)?
Spring-parallel would require one thread per spring, with `atomicAdd` to accumulate forces on
both endpoints. At ~6 springs/particle (12 adjacency entries/particle), the atomic contention
would be severe. Node-parallel avoids atomics entirely at the cost of some redundant position reads.

### Expected Bottlenecks
1. **AoS striding**: Threads in a warp read `particles[i].pos` — addresses are ~40 bytes apart, wasting most of each 128-byte cache line
2. **5 kernel launches per step**: launch overhead (~5-10us each) dominates at small grid sizes
3. **CSR adjacency indirection**: random-looking global memory reads for neighbor positions

### Timing

| Grid Size | Seq (ms) | V1 (ms) | Speedup |
|-----------|----------|---------|---------|
| 100x100   | _____    | _____   | _____x  |
| 250x250   | _____    | _____   | _____x  |
| 500x500   | _____    | _____   | _____x  |
| 1000x1000 | _____    | _____   | _____x  |

### Key Takeaway
_[Fill: Was the bottleneck as predicted? What does profiling show?]_

---

## V1 → V2: SoA Memory Layout

**File**: `simCode/cloth_sim_v2.cu`

### Hypothesis
Converting from AoS to SoA will dramatically improve memory coalescing. When 32 threads
in a warp read `pos_x[i..i+31]`, they access 128 contiguous bytes — a single coalesced
transaction. AoS requires 32 separate cache lines for the same data.

### What Changed
- Replaced `Particle*` with separate `float*` arrays: `pos_x`, `pos_y`, `pos_z`, `vel_x`, `vel_y`, `vel_z`, `force_x`, `force_y`, `force_z`
- Fused `clear_forces` + `apply_external` into one kernel (trivial: just overwrite `f[i]` with gravity+drag)
- Kernel count per step: 4 (down from 5)
- Same CSR adjacency for spring forces, same block size
- Mass stored as constant (uniform across all particles)

### Why This Should Help
The RTX 2080 has 448 GB/s memory bandwidth. AoS wastes most of it loading unused struct fields
per cache line. SoA ensures every byte loaded is used, approaching theoretical peak bandwidth
for the integrate and collision kernels (which are purely bandwidth-bound).

The spring force kernel benefits too: reading `pos_x[neighbor]` for 12 neighbors in SoA generates
far fewer cache lines than reading 12 full `Particle` structs.

### Timing

| Grid Size | V1 (ms) | V2 (ms) | V2 Speedup | vs Seq |
|-----------|---------|---------|------------|--------|
| 100x100   | _____   | _____   | _____x     | _____x |
| 250x250   | _____   | _____   | _____x     | _____x |
| 500x500   | _____   | _____   | _____x     | _____x |
| 1000x1000 | _____   | _____   | _____x     | _____x |

### Key Takeaway
_[Fill: How much did coalescing help? Compare bandwidth utilization if Nsight data available.]_

---

## V2 → V3: Shared Memory Tiling

**File**: `simCode/cloth_sim_v3.cu`

### Hypothesis
The cloth grid has strong spatial locality: every spring connects particles within 2 grid
hops. By loading a tile of positions into shared memory (including a 2-cell halo for bend
springs), we eliminate redundant global memory reads. Interior particles in V2 read each
neighbor from global memory independently; with tiling, each position is loaded once per block.

### What Changed
- **2D thread blocks** (16x16 = 256 threads) matching grid topology
- **Shared memory tile**: `(16+4) x (16+4) x 3 floats` = 4.8 KB per block
- Spring force computation now directly encodes the 12 neighbor offsets (structural ±1, shear ±1/±1, bend ±2) rather than using CSR adjacency
- Removed CSR adjacency structure entirely (saves device memory and eliminates indirection)
- `__syncthreads()` after cooperative halo load

### Why Encode Neighbors Directly?
For a regular grid, every interior particle has exactly the same 12 neighbor pattern. Encoding
these as compile-time offsets eliminates the CSR offset+entry lookups (2 global reads per neighbor)
and allows the compiler to generate better code. This is only valid for regular grids — irregular
meshes would still need CSR.

### Shared Memory Math
- Tile + halo: 20x20 = 400 cells x 3 floats x 4 bytes = 4,800 bytes
- RTX 2080 shared memory: 48 KB per SM
- We can fit ~10 blocks per SM from shared memory alone (occupancy limited by other factors)

### Timing

| Grid Size | V2 (ms) | V3 (ms) | V3 vs V2 | vs Seq |
|-----------|---------|---------|----------|--------|
| 100x100   | _____   | _____   | _____x   | _____x |
| 250x250   | _____   | _____   | _____x   | _____x |
| 500x500   | _____   | _____   | _____x   | _____x |
| 1000x1000 | _____   | _____   | _____x   | _____x |

### Key Takeaway
_[Fill: Did shared memory help significantly, or was L1/L2 cache already capturing locality?]_

---

## V3 → V4: Fully Optimized

**File**: `simCode/cloth_sim_v4.cu`

### Hypothesis
The remaining overhead in V3 comes from: (1) separate kernel launches for force computation
vs integration, requiring intermediate global memory writes of the force array, and (2) reading
simulation constants from global memory on every access. Fusing kernels and using constant
memory should eliminate these costs.

### What Changed
1. **Kernel fusion**: Merged clear + external + spring forces into `fused_force_kernel`.
   Force accumulator (`ax, ay, az`) lives entirely in registers — computed from shared memory
   positions, written to global `f[i]` once. Previous versions wrote forces to global memory
   in `clear_and_apply_external` then read them back in `spring_forces`.
2. **Fused integrate + collision**: Single kernel reads forces, updates velocity/position,
   handles ground and sphere collision, writes final pos/vel. New position is computed in
   registers before any global write.
3. **`__constant__` memory**: All simulation parameters (dt, stiffnesses, gravity, sphere
   geometry) stored in constant memory for warp-broadcast access.
4. **`__restrict__` pointers**: Tells nvcc that pos (read), vel (read), and force (write)
   arrays don't alias, enabling more aggressive load/store optimization.
5. **Configurable block size**: Accepts `BLOCK_W BLOCK_H` via CLI for easy sweep without
   recompilation.
6. **Only 2 kernel launches per step** (down from 4 in V3, 5 in V1).

### Why Kernel Fusion Matters
Each kernel launch has ~5-10μs overhead. More importantly, between V3's `clear_and_apply_external`
and `spring_forces` kernels, the force array is written to global memory and then read back —
2 x N x 3 x 4 bytes of unnecessary traffic. Fusion keeps forces in registers throughout the
entire force computation.

### Block Size Tuning
Run V4 with different tile sizes to find the optimum:

| Tile Size | Threads/Block | Shared Mem | Time (ms) at 500x500 |
|-----------|---------------|------------|----------------------|
| 8x8       | 64            | ~2 KB      | _____                |
| 16x16     | 256           | ~5 KB      | _____                |
| 32x8      | 256           | ~5 KB      | _____                |
| 32x16     | 512           | ~9 KB      | _____                |

### Final Timing

| Grid Size | V3 (ms) | V4 (ms) | V4 vs V3 | vs Seq |
|-----------|---------|---------|----------|--------|
| 100x100   | _____   | _____   | _____x   | _____x |
| 250x250   | _____   | _____   | _____x   | _____x |
| 500x500   | _____   | _____   | _____x   | _____x |
| 1000x1000 | _____   | _____   | _____x   | _____x |

### Key Takeaway
_[Fill: Was kernel fusion the dominant improvement, or constant memory?]_

---

## Summary: Full Optimization Progression

| Version | Key Change            | Kernels/Step | Speedup (1000x1000) |
|---------|-----------------------|--------------|---------------------|
| V0      | Sequential CPU        | N/A          | 1.0x (baseline)     |
| V1      | Naive CUDA (AoS)      | 5            | _____x              |
| V2      | SoA layout            | 4            | _____x              |
| V3      | Shared memory tiling  | 4            | _____x              |
| V4      | Fused + optimized     | 2            | _____x              |

### What Mattered Most
_[Fill after benchmarks: rank the optimizations by impact]_

1. ____
2. ____
3. ____

### What Didn't Help (or Helped Less Than Expected)
_[Fill: any optimizations that profiling showed were minor]_

---

## Correctness Validation

All CUDA versions validated against sequential CPU reference using L2-norm and max-norm
of final position differences.

| Version | 100x100 Max Dev | 250x250 Max Dev | 500x500 Max Dev | 1000x1000 Max Dev |
|---------|-----------------|-----------------|------------------|--------------------|
| V1      | _____           | _____           | _____            | _____              |
| V2      | _____           | _____           | _____            | _____              |
| V3      | _____           | _____           | _____            | _____              |
| V4      | _____           | _____           | _____            | _____              |

Threshold: max deviation < 1e-3 (expected due to float operation reordering).

---

## Future Work
- Multi-GPU: partition grid across GPUs with halo exchange
- Adaptive time-stepping for stability at large deformations
- Irregular mesh support (triangle soup) requiring CSR adjacency on GPU
- Signed distance field collision for arbitrary surfaces
- Warp-level shuffle operations for neighbor exchange within a warp

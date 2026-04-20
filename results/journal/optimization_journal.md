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

### Timing

| Grid Size | Particles | Springs | Time (ms) | Steps/s |
|-----------|-----------|---------|-----------|---------|
| 100x100   | 10,000    | 59,002  | 436.2     | 2751.0  |
| 250x250   | 62,500    | 371,252 | 2752.5    | 436.0   |
| 500x500   | 250,000   | 1,489,002 | 12208.8 | 98.3    |
| 1000x1000 | 1,000,000 | 5,970,002 | 55321.5 | 21.7    |

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
| 100x100   | 436.2    | 29.5027 | 14.78x  |
| 250x250   | 2752.5   | 324.3722| 8.48x   |
| 500x500   | 12208.8  | 1503.6295 | 8.11x |
| 1000x1000 | 55321.5  | 6246.5688 | 8.85x |

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
| 100x100   | 29.5027 | 22.0086 | 1.34x      | 19.81x |
| 250x250   | 324.3722| 215.3335| 1.51x      | 12.78x |
| 500x500   | 1503.6295 | 852.4832 | 1.76x   | 14.32x |
| 1000x1000 | 6246.5688 | 3561.0596 | 1.75x  | 15.53x |

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
| 100x100   | 22.0086 | 15.5847 | 1.41x    | 27.98x |
| 250x250   | 215.3335 | 27.2366 | 7.91x   | 101.05x |
| 500x500   | 852.4832 | 151.4568 | 5.63x  | 80.60x |
| 1000x1000 | 3561.0596 | 551.5203 | 6.46x | 100.30x |

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
| 8x8       | 64            | ~2 KB      | 95.2563        |
| 16x16     | 256           | ~5 KB      | 93.0798              |
| 32x8      | 256           | ~5 KB      | 91.2579        |
| 32x16     | 512           | ~9 KB      | 92.8524        |

### Final Timing

| Grid Size | V3 (ms) | V4 (ms) | V4 vs V3 | vs Seq |
|-----------|---------|---------|----------|--------|
| 100x100   | 15.5847 | 9.1054  | 1.71x    | 47.90x |
| 250x250   | 27.2366 | 18.4992 | 1.47x    | 148.79x |
| 500x500   | 151.4568| 93.0798 | 1.63x    | 131.16x |
| 1000x1000 | 551.5203| 344.2668| 1.60x    | 160.69x |

### Key Takeaway
V4 improved over V3 by about **1.47x-1.71x** across tested sizes. This is a strong gain, but
smaller than the V2->V3 jump; the largest observed step-change in this project came from
the regular-grid/shared-memory reformulation. V4 still delivered the best absolute runtime
and highest end-to-end speedup.

---

## Summary: Full Optimization Progression

| Version | Key Change            | Kernels/Step | Speedup (1000x1000) |
|---------|-----------------------|--------------|---------------------|
| V0      | Sequential CPU        | N/A          | 1.0x (baseline)     |
| V1      | Naive CUDA (AoS)      | 5            | 8.85x               |
| V2      | SoA layout            | 4            | 15.53x              |
| V3      | Shared memory tiling  | 4            | 100.30x             |
| V4      | Fused + optimized     | 2            | 160.69x             |

### What Mattered Most
Ranking by observed median runtime impact (largest to smaller):

1. **V2 -> V3 shared-memory tiled regular-grid spring kernel** (largest jump at medium/large N)
2. **V3 -> V4 kernel fusion + constant memory + restrict qualifiers** (consistent additional 1.47x-1.71x)
3. **V1 -> V2 SoA layout and partial kernel fusion** (solid 1.34x-1.76x over V1)
4. **Tile Configuration** (minor ~1.02x in V4)


// =============================================================================
// cloth_sim_v4.cu
// V4: Fully Optimized — fused kernels, constant memory, __restrict__, tunable
// Ankita Kundu & Michael Nguyen — 15-418/618 Final Project
//
// Key changes from V3:
//  1. KERNEL FUSION: clear_forces + external_forces + spring_forces merged
//     into a single kernel. Force accumulator lives entirely in registers —
//     never written to or read from global memory as an intermediate.
//     This eliminates 2 kernel launches and 2 global R/W passes per step.
//  2. CONSTANT MEMORY: All simulation parameters loaded into __constant__
//     memory for broadcast-efficient access across warps.
//  3. __restrict__ POINTERS: Enables compiler to assume no aliasing between
//     position (read) and velocity (read/write) arrays.
//  4. CONFIGURABLE BLOCK SIZE: Accepts BLOCK_W/BLOCK_H via CLI for sweeping
//     optimal tile size without recompilation (defaults to 16x16).
//  5. FUSED INTEGRATE+COLLISION: Single kernel for velocity update, position
//     update, ground collision, and sphere collision.
//
// Compile:  nvcc -O2 -std=c++17 -arch=sm_75 -o cloth_sim_v4 cloth_sim_v4.cu
// Run:      ./cloth_sim_v4 <GRID_W> <GRID_H> [NUM_STEPS] [BLOCK_W] [BLOCK_H]
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Constant memory for simulation parameters
// ---------------------------------------------------------------------------
struct SimParams {
    float cloth_size;
    float dt;
    float k_struct, k_shear, k_bend;
    float damping, air_drag;
    float gravity_y;
    float ground_y, ground_restitution, ground_friction;
    int   sphere_enabled;
    float sphere_radius, sphere_cx, sphere_cy, sphere_cz;
    float mass, inv_mass;
};

__constant__ SimParams d_params;

// ---------------------------------------------------------------------------
// Max tile size for shared memory allocation (compile-time upper bound).
// Actual tile size is set at runtime; shared memory is dynamically sized.
// ---------------------------------------------------------------------------
constexpr int HALO     = 2;

// ---------------------------------------------------------------------------
// Host setup (SoA)
// ---------------------------------------------------------------------------
struct HostSoA {
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    int n;
};

void init_particles_host(HostSoA& soa, int grid_w, int grid_h, float cloth_size, float drop_y) {
    int n = grid_w * grid_h;
    soa.n = n;
    soa.pos_x.resize(n); soa.pos_y.resize(n); soa.pos_z.resize(n);
    soa.vel_x.resize(n, 0); soa.vel_y.resize(n, 0); soa.vel_z.resize(n, 0);

    const float dx = cloth_size / (grid_w - 1);
    const float dz = cloth_size / (grid_h - 1);

    for (int r = 0; r < grid_h; ++r)
        for (int c = 0; c < grid_w; ++c) {
            int i = r * grid_w + c;
            soa.pos_x[i] = c * dx - cloth_size * 0.5f;
            soa.pos_y[i] = drop_y;
            soa.pos_z[i] = r * dz - cloth_size * 0.5f;
        }
}

// ---------------------------------------------------------------------------
// FUSED KERNEL: clear + external + spring forces
// Computes total force per particle entirely in registers, writes once.
// ---------------------------------------------------------------------------
__global__
void fused_force_kernel(const float* __restrict__ px,
                        const float* __restrict__ py,
                        const float* __restrict__ pz,
                        const float* __restrict__ vx,
                        const float* __restrict__ vy,
                        const float* __restrict__ vz,
                        float* __restrict__ fx,
                        float* __restrict__ fy,
                        float* __restrict__ fz,
                        int grid_w, int grid_h,
                        int tile_w, int tile_h) {
    // Dynamic shared memory: 3 arrays of (tile_w+4) x (tile_h+4)
    extern __shared__ float smem[];
    int sm_w = tile_w + 2 * HALO;
    int sm_h = tile_h + 2 * HALO;
    int sm_plane = sm_w * sm_h;

    float* s_px = smem;
    float* s_py = smem + sm_plane;
    float* s_pz = smem + 2 * sm_plane;

    int g_col = blockIdx.x * tile_w + threadIdx.x;
    int g_row = blockIdx.y * tile_h + threadIdx.y;

    int tile_r0 = blockIdx.y * tile_h - HALO;
    int tile_c0 = blockIdx.x * tile_w - HALO;

    // Cooperative load of tile + halo
    int tid = threadIdx.y * tile_w + threadIdx.x;
    int total_sm = sm_w * sm_h;
    int tpb = tile_w * tile_h;

    for (int idx = tid; idx < total_sm; idx += tpb) {
        int sr = idx / sm_w;
        int sc = idx % sm_w;
        int gr = tile_r0 + sr;
        int gc = tile_c0 + sc;
        gr = max(0, min(gr, grid_h - 1));
        gc = max(0, min(gc, grid_w - 1));
        int gi = gr * grid_w + gc;
        s_px[idx] = px[gi];
        s_py[idx] = py[gi];
        s_pz[idx] = pz[gi];
    }

    __syncthreads();

    if (g_col >= grid_w || g_row >= grid_h) return;

    int gi = g_row * grid_w + g_col;

    int lr = threadIdx.y + HALO;
    int lc = threadIdx.x + HALO;
    int li = lr * sm_w + lc;

    float my_x = s_px[li];
    float my_y = s_py[li];
    float my_z = s_pz[li];

    // Start with external forces (gravity + drag) — fused with clear
    float ax = vx[gi] * (-d_params.air_drag);
    float ay = d_params.gravity_y * d_params.mass + vy[gi] * (-d_params.air_drag);
    float az = vz[gi] * (-d_params.air_drag);

    // Rest lengths computed from grid spacing
    float dx_s = d_params.cloth_size / (grid_w - 1);
    float dz_s = d_params.cloth_size / (grid_h - 1);
    float rest_h  = dx_s;
    float rest_v  = dz_s;
    float rest_d  = sqrtf(dx_s * dx_s + dz_s * dz_s);
    float rest_bh = 2.0f * dx_s;
    float rest_bv = 2.0f * dz_s;

    #define ACCUMULATE(dr, dc, K, REST)                                        \
    {                                                                           \
        int nr = g_row + (dr);                                                  \
        int nc = g_col + (dc);                                                  \
        if (nr >= 0 && nr < grid_h && nc >= 0 && nc < grid_w) {                \
            int ni = (lr + (dr)) * sm_w + (lc + (dc));                          \
            float ddx = s_px[ni] - my_x;                                        \
            float ddy = s_py[ni] - my_y;                                        \
            float ddz = s_pz[ni] - my_z;                                        \
            float len = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);                     \
            if (len > 1e-9f) {                                                  \
                float f = (K) * (len - (REST)) / len;                           \
                ax += ddx * f;  ay += ddy * f;  az += ddz * f;                  \
            }                                                                   \
        }                                                                       \
    }

    ACCUMULATE( 0,  1, d_params.k_struct, rest_h);
    ACCUMULATE( 0, -1, d_params.k_struct, rest_h);
    ACCUMULATE( 1,  0, d_params.k_struct, rest_v);
    ACCUMULATE(-1,  0, d_params.k_struct, rest_v);

    ACCUMULATE( 1,  1, d_params.k_shear, rest_d);
    ACCUMULATE( 1, -1, d_params.k_shear, rest_d);
    ACCUMULATE(-1,  1, d_params.k_shear, rest_d);
    ACCUMULATE(-1, -1, d_params.k_shear, rest_d);

    ACCUMULATE( 0,  2, d_params.k_bend, rest_bh);
    ACCUMULATE( 0, -2, d_params.k_bend, rest_bh);
    ACCUMULATE( 2,  0, d_params.k_bend, rest_bv);
    ACCUMULATE(-2,  0, d_params.k_bend, rest_bv);

    #undef ACCUMULATE

    // Single global write of total force
    fx[gi] = ax;
    fy[gi] = ay;
    fz[gi] = az;
}

// ---------------------------------------------------------------------------
// FUSED KERNEL: integrate + collision
// ---------------------------------------------------------------------------
__global__
void fused_integrate_collision_kernel(float* __restrict__ px,
                                      float* __restrict__ py,
                                      float* __restrict__ pz,
                                      float* __restrict__ vx,
                                      float* __restrict__ vy,
                                      float* __restrict__ vz,
                                      const float* __restrict__ fx,
                                      const float* __restrict__ fy,
                                      const float* __restrict__ fz,
                                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Integration
    float nvx = vx[i] + fx[i] * d_params.inv_mass * d_params.dt;
    float nvy = vy[i] + fy[i] * d_params.inv_mass * d_params.dt;
    float nvz = vz[i] + fz[i] * d_params.inv_mass * d_params.dt;

    nvx *= d_params.damping;
    nvy *= d_params.damping;
    nvz *= d_params.damping;

    float npx = px[i] + nvx * d_params.dt;
    float npy = py[i] + nvy * d_params.dt;
    float npz = pz[i] + nvz * d_params.dt;

    // Ground collision
    if (npy < d_params.ground_y) {
        npy  = d_params.ground_y;
        nvy  = fabsf(nvy) * d_params.ground_restitution;
        nvx *= d_params.ground_friction;
        nvz *= d_params.ground_friction;
    }

    // Sphere collision
    if (d_params.sphere_enabled) {
        float sdx = npx - d_params.sphere_cx;
        float sdy = npy - d_params.sphere_cy;
        float sdz = npz - d_params.sphere_cz;
        float dist = sqrtf(sdx*sdx + sdy*sdy + sdz*sdz);
        if (dist < d_params.sphere_radius && dist > 1e-9f) {
            float inv_d = 1.0f / dist;
            float snx = sdx * inv_d, sny = sdy * inv_d, snz = sdz * inv_d;
            npx = d_params.sphere_cx + snx * d_params.sphere_radius;
            npy = d_params.sphere_cy + sny * d_params.sphere_radius;
            npz = d_params.sphere_cz + snz * d_params.sphere_radius;
            float v_n = nvx*snx + nvy*sny + nvz*snz;
            if (v_n < 0.0f) {
                nvx += snx * (-v_n);
                nvy += sny * (-v_n);
                nvz += snz * (-v_n);
            }
        }
    }

    px[i] = npx;  py[i] = npy;  pz[i] = npz;
    vx[i] = nvx;  vy[i] = nvy;  vz[i] = nvz;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
void validate_against_reference(const HostSoA& soa, int grid_w, int grid_h) {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "results/outputs/ref_%dx%d.bin", grid_w, grid_h);
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Warning: no reference file %s — skipping validation\n", fname);
        return;
    }

    int ref_n;
    in.read(reinterpret_cast<char*>(&ref_n), sizeof(int));
    if (ref_n != soa.n) {
        std::fprintf(stderr, "Warning: reference size mismatch\n");
        return;
    }

    double l2_sum = 0.0;
    float  max_dev = 0.0f;
    for (int i = 0; i < ref_n; ++i) {
        float rx, ry, rz;
        in.read(reinterpret_cast<char*>(&rx), sizeof(float));
        in.read(reinterpret_cast<char*>(&ry), sizeof(float));
        in.read(reinterpret_cast<char*>(&rz), sizeof(float));
        float dx = soa.pos_x[i] - rx;
        float dy = soa.pos_y[i] - ry;
        float dz = soa.pos_z[i] - rz;
        float d2 = dx*dx + dy*dy + dz*dz;
        l2_sum += d2;
        float d = std::sqrt(d2);
        if (d > max_dev) max_dev = d;
    }

    double l2_norm = std::sqrt(l2_sum);
    bool pass = max_dev < 5e-2f;
    std::fprintf(stderr, "Validation vs %s: L2=%.6e  max=%.6e  %s\n",
                 fname, l2_norm, (double)max_dev, pass ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int grid_w    = 25;
    int grid_h    = 25;
    int num_steps = 1200;

    int tile_w = 16;
    int tile_h = 16;

    float sphere_r = 0.6f;
    float drop_y   = 3.0f;

    std::string frames_path;
    constexpr int SAVE_EVERY = 10;

    // Parse positional args, then scan for --frames <path>
    if (argc >= 3) {
        grid_w = std::atoi(argv[1]);
        grid_h = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        num_steps = std::atoi(argv[3]);
    }

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--tile_w" && i + 1 < argc) {
            tile_w = std::atoi(argv[++i]);
        } else if (arg == "--tile_h" && i + 1 < argc) {
            tile_h = std::atoi(argv[++i]);
        } else if (arg == "--sphere_r" && i + 1 < argc) {
            sphere_r = std::atof(argv[++i]);
        } else if (arg == "--drop_y" && i + 1 < argc) {
            drop_y = std::atof(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            frames_path = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
            return 1;
        }
    }

    const int N = grid_w * grid_h;
    const float cloth_size = 3.0f;

    // Upload constant parameters
    SimParams h_params;
    h_params.cloth_size   = cloth_size;
    h_params.dt           = 0.005f;
    h_params.k_struct     = 2000.0f;
    h_params.k_shear      = 600.0f;
    h_params.k_bend       = 400.0f;
    h_params.damping      = 0.998f;
    h_params.air_drag     = 0.3f;
    h_params.gravity_y    = -9.81f;
    h_params.ground_y     = -1.0f;
    h_params.ground_restitution = 0.02f;
    h_params.ground_friction    = 0.92f;
    h_params.sphere_enabled = 1;
    h_params.sphere_radius = sphere_r;
    h_params.sphere_cy = h_params.ground_y + sphere_r;
    h_params.sphere_cx = 0.0f;
    h_params.sphere_cz = 0.0f;
    h_params.mass     = 0.1f;
    h_params.inv_mass = 1.0f / h_params.mass;

    CUDA_CHECK(cudaMemcpyToSymbol(d_params, &h_params, sizeof(SimParams)));

    // Launch configs
    constexpr int BLOCK_1D = 256;
    int blocks_1d = (N + BLOCK_1D - 1) / BLOCK_1D;

    dim3 block_2d(tile_w, tile_h);
    dim3 grid_2d((grid_w + tile_w - 1) / tile_w,
                 (grid_h + tile_h - 1) / tile_h);

    int sm_w = tile_w + 2 * HALO;
    int sm_h = tile_h + 2 * HALO;
    size_t shmem_bytes = 3 * sm_w * sm_h * sizeof(float);

    std::fprintf(stderr, "=== V4: Fully Optimized CUDA ===\n");
    std::fprintf(stderr, "Grid:  %d x %d  (%d particles)\n", grid_w, grid_h, N);
    std::fprintf(stderr, "Steps: %d   dt=%.4f\n", num_steps, h_params.dt);
    std::fprintf(stderr, "Tile:  %dx%d  Shared mem: %zu bytes\n", tile_w, tile_h, shmem_bytes);
    std::fprintf(stderr, "2D grid: (%d, %d)   1D blocks: %d\n\n", grid_2d.x, grid_2d.y, blocks_1d);
    std::fprintf(stderr, "Drop height: %.2f\n", drop_y);
    std::fprintf(stderr, "Sphere radius: %.2f\n", sphere_r);

    // --- Host setup ---
    HostSoA h;
    init_particles_host(h, grid_w, grid_h, cloth_size, drop_y);

    // --- Device allocation ---
    float *d_px, *d_py, *d_pz;
    float *d_vx, *d_vy, *d_vz;
    float *d_fx, *d_fy, *d_fz;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fz, N * sizeof(float)));

    auto upload = [&]() {
        CUDA_CHECK(cudaMemcpy(d_px, h.pos_x.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, h.pos_y.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pz, h.pos_z.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx, h.vel_x.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy, h.vel_y.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz, h.vel_z.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    };

    auto download = [&]() {
        CUDA_CHECK(cudaMemcpy(h.pos_x.data(), d_px, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.pos_y.data(), d_py, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.pos_z.data(), d_pz, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vel_x.data(), d_vx, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vel_y.data(), d_vy, N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vel_z.data(), d_vz, N*sizeof(float), cudaMemcpyDeviceToHost));
    };

    upload();

    // --- Warm-up ---
    fused_force_kernel<<<grid_2d, block_2d, shmem_bytes>>>(
        d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz,
        grid_w, grid_h, tile_w, tile_h);
    fused_integrate_collision_kernel<<<blocks_1d, BLOCK_1D>>>(
        d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    upload();  // reset state

    // --- Timed simulation ---
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    CUDA_CHECK(cudaEventRecord(start_ev));

    for (int s = 0; s < num_steps; ++s) {
        fused_force_kernel<<<grid_2d, block_2d, shmem_bytes>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz,
            grid_w, grid_h, tile_w, tile_h);
        fused_integrate_collision_kernel<<<blocks_1d, BLOCK_1D>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, N);
    }

    CUDA_CHECK(cudaEventRecord(stop_ev));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    download();

    bool has_nan = false;
    for (int i = 0; i < N; ++i) {
        if (std::isnan(h.pos_x[i]) || std::isnan(h.pos_y[i]) || std::isnan(h.pos_z[i]) ||
            std::isinf(h.pos_x[i]) || std::isinf(h.pos_y[i]) || std::isinf(h.pos_z[i])) {
            has_nan = true; break;
        }
    }
    if (has_nan) std::fprintf(stderr, "ERROR: NaN/Inf detected!\n");

    validate_against_reference(h, grid_w, grid_h);

    int num_springs = 0;
    for (int r = 0; r < grid_h; ++r)
        for (int c = 0; c < grid_w; ++c) {
            if (c+1 < grid_w) num_springs++;
            if (r+1 < grid_h) num_springs++;
            if (r+1 < grid_h && c+1 < grid_w) num_springs++;
            if (r+1 < grid_h && c-1 >= 0) num_springs++;
            if (c+2 < grid_w) num_springs++;
            if (r+2 < grid_h) num_springs++;
        }

    double steps_per_sec = num_steps / ((double)elapsed_ms / 1000.0);
    std::fprintf(stderr, "Springs: %d\n", num_springs);
    std::fprintf(stderr, "Done in %.3f ms  (%.1f steps/s)\n", elapsed_ms, steps_per_sec);

    std::printf("v4,%d,%d,%d,%d,%d,%.4f\n",
                grid_w, grid_h, N, num_springs, num_steps, elapsed_ms);

    // --- Optional frame-saving pass (separate from timed benchmark) ---
    if (!frames_path.empty()) {
        std::fprintf(stderr, "\nFrame-saving pass -> %s  (every %d steps)\n",
                     frames_path.c_str(), SAVE_EVERY);

        // Re-initialise host arrays to initial state (download() above overwrote them)
        init_particles_host(h, grid_w, grid_h, cloth_size, drop_y);
        upload();

        std::ofstream fout(frames_path);
        // Error checking 
        if (!fout) {
            std::fprintf(stderr, "ERROR: could not open frames file: %s\n", frames_path.c_str());
            return 1;
        }
        fout << "step,row,col,x,y,z\n";

        // Dump step 0
        for (int r = 0; r < grid_h; ++r)
            for (int c = 0; c < grid_w; ++c) {
                int i = r * grid_w + c;
                fout << 0 << ',' << r << ',' << c << ','
                     << h.pos_x[i] << ',' << h.pos_y[i] << ',' << h.pos_z[i] << '\n';
            }

        for (int s = 1; s <= num_steps; ++s) {
            fused_force_kernel<<<grid_2d, block_2d, shmem_bytes>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz,
                grid_w, grid_h, tile_w, tile_h);
            fused_integrate_collision_kernel<<<blocks_1d, BLOCK_1D>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, N);

            if (s % SAVE_EVERY == 0) {
                CUDA_CHECK(cudaDeviceSynchronize());
                download();
                for (int r = 0; r < grid_h; ++r)
                    for (int c = 0; c < grid_w; ++c) {
                        int i = r * grid_w + c;
                        fout << s << ',' << r << ',' << c << ','
                             << h.pos_x[i] << ',' << h.pos_y[i] << ',' << h.pos_z[i] << '\n';
                    }
            }
        }
        fout.close();
        std::fprintf(stderr, "Frames saved -> %s\n", frames_path.c_str());
    }

    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));
    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_fx)); CUDA_CHECK(cudaFree(d_fy)); CUDA_CHECK(cudaFree(d_fz));

    return 0;
}

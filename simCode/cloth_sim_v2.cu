// =============================================================================
// cloth_sim_v2.cu
// V2: SoA Memory Layout — coalesced global memory access
// Ankita Kundu & Michael Nguyen — 15-418/618 Final Project
//
// Key change from V1: Particle data stored as separate float arrays (SoA)
// instead of an array of Particle structs (AoS). This allows consecutive
// threads in a warp to read/write contiguous memory, enabling coalesced
// 128-byte transactions and dramatically improving effective bandwidth.
//
// Compile:  nvcc -O2 -std=c++17 -arch=sm_75 -o cloth_sim_v2 cloth_sim_v2.cu
// Run:      ./cloth_sim_v2 <GRID_W> <GRID_H> [NUM_STEPS]
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
// Simulation parameters
// ---------------------------------------------------------------------------
namespace Params {
    constexpr float CLOTH_SIZE   = 3.0f;
    constexpr float DROP_Y       = 3.0f;
    constexpr float DT           = 0.005f;

    constexpr float K_STRUCTURAL = 500.0f;
    constexpr float K_SHEAR      = 200.0f;
    constexpr float K_BEND       = 100.0f;

    constexpr float DAMPING      = 0.998f;
    constexpr float AIR_DRAG     = 0.01f;

    constexpr float GRAVITY_Y    = -9.81f;

    constexpr float GROUND_Y           = -1.0f;
    constexpr float GROUND_RESTITUTION = 0.05f;
    constexpr float GROUND_FRICTION    = 0.80f;

    constexpr bool  SPHERE_ENABLED = true;
    constexpr float SPHERE_RADIUS  = 0.6f;
    constexpr float SPHERE_CX     = 0.0f;
    constexpr float SPHERE_CY     = GROUND_Y + SPHERE_RADIUS;
    constexpr float SPHERE_CZ     = 0.0f;
}

// ---------------------------------------------------------------------------
// Adjacency structure (unchanged from V1)
// ---------------------------------------------------------------------------
struct AdjEntry {
    int   neighbor;
    float rest_length;
    float stiffness;
};

// ---------------------------------------------------------------------------
// Host setup (produces SoA arrays)
// ---------------------------------------------------------------------------
struct HostSoA {
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    int n;
};

void init_particles_host(HostSoA& soa, int grid_w, int grid_h) {
    int n = grid_w * grid_h;
    soa.n = n;
    soa.pos_x.resize(n); soa.pos_y.resize(n); soa.pos_z.resize(n);
    soa.vel_x.resize(n, 0); soa.vel_y.resize(n, 0); soa.vel_z.resize(n, 0);

    const float dx = Params::CLOTH_SIZE / (grid_w - 1);
    const float dz = Params::CLOTH_SIZE / (grid_h - 1);

    for (int r = 0; r < grid_h; ++r) {
        for (int c = 0; c < grid_w; ++c) {
            int i = r * grid_w + c;
            soa.pos_x[i] = c * dx - Params::CLOTH_SIZE * 0.5f;
            soa.pos_y[i] = Params::DROP_Y;
            soa.pos_z[i] = r * dz - Params::CLOTH_SIZE * 0.5f;
        }
    }
}

struct SpringHost {
    int a, b;
    float rest_length, stiffness;
};

void build_springs_host(std::vector<SpringHost>& springs,
                        const HostSoA& soa, int grid_w, int grid_h) {
    auto I = [grid_w](int r, int c) { return r * grid_w + c; };
    auto add = [&](int a, int b, float k) {
        float dx = soa.pos_x[b] - soa.pos_x[a];
        float dy = soa.pos_y[b] - soa.pos_y[a];
        float dz = soa.pos_z[b] - soa.pos_z[a];
        float rest = std::sqrt(dx*dx + dy*dy + dz*dz);
        springs.push_back({a, b, rest, k});
    };

    for (int r = 0; r < grid_h; ++r) {
        for (int c = 0; c < grid_w; ++c) {
            if (c+1 < grid_w) add(I(r,c), I(r,c+1), Params::K_STRUCTURAL);
            if (r+1 < grid_h) add(I(r,c), I(r+1,c), Params::K_STRUCTURAL);
            if (r+1 < grid_h && c+1 < grid_w) add(I(r,c), I(r+1,c+1), Params::K_SHEAR);
            if (r+1 < grid_h && c-1 >= 0)     add(I(r,c), I(r+1,c-1), Params::K_SHEAR);
            if (c+2 < grid_w) add(I(r,c), I(r,c+2), Params::K_BEND);
            if (r+2 < grid_h) add(I(r,c), I(r+2,c), Params::K_BEND);
        }
    }
}

void build_adjacency(const std::vector<SpringHost>& springs, int n,
                     std::vector<int>& offsets, std::vector<AdjEntry>& entries) {
    std::vector<std::vector<AdjEntry>> per_particle(n);
    for (const auto& s : springs) {
        per_particle[s.a].push_back({s.b, s.rest_length, s.stiffness});
        per_particle[s.b].push_back({s.a, s.rest_length, s.stiffness});
    }
    offsets.resize(n + 1);
    offsets[0] = 0;
    for (int i = 0; i < n; ++i)
        offsets[i + 1] = offsets[i] + (int)per_particle[i].size();
    entries.resize(offsets[n]);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < (int)per_particle[i].size(); ++j)
            entries[offsets[i] + j] = per_particle[i][j];
}

// ---------------------------------------------------------------------------
// CUDA Kernels — SoA layout
// ---------------------------------------------------------------------------

__global__
void clear_and_apply_external_kernel(float* fx, float* fy, float* fz,
                                     const float* vx, const float* vy, const float* vz,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Clear + gravity + drag in a single pass (fused from V1's 2 kernels)
    float mass = 0.1f;
    fx[i] = vx[i] * (-Params::AIR_DRAG);
    fy[i] = Params::GRAVITY_Y * mass + vy[i] * (-Params::AIR_DRAG);
    fz[i] = vz[i] * (-Params::AIR_DRAG);
}

__global__
void compute_spring_forces_kernel(const float* px, const float* py, const float* pz,
                                  float* fx, float* fy, float* fz,
                                  const int* adj_offsets, const AdjEntry* adj_entries,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float my_x = px[i], my_y = py[i], my_z = pz[i];
    float accum_x = 0, accum_y = 0, accum_z = 0;

    int start = adj_offsets[i];
    int end   = adj_offsets[i + 1];

    for (int e = start; e < end; ++e) {
        int nb = adj_entries[e].neighbor;
        float dx = px[nb] - my_x;
        float dy = py[nb] - my_y;
        float dz = pz[nb] - my_z;
        float len = sqrtf(dx*dx + dy*dy + dz*dz);
        if (len < 1e-9f) continue;
        float f = adj_entries[e].stiffness * (len - adj_entries[e].rest_length) / len;
        accum_x += dx * f;
        accum_y += dy * f;
        accum_z += dz * f;
    }

    fx[i] += accum_x;
    fy[i] += accum_y;
    fz[i] += accum_z;
}

__global__
void integrate_kernel(float* px, float* py, float* pz,
                      float* vx, float* vy, float* vz,
                      const float* fx, const float* fy, const float* fz,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float inv_mass = 10.0f;  // 1.0 / 0.1
    vx[i] += fx[i] * inv_mass * Params::DT;
    vy[i] += fy[i] * inv_mass * Params::DT;
    vz[i] += fz[i] * inv_mass * Params::DT;

    vx[i] *= Params::DAMPING;
    vy[i] *= Params::DAMPING;
    vz[i] *= Params::DAMPING;

    px[i] += vx[i] * Params::DT;
    py[i] += vy[i] * Params::DT;
    pz[i] += vz[i] * Params::DT;
}

__global__
void collision_kernel(float* px, float* py, float* pz,
                      float* vx, float* vy, float* vz,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (py[i] < Params::GROUND_Y) {
        py[i]  = Params::GROUND_Y;
        vy[i]  = fabsf(vy[i]) * Params::GROUND_RESTITUTION;
        vx[i] *= Params::GROUND_FRICTION;
        vz[i] *= Params::GROUND_FRICTION;
    }

    if (Params::SPHERE_ENABLED) {
        float dx = px[i] - Params::SPHERE_CX;
        float dy = py[i] - Params::SPHERE_CY;
        float dz = pz[i] - Params::SPHERE_CZ;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < Params::SPHERE_RADIUS && dist > 1e-9f) {
            float inv_d = 1.0f / dist;
            float nx = dx * inv_d, ny = dy * inv_d, nz = dz * inv_d;
            px[i] = Params::SPHERE_CX + nx * Params::SPHERE_RADIUS;
            py[i] = Params::SPHERE_CY + ny * Params::SPHERE_RADIUS;
            pz[i] = Params::SPHERE_CZ + nz * Params::SPHERE_RADIUS;
            float v_n = vx[i]*nx + vy[i]*ny + vz[i]*nz;
            if (v_n < 0.0f) {
                vx[i] += nx * (-v_n);
                vy[i] += ny * (-v_n);
                vz[i] += nz * (-v_n);
            }
        }
    }
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
    bool pass = max_dev < 1e-3f;
    // std::fprintf(stderr, "Validation vs %s: L2=%.6e  max=%.6e  %s\n",
    //              fname, l2_norm, (double)max_dev, pass ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int grid_w    = 25;
    int grid_h    = 25;
    int num_steps = 1200;

    if (argc >= 3) {
        grid_w = std::atoi(argv[1]);
        grid_h = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        num_steps = std::atoi(argv[3]);
    }

    const int N = grid_w * grid_h;
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::fprintf(stderr, "=== V2: SoA Layout CUDA ===\n");
    std::fprintf(stderr, "Grid:  %d x %d  (%d particles)\n", grid_w, grid_h, N);
    std::fprintf(stderr, "Steps: %d   dt=%.4f\n", num_steps, Params::DT);
    std::fprintf(stderr, "Block size: %d   Blocks: %d\n\n", BLOCK_SIZE, num_blocks);

    // --- Host setup ---
    HostSoA h;
    std::vector<SpringHost> h_springs;
    init_particles_host(h, grid_w, grid_h);
    build_springs_host(h_springs, h, grid_w, grid_h);

    const int num_springs = (int)h_springs.size();
    std::fprintf(stderr, "Particles: %d\nSprings:   %d\n", N, num_springs);

    std::vector<int>      h_adj_offsets;
    std::vector<AdjEntry> h_adj_entries;
    build_adjacency(h_springs, N, h_adj_offsets, h_adj_entries);
    std::fprintf(stderr, "Adjacency entries: %d\n\n", (int)h_adj_entries.size());

    // --- Device allocation (SoA) ---
    float *d_px, *d_py, *d_pz;
    float *d_vx, *d_vy, *d_vz;
    float *d_fx, *d_fy, *d_fz;
    int      *d_adj_offsets;
    AdjEntry *d_adj_entries;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_adj_offsets, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj_entries, h_adj_entries.size() * sizeof(AdjEntry)));

    auto upload_state = [&]() {
        CUDA_CHECK(cudaMemcpy(d_px, h.pos_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, h.pos_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pz, h.pos_z.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx, h.vel_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy, h.vel_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz, h.vel_z.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    };

    auto download_state = [&]() {
        CUDA_CHECK(cudaMemcpy(h.pos_x.data(), d_px, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.pos_y.data(), d_py, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.pos_z.data(), d_pz, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vel_x.data(), d_vx, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vel_y.data(), d_vy, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vel_z.data(), d_vz, N * sizeof(float), cudaMemcpyDeviceToHost));
    };

    upload_state();
    CUDA_CHECK(cudaMemcpy(d_adj_offsets, h_adj_offsets.data(), (N+1)*sizeof(int),                       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_entries, h_adj_entries.data(), h_adj_entries.size()*sizeof(AdjEntry),    cudaMemcpyHostToDevice));

    // --- Warm-up ---
    clear_and_apply_external_kernel<<<num_blocks, BLOCK_SIZE>>>(d_fx, d_fy, d_fz, d_vx, d_vy, d_vz, N);
    compute_spring_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz, d_fx, d_fy, d_fz, d_adj_offsets, d_adj_entries, N);
    integrate_kernel<<<num_blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, N);
    collision_kernel<<<num_blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-upload after warm-up
    upload_state();

    // --- Timed simulation ---
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    CUDA_CHECK(cudaEventRecord(start_ev));

    for (int s = 0; s < num_steps; ++s) {
        clear_and_apply_external_kernel<<<num_blocks, BLOCK_SIZE>>>(d_fx, d_fy, d_fz, d_vx, d_vy, d_vz, N);
        compute_spring_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz, d_fx, d_fy, d_fz, d_adj_offsets, d_adj_entries, N);
        integrate_kernel<<<num_blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, N);
        collision_kernel<<<num_blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, N);
    }

    CUDA_CHECK(cudaEventRecord(stop_ev));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    // --- Copy back & validate ---
    download_state();

    bool has_nan = false;
    for (int i = 0; i < N; ++i) {
        if (std::isnan(h.pos_x[i]) || std::isnan(h.pos_y[i]) || std::isnan(h.pos_z[i]) ||
            std::isinf(h.pos_x[i]) || std::isinf(h.pos_y[i]) || std::isinf(h.pos_z[i])) {
            has_nan = true; break;
        }
    }
    if (has_nan) std::fprintf(stderr, "ERROR: NaN/Inf detected in output!\n");

    validate_against_reference(h, grid_w, grid_h);

    double steps_per_sec = num_steps / ((double)elapsed_ms / 1000.0);
    std::fprintf(stderr, "Done in %.3f ms  (%.1f steps/s)\n", elapsed_ms, steps_per_sec);

    std::printf("v2,%d,%d,%d,%d,%d,%.4f\n",
                grid_w, grid_h, N, num_springs, num_steps, elapsed_ms);

    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));
    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_fx)); CUDA_CHECK(cudaFree(d_fy)); CUDA_CHECK(cudaFree(d_fz));
    CUDA_CHECK(cudaFree(d_adj_offsets));
    CUDA_CHECK(cudaFree(d_adj_entries));

    return 0;
}

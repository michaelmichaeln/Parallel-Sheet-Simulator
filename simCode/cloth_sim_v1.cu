// =============================================================================
// cloth_sim_v1.cu
// V1: Naive CUDA Port — AoS layout, node-parallel, 5 separate kernels
// Ankita Kundu & Michael Nguyen — 15-418/618 Final Project
//
// Compile:  nvcc -O2 -std=c++17 -arch=sm_75 -o cloth_sim_v1 cloth_sim_v1.cu
// Run:      ./cloth_sim_v1 <GRID_W> <GRID_H> [NUM_STEPS]
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
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
// Vec3 (host + device)
// ---------------------------------------------------------------------------
struct Vec3 {
    float x = 0, y = 0, z = 0;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3  operator+(const Vec3& o)  const { return {x+o.x, y+o.y, z+o.z}; }
    __host__ __device__ Vec3  operator-(const Vec3& o)  const { return {x-o.x, y-o.y, z-o.z}; }
    __host__ __device__ Vec3  operator*(float s)        const { return {x*s,   y*s,   z*s};   }
    __host__ __device__ Vec3  operator-()               const { return {-x,    -y,    -z};    }
    __host__ __device__ Vec3& operator+=(const Vec3& o)       { x+=o.x; y+=o.y; z+=o.z; return *this; }
    __host__ __device__ Vec3& operator*=(float s)             { x*=s;   y*=s;   z*=s;   return *this; }

    __host__ __device__ float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    __host__ __device__ float length()           const { return sqrtf(x*x + y*y + z*z); }
    __host__ __device__ Vec3  normalized()       const {
        float l = length();
        return l > 1e-9f ? (*this)*(1.f/l) : Vec3();
    }
};

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
// Data structures (AoS — same as sequential)
// ---------------------------------------------------------------------------
struct Particle {
    Vec3  pos, vel, force;
    float mass;
    int   pinned;  // int instead of bool for GPU alignment
};

struct Spring {
    int   a, b;
    float rest_length;
    float stiffness;
};

// Per-particle adjacency: which springs connect to this particle, and
// which end of the spring this particle is (so we know the neighbor index).
struct AdjEntry {
    int   neighbor;     // index of the other particle
    float rest_length;
    float stiffness;
};

// ---------------------------------------------------------------------------
// Host initialization
// ---------------------------------------------------------------------------
void init_particles_host(std::vector<Particle>& particles, int grid_w, int grid_h) {
    particles.resize(grid_w * grid_h);
    const float dx = Params::CLOTH_SIZE / (grid_w - 1);
    const float dz = Params::CLOTH_SIZE / (grid_h - 1);

    for (int r = 0; r < grid_h; ++r) {
        for (int c = 0; c < grid_w; ++c) {
            Particle& p = particles[r * grid_w + c];
            p.pos    = { c * dx - Params::CLOTH_SIZE * 0.5f,
                         Params::DROP_Y,
                         r * dz - Params::CLOTH_SIZE * 0.5f };
            p.vel    = Vec3();
            p.force  = Vec3();
            p.mass   = 0.1f;
            p.pinned = 0;
        }
    }
}

void build_springs_host(std::vector<Spring>& springs,
                        const std::vector<Particle>& particles,
                        int grid_w, int grid_h) {
    auto I = [grid_w](int r, int c) { return r * grid_w + c; };
    auto add = [&](int a, int b, float k) {
        float rest = (particles[a].pos - particles[b].pos).length();
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

// Build CSR-like adjacency from springs: for each particle, list its neighbors
void build_adjacency(const std::vector<Spring>& springs, int num_particles,
                     std::vector<int>& offsets, std::vector<AdjEntry>& entries) {
    std::vector<std::vector<AdjEntry>> per_particle(num_particles);

    for (const auto& s : springs) {
        per_particle[s.a].push_back({s.b, s.rest_length, s.stiffness});
        per_particle[s.b].push_back({s.a, s.rest_length, s.stiffness});
    }

    offsets.resize(num_particles + 1);
    offsets[0] = 0;
    for (int i = 0; i < num_particles; ++i)
        offsets[i + 1] = offsets[i] + (int)per_particle[i].size();

    entries.resize(offsets[num_particles]);
    for (int i = 0; i < num_particles; ++i)
        for (int j = 0; j < (int)per_particle[i].size(); ++j)
            entries[offsets[i] + j] = per_particle[i][j];
}

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

__global__
void clear_forces_kernel(Particle* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    particles[i].force = Vec3();
}

__global__
void apply_external_forces_kernel(Particle* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle& p = particles[i];
    if (p.pinned) return;

    p.force.x += 0.0f;
    p.force.y += Params::GRAVITY_Y * p.mass;
    p.force.z += 0.0f;

    p.force.x += p.vel.x * (-Params::AIR_DRAG);
    p.force.y += p.vel.y * (-Params::AIR_DRAG);
    p.force.z += p.vel.z * (-Params::AIR_DRAG);
}

__global__
void compute_spring_forces_kernel(Particle* particles, int n,
                                  const int* adj_offsets,
                                  const AdjEntry* adj_entries) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (particles[i].pinned) return;

    Vec3 my_pos = particles[i].pos;
    Vec3 f_accum = Vec3();

    int start = adj_offsets[i];
    int end   = adj_offsets[i + 1];

    for (int e = start; e < end; ++e) {
        const AdjEntry& adj = adj_entries[e];
        Vec3 other_pos = particles[adj.neighbor].pos;
        Vec3 diff = other_pos - my_pos;
        float len = diff.length();
        if (len < 1e-9f) continue;
        float force_mag = adj.stiffness * (len - adj.rest_length);
        f_accum += diff * (force_mag / len);
    }

    particles[i].force += f_accum;
}

__global__
void integrate_kernel(Particle* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle& p = particles[i];
    if (p.pinned) return;

    float inv_mass = 1.0f / p.mass;
    p.vel.x += p.force.x * inv_mass * Params::DT;
    p.vel.y += p.force.y * inv_mass * Params::DT;
    p.vel.z += p.force.z * inv_mass * Params::DT;

    p.vel.x *= Params::DAMPING;
    p.vel.y *= Params::DAMPING;
    p.vel.z *= Params::DAMPING;

    p.pos.x += p.vel.x * Params::DT;
    p.pos.y += p.vel.y * Params::DT;
    p.pos.z += p.vel.z * Params::DT;
}

__global__
void collision_kernel(Particle* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Particle& p = particles[i];

    // Ground plane
    if (p.pos.y < Params::GROUND_Y) {
        p.pos.y  = Params::GROUND_Y;
        p.vel.y  = fabsf(p.vel.y) * Params::GROUND_RESTITUTION;
        p.vel.x *= Params::GROUND_FRICTION;
        p.vel.z *= Params::GROUND_FRICTION;
    }

    // Sphere collision
    if (Params::SPHERE_ENABLED) {
        float dx = p.pos.x - Params::SPHERE_CX;
        float dy = p.pos.y - Params::SPHERE_CY;
        float dz = p.pos.z - Params::SPHERE_CZ;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < Params::SPHERE_RADIUS && dist > 1e-9f) {
            float inv_dist = 1.0f / dist;
            float nx = dx * inv_dist;
            float ny = dy * inv_dist;
            float nz = dz * inv_dist;
            p.pos.x = Params::SPHERE_CX + nx * Params::SPHERE_RADIUS;
            p.pos.y = Params::SPHERE_CY + ny * Params::SPHERE_RADIUS;
            p.pos.z = Params::SPHERE_CZ + nz * Params::SPHERE_RADIUS;
            float v_n = p.vel.x * nx + p.vel.y * ny + p.vel.z * nz;
            if (v_n < 0.0f) {
                p.vel.x += nx * (-v_n);
                p.vel.y += ny * (-v_n);
                p.vel.z += nz * (-v_n);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validation against CPU reference
// ---------------------------------------------------------------------------
void validate_against_reference(const std::vector<Particle>& particles,
                                int grid_w, int grid_h) {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "results/outputs/ref_%dx%d.bin", grid_w, grid_h);
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Warning: no reference file %s — skipping validation\n", fname);
        return;
    }

    int ref_n;
    in.read(reinterpret_cast<char*>(&ref_n), sizeof(int));
    if (ref_n != (int)particles.size()) {
        std::fprintf(stderr, "Warning: reference size mismatch (%d vs %d)\n",
                     ref_n, (int)particles.size());
        return;
    }

    double l2_sum = 0.0;
    float  max_dev = 0.0f;
    for (int i = 0; i < ref_n; ++i) {
        float rx, ry, rz;
        in.read(reinterpret_cast<char*>(&rx), sizeof(float));
        in.read(reinterpret_cast<char*>(&ry), sizeof(float));
        in.read(reinterpret_cast<char*>(&rz), sizeof(float));
        float dx = particles[i].pos.x - rx;
        float dy = particles[i].pos.y - ry;
        float dz = particles[i].pos.z - rz;
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

    std::fprintf(stderr, "=== V1: Naive CUDA (AoS, node-parallel) ===\n");
    std::fprintf(stderr, "Grid:  %d x %d  (%d particles)\n", grid_w, grid_h, N);
    std::fprintf(stderr, "Steps: %d   dt=%.4f\n", num_steps, Params::DT);
    std::fprintf(stderr, "Block size: %d   Blocks: %d\n\n", BLOCK_SIZE, num_blocks);

    // --- Host setup ---
    std::vector<Particle> h_particles;
    std::vector<Spring>   h_springs;
    init_particles_host(h_particles, grid_w, grid_h);
    build_springs_host(h_springs, h_particles, grid_w, grid_h);

    const int num_springs = (int)h_springs.size();
    std::fprintf(stderr, "Particles: %d\nSprings:   %d\n", N, num_springs);

    std::vector<int>      h_adj_offsets;
    std::vector<AdjEntry> h_adj_entries;
    build_adjacency(h_springs, N, h_adj_offsets, h_adj_entries);
    std::fprintf(stderr, "Adjacency entries: %d\n\n", (int)h_adj_entries.size());

    // --- Device allocation ---
    Particle* d_particles;
    int*      d_adj_offsets;
    AdjEntry* d_adj_entries;

    CUDA_CHECK(cudaMalloc(&d_particles,   N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_adj_offsets,  (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj_entries,  h_adj_entries.size() * sizeof(AdjEntry)));

    CUDA_CHECK(cudaMemcpy(d_particles,   h_particles.data(),   N * sizeof(Particle),                     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_offsets,  h_adj_offsets.data(), (N + 1) * sizeof(int),                   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_entries,  h_adj_entries.data(), h_adj_entries.size() * sizeof(AdjEntry),  cudaMemcpyHostToDevice));

    // --- Warm-up ---
    clear_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N);
    apply_external_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N);
    compute_spring_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N, d_adj_offsets, d_adj_entries);
    integrate_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N);
    collision_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-upload initial state after warm-up
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles.data(), N * sizeof(Particle), cudaMemcpyHostToDevice));

    // --- Timed simulation ---
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    CUDA_CHECK(cudaEventRecord(start_ev));

    for (int s = 0; s < num_steps; ++s) {
        clear_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N);
        apply_external_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N);
        compute_spring_forces_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N, d_adj_offsets, d_adj_entries);
        integrate_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N);
        collision_kernel<<<num_blocks, BLOCK_SIZE>>>(d_particles, N);
    }

    CUDA_CHECK(cudaEventRecord(stop_ev));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    // --- Copy back & validate ---
    CUDA_CHECK(cudaMemcpy(h_particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost));

    // NaN/Inf check
    bool has_nan = false;
    for (int i = 0; i < N; ++i) {
        const auto& p = h_particles[i];
        if (std::isnan(p.pos.x) || std::isnan(p.pos.y) || std::isnan(p.pos.z) ||
            std::isinf(p.pos.x) || std::isinf(p.pos.y) || std::isinf(p.pos.z)) {
            has_nan = true;
            break;
        }
    }
    if (has_nan) std::fprintf(stderr, "ERROR: NaN/Inf detected in output!\n");

    validate_against_reference(h_particles, grid_w, grid_h);

    double steps_per_sec = num_steps / ((double)elapsed_ms / 1000.0);
    std::fprintf(stderr, "Done in %.3f ms  (%.1f steps/s)\n", elapsed_ms, steps_per_sec);

    // CSV output to stdout
    std::printf("v1,%d,%d,%d,%d,%d,%.4f\n",
                grid_w, grid_h, N, num_springs, num_steps, elapsed_ms);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));
    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_adj_offsets));
    CUDA_CHECK(cudaFree(d_adj_entries));

    return 0;
}

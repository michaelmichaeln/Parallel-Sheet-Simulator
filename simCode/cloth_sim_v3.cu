// =============================================================================
// cloth_sim_v3.cu
// V3: Shared Memory Tiling — 2D thread blocks, halo-based spring forces
// Ankita Kundu & Michael Nguyen — 15-418/618 Final Project
//
// Key change from V2: Instead of CSR adjacency lookups into global memory,
// we exploit the regular grid topology. Each 2D thread block loads a tile of
// position data plus a 2-cell halo (needed for bend springs) into shared
// memory. Spring force computation reads from shared memory for all neighbors,
// eliminating redundant global memory reads across threads in a block.
//
// Compile:  nvcc -O2 -std=c++17 -arch=sm_75 -o cloth_sim_v3 cloth_sim_v3.cu
// Run:      ./cloth_sim_v3 <GRID_W> <GRID_H> [NUM_STEPS]
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

// Tile dimensions for the spring force kernel
constexpr int TILE_W = 16;
constexpr int TILE_H = 16;
constexpr int HALO   = 2;  // bend springs reach 2 cells away
constexpr int SM_W   = TILE_W + 2 * HALO;   // 20
constexpr int SM_H   = TILE_H + 2 * HALO;   // 20

// ---------------------------------------------------------------------------
// Host setup (SoA — same as V2)
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

    for (int r = 0; r < grid_h; ++r)
        for (int c = 0; c < grid_w; ++c) {
            int i = r * grid_w + c;
            soa.pos_x[i] = c * dx - Params::CLOTH_SIZE * 0.5f;
            soa.pos_y[i] = Params::DROP_Y;
            soa.pos_z[i] = r * dz - Params::CLOTH_SIZE * 0.5f;
        }
}

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

// Same simple kernels for clear/external, integrate, collision (from V2)
__global__
void clear_and_apply_external_kernel(float* fx, float* fy, float* fz,
                                     const float* vx, const float* vy, const float* vz,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mass = 0.1f;
    fx[i] = vx[i] * (-Params::AIR_DRAG);
    fy[i] = Params::GRAVITY_Y * mass + vy[i] * (-Params::AIR_DRAG);
    fz[i] = vz[i] * (-Params::AIR_DRAG);
}

// Spring forces using shared memory tile + halo.
// Thread block is TILE_W x TILE_H; each block owns a tile of the grid.
// We load a (TILE_W+4) x (TILE_H+4) region of positions into shared memory.
__global__
void spring_forces_tiled_kernel(const float* __restrict__ px,
                                const float* __restrict__ py,
                                const float* __restrict__ pz,
                                float* __restrict__ fx,
                                float* __restrict__ fy,
                                float* __restrict__ fz,
                                int grid_w, int grid_h) {
    // Shared memory for the tile + halo (3 floats per position)
    __shared__ float s_px[SM_H][SM_W];
    __shared__ float s_py[SM_H][SM_W];
    __shared__ float s_pz[SM_H][SM_W];

    // Global grid coords for this thread's particle
    int g_col = blockIdx.x * TILE_W + threadIdx.x;
    int g_row = blockIdx.y * TILE_H + threadIdx.y;

    // Top-left corner of this tile in global coords (including halo offset)
    int tile_r0 = blockIdx.y * TILE_H - HALO;
    int tile_c0 = blockIdx.x * TILE_W - HALO;

    // Cooperatively load the SM_W x SM_H region into shared memory.
    // Each thread may need to load multiple elements.
    int tid = threadIdx.y * TILE_W + threadIdx.x;
    int total_sm = SM_W * SM_H;
    int threads_per_block = TILE_W * TILE_H;

    for (int idx = tid; idx < total_sm; idx += threads_per_block) {
        int sr = idx / SM_W;
        int sc = idx % SM_W;
        int gr = tile_r0 + sr;
        int gc = tile_c0 + sc;

        // Clamp to grid bounds (repeat edge)
        gr = max(0, min(gr, grid_h - 1));
        gc = max(0, min(gc, grid_w - 1));

        int gi = gr * grid_w + gc;
        s_px[sr][sc] = px[gi];
        s_py[sr][sc] = py[gi];
        s_pz[sr][sc] = pz[gi];
    }

    __syncthreads();

    if (g_col >= grid_w || g_row >= grid_h) return;

    // This thread's position in shared memory
    int lr = threadIdx.y + HALO;
    int lc = threadIdx.x + HALO;

    float my_x = s_px[lr][lc];
    float my_y = s_py[lr][lc];
    float my_z = s_pz[lr][lc];

    float ax = 0, ay = 0, az = 0;

    // Precompute rest lengths from initial grid spacing
    float dx_struct = Params::CLOTH_SIZE / (grid_w - 1);
    float dz_struct = Params::CLOTH_SIZE / (grid_h - 1);
    float rest_struct_h = dx_struct;                               // horizontal neighbor
    float rest_struct_v = dz_struct;                               // vertical neighbor
    float rest_shear    = sqrtf(dx_struct*dx_struct + dz_struct*dz_struct);
    float rest_bend_h   = 2.0f * dx_struct;
    float rest_bend_v   = 2.0f * dz_struct;

    // Inline spring force accumulation for all 12 neighbor types
    // Structural: (0,+1) (0,-1) (+1,0) (-1,0)
    #define DO_SPRING(dr, dc, K, REST)                                         \
    {                                                                           \
        int nr = g_row + (dr);                                                  \
        int nc = g_col + (dc);                                                  \
        if (nr >= 0 && nr < grid_h && nc >= 0 && nc < grid_w) {                \
            float nx = s_px[lr+(dr)][lc+(dc)];                                  \
            float ny = s_py[lr+(dr)][lc+(dc)];                                  \
            float nz = s_pz[lr+(dr)][lc+(dc)];                                  \
            float ddx = nx - my_x;                                              \
            float ddy = ny - my_y;                                              \
            float ddz = nz - my_z;                                              \
            float len = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);                     \
            if (len > 1e-9f) {                                                  \
                float f = (K) * (len - (REST)) / len;                           \
                ax += ddx * f;  ay += ddy * f;  az += ddz * f;                  \
            }                                                                   \
        }                                                                       \
    }

    // Structural (4 neighbors)
    DO_SPRING( 0,  1, Params::K_STRUCTURAL, rest_struct_h);
    DO_SPRING( 0, -1, Params::K_STRUCTURAL, rest_struct_h);
    DO_SPRING( 1,  0, Params::K_STRUCTURAL, rest_struct_v);
    DO_SPRING(-1,  0, Params::K_STRUCTURAL, rest_struct_v);

    // Shear (4 diagonal neighbors)
    DO_SPRING( 1,  1, Params::K_SHEAR, rest_shear);
    DO_SPRING( 1, -1, Params::K_SHEAR, rest_shear);
    DO_SPRING(-1,  1, Params::K_SHEAR, rest_shear);
    DO_SPRING(-1, -1, Params::K_SHEAR, rest_shear);

    // Bend (4 skip-one neighbors)
    DO_SPRING( 0,  2, Params::K_BEND, rest_bend_h);
    DO_SPRING( 0, -2, Params::K_BEND, rest_bend_h);
    DO_SPRING( 2,  0, Params::K_BEND, rest_bend_v);
    DO_SPRING(-2,  0, Params::K_BEND, rest_bend_v);

    #undef DO_SPRING

    int gi = g_row * grid_w + g_col;
    fx[gi] += ax;
    fy[gi] += ay;
    fz[gi] += az;
}

__global__
void integrate_kernel(float* px, float* py, float* pz,
                      float* vx, float* vy, float* vz,
                      const float* fx, const float* fy, const float* fz,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float inv_mass = 10.0f;
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

    if (argc >= 3) {
        grid_w = std::atoi(argv[1]);
        grid_h = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        num_steps = std::atoi(argv[3]);
    }

    const int N = grid_w * grid_h;

    // 1D launch params for simple kernels
    constexpr int BLOCK_1D = 256;
    int blocks_1d = (N + BLOCK_1D - 1) / BLOCK_1D;

    // 2D launch params for tiled spring kernel
    dim3 block_2d(TILE_W, TILE_H);
    dim3 grid_2d((grid_w + TILE_W - 1) / TILE_W,
                 (grid_h + TILE_H - 1) / TILE_H);

    std::fprintf(stderr, "=== V3: Shared Memory Tiling CUDA ===\n");
    std::fprintf(stderr, "Grid:  %d x %d  (%d particles)\n", grid_w, grid_h, N);
    std::fprintf(stderr, "Steps: %d   dt=%.4f\n", num_steps, Params::DT);
    std::fprintf(stderr, "Tile:  %dx%d  Halo: %d  Shared mem tile: %dx%d\n",
                 TILE_W, TILE_H, HALO, SM_W, SM_H);
    std::fprintf(stderr, "2D grid: (%d, %d)  block: (%d, %d)\n\n",
                 grid_2d.x, grid_2d.y, block_2d.x, block_2d.y);

    // --- Host setup ---
    HostSoA h;
    init_particles_host(h, grid_w, grid_h);

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
    clear_and_apply_external_kernel<<<blocks_1d, BLOCK_1D>>>(d_fx, d_fy, d_fz, d_vx, d_vy, d_vz, N);
    spring_forces_tiled_kernel<<<grid_2d, block_2d>>>(d_px, d_py, d_pz, d_fx, d_fy, d_fz, grid_w, grid_h);
    integrate_kernel<<<blocks_1d, BLOCK_1D>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, N);
    collision_kernel<<<blocks_1d, BLOCK_1D>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    upload();  // reset state

    // --- Timed simulation ---
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    CUDA_CHECK(cudaEventRecord(start_ev));

    for (int s = 0; s < num_steps; ++s) {
        clear_and_apply_external_kernel<<<blocks_1d, BLOCK_1D>>>(d_fx, d_fy, d_fz, d_vx, d_vy, d_vz, N);
        spring_forces_tiled_kernel<<<grid_2d, block_2d>>>(d_px, d_py, d_pz, d_fx, d_fy, d_fz, grid_w, grid_h);
        integrate_kernel<<<blocks_1d, BLOCK_1D>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, N);
        collision_kernel<<<blocks_1d, BLOCK_1D>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, N);
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

    // Count springs for CSV output
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

    std::printf("v3,%d,%d,%d,%d,%d,%.4f\n",
                grid_w, grid_h, N, num_springs, num_steps, elapsed_ms);

    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));
    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_fx)); CUDA_CHECK(cudaFree(d_fy)); CUDA_CHECK(cudaFree(d_fz));

    return 0;
}

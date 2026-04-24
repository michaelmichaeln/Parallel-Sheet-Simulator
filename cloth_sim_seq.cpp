// =============================================================================
// cloth_sim_seq.cpp
// Sequential Mass-Spring Cloth Simulator  (benchmarkable version)
// Ankita Kundu & Michael Nguyen — 15-418/618 Final Project
//
// Compile:  g++ -O2 -std=c++17 -o cloth_sim_seq cloth_sim_seq.cpp
// Run:      ./cloth_sim_seq <GRID_W> <GRID_H> [NUM_STEPS]
//           ./cloth_sim_seq                  (defaults: 25 25 1200)
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------
struct Vec3 {
    float x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3  operator+(const Vec3& o)  const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3  operator-(const Vec3& o)  const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3  operator*(float s)        const { return {x*s,   y*s,   z*s};   }
    Vec3  operator-()               const { return {-x,    -y,    -z};    }
    Vec3& operator+=(const Vec3& o)       { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator*=(float s)             { x*=s;   y*=s;   z*=s;   return *this; }

    float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    float length()           const { return std::sqrt(x*x + y*y + z*z); }
    Vec3  normalized()       const {
        float l = length();
        return l > 1e-9f ? (*this)*(1.f/l) : Vec3{};
    }
};

// ---------------------------------------------------------------------------
// Simulation parameters (compile-time constants except grid size / steps)
// ---------------------------------------------------------------------------
namespace Params {
    constexpr float CLOTH_SIZE   = 3.0f;

    constexpr float DROP_Y       = 3.0f;
    constexpr float DT           = 0.005f;
    constexpr int   SAVE_EVERY   = 10;

    constexpr float K_STRUCTURAL = 500.0f;
    constexpr float K_SHEAR      = 200.0f;
    constexpr float K_BEND       = 100.0f;

    constexpr float DAMPING      = 0.998f;
    constexpr float AIR_DRAG     = 0.01f;

    const     Vec3  GRAVITY      = {0.0f, -9.81f, 0.0f};

    constexpr float GROUND_Y           = -1.0f;
    constexpr float GROUND_RESTITUTION = 0.05f;
    constexpr float GROUND_FRICTION    = 0.80f;

    constexpr bool  SPHERE_ENABLED = true;
    constexpr float SPHERE_RADIUS  = 0.6f;
    const     Vec3  SPHERE_CENTER  = {0.0f, GROUND_Y + SPHERE_RADIUS, 0.0f};
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct Particle {
    Vec3  pos, vel, force;
    float mass;
    bool  pinned;
};

struct Spring {
    int   a, b;
    float rest_length;
    float stiffness;
};

// ---------------------------------------------------------------------------
// Initialise particles
// ---------------------------------------------------------------------------
void init_particles(std::vector<Particle>& particles, int grid_w, int grid_h) {
    particles.resize(grid_w * grid_h);

    const float dx = Params::CLOTH_SIZE / (grid_w - 1);
    const float dz = Params::CLOTH_SIZE / (grid_h - 1);

    for (int r = 0; r < grid_h; ++r) {
        for (int c = 0; c < grid_w; ++c) {
            Particle& p = particles[r * grid_w + c];
            p.pos    = { c * dx - Params::CLOTH_SIZE * 0.5f,
                         Params::DROP_Y,
                         r * dz - Params::CLOTH_SIZE * 0.5f };
            p.vel    = {};
            p.force  = {};
            p.mass   = 0.1f;
            p.pinned = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Build spring network (structural + shear + bend)
// ---------------------------------------------------------------------------
void build_springs(std::vector<Spring>& springs,
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

// ---------------------------------------------------------------------------
// One simulation timestep
// ---------------------------------------------------------------------------
void step(std::vector<Particle>& particles,
          const std::vector<Spring>& springs) {

    for (auto& p : particles) p.force = {};

    for (auto& p : particles) {
        if (p.pinned) continue;
        p.force += Params::GRAVITY * p.mass;
        p.force += p.vel * (-Params::AIR_DRAG);
    }

    for (const auto& s : springs) {
        Particle& a = particles[s.a];
        Particle& b = particles[s.b];
        Vec3  diff = b.pos - a.pos;
        float len  = diff.length();
        if (len < 1e-9f) continue;
        Vec3  f = diff * (1.0f / len) * (s.stiffness * (len - s.rest_length));
        if (!a.pinned) a.force += f;
        if (!b.pinned) b.force += -f;
    }

    for (auto& p : particles) {
        if (p.pinned) continue;
        p.vel += (p.force * (1.0f / p.mass)) * Params::DT;
        p.vel *= Params::DAMPING;
        p.pos += p.vel * Params::DT;
    }

    for (auto& p : particles) {
        if (p.pos.y < Params::GROUND_Y) {
            p.pos.y  = Params::GROUND_Y;
            p.vel.y  = std::abs(p.vel.y) * Params::GROUND_RESTITUTION;
            p.vel.x *= Params::GROUND_FRICTION;
            p.vel.z *= Params::GROUND_FRICTION;
        }
    }

    if (Params::SPHERE_ENABLED) {
        for (auto& p : particles) {
            Vec3  to_p = p.pos - Params::SPHERE_CENTER;
            float dist = to_p.length();
            if (dist < Params::SPHERE_RADIUS) {
                Vec3  n   = to_p.normalized();
                p.pos     = Params::SPHERE_CENTER + n * Params::SPHERE_RADIUS;
                float v_n = p.vel.dot(n);
                if (v_n < 0.0f) p.vel += n * (-v_n);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dump final positions as binary reference for CUDA validation
// ---------------------------------------------------------------------------
void dump_reference(const std::vector<Particle>& particles,
                    int grid_w, int grid_h) {
    char fname[256];
    std::snprintf(fname, sizeof(fname), "results/outputs/ref_%dx%d.bin", grid_w, grid_h);
    std::ofstream out(fname, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "Warning: could not write reference file %s\n", fname);
        return;
    }
    int n = (int)particles.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(int));
    for (const auto& p : particles) {
        out.write(reinterpret_cast<const char*>(&p.pos.x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.pos.y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&p.pos.z), sizeof(float));
    }
    std::fprintf(stderr, "Reference -> %s\n", fname);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int grid_w    = 25;
    int grid_h    = 25;
    int num_steps = 1200;
    bool write_frames = false;

    if (argc >= 3) {
        grid_w = std::atoi(argv[1]);
        grid_h = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        num_steps = std::atoi(argv[3]);
    }
    if (argc >= 5 && std::string(argv[4]) == "--frames") {
        write_frames = true;
    }

    const int num_particles = grid_w * grid_h;

    std::fprintf(stderr, "=== Sequential Cloth Simulator ===\n");
    std::fprintf(stderr, "Grid:  %d x %d  (%d particles)\n", grid_w, grid_h, num_particles);
    std::fprintf(stderr, "Steps: %d   dt=%.4fs\n", num_steps, Params::DT);
    std::fprintf(stderr, "Sphere: %s  radius=%.2f\n\n",
                 Params::SPHERE_ENABLED ? "ON" : "OFF", Params::SPHERE_RADIUS);

    std::vector<Particle> particles;
    std::vector<Spring>   springs;
    init_particles(particles, grid_w, grid_h);
    build_springs(springs, particles, grid_w, grid_h);

    const int num_springs = (int)springs.size();
    std::fprintf(stderr, "Particles: %d\nSprings:   %d\n\n", num_particles, num_springs);

    std::ofstream frame_out;
    if (write_frames) {
        frame_out.open("cloth_frames.csv");
        frame_out << "step,row,col,x,y,z\n";
    }

    auto dump_frame = [&](int step_num) {
        if (!write_frames) return;
        for (int r = 0; r < grid_h; ++r)
            for (int c = 0; c < grid_w; ++c) {
                const Vec3& pos = particles[r * grid_w + c].pos;
                frame_out << step_num << ',' << r << ',' << c << ','
                          << pos.x << ',' << pos.y << ',' << pos.z << '\n';
            }
    };

    dump_frame(0);

    // --- Simulation loop ---
    // When write_frames is true, I/O is interleaved (timing will include I/O overhead).
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int s = 1; s <= num_steps; ++s) {
        step(particles, springs);
        if (write_frames && s % Params::SAVE_EVERY == 0) {
            dump_frame(s);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    double steps_per_sec = num_steps / (elapsed_ms / 1000.0);

    std::fprintf(stderr, "Done in %.3f ms  (%.1f steps/s)\n", elapsed_ms, steps_per_sec);

    // CSV timing output to stdout for benchmark scripts
    // Format: version,grid_w,grid_h,num_particles,num_springs,num_steps,elapsed_ms
    std::printf("seq,%d,%d,%d,%d,%d,%.4f\n",
                grid_w, grid_h, num_particles, num_springs, num_steps, elapsed_ms);

    dump_reference(particles, grid_w, grid_h);

    return 0;
}

// =============================================================================
// cloth_sim_sequential.cpp
// Sequential Mass-Spring Cloth Simulator
// Ankita Kundu & Michael Nguyen — 15-418/618 Final Project
//
// Compile:  g++ -O2 -std=c++17 -o /tmp/cloth_sim cloth_sim_sequential.cpp
// Run:      /tmp/cloth_sim
//
// Hyperparameters (edit the Params namespace below):
//   CLOTH_SIZE       — physical size of cloth (world units)
//   GRID_W/GRID_H    — particle resolution of the cloth
//   SPHERE_RADIUS    — radius of the collision object on the ground
//   SPHERE_ENABLED   — toggle sphere collision on/off
//   DROP_Y           — height from which the cloth is dropped
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <string>

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
// *** HYPERPARAMETERS — edit here ***
// ---------------------------------------------------------------------------
namespace Params {

    // --- Cloth size & resolution -------------------------------------------
    // CLOTH_SIZE controls the physical width/depth of the cloth in world units.
    // GRID_W/GRID_H control how many particles make up the cloth.
    constexpr float CLOTH_SIZE   = 3.0f;   // world-unit width & depth
    constexpr int   GRID_W       = 25;     // particles along X
    constexpr int   GRID_H       = 25;     // particles along Z

    // --- Drop height ----------------------------------------------------------
    // The cloth starts flat and horizontal at DROP_Y, then falls freely.
    // No particles are pinned — the whole cloth drops straight down.
    constexpr float DROP_Y       = 3.0f;

    // --- Integration ----------------------------------------------------------
    constexpr float DT           = 0.005f; // timestep (s) — lower = more stable
    constexpr int   NUM_STEPS    = 1200;   // total simulation steps
    constexpr int   SAVE_EVERY   = 10;     // write a frame every N steps

    // --- Spring stiffnesses ---------------------------------------------------
    constexpr float K_STRUCTURAL = 500.0f;
    constexpr float K_SHEAR      = 200.0f;
    constexpr float K_BEND       = 100.0f;

    // --- Damping & drag -------------------------------------------------------
    constexpr float DAMPING      = 0.998f;
    constexpr float AIR_DRAG     = 0.01f;

    // --- Forces ---------------------------------------------------------------
    const     Vec3  GRAVITY      = {0.0f, -9.81f, 0.0f};

    // --- Ground plane ---------------------------------------------------------
    constexpr float GROUND_Y           = -1.0f;
    constexpr float GROUND_RESTITUTION = 0.05f;
    constexpr float GROUND_FRICTION    = 0.80f;

    // --- Sphere collision object on the ground --------------------------------
    // Toggle SPHERE_ENABLED to turn the object on/off.
    // The sphere sits on the ground plane, centred under the cloth.
    constexpr bool  SPHERE_ENABLED = true;
    constexpr float SPHERE_RADIUS  = 0.6f;
    const     Vec3  SPHERE_CENTER  = {0.0f,
                                      GROUND_Y + SPHERE_RADIUS,
                                      0.0f};
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

inline int idx(int row, int col) { return row * Params::GRID_W + col; }

// ---------------------------------------------------------------------------
// Initialise particles — flat cloth at DROP_Y, no pins, free drop
// ---------------------------------------------------------------------------
void init_particles(std::vector<Particle>& particles) {
    particles.resize(Params::GRID_W * Params::GRID_H);

    const float dx = Params::CLOTH_SIZE / (Params::GRID_W - 1);
    const float dz = Params::CLOTH_SIZE / (Params::GRID_H - 1);

    for (int r = 0; r < Params::GRID_H; ++r) {
        for (int c = 0; c < Params::GRID_W; ++c) {
            Particle& p = particles[idx(r, c)];
            p.pos    = { c * dx - Params::CLOTH_SIZE * 0.5f,
                          Params::DROP_Y,
                          r * dz - Params::CLOTH_SIZE * 0.5f };
            p.vel    = {};
            p.force  = {};
            p.mass   = 0.1f;
            p.pinned = false;  // free drop — no pinned corners
        }
    }
}

// ---------------------------------------------------------------------------
// Build spring network (structural + shear + bend)
// ---------------------------------------------------------------------------
void build_springs(std::vector<Spring>& springs,
                   const std::vector<Particle>& particles) {
    auto add = [&](int a, int b, float k) {
        float rest = (particles[a].pos - particles[b].pos).length();
        springs.push_back({a, b, rest, k});
    };

    for (int r = 0; r < Params::GRID_H; ++r) {
        for (int c = 0; c < Params::GRID_W; ++c) {
            // Structural
            if (c+1 < Params::GRID_W) add(idx(r,c), idx(r,c+1), Params::K_STRUCTURAL);
            if (r+1 < Params::GRID_H) add(idx(r,c), idx(r+1,c), Params::K_STRUCTURAL);
            // Shear
            if (r+1 < Params::GRID_H && c+1 < Params::GRID_W) add(idx(r,c), idx(r+1,c+1), Params::K_SHEAR);
            if (r+1 < Params::GRID_H && c-1 >= 0)             add(idx(r,c), idx(r+1,c-1), Params::K_SHEAR);
            // Bend (skip-one)
            if (c+2 < Params::GRID_W) add(idx(r,c), idx(r,c+2), Params::K_BEND);
            if (r+2 < Params::GRID_H) add(idx(r,c), idx(r+2,c), Params::K_BEND);
        }
    }
}

// ---------------------------------------------------------------------------
// One simulation timestep
// ---------------------------------------------------------------------------
void step(std::vector<Particle>& particles,
          const std::vector<Spring>& springs) {

    // 1. Clear forces
    for (auto& p : particles) p.force = {};

    // 2. External forces: gravity + air drag
    for (auto& p : particles) {
        if (p.pinned) continue;
        p.force += Params::GRAVITY * p.mass;
        p.force += p.vel * (-Params::AIR_DRAG);
    }

    // 3. Spring forces (Hooke's law)
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

    // 4. Semi-implicit Euler integration
    for (auto& p : particles) {
        if (p.pinned) continue;
        p.vel += (p.force * (1.0f / p.mass)) * Params::DT;
        p.vel *= Params::DAMPING;
        p.pos += p.vel * Params::DT;
    }

    // 5. Ground plane collision
    for (auto& p : particles) {
        if (p.pos.y < Params::GROUND_Y) {
            p.pos.y  = Params::GROUND_Y;
            p.vel.y  = std::abs(p.vel.y) * Params::GROUND_RESTITUTION;
            p.vel.x *= Params::GROUND_FRICTION;
            p.vel.z *= Params::GROUND_FRICTION;
        }
    }

    // 6. Sphere collision
    // Push any particle inside the sphere back to its surface,
    // then cancel the inward component of its velocity.
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
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== Sequential Cloth Simulator ===\n"
              << "Cloth size:  " << Params::CLOTH_SIZE << " x " << Params::CLOTH_SIZE << "\n"
              << "Grid:        " << Params::GRID_W << " x " << Params::GRID_H << "\n"
              << "Drop height: " << Params::DROP_Y << "\n"
              << "Sphere:      " << (Params::SPHERE_ENABLED ? "ON" : "OFF")
              << "  radius=" << Params::SPHERE_RADIUS << "\n"
              << "Steps:       " << Params::NUM_STEPS
              << "  dt=" << Params::DT << "s\n\n";

    std::vector<Particle> particles;
    std::vector<Spring>   springs;
    init_particles(particles);
    build_springs(springs, particles);

    std::cout << "Particles: " << particles.size() << "\n"
              << "Springs:   " << springs.size()   << "\n\n";

    // Write scene metadata for the visualizer
    {
        std::ofstream meta("cloth_meta.csv");
        meta << "ground_y,sphere_enabled,sphere_cx,sphere_cy,sphere_cz,sphere_r\n";
        meta << Params::GROUND_Y << ","
             << (int)Params::SPHERE_ENABLED << ","
             << Params::SPHERE_CENTER.x << ","
             << Params::SPHERE_CENTER.y << ","
             << Params::SPHERE_CENTER.z << ","
             << Params::SPHERE_RADIUS   << "\n";
    }

    std::ofstream out("cloth_frames.csv");
    out << "step,row,col,x,y,z\n";

    auto dump = [&](int step_num) {
        for (int r = 0; r < Params::GRID_H; ++r)
            for (int c = 0; c < Params::GRID_W; ++c) {
                const Vec3& pos = particles[idx(r,c)].pos;
                out << step_num << ',' << r << ',' << c << ','
                    << pos.x << ',' << pos.y << ',' << pos.z << '\n';
            }
    };

    dump(0);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int s = 1; s <= Params::NUM_STEPS; ++s) {
        step(particles, springs);
        if (s % Params::SAVE_EVERY == 0) dump(s);
        if (s % 100 == 0) {
            const Vec3& centre = particles[idx(Params::GRID_H/2, Params::GRID_W/2)].pos;
            std::printf("  step %4d — centre y = %+.4f\n", s, centre.y);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nDone in " << elapsed << "s  ("
              << (Params::NUM_STEPS / elapsed) << " steps/s)\n"
              << "Frames   -> cloth_frames.csv\n"
              << "Metadata -> cloth_meta.csv\n";
    return 0;
}
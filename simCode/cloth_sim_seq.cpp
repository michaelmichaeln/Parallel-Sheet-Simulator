// =============================================================================
// cloth_sim_seq.cpp
// Sequential Mass-Spring Cloth Simulator with Config-Driven Experiments
// Ankita Kundu & Michael Nguyen — 15-418/618 Final Project
//
// Compile:  g++ -O2 -std=c++17 -I include -o cloth_sim cloth_sim_seq.cpp
// Run:      ./cloth_sim ../configs/experiments/drop_on_sphere.json
//           ./cloth_sim   (uses default parameters if no config provided)
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <string>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <algorithm>
#include <limits>
#include "include/json.hpp"

using json = nlohmann::json;

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
// Runtime Configuration (loaded from JSON or defaults)
// ---------------------------------------------------------------------------
struct Config {
    std::string scene_name = "default";
    
    // Cloth size & resolution
    float cloth_size = 3.0f;
    int   grid_w     = 25;
    int   grid_h     = 25;
    
    // Drop height
    float drop_y = 3.0f;
    
    // Integration
    float dt        = 0.005f;
    int   num_steps = 1200;
    int   save_every = 10;
    
    // Spring stiffnesses
    float k_structural = 500.0f;
    float k_shear      = 200.0f;
    float k_bend       = 100.0f;
    
    // Damping & drag
    float damping  = 0.998f;
    float air_drag = 0.01f;
    
    // Gravity
    Vec3 gravity = {0.0f, -9.81f, 0.0f};
    
    // Ground plane
    float ground_y           = -1.0f;
    float ground_restitution = 0.05f;
    float ground_friction    = 0.80f;
    
    // Sphere collision
    bool  sphere_enabled = true;
    float sphere_radius  = 0.6f;
    Vec3  sphere_center  = {0.0f, -0.4f, 0.0f};  // Updated after ground_y is set
    
    // Output paths
    std::string frames_path  = "results/outputs/cloth_frames.csv";
    std::string meta_path    = "results/outputs/cloth_meta.csv";
    std::string metrics_path = "results/metrics/";
    
    void update_derived() {
        sphere_center = {0.0f, ground_y + sphere_radius, 0.0f};
    }
};

Config load_config(const std::string& path) {
    Config cfg;
    
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file '" << path << "', using defaults.\n";
        cfg.update_derived();
        return cfg;
    }
    
    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << "\nUsing defaults.\n";
        cfg.update_derived();
        return cfg;
    }
    
    if (j.contains("scene")) cfg.scene_name = j["scene"].get<std::string>();
    
    if (j.contains("cloth")) {
        auto& c = j["cloth"];
        if (c.contains("size"))   cfg.cloth_size = c["size"].get<float>();
        if (c.contains("grid_w")) cfg.grid_w = c["grid_w"].get<int>();
        if (c.contains("grid_h")) cfg.grid_h = c["grid_h"].get<int>();
        if (c.contains("drop_y")) cfg.drop_y = c["drop_y"].get<float>();
    }
    
    if (j.contains("physics")) {
        auto& p = j["physics"];
        if (p.contains("dt"))           cfg.dt = p["dt"].get<float>();
        if (p.contains("steps"))        cfg.num_steps = p["steps"].get<int>();
        if (p.contains("k_structural")) cfg.k_structural = p["k_structural"].get<float>();
        if (p.contains("k_shear"))      cfg.k_shear = p["k_shear"].get<float>();
        if (p.contains("k_bend"))       cfg.k_bend = p["k_bend"].get<float>();
        if (p.contains("damping"))      cfg.damping = p["damping"].get<float>();
        if (p.contains("air_drag"))     cfg.air_drag = p["air_drag"].get<float>();
    }
    
    if (j.contains("collision")) {
        auto& col = j["collision"];
        if (col.contains("ground_y"))           cfg.ground_y = col["ground_y"].get<float>();
        if (col.contains("ground_restitution")) cfg.ground_restitution = col["ground_restitution"].get<float>();
        if (col.contains("ground_friction"))    cfg.ground_friction = col["ground_friction"].get<float>();
        if (col.contains("sphere_enabled"))     cfg.sphere_enabled = col["sphere_enabled"].get<bool>();
        if (col.contains("sphere_radius"))      cfg.sphere_radius = col["sphere_radius"].get<float>();
    }
    
    if (j.contains("output")) {
        auto& o = j["output"];
        if (o.contains("save_every"))   cfg.save_every = o["save_every"].get<int>();
        if (o.contains("frames_path"))  cfg.frames_path = o["frames_path"].get<std::string>();
        if (o.contains("meta_path"))    cfg.meta_path = o["meta_path"].get<std::string>();
        if (o.contains("metrics_path")) cfg.metrics_path = o["metrics_path"].get<std::string>();
    }
    
    cfg.update_derived();
    return cfg;
}

// Global config instance
Config cfg;

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

inline int idx(int row, int col) { return row * cfg.grid_w + col; }

// ---------------------------------------------------------------------------
// Correctness Metrics
// ---------------------------------------------------------------------------
struct Metrics {
    float max_stretch_error = 0.0f;
    int   ground_penetrations = 0;
    float max_ground_penetration = 0.0f;
    int   sphere_penetrations = 0;
    float max_sphere_penetration = 0.0f;
    float total_energy = 0.0f;
    float kinetic_energy = 0.0f;
    float potential_energy = 0.0f;
    float spring_energy = 0.0f;
    bool  has_nan_inf = false;
    double step_time_ms = 0.0;
};

bool check_finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

Metrics compute_metrics(const std::vector<Particle>& particles,
                        const std::vector<Spring>& springs) {
    Metrics m;
    
    // Check for NaN/Inf and compute energies
    for (const auto& p : particles) {
        if (!check_finite(p.pos) || !check_finite(p.vel)) {
            m.has_nan_inf = true;
        }
        
        // Kinetic energy: 0.5 * m * v^2
        float v2 = p.vel.dot(p.vel);
        m.kinetic_energy += 0.5f * p.mass * v2;
        
        // Gravitational potential energy: m * g * h (relative to ground)
        float h = p.pos.y - cfg.ground_y;
        m.potential_energy += p.mass * std::abs(cfg.gravity.y) * h;
        
        // Ground penetration check
        if (p.pos.y < cfg.ground_y) {
            m.ground_penetrations++;
            float depth = cfg.ground_y - p.pos.y;
            m.max_ground_penetration = std::max(m.max_ground_penetration, depth);
        }
        
        // Sphere penetration check
        if (cfg.sphere_enabled) {
            Vec3 to_p = p.pos - cfg.sphere_center;
            float dist = to_p.length();
            if (dist < cfg.sphere_radius) {
                m.sphere_penetrations++;
                float depth = cfg.sphere_radius - dist;
                m.max_sphere_penetration = std::max(m.max_sphere_penetration, depth);
            }
        }
    }
    
    // Spring stretch error and elastic potential energy
    for (const auto& s : springs) {
        Vec3 diff = particles[s.b].pos - particles[s.a].pos;
        float len = diff.length();
        float stretch = std::abs(len - s.rest_length) / s.rest_length;
        m.max_stretch_error = std::max(m.max_stretch_error, stretch);
        
        // Spring potential energy: 0.5 * k * (x - x0)^2
        float dx = len - s.rest_length;
        m.spring_energy += 0.5f * s.stiffness * dx * dx;
    }
    
    m.total_energy = m.kinetic_energy + m.potential_energy + m.spring_energy;
    
    return m;
}

std::string generate_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

// ---------------------------------------------------------------------------
// Initialise particles — flat cloth at drop_y, no pins, free drop
// ---------------------------------------------------------------------------
void init_particles(std::vector<Particle>& particles) {
    particles.resize(cfg.grid_w * cfg.grid_h);

    const float dx = cfg.cloth_size / (cfg.grid_w - 1);
    const float dz = cfg.cloth_size / (cfg.grid_h - 1);

    for (int r = 0; r < cfg.grid_h; ++r) {
        for (int c = 0; c < cfg.grid_w; ++c) {
            Particle& p = particles[idx(r, c)];
            p.pos    = { c * dx - cfg.cloth_size * 0.5f,
                         cfg.drop_y,
                         r * dz - cfg.cloth_size * 0.5f };
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
                   const std::vector<Particle>& particles) {
    auto add = [&](int a, int b, float k) {
        float rest = (particles[a].pos - particles[b].pos).length();
        springs.push_back({a, b, rest, k});
    };

    for (int r = 0; r < cfg.grid_h; ++r) {
        for (int c = 0; c < cfg.grid_w; ++c) {
            // Structural
            if (c+1 < cfg.grid_w) add(idx(r,c), idx(r,c+1), cfg.k_structural);
            if (r+1 < cfg.grid_h) add(idx(r,c), idx(r+1,c), cfg.k_structural);
            // Shear
            if (r+1 < cfg.grid_h && c+1 < cfg.grid_w) add(idx(r,c), idx(r+1,c+1), cfg.k_shear);
            if (r+1 < cfg.grid_h && c-1 >= 0)         add(idx(r,c), idx(r+1,c-1), cfg.k_shear);
            // Bend (skip-one)
            if (c+2 < cfg.grid_w) add(idx(r,c), idx(r,c+2), cfg.k_bend);
            if (r+2 < cfg.grid_h) add(idx(r,c), idx(r+2,c), cfg.k_bend);
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
        p.force += cfg.gravity * p.mass;
        p.force += p.vel * (-cfg.air_drag);
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
        p.vel += (p.force * (1.0f / p.mass)) * cfg.dt;
        p.vel *= cfg.damping;
        p.pos += p.vel * cfg.dt;
    }

    // 5. Ground plane collision
    for (auto& p : particles) {
        if (p.pos.y < cfg.ground_y) {
            p.pos.y  = cfg.ground_y;
            p.vel.y  = std::abs(p.vel.y) * cfg.ground_restitution;
            p.vel.x *= cfg.ground_friction;
            p.vel.z *= cfg.ground_friction;
        }
    }

    // 6. Sphere collision
    if (cfg.sphere_enabled) {
        for (auto& p : particles) {
            Vec3  to_p = p.pos - cfg.sphere_center;
            float dist = to_p.length();
            if (dist < cfg.sphere_radius) {
                Vec3  n   = to_p.normalized();
                p.pos     = cfg.sphere_center + n * cfg.sphere_radius;
                float v_n = p.vel.dot(n);
                if (v_n < 0.0f) p.vel += n * (-v_n);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Load configuration
    if (argc > 1) {
        cfg = load_config(argv[1]);
        std::cout << "Loaded config: " << argv[1] << "\n";
    } else {
        cfg.update_derived();
        std::cout << "Using default configuration.\n";
    }

    std::cout << "\n=== Sequential Cloth Simulator ===\n"
              << "Scene:       " << cfg.scene_name << "\n"
              << "Cloth size:  " << cfg.cloth_size << " x " << cfg.cloth_size << "\n"
              << "Grid:        " << cfg.grid_w << " x " << cfg.grid_h << "\n"
              << "Drop height: " << cfg.drop_y << "\n"
              << "Sphere:      " << (cfg.sphere_enabled ? "ON" : "OFF")
              << "  radius=" << cfg.sphere_radius << "\n"
              << "Steps:       " << cfg.num_steps
              << "  dt=" << cfg.dt << "s\n\n";

    std::vector<Particle> particles;
    std::vector<Spring>   springs;
    init_particles(particles);
    build_springs(springs, particles);

    std::cout << "Particles: " << particles.size() << "\n"
              << "Springs:   " << springs.size()   << "\n\n";

    // Write scene metadata for the visualizer
    {
        std::ofstream meta(cfg.meta_path);
        meta << "ground_y,sphere_enabled,sphere_cx,sphere_cy,sphere_cz,sphere_r\n";
        meta << cfg.ground_y << ","
             << (int)cfg.sphere_enabled << ","
             << cfg.sphere_center.x << ","
             << cfg.sphere_center.y << ","
             << cfg.sphere_center.z << ","
             << cfg.sphere_radius   << "\n";
    }

    // Open frames output
    std::ofstream frames_out(cfg.frames_path);
    frames_out << "step,row,col,x,y,z\n";

    auto dump_frame = [&](int step_num) {
        for (int r = 0; r < cfg.grid_h; ++r)
            for (int c = 0; c < cfg.grid_w; ++c) {
                const Vec3& pos = particles[idx(r,c)].pos;
                frames_out << step_num << ',' << r << ',' << c << ','
                           << pos.x << ',' << pos.y << ',' << pos.z << '\n';
            }
    };

    // Open metrics output
    std::string timestamp = generate_timestamp();
    std::string metrics_file = cfg.metrics_path + cfg.scene_name + "_" + timestamp + ".csv";
    std::ofstream metrics_out(metrics_file);
    metrics_out << "step,max_stretch_error,ground_penetrations,max_ground_pen,"
                << "sphere_penetrations,max_sphere_pen,kinetic_energy,potential_energy,"
                << "spring_energy,total_energy,has_nan_inf,step_time_ms,cumulative_time_ms\n";

    // Initial state
    dump_frame(0);
    Metrics m0 = compute_metrics(particles, springs);
    float initial_energy = m0.total_energy;
    metrics_out << 0 << "," << m0.max_stretch_error << "," << m0.ground_penetrations << ","
                << m0.max_ground_penetration << "," << m0.sphere_penetrations << ","
                << m0.max_sphere_penetration << "," << m0.kinetic_energy << ","
                << m0.potential_energy << "," << m0.spring_energy << "," << m0.total_energy << ","
                << (m0.has_nan_inf ? 1 : 0) << ",0,0\n";

    // Simulation loop with timing
    auto t_total_start = std::chrono::high_resolution_clock::now();
    double cumulative_ms = 0.0;
    
    float max_stretch_overall = 0.0f;
    int   max_ground_pen_count = 0;
    int   max_sphere_pen_count = 0;
    float max_energy_ratio = 1.0f;
    bool  any_nan_inf = m0.has_nan_inf;

    for (int s = 1; s <= cfg.num_steps; ++s) {
        auto t_step_start = std::chrono::high_resolution_clock::now();
        
        step(particles, springs);
        
        auto t_step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
        cumulative_ms += step_ms;

        // Save frame and metrics at intervals
        if (s % cfg.save_every == 0) {
            dump_frame(s);
            
            Metrics m = compute_metrics(particles, springs);
            m.step_time_ms = step_ms;
            
            metrics_out << s << "," << m.max_stretch_error << "," << m.ground_penetrations << ","
                        << m.max_ground_penetration << "," << m.sphere_penetrations << ","
                        << m.max_sphere_penetration << "," << m.kinetic_energy << ","
                        << m.potential_energy << "," << m.spring_energy << "," << m.total_energy << ","
                        << (m.has_nan_inf ? 1 : 0) << "," << step_ms << "," << cumulative_ms << "\n";
            
            // Track worst-case metrics
            max_stretch_overall = std::max(max_stretch_overall, m.max_stretch_error);
            max_ground_pen_count = std::max(max_ground_pen_count, m.ground_penetrations);
            max_sphere_pen_count = std::max(max_sphere_pen_count, m.sphere_penetrations);
            if (initial_energy > 1e-6f) {
                max_energy_ratio = std::max(max_energy_ratio, m.total_energy / initial_energy);
            }
            if (m.has_nan_inf) any_nan_inf = true;
        }

        // Progress output
        if (s % 100 == 0) {
            const Vec3& centre = particles[idx(cfg.grid_h/2, cfg.grid_w/2)].pos;
            std::printf("  step %4d — centre y = %+.4f  (%.2f ms/step)\n", 
                        s, centre.y, step_ms);
        }
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_total_end - t_total_start).count();

    // Close files
    frames_out.close();
    metrics_out.close();

    // Write run summary
    std::string summary_file = cfg.metrics_path + cfg.scene_name + "_" + timestamp + "_summary.json";
    {
        std::ofstream summary(summary_file);
        summary << "{\n"
                << "  \"scene\": \"" << cfg.scene_name << "\",\n"
                << "  \"timestamp\": \"" << timestamp << "\",\n"
                << "  \"config\": {\n"
                << "    \"grid_w\": " << cfg.grid_w << ",\n"
                << "    \"grid_h\": " << cfg.grid_h << ",\n"
                << "    \"cloth_size\": " << cfg.cloth_size << ",\n"
                << "    \"drop_y\": " << cfg.drop_y << ",\n"
                << "    \"dt\": " << cfg.dt << ",\n"
                << "    \"num_steps\": " << cfg.num_steps << ",\n"
                << "    \"sphere_enabled\": " << (cfg.sphere_enabled ? "true" : "false") << ",\n"
                << "    \"sphere_radius\": " << cfg.sphere_radius << "\n"
                << "  },\n"
                << "  \"results\": {\n"
                << "    \"particles\": " << particles.size() << ",\n"
                << "    \"springs\": " << springs.size() << ",\n"
                << "    \"total_time_s\": " << total_elapsed << ",\n"
                << "    \"steps_per_second\": " << (cfg.num_steps / total_elapsed) << ",\n"
                << "    \"avg_step_ms\": " << (cumulative_ms / cfg.num_steps) << "\n"
                << "  },\n"
                << "  \"correctness\": {\n"
                << "    \"max_stretch_error\": " << max_stretch_overall << ",\n"
                << "    \"max_ground_penetrations\": " << max_ground_pen_count << ",\n"
                << "    \"max_sphere_penetrations\": " << max_sphere_pen_count << ",\n"
                << "    \"max_energy_ratio\": " << max_energy_ratio << ",\n"
                << "    \"has_nan_inf\": " << (any_nan_inf ? "true" : "false") << "\n"
                << "  }\n"
                << "}\n";
    }

    std::cout << "\n=== Simulation Complete ===\n"
              << "Time:      " << total_elapsed << "s  ("
              << (cfg.num_steps / total_elapsed) << " steps/s)\n"
              << "Frames:    " << cfg.frames_path << "\n"
              << "Metadata:  " << cfg.meta_path << "\n"
              << "Metrics:   " << metrics_file << "\n"
              << "Summary:   " << summary_file << "\n\n"
              << "=== Correctness Summary ===\n"
              << "Max stretch error:       " << max_stretch_overall << "\n"
              << "Max ground penetrations: " << max_ground_pen_count << "\n"
              << "Max sphere penetrations: " << max_sphere_pen_count << "\n"
              << "Max energy ratio:        " << max_energy_ratio << "\n"
              << "NaN/Inf detected:        " << (any_nan_inf ? "YES" : "NO") << "\n";

    return any_nan_inf ? 1 : 0;
}
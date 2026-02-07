#version 450

// === SEPARABLE CONVOLUTION - HORIZONTAL PASS ===
// Reduces O(R²) to O(R) by computing row-wise convolution
// Output: Intermediate buffer with horizontally-convolved data

#define WORKGROUP_SIZE 256
#define MAX_RADIUS 20

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float u_theta_A;
    float u_alpha_n;
    float u_temperature;
    float u_signal_advect;
    float u_beta;
    float u_signal_diff;
    float u_signal_decay;
    float u_flow_speed;
    float u_init_clusters;
    float u_init_density;
    float u_colonize_thr;
    
    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_inertia;
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;
    
    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;
    
    float u_signal_force_strength;
    float u_signal_emission_strength;
    float u_pad1;
    float u_pad2;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_signal;
layout(set = 0, binding = 3, rgba32f) uniform image2D img_intermediate;

// Shared memory for row + halo
shared vec4 row_cache[WORKGROUP_SIZE + 2 * MAX_RADIUS];

// 1D Gaussian kernel weight
float kernel_1d(float x, float R) {
    float norm_x = abs(x) / R;
    if (norm_x > 1.0) return 0.0;
    
    // Simple smooth bump (approximation of radial kernel projected to 1D)
    // Using cosine-based smooth falloff
    float t = norm_x * 3.14159265;
    return (1.0 + cos(t)) * 0.5;
}

void main() {
    ivec2 res_i = ivec2(p.u_res);
    int row = int(gl_WorkGroupID.y);
    int local_x = int(gl_LocalInvocationID.x);
    int global_x = int(gl_WorkGroupID.x) * WORKGROUP_SIZE + local_x;
    
    if (row >= res_i.y) return;
    
    int R_int = int(ceil(min(p.u_R, float(MAX_RADIUS))));
    
    // === 1. COLLABORATIVE LOADING ===
    // Each thread loads its position + helps with halo
    int cache_size = WORKGROUP_SIZE + 2 * MAX_RADIUS;
    int base_x = int(gl_WorkGroupID.x) * WORKGROUP_SIZE - MAX_RADIUS;
    
    for (int i = local_x; i < cache_size; i += WORKGROUP_SIZE) {
        int src_x = (base_x + i + res_i.x) % res_i.x;
        ivec2 src_pos = ivec2(src_x, row);
        
        float mass = texelFetch(tex_state, src_pos, 0).r;
        vec3 sig = texelFetch(tex_signal, src_pos, 0).rgb;
        row_cache[i] = vec4(mass, sig);
    }
    
    barrier();
    
    // === 2. HORIZONTAL CONVOLUTION ===
    if (global_x >= res_i.x) return;
    
    int my_cache_pos = local_x + MAX_RADIUS;
    
    float R_actual = p.u_R;
    R_actual = min(R_actual, float(MAX_RADIUS));
    
    float sum_mass = 0.0;
    vec3 sum_signal = vec3(0.0);
    float total_weight = 0.0;
    
    for (int dx = -R_int; dx <= R_int; dx++) {
        int cache_idx = my_cache_pos + dx;
        vec4 data = row_cache[cache_idx];
        
        float w = kernel_1d(float(dx), R_actual);
        
        sum_mass += data.r * w;
        sum_signal += data.gba * w;
        total_weight += w;
    }
    
    // Normalize
    if (total_weight > 0.0) {
        sum_mass /= total_weight;
        sum_signal /= total_weight;
    }
    
    // Store intermediate result
    imageStore(img_intermediate, ivec2(global_x, row), vec4(sum_mass, sum_signal));
}

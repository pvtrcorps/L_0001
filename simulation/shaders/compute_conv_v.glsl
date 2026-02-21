#version 450

// === SEPARABLE CONVOLUTION - VERTICAL PASS ===
// Second pass: reads horizontally-convolved intermediate buffer
// Applies vertical convolution + growth function → outputs potential

#define WORKGROUP_SIZE 256
#define MAX_RADIUS 20

layout(local_size_x = 1, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    // 0. Globals
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
    float u_fluid_momentum;
    
    // 1. Gene Ranges (16 Genes * 2) = 32 floats
    // Block A: Physiology
    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    // Block B: Morphology
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_inertia;
    // Block C: Social / Motor
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    // Block D: Senses
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;
    
    // 2. Wind / Atmosphere
    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;
    
    // 3. Signal + Morph Extras
    float u_signal_force_strength;
    float u_signal_emission_strength;
    float u_interaction_beta;
    float u_morph_anisotropy_gain;
    
    // 4. Morph Controls + Cleanup
    float u_colonize_thr;
    float u_morph_polarity_gain;
    float u_morph_plasticity_gain;
    float u_morph_self_propulsion_gain;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_intermediate;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_genome_ext;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_potential;

// Shared memory for column + halo
shared vec4 col_cache[WORKGROUP_SIZE + 2 * MAX_RADIUS];

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u;
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

vec3 HueToRGB(float hue) {
    float h = hue * 6.0;
    float c = 1.0;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    vec3 rgb;
    if (h < 1.0)      rgb = vec3(c, x, 0.0);
    else if (h < 2.0) rgb = vec3(x, c, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, c, x);
    else if (h < 4.0) rgb = vec3(0.0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0.0, c);
    else              rgb = vec3(c, 0.0, x);
    return rgb;
}

float kernel_1d(float x, float R) {
    float norm_x = abs(x) / R;
    if (norm_x > 1.0) return 0.0;
    float t = norm_x * 3.14159265;
    return (1.0 + cos(t)) * 0.5;
}

void main() {
    ivec2 res_i = ivec2(p.u_res);
    int col = int(gl_WorkGroupID.x);
    int local_y = int(gl_LocalInvocationID.y);
    int global_y = int(gl_WorkGroupID.y) * WORKGROUP_SIZE + local_y;
    
    if (col >= res_i.x) return;
    
    int R_int = int(ceil(min(p.u_R, float(MAX_RADIUS))));
    
    // === 1. COLLABORATIVE LOADING ===
    int cache_size = WORKGROUP_SIZE + 2 * MAX_RADIUS;
    int base_y = int(gl_WorkGroupID.y) * WORKGROUP_SIZE - MAX_RADIUS;
    
    for (int i = local_y; i < cache_size; i += WORKGROUP_SIZE) {
        int src_y = (base_y + i + res_i.y) % res_i.y;
        ivec2 src_pos = ivec2(col, src_y);
        
        vec4 data = texelFetch(tex_intermediate, src_pos, 0);
        col_cache[i] = data;
    }
    
    barrier();
    
    // === 2. VERTICAL CONVOLUTION ===
    if (global_y >= res_i.y) return;
    
    ivec2 gid = ivec2(col, global_y);
    vec2 uv = (vec2(gid) + 0.5) / p.u_res;
    
    // Read genome for this pixel
    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);
    
    // Void check
    if (dot(g1, g1) < 0.0001) {
        imageStore(img_potential, gid, vec4(0.0));
        return;
    }
    
    // Unpack genes
    vec2 mu_sigma = unpack2(g1.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    
    vec2 rad_visc = unpack2(g1.g);
    float g_radius = rad_visc.x;
    
    vec2 hue_hue = unpack2(g2.a);
    float g_detection_hue = hue_hue.y;
    vec3 myDetector = HueToRGB(g_detection_hue);
    
    float R_actual = max(p.u_R * g_radius, 1.0);
    R_actual = min(R_actual, float(MAX_RADIUS));
    
    int my_cache_pos = local_y + MAX_RADIUS;
    
    float sum_mass = 0.0;
    float sum_signal = 0.0;
    float total_weight = 0.0;
    
    for (int dy = -R_int; dy <= R_int; dy++) {
        int cache_idx = my_cache_pos + dy;
        vec4 data = col_cache[cache_idx];
        
        float w = kernel_1d(float(dy), R_actual);
        
        if (w > 0.0001) {
            float neighborMass = data.r;
            vec3 neighborSignal = data.gba;
            
            // Signal matching
            float signalMatch = dot(neighborSignal, myDetector);
            float totalIntensity = dot(neighborSignal, vec3(1.0));
            float signedScore = 2.0 * signalMatch - totalIntensity;
            
            sum_mass += neighborMass * w;
            sum_signal += signedScore * w;
            total_weight += w;
        }
    }
    
    float U_raw = (total_weight > 0.0) ? sum_mass / total_weight : 0.0;
    float U_signal_smooth = (total_weight > 0.0) ? sum_signal / total_weight : 0.0;
    
    // === GROWTH G(U) ===
    float mu = g_mu;
    float sigma = 0.001 + g_sigma * 0.2;
    
    float diff = (U_raw - mu);
    float exp_term = exp(-0.5 * (diff * diff) / max(sigma * sigma, 0.0001));
    float U_growth = 2.0 * exp_term - 1.0;
    
    imageStore(img_potential, gid, vec4(U_growth, 0.0, 0.0, U_signal_smooth));
}

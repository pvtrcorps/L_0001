#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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
    
    // Wind / Atmosphere
    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;
    
    // Signal Extras
    float u_signal_force_strength;
    float u_signal_emission_strength;
    float u_pad1;
    float u_pad2;
    float u_colonize_thr;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_signal_src;
layout(set = 0, binding = 2, rgba32f) uniform image2D img_signal_dst;
layout(set = 0, binding = 3) uniform sampler2D tex_state;  // For velocity field

// Simple pseudo-random hash
vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// 2D Gradient Noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dot(hash22(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
                   dot(hash22(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
               mix(dot(hash22(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
                   dot(hash22(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
}

// Curl Noise for divergence-free flow
vec2 curl_noise(vec2 p, float t) {
    float eps = 0.1;
    // Time evolution
    vec2 p_t = p + vec2(t * 0.1, -t * 0.05); 
    
    // Finite difference curl
    float n1 = noise(p_t + vec2(0, eps));
    float n2 = noise(p_t - vec2(0, eps));
    float n3 = noise(p_t + vec2(eps, 0));
    float n4 = noise(p_t - vec2(eps, 0));
    
    float x = (n1 - n2) / (2.0 * eps);
    float y = (n3 - n4) / (2.0 * eps);
    
    return vec2(x, -y);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    ivec2 res_i = ivec2(p.u_res);
    
    // Read current signal
    vec3 center = texture(tex_signal_src, uv).rgb;
    
    // === 1. DIFFUSION (Laplacian) ===
    vec3 laplacian = vec3(0.0);
    
    // Neighbors (cardinal + ordinal with weights)
    laplacian += texture(tex_signal_src, uv + vec2(px.x, 0.0)).rgb;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, 0.0)).rgb;
    laplacian += texture(tex_signal_src, uv + vec2(0.0, px.y)).rgb;
    laplacian += texture(tex_signal_src, uv + vec2(0.0, -px.y)).rgb;
    
    laplacian += texture(tex_signal_src, uv + vec2(px.x, px.y)).rgb * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, px.y)).rgb * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(px.x, -px.y)).rgb * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, -px.y)).rgb * 0.5;
    
    laplacian = (laplacian / 6.0) - center;
    
    vec3 diffused = center + (laplacian * p.u_signal_diff * p.u_dt);
    
    // === 2. ADVECTION (Partial, based on mass velocity + WIND) ===
    // Read local velocity from state texture (stored in .gb channels)
    vec4 state = texture(tex_state, uv);
    vec2 velocity = state.gb;
    
    // Wind Effect
    if (p.u_wind_strength > 0.0) {
        // Use time and position for noise
        // Scale UV for noise frequency
        vec2 noise_uv = uv * p.u_wind_scale;
        
        // Generate Curl direction
        vec2 wind_dir = curl_noise(noise_uv, p.u_time * p.u_wind_speed);
        
        // Apply strength
        velocity += wind_dir * p.u_wind_strength;
    }
    
    // Scale advection by the global weight parameter
    float advect_weight = clamp(p.u_signal_advect, 0.0, 1.0);
    
    vec3 advected = diffused;
    if (advect_weight > 0.001) { // Removed velocity check to allow wind only
        // Semi-Lagrangian advection: sample from upstream position
        vec2 upstream_uv = uv - velocity * advect_weight * p.u_dt * px;
        vec3 upstream_signal = texture(tex_signal_src, upstream_uv).rgb;
        
        // Blend between diffused result and advected result
        advected = mix(diffused, upstream_signal, advect_weight * 0.5);
    }
    
    // === 3. DECAY (Exponential) ===
    // S_new = S_old * (1.0 - k * dt)
    // Scaled so "0.1" in UI is meaningful
    float decay_factor = clamp(p.u_signal_decay * p.u_dt * 2.0, 0.0, 1.0);
    vec3 next_signal = advected * (1.0 - decay_factor);
    
    // Clamp to prevent negative values
    next_signal = max(vec3(0.0), next_signal);
    
    imageStore(img_signal_dst, uv_i, vec4(next_signal, 0.0));
}


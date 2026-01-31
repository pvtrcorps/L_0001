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
    float u_colonize_thr;
    
    // 1. Gene Ranges (16 Genes * 2) = 32 floats
    // Block A: Physiology
    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    // Block B: Morphology
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_ring_width;
    // Block C: Social / Motor
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    // Block D: Senses
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_signal_src;
layout(set = 0, binding = 2, rgba32f) uniform image2D img_signal_dst;
layout(set = 0, binding = 3) uniform sampler2D tex_state;  // For velocity field

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
    
    // === 2. ADVECTION (Partial, based on mass velocity) ===
    // Read local velocity from state texture (stored in .gb channels)
    vec4 state = texture(tex_state, uv);
    vec2 velocity = state.gb;
    
    // Scale advection by the global weight parameter
    float advect_weight = clamp(p.u_signal_advect, 0.0, 1.0);
    
    vec3 advected = diffused;
    if (advect_weight > 0.001 && length(velocity) > 0.001) {
        // Semi-Lagrangian advection: sample from upstream position
        vec2 upstream_uv = uv - velocity * advect_weight * p.u_dt * px;
        vec3 upstream_signal = texture(tex_signal_src, upstream_uv).rgb;
        
        // Blend between diffused result and advected result
        advected = mix(diffused, upstream_signal, advect_weight * 0.5);
    }
    
    // === 3. DECAY (Quadratic) ===
    vec3 decay_amount = (advected * advected) * p.u_signal_decay * p.u_dt * 5.0;
    vec3 next_signal = advected - decay_amount;
    
    // Clamp to prevent negative values
    next_signal = max(vec3(0.0), next_signal);
    
    imageStore(img_signal_dst, uv_i, vec4(next_signal, 0.0));
}


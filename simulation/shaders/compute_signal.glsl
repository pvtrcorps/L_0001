#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float u_theta_A; 
    float u_alpha_n; 
    float u_temperature; // Temperature (s)
    float _pad4;
    float _pad5;
    float u_init_clusters;
    float u_init_density;
    float u_colonize_thr;
    vec2 u_range_mu;
    vec2 u_range_sigma;
    vec2 u_range_radius;
    vec2 u_range_flow;
    vec2 u_range_affinity;
    vec2 u_range_lambda;
    float u_signal_diff;
    float u_signal_decay;
    vec2 u_range_secretion;
    vec2 u_range_perception;
    float _pad_end;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_signal_src;
layout(set = 0, binding = 2, rgba32f) uniform image2D img_signal_dst;

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    // 3x3 Laplacian Diffusion (RGB Vector)
    vec3 center = texture(tex_signal_src, uv).rgb;
    vec3 laplacian = vec3(0.0);
    
    // Neighbors (cardinal + ordinal)
    laplacian += texture(tex_signal_src, uv + vec2(px.x, 0.0)).rgb;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, 0.0)).rgb;
    laplacian += texture(tex_signal_src, uv + vec2(0.0, px.y)).rgb;
    laplacian += texture(tex_signal_src, uv + vec2(0.0, -px.y)).rgb;
    
    laplacian += texture(tex_signal_src, uv + vec2(px.x, px.y)).rgb * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, px.y)).rgb * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(px.x, -px.y)).rgb * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, -px.y)).rgb * 0.5;
    
    laplacian = (laplacian / 6.0) - center;
    
    vec3 next_signal = center + (laplacian * p.u_signal_diff * p.u_dt);
    
    // Quadratic Decay (Second-Order Kinetics)
    // dS/dt = -k * S^2
    // High concentrations decay much faster than low concentrations.
    // We add a specific 'quadratic_factor' or just reuse u_signal_decay.
    // To keep it controllable with existing slider [0..1], we scale it.
    
    vec3 decay_amount = (next_signal * next_signal) * p.u_signal_decay * p.u_dt * 5.0;
    next_signal -= decay_amount;
    
    // Clamp to prevent negative noise
    next_signal = max(vec3(0.0), next_signal);
    
    imageStore(img_signal_dst, uv_i, vec4(next_signal, 0.0));
}

#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float _pad1;
    float _pad2;
    float _pad3;
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
    
    // 3x3 Laplacian Diffusion
    float center = texture(tex_signal_src, uv).r;
    float laplacian = 0.0;
    
    // Neighbors (cardinal + ordinal)
    laplacian += texture(tex_signal_src, uv + vec2(px.x, 0.0)).r;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, 0.0)).r;
    laplacian += texture(tex_signal_src, uv + vec2(0.0, px.y)).r;
    laplacian += texture(tex_signal_src, uv + vec2(0.0, -px.y)).r;
    
    laplacian += texture(tex_signal_src, uv + vec2(px.x, px.y)).r * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, px.y)).r * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(px.x, -px.y)).r * 0.5;
    laplacian += texture(tex_signal_src, uv + vec2(-px.x, -px.y)).r * 0.5;
    
    laplacian = (laplacian / 6.0) - center;
    
    float next_signal = center + (laplacian * p.u_signal_diff * p.u_dt);
    
    // Decay
    next_signal *= (1.0 - p.u_signal_decay * p.u_dt);
    
    // Clamp to prevent negative noise
    next_signal = max(0.0, next_signal);
    
    imageStore(img_signal_dst, uv_i, vec4(next_signal, 0.0, 0.0, 0.0));
}

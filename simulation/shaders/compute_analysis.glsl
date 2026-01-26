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
    float u_temperature; 
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

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;

layout(set = 0, binding = 3, std430) buffer Analysis {
    float data[]; // 4096 * 10 floats
} a;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u; // Clear normalization bit
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

void main() {
    uint idx_x = gl_GlobalInvocationID.x;
    uint idx_y = gl_GlobalInvocationID.y;
    if (idx_x >= 64 || idx_y >= 64) return;
    
    // Sample the 1024x1024 grid at 64x64 intervals
    vec2 uv = (vec2(idx_x, idx_y) + 0.5) / 64.0;
    
    vec4 state = texture(tex_state, uv);
    vec4 g = texture(tex_genome, uv);
    
    vec2 mu_sigma = unpack2(g.r);
    vec2 radius_flow = unpack2(g.g);
    vec2 affinity_lambda = unpack2(g.b);
    vec2 secre_percep = unpack2(g.a);
    
    uint base = (idx_y * 64 + idx_x) * 10;
    
    a.data[base + 0] = state.r; // Mass
    a.data[base + 1] = mu_sigma.x; // mu
    a.data[base + 2] = mu_sigma.y; // sigma
    a.data[base + 3] = radius_flow.x; // radius
    a.data[base + 4] = radius_flow.y; // flow
    a.data[base + 5] = affinity_lambda.x; // affinity
    a.data[base + 6] = affinity_lambda.y; // lambda
    a.data[base + 7] = secre_percep.x; // secretion
    a.data[base + 8] = secre_percep.y; // perception
    a.data[base + 9] = 0.0; // pad
}

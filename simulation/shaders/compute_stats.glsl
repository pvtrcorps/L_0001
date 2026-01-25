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

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;

layout(set = 0, binding = 3, std430) buffer Stats {
    uint total_mass;
    uint population;
    uint histograms[80]; // 10 bins * 8 genes
} s;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed);
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    float mass = texture(tex_state, uv).r;
    
    if (mass > 0.05) {
        atomicAdd(s.total_mass, uint(mass * 1000.0));
        atomicAdd(s.population, 1);
        
        vec4 g = texture(tex_genome, uv);
        float genes[8];
        vec2 g0 = unpack2(g.r); genes[0] = g0.x; genes[1] = g0.y;
        vec2 g1 = unpack2(g.g); genes[2] = g1.x; genes[3] = g1.y;
        vec2 g2 = unpack2(g.b); genes[4] = g2.x; genes[5] = g2.y;
        vec2 g3 = unpack2(g.a); genes[6] = g3.x; genes[7] = g3.y;
        
        for (int i = 0; i < 8; i++) {
            int bin = int(clamp(genes[i], 0.0, 0.99) * 10.0);
            atomicAdd(s.histograms[i * 10 + bin], 1);
        }
    }
}

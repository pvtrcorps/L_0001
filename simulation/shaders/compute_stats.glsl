#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    // ... we just need u_res from the common block
    // but we must match the layout size/structure to bind the same UBO
    float u_dt;
    float u_seed;
    float u_R;
    float _u1, _u2, _u3;
    float u_mutation_rate;
    float u_base_decay;
    float u_init_clusters;
    float u_init_density;
    float _pad0;
    vec2 u_range_mu;
    vec2 u_range_sigma;
    vec2 u_range_radius;
    vec2 u_range_flow;
    vec2 u_range_affinity;
    vec2 u_range_lambda;
    float _pad1, _pad2, _pad3;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;

// Statistics Buffer
// Binding 3 is used for img_potential in other shaders, 
// but here we use valid binding index. Let's say binding 3.
layout(set = 0, binding = 3, std430) buffer Stats {
    uint total_mass_int;     // Scaled by 1000
    uint living_cell_count;
    // Histograms: 6 genes * 10 buckets = 60 uints
    // 0-9: Mu, 10-19: Sigma, 20-29: Radius, 30-39: Flow, 40-49: Affinity, 50-59: Lambda
    uint buckets[60];
} stats;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed);
    float a = float(bits >> 16) / 65535.0;
    float b = float(bits & 0xFFFFu) / 65535.0;
    return vec2(a, b);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    vec4 state = texture(tex_state, uv);
    float mass = state.r;
    
    // Accumulate total mass (scaled)
    if (mass > 0.0) {
        atomicAdd(stats.total_mass_int, uint(mass * 1000.0));
    }
    
    // Only count living cells for histograms
    if (mass > 0.05) {
        atomicAdd(stats.living_cell_count, 1);
        
        vec4 genome = texture(tex_genome, uv);
        float g_mu = genome.r;
        float g_sigma = genome.g;
        vec2 radius_flow = unpack2(genome.b);
        float g_radius = radius_flow.x;
        float g_flow = radius_flow.y; // note: unpack returns vec2(radius, flow)
        
        vec2 affinity_lambda = unpack2(genome.a);
        float g_affinity = affinity_lambda.x;
        float g_lambda = affinity_lambda.y;
        
        // Genes to array
        float genes[6];
        genes[0] = g_mu;
        genes[1] = g_sigma;
        genes[2] = g_radius;
        genes[3] = g_flow;
        genes[4] = g_affinity;
        genes[5] = g_lambda;
        
        for (int i = 0; i < 6; i++) {
            // Determine bucket 0-9
            uint bucket = uint(clamp(genes[i] * 10.0, 0.0, 9.0));
            uint index = uint(i) * 10u + bucket;
            atomicAdd(stats.buckets[index], 1);
        }
    }
}

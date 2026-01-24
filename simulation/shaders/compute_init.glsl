#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float u_repulsion_strength;
    float u_combat_damage;
    float u_identity_thr;
    float u_mutation_rate;
    float u_base_decay;
    float u_init_clusters;
    float u_init_density;
    float u_colonize_thr;
    vec2 u_range_mu;
    vec2 u_range_sigma;
    vec2 u_range_radius;
    vec2 u_range_flow;
    vec2 u_range_affinity;
    vec2 u_range_lambda;
    float _pad1, _pad2, _pad3;
} p;

layout(set = 0, binding = 1, rgba32f) uniform image2D img_state;
layout(set = 0, binding = 2, rgba32f) uniform image2D img_genome;

float hash(vec2 pt) {
    return fract(sin(dot(pt, vec2(12.9898, 78.233))) * 43758.5453);
}

float rand_range(vec2 seed, float minVal, float maxVal) {
    return minVal + hash(seed) * (maxVal - minVal);
}

// Generate a random genome based on seed
vec4 generate_genome(vec2 seed) {
    float mu = rand_range(seed + 0.1, p.u_range_mu.x, p.u_range_mu.y);
    float sigma = rand_range(seed + 0.2, p.u_range_sigma.x, p.u_range_sigma.y);
    float radius = rand_range(seed + 0.3, p.u_range_radius.x, p.u_range_radius.y);
    float flow = rand_range(seed + 0.4, p.u_range_flow.x, p.u_range_flow.y);
    float affinity = rand_range(seed + 0.5, p.u_range_affinity.x, p.u_range_affinity.y);
    float lambda = rand_range(seed + 0.6, p.u_range_lambda.x, p.u_range_lambda.y);
    
    // Pack
    uint p1 = uint(clamp(radius, 0.0, 1.0) * 65535.0) << 16 | uint(clamp(flow, 0.0, 1.0) * 65535.0);
    uint p2 = uint(clamp(affinity, 0.0, 1.0) * 65535.0) << 16 | uint(clamp(lambda, 0.0, 1.0) * 65535.0);
    
    float packed1 = uintBitsToFloat(p1);
    float packed2 = uintBitsToFloat(p2);
    
    return vec4(mu, sigma, packed1, packed2);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    
    // Determine Quadrant
    // Grid size from u_init_clusters (e.g., 2.0 -> 2x2 grid)
    float grid_n = max(1.0, floor(p.u_init_clusters));
    vec2 grid_pos = floor(uv * grid_n);
    
    // Seed for this quadrant (Constant for the whole block)
    vec2 quad_seed = grid_pos * 13.0 + p.u_seed;
    
    // Generate Genome for this Quadrant/Nation
    vec4 genome = generate_genome(quad_seed);
    
    // Generate State (Mass) inside the quadrant
    // Use salt density
    float local_seed = hash(uv * 100.0 + p.u_seed);
    float mass = 0.0;
    
    if (local_seed < p.u_init_density) {
        // Random start mass
        mass = 0.2 + hash(uv * 50.0) * 0.3;
    }
    
    // Store
    imageStore(img_state, uv_i, vec4(mass, 0.0, 0.0, 0.0));
    imageStore(img_genome, uv_i, genome);
}

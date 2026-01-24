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

layout(set = 0, binding = 1, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 2) uniform sampler2D tex_potential;
layout(set = 0, binding = 3) uniform sampler2D tex_new_genome; // Sampler (Reverted)
layout(set = 0, binding = 4, rgba32f) uniform image2D img_new_state;

const float MASS_SCALE = 100000.0; 

float growth(float U, float g_mu, float g_sigma) {
    float optimalU = 0.15 + g_mu * 0.35; 
    float tolerance = 0.03 + g_sigma * 0.12;
    float d = (U - optimalU) / tolerance;
    return 2.0 * exp(-0.5 * d * d) - 1.0;
}

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed);
    float a = float(bits >> 16) / 65535.0;
    float b = float(bits & 0xFFFFu) / 65535.0;
    return vec2(a, b);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    // 1. Read Atomic Mass
    uint m_uint = imageLoad(img_mass_accum, uv_i).r;
    
    // 2. Clear Atomic Mass
    imageStore(img_mass_accum, uv_i, uvec4(0));
    
    float mass = float(m_uint) / MASS_SCALE;
    
    // 3. Read preserved Vector/Age
    vec4 prevData = imageLoad(img_new_state, uv_i);
    vec2 velocity = prevData.gb; 
    float age = prevData.a;
    
    // 4. Growth
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    vec4 genome = texture(tex_new_genome, uv);
    float U = texture(tex_potential, uv).r;
    
    vec2 affinity_lambda = unpack2(genome.a);
    float g_lambda = affinity_lambda.y;
    
    float G = growth(U, genome.r, genome.g);
    float metabolism = 0.5 + g_lambda * 1.5; 
    
    float deltaMass = G * p.u_dt * metabolism;
    
    float finalMass = mass + deltaMass;
    
    // Cleanup Dust (Logic for mass only)
    if (finalMass < 0.01) {
        finalMass = 0.0;
        // Cannot wipe genome here. Handled by visualization threshold.
    }
    
    finalMass = clamp(finalMass, 0.0, 1.0);
    imageStore(img_new_state, uv_i, vec4(finalMass, velocity, age));
}

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
    float u_signal_diff;
    float u_signal_decay;
    vec2 u_range_secretion;
    vec2 u_range_perception;
    float _pad;
} p;

layout(set = 0, binding = 1, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 2) uniform sampler2D tex_potential;
layout(set = 0, binding = 3) uniform sampler2D tex_new_genome;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_signal;

const float MASS_SCALE = 100000.0; 

float growth(float U, float g_mu, float g_sigma) {
    float optimalU = 0.15 + g_mu * 0.35; 
    float tolerance = 0.03 + g_sigma * 0.12;
    float d = (U - optimalU) / max(tolerance, 0.001);
    return 2.0 * exp(-0.5 * d * d) - 1.0;
}

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed);
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    // 1. Read Atomic Mass
    uint m_uint = imageLoad(img_mass_accum, uv_i).r;
    imageStore(img_mass_accum, uv_i, uvec4(0));
    float mass = float(m_uint) / MASS_SCALE;
    
    // 2. Read preserved Vector/Age
    vec4 prevData = imageLoad(img_new_state, uv_i);
    vec2 velocity = prevData.gb; 
    float age = prevData.a;
    
    // 3. Genome & Potential
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    vec4 genome = texture(tex_new_genome, uv);
    vec2 mu_sigma = unpack2(genome.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    
    vec2 affinity_lambda = unpack2(genome.b);
    float g_lambda = affinity_lambda.y;
    
    vec2 secre_percep = unpack2(genome.a);
    float g_secretion = secre_percep.x;
    
    float U = texture(tex_potential, uv).r;
    
    // 4. Growth
    float G = growth(U, g_mu, g_sigma);
    float metabolism = 0.5 + g_lambda * 1.5; 
    
    float deltaMass = G * p.u_dt * metabolism;
    
    // --- METABOLIC TAX ---
    // Emitting signals costs mass!
    float metabolic_tax = g_secretion * 0.05 * p.u_dt; // [NEW]
    deltaMass -= metabolic_tax;
    
    float finalMass = mass + deltaMass;
    
    // Cleanup Dust
    if (finalMass < 0.01) finalMass = 0.0;
    finalMass = clamp(finalMass, 0.0, 1.0);
    
    // 5. Secretion
    // Only if cell is healthy
    if (finalMass > 0.05) {
        float currentSignal = imageLoad(img_new_signal, uv_i).r;
        float addedSignal = g_secretion * finalMass * p.u_dt * 0.5;
        float nextSignal = clamp(currentSignal + addedSignal, 0.0, 2.0); // HARD CAP
        imageStore(img_new_signal, uv_i, vec4(nextSignal, 0.0, 0.0, 0.0));
    }
    
    imageStore(img_new_state, uv_i, vec4(finalMass, velocity, age));
}

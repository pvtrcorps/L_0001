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

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_signal;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_potential;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed);
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

float gaussian(float x, float mu, float sigma) {
    float d = (x - mu) / max(sigma, 0.001);
    return exp(-0.5 * d * d);
}

// Species-Dependent Kernel
float kernel(float r, float g_mu, float g_sigma, float g_radius, float g_affinity) {
    float R = p.u_R * (0.5 + g_radius); 
    float norm_r = r / R;
    if (norm_r > 1.0) return 0.0;
    
    float r2_pos = 0.2 + g_mu * 0.45;
    float w1 = g_affinity * 1.5;
    float w3 = (1.0 - g_sigma) * 1.2;
    float w2 = 1.0; 
    float sig_base = 0.05 + g_sigma * 0.1;
    
    float k1 = gaussian(norm_r, 0.1, 0.07) * w1;
    float k2 = gaussian(norm_r, r2_pos, sig_base) * w2;
    float k3 = gaussian(norm_r, 0.85, 0.09) * w3;
    
    return k1 + k2 + k3;
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    
    // Genome Unpacking (8 Genes)
    vec4 g_tex = texture(tex_genome, uv);
    vec2 mu_sigma = unpack2(g_tex.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    
    vec2 radius_flow = unpack2(g_tex.g);
    float g_radius = radius_flow.x;
    
    vec2 affinity_lambda = unpack2(g_tex.b);
    float g_affinity = affinity_lambda.x;
    
    vec2 secre_percep = unpack2(g_tex.a);
    float g_perception = secre_percep.y;
    
    // Read Signal
    float signal_val = texture(tex_signal, uv).r;
    
    float R_max = p.u_R * 1.6;
    int maxR = int(R_max) + 1;
    
    float sum = 0.0;
    float totalWeight = 0.0;
    vec2 weightedGradient = vec2(0.0);
    
    for (int dy = -maxR; dy <= maxR; dy++) {
        for (int dx = -maxR; dx <= maxR; dx++) {
            vec2 offset = vec2(float(dx), float(dy));
            float r = length(offset);
            
            float w = kernel(r, g_mu, g_sigma, g_radius, g_affinity);
            if (w > 0.0) {
                vec2 sampleUV = uv + offset / p.u_res;
                float neighborMass = texture(tex_state, sampleUV).r;
                
                sum += neighborMass * w;
                totalWeight += w;
                
                if (r > 0.0) {
                    vec2 dir = offset / r;
                    weightedGradient += dir * neighborMass * w;
                }
            }
        }
    }
    
    float U = (totalWeight > 0.0) ? sum / totalWeight : 0.0;
    
    // Signal perception biases growth potential (Subtle influence)
    U += signal_val * (g_perception - 0.5) * 0.04; 
    U = max(0.0, U); // HARD CLAMP to prevent extinction
    
    vec2 gradU = (totalWeight > 0.0) ? weightedGradient / totalWeight : vec2(0.0);
    
    imageStore(img_potential, uv_i, vec4(U, gradU, 1.0));
}

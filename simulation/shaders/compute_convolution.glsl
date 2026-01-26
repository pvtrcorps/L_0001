#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float u_theta_A; // Previously _pad1
    float u_alpha_n; // Previously _pad2
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

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_signal;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_potential;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u; // Clear the forced normalization bit
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
            if (w > 0.0) {
                // Precise Integer Sampling to avoid sub-pixel drift
                ivec2 res_i = ivec2(p.u_res);
                ivec2 neighbor_coord = (uv_i + ivec2(dx, dy) + res_i) % res_i; // True Torus Wrap
                
                // Use texelFetch instead of texture(linear)
                float neighborMass = texelFetch(tex_state, neighbor_coord, 0).r;
                
                sum += neighborMass * w;
                totalWeight += w;
                
                if (r > 0.0) {
                    vec2 dir = offset / r;
                    weightedGradient += dir * neighborMass * w;
                }
            }
            }
        }
    }
    
    float U = (totalWeight > 0.0) ? sum / totalWeight : 0.0;
    
    // Growth derivative approximation G'(U)
    // For G(U) = 2*exp(-0.5*((U-mu)/sigma)^2) - 1
    // G'(U) = -2 * ((U-mu)/sigma^2) * exp(-0.5*((U-mu)/sigma)^2)
    float optimalU = 0.15 + g_mu * 0.35; 
    float tolerance = 0.03 + g_sigma * 0.12;
    float diff = (U - optimalU);
    float g_prime = -2.0 * (diff / max(tolerance * tolerance, 0.0001)) * exp(-0.5 * (diff * diff) / max(tolerance * tolerance, 0.0001));
    
    vec2 gradU = (totalWeight > 0.0) ? weightedGradient / totalWeight : vec2(0.0);
    
    // Store U in R, G' (growth force) in G, and keep original gradU in B,A or pack it
    // Actually, let's just use R for U and GB for gradU. The flow shader will compute G'.
    imageStore(img_potential, uv_i, vec4(U, gradU, g_prime));
}

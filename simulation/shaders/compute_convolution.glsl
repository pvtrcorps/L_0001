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
    float u_signal_advect; // Signal advection weight
    float u_beta; // Selection pressure for negotiation rule
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

// Mexican Hat Kernel (Negative Weights Enabled)
// Paper Eq 1: K(x) = Σ_j b_j * exp(-0.5 * ((x/(r*R) - a_j)/w_j)²)
float kernel(float r, vec4 genome_rg, vec4 genome_ba) {
    // Unpack kernel genes
    vec2 b1b2 = unpack2(genome_rg.r);
    vec2 b3a2 = unpack2(genome_rg.g);
    vec2 width_radius = unpack2(genome_rg.b);
    
    float b1_weight = b1b2.x;
    float b2_weight = b1b2.y;
    float b3_weight = b3a2.x;
    float a2_pos = b3a2.y;
    float kernel_width = width_radius.x;
    float kernel_radius = width_radius.y;
    
    // Map genes to kernel parameters
    
    // OFFICIAL FLOW LENIA: b weights are POSITIVE ONLY [0.001, 1.0]
    float b1 = 0.001 + b1_weight * 0.999;
    float b2 = 0.001 + b2_weight * 0.999; 
    float b3 = 0.001 + b3_weight * 0.999;
    
    float a1 = 0.15;
    float a2 = 0.3 + a2_pos * 0.4;
    float a3 = 0.85;
    
    // Widths
    float width_scale = 0.04 + kernel_width * 0.12;
    float w1 = width_scale;
    float w2 = width_scale * 1.2;
    float w3 = width_scale * 0.8;
    
    float r_scale = 0.5 + kernel_radius;
    float R_actual = p.u_R * r_scale;
    float norm_r = r / R_actual;
    if (norm_r > 1.0) return 0.0;
    
    // Gaussian bumps
    float k1 = b1 * gaussian(norm_r, a1, w1);
    float k2 = b2 * gaussian(norm_r, a2, w2);
    float k3 = b3 * gaussian(norm_r, a3, w3);
    
    return k1 + k2 + k3; // Raw kernel value, will be normalized by sum in loop
}

// Convert hue [0,1] to RGB using HSV with S=1, V=1
vec3 HueToRGB(float hue) {
    float h = hue * 6.0;
    float c = 1.0;  // Chroma (S*V = 1*1)
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    
    vec3 rgb;
    if (h < 1.0)      rgb = vec3(c, x, 0.0);
    else if (h < 2.0) rgb = vec3(x, c, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, c, x);
    else if (h < 4.0) rgb = vec3(0.0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0.0, c);
    else              rgb = vec3(c, 0.0, x);
    
    return rgb;
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    ivec2 res_i = ivec2(p.u_res);
    
    // Genome Unpacking (8 Genes from genome texture)
    vec4 g_tex = texture(tex_genome, uv);
    vec2 mu_sigma = unpack2(g_tex.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    
    vec2 radius_flow = unpack2(g_tex.g);
    float g_radius = radius_flow.x;
    
    vec2 affinity_lambda = unpack2(g_tex.b);
    float g_affinity = affinity_lambda.x;
    
    // Read Spectral Genes from state.a
    vec4 state = texture(tex_state, uv);
    vec2 spectral_genes = unpack2(state.a);
    float g_detection_hue = spectral_genes.y;  // Hue this species seeks
    
    // === SPECTRAL SIGNAL DETECTION ===
    vec3 myDetector = HueToRGB(g_detection_hue);
    
    // Read local signal and compute similarity
    vec3 signalVec = texture(tex_signal, uv).rgb;
    float U_signal = dot(signalVec, myDetector);  // Spectral similarity [-1, 1] -> [0, 3] for pure colors
    
    // Compute signal gradient using central differences
    ivec2 l_uv = (uv_i + ivec2(-1, 0) + res_i) % res_i;
    ivec2 r_uv = (uv_i + ivec2(1, 0) + res_i) % res_i;
    ivec2 u_uv = (uv_i + ivec2(0, -1) + res_i) % res_i;
    ivec2 d_uv = (uv_i + ivec2(0, 1) + res_i) % res_i;
    
    vec3 sL = texelFetch(tex_signal, l_uv, 0).rgb;
    vec3 sR = texelFetch(tex_signal, r_uv, 0).rgb;
    vec3 sU = texelFetch(tex_signal, u_uv, 0).rgb;
    vec3 sD = texelFetch(tex_signal, d_uv, 0).rgb;
    
    // Gradient of spectral similarity
    float simL = dot(sL, myDetector);
    float simR = dot(sR, myDetector);
    float simU = dot(sU, myDetector);
    float simD = dot(sD, myDetector);
    vec2 gradSignal = vec2(simR - simL, simD - simU);
    
    // === MASS CONVOLUTION (U_growth) ===
    float R_max = p.u_R * 1.6;
    int maxR = int(R_max) + 1;
    
    float sum = 0.0;
    float totalWeight = 0.0; // Restoring normalization accumulator
    vec2 weightedGradient = vec2(0.0);
    
    for (int dy = -maxR; dy <= maxR; dy++) {
        for (int dx = -maxR; dx <= maxR; dx++) {
            vec2 offset = vec2(float(dx), float(dy));
            float r = length(offset);
            
            float w = kernel(r, g_tex, g_tex);
            if (w > 0.0001) { // Positive weights only now
                // Precise Integer Sampling
                ivec2 neighbor_coord = (uv_i + ivec2(dx, dy) + res_i) % res_i;
                float neighborMass = texelFetch(tex_state, neighbor_coord, 0).r;
                
                sum += neighborMass * w;
                totalWeight += w; // Accumulate spatial weight
                
                if (r > 0.0) {
                    vec2 dir = offset / r;
                    weightedGradient += dir * neighborMass * w;
                }
            }
        }
    }
    
    // Step 1: Compute convolution K*A (NORMALIZED by spatial sum)
    // This matches: nK = K / sum(K) in official code
    float U_raw = (totalWeight > 0.0) ? sum / totalWeight : 0.0;
    
    // Step 2: Apply Growth Function G(U)
    // Read growth genes
    vec2 growth_genes = unpack2(g_tex.a);
    float growth_mu = growth_genes.x;
    float growth_sigma = growth_genes.y;
    
    // Restore Lenia ranges (U is normalized [0,1])
    float mu = growth_mu; // Use direct [0,1] from initialization (mapped by UI)
    float sigma = 0.001 + growth_sigma * 0.199; // Keep sigma scaled [0, 0.2] for stability
    
    float diff = (U_raw - mu);
    float exp_term = exp(-0.5 * (diff * diff) / max(sigma * sigma, 0.0001));
    float U_growth = 2.0 * exp_term - 1.0;  // Range: [-1, 1]
    
    // Normalize gradient relative to kernel strength
    vec2 gradU_growth = (totalWeight > 0.0) ? weightedGradient / totalWeight : vec2(0.0);
    
    // === OUTPUT ===
    // R: U_growth (for growth function G(U))
    // GB: gradU_growth (mass gradient for flow)
    // A: U_signal (spectral similarity for chemotaxis)
    // Note: gradSignal will be computed in flow shader from neighbors' U_signal values
    imageStore(img_potential, uv_i, vec4(U_growth, gradU_growth, U_signal));
}

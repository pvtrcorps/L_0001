#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    // 0. Globals
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float u_theta_A;
    float u_alpha_n;
    float u_temperature;
    float u_signal_advect;
    float u_beta;
    float u_signal_diff;
    float u_signal_decay;
    float u_flow_speed;
    float u_init_clusters;
    float u_init_density;
    float u_colonize_thr;
    
    // 1. Gene Ranges (16 Genes * 2) = 32 floats
    // Block A: Physiology
    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    // Block B: Morphology
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_growth_rate;
    // Block C: Social / Motor
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    // Block D: Senses
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;     // Genes 1-8
layout(set = 0, binding = 3) uniform sampler2D tex_signal;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_potential;
layout(set = 0, binding = 5) uniform sampler2D tex_genome_ext; // Genes 9-16 [NEW]

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u; // Clear normalized bit
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

float gaussian(float x, float mu, float sigma) {
    float d = (x - mu) / max(sigma, 0.001);
    return exp(-0.5 * d * d);
}

// Dynamic Kernel Generation based on Abstract Shape Genes
float kernel(float r, float R_actual, vec4 genome_1) {
    // Unpack Morphology Genes
    // R: Mu, Sigma
    // G: Radius, Viscosity
    // B: Shape A, Shape B
    // A: Shape C, Growth Rate
    
    vec2 shape_ab = unpack2(genome_1.b);
    vec2 shape_c_gr = unpack2(genome_1.a);
    
    float shape_a = shape_ab.x; // Ring Balance (Inner vs Outer)
    float shape_b = shape_ab.y; // Complexity / Texture
    float shape_c = shape_c_gr.x; // Ring Spacing / Position
    
    // Map Abstract Shape -> Concrete Kernel Weights (b1, b2, b3)
    // Shape A (0-1): 0 = Outer Ring Dominant, 1 = Inner Ring Dominant
    float b1 = 0.1 + shape_a * 0.9;
    float b3 = 0.1 + (1.0 - shape_a) * 0.9;
    
    // Shape B (0-1): Modulates the middle ring (b2)
    float b2 = shape_b;
    
    // Fixed positions for standard Lenia life
    float a1 = 0.15;
    float a2 = 0.35 + shape_c * 0.3; // Shape C modulates middle ring position
    float a3 = 0.85;
    
    // Widths (Standardized for stability)
    float w1 = 0.15;
    float w2 = 0.20;
    float w3 = 0.15;
    
    // Normalize distance by Species Radius
    float norm_r = r / R_actual;
    if (norm_r > 1.0) return 0.0;
    
    // Gaussian bumps
    float k1 = b1 * gaussian(norm_r, a1, w1);
    float k2 = b2 * gaussian(norm_r, a2, w2);
    float k3 = b3 * gaussian(norm_r, a3, w3);
    
    return k1 + k2 + k3;
}

vec3 HueToRGB(float hue) {
    float h = hue * 6.0;
    float c = 1.0;
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
    
    // 1. Read Genome 1 (Physiology/Morphology)
    vec4 g1 = texture(tex_genome, uv);
    
    // 2. Read Genome 2 (Behavior/Senses)
    vec4 g2 = texture(tex_genome_ext, uv);
    
    // Unpack Key Genes
    vec2 mu_sigma = unpack2(g1.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    
    vec2 rad_visc = unpack2(g1.g);
    float g_radius = rad_visc.x; // [0-1] relative to p.u_R? No, absolute multiplier?
    // Init maps it to u_range_radius. Let's assume that range is e.g. [0.5, 2.0]
    // So R_actual = p.u_R * g_radius
    
    vec2 hue_hue = unpack2(g2.a);
    float g_detection_hue = hue_hue.y;
    
    // === SPECTRAL SIGNAL ===
    vec3 myDetector = HueToRGB(g_detection_hue);
    vec3 signalVec = texture(tex_signal, uv).rgb;
    float U_signal = dot(signalVec, myDetector); 
    
    // === CONVOLUTION ===
    // Dynamic Radius!
    float R_actual = max(p.u_R * g_radius, 1.0);
    int maxR = int(R_actual) + 1; // Conservative bound
    
    // Optimization: Hard cap maxR to prevent massive loops if genes go wild
    maxR = min(maxR, 60); 
    
    float sum = 0.0;
    float totalWeight = 0.0;
    
    for (int dy = -maxR; dy <= maxR; dy++) {
        for (int dx = -maxR; dx <= maxR; dx++) {
            vec2 offset = vec2(float(dx), float(dy));
            float r = length(offset);
            
            // Pass R_actual to kernel so it normalizes correctly [0,1] inside the species radius
            float w = kernel(r, R_actual, g1);
            
            if (w > 0.0001) {
                ivec2 neighbor_coord = (uv_i + ivec2(dx, dy) + res_i) % res_i;
                float neighborMass = texelFetch(tex_state, neighbor_coord, 0).r;
                
                sum += neighborMass * w;
                totalWeight += w;
            }
        }
    }
    
    float U_raw = (totalWeight > 0.0) ? sum / totalWeight : 0.0;
    
    // === GROWTH G(U) ===
    // Use Species Specific Mu and Sigma
    float mu = g_mu; 
    float sigma = 0.001 + g_sigma * 0.2; // Scaling for stability
    
    float diff = (U_raw - mu);
    float exp_term = exp(-0.5 * (diff * diff) / max(sigma * sigma, 0.0001));
    float U_growth = 2.0 * exp_term - 1.0; 
    
    imageStore(img_potential, uv_i, vec4(U_growth, 0.0, 0.0, U_signal));
}

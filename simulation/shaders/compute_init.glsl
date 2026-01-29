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
    float u_signal_advect;
    float u_beta;
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

layout(set = 0, binding = 1, rgba32f) uniform image2D img_state;
layout(set = 0, binding = 2, rgba32f) uniform image2D img_genome;

float hash(vec2 pt) {
    return fract(sin(dot(pt, vec2(12.9898, 78.233))) * 43758.5453);
}

float pack2(float a, float b) {
    uint ia = uint(clamp(a, 0.0, 1.0) * 32767.0);
    uint ib = uint(clamp(b, 0.0, 1.0) * 32767.0);
    // Force bit 30 (0x40000000) to ensure Exponent is non-zero (Normalized Float)
    return uintBitsToFloat((ia << 15) | ib | 0x40000000u);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    
    // 1. Density Initialization (Grid/Cluster)
    float density = 0.0;
    
    float cell_x = floor(uv.x * p.u_init_clusters);
    float cell_y = floor(uv.y * p.u_init_clusters);
    float cell_hash = hash(vec2(cell_x, cell_y) + p.u_seed);
    
    if (cell_hash < p.u_init_density) {
        vec2 cell_center = (vec2(cell_x, cell_y) + 0.5) / p.u_init_clusters;
        float d = length(uv - cell_center) * p.u_init_clusters;
        density = smoothstep(0.4, 0.2, d);
    }
    
    // 2. Genome Generation (8 Kernel/Growth Genes + 2 Spectral Genes)
    // Species-specific genome per cluster
    vec2 species_seed = vec2(cell_x, cell_y) + p.u_seed;
    
    // === KERNEL GENES (6) ===
    float b1_weight = hash(species_seed + 1.0);  // Range [0, 1] - POSITIVE ONLY
    float b2_weight = hash(species_seed + 2.0);  // Range [0, 1]
    float b3_weight = hash(species_seed + 3.0);  // Range [0, 1]
    float a2_pos = hash(species_seed + 4.0);         // Middle ring position [0,1]
    float kernel_width_raw = hash(species_seed + 5.0);
    // Map Flow (Mobility) to Kernel Width [0,1] using u_range_flow
    float kernel_width = mix(p.u_range_flow.x, p.u_range_flow.y, kernel_width_raw);
    
    float kernel_radius_raw = hash(species_seed + 6.0); 
    // Map Radius using u_range_radius
    float kernel_radius = mix(p.u_range_radius.x, p.u_range_radius.y, kernel_radius_raw);
    
    // === GROWTH GENES (2) ===
    float growth_mu_raw = hash(species_seed + 7.0);
    // Map Mu (Archetype) using u_range_mu
    float growth_mu = mix(p.u_range_mu.x, p.u_range_mu.y, growth_mu_raw);
    
    float growth_sigma_raw = hash(species_seed + 8.0); 
    // Map Sigma (Stability) using u_range_sigma
    float growth_sigma = mix(p.u_range_sigma.x, p.u_range_sigma.y, growth_sigma_raw);
    
    // === SPECTRAL GENES (2) - For Chemical Signals ===
    float emission_hue = hash(species_seed + 9.0);   // Emitted signal hue
    float detection_hue = hash(species_seed + 10.0); // Detected signal hue
    
    // Pack 8 base genes into genome texture (RGBA32F)
    vec4 packedGenome = vec4(
        pack2(b1_weight, b2_weight),        // R: inner/middle weights
        pack2(b3_weight, a2_pos),           // G: outer weight + position
        pack2(kernel_width, kernel_radius), // B: width + size
        pack2(growth_mu, growth_sigma)      // A: growth params
    );
    
    // Pack spectral genes into state.a (replaces age tracking)
    float packedSpectral = pack2(emission_hue, detection_hue);
    
    imageStore(img_state, uv_i, vec4(density, 0.0, 0.0, packedSpectral));
    // Write genome everywhere in the cluster to prevent "Void Border" artifacts
    // when using bilinear sampling for mass but nearest for genome.
    imageStore(img_genome, uv_i, packedGenome);
}

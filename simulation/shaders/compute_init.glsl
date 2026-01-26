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
    
    // 2. Genome Generation (8 Base Genes + 2 Spectral Genes)
    // Species-specific base genome per cluster
    vec2 species_seed = vec2(cell_x, cell_y) + p.u_seed;
    
    float g_mu = mix(p.u_range_mu.x, p.u_range_mu.y, hash(species_seed + 1.0));
    float g_sigma = mix(p.u_range_sigma.x, p.u_range_sigma.y, hash(species_seed + 2.0));
    float g_radius = mix(p.u_range_radius.x, p.u_range_radius.y, hash(species_seed + 3.0));
    float g_flow = mix(p.u_range_flow.x, p.u_range_flow.y, hash(species_seed + 4.0));
    float g_affinity = mix(p.u_range_affinity.x, p.u_range_affinity.y, hash(species_seed + 5.0));
    float g_lambda = mix(p.u_range_lambda.x, p.u_range_lambda.y, hash(species_seed + 6.0));
    float g_secretion = mix(p.u_range_secretion.x, p.u_range_secretion.y, hash(species_seed + 7.0));
    float g_perception = mix(p.u_range_perception.x, p.u_range_perception.y, hash(species_seed + 8.0));
    
    // Spectral Genes (for chemotaxis signaling)
    float g_emission_hue = hash(species_seed + 9.0);   // [0, 1] hue of emitted signal
    float g_detection_hue = hash(species_seed + 10.0); // [0, 1] hue this species seeks
    
    // Pack 8 base genes into RGBA32F (genome texture)
    vec4 packedGenome = vec4(
        pack2(g_mu, g_sigma),
        pack2(g_radius, g_flow),
        pack2(g_affinity, g_lambda),
        pack2(g_secretion, g_perception)
    );
    
    // Pack spectral genes into state.a (replaces age tracking)
    float packedSpectral = pack2(g_emission_hue, g_detection_hue);
    
    imageStore(img_state, uv_i, vec4(density, 0.0, 0.0, packedSpectral));
    // Write genome everywhere in the cluster to prevent "Void Border" artifacts
    // when using bilinear sampling for mass but nearest for genome.
    imageStore(img_genome, uv_i, packedGenome);
}

#[compute]
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
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 4) uniform sampler2D tex_genome_ext; // Using binding 4 for extension texture

layout(set = 0, binding = 3, std430) buffer Stats {
    uint total_mass;
    uint population;
    uint histograms[160]; // 10 bins * 16 genes
} s;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u; // Clear normalization bit
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    float mass = texture(tex_state, uv).r;
    
    if (mass > 0.05) {
        atomicAdd(s.total_mass, uint(mass * 1000.0));
        atomicAdd(s.population, 1);
        
        vec4 g1 = texture(tex_genome, uv);
        vec4 g2 = texture(tex_genome_ext, uv);
        
        float genes[16];
        
        // Genome 1
        vec2 d1 = unpack2(g1.r); genes[0] = d1.x; genes[1] = d1.y;
        vec2 d2 = unpack2(g1.g); genes[2] = d2.x; genes[3] = d2.y;
        vec2 d3 = unpack2(g1.b); genes[4] = d3.x; genes[5] = d3.y;
        vec2 d4 = unpack2(g1.a); genes[6] = d4.x; genes[7] = d4.y;
        
        // Genome 2
        vec2 d5 = unpack2(g2.r); genes[8] = d5.x; genes[9] = d5.y;
        vec2 d6 = unpack2(g2.g); genes[10] = d6.x; genes[11] = d6.y;
        vec2 d7 = unpack2(g2.b); genes[12] = d7.x; genes[13] = d7.y;
        vec2 d8 = unpack2(g2.a); genes[14] = d8.x; genes[15] = d8.y;
        
        for (int i = 0; i < 16; i++) {
            // Check for NaNs just in case
            float val = genes[i];
            if (isnan(val)) val = 0.0;
            int bin = int(clamp(val, 0.0, 0.99) * 10.0);
            atomicAdd(s.histograms[i * 10 + bin], 1);
        }
    }
}

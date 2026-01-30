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
layout(set = 0, binding = 3) uniform sampler2D tex_genome_ext; // [NEW]

layout(set = 0, binding = 4, std430) buffer Analysis {
    float data[]; // 4096 * 18 floats (aligned)
} a;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u; // Clear normalization bit
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

void main() {
    uint idx_x = gl_GlobalInvocationID.x;
    uint idx_y = gl_GlobalInvocationID.y;
    if (idx_x >= 64 || idx_y >= 64) return;
    
    // Sample the grid at 64x64 intervals
    vec2 uv = (vec2(idx_x, idx_y) + 0.5) / 64.0;
    
    vec4 state = texture(tex_state, uv);
    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);
    
    // Unpack Genome 1 (Physiology/Morphology)
    vec2 mu_sigma = unpack2(g1.r);
    vec2 rad_visc = unpack2(g1.g);
    vec2 shape_ab = unpack2(g1.b);
    vec2 shape_c_gr = unpack2(g1.a);
    
    // Unpack Genome 2 (Behavior/Senses)
    vec2 aff_rep = unpack2(g2.r);
    vec2 tol_mob = unpack2(g2.g);
    vec2 sec_sens = unpack2(g2.b);
    vec2 hues = unpack2(g2.a);
    
    // Stride = 18 floats per cell
    uint base = (idx_y * 64 + idx_x) * 18;
    
    // Standard Analysis Block (18 floats)
    a.data[base + 0] = state.r; // Mass (0)
    
    // Physiology (1-4)
    a.data[base + 1] = mu_sigma.x;
    a.data[base + 2] = mu_sigma.y;
    a.data[base + 3] = rad_visc.x;
    a.data[base + 4] = rad_visc.y;
    
    // Morphology (5-8)
    a.data[base + 5] = shape_ab.x;
    a.data[base + 6] = shape_ab.y;
    a.data[base + 7] = shape_c_gr.x;
    a.data[base + 8] = shape_c_gr.y;
    
    // Behavior (9-12)
    a.data[base + 9] = aff_rep.x;
    a.data[base + 10] = aff_rep.y;
    a.data[base + 11] = tol_mob.x;
    a.data[base + 12] = tol_mob.y;
    
    // Senses (13-16)
    a.data[base + 13] = sec_sens.x;
    a.data[base + 14] = sec_sens.y;
    a.data[base + 15] = hues.x; // Emission Hue
    a.data[base + 16] = hues.y; // Detection Hue
    
    a.data[base + 17] = 0.0; // Padding/Flag
}

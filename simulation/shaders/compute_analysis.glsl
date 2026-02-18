#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
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

    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_inertia;
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;

    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_genome_ext;

layout(set = 0, binding = 4, std430) buffer Analysis {
    float data[];
} a;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u;
    float av = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float bv = float(bits & 0x7FFFu) / 32767.0;
    return vec2(av, bv);
}

void main() {
    uint idx_x = gl_GlobalInvocationID.x;
    uint idx_y = gl_GlobalInvocationID.y;
    if (idx_x >= 64 || idx_y >= 64) return;

    vec2 uv = (vec2(idx_x, idx_y) + 0.5) / 64.0;

    vec4 state = texture(tex_state, uv);
    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);

    vec2 mu_sigma = unpack2(g1.r);
    vec2 rad_visc = unpack2(g1.g);
    vec2 shape_ab = unpack2(g1.b);
    vec2 shape_c_aniso = unpack2(g1.a);

    vec2 comp_rep = unpack2(g2.r);
    vec2 plast_mob = unpack2(g2.g);
    vec2 sec_sens = unpack2(g2.b);
    vec2 hues = unpack2(g2.a);

    uint base = (idx_y * 64 + idx_x) * 18;

    a.data[base + 0] = state.r;

    // Physiology
    a.data[base + 1] = mu_sigma.x;
    a.data[base + 2] = mu_sigma.y;
    a.data[base + 3] = rad_visc.x;
    a.data[base + 4] = rad_visc.y;

    // Morphology
    a.data[base + 5] = shape_ab.x;
    a.data[base + 6] = shape_ab.y;
    a.data[base + 7] = shape_c_aniso.x;
    a.data[base + 8] = shape_c_aniso.y; // anisotropy

    // Body plan / Motor
    a.data[base + 9] = comp_rep.x;      // compactness
    a.data[base + 10] = comp_rep.y;     // hollow core repulsion
    a.data[base + 11] = plast_mob.x;    // plasticity
    a.data[base + 12] = plast_mob.y;    // mobility

    // Senses
    a.data[base + 13] = sec_sens.x;
    a.data[base + 14] = sec_sens.y;
    a.data[base + 15] = hues.x;
    a.data[base + 16] = hues.y;

    // Morphology state signal
    a.data[base + 17] = state.a;        // polarity
}

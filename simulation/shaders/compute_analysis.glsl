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
    float u_fluid_momentum;

    // 1. Gene Ranges (16 Genes * 2) = 32 floats
    // Block A: Physiology
    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    // Block B: Morphology
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_inertia;
    // Block C: Social / Motor
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    // Block D: Senses
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;

    // 2. Wind / Atmosphere
    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;

    // 3. Signal + Morph Extras
    float u_signal_force_strength;
    float u_signal_emission_strength;
    float u_interaction_beta;
    float u_morph_anisotropy_gain;

    // 4. Morph Controls + Cleanup
    float u_colonize_thr;
    float u_morph_polarity_gain;
    float u_morph_plasticity_gain;
    float u_morph_self_propulsion_gain;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_genome_ext;
layout(set = 0, binding = 4) uniform sampler2D tex_polarity;

layout(set = 0, binding = 5, std430) buffer Analysis {
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

    // Selection Bias: Favor center pixels of the analysis block for this resolution.
    float best_score = -1.0;
    vec2 best_uv = (vec2(idx_x, idx_y) + 0.5) / 64.0;

    ivec2 res_i = ivec2(p.u_res);
    ivec2 block_min = ivec2(
        int((idx_x * uint(res_i.x)) / 64u),
        int((idx_y * uint(res_i.y)) / 64u)
    );
    ivec2 block_max = ivec2(
        int(((idx_x + 1u) * uint(res_i.x)) / 64u),
        int(((idx_y + 1u) * uint(res_i.y)) / 64u)
    );
    block_max = max(block_max, block_min + ivec2(1));

    vec2 block_center = (vec2(block_min + block_max) * 0.5) - vec2(0.5);

    for (int py = block_min.y; py < block_max.y; py++) {
        for (int px_i = block_min.x; px_i < block_max.x; px_i++) {
            vec2 sample_uv = (vec2(float(px_i), float(py)) + 0.5) / p.u_res;
            float m = texture(tex_state, sample_uv).r;

            // Score = mass / (1 + center-distance bias), computed in pixel space
            // so behavior scales correctly across 1024/2048/4096 and non-power-of-two sizes.
            vec2 d = vec2(float(px_i), float(py)) - block_center;
            float dist_sq = dot(d, d);
            float score = m / (1.0 + 0.05 * dist_sq);

            if (score > best_score) {
                best_score = score;
                best_uv = sample_uv;
            }
        }
    }

    vec4 state = texture(tex_state, best_uv);
    vec4 g1 = texture(tex_genome, best_uv);
    vec4 g2 = texture(tex_genome_ext, best_uv);

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

    // Morphology state signal (polarity angle 0..1 from dedicated vector field)
    vec2 pol = texture(tex_polarity, best_uv).xy;
    float pol_len = length(pol);
    float pol_angle = 0.0;
    if (pol_len > 0.0001) {
        pol_angle = atan(pol.y, pol.x) / 6.28318530718;
        if (pol_angle < 0.0) pol_angle += 1.0;
    }
    a.data[base + 17] = pol_angle;
}

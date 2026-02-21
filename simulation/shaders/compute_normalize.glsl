#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#define PI 3.14159265359

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float u_theta_A;
    float u_alpha_n;
    float u_temperature;
    float u_detritus_advect;     // was: u_signal_advect
    float u_beta;
    float u_detritus_diff;       // was: u_signal_diff
    float u_mass_decay_rate;     // was: u_signal_decay
    float u_flow_speed;
    float u_init_clusters;
    float u_init_density;
    float u_fluid_momentum;

    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_inertia;
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;

    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;

    float u_detritus_force_strength;  // was: u_signal_force_strength
    float u_mass_digest_rate;         // was: u_signal_emission_strength
    float u_interaction_beta;
    float u_morph_anisotropy_gain;
    float u_colonize_thr;
    float u_morph_polarity_gain;
    float u_morph_plasticity_gain;
    float u_morph_self_propulsion_gain;
} p;

layout(set = 0, binding = 1, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 2) uniform sampler2D tex_potential;
layout(set = 0, binding = 3) uniform sampler2D tex_old_state;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_detritus;  // was: img_new_signal. R=mass, G=hue (read/write)
layout(set = 0, binding = 6, r32ui) uniform uimage2D img_winner_tracker;
layout(set = 0, binding = 7, rgba32f) uniform image2D img_new_genome;
layout(set = 0, binding = 8) uniform sampler2D tex_old_genome;
layout(set = 0, binding = 9) uniform sampler2D tex_genome_ext;
layout(set = 0, binding = 10, rgba32f) uniform image2D img_new_genome_ext;
layout(set = 0, binding = 11) uniform sampler2D tex_old_polarity;
layout(set = 0, binding = 12, rgba32f) uniform image2D img_new_polarity;

// Detritus accumulators from compute_detritus.glsl
layout(set = 0, binding = 13, r32ui) uniform uimage2D img_detritus_mass_accum;
layout(set = 0, binding = 14, r32ui) uniform uimage2D img_detritus_hue_accum;

const float MASS_SCALE = 100000000.0;
const float DETRITUS_SCALE = 100000000.0;
const float HUE_SCALE = 10000.0;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u;
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

vec2 normalize_or(vec2 v, vec2 fallback) {
    float l = length(v);
    if (l < 0.0001) return fallback;
    return v / l;
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;

    // === 1. Reconstruct living mass from flow accumulator ===
    uint m_uint = imageLoad(img_mass_accum, uv_i).r;
    imageStore(img_mass_accum, uv_i, uvec4(0));
    float mass = float(m_uint) / MASS_SCALE;

    vec4 flowData = imageLoad(img_new_state, uv_i);
    vec2 velocity = flowData.gb;

    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    vec2 final_polarity = normalize_or(texture(tex_old_polarity, uv).xy, vec2(1.0, 0.0));

    // === 2. Winner tracking (identity transfer) ===
    uint packed = imageLoad(img_winner_tracker, uv_i).r;
    imageStore(img_winner_tracker, uv_i, uvec4(0));

    vec4 finalGenome1 = texture(tex_old_genome, uv);
    vec4 finalGenome2 = texture(tex_genome_ext, uv);

    if (packed != 0u) {
        uint winner_idx = packed & 0x3FFFFFu;

        ivec2 res = ivec2(p.u_res);
        ivec2 winner_coords = ivec2(winner_idx % uint(res.x), winner_idx / uint(res.x));
        vec2 winner_uv = (vec2(winner_coords) + 0.5) / p.u_res;

        finalGenome1 = texture(tex_old_genome, winner_uv);
        finalGenome2 = texture(tex_genome_ext, winner_uv);
        float winner_angle = imageLoad(img_new_state, winner_coords).a;
        final_polarity = vec2(cos(winner_angle * 6.28318530718), sin(winner_angle * 6.28318530718));

        bool is_raw_null = dot(finalGenome1, finalGenome1) < 0.0001;

        vec2 t_r = unpack2(finalGenome1.r);
        vec2 t_g = unpack2(finalGenome1.g);
        vec2 t_a = unpack2(finalGenome1.a);
        float trait_sum = t_r.x + t_r.y + t_g.x + t_a.y;

        if (is_raw_null || trait_sum < 0.01) {
            finalGenome1 = vec4(0.0);
            finalGenome2 = vec4(0.0);
            final_polarity = vec2(0.0);
        }
    }

    imageStore(img_new_genome, uv_i, finalGenome1);
    imageStore(img_new_genome_ext, uv_i, finalGenome2);

    // === 3. Reconstruct detritus from transport accumulators ===
    uint det_mass_uint = imageLoad(img_detritus_mass_accum, uv_i).r;
    imageStore(img_detritus_mass_accum, uv_i, uvec4(0));
    uint det_hue_uint = imageLoad(img_detritus_hue_accum, uv_i).r;
    imageStore(img_detritus_hue_accum, uv_i, uvec4(0));

    float det_mass = float(det_mass_uint) / DETRITUS_SCALE;
    float det_hue = (det_mass_uint > 0u)
        ? (float(det_hue_uint) / float(det_mass_uint)) / HUE_SCALE
        : 0.0;
    det_hue = clamp(det_hue, 0.0, 1.0);

    // === 4. Extract genes ===
    vec2 sec_sens = unpack2(finalGenome2.b);
    float g_secretion = sec_sens.x;    // Repurposed as metabolism rate

    vec2 hues = unpack2(finalGenome2.a);
    float g_emission_hue = hues.x;

    float finalMass = mass;
    float identity_prune_thr = max(0.00005, p.u_colonize_thr * 0.35);
    bool has_identity = dot(finalGenome1, finalGenome1) > 0.0001;

    // === 5. Metabolic Decay: living mass → detritus ===
    if (finalMass > 0.0001 && has_identity) {
        // Decay rate scaled by metabolism gene (secretion repurposed)
        float decay = finalMass * g_secretion * p.u_mass_decay_rate * p.u_dt;
        decay = min(decay, finalMass * 0.1);  // Cap at 10% per step for stability
        finalMass -= decay;

        // Add decayed mass to detritus with this species' hue
        float new_det = det_mass + decay;
        // Mass-weighted hue average
        det_hue = (det_mass > 0.0001)
            ? (det_hue * det_mass + g_emission_hue * decay) / new_det
            : g_emission_hue;
        det_mass = new_det;
    }

    // === 6. Feeding: detritus → living mass (anti-cannibalism) ===
    if (finalMass > 0.0001 && has_identity && det_mass > 0.0001) {
        // Anti-cannibalism via sin²(π × circular_hue_distance)
        float hue_dist = abs(g_emission_hue - det_hue);
        if (hue_dist > 0.5) hue_dist = 1.0 - hue_dist;
        float feed_factor = sin(PI * hue_dist);
        feed_factor *= feed_factor;  // sin² — 0 for self, 1 for opposite

        float absorbed = min(det_mass, finalMass * p.u_mass_digest_rate * feed_factor * p.u_dt);
        finalMass += absorbed;
        det_mass -= absorbed;
    }

    // === 7. Identity prune → transfer to detritus (unified orphan handling) ===
    if (finalMass > 0.0 && finalMass < identity_prune_thr && has_identity) {
        // Instead of creating orphan mass, transfer to detritus layer
        float new_det = det_mass + finalMass;
        det_hue = (det_mass > 0.0001)
            ? (det_hue * det_mass + g_emission_hue * finalMass) / new_det
            : g_emission_hue;
        det_mass = new_det;
        finalMass = 0.0;
        finalGenome1 = vec4(0.0);
        finalGenome2 = vec4(0.0);
        final_polarity = vec2(0.0);
    }

    // Handle mass without identity — also goes to detritus (cleanup rule)
    if (finalMass > 0.0 && !has_identity) {
        det_mass += finalMass;
        // No hue contribution from identity-less mass (keep existing hue)
        finalMass = 0.0;
    }

    // === 8. Write detritus ===
    imageStore(img_detritus, uv_i, vec4(max(0.0, det_mass), clamp(det_hue, 0.0, 1.0), 0.0, 0.0));

    // === 9. Write final state ===
    if (finalMass <= 0.0) {
        imageStore(img_new_state, uv_i, vec4(0.0));
        imageStore(img_new_genome, uv_i, vec4(0.0));
        imageStore(img_new_genome_ext, uv_i, vec4(0.0));
        imageStore(img_new_polarity, uv_i, vec4(0.0));
    } else {
        imageStore(img_new_state, uv_i, vec4(finalMass, velocity, 0.0));
        imageStore(img_new_genome, uv_i, finalGenome1);
        imageStore(img_new_genome_ext, uv_i, finalGenome2);
        imageStore(img_new_polarity, uv_i, vec4(final_polarity, 0.0, 0.0));
    }
}

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

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_potential;
layout(set = 0, binding = 4, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 7) uniform sampler2D tex_detritus;  // R=mass, G=hue
layout(set = 0, binding = 8, r32ui) uniform uimage2D img_winner_tracker;
layout(set = 0, binding = 9) uniform sampler2D tex_genome_ext;
layout(set = 0, binding = 10) uniform sampler2D tex_polarity;

const float MASS_SCALE = 100000000.0;
const float TWO_PI = 6.28318530718;
const float PI = 3.14159265359;

uint pcg_hash(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v.x + v.y;
}

float hash(vec2 pt) {
    uvec2 p = uvec2(floatBitsToUint(pt.x), floatBitsToUint(pt.y));
    return float(pcg_hash(p)) * (1.0 / 4294967296.0);
}

uint pcg_hash_1d(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u;
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
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

uint calculate_gumbel_score(uint amt, float pot, float beta, vec2 seed_uv) {
    if (amt == 0u) return 0u;

    float log_mass = log(float(amt));
    float pot_term = pot * beta;

    float u = hash(seed_uv);
    u = clamp(u, 0.0000001, 0.9999999);
    float gumbel = -log(-log(u));

    float score_float = log_mass + pot_term + gumbel;
    float map_s = (score_float + 10.0) * 4.0;

    return uint(clamp(map_s, 1.0, 255.0));
}

vec2 normalize_or(vec2 v, vec2 fallback) {
    float l = length(v);
    if (l < 0.0001) return fallback;
    return v / l;
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;

    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;

    vec4 state = texture(tex_state, uv);
    float myMass = state.r;

    if (myMass <= 0.0) {
        imageStore(img_new_state, uv_i, vec4(0.0));
        return;
    }

    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);
    bool is_void = dot(g1, g1) < 0.0001;

    vec2 rad_visc = unpack2(g1.g);
    float g_viscosity = rad_visc.y;

    vec2 shape_c_aniso = unpack2(g1.a);
    float base_anisotropy = clamp(shape_c_aniso.y * p.u_morph_anisotropy_gain, 0.0, 1.0);

    vec2 comp_rep = unpack2(g2.r);
    float g_compactness = comp_rep.x;

    vec2 plast_mob = unpack2(g2.g);
    float g_plasticity = clamp(plast_mob.x * p.u_morph_plasticity_gain, 0.0, 1.0);
    // Keep anisotropy active on peripheral mass so body plans do not collapse to cores.
    float mass_gate = 0.32 + 0.68 * smoothstep(0.02, 0.22, myMass);
    float g_anisotropy = base_anisotropy * mass_gate * (1.0 - 0.35 * g_plasticity);
    float g_mobility = plast_mob.y;

    vec2 sec_sens = unpack2(g2.b);
    float g_sensitivity = sec_sens.y;

    vec2 hues = unpack2(g2.a);
    float g_emission_hue = hues.x;
    float g_detection_hue = hues.y;

    vec2 pixel_size = 1.0 / p.u_res;

    vec4 pot_TL = texture(tex_potential, uv + vec2(-1, -1) * pixel_size);
    vec4 pot_TC = texture(tex_potential, uv + vec2(0, -1) * pixel_size);
    vec4 pot_TR = texture(tex_potential, uv + vec2(1, -1) * pixel_size);
    vec4 pot_ML = texture(tex_potential, uv + vec2(-1, 0) * pixel_size);
    vec4 pot_MC = texture(tex_potential, uv);
    vec4 pot_MR = texture(tex_potential, uv + vec2(1, 0) * pixel_size);
    vec4 pot_BL = texture(tex_potential, uv + vec2(-1, 1) * pixel_size);
    vec4 pot_BC = texture(tex_potential, uv + vec2(0, 1) * pixel_size);
    vec4 pot_BR = texture(tex_potential, uv + vec2(1, 1) * pixel_size);

    float beta = p.u_interaction_beta;
    float cp_TL = pot_TL.r + beta * pot_TL.b;
    float cp_TC = pot_TC.r + beta * pot_TC.b;
    float cp_TR = pot_TR.r + beta * pot_TR.b;
    float cp_ML = pot_ML.r + beta * pot_ML.b;
    float cp_MR = pot_MR.r + beta * pot_MR.b;
    float cp_BL = pot_BL.r + beta * pot_BL.b;
    float cp_BC = pot_BC.r + beta * pot_BC.b;
    float cp_BR = pot_BR.r + beta * pot_BR.b;

    float gx = -1.0 * cp_TL + -2.0 * cp_ML + -1.0 * cp_BL
             + 1.0 * cp_TR + 2.0 * cp_MR + 1.0 * cp_BR;
    float gy = -1.0 * cp_TL + -2.0 * cp_TC + -1.0 * cp_TR
             + 1.0 * cp_BL + 2.0 * cp_BC + 1.0 * cp_BR;
    vec2 gradGrowth = vec2(gx, gy);

    float dx = -1.0 * pot_TL.g + -2.0 * pot_ML.g + -1.0 * pot_BL.g
             + 1.0 * pot_TR.g + 2.0 * pot_MR.g + 1.0 * pot_BR.g;
    float dy = -1.0 * pot_TL.g + -2.0 * pot_TC.g + -1.0 * pot_TR.g
             + 1.0 * pot_BL.g + 2.0 * pot_BC.g + 1.0 * pot_BR.g;
    vec2 gradDensity = vec2(dx, dy);

    // Detritus chemotaxis: navigate toward edible detritus
    float dp_TL = texture(tex_detritus, uv + vec2(-1, -1) * pixel_size).r;
    float dp_TC = texture(tex_detritus, uv + vec2(0, -1) * pixel_size).r;
    float dp_TR = texture(tex_detritus, uv + vec2(1, -1) * pixel_size).r;
    float dp_ML = texture(tex_detritus, uv + vec2(-1, 0) * pixel_size).r;
    float dp_MR = texture(tex_detritus, uv + vec2(1, 0) * pixel_size).r;
    float dp_BL = texture(tex_detritus, uv + vec2(-1, 1) * pixel_size).r;
    float dp_BC = texture(tex_detritus, uv + vec2(0, 1) * pixel_size).r;
    float dp_BR = texture(tex_detritus, uv + vec2(1, 1) * pixel_size).r;

    float dgx = -1.0 * dp_TL + -2.0 * dp_ML + -1.0 * dp_BL
              + 1.0 * dp_TR + 2.0 * dp_MR + 1.0 * dp_BR;
    float dgy = -1.0 * dp_TL + -2.0 * dp_TC + -1.0 * dp_TR
              + 1.0 * dp_BL + 2.0 * dp_BC + 1.0 * dp_BR;
    vec2 gradDetritus = vec2(dgx, dgy);

    // Modulate navigation by hue selectivity: use detection_hue (prey preference)
    // cos²(π × Δhue) peaks at 1.0 when detritus matches detection target, drops to 0 at opposite
    float local_det_hue = texture(tex_detritus, uv).g;
    float nav_hue_dist = abs(g_detection_hue - local_det_hue);
    if (nav_hue_dist > 0.5) nav_hue_dist = 1.0 - nav_hue_dist;
    float nav_factor = cos(PI * nav_hue_dist);
    nav_factor *= nav_factor;  // cos² — 1 when det_hue matches detection target, 0 at opposite
    gradDetritus *= nav_factor;

    float alpha = clamp((myMass / 2.0) * (myMass / 2.0), 0.0, 1.0);

    vec2 polarity = normalize_or(texture(tex_polarity, uv).xy, vec2(1.0, 0.0));
    vec2 polarity_perp = vec2(-polarity.y, polarity.x);

    float along = dot(gradGrowth, polarity);
    float across = dot(gradGrowth, polarity_perp);
    float anis_gain = mix(1.0, 1.55, g_anisotropy);
    vec2 gradGrowthAniso = polarity * (along * anis_gain) + polarity_perp * (across / anis_gain);

    float plastic_boost = clamp(length(gradDensity) * (0.25 + g_plasticity * 2.0), 0.0, 1.5);

    float polarity_gain = max(0.0, p.u_morph_polarity_gain);
    vec2 totalAttraction = gradGrowthAniso
                         + gradDetritus * (g_sensitivity * p.u_detritus_force_strength)
                         + polarity * (0.05 + 0.45 * g_anisotropy) * polarity_gain;

    vec2 totalRepulsion = gradDensity * mix(1.35, 0.5, g_compactness);

    vec2 flow_field = (1.0 - alpha) * totalAttraction - alpha * totalRepulsion;

    float force_mult = p.u_flow_speed * (0.2 + g_mobility * 1.8) * (1.0 + 0.35 * plastic_boost);
    vec2 target_vel = force_mult * flow_field;

    // Active locomotion term along internal polarity, so species can still move
    // in weak-gradient scenarios (e.g. no signal pull / low inter-species forces).
    float polarity_gate = clamp(polarity_gain, 0.0, 1.0);
    float self_propulsion_gain = max(0.0, p.u_morph_self_propulsion_gain) * polarity_gate;
    // Separate gate so active propulsion does not vanish on thin structures.
    float propulsion_gate = 0.45 + 0.55 * smoothstep(0.01, 0.18, myMass);
    float self_propulsion_speed = self_propulsion_gain
                                * (0.10 + 0.90 * g_mobility)
                                * (0.25 + 0.75 * g_anisotropy)
                                * (0.85 + 0.30 * g_plasticity)
                                * propulsion_gate;
    vec2 self_propulsion = polarity * self_propulsion_speed;
    target_vel += self_propulsion;

    float tv_len = length(target_vel);
    float max_v = 2.0 + g_mobility * 1.5;
    if (tv_len > max_v) target_vel = (target_vel / tv_len) * max_v;

    vec2 old_vel = state.gb;
    float gene_momentum = mix(0.05, 0.95, g_anisotropy);
    float momentum = clamp(0.6 * p.u_fluid_momentum + 0.4 * gene_momentum, 0.0, 1.0);
    float responsiveness = (1.0 - momentum) * 10.0;
    vec2 vel = mix(old_vel, target_vel, clamp(responsiveness * p.u_dt, 0.05, 1.0));
    vel *= clamp(1.0 - g_viscosity * p.u_dt * 2.0, 0.0, 1.0);

    // Explicit polarity update: history + growth gradient + detritus gradient.
    vec2 growth_dir = normalize_or(gradGrowthAniso, polarity);
    vec2 detritus_dir = normalize_or(gradDetritus, growth_dir);
    vec2 vel_dir = normalize_or(vel, growth_dir);

    float w_hist = (0.55 + 0.25 * g_anisotropy) * polarity_gain;
    float w_growth = (0.30 + 0.25 * g_anisotropy) * polarity_gain;
    // If detritus force is 0, detritus must not affect direction update.
    float detritus_gate = clamp(p.u_detritus_force_strength, 0.0, 1.0);
    float w_detritus = (0.15 + 0.35 * g_sensitivity) * polarity_gain * detritus_gate;
    float w_vel = 0.15 + 0.20 * g_plasticity;

    vec2 desired_raw = polarity * w_hist + growth_dir * w_growth + detritus_dir * w_detritus + vel_dir * w_vel;
    vec2 desired_dir = normalize_or(desired_raw, polarity);

    float new_angle = atan(desired_dir.y, desired_dir.x) / TWO_PI;
    if (new_angle < 0.0) new_angle += 1.0;

    vec2 pos_next = uv * p.u_res + vel * p.u_dt - 0.5;
    pos_next = mod(pos_next, p.u_res);
    if (pos_next.x < 0.0) pos_next.x += p.u_res.x;
    if (pos_next.y < 0.0) pos_next.y += p.u_res.y;

    vec2 center_f = floor(pos_next + 0.5);
    ivec2 center_i = ivec2(center_f);
    vec2 delta = pos_next - center_f;

    float sigma = max(p.u_temperature, 0.1) + 0.25 * g_plasticity;

    float total_weight = 0.0;
    float weights[9];
    ivec2 offsets[9];

    int idx = 0;
    for (int dy_i = -1; dy_i <= 1; dy_i++) {
        for (int dx_i = -1; dx_i <= 1; dx_i++) {
            vec2 dist_vec = abs(delta - vec2(float(dx_i), float(dy_i)));
            vec2 sz = 0.5 - dist_vec + sigma;
            vec2 w_axis = clamp(sz, 0.0, 1.0);
            float w = w_axis.x * w_axis.y;

            weights[idx] = w;
            offsets[idx] = ivec2(dx_i, dy_i);
            total_weight += w;
            idx++;
        }
    }

    if (total_weight < 0.0001) total_weight = 1.0;
    float norm_factor = 1.0 / total_weight;

    uint total_amount = uint(round(myMass * MASS_SCALE));
    uint identity_transfer_thr = uint(max(0.00005, p.u_colonize_thr * 0.35) * MASS_SCALE);

    if (total_amount > 0u) {
        uint remaining = total_amount;
        int target_remainder_idx = 4;
        float max_w = -1.0;

        for (int i = 0; i < 9; i++) {
            float w = (i == 4) ? weights[i] + 0.0001 : weights[i];
            if (w > max_w) {
                max_w = w;
                target_remainder_idx = i;
            }
        }

        for (int i = 0; i < 9; i++) {
            if (i == target_remainder_idx) continue;

            float w = weights[i] * norm_factor;
            if (w < 0.001) continue;

            uint amount = uint(floor(float(total_amount) * w));
            if (amount > remaining) amount = remaining;
            remaining -= amount;

            if (amount == 0u) continue;

            ivec2 target_uv = (center_i + offsets[i] + ivec2(p.u_res)) % ivec2(p.u_res);

            imageAtomicAdd(img_mass_accum, target_uv, amount);

            float source_pot = texture(tex_potential, uv).r;
            uint src_idx = (uint(uv_i.y) * uint(p.u_res.x) + uint(uv_i.x));

            vec2 noise_uv = uv + vec2(float(i) * 0.1, p.u_seed);
            uint score_8bit = calculate_gumbel_score(amount, source_pot, p.u_beta, noise_uv);

            uint jitter_2bit = pcg_hash_1d(src_idx) & 0x3u;
            uint packed_comp = (score_8bit << 24u) | (jitter_2bit << 22u) | (src_idx & 0x3FFFFFu);

            if (score_8bit > 0u && amount > identity_transfer_thr && !is_void) {
                imageAtomicMax(img_winner_tracker, target_uv, packed_comp);
            }
        }

        if (remaining > 0u) {
            ivec2 remainder_uv = (center_i + offsets[target_remainder_idx] + ivec2(p.u_res)) % ivec2(p.u_res);
            imageAtomicAdd(img_mass_accum, remainder_uv, remaining);
        }
    }

    imageStore(img_new_state, uv_i, vec4(0.0, vel.x, vel.y, new_angle));
}

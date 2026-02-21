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
    float u_detritus_advect;     // was: u_signal_advect
    float u_beta;
    float u_detritus_diff;       // was: u_signal_diff
    float u_mass_decay_rate;     // was: u_signal_decay
    float u_flow_speed;
    float u_init_clusters;
    float u_init_density;
    float u_fluid_momentum;

    // 1. Gene Ranges
    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_inertia;
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;

    // 2. Wind / Atmosphere
    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;

    // 3. Detritus + Morph Extras
    float u_detritus_force_strength;  // was: u_signal_force_strength
    float u_mass_digest_rate;         // was: u_signal_emission_strength
    float u_interaction_beta;
    float u_morph_anisotropy_gain;

    // 4. Morph Controls + Cleanup
    float u_colonize_thr;
    float u_morph_polarity_gain;
    float u_morph_plasticity_gain;
    float u_morph_self_propulsion_gain;
} p;

// Source detritus (read): R=mass, G=hue
layout(set = 0, binding = 1) uniform sampler2D tex_detritus_src;
// Atomic accumulator for conservative mass transport
layout(set = 0, binding = 2, r32ui) uniform uimage2D img_detritus_mass_accum;
// Atomic accumulator for hue (mass-weighted, scaled to integer)
layout(set = 0, binding = 3, r32ui) uniform uimage2D img_detritus_hue_accum;

const float DETRITUS_SCALE = 100000000.0;  // Same precision as mass
const float HUE_SCALE = 10000.0;           // Hue precision (4 decimal places)

// === Noise functions (same as old compute_signal.glsl) ===

vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(dot(hash22(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
                   dot(hash22(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
               mix(dot(hash22(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
                   dot(hash22(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
}

vec2 curl_noise(vec2 p, float t) {
    float eps = 0.1;
    vec2 p_t = p + vec2(t * 0.1, -t * 0.05);
    float n1 = noise(p_t + vec2(0, eps));
    float n2 = noise(p_t - vec2(0, eps));
    float n3 = noise(p_t + vec2(eps, 0));
    float n4 = noise(p_t - vec2(eps, 0));
    float x = (n1 - n2) / (2.0 * eps);
    float y = (n3 - n4) / (2.0 * eps);
    return vec2(x, -y);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;

    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;

    // Read current detritus
    vec4 det = texture(tex_detritus_src, uv);
    float myMass = det.r;
    float myHue = det.g;

    if (myMass <= 0.0) return;  // Nothing to transport

    // === 1. Compute velocity field (diffusion + wind) ===
    
    // Diffusion: move toward neighbors with lower density (Sobel gradient)
    float d_TL = texture(tex_detritus_src, uv + vec2(-1, -1) * px).r;
    float d_TC = texture(tex_detritus_src, uv + vec2( 0, -1) * px).r;
    float d_TR = texture(tex_detritus_src, uv + vec2( 1, -1) * px).r;
    float d_ML = texture(tex_detritus_src, uv + vec2(-1,  0) * px).r;
    float d_MR = texture(tex_detritus_src, uv + vec2( 1,  0) * px).r;
    float d_BL = texture(tex_detritus_src, uv + vec2(-1,  1) * px).r;
    float d_BC = texture(tex_detritus_src, uv + vec2( 0,  1) * px).r;
    float d_BR = texture(tex_detritus_src, uv + vec2( 1,  1) * px).r;

    // Sobel gradient of detritus density
    float gx = -1.0 * d_TL + -2.0 * d_ML + -1.0 * d_BL
              + 1.0 * d_TR + 2.0 * d_MR + 1.0 * d_BR;
    float gy = -1.0 * d_TL + -2.0 * d_TC + -1.0 * d_TR
              + 1.0 * d_BL + 2.0 * d_BC + 1.0 * d_BR;
    vec2 gradDensity = vec2(gx, gy);

    // Diffusion velocity: move DOWN the gradient (from high to low density)
    vec2 vel_diff = -gradDensity * p.u_detritus_diff;

    // Wind velocity (curl noise)
    vec2 vel_wind = vec2(0.0);
    if (p.u_wind_strength > 0.0) {
        vec2 noise_uv = uv * p.u_wind_scale;
        vel_wind = curl_noise(noise_uv, p.u_time * p.u_wind_speed) * p.u_wind_strength;
    }

    // Combined velocity
    vec2 vel = (vel_diff + vel_wind * p.u_detritus_advect) * p.u_dt;

    // === 2. Reintegration Tracking (conservative transport) ===
    // Same algorithm as compute_flow_conservative.glsl
    
    vec2 pos_next = uv * p.u_res + vel - 0.5;
    pos_next = mod(pos_next, p.u_res);
    if (pos_next.x < 0.0) pos_next.x += p.u_res.x;
    if (pos_next.y < 0.0) pos_next.y += p.u_res.y;

    vec2 center_f = floor(pos_next + 0.5);
    ivec2 center_i = ivec2(center_f);
    vec2 delta = pos_next - center_f;

    // Temperature/spread of the distribution
    float sigma = max(p.u_temperature, 0.1);

    float total_weight = 0.0;
    float weights[9];
    ivec2 offsets[9];

    int idx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 dist_vec = abs(delta - vec2(float(dx), float(dy)));
            vec2 sz = 0.5 - dist_vec + sigma;
            vec2 w_axis = clamp(sz, 0.0, 1.0);
            float w = w_axis.x * w_axis.y;

            weights[idx] = w;
            offsets[idx] = ivec2(dx, dy);
            total_weight += w;
            idx++;
        }
    }

    if (total_weight < 0.0001) total_weight = 1.0;
    float norm_factor = 1.0 / total_weight;

    // === 3. Distribute mass via atomics (perfectly conservative) ===
    uint total_amount = uint(round(myMass * DETRITUS_SCALE));
    uint hue_encoded = uint(round(clamp(myHue, 0.0, 1.0) * HUE_SCALE));

    if (total_amount > 0u) {
        uint remaining = total_amount;
        int target_remainder_idx = 4;  // center
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
            imageAtomicAdd(img_detritus_mass_accum, target_uv, amount);
            // Mass-weighted hue: accumulate (hue × mass) for later averaging
            imageAtomicAdd(img_detritus_hue_accum, target_uv, amount * hue_encoded);
        }

        if (remaining > 0u) {
            ivec2 remainder_uv = (center_i + offsets[target_remainder_idx] + ivec2(p.u_res)) % ivec2(p.u_res);
            imageAtomicAdd(img_detritus_mass_accum, remainder_uv, remaining);
            imageAtomicAdd(img_detritus_hue_accum, remainder_uv, remaining * hue_encoded);
        }
    }
}

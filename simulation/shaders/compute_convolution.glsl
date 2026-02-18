#version 450

#define WORKGROUP_SIZE 16
#define MAX_RADIUS 20
#define TILE_WIDTH (WORKGROUP_SIZE + 2 * MAX_RADIUS)
#define TILE_AREA (TILE_WIDTH * TILE_WIDTH)
#define PI 3.14159265359

layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;

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
    float u_fluid_momentum;

    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_inertia;
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;

    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;

    float u_signal_force_strength;
    float u_signal_emission_strength;
    float u_interaction_beta;
    float u_morph_anisotropy_gain;
    float u_colonize_thr;
    float u_morph_polarity_gain;
    float u_morph_plasticity_gain;
    float u_pad0;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_signal;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_potential;
layout(set = 0, binding = 5) uniform sampler2D tex_genome_ext;

shared vec2 tile_cache[TILE_AREA];

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u;
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

struct KernelParams {
    float b1, b2, b3;
    float a1, a2, a3;
    float w1, w2, w3;
};

float eval_kernel(float r, float R_actual, KernelParams k) {
    float norm_r = r / R_actual;
    if (norm_r > 1.0) return 0.0;

    float k1 = k.b1 * exp(-0.5 * ((norm_r - k.a1) / k.w1) * ((norm_r - k.a1) / k.w1));
    float k2 = k.b2 * exp(-0.5 * ((norm_r - k.a2) / k.w2) * ((norm_r - k.a2) / k.w2));
    float k3 = k.b3 * exp(-0.5 * ((norm_r - k.a3) / k.w3) * ((norm_r - k.a3) / k.w3));

    return k1 + k2 + k3;
}

float get_interaction_strength(float my_detection, float their_emission) {
    float diff = abs(my_detection - their_emission);
    if (diff > 0.5) diff = 1.0 - diff;
    return cos(2.0 * PI * diff);
}

void main() {
    ivec2 res_i = ivec2(p.u_res);
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);
    ivec2 group_id = ivec2(gl_WorkGroupID.xy);

    ivec2 tile_base = group_id * WORKGROUP_SIZE - ivec2(MAX_RADIUS);

    uint thread_idx = gl_LocalInvocationIndex;
    uint total_threads = WORKGROUP_SIZE * WORKGROUP_SIZE;

    for (uint i = thread_idx; i < TILE_AREA; i += total_threads) {
        int tx = int(i) % TILE_WIDTH;
        int ty = int(i) / TILE_WIDTH;

        ivec2 global_pos = (tile_base + ivec2(tx, ty));
        global_pos = (global_pos % res_i + res_i) % res_i;

        float mass = texelFetch(tex_state, global_pos, 0).r;
        vec4 g_ext_packed = texelFetch(tex_genome_ext, global_pos, 0);
        vec2 hues = unpack2(g_ext_packed.a);
        float emission_hue = hues.x;

        tile_cache[i] = vec2(mass, emission_hue);
    }

    barrier();

    if (gid.x >= int(p.u_res.x) || gid.y >= int(p.u_res.y)) return;

    vec2 uv = (vec2(gid) + 0.5) / p.u_res;

    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);
    vec4 state = texture(tex_state, uv);

    if (dot(g1, g1) < 0.0001) {
        imageStore(img_potential, gid, vec4(0.0));
        return;
    }

    vec2 mu_sigma = unpack2(g1.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;

    vec2 rad_visc = unpack2(g1.g);
    float g_radius = rad_visc.x;

    vec2 hue_hue = unpack2(g2.a);
    float my_emission_hue = hue_hue.x;
    float g_detection_hue = hue_hue.y;

    vec2 shape_ab = unpack2(g1.b);
    vec2 shape_c_aniso = unpack2(g1.a);
    float g_shape_c = shape_c_aniso.x;
    float base_anisotropy = clamp(shape_c_aniso.y * p.u_morph_anisotropy_gain, 0.0, 1.0);

    vec2 comp_rep = unpack2(g2.r);
    float g_compactness = comp_rep.x;
    float g_repulsion = comp_rep.y;

    vec2 plast_mob = unpack2(g2.g);
    float g_plasticity = clamp(plast_mob.x * p.u_morph_plasticity_gain, 0.0, 1.0);
    float mass_gate = smoothstep(0.06, 0.25, state.r);
    float g_anisotropy = base_anisotropy * mass_gate * (1.0 - 0.35 * g_plasticity);

    float R_actual = max(p.u_R * g_radius, 1.0);
    R_actual = min(R_actual, float(MAX_RADIUS));

    KernelParams kp;
    kp.b1 = (0.1 + shape_ab.x * 0.9) - g_repulsion * 1.5;
    kp.b2 = shape_ab.y - g_repulsion * 0.5;
    kp.b3 = mix(1.15, 0.7, g_compactness) * (0.1 + (1.0 - shape_ab.x) * 0.9);

    kp.a1 = 0.15;
    kp.a2 = 0.35 + g_shape_c * 0.3;
    kp.a3 = 0.85;

    kp.w1 = 0.15;
    kp.w2 = 0.20;
    kp.w3 = 0.15;

    float angle = state.a * (2.0 * PI);
    vec2 axis = vec2(cos(angle), sin(angle));
    vec2 axis_perp = vec2(-axis.y, axis.x);
    float axis_stretch = mix(1.0, 1.65, g_anisotropy);

    int loopR = int(ceil(R_actual));

    float sumAll = 0.0;
    float sumInteract = 0.0;
    float weightAll = 0.0;
    float weightInteract = 0.0;

    ivec2 my_tile_pos = lid + ivec2(MAX_RADIUS);

    for (int dy = -loopR; dy <= loopR; dy++) {
        for (int dx = -loopR; dx <= loopR; dx++) {
            int tx = my_tile_pos.x + dx;
            int ty = my_tile_pos.y + dy;
            int idx = ty * TILE_WIDTH + tx;

            vec2 cell_data = tile_cache[idx];
            float neighborMass = cell_data.x;
            float neighborHue = cell_data.y;

            vec2 d = vec2(float(dx), float(dy));
            float d_long = dot(d, axis);
            float d_lat = dot(d, axis_perp);
            float r_ell = length(vec2(d_long / axis_stretch, d_lat * axis_stretch));

            float w = eval_kernel(r_ell, R_actual, kp);

            if (w > 0.0001 && neighborMass >= p.u_colonize_thr) {
                sumAll += neighborMass * w;
                weightAll += w;

                float strength = get_interaction_strength(g_detection_hue, neighborHue);
                strength = mix(strength, strength * (1.0 + 0.75 * g_plasticity), g_plasticity);
                sumInteract += neighborMass * w * strength;
                weightInteract += w;
            }
        }
    }

    float U_density = (weightAll > 0.0) ? sumAll / weightAll : 0.0;

    float sigma = 0.001 + g_sigma * 0.2;
    float diff = (U_density - g_mu);
    float exp_term = exp(-0.5 * (diff * diff) / max(sigma * sigma, 0.0001));
    float U_growth = 2.0 * exp_term - 1.0;

    float U_interact = (weightInteract > 0.0) ? sumInteract / weightInteract : 0.0;

    imageStore(img_potential, gid, vec4(U_growth, U_density, U_interact, g_anisotropy));
}

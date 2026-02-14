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
    float u_genetic_barrier;
} p;

layout(set = 0, binding = 1, std430) buffer SpawnData {
    vec4 spawn_a; // uv_x, uv_y, radius_px, mass
    vec4 spawn_b; // mode_id, pad0, pad1, pad2
    vec4 g0;
    vec4 g1;
    vec4 g2;
    vec4 g3;
} s;

layout(set = 0, binding = 2, rgba32f) uniform image2D img_state;
layout(set = 0, binding = 3, rgba32f) uniform image2D img_genome;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_genome_ext;

float pack2(float a, float b) {
    uint ia = uint(clamp(a, 0.0, 1.0) * 32767.0);
    uint ib = uint(clamp(b, 0.0, 1.0) * 32767.0);
    return uintBitsToFloat((ia << 15) | ib | 0x40000000u);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;

    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    vec2 to_center = uv - s.spawn_a.xy;
    to_center.x = (to_center.x > 0.5) ? to_center.x - 1.0 : to_center.x;
    to_center.x = (to_center.x < -0.5) ? to_center.x + 1.0 : to_center.x;
    to_center.y = (to_center.y > 0.5) ? to_center.y - 1.0 : to_center.y;
    to_center.y = (to_center.y < -0.5) ? to_center.y + 1.0 : to_center.y;

    float r_uv = s.spawn_a.z / max(p.u_res.x, p.u_res.y);
    float dist = length(to_center);
    if (dist > r_uv) return;

    float falloff = 1.0 - smoothstep(0.0, r_uv, dist);
    float add_mass = s.spawn_a.w * falloff;

    vec4 old_state = imageLoad(img_state, uv_i);
    float mode_norm = clamp(round(s.spawn_b.x), 0.0, 3.0) / 3.0;
    imageStore(img_state, uv_i, vec4(max(old_state.r, add_mass), 0.0, 0.0, mode_norm));

    vec4 genome_lo = vec4(
        pack2(s.g0.x, s.g0.y),
        pack2(s.g0.z, s.g0.w),
        pack2(s.g1.x, s.g1.y),
        pack2(s.g1.z, s.g1.w)
    );

    vec4 genome_hi = vec4(
        pack2(s.g2.x, s.g2.y),
        pack2(s.g2.z, s.g2.w),
        pack2(s.g3.x, s.g3.y),
        pack2(s.g3.z, s.g3.w)
    );

    imageStore(img_genome, uv_i, genome_lo);
    imageStore(img_genome_ext, uv_i, genome_hi);
}

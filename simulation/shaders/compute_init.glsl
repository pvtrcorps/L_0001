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

layout(set = 0, binding = 1, rgba32f) uniform image2D img_state;
layout(set = 0, binding = 2, rgba32f) uniform image2D img_genome;
layout(set = 0, binding = 3, rgba32f) uniform image2D img_genome_ext;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_polarity;

float hash(vec2 pt) {
    return fract(sin(dot(pt, vec2(12.9898, 78.233))) * 43758.5453);
}

float pack2(float a, float b) {
    uint ia = uint(clamp(a, 0.0, 1.0) * 32767.0);
    uint ib = uint(clamp(b, 0.0, 1.0) * 32767.0);
    return uintBitsToFloat((ia << 15) | ib | 0x40000000u);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;

    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;

    // 1. Density Initialization
    float density = 0.0;
    float cell_x = floor(uv.x * p.u_init_clusters);
    float cell_y = floor(uv.y * p.u_init_clusters);
    float cell_hash = hash(vec2(cell_x, cell_y) + p.u_seed);

    if (cell_hash < p.u_init_density) {
        vec2 cell_center = (vec2(cell_x, cell_y) + 0.5) / p.u_init_clusters;
        float d = length(uv - cell_center) * p.u_init_clusters;
        density = smoothstep(0.30001, 0.3, d)*0.25;
    }

    if (density < 0.0001) {
        imageStore(img_state, uv_i, vec4(0.0));
        imageStore(img_genome, uv_i, vec4(0.0));
        imageStore(img_genome_ext, uv_i, vec4(0.0));
        imageStore(img_polarity, uv_i, vec4(0.0));
        return;
    }

    // 2. Gene Generation (Species Seed)
    vec2 species_seed = vec2(cell_x, cell_y) + p.u_seed;

    #define GEN_GENE(offset, range_vec) mix(range_vec.x, range_vec.y, hash(species_seed + offset))

    // Block A: Physiology
    float g_mu = GEN_GENE(1.0, p.r_mu);
    float g_sigma = GEN_GENE(2.0, p.r_sigma);
    float g_radius = GEN_GENE(3.0, p.r_radius);
    float g_viscosity = GEN_GENE(4.0, p.r_viscosity);

    // Block B: Morphology
    float g_shape_a = GEN_GENE(5.0, p.r_shape_a);
    float g_shape_b = GEN_GENE(6.0, p.r_shape_b);
    float g_shape_c = GEN_GENE(7.0, p.r_shape_c);
    float g_anisotropy = GEN_GENE(8.0, p.r_inertia);     // repurposed inertia gene

    // Block C: Body Plan / Social / Motor
    float g_compactness = GEN_GENE(9.0, p.r_affinity);   // repurposed affinity gene
    float g_repulsion = GEN_GENE(10.0, p.r_repulsion);
    float g_plasticity = GEN_GENE(11.0, p.r_density_tol);// repurposed density_tol gene
    float g_mobility = GEN_GENE(12.0, p.r_mobility);

    // Block D: Senses
    float g_secretion = GEN_GENE(13.0, p.r_secretion);
    float g_sensitivity = GEN_GENE(14.0, p.r_sensitivity);
    float g_emission_hue = GEN_GENE(15.0, p.r_emission_hue);
    float g_detection_hue = GEN_GENE(16.0, p.r_detection_hue);

    // 3. Packing
    vec4 packed_1 = vec4(
        pack2(g_mu, g_sigma),
        pack2(g_radius, g_viscosity),
        pack2(g_shape_a, g_shape_b),
        pack2(g_shape_c, g_anisotropy)
    );

    vec4 packed_2 = vec4(
        pack2(g_compactness, g_repulsion),
        pack2(g_plasticity, g_mobility),
        pack2(g_secretion, g_sensitivity),
        pack2(g_emission_hue, g_detection_hue)
    );

    // Polarity is now stored in a dedicated texture as a unit direction vector.
    float polarity_angle = hash(species_seed + 17.0) * 6.28318530718;
    vec2 polarity = vec2(cos(polarity_angle), sin(polarity_angle));

    imageStore(img_state, uv_i, vec4(density, 0.0, 0.0, 0.0));
    imageStore(img_genome, uv_i, packed_1);
    imageStore(img_genome_ext, uv_i, packed_2);
    imageStore(img_polarity, uv_i, vec4(polarity, 0.0, 0.0));
}

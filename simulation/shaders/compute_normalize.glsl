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
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_ring_width;
    // Block C: Social / Motor
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    // Block D: Senses
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;
} p;

layout(set = 0, binding = 1, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 2) uniform sampler2D tex_potential;
layout(set = 0, binding = 3) uniform sampler2D tex_old_state; 
layout(set = 0, binding = 4, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_signal;
layout(set = 0, binding = 6, r32ui) uniform uimage2D img_winner_tracker;
layout(set = 0, binding = 7, rgba32f) uniform image2D img_new_genome;
layout(set = 0, binding = 8) uniform sampler2D tex_old_genome;
layout(set = 0, binding = 9) uniform sampler2D tex_genome_ext;      // Source Ext (Read Old)
layout(set = 0, binding = 10, rgba32f) uniform image2D img_new_genome_ext; // Target Ext (Write New)

const float MASS_SCALE = 100000.0; 

// PCG Hash (1d)
uint pcg_hash_1d(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u; // Clear normalization bit
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

// Convert hue [0,1] to RGB using HSV with S=1, V=1
vec3 HueToRGB(float hue) {
    float h = hue * 6.0;
    float c = 1.0;  // Chroma (S*V = 1*1)
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

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    // 1. Read Atomic Mass
    uint m_uint = imageLoad(img_mass_accum, uv_i).r;
    imageStore(img_mass_accum, uv_i, uvec4(0)); // Reset
    float mass = float(m_uint) / MASS_SCALE;
    
    // 2. Retrieve Velocity calculated by Flow Shader
    // We stored it in img_new_state.gb in the previous pass
    vec4 flowData = imageLoad(img_new_state, uv_i);
    vec2 velocity = flowData.gb; 
    
    // Optional: Smooth or Damping could happen here, but we keep it raw for now.
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    // 3. Genome Update (Winner-Takes-All)
    // If multiple source blocks moved mass here, the one with largest contribution (tracked in winner_tracker) wins.
    uint packed = imageLoad(img_winner_tracker, uv_i).r;
    imageStore(img_winner_tracker, uv_i, uvec4(0));
    
    vec4 finalGenome1 = texture(tex_old_genome, uv);
    vec4 finalGenome2 = texture(tex_genome_ext, uv); 
    
    if (packed != 0) {
        uint winner_idx = packed & 0xFFFFFFu; 
        
        ivec2 res = ivec2(p.u_res);
        ivec2 winner_coords = ivec2(winner_idx % res.x, winner_idx / res.x);
        vec2 winner_uv = (vec2(winner_coords) + 0.5) / p.u_res;
        
        finalGenome1 = texture(tex_old_genome, winner_uv);
        finalGenome2 = texture(tex_genome_ext, winner_uv);
    }
    
    // Write BOTH genomes to new generation (Move identity)
    imageStore(img_new_genome, uv_i, finalGenome1);
    imageStore(img_new_genome_ext, uv_i, finalGenome2);
    
    // 4. Secretion Logic using Real Genes (Genome 2)
    // Secretion: Ext B.x
    vec2 sec_sens = unpack2(finalGenome2.b);
    float g_secretion = sec_sens.x;
    
    // Emission Hue: Ext A.x
    vec2 hues = unpack2(finalGenome2.a);
    float g_emission_hue = hues.x;
    
    // 5. Final Mass & Emission
    float finalMass = mass;
    
    if (finalMass > 0.05) {
        vec3 emittedColor = HueToRGB(g_emission_hue);
        vec4 currentSignal = imageLoad(img_new_signal, uv_i); 
        
        // Add emission
        vec3 addedSignal = emittedColor * (finalMass * g_secretion * p.u_dt * 0.5);
        vec3 nextSignal = currentSignal.rgb + addedSignal;
        
        // Decay/Clamp
        imageStore(img_new_signal, uv_i, vec4(nextSignal, 0.0));
    }
    
    // Store final state (Mass, Velocity, Debug/Extra)
    imageStore(img_new_state, uv_i, vec4(finalMass, velocity, 0.0));
}

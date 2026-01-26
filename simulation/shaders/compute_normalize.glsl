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
    float _pad4;
    float _pad5;
    float u_init_clusters;
    float u_init_density;
    float u_colonize_thr;
    vec2 u_range_mu;
    vec2 u_range_sigma;
    vec2 u_range_radius;
    vec2 u_range_flow;
    vec2 u_range_affinity;
    vec2 u_range_lambda;
    float u_signal_diff;
    float u_signal_decay;
    vec2 u_range_secretion;
    vec2 u_range_perception;
    float _pad_end;
} p;

layout(set = 0, binding = 1, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 2) uniform sampler2D tex_potential;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_signal;
layout(set = 0, binding = 6, r32ui) uniform uimage2D img_winner_tracker;
layout(set = 0, binding = 7, rgba32f) uniform image2D img_new_genome;
layout(set = 0, binding = 8) uniform sampler2D tex_old_genome;

const float MASS_SCALE = 100000.0; 

// 100% Flow Lenia: No additive growth.
// Final mass is strictly determined by advection.

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

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    // 1. Read Atomic Mass
    uint m_uint = imageLoad(img_mass_accum, uv_i).r;
    imageStore(img_mass_accum, uv_i, uvec4(0));
    float mass = float(m_uint) / MASS_SCALE;
    
    // 2. Read preserved Vector/Age
    vec4 prevData = imageLoad(img_new_state, uv_i);
    vec2 velocity = prevData.gb; 
    float age = prevData.a;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    // 3. Genome Update (Winner-Takes-All)
    uint packed = imageLoad(img_winner_tracker, uv_i).r;
    imageStore(img_winner_tracker, uv_i, uvec4(0)); // Clear for next frame
    
    vec4 finalGenome = texture(tex_old_genome, uv); // Preserve by default if possible
    if (packed != 0) {
        // Unpack Lower 20 bits for Scrambled Index
        uint scrambled_idx = packed & 0xFFFFF; 
        
        // Descramble using same mask as flow shader (u_seed)
        // Ensure strictly matching hash function!
        uint seed_bits = floatBitsToUint(p.u_seed);
        uint frame_mask = pcg_hash_1d(seed_bits) & 0xFFFFFFu;
        uint winner_idx = scrambled_idx ^ frame_mask;
        
        ivec2 res = ivec2(p.u_res);
        ivec2 winner_coords = ivec2(winner_idx % res.x, winner_idx / res.x);
        vec2 winner_uv = (vec2(winner_coords) + 0.5) / p.u_res;
        finalGenome = texture(tex_old_genome, winner_uv);
    }
    // Removed "Clear on Low Mass" logic to prevent visual artifacts at edges.
    // The genome persists even if mass is zero ("trace").
    
    imageStore(img_new_genome, uv_i, finalGenome);
    
    vec2 secre_percep = unpack2(finalGenome.a);
    float g_secretion = secre_percep.x;
    
    float U = texture(tex_potential, uv).r;
    
    // 4. Finalization
    float finalMass = mass;
    
    // Cleanup Dust (Disabled for strict mass conservation)
    // if (finalMass < 0.001) finalMass = 0.0;
    
    // 5. Secretion (Optional signaling component)
    // 5. Secretion (Optional signaling component) - RGB VECTOR IMPLEMENTATION
    if (finalMass > 0.05) {
        // Unpack genes to create Sensory Identity Vector
        vec2 mu_sigma = unpack2(finalGenome.r);
        vec2 radius_flow = unpack2(finalGenome.g);
        vec2 affinity_lambda = unpack2(finalGenome.b);
        
        // Identity Vector: Mu (Physiology), Flow (Movement), Affinity (Structure)
        vec3 identityVec = vec3(mu_sigma.x, radius_flow.y, affinity_lambda.x);
        
        vec4 currentSignal = imageLoad(img_new_signal, uv_i);
        
        // Emit vector signal: Amount = mass * secretion * identity
        vec3 addedSignal = identityVec * (finalMass * g_secretion * p.u_dt * 0.5);
        
        // Add to existing signal and decay/clamp
        vec3 nextSignal = currentSignal.rgb + addedSignal;
        nextSignal = clamp(nextSignal, 0.0, 2.0); // Allow some saturation
        
        imageStore(img_new_signal, uv_i, vec4(nextSignal, 0.0));
    }
    
    imageStore(img_new_state, uv_i, vec4(finalMass, velocity, age));
}

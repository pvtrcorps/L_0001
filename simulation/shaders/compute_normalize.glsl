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
layout(set = 0, binding = 3) uniform sampler2D tex_old_state;  // For spectral gene advection
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

float pack2(float a, float b) {
    uint ia = uint(clamp(a, 0.0, 1.0) * 32767.0);
    uint ib = uint(clamp(b, 0.0, 1.0) * 32767.0);
    return uintBitsToFloat((ia << 15) | ib | 0x40000000u);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    // 1. Read Atomic Mass
    uint m_uint = imageLoad(img_mass_accum, uv_i).r;
    imageStore(img_mass_accum, uv_i, uvec4(0));
    float mass = float(m_uint) / MASS_SCALE;
    
    // 2. Read preserved Velocity (spectral genes will come from winner)
    vec4 prevData = imageLoad(img_new_state, uv_i);
    vec2 velocity = prevData.gb;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    // 3. Genome + Spectral Gene Update (Winner-Takes-All)
    uint packed = imageLoad(img_winner_tracker, uv_i).r;
    imageStore(img_winner_tracker, uv_i, uvec4(0)); // Clear for next frame
    
    vec4 finalGenome = texture(tex_old_genome, uv); // Preserve by default
    vec4 winnerState = texture(tex_old_state, uv);  // For spectral genes
    float packedSpectral = winnerState.a;           // Default: keep local spectral genes
    
    if (packed != 0) {
        // Unpack Lower 20 bits for Scrambled Index
        uint scrambled_idx = packed & 0xFFFFF; 
        
        // Descramble using same mask as flow shader (u_seed)
        uint seed_bits = floatBitsToUint(p.u_seed);
        uint frame_mask = pcg_hash_1d(seed_bits) & 0xFFFFFFu;
        uint winner_idx = scrambled_idx ^ frame_mask;
        
        ivec2 res = ivec2(p.u_res);
        ivec2 winner_coords = ivec2(winner_idx % res.x, winner_idx / res.x);
        vec2 winner_uv = (vec2(winner_coords) + 0.5) / p.u_res;
        
        finalGenome = texture(tex_old_genome, winner_uv);
        vec4 winnerStateData = texture(tex_old_state, winner_uv);
        packedSpectral = winnerStateData.a;  // Copy spectral genes from winner
    }
    
    imageStore(img_new_genome, uv_i, finalGenome);
    
    // Read genome for dynamic secretion calculation
    vec2 growth_genes = unpack2(finalGenome.a);
    vec2 width_radius = unpack2(finalGenome.b);
    float growth_mu = growth_genes.x;
    float kernel_width = width_radius.x;
    
    // Unpack spectral genes for emission
    vec2 spectral_genes = unpack2(packedSpectral);
    float g_emission_hue = spectral_genes.x;
    
    // Dynamic secretion: high μ species + wide kernels secrete more
    float secretion_base = 0.3 + growth_mu * 0.7;     // [0.3, 1.0]
    float secretion_diffusion = 0.5 + kernel_width;   // [0.5, 1.5]
    float raw_secretion = clamp((secretion_base * secretion_diffusion) / 1.5, 0.0, 1.0);
    
    // Map to UI range
    float g_secretion = mix(p.u_range_secretion.x, p.u_range_secretion.y, raw_secretion);
    vec2 sec_per = unpack2(finalGenome.a); // Actually... wait.
    
    // We need to verify where secretion is packed.
    // In compute_init:
    // R=(b1, b2), G=(b3, a2), B=(width, radius), A=(mu, sigma)
    // Wait... 8 genes.
    // Texture is RGBA32F -> 4 channels.
    // 2 genes per channel (unpack2).
    // R -> b1, b2
    // G -> b3, a2
    // B -> width, radius
    // A -> mu, sigma
    
    // WHERE are secretion and perception?
    // They are NOT in the 8-gene genome texture currently!
    // The genome texture only holds 8 core genes.
    // compute_init.glsl lines 67-85 define 8 genes.
    
    // If they are not packed, we cannot read them!
    // We must stick to the hardcoded dependency for now OR repack the genome.
    // The user wants to use the sliders/histograms.
    
    // CHECK compute_init.glsl Packing!
    
    // 4. Finalization
    float finalMass = mass;
    
    // 5. Spectral Secretion (HueToRGB emission)
    if (finalMass > 0.05) {
        // Convert emission hue to RGB color
        vec3 emittedColor = HueToRGB(g_emission_hue);
        
        vec4 currentSignal = imageLoad(img_new_signal, uv_i);
        
        // Emit spectral signal: Amount = mass * secretion * color
        vec3 addedSignal = emittedColor * (finalMass * g_secretion * p.u_dt * 0.5);
        
        // Add to existing signal and clamp
        vec3 nextSignal = currentSignal.rgb + addedSignal;
        nextSignal = clamp(nextSignal, 0.0, 2.0);
        
        imageStore(img_new_signal, uv_i, vec4(nextSignal, 0.0));
    }
    
    // Store final state with spectral genes preserved in .a
    imageStore(img_new_state, uv_i, vec4(finalMass, velocity, packedSpectral));
}

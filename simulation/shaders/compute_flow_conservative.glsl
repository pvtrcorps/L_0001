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
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_growth_rate;
    // Block C: Social / Motor
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    // Block D: Senses
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;      // Genes 1-8
layout(set = 0, binding = 3) uniform sampler2D tex_potential;
layout(set = 0, binding = 4, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 6, rgba32f) uniform image2D img_new_genome; // Unused for advection?
layout(set = 0, binding = 7) uniform sampler2D tex_signal;
layout(set = 0, binding = 8, r32ui) uniform uimage2D img_winner_tracker;
layout(set = 0, binding = 9) uniform sampler2D tex_genome_ext;  // Genes 9-16 [NEW]

// PCG Hash (2d -> 1d)
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
    return float(pcg_hash(p)) * (1.0/4294967296.0);
}

// 1D PCG Hash
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

const float MASS_SCALE = 100000.0; 

// Stochastic Rounding
uint get_rounded_amount(float amount, vec2 seed) {
    uint i = uint(amount);
    float f = fract(amount);
    float h = hash(seed + vec2(amount, amount * 0.123));
    if (h < f) i += 1u;
    return i;
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


void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    vec4 state = texture(tex_state, uv);
    float myMass = state.r;
    
    if (myMass < 0.0001) {
        imageStore(img_new_state, uv_i, vec4(0.0));
        return; 
    }

    // === 1. Unpack Genes ===
    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);
    
    // Physiology
    vec2 rad_visc = unpack2(g1.g);
    float g_viscosity = rad_visc.y; // [0-1] Inertia/Drag
    
    // Social / Motor
    vec2 aff_rep = unpack2(g2.r);
    float g_affinity = aff_rep.x;   // [0-1] Cohesion
    float g_repulsion = aff_rep.y;  // [0-1] Personal Space
    
    vec2 tol_mob = unpack2(g2.g);
    float g_density_tol = tol_mob.x;// [0-1] Pressure Resistance
    float g_mobility = tol_mob.y;   // [0-1] Speed Multiplier
    
    // Senses
    vec2 sig_gain = unpack2(g2.b);
    float g_sensitivity = sig_gain.y; // [0-1] Signal Gain
    
    vec2 hues = unpack2(g2.a);
    float g_detection_hue = hues.y; // [0-1] Target Signal
    
    // === 2. Calculate Forces ===
    
    // A. Mass Potential Gradient (Attraction / Growth Direction)
    // Sobel Filter for dU/dx, dU/dy
    vec2 pixel_size = 1.0 / p.u_res;
    float gx = 0.0;
    gx += -1.0 * texture(tex_potential, uv + vec2(-1, -1)*pixel_size).r;
    gx += -2.0 * texture(tex_potential, uv + vec2(-1,  0)*pixel_size).r;
    gx += -1.0 * texture(tex_potential, uv + vec2(-1,  1)*pixel_size).r;
    gx +=  1.0 * texture(tex_potential, uv + vec2( 1, -1)*pixel_size).r;
    gx +=  2.0 * texture(tex_potential, uv + vec2( 1,  0)*pixel_size).r;
    gx +=  1.0 * texture(tex_potential, uv + vec2( 1,  1)*pixel_size).r;
    
    float gy = 0.0;
    gy += -1.0 * texture(tex_potential, uv + vec2(-1, -1)*pixel_size).r;
    gy += -2.0 * texture(tex_potential, uv + vec2( 0, -1)*pixel_size).r;
    gy += -1.0 * texture(tex_potential, uv + vec2( 1, -1)*pixel_size).r;
    gy +=  1.0 * texture(tex_potential, uv + vec2(-1,  1)*pixel_size).r;
    gy +=  2.0 * texture(tex_potential, uv + vec2( 0,  1)*pixel_size).r;
    gy +=  1.0 * texture(tex_potential, uv + vec2( 1,  1)*pixel_size).r;
    
    vec2 gradU = vec2(gx, gy);
    
    // APPLY AFFINITY (Cohesion)
    // High affinity = Follows potential gradient strongly (Clumps)
    // Low affinity = Drifts more (Cloud)
    // Base affinity 1.0, scalable up to 3.0 via gene
    gradU *= (0.5 + g_affinity * 2.5);
    
    // B. Signal Gradient (Chemotaxis)
    // Compute gradient of spectral similarity (U_signal)
    ivec2 res_i = ivec2(p.u_res);
    ivec2 l_uv = (uv_i + ivec2(-1, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 r_uv = (uv_i + ivec2(1, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 u_uv = (uv_i + ivec2(0, -1) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 d_uv = (uv_i + ivec2(0, 1) + ivec2(p.u_res)) % ivec2(p.u_res);
    
    // We must re-compute signal matching here or assume tex_potential.a has it.
    // tex_potential.a stores U_signal computed in Step 1.
    // U_signal = dot(signal, myDetector).
    // This is correct because compute_convolution updated U_signal based on *my* genome at that pixel.
    float sL = texelFetch(tex_potential, l_uv, 0).a;
    float sR = texelFetch(tex_potential, r_uv, 0).a;
    float sU = texelFetch(tex_potential, u_uv, 0).a;
    float sD = texelFetch(tex_potential, d_uv, 0).a;
    vec2 gradSignal = vec2(sR - sL, sD - sU);
    
    // APPLY SENSITIVITY
    // Combine Growth Gradient + Signal Gradient
    // Signal Advect controls global weight, Sensitivity controls per-species gain
    vec2 totalAttraction = gradU + gradSignal * p.u_signal_advect * (g_sensitivity * 3.0);
    
    // C. Density Gradient (Repulsion)
    // High density pressure
    float mR = texelFetch(tex_state, r_uv, 0).r;
    float mL = texelFetch(tex_state, l_uv, 0).r;
    float mD = texelFetch(tex_state, d_uv, 0).r;
    float mU = texelFetch(tex_state, u_uv, 0).r;
    vec2 gradA = vec2(mR - mL, mD - mU);
    
    // APPLY REPULSION GENE
    gradA *= (0.5 + g_repulsion * 2.5);
    
    // D. Compute Velocity Field
    // alpha = (mass / theta)^n
    // APPLY DENSITY TOLERANCE: Modulates theta_A (Critical Mass)
    // High tolerance = High Theta = Low Alpha (Less repulsion)
    float local_theta = p.u_theta_A * (0.5 + g_density_tol * 2.0);
    float alpha = pow(max(myMass, 0.0) / max(local_theta, 0.001), p.u_alpha_n);
    
    // V = Speed * (Attraction - Alpha * Repulsion)
    // APPLY MOBILITY
    float speed_mult = p.u_flow_speed * (0.2 + g_mobility * 1.8);
    vec2 vel = speed_mult * (totalAttraction - alpha * gradA);
    
    // APPLY VISCOSITY (Drag)
    // V_final = V * (1 - viscosity)
    vel *= clamp(1.0 - g_viscosity * 0.9, 0.1, 1.0);
    
    
    // === 3. Mass Advection (Scatter) ===
    // Distribution to 4 neighbors
    vec2 pos_next = uv * p.u_res + vel * p.u_dt;
    
    // Wrap coordinates
    pos_next = mod(pos_next, p.u_res);
    
    // Bilinear Scatter
    vec2 start_cell_f = floor(pos_next - 0.5);
    ivec2 start_cell = ivec2(start_cell_f);
    vec2 f = pos_next - 0.5 - start_cell_f; // Fractional part
    
    // Distribute mass
    // 00 10
    // 01 11
    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y);
    float w01 = (1.0 - f.x) * f.y;
    float w11 = f.x * f.y;
    
    ivec2 c00 = (start_cell + ivec2(0, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 c10 = (start_cell + ivec2(1, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 c01 = (start_cell + ivec2(0, 1) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 c11 = (start_cell + ivec2(1, 1) + ivec2(p.u_res)) % ivec2(p.u_res);
    
    // Atomic accumulation (using robust int mapping)
    uint amount = uint(myMass * MASS_SCALE);
    if (amount > 0) {
        imageAtomicAdd(img_mass_accum, c00, uint(float(amount) * w00));
        imageAtomicAdd(img_mass_accum, c10, uint(float(amount) * w10));
        imageAtomicAdd(img_mass_accum, c01, uint(float(amount) * w01));
        imageAtomicAdd(img_mass_accum, c11, uint(float(amount) * w11));
        
        // WINNER TRACKING (For Genome Inheritance)
        // We pack (Mass Contribution << 20) | (Source Index)
        // Mass Contribution uses 12 bits (Max 4096 -> ~40.0 mass units)
        // Source Index uses 20 bits (Max 1M -> 1024x1024 resolution)
        
        uint src_idx = uint(uv_i.y) * uint(p.u_res.x) + uint(uv_i.x);
        // Ensure index fits in 20 bits
        src_idx = src_idx & 0xFFFFFu; 
        
        // Calculate contribution for each neighbor (scaled by 100 for precision)
        // Max mass 40.0 * 100 = 4000 < 4096 (12 bits)
        uint m00 = uint(clamp(myMass * w00 * 100.0, 0.0, 40.0));
        uint m10 = uint(clamp(myMass * w10 * 100.0, 0.0, 40.0));
        uint m01 = uint(clamp(myMass * w01 * 100.0, 0.0, 40.0));
        uint m11 = uint(clamp(myMass * w11 * 100.0, 0.0, 40.0));
        
        if (m00 > 0) imageAtomicMax(img_winner_tracker, c00, (m00 << 20u) | src_idx);
        if (m10 > 0) imageAtomicMax(img_winner_tracker, c10, (m10 << 20u) | src_idx);
        if (m01 > 0) imageAtomicMax(img_winner_tracker, c01, (m01 << 20u) | src_idx);
        if (m11 > 0) imageAtomicMax(img_winner_tracker, c11, (m11 << 20u) | src_idx);
    }
    
    // Store calculated velocity (source, instantaneous) into the G/B channels of the destination state
    // This allows compute_normalize to pick it up and preserve it for the next frame's signal advection.
    // We store it in .gb to match the standard format (Mass, VelX, VelY, Aux)
    imageStore(img_new_state, uv_i, vec4(0.0, vel.x, vel.y, 0.0));
}

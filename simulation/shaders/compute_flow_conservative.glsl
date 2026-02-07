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
    float u_fluid_momentum; // Was u_pad_col
    
    // 1. Gene Ranges (16 Genes * 2) = 32 floats
    // Block A: Physiology
    vec2 r_mu; vec2 r_sigma; vec2 r_radius; vec2 r_viscosity;
    // Block B: Morphology
    vec2 r_shape_a; vec2 r_shape_b; vec2 r_shape_c; vec2 r_inertia;
    // Block C: Social / Motor
    vec2 r_affinity; vec2 r_repulsion; vec2 r_density_tol; vec2 r_mobility;
    // Block D: Senses
    vec2 r_secretion; vec2 r_sensitivity; vec2 r_emission_hue; vec2 r_detection_hue;
    
    // Wind / Atmosphere
    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;
    
    // Signal Extras
    float u_signal_force_strength;
    float u_signal_emission_strength;
    float u_pad1;
    float u_pad2;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;      // Genes 1-8
layout(set = 0, binding = 3) uniform sampler2D tex_potential;
layout(set = 0, binding = 4, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_state;
// binding 6 was img_new_genome, removed as unused
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

const float MASS_SCALE = 100000000.0; // 1e8 for High Precision

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
    
    // 1. Log Mass Term
    // amt is up to 1e8. log(1e8) ~ 18.4
    float log_mass = log(float(amt)); 
    
    // 2. Potential Term
    // pot is [-1, 1], beta is [0, 5]
    float pot_term = pot * beta;
    
    // 3. Gumbel Noise
    // u ~ Uniform(0,1)
    float u = hash(seed_uv);
    // Avoid log(0)
    u = clamp(u, 0.0000001, 0.9999999);
    float gumbel = -log(-log(u));
    
    // 4. Total Score
    // Range approx: [0, 18] + [-5, 5] + [-2, 10] = [-7, 33]
    // We map this to [1, 255] for the 8-bit storage.
    float score_float = log_mass + pot_term + gumbel;
    
    // Shift (+10) and Scale (*4) to fit typical range into 0-255
    float map_s = (score_float + 10.0) * 4.0;
    
    return uint(clamp(map_s, 1.0, 255.0));
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
    // A. Mass Potential Gradient (Attraction / Growth Direction)
    // Sobel Filter for dU/dx, dU/dy (Canonical)
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
    gradU *= (0.5 + g_affinity * 2.5);
    
    // B. Signal Gradient (Chemotaxis)
    // Compute gradient of spectral similarity (U_signal)
    ivec2 res_i = ivec2(p.u_res);
    ivec2 l_uv = (uv_i + ivec2(-1, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 r_uv = (uv_i + ivec2(1, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 u_uv = (uv_i + ivec2(0, -1) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 d_uv = (uv_i + ivec2(0, 1) + ivec2(p.u_res)) % ivec2(p.u_res);
    
    float sL = texelFetch(tex_potential, l_uv, 0).a;
    float sR = texelFetch(tex_potential, r_uv, 0).a;
    float sU = texelFetch(tex_potential, u_uv, 0).a;
    float sD = texelFetch(tex_potential, d_uv, 0).a;
    vec2 gradSignal = vec2(sR - sL, sD - sU);
    
    vec2 totalAttraction = gradU + gradSignal * p.u_signal_advect * (g_sensitivity * p.u_signal_force_strength);
    
    // C. Density Gradient (Repulsion)
    // High density pressure
    float mR = texelFetch(tex_state, r_uv, 0).r;
    float mL = texelFetch(tex_state, l_uv, 0).r;
    float mD = texelFetch(tex_state, d_uv, 0).r;
    float mU = texelFetch(tex_state, u_uv, 0).r;
    vec2 gradA = vec2(mR - mL, mD - mU);
    
    gradA *= (0.5 + g_repulsion * 2.5);
    
    // D. Compute Velocity Field
    vec2 shape_c_inertia = unpack2(g1.a);
    float g_inertia = shape_c_inertia.y;

    // D. Compute Acceleration
    float local_theta = p.u_theta_A * (0.5 + g_density_tol * 2.0);
    float alpha = pow(max(myMass, 0.0) / max(local_theta, 0.001), p.u_alpha_n);
    
    float force_mult = p.u_flow_speed * (0.2 + g_mobility * 1.8);
    vec2 force = force_mult * (totalAttraction - alpha * gradA);
    
    // === INERTIAL INTEGRATION ===
    vec2 old_vel = state.gb;
    
    // FLUID MOMENTUM CONTROL
    // If u_fluid_momentum < 1.0, we dampen the old velocity.
    // If 0.0, it becomes Aristotelian (velocity = force * dt), no inertia.
    old_vel *= p.u_fluid_momentum;
    
    float responsiveness = mix(1.0, 0.05, g_inertia);
    vec2 integrated_vel = mix(old_vel, old_vel + force * p.u_dt, responsiveness);
    integrated_vel *= clamp(1.0 - g_viscosity * p.u_dt * 2.0, 0.0, 1.0);
    
    vec2 vel = integrated_vel;
    
    
    // === 3. Mass Advection (Scatter) ===
    vec2 pos_next = uv * p.u_res + vel * p.u_dt;
    pos_next = mod(pos_next, p.u_res);
    
    vec2 start_cell_f = floor(pos_next - 0.5);
    ivec2 start_cell = ivec2(start_cell_f);
    vec2 f = pos_next - 0.5 - start_cell_f;
    
    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y);
    float w01 = (1.0 - f.x) * f.y;
    float w11 = f.x * f.y;
    
    ivec2 c00 = (start_cell + ivec2(0, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 c10 = (start_cell + ivec2(1, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 c01 = (start_cell + ivec2(0, 1) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 c11 = (start_cell + ivec2(1, 1) + ivec2(p.u_res)) % ivec2(p.u_res);
    
    // Mass Accumulation (High Precision)
    uint total_amount = uint(myMass * MASS_SCALE);
    
    if (total_amount > 0) {
        // SAFE PARTITIONING (Cascade Remainder)
        // Prevents underflow where a00+a10+a01 > total (due to float rounding up)
        // forcing a11 to wrap around to uint_max (Mass Explosion).
        
        uint remaining = total_amount;
        
        uint a00 = uint(float(total_amount) * w00);
        if (a00 > remaining) a00 = remaining;
        remaining -= a00;
        
        uint a10 = uint(float(total_amount) * w10);
        if (a10 > remaining) a10 = remaining;
        remaining -= a10;
        
        uint a01 = uint(float(total_amount) * w01);
        if (a01 > remaining) a01 = remaining;
        remaining -= a01;
        
        uint a11 = remaining; // The rest goes here
        
        if (a00 > 0) imageAtomicAdd(img_mass_accum, c00, a00);
        if (a10 > 0) imageAtomicAdd(img_mass_accum, c10, a10);
        if (a01 > 0) imageAtomicAdd(img_mass_accum, c01, a01);
        if (a11 > 0) imageAtomicAdd(img_mass_accum, c11, a11);
        
        // WINNER TRACKING
        uint src_idx = uint(uv_i.y) * uint(p.u_res.x) + uint(uv_i.x);
        src_idx = src_idx & 0xFFFFFFu; 
        
        // Score based on ACTUAL sent mass amount (a00 etc)
        // Using 'total_amount' as denominator to normalize would suffice, but raw amount is fine.
        // We multiply by constant to map to 0-255 roughly (max mass for 1 pixel is ~1-5?)
        // Actually, just checking > 0 is enough to verify existence.
        // But for competition, we want Larger Mass > Smaller Mass.
        // a00 is typically 1e7 or 1e8 range.
        // We can just use high bits or log score.
        // Simple linear map: if a00 is 100% of mass, score is 255.
        // But 'total_amount' varies.
        // Let's stick to the previous robust logic:
        
        #define CALC_SCORE_SAFE(amt, tot) ( (amt > 0) ? max(1u, uint( (float(amt)/max(float(tot),1.0)) * 255.0 )) : 0u )
        
        // Revised Score: Just use the raw amount scaled? No, we need 0-255? 
        // Actually img_winner_tracker seems to support High Bits for score?
        // compute_normalize reads: uint packed = imageLoad...
        // Format R32UI.
        // Packed = (Score << 24) | Index.
        // Score is 8 bits (255).
        
        // Proper normalized score: fraction of MY mass sent.
        // NO! We want Absolute Mass comparison.
        // A giant blob contributing 10% should beat a tiny speck contributing 100%.
        // Score = min(255, amount / (SCALE/255) )
        // MASS_SCALE = 1e8.
        // If amount = 1e6 (0.01 mass), Score = 1?
        // Let's scale so 1.0 mass = 255 score.
        // score = amount / (1e8 / 255) = amount * 2.55e-6
        
        // NEGOTIATION RULE (Gumbel-Max)
        // Score = log(Mass) + Beta * Potential + GumbelNoise
        // Mass is 'amount' (uint). Potential is 'source_pot' [-1, 1].
        
        // 1. Get Source Potential (Growth Affinity)
        float source_pot = texture(tex_potential, uv).r; 
        
        // 2. Pre-calc random seed base for this pixel/frame
        // We use the pixel index and the global seed to get a unique hash base
        vec2 noise_base_uv = uv + vec2(p.u_seed, p.u_seed * 0.1);

        #define CALC_SCORE_GUMBEL(amt, offset_idx) \
            ( (amt > 0) ? calculate_gumbel_score(amt, source_pot, p.u_beta, noise_base_uv + vec2(float(offset_idx)*0.01)) : 0u )
            
        uint s00 = CALC_SCORE_GUMBEL(a00, 0);
        uint s10 = CALC_SCORE_GUMBEL(a10, 1);
        uint s01 = CALC_SCORE_GUMBEL(a01, 2);
        uint s11 = CALC_SCORE_GUMBEL(a11, 3);

        if (s00 > 0) imageAtomicMax(img_winner_tracker, c00, (s00 << 24u) | src_idx);
        if (s10 > 0) imageAtomicMax(img_winner_tracker, c10, (s10 << 24u) | src_idx);
        if (s01 > 0) imageAtomicMax(img_winner_tracker, c01, (s01 << 24u) | src_idx);
        if (s11 > 0) imageAtomicMax(img_winner_tracker, c11, (s11 << 24u) | src_idx);
    }
    
    // Store calculated velocity (source, instantaneous) into the G/B channels of the destination state
    // This allows compute_normalize to pick it up and preserve it for the next frame's signal advection.
    // We store it in .gb to match the standard format (Mass, VelX, VelY, Aux)
    imageStore(img_new_state, uv_i, vec4(0.0, vel.x, vel.y, 0.0));
}

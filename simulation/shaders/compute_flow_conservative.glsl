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
    
    // Wind / Atmosphere
    float u_time;
    float u_wind_scale;
    float u_wind_strength;
    float u_wind_speed;
    
    // Signal Extras
    float u_signal_force_strength;
    float u_signal_emission_strength;
    float u_interaction_beta; 
    float u_genetic_barrier;  
    float u_colonize_thr;
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
    
    if (myMass <= 0.0) {
        imageStore(img_new_state, uv_i, vec4(0.0));
        return; 
    }

    // === 1. Unpack Genes ===
    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);
    
    // VOID CHECK: Mass with no genome is "dead matter"
    // It still advects (mass conservation) but cannot colonize (no identity propagation)
    bool is_void = dot(g1, g1) < 0.0001;
    
    // Physiology
    vec2 rad_visc = unpack2(g1.g);
    float g_viscosity = rad_visc.y; // [0-1] Inertia/Drag
    
    // Social / Motor
    vec2 aff_rep = unpack2(g2.r);  // .x=affinity, .y=repulsion (reserved for future)
    
    vec2 tol_mob = unpack2(g2.g);
    float g_mobility = tol_mob.y;   // [0-1] Speed Multiplier
    
    // Senses
    vec2 sig_gain = unpack2(g2.b);
    float g_sensitivity = sig_gain.y; // [0-1] Signal Gain
    
    vec2 hues = unpack2(g2.a);
    float g_emission_hue = hues.x;  // [0-1] My Color / Emission Hue
    float g_detection_hue = hues.y; // [0-1] Target Signal
    
    // === 2. Calculate Forces ===
    
    // A. Growth Potential Gradient (Attraction toward kin)
    // Uses tex_potential: R = U_growth, G = U_density, B = U_interact
    vec2 pixel_size = 1.0 / p.u_res;
    
    // Cache 3x3 potential samples (avoids redundant texture fetches)
    vec4 pot_TL = texture(tex_potential, uv + vec2(-1, -1)*pixel_size);
    vec4 pot_TC = texture(tex_potential, uv + vec2( 0, -1)*pixel_size);
    vec4 pot_TR = texture(tex_potential, uv + vec2( 1, -1)*pixel_size);
    vec4 pot_ML = texture(tex_potential, uv + vec2(-1,  0)*pixel_size);
    vec4 pot_MR = texture(tex_potential, uv + vec2( 1,  0)*pixel_size);
    vec4 pot_BL = texture(tex_potential, uv + vec2(-1,  1)*pixel_size);
    vec4 pot_BC = texture(tex_potential, uv + vec2( 0,  1)*pixel_size);
    vec4 pot_BR = texture(tex_potential, uv + vec2( 1,  1)*pixel_size);
    
    // A. Combined Growth + Interaction Potential Gradient
    // ARCHITECTURAL DECISION: Interaction is merged into growth potential
    // so it's naturally balanced by alpha repulsion.
    // combined = U_growth + interaction_beta * U_interact
    float beta = p.u_interaction_beta;
    float cp_TL = pot_TL.r + beta * pot_TL.b;
    float cp_TC = pot_TC.r + beta * pot_TC.b;
    float cp_TR = pot_TR.r + beta * pot_TR.b;
    float cp_ML = pot_ML.r + beta * pot_ML.b;
    float cp_MR = pot_MR.r + beta * pot_MR.b;
    float cp_BL = pot_BL.r + beta * pot_BL.b;
    float cp_BC = pot_BC.r + beta * pot_BC.b;
    float cp_BR = pot_BR.r + beta * pot_BR.b;
    
    // Sobel on combined potential
    float gx = -1.0*cp_TL + -2.0*cp_ML + -1.0*cp_BL
              + 1.0*cp_TR +  2.0*cp_MR +  1.0*cp_BR;
    float gy = -1.0*cp_TL + -2.0*cp_TC + -1.0*cp_TR
              + 1.0*cp_BL +  2.0*cp_BC +  1.0*cp_BR;
    vec2 gradGrowth = vec2(gx, gy);
    
    // B. Density Gradient (reuse cached samples, G channel)
    float dx = -1.0*pot_TL.g + -2.0*pot_ML.g + -1.0*pot_BL.g
              + 1.0*pot_TR.g +  2.0*pot_MR.g +  1.0*pot_BR.g;
    float dy = -1.0*pot_TL.g + -2.0*pot_TC.g + -1.0*pot_TR.g
              + 1.0*pot_BL.g +  2.0*pot_BC.g +  1.0*pot_BR.g;
    vec2 gradDensity = vec2(dx, dy);
    
    // C. Signal Gradient (Chemotaxis) — Sobel filtered
    // Uses same hue interaction as convolution: cos(2π · circular_hue_distance)
    // This makes gradSignal consistent with gradInteract's selectivity.
    
    // Helper: Convert signal RGB back to dominant hue and compute scalar potential
    // For each neighbor: compute how much we "like" the signal at that location
    // using the SAME function as get_interaction_strength in convolution
    
    // Sample 3x3 neighborhood and compute scalar signal potential
    float sp_TL = 0.0, sp_TC = 0.0, sp_TR = 0.0;
    float sp_ML = 0.0, sp_MC = 0.0, sp_MR = 0.0;
    float sp_BL = 0.0, sp_BC = 0.0, sp_BR = 0.0;
    {
        // We project signal RGB onto our detector RGB and use that as potential
        // This is equivalent to cos-based hue matching when signals are pure hues
        vec3 myDetector = HueToRGB(g_detection_hue);
        
        sp_TL = dot(texture(tex_signal, uv + vec2(-1, -1)*pixel_size).rgb, myDetector);
        sp_TC = dot(texture(tex_signal, uv + vec2( 0, -1)*pixel_size).rgb, myDetector);
        sp_TR = dot(texture(tex_signal, uv + vec2( 1, -1)*pixel_size).rgb, myDetector);
        sp_ML = dot(texture(tex_signal, uv + vec2(-1,  0)*pixel_size).rgb, myDetector);
        sp_MC = dot(texture(tex_signal, uv                           ).rgb, myDetector);
        sp_MR = dot(texture(tex_signal, uv + vec2( 1,  0)*pixel_size).rgb, myDetector);
        sp_BL = dot(texture(tex_signal, uv + vec2(-1,  1)*pixel_size).rgb, myDetector);
        sp_BC = dot(texture(tex_signal, uv + vec2( 0,  1)*pixel_size).rgb, myDetector);
        sp_BR = dot(texture(tex_signal, uv + vec2( 1,  1)*pixel_size).rgb, myDetector);
    }
    
    // Sobel filter on scalar signal potential
    float sgx = -1.0*sp_TL + -2.0*sp_ML + -1.0*sp_BL
               + 1.0*sp_TR +  2.0*sp_MR +  1.0*sp_BR;
    float sgy = -1.0*sp_TL + -2.0*sp_TC + -1.0*sp_TR
               + 1.0*sp_BL +  2.0*sp_BC +  1.0*sp_BR;
    
    vec2 gradSignal = vec2(sgx, sgy);
    
    // === TOTAL ATTRACTION ===
    // gradGrowth: includes growth + interaction potential (merged, alpha-balanced)
    // gradSignal: long-range chemotaxis toward preferred signals

    
    // === CANONICAL ALPHA ===
    // Reference: alpha = clip((A/2)², 0, 1)
    // When mass is low → alpha ≈ 0 → creature follows growth gradient (exploration)
    // When mass is high → alpha ≈ 1 → creature pushed away by density gradient (pressure)
    float alpha = clamp((myMass / 2.0) * (myMass / 2.0), 0.0, 1.0);
    
    // === CANONICAL FORCE: F = (1-α)·∇U - α·∇A ===
    vec2 totalAttraction = gradGrowth 
                         + gradSignal * (g_sensitivity * p.u_signal_force_strength);
    
    // CANONICAL REPULSION: Sobel-filtered total density field (diffuse, medium-range)
    vec2 totalRepulsion = gradDensity;
    
    vec2 flow_field = (1.0 - alpha) * totalAttraction - alpha * totalRepulsion;
    
    float force_mult = p.u_flow_speed * (0.2 + g_mobility * 1.8);
    vec2 target_vel = force_mult * flow_field;
    
    // SAFETY 2: Clamp Target Velocity (CFL Condition)
    // Prevent moving more than ~1.0 pixel per frame effectively
    float tv_len = length(target_vel);
    float max_v = 2.0; // Max pixels/sec roughly, let's say 2.0 relative speed
    if (tv_len > max_v) target_vel = (target_vel / tv_len) * max_v;
    
    // === ARISTOTELIAN INTEGRATION ===
    // In Low Reynolds numbers, Force ~ Velocity (not Acceleration).
    // We blend from current velocity to target velocity based on 'Fluid Momentum' (Inertia).
    // fluid_momentum 0.0 = Instant Change (Aristotelian/Massless)
    // fluid_momentum 1.0 = High Inertia (Newtonian-ishish)
    
    vec2 old_vel = state.gb;
    
    // Smooth transition factor
    // If momentum is 0, factor is high (fast response).
    // We multiply by dt to ensure frame-rate independence.
    float responsiveness = (1.0 - p.u_fluid_momentum) * 10.0;
    vec2 vel = mix(old_vel, target_vel, clamp(responsiveness * p.u_dt, 0.05, 1.0));
    
    // Friction / Viscosity
    vel *= clamp(1.0 - g_viscosity * p.u_dt * 2.0, 0.0, 1.0);
    
    
    // === 3. Mass Advection (Smooth Splatting) ===
    // Flow Lenia Reference: 
    // Target pos = pos + dt * vel
    // Distribute to neighbors within range [pos - 0.5 - sigma, pos + 0.5 + sigma]
    // Weight = max(0, 0.5 - |dist| + sigma)
    // Here we implement a 3x3 kernel which covers sigma up to ~0.5-0.8 efficiently.
    
    // IMPORTANT: uv * p.u_res is centered on 0.5 (e.g., 0.5, 1.5, etc.)
    // But our splatting logic uses integer offsets (dx, dy).
    // We shift by -0.5 so that 0.5 becomes 0.0, aligning the center of mass with integer index 0.
    vec2 pos_next = uv * p.u_res + vel * p.u_dt - 0.5;
    
    // Handle Boundary Checks (Wrap)
    // GLSL mod can return negative for negative inputs, so we do standard positive mod
    pos_next = mod(pos_next, p.u_res);
    if (pos_next.x < 0.0) pos_next.x += p.u_res.x;
    if (pos_next.y < 0.0) pos_next.y += p.u_res.y;
    
    // Nearest integer coordinate (base index of distribution)
    vec2 center_f = floor(pos_next + 0.5);
    ivec2 center_i = ivec2(center_f);
    
    // Relative position from center [-0.5, 0.5]
    vec2 delta = pos_next - center_f;
    
    // Sigma (Spread) from Temperature parameter
    // Reference default is 0.65.
    float sigma = max(p.u_temperature, 0.1); 
    
    // Pre-calculate normalization factor
    // The sum of weights for a Gaussian-like splat isn't strictly 1.0 automatically,
    // but the reference implementation divides by total area.
    // Lenia Reference: area = prod(clip(0.5 - |dx| + sigma, 0, 1)) / (4 * sigma^2) ??
    // Let's stick to the weight formula: w = calc_weight(dx) * calc_weight(dy)
    // And normalize explicitly to conserve mass.
    
    float total_weight = 0.0;
    float weights[9];
    ivec2 offsets[9];
    
    int idx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            // Distance from exact float position `pos_next` to neighbor center `center_f + vec2(dx, dy)`
            // pos_next = center_f + delta
            // neighbor = center_f + vec2(dx, dy)
            // dist = |delta - vec2(dx, dy)|
            
            vec2 dist_vec = abs(delta - vec2(float(dx), float(dy)));
            
            // Reference Weight Function: sz = 0.5 - dist + sigma
            vec2 sz = 0.5 - dist_vec + sigma;
            
            // Per-axis weight = clip(sz, 0, min(1, 2*sigma))
            // We use simple max(0, ...) as per standard splatting approximation
            vec2 w_axis = clamp(sz, 0.0, 1.0);
            
            float w = w_axis.x * w_axis.y;
            
            weights[idx] = w;
            offsets[idx] = ivec2(dx, dy);
            total_weight += w;
            idx++;
        }
    }
    
    // Normalize weights to ensure Conservation of Mass
    if (total_weight < 0.0001) total_weight = 1.0; // Prevent div/0
    float norm_factor = 1.0 / total_weight;
    
    
    // Mass Accumulation (High Precision)
    uint total_amount = uint(round(myMass * MASS_SCALE));
    uint kept_mass = 0u; // Mass blocked by barrier
    
    if (total_amount > 0) {
        
        uint remaining = total_amount;
        int target_remainder_idx = 4; // Center by default
        float max_w = -1.0;
        
        // Find highest weight neighbor (prioritize center if weights are equal)
        for (int i = 0; i < 9; i++) {
            float w = (i == 4) ? weights[i] + 0.0001 : weights[i];
            if (w > max_w) {
                max_w = w;
                target_remainder_idx = i;
            }
        }
        
        // Distribute to 9 neighbors
        for (int i = 0; i < 9; i++) {
            if (i == target_remainder_idx) continue;
            
            float w = weights[i] * norm_factor;
            if (w < 0.001) continue; 
            
            uint amount = uint(floor(float(total_amount) * w)); 
            if (amount > remaining) amount = remaining;
            remaining -= amount;
            
            if (amount == 0u) continue;
            
            ivec2 target_uv = (center_i + offsets[i] + ivec2(p.u_res)) % ivec2(p.u_res);
            
            if (true) { // Barrier is now handled by Force Fields in the flow step
                imageAtomicAdd(img_mass_accum, target_uv, amount);
                
                float source_pot = texture(tex_potential, uv).r; 
                uint src_idx = (uint(uv_i.y) * uint(p.u_res.x) + uint(uv_i.x));
                
                vec2 noise_uv = uv + vec2(float(i)*0.1, p.u_seed); 
                uint score_8bit = calculate_gumbel_score(amount, source_pot, p.u_beta, noise_uv);
                
                // Unbiased Tie-breaker: Spatial hash of the source index
                uint jitter_2bit = pcg_hash_1d(src_idx) & 0x3u;
                
                uint packed_comp = (score_8bit << 24u) | (jitter_2bit << 22u) | (src_idx & 0x3FFFFFu);
                
                uint thr_uint = uint(p.u_colonize_thr * MASS_SCALE);
                if (score_8bit > 0u && amount > thr_uint && !is_void) {
                    imageAtomicMax(img_winner_tracker, target_uv, packed_comp);
                }
            }
        }

        // Any residual quantization error remains in place to preserve strict conservation.
        // We assign it to the target_remainder_idx instead of the last valid index.
        if (remaining > 0u) {
            ivec2 remainder_uv = (center_i + offsets[target_remainder_idx] + ivec2(p.u_res)) % ivec2(p.u_res);
            imageAtomicAdd(img_mass_accum, remainder_uv, remaining);
        }
        
        // Return kept mass to self (bounce back)
        if (kept_mass > 0u) {
            imageAtomicAdd(img_mass_accum, uv_i, kept_mass);
        }
    }
    
    // Store calculated velocity (source, instantaneous) into the G/B channels of the destination state
    // This allows compute_normalize to pick it up and preserve it for the next frame's signal advection.
    // We store it in .gb to match the standard format (Mass, VelX, VelY, Aux)
    imageStore(img_new_state, uv_i, vec4(0.0, vel.x, vel.y, 0.0));
}

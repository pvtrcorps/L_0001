#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float u_theta_A; // Previously _pad1
    float u_alpha_n; // Previously _pad2
    float u_temperature; // Temperature (s) for box distribution
    float u_signal_advect; // Signal advection weight
    float u_beta; // Selection pressure for negotiation rule
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

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_potential;
layout(set = 0, binding = 4, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 6, rgba32f) uniform image2D img_new_genome;
layout(set = 0, binding = 7) uniform sampler2D tex_signal;
layout(set = 0, binding = 8, r32ui) uniform uimage2D img_winner_tracker;

// PCG Hash (2d -> 1d) - High Quality, no Trig functions
uint pcg_hash(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v.x + v.y; // XOR mixing
}

float hash(vec2 pt) {
    // Map float coordinate to uints for hashing
    // Assuming pt is UV or similar, scaling helps avoid correlation
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

float pack2(float a, float b) {
    uint ia = uint(clamp(a, 0.0, 1.0) * 32767.0);
    uint ib = uint(clamp(b, 0.0, 1.0) * 32767.0);
    return uintBitsToFloat((ia << 15) | ib);
}

const float MASS_SCALE = 100000.0; 

// Stochastic Rounding to preserve mass statistically
// Using Robust Hash to avoid mass drift
uint get_rounded_amount(float amount, vec2 seed) {
    uint i = uint(amount);
    float f = fract(amount);
    
    // Mix seed with amount to decorrelate
    float h = hash(seed + vec2(amount, amount * 0.123));
    
    if (h < f) {
        i += 1u;
    }
    return i;
}

// 1D Box Intersection Area
// Overlap between pixel interval [px_min, px_max] and cloud interval [cloud_min, cloud_max]
float intersect_len(float px_min, float px_max, float cloud_min, float cloud_max) {
    return max(0.0, min(px_max, cloud_max) - max(px_min, cloud_min));
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    vec4 state = texture(tex_state, uv);
    float myMass = state.r;
    float age = state.a;
    
    // Genome Unpacking (New 8-Gene System)
    vec4 myGenome = texture(tex_genome, uv);
    // R: (b1_weight, b2_weight)
    // G: (b3_weight, a2_pos)
    // B: (kernel_width, kernel_radius)
    // A: (growth_mu, growth_sigma)
    vec2 growth_genes = unpack2(myGenome.a);
    vec2 width_radius = unpack2(myGenome.b);
    vec2 b3a2 = unpack2(myGenome.g);
    
    float growth_mu = growth_genes.x;
    float growth_sigma = growth_genes.y;
    float kernel_width = width_radius.x;
    float kernel_radius = width_radius.y;
    float a2_pos = b3a2.y;
    
    // Spectral genes for perception
    vec2 spectral_genes = unpack2(state.a);
    float detection_hue = spectral_genes.y;
    
    // Perception strength from gene (derived from stability)
    // Low sigma (unstable/picky) -> High perception needed
    float raw_perception = clamp(1.0 - growth_sigma, 0.0, 1.0);
    
    // Map to UI range
    float perception_strength = mix(p.u_range_perception.x, p.u_range_perception.y, raw_perception);
    // === 1. Calculate Gradient of Affinity U (using Sobel) ===
    // We compute gradient of U (Growth) NOT density. 
    // This allows flow away from center when density > mu due to G(u) shape.
    
    vec2 pixel_size = 1.0 / p.u_res;
    
    // Sobel Kernels (transposed relative to typical CPU definition to match GLSL axes)
    // dU/dx
    float gx = 0.0;
    gx += -1.0 * texture(tex_potential, uv + vec2(-1, -1)*pixel_size).r;
    gx += -2.0 * texture(tex_potential, uv + vec2(-1,  0)*pixel_size).r;
    gx += -1.0 * texture(tex_potential, uv + vec2(-1,  1)*pixel_size).r;
    gx +=  1.0 * texture(tex_potential, uv + vec2( 1, -1)*pixel_size).r;
    gx +=  2.0 * texture(tex_potential, uv + vec2( 1,  0)*pixel_size).r;
    gx +=  1.0 * texture(tex_potential, uv + vec2( 1,  1)*pixel_size).r;
    
    // dU/dy
    float gy = 0.0;
    gy += -1.0 * texture(tex_potential, uv + vec2(-1, -1)*pixel_size).r;
    gy += -2.0 * texture(tex_potential, uv + vec2( 0, -1)*pixel_size).r;
    gy += -1.0 * texture(tex_potential, uv + vec2( 1, -1)*pixel_size).r;
    gy +=  1.0 * texture(tex_potential, uv + vec2(-1,  1)*pixel_size).r;
    gy +=  2.0 * texture(tex_potential, uv + vec2( 0,  1)*pixel_size).r;
    gy +=  1.0 * texture(tex_potential, uv + vec2( 1,  1)*pixel_size).r;
    
    vec2 gradU_mass = vec2(gx, gy); // Magnitude is usually higher with Sobel, check scale
    
    // For Signal Gradient, we can still use the passed value or apply Sobel if needed.
    // Keeping existing signal logic for now but applying Sobel to Mass Affinity is CRITICAL.
    
    vec4 potential = texture(tex_potential, uv);
    float U_growth = potential.r;     // Mass convolution result
    // vec2 gradU_mass = potential.gb;   // Gradient of mass potential (now computed via Sobel)
    float U_signal = potential.a;     // Spectral similarity at this pixel
    
    // === SPECTRAL SIGNAL GRADIENT ===
    // Compute gradient of U_signal from neighboring pixels
    ivec2 res_i = ivec2(p.u_res);
    ivec2 l_uv = (uv_i + ivec2(-1, 0) + res_i) % res_i;
    ivec2 r_uv = (uv_i + ivec2(1, 0) + res_i) % res_i;
    ivec2 u_uv = (uv_i + ivec2(0, -1) + res_i) % res_i;
    ivec2 d_uv = (uv_i + ivec2(0, 1) + res_i) % res_i;
    
    // Read U_signal from neighboring potential textures (already contains spectral similarity)
    float sigL = texelFetch(tex_potential, l_uv, 0).a;
    float sigR = texelFetch(tex_potential, r_uv, 0).a;
    float sigU = texelFetch(tex_potential, u_uv, 0).a;
    float sigD = texelFetch(tex_potential, d_uv, 0).a;
    vec2 gradU_signal = vec2(sigR - sigL, sigD - sigU);
    
    // === UNIFIED FLOW POTENTIAL ===
    // U_flow = U_growth + perception * U_signal
    vec2 gradU = gradU_mass + perception_strength * gradU_signal;
    
    // === DENSITY GRADIENT (Repulsion / Pressure) ===
    float mR = texelFetch(tex_state, r_uv, 0).r;
    float mL = texelFetch(tex_state, l_uv, 0).r;
    float mD = texelFetch(tex_state, d_uv, 0).r;
    float mU = texelFetch(tex_state, u_uv, 0).r;
    vec2 gradA = vec2(mR - mL, mD - mU);

    // === EQUATION 5: Flow F = (1-alpha)*grad(G(U)) - alpha*grad(A) ===
    
    // alpha = [(A / theta_A)^n] clamped to [0,1]
    // 3. Alpha Calculation
    // Use UI parameter for theta_A (Critical Mass)
    float theta_A = p.u_theta_A;
    // Use UI parameter for alpha_n (Sharpness) 
    float alpha_n = p.u_alpha_n;
    float alpha_val = clamp(pow(myMass / max(theta_A, 0.001), alpha_n), 0.0, 1.0);
    
    // Official Flow Formula: nU * (1-alpha) - nA * alpha
    // Equivalent to: mix(gradU, -gradA, alpha)
    // Official code does NOT use a flow_speed multiplier (implies 1.0)
    // But we check kernel_width to give some variety
    
    float base_flow_speed = 1.0;
    vec2 force_affinity = gradU * base_flow_speed;
    vec2 force_repulsion = -gradA * base_flow_speed;
    
    vec2 totalVelocity = mix(force_affinity, force_repulsion, alpha_val);
    
    // Velocity Clipping (Critical for stability)
    // Official: clip to [-(dd-sigma), +(dd-sigma)]
    // dd=5, sigma=0.65 -> +/- 4.35
    float max_vel = 4.0; 
    totalVelocity = clamp(totalVelocity, -max_vel, max_vel);
    
    // === MASS ADVECTION (Conservative) with Temperature ===
    vec2 targetUV = uv + totalVelocity * p.u_dt * px;
    // Fix: Remove -0.5 offset. 
    // Pixel 'i' covers continuous range [i, i+1]. Its center is i+0.5.
    // targetUV * res gives the continuous coordinate where 0.5 is centers.
    vec2 targetPx = targetUV * p.u_res; 
    
    float survivingMass = myMass; 
    
    // Temperature S: Half-width of the square distribution
    float s = max(p.u_temperature, 0.01);
    
    if (survivingMass > 0.0) {
        // Target Cloud bounds (in pixel coords)
        vec2 cloudMax = targetPx + vec2(s);
        vec2 cloudMin = targetPx - vec2(s);
        
        // Affected Pixels Range
        ivec2 iMin = ivec2(floor(cloudMin));
        ivec2 iMax = ivec2(ceil(cloudMax)); // Exclusive? No, ceil gives upper integer
        // Actually, if cloudMax is 5.2, ceil is 6. The pixel at 5 (5.0-6.0) is included.
        // We iterate pixels p. Bounds of pixel p are [p, p+1].
        // Intersection of [p, p+1] with [cloudMin, cloudMax].
        
        // Optimization: Don't loop too far. With s=0.6, range is small (~2x2 or 3x3).
        
        ivec2 res = ivec2(p.u_res);
        float totalArea = (2.0 * s) * (2.0 * s); // Should be this analytically
        
        // Iterate over potential candidate pixels
        for (int y = iMin.y; y < iMax.y; y++) {
            float y_min = float(y);
            float y_max = float(y + 1);
            float h = intersect_len(y_min, y_max, cloudMin.y, cloudMax.y);
            if (h <= 0.0) continue;
            
            for (int x = iMin.x; x < iMax.x; x++) {
                float x_min = float(x);
                float x_max = float(x + 1);
                float w = intersect_len(x_min, x_max, cloudMin.x, cloudMax.x);
                
                if (w > 0.0) {
                    float area = w * h;
                    float weight = area / totalArea; // Normalize so sum(weights) = 1.0 across all pixels
                    
                    // Add mass to this neighbor
                    ivec2 p_neighbor = (ivec2(x, y) % res + res) % res; // Wrap
                    
                    uint m_int = uint(survivingMass * MASS_SCALE);
                    imageAtomicAdd(img_mass_accum, p_neighbor, get_rounded_amount(float(m_int) * weight, uv + vec2(float(x)*0.1, float(y)*0.1)));
                }
            }
        }
    }
    
    // === GENOME ADVECTION with BOX DISTRIBUTION & NEGOTIATION RULE ===
    // Paper-compliant: Use SAME box distribution as mass (Equations 6-7, ISAL 2023)
    // Negotiation Rule: Priority based on mass × exp(β × V(x_src)) (IMGEP 2025)
    if (survivingMass > 0.0) {
        // Use SAME box distribution parameters as mass transport
        // This ensures genome spreads to ~4-9 pixels (not just 4)
        vec2 cloudMax = targetPx + vec2(s);
        vec2 cloudMin = targetPx - vec2(s);
        
        ivec2 iMin = ivec2(floor(cloudMin));
        ivec2 iMax = ivec2(ceil(cloudMax));
        ivec2 res = ivec2(p.u_res);
        float totalArea = (2.0 * s) * (2.0 * s);
        
        // Compute negotiation priority using affinity map
        // V(x_src) = U_growth (already computed in convolution pass)
        // β = selection pressure parameter (higher = more competitive)
        
        // Normalize U_growth to typical range [0, 1] before exponential
        // Typical U_growth range: [0.0, 0.5] → normalize to [0, 1]
        float normalized_U = clamp((U_growth - 0.1) / 0.3, 0.0, 1.0);
        
        // Centered exponential: exp(β × (U - 0.5))
        // At U=0.5 (neutral): affinity_factor = 1.0
        // At U=0.0 (poor): affinity_factor = exp(-β/2)
        // At U=1.0 (excellent): affinity_factor = exp(β/2)
        float affinity_exponent = p.u_beta * (normalized_U - 0.5);
        float affinity_factor = exp(affinity_exponent);
        
        // Cap maximum advantage to prevent single-species dominance
        // With beta=1.0: max factor ≈ 1.65x (reasonable)
        // With beta=2.0: max factor ≈ 2.72x (competitive)
        affinity_factor = min(affinity_factor, 5.0);
        
        // Anti-Bias: Hash the source index to prevent spatial preference
        uint raw_idx = uint(uv_i.y * res.x + uv_i.x);
        uint seed_bits = floatBitsToUint(p.u_seed);
        uint frame_mask = pcg_hash_1d(seed_bits) & 0xFFFFFFu;
        uint scrambled_idx = raw_idx ^ frame_mask;
        
        // Iterate over all pixels in the box distribution
        for (int y = iMin.y; y < iMax.y; y++) {
            float y_min = float(y);
            float y_max = float(y + 1);
            float h = intersect_len(y_min, y_max, cloudMin.y, cloudMax.y);
            if (h <= 0.0) continue;
            
            for (int x = iMin.x; x < iMax.x; x++) {
                float x_min = float(x);
                float x_max = float(x + 1);
                float w = intersect_len(x_min, x_max, cloudMin.x, cloudMax.x);
                
                if (w > 0.0) {
                    float area = w * h;
                    float weight = area / totalArea;
                    
                    // Negotiation Rule Priority:
                    // P ∝ mass × I(src,dest) × exp(β × V(src))
                    float mass_contribution = survivingMass * weight;
                    float negotiation_priority = mass_contribution * affinity_factor;
                    
                    // Add small jitter to break ties
                    float jitter = hash(uv + vec2(float(x) * 0.1, float(y) * 0.1)) * 0.01;
                    negotiation_priority += jitter;
                    
                    // Pack into atomic-friendly format
                    // [Priority (12b)] [Scrambled Index (20b)]
                    uint priority_bits = uint(clamp(negotiation_priority * 1000.0, 0.0, 4095.0));
                    uint packed = (priority_bits << 20) | (scrambled_idx & 0xFFFFFu);
                    
                    // Compete for winner position in destination pixel
                    ivec2 p_neighbor = (ivec2(x, y) % res + res) % res;
                    imageAtomicMax(img_winner_tracker, p_neighbor, packed);
                }
            }
        }
    }
    
    // Write State (Advected Mass is accumulated in img_mass_accum via atomicAdd above)
    // We store Velocity and Age here for the next frame's "Previous State"
    imageStore(img_new_state, uv_i, vec4(0.0, totalVelocity, age + p.u_dt));
}

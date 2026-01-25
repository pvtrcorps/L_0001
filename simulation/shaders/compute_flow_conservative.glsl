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
    float _pad4; // mutation_rate
    float _pad5; // base_decay
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
    uint bits = floatBitsToUint(packed);
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
    
    // Genome Unpacking (8 Genes)
    vec4 myGenome = texture(tex_genome, uv);
    vec2 mu_sigma = unpack2(myGenome.r);
    vec2 radius_flow = unpack2(myGenome.g);
    float g_flow = radius_flow.y;
    vec2 affinity_lambda = unpack2(myGenome.b);
    float myAffinity = affinity_lambda.x;
    vec2 secre_percep = unpack2(myGenome.a);
    float g_perception = secre_percep.y;
    
    vec4 potential = texture(tex_potential, uv);
    vec2 gradU = potential.gb; 
    float g_prime = potential.a; // Growth force (G'(U))
    
    // === SIGNAL GRADIENT (Chemotaxis) ===
    vec2 signalGrad = vec2(0.0);
    ivec2 res_i = ivec2(p.u_res);
    
    // Helper for wrapping texelFetch
    ivec2 l_uv = (uv_i + ivec2(-1, 0) + res_i) % res_i;
    ivec2 r_uv = (uv_i + ivec2(1, 0) + res_i) % res_i;
    ivec2 u_uv = (uv_i + ivec2(0, -1) + res_i) % res_i;
    ivec2 d_uv = (uv_i + ivec2(0, 1) + res_i) % res_i;
    
    float sL = texelFetch(tex_signal, l_uv, 0).r;
    float sR = texelFetch(tex_signal, r_uv, 0).r;
    float sU = texelFetch(tex_signal, u_uv, 0).r;
    float sD = texelFetch(tex_signal, d_uv, 0).r;
    signalGrad = vec2(sR - sL, sD - sU);
    
    // === FLOW LENIA VELOCITY ===
    // === DENSITY GRADIENT (Repulsion / Pressure) ===
    // Use texelFetch for exact neighbors to avoid linear interpolation shift
    float mR = texelFetch(tex_state, r_uv, 0).r;
    float mL = texelFetch(tex_state, l_uv, 0).r;
    float mD = texelFetch(tex_state, d_uv, 0).r;
    float mU = texelFetch(tex_state, u_uv, 0).r;
    vec2 gradA = vec2(mR - mL, mD - mU);

    // === EQUATION 5: Flow F = (1-alpha)*grad(U) - alpha*grad(A) ===
    
    // 1. Calculate Alpha (Weighting Factor)
    // alpha = [(A / theta_A)^n] clamped to [0,1]
    
    // Use Gene for Theta (Repulsion Limit).
    // Original gene was 'lambda' (in affinity_lambda.y).
    // Map lambda [0..1] to reasonable Theta range [0.1 .. 5.0]
    float g_lambda = affinity_lambda.y;
    float gene_theta = 0.1 + g_lambda * 4.9;
    
    // Allow Global Toggle/Override? For now, let's mix or just use the gene dynamics.
    // Let's say the global 'theta_A' acts as a multiplier or base?
    // User requested "Species specific genes". So we rely on gene.
    // BUT we keep the global u_theta_A as a "Master Scalar" to tweak the whole balanced system.
    
    float theta_A = gene_theta * p.u_theta_A;
    float alpha_n = max(p.u_alpha_n, 1.0);
    
    float alpha_val = clamp(pow(myMass / theta_A, alpha_n), 0.0, 1.0);
    
    // 2. Term 1: Affinity Gradient Force (Attraction)
    // Note: Our g_prime * gradU is effectively grad(G(K*A)) = grad(Affinity)
    // movement_scale acts as the magnitude or 'dt' for the flow field integration over time step
    float flow_speed = 10.0 * (0.5 + g_flow);
    vec2 force_affinity = (g_prime * gradU) * flow_speed;
    
    // 3. Term 2: Density Gradient Force (Repulsion)
    // We negate gradA to move AWAY from high density
    vec2 force_repulsion = -gradA * flow_speed; 
    
    // Mix based on alpha
    // When mass is low (alpha~0), we follow affinity (form patterns)
    // When mass is high (alpha~1), we follow pressure (avoid collapse)
    vec2 flowVelocity = mix(force_affinity, force_repulsion, alpha_val);
    
    // Add Chemotaxis (External Signal)
    float chemotaxis_strength = (g_perception - 0.5) * 2.0;
    vec2 totalVelocity = flowVelocity + (signalGrad * chemotaxis_strength);
    
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
    
    // === DISCRETE GENOME ADVECTION (High-Precision ArgMax) ===
    // "Softmax Sampling" limit (Temperature -> 0): The source with the highest mass contribution 
    // determines the new identity. We use atomicMax to find the winner.
    if (survivingMass > 0.0) {
        // Fix for Genome Bias:
        // 'targetPx' is a CENTER coordinate (e.g. 10.5 is center of pixel 10).
        // The splatting logic below assumes CORNER coordinates (e.g. 10.0 is center of pixel 10).
        // We must subtract 0.5 so that 10.5 becomes 10.0, producing fract=0, writing purely to pixel 10.
        // Without this, 10.5 gives fract=0.5, distributing genome to 10,11,10,11 (Down-Right bias).
        vec2 targetPx_ID = targetPx - 0.5;
        
        ivec2 tl = ivec2(floor(targetPx_ID));
        vec2 f = fract(targetPx_ID);
        ivec2 res = ivec2(p.u_res);
        
        // Anti-Bias: Hash the index to prevent "Bottom-Right" preference in ties.
        // We use a XOR mask derived from the random seed to shuffle the index preference every frame.
        // This ensures that "who wins the tie" is spatially and temporally random (white noise), 
        // eliminating global diagonal drift.
        
        uint raw_idx = uint(uv_i.y * res.x + uv_i.x);
        
        // Create a mask from the seed (which changes every frame)
        // Use robust hash of the float bits to ensure the mask changes drastically and covers high bits.
        // floatBitsToUint preserves entropy of the seed update.
        uint seed_bits = floatBitsToUint(p.u_seed);
        uint frame_mask = pcg_hash_1d(seed_bits) & 0xFFFFFFu; 
        
        // Scramble the index for the comparison key
        uint scrambled_idx = raw_idx ^ frame_mask;
        
        // 1. Add random jitter to mass for priority
        float jitter = hash(uv + vec2(p.u_dt, p.u_seed)) * 0.9; 
        uint p_mass = uint(clamp(survivingMass * 4000.0, 0.0, 4000.0));
        uint final_priority = uint(clamp(survivingMass * 4000.0 + jitter * 10.0, 0.0, 4095.0));

        // Packed Layout: [Priority Mass (12b)] [Scrambled Index (20b)]
        uint packed = (final_priority << 20) | (scrambled_idx & 0xFFFFFFu);

        if ((1.0-f.x)*(1.0-f.y) > 0.0) imageAtomicMax(img_winner_tracker, (tl + res) % res, packed);
        if (f.x*(1.0-f.y) > 0.0)       imageAtomicMax(img_winner_tracker, (ivec2(tl.x+1, tl.y) + res) % res, packed);
        if ((1.0-f.x)*f.y > 0.0)       imageAtomicMax(img_winner_tracker, (ivec2(tl.x, tl.y+1) + res) % res, packed);
        if (f.x*f.y > 0.0)             imageAtomicMax(img_winner_tracker, (ivec2(tl.x+1, tl.y+1) + res) % res, packed);
    }
    
    // Write State (Advected Mass is accumulated in img_mass_accum via atomicAdd above)
    // We store Velocity and Age here for the next frame's "Previous State"
    imageStore(img_new_state, uv_i, vec4(0.0, totalVelocity, age + p.u_dt));
}

#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float _pad1; // repulsion
    float _pad2; // damage
    float _pad3; // identity_thr
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

float hash(vec2 pt) {
    return fract(sin(dot(pt, vec2(12.9898, 78.233))) * 43758.5453);
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
uint get_rounded_amount(float amount, vec2 seed) {
    uint i = uint(amount);
    float f = fract(amount);
    if (hash(seed) < f) {
        i += 1u;
    }
    return i;
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
    float sL = texture(tex_signal, uv + vec2(-px.x, 0.0)).r;
    float sR = texture(tex_signal, uv + vec2(px.x, 0.0)).r;
    float sU = texture(tex_signal, uv + vec2(0.0, -px.y)).r;
    float sD = texture(tex_signal, uv + vec2(0.0, px.y)).r;
    signalGrad = vec2(sR - sL, sD - sU);
    
    // === FLOW LENIA VELOCITY ===
    // v = C * G'(U) * gradU
    // C is a global or gene-dependent scaling factor
    float movement_scale = 10.0 * (0.5 + g_flow);
    vec2 flowVelocity = g_prime * gradU * movement_scale;
    
    // === AFFINITY INTERACTION ===
    // Creatures with similar affinity attract, different affinity might repel or be neutral
    // In strict Flow Lenia, interaction often emerges from the kernel itself, but
    // we can add a subtle affinity-driven flux if needed.
    
    float chemotaxis_strength = (g_perception - 0.5) * 2.0;
    
    vec2 totalVelocity = flowVelocity + (signalGrad * chemotaxis_strength);
    
    // === MASS ADVECTION (Conservative PUSH) ===
    vec2 targetUV = uv + totalVelocity * p.u_dt * px;
    vec2 targetPx = targetUV * p.u_res - 0.5; 
    
    // STRICT MASS CONSERVATION: No damage, no decay
    float survivingMass = myMass; 
    
    if (survivingMass > 0.0) {
        ivec2 tl = ivec2(floor(targetPx));
        vec2 f = fract(targetPx);
        
        float w00 = (1.0-f.x)*(1.0-f.y);
        float w10 = f.x*(1.0-f.y);
        float w01 = (1.0-f.x)*f.y;
        float w11 = f.x*f.y;
        
        uint m_int = uint(survivingMass * MASS_SCALE);
        ivec2 res = ivec2(p.u_res);
        float f_m = float(m_int);
        
        ivec2 p00 = (tl + res) % res; 
        if (w00 > 0.0) imageAtomicAdd(img_mass_accum, p00, get_rounded_amount(f_m * w00, uv + vec2(0.1, p.u_dt)));
        
        ivec2 p10 = (ivec2(tl.x+1, tl.y) + res) % res;
        if (w10 > 0.0) imageAtomicAdd(img_mass_accum, p10, get_rounded_amount(f_m * w10, uv + vec2(0.2, p.u_dt)));
        
        ivec2 p01 = (ivec2(tl.x, tl.y+1) + res) % res;
        if (w01 > 0.0) imageAtomicAdd(img_mass_accum, p01, get_rounded_amount(f_m * w01, uv + vec2(0.3, p.u_dt)));
        
        ivec2 p11 = (ivec2(tl.x+1, tl.y+1) + res) % res;
        if (w11 > 0.0) imageAtomicAdd(img_mass_accum, p11, get_rounded_amount(f_m * w11, uv + vec2(0.4, p.u_dt)));
    }
    
    // === DISCRETE GENOME ADVECTION (High-Precision ArgMax) ===
    // "Softmax Sampling" limit (Temperature -> 0): The source with the highest mass contribution 
    // determines the new identity. We use atomicMax to find the winner.
    if (survivingMass > 0.0) {
        ivec2 tl = ivec2(floor(targetPx));
        vec2 f = fract(targetPx);
        ivec2 res = ivec2(p.u_res);
        
        // My unique identifier (1D index) - Max 1024x1024 = 1,048,576 (Needs 20 bits)
        uint my_idx = uint(uv_i.y * res.x + uv_i.x);
        
        // Pack (12-bit Mass, 20-bit Index)
        // 12 bits = 0..4095. Scale mass appropriately.
        // Assuming max mass per packet is < 1.0 (usually much less due to splitting).
        // Using a non-linear scale (sqrt) or just high gain could help preserve small differences.
        // Let's use simple linear scaling with a higher clamp.
        
        uint p_mass = uint(clamp(survivingMass * 4095.0, 0.0, 4095.0));
        
        // Packed Layout: [Mass (12b from 31..20)] [Index (20b from 19..0)]
        uint packed = (p_mass << 20) | (my_idx & 0xFFFFFFu); // 0xFFFFF is 20 bits, strictly it should be 0xFFFFF

        // NOTE: atomicMax compares the whole uint. 
        // Higher mass (MSB) wins. Loops/Ties broken by Index (LSB).
        
        if ((1.0-f.x)*(1.0-f.y) > 0.0) imageAtomicMax(img_winner_tracker, (tl + res) % res, packed);
        if (f.x*(1.0-f.y) > 0.0)       imageAtomicMax(img_winner_tracker, (ivec2(tl.x+1, tl.y) + res) % res, packed);
        if ((1.0-f.x)*f.y > 0.0)       imageAtomicMax(img_winner_tracker, (ivec2(tl.x, tl.y+1) + res) % res, packed);
        if (f.x*f.y > 0.0)             imageAtomicMax(img_winner_tracker, (ivec2(tl.x+1, tl.y+1) + res) % res, packed);
    }
    
    // Write State (Advected Mass is accumulated in img_mass_accum via atomicAdd above)
    // We store Velocity and Age here for the next frame's "Previous State"
    imageStore(img_new_state, uv_i, vec4(0.0, totalVelocity, age + p.u_dt));
}

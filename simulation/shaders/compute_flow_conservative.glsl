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

// Helper for Square Cloud Intersection
float intersect_len(float min1, float max1, float min2, float max2) {
    return max(0.0, min(max1, max2) - max(min1, min2));
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
    vec2 mu_sigma = unpack2(g1.r); 
    float g_mu = mu_sigma.x;
    float g_sigma = 0.001 + mu_sigma.y * 0.2; // Consistent with convolution shader

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
    
    // === 2. UNIFIED FLOW POTENTIAL ===
    
    // Read Potentials (Computed in Convolution Step)
    // .r = G(U) (Growth function result)
    // .a = U_signal (Spectral similarity)
    // We assume convolution shader calculates these.
    // However, to compute G'(U), we ideally need the raw potential U before growth function.
    // But since we don't store raw U, we can approximate the gradient direction using the gradient of G(U).
    // Or we can rely on the fact that G(U) is monotonic on either side of the peak.
    
    vec4 potential = texture(tex_potential, uv);
    float U_growth = potential.r; // G(U)
    // float U_signal = potential.a; // U_signal (already computed)

    // A. Mass/Growth Gradient
    // Sobel Filter for d(U_growth)/dx, d(U_growth)/dy
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
    
    vec2 gradU_mass = vec2(gx, gy);
    
    // B. Signal Gradient (Chemotaxis)
    // Compute gradient of spectral similarity (U_signal)
    ivec2 res_i = ivec2(p.u_res);
    ivec2 l_uv = (uv_i + ivec2(-1, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 r_uv = (uv_i + ivec2(1, 0) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 u_uv = (uv_i + ivec2(0, -1) + ivec2(p.u_res)) % ivec2(p.u_res);
    ivec2 d_uv = (uv_i + ivec2(0, 1) + ivec2(p.u_res)) % ivec2(p.u_res);
    
    // tex_potential.a stores U_signal
    float sL = texelFetch(tex_potential, l_uv, 0).a;
    float sR = texelFetch(tex_potential, r_uv, 0).a;
    float sU = texelFetch(tex_potential, u_uv, 0).a;
    float sD = texelFetch(tex_potential, d_uv, 0).a;
    vec2 gradU_signal = vec2(sR - sL, sD - sU);
    
    // Unified Gradient Construction
    // Perception Weight: Mapped to [-1, 1] to allow repulsion
    float perception_weight = (g_sensitivity - 0.5) * 2.0; 
    
    // Combine gradients. Signal affects the slope of the effective potential.
    vec2 gradUnified = gradU_mass + perception_weight * gradU_signal * p.u_signal_advect * 3.0; // Added global generic multiplier for signal strength

    // Compute derivative G'(U) Locally (HEURISTIC)
    // We approximate the direction "Towards Optimal U".
    // Optimal U is roughly mu.
    // If G(U) is high, we are at peak. G'(U) ~ 0.
    // If G(U) is low, we are far from peak.
    // BUT, simple gradient of G(U) already points towards the peak!
    // The previous implementation used an explicit G'(U) calculation.
    // Since we don't have raw U, we will trust that gradUnified (which is approx grad(G(U)))
    // already encodes the "direction of improvement".
    // To restore the *feeling* of the old commit where "Stable mass doesn't move", 
    // we should note that grad(G(U)) is naturally zero at the peak (where G(U) is max).
    // So the "G'(U)" term is actually redundant if we use the gradient of G(U) directly,
    // *provided* we aren't adding other forces that ignore this stability.
    
    // However, the commit did this:
    // float diff = (U_growth - optimalU);
    // float g_prime = ... exp(...)
    // force = g_prime * gradU;
    
    // We need to simulate this modulation. 
    // If we are at the peak (U_growth ~ 1.0), force should be zero.
    // If we are at the edge, force should be high.
    // Let's use (1.0 - U_growth) as a proxy for "Distance from Peak" to modulate forces.
    // Or simpler: gradU_mass IS the derivative. It is zero at peak.
    // The signal term needs to be modulated by this "Instability" so it doesn't move stable mass.
    // So:
    
    float instability = 1.0 - U_growth; // 0.0 at peak, 1.0 at void
    // Actually, we want movement at the "skin" (surface).
    // Let's stick to the Unified Potential logic:
    // Force = (grad G(U) + w * grad S)
    // This naturally stops at the peak of G(U) *if* grad S isn't too strong.
    // But the user liked the specific physics of that commit.
    // Let's assume the "Unified Gradient" approach above captures the essence:
    // Adding the signal gradient TO the mass gradient effectively shifts the "peak" of the potential hill.
    
    // Force 1: Affinity (Growth Directed)
    vec2 force_affinity = gradUnified * (0.5 + g_affinity * 2.5);
    
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
    vec2 totalVelocity = speed_mult * mix(force_affinity, -gradA * (0.5 + g_repulsion * 2.5), alpha);
    
    // APPLY VISCOSITY (Drag)
    // V_final = V * (1 - viscosity)
    totalVelocity *= clamp(1.0 - g_viscosity * 0.9, 0.1, 1.0);
    
    
    // === 3. Mass Advection (Square Cloud Sampling) ===
    
    // Target position (Forward advection)
    vec2 targetUV = uv + totalVelocity * p.u_dt * px;
    vec2 targetPx = targetUV * p.u_res;  // Continuous pixel coordinates (e.g. 10.5, 20.3)
    
    float survivingMass = myMass;
    
    // Temperature S: Half-width of the square distribution
    float s = max(p.u_temperature, 0.01);
    
    if (survivingMass > 0.0001) {
        // Target Cloud bounds (in pixel coords)
        vec2 cloudMax = targetPx + vec2(s);
        vec2 cloudMin = targetPx - vec2(s);
        
        // Affected Pixels Range
        ivec2 iMin = ivec2(floor(cloudMin));
        ivec2 iMax = ivec2(ceil(cloudMax));
        
        ivec2 res = ivec2(p.u_res);
        float totalArea = (2.0 * s) * (2.0 * s); // Theoretical area
        
        // Iterate over potential candidate pixels
        for (int y = iMin.y; y < iMax.y; y++) {
            float h = intersect_len(float(y), float(y + 1), cloudMin.y, cloudMax.y);
            if (h <= 0.0) continue;
            
            for (int x = iMin.x; x < iMax.x; x++) {
                float w = intersect_len(float(x), float(x + 1), cloudMin.x, cloudMax.x);
                
                if (w > 0.0) {
                    float area = w * h;
                    float weight = area / totalArea; // Normalize
                    
                    // Add mass to this neighbor
                    ivec2 p_neighbor = (ivec2(x, y) % res + res) % res; // Wrap
                    
                    uint m_int = uint(survivingMass * MASS_SCALE);
                    imageAtomicAdd(img_mass_accum, p_neighbor, get_rounded_amount(float(m_int) * weight, uv + vec2(float(x)*0.1, float(y)*0.1)));

                    // WINNER TRACKING (Genome Advection)
                    // Pack (Mass Contribution << 24) | (Source Index)
                    uint src_idx = uint(uv_i.y) * uint(p.u_res.x) + uint(uv_i.x);
                    src_idx = src_idx & 0xFFFFFFu;
                    
                    // Calculate contribution for this specific neighbor
                    // FIX: Use Sqrt Curve for Advection Score
                    // 1. Allows small mass to pass threshold (> 1) -> Breaks Invisible Walls
                    // 2. Prevents saturation at 250 -> Prevents "Left-Up Bias" (Index War)
                    // 3. Add DITHERING to prevent "Diagonal Cuts" (Quantization Banding)
                    uint m_contrib = 0;
                    float mass_contribution = myMass * weight;
                    if (mass_contribution > 1.0e-7) {
                         // Dithering: Add small noise to break ties in discrete bands
                         float noise = fract(sin(dot(vec2(p_neighbor), vec2(12.9898, 78.233))) * 43758.5453);
                         float dither = noise * 0.9;
                         
                         // Sqrt curve + Dither
                         m_contrib = uint(clamp(sqrt(mass_contribution) * 200.0 + dither, 1.0, 250.0));
                    }

                    if (m_contrib > 0) {
                         imageAtomicMax(img_winner_tracker, p_neighbor, (m_contrib << 24u) | src_idx);
                    }
                }
            }
        }
    }
    
    // Store calculated velocity (source, instantaneous) into the G/B channels of the destination state
    // This allows compute_normalize to pick it up and preserve it for the next frame's signal advection.
    // We store it in .gb to match the standard format (Mass, VelX, VelY, Aux)
    imageStore(img_new_state, uv_i, vec4(0.0, totalVelocity.x, totalVelocity.y, 0.0));
}

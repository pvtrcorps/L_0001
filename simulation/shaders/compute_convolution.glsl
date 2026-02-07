#version 450

// === OPTIMIZATION CONSTANTS ===
// Phase 1 Optimizations:
// - Workgroup 16×16 for better GPU occupancy (256 threads vs 64)
// - Kernel LUT in shared memory to avoid redundant exp() calls
#define WORKGROUP_SIZE 16
#define MAX_RADIUS 20
#define TILE_WIDTH (WORKGROUP_SIZE + 2 * MAX_RADIUS)
#define TILE_AREA (TILE_WIDTH * TILE_WIDTH)
#define PI 3.14159265359

layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;

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
    float u_interaction_beta; // [NEW] Interaction Strength
    float u_pad2;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;     // Genes 1-8
layout(set = 0, binding = 3) uniform sampler2D tex_signal;
layout(set = 0, binding = 4, rgba32f) uniform image2D img_potential;
layout(set = 0, binding = 5) uniform sampler2D tex_genome_ext; // Genes 9-16 [NEW]

// === SHARED MEMORY ===
// Stores: .x = Mass, .y = Emission Hue (Identity)
shared vec2 tile_cache[TILE_AREA];

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u; // Clear normalized bit
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

float gaussian(float x, float mu, float sigma) {
    float d = (x - mu) / max(sigma, 0.001);
    return exp(-0.5 * d * d);
}

// Pre-calculated Kernel Parameters
struct KernelParams {
    float b1, b2, b3;
    float a1, a2, a3;
    float w1, w2, w3;
};

// Optimized Kernel: Params are pre-calculated, only distance `r` varies
float eval_kernel(float r, float R_actual, KernelParams k) {
    float norm_r = r / R_actual;
    if (norm_r > 1.0) return 0.0;
    
    // Gaussian bumps
    // We can inline Gaussian here for speed if needed, but the compiler usually buffers it well
    float k1 = k.b1 * exp(-0.5 * ((norm_r - k.a1)/k.w1) * ((norm_r - k.a1)/k.w1));
    float k2 = k.b2 * exp(-0.5 * ((norm_r - k.a2)/k.w2) * ((norm_r - k.a2)/k.w2));
    float k3 = k.b3 * exp(-0.5 * ((norm_r - k.a3)/k.w3) * ((norm_r - k.a3)/k.w3));
    
    return k1 + k2 + k3;
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

// === INTERACTION FUNCTION ===
// Returns affinity multiplier [0.0, 2.0] based on genetic match
// my_target: The Hue I want to see (Detection)
// their_id: The Hue they are (Emission)
// beta: Strength of preference
float get_interaction_affinity(float my_target, float their_id, float beta) {
    if (beta <= 0.001) return 1.0;
    
    // Circular difference [0, 1]
    float diff = abs(my_target - their_id);
    if (diff > 0.5) diff = 1.0 - diff;
    
    // Cosine similarity mapped to modulation
    // If diff = 0 (Match) -> cos(0) = 1 -> Return 1 + beta
    // If diff = 0.5 (Opposite) -> cos(PI) = -1 -> Return 1 - beta
    float match = cos(2.0 * PI * diff);
    
    // Result range: [1-beta, 1+beta]
    // If beta=1, Range [0, 2].
    return 1.0 + beta * match;
}

void main() {
    ivec2 res_i = ivec2(p.u_res);
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);
    ivec2 group_id = ivec2(gl_WorkGroupID.xy);
    
    // Base coordinate of the tile (top-left of the halo region)
    // The tile centers on the WorkGroup.
    // WorkGroup covers [group_id * 8, group_id * 8 + 7]
    // Tile starts at [group_id * 8 - MAX_RADIUS, group_id * 8 - MAX_RADIUS]
    ivec2 tile_base = group_id * WORKGROUP_SIZE - ivec2(MAX_RADIUS);
    
    // === 1. COLLABORATIVE LOADING ===
    // Total pixels to load = TILE_AREA
    // Total threads = WORKGROUP_SIZE * WORKGROUP_SIZE = 64
    uint thread_idx = gl_LocalInvocationIndex; // 0..63
    uint total_threads = WORKGROUP_SIZE * WORKGROUP_SIZE;
    
    for (uint i = thread_idx; i < TILE_AREA; i += total_threads) {
        // Map linear index 'i' to tile local coord (tx, ty)
        int tx = int(i) % TILE_WIDTH;
        int ty = int(i) / TILE_WIDTH;
        
        // Calculate global coordinate with wrapping
        ivec2 global_pos = (tile_base + ivec2(tx, ty) + res_i) % res_i; // True standard modulo
        // Fix standard modulo for negative numbers in GLSL: (a % n + n) % n
        global_pos = (tile_base + ivec2(tx, ty));
        global_pos = (global_pos % res_i + res_i) % res_i;
        
        // Fetch Global Memory
        // 1. Mass
        float mass = texelFetch(tex_state, global_pos, 0).r;
        
        // 2. Emission Hue (Identity) from Genome Ext (Genes 9-16)
        // Texture packs 4 channels. 
        // We know from LeniaSimulation that params["g_emission_hue"] is Gene 15.
        // Gene 15/16 are in Bound Texture 'tex_genome_ext' Channel Alpha.
        vec4 g_ext_packed = texelFetch(tex_genome_ext, global_pos, 0);
        
        // Unpack Alpha channel to get (Emission, Detection)
        vec2 hues = unpack2(g_ext_packed.a);
        float emission_hue = hues.x; // Gene 15
        
        // Store in Shared Memory
        tile_cache[i] = vec2(mass, emission_hue);
    }
    
    // Wait for all threads to finish loading
    barrier();
    
    // === 2. COMPUTE PIXEL ===
    // If outside screen, we still helped load, but we don't compute.
    // (Wait, we can't return early easily because 'barrier' must be hit by all flows if we had another one, 
    // but here we are done with barriers, so early exit is fine for computation).
    if (gid.x >= int(p.u_res.x) || gid.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(gid) + 0.5) / p.u_res;
    
    // 1. Read Genome
    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);
    
    // === VOID CHECK ===
    if (dot(g1, g1) < 0.0001) {
        imageStore(img_potential, gid, vec4(0.0));
        return;
    }

    // Unpack Key Genes
    vec2 mu_sigma = unpack2(g1.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    
    vec2 rad_visc = unpack2(g1.g);
    float g_radius = rad_visc.x; 
    
    vec2 hue_hue = unpack2(g2.a);
    float my_emission_hue = hue_hue.x; // My Identity (Gene 15)
    float g_detection_hue = hue_hue.y; // My Target Hue (Gene 16)
    
    // === PREPARE KERNEL ===
    float R_actual = max(p.u_R * g_radius, 1.0);
    
    // Clamp R to our Max Radius allocated in Shared Mem
    // This is the physical limit of our optimization.
    R_actual = min(R_actual, float(MAX_RADIUS)); 
    
    vec2 shape_ab = unpack2(g1.b);
    vec2 shape_c_gr = unpack2(g1.a);
    
    KernelParams kp;
    kp.b1 = 0.1 + shape_ab.x * 0.9;
    kp.b3 = 0.1 + (1.0 - shape_ab.x) * 0.9;
    kp.b2 = shape_ab.y;
    
    kp.a1 = 0.15;
    kp.a2 = 0.35 + shape_c_gr.x * 0.3;
    kp.a3 = 0.85;
    
    kp.w1 = 0.15;
    kp.w2 = 0.20;
    kp.w3 = 0.15;
    
    int loopR = int(ceil(R_actual));
    
    float sum = 0.0;
    float totalWeight = 0.0;
    
    // Our position within the tile
    // lid is (0..7), tile starts at -MAX_RADIUS.
    // So our index in tile is (lid.x + MAX_RADIUS, lid.y + MAX_RADIUS)
    ivec2 my_tile_pos = lid + ivec2(MAX_RADIUS);
    
    // Loop only the necessary radius
    for (int dy = -loopR; dy <= loopR; dy++) {
        for (int dx = -loopR; dx <= loopR; dx++) {
            
            // Shared Memory Lookup
            int tx = my_tile_pos.x + dx;
            int ty = my_tile_pos.y + dy;
            
            // Flatten index
            int idx = ty * TILE_WIDTH + tx;
            
            vec2 cell_data = tile_cache[idx];
            float neighborMass = cell_data.x;
            float neighborHue = cell_data.y; // Their Identity
            
            float dist = length(vec2(float(dx), float(dy)));
            
            float w = eval_kernel(dist, R_actual, kp);
           
            if (w > 0.0001) {
                // === KERNEL INTERACTION ===
                
                // 1. IDENTITY CHECK (Self/Kin Recognition)
                // Compare MY Identity (Emission) vs THEIR Identity (Emission)
                float genetic_dist = abs(my_emission_hue - neighborHue);
                if (genetic_dist > 0.5) genetic_dist = 1.0 - genetic_dist;
                
                float affinity = 1.0;
                
                if (genetic_dist < 0.1) {
                    // SAME SPECIES: Standard Physics (Affinity = 1.0)
                    // We ignore our "Target Preference" when dealing with family.
                    affinity = 1.0; 
                } else {
                    // DIFFERENT SPECIES: Apply Interaction Rule
                    // Compare MY Target (Detection) vs THEIR Identity (Emission)
                    affinity = get_interaction_affinity(g_detection_hue, neighborHue, p.u_interaction_beta);
                }
                
                sum += neighborMass * w * affinity;
                totalWeight += w; // Normalize by PHYSICAL weight
            }
        }
    }
    
    float U_raw = (totalWeight > 0.0) ? sum / totalWeight : 0.0;
    
    // === GROWTH G(U) ===
    float mu = g_mu; 
    float sigma = 0.001 + g_sigma * 0.2; 
    
    float diff = (U_raw - mu);
    float exp_term = exp(-0.5 * (diff * diff) / max(sigma * sigma, 0.0001));
    float U_growth = 2.0 * exp_term - 1.0; 
    
    // Output:
    // R: Growth Potential (G(U))
    // A: Signal Potential (Now 0.0, disabled signal force)
    imageStore(img_potential, gid, vec4(U_growth, 0.0, 0.0, 0.0));
}


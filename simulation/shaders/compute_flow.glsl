#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float _unused1, _unused2, _unused3;
    float u_mutation_rate;
    float u_base_decay;
    float u_init_clusters;
    float u_init_density;
    float _pad0;
    vec2 u_range_mu;
    vec2 u_range_sigma;
    vec2 u_range_radius;
    vec2 u_range_flow;
    vec2 u_range_affinity;
    vec2 u_range_lambda;
    float _pad1, _pad2, _pad3;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_potential;

layout(set = 0, binding = 4, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_genome;

float hash(vec2 pt) {
    return fract(sin(dot(pt, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed);
    float a = float(bits >> 16) / 65535.0;
    float b = float(bits & 0xFFFFu) / 65535.0;
    return vec2(a, b);
}

float pack2(float a, float b) {
    uint ia = uint(clamp(a, 0.0, 1.0) * 65535.0);
    uint ib = uint(clamp(b, 0.0, 1.0) * 65535.0);
    return uintBitsToFloat((ia << 16) | ib);
}

float growth(float U, float g_mu, float g_sigma) {
    float optimalU = 0.15 + g_mu * 0.35; 
    float tolerance = 0.03 + g_sigma * 0.12;
    float d = (U - optimalU) / tolerance;
    return 2.0 * exp(-0.5 * d * d) - 1.0;
}

// Mix two genomes based on ratio
vec4 mix_genomes(vec4 g1, vec4 g2, float ratio) {
    // Unpack, mix, repack.
    // Optimization: Just lerp the vec4?
    // Packed floats are linear [0,1], so linear interpolation is roughly valid approximation
    // pack2(lerp(a,b,t)) approx lerp(pack2(a), pack2(b), t) for small differences
    // But bitwise limits precision. Let's do full unpack for correctness.
    
    vec2 rf1 = unpack2(g1.b);
    vec2 al1 = unpack2(g1.a);
    vec2 rf2 = unpack2(g2.b);
    vec2 al2 = unpack2(g2.a);
    
    float mu = mix(g1.r, g2.r, ratio);
    float sigma = mix(g1.g, g2.g, ratio);
    float rad = mix(rf1.x, rf2.x, ratio);
    float flow = mix(rf1.y, rf2.y, ratio);
    float aff = mix(al1.x, al2.x, ratio);
    float lam = mix(al1.y, al2.y, ratio);
    
    return vec4(mu, sigma, pack2(rad, flow), pack2(aff, lam));
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;
    
    vec4 state = texture(tex_state, uv);
    float myMass = state.r;
    float age = state.a;
    
    // Self Genome
    vec4 myGenome = texture(tex_genome, uv);
    float myAffinity = unpack2(myGenome.a).x;
    
    vec4 potential = texture(tex_potential, uv);
    float U = potential.r;
    vec2 gradU = potential.gb; 
    
    vec2 radius_flow = unpack2(myGenome.b);
    float g_flow = radius_flow.y;
    vec2 affinity_lambda = unpack2(myGenome.a);
    float g_lambda = affinity_lambda.y;
    
    // === REPULSION & COMBAT SCAN ===
    vec2 repulsion = vec2(0.0);
    float damage = 0.0;
    
    // Scan immediate 3x3 neighbors
    if (myMass > 0.01) {
        for(int y=-2; y<=2; y++) {
            for(int x=-2; x<=2; x++) {
                if(x==0 && y==0) continue;
                vec2 off = vec2(float(x), float(y));
                vec2 nUV = uv + off * px;
                
                vec4 nVal = texture(tex_state, nUV);
                if (nVal.r > 0.05) {
                    float nAff = unpack2(texture(tex_genome, nUV).a).x;
                    if (abs(nAff - myAffinity) > 0.2) {
                        // Enemy found! 
                        float dist = length(off);
                        vec2 dir = off / dist;
                        // Push away strongly
                        repulsion -= dir * nVal.r;
                        // Burn damage
                        damage += nVal.r * 2.0;
                    }
                }
            }
        }
    }
    
    float force_flow = 4.0 * max(0.2, g_flow); 
    // Combine natural flow with active repulsion
    vec2 totalVelocity = (gradU * force_flow) + (repulsion * 8.0);
    
    vec2 sourceUV = uv - totalVelocity * p.u_dt * px;
    
    vec4 srcState = texture(tex_state, sourceUV);
    vec4 srcGenome = texture(tex_genome, sourceUV);
    float incomingMass = srcState.r;
    float srcAffinity = unpack2(srcGenome.a).x;
    
    float srcU = texture(tex_potential, sourceUV).r;
    float G = growth(srcU, srcGenome.r, srcGenome.g);
    
    float metabolism = 0.5 + g_lambda * 1.5; 
    float deltaMass = G * p.u_dt * metabolism;
    
    // === INTERACTION LOGIC ===
    // If I am occupied (myMass > 0.1), I defend my identity.
    // If incoming mass is different species (affinity diff high), I eat it (gain mass) but ignore its genome.
    
    vec4 nextGenome = srcGenome;
    
    if (myMass > 0.05) {
        float affinityDiff = abs(myAffinity - srcAffinity);
        if (affinityDiff > 0.2) {
            // PREDATION / COMPETITION
            // Different species.
            // If incoming mass > my mass * 1.5, I get overrun (replaced).
            // Else, I keep my genome (I eat them).
            
            if (incomingMass > myMass * 1.2) {
                 // They conquer me
                 nextGenome = srcGenome;
            } else {
                 // I resist (eat/defend)
                 nextGenome = myGenome;
            }
        } else {
            // COOPERATION / SAME SPECIES
            // Smoothly blend genomes to maintain local coherence
            // Bias towards the one with more mass? Or just incoming (advection)?
            // Advection implies moving mass carries its properties.
            // But if I am effectively staying here (Lagrangian view), I should mix.
            // Simple advection takes srcGenome.
            // Let's stick to srcGenome for same species (flow carries DNA).
            nextGenome = srcGenome;
        }
    } else {
        // I am empty/weak, I just get colonized
        nextGenome = srcGenome;
    }
    
    
    float totalDecay = p.u_base_decay * metabolism * p.u_dt;
    
    // Apply active combat damage
    float newMass = incomingMass + deltaMass - totalDecay - (damage * p.u_dt);
    newMass = clamp(newMass, 0.0, 1.0);
    
    // Mutation
    if (hash(uv + vec2(p.u_seed, p.u_dt)) < p.u_mutation_rate) {
        float drift = (hash(uv * 2.0 + p.u_seed) - 0.5) * 0.1;
        nextGenome.r = fract(nextGenome.r + drift);        
        nextGenome.g = clamp(nextGenome.g + drift * 0.5, 0.01, 1.0); 
        
        vec2 rf = unpack2(nextGenome.b);
        rf.x = clamp(rf.x + drift * 0.5, 0.0, 1.0); 
        rf.y = clamp(rf.y + drift * 0.5, 0.0, 1.0); 
        nextGenome.b = pack2(rf.x, rf.y);
        
        vec2 al = unpack2(nextGenome.a);
        al.x = clamp(al.x + drift * 0.5, 0.0, 1.0); 
        al.y = clamp(al.y + drift * 0.5, 0.0, 1.0); 
        nextGenome.a = pack2(al.x, al.y);
    }
    
    // Store new state with Repulsion Velocity
    imageStore(img_new_state, uv_i, vec4(newMass, totalVelocity, age + p.u_dt));
    imageStore(img_new_genome, uv_i, nextGenome);
}

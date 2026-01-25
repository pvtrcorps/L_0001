#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    float u_dt;
    float u_seed;
    float u_R;
    float u_repulsion_strength;
    float u_combat_damage;
    float u_identity_thr;
    float u_mutation_rate;
    float u_base_decay;
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
    float _pad;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;
layout(set = 0, binding = 3) uniform sampler2D tex_potential;
layout(set = 0, binding = 4, r32ui) uniform uimage2D img_mass_accum;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_state;
layout(set = 0, binding = 6, rgba32f) uniform image2D img_new_genome;
layout(set = 0, binding = 7) uniform sampler2D tex_signal;

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
    
    // === SIGNAL GRADIENT (Chemotaxis) ===
    vec2 signalGrad = vec2(0.0);
    float sL = texture(tex_signal, uv + vec2(-px.x, 0.0)).r;
    float sR = texture(tex_signal, uv + vec2(px.x, 0.0)).r;
    float sU = texture(tex_signal, uv + vec2(0.0, -px.y)).r;
    float sD = texture(tex_signal, uv + vec2(0.0, px.y)).r;
    signalGrad = vec2(sR - sL, sD - sU);
    
    // Movement logic
    float force_flow = 4.0 * max(0.2, g_flow); 
    float chemotaxis_strength = (g_perception - 0.5) * 1.5; // Reduced from 10.0
    
    // === REPULSION SCAN ===
    vec2 repulsion = vec2(0.0);
    float damage = 0.0;
    
    if (myMass > 0.01) {
        for(int y=-2; y<=2; y++) {
            for(int x=-2; x<=2; x++) {
                if(x==0 && y==0) continue;
                vec2 off = vec2(float(x), float(y));
                vec2 nUV = uv + off * px;
                
                vec4 nVal = texture(tex_state, nUV);
                if (nVal.r > 0.05) {
                    vec2 nAff_lam = unpack2(texture(tex_genome, nUV).b);
                    float nAff = nAff_lam.x;
                    if (abs(nAff - myAffinity) > p.u_identity_thr) {
                        float dist = length(off);
                        vec2 dir = off / dist;
                        repulsion -= dir * nVal.r;
                        damage += nVal.r * p.u_combat_damage;
                    }
                }
            }
        }
    }
    
    vec2 totalVelocity = (gradU * force_flow) + 
                         (repulsion * p.u_repulsion_strength) +
                         (signalGrad * chemotaxis_strength);
    
    // === MASS ADVECTION (Conservative PUSH) ===
    vec2 targetUV = uv + totalVelocity * p.u_dt * px;
    vec2 targetPx = targetUV * p.u_res - 0.5; 
    
    float survivingMass = myMass - (damage * p.u_dt) - (p.u_base_decay * p.u_dt); 
    survivingMass = max(0.0, survivingMass);
    
    if (survivingMass > 0.0) {
        ivec2 tl = ivec2(floor(targetPx));
        vec2 f = fract(targetPx);
        
        float w00 = (1.0-f.x)*(1.0-f.y);
        float w10 = f.x*(1.0-f.y);
        float w01 = (1.0-f.x)*f.y;
        float w11 = f.x*f.y;
        
        uint m_int = uint(survivingMass * MASS_SCALE);
        ivec2 res = ivec2(p.u_res);
        
        ivec2 p00 = (tl + res) % res; 
        if (w00 > 0.0) imageAtomicAdd(img_mass_accum, p00, uint(float(m_int) * w00));
        
        ivec2 p10 = (ivec2(tl.x+1, tl.y) + res) % res;
        if (w10 > 0.0) imageAtomicAdd(img_mass_accum, p10, uint(float(m_int) * w10));
        
        ivec2 p01 = (ivec2(tl.x, tl.y+1) + res) % res;
        if (w01 > 0.0) imageAtomicAdd(img_mass_accum, p01, uint(float(m_int) * w01));
        
        ivec2 p11 = (ivec2(tl.x+1, tl.y+1) + res) % res;
        if (w11 > 0.0) imageAtomicAdd(img_mass_accum, p11, uint(float(m_int) * w11));
    }
    
    // === GENOME ADVECTION (Semi-Lagrangian PULL) ===
    vec2 sourceUV = uv - totalVelocity * p.u_dt * px;
    vec4 nextGenome = texture(tex_genome, sourceUV);
    
    vec4 srcGenome = nextGenome;
    float srcAffinity = unpack2(srcGenome.b).x;
    
    // Identity Defense
    if (myMass > p.u_colonize_thr) {
        if (abs(myAffinity - srcAffinity) > p.u_identity_thr) {
             nextGenome = myGenome;
        }
    }
    
    // Mutation (supports 8 genes naturally now)
    if (hash(uv + vec2(p.u_seed, p.u_dt)) < p.u_mutation_rate) {
        float drift = (hash(uv * 2.0 + p.u_seed) - 0.5) * 0.1;
        
        // Mutate packed channels
        for(int i=0; i<4; i++) {
            vec2 genes = unpack2(nextGenome[i]);
            genes.x = clamp(genes.x + drift, 0.0, 1.0);
            genes.y = clamp(genes.y + drift * 0.5, 0.0, 1.0);
            nextGenome[i] = pack2(genes.x, genes.y);
        }
    }
    
    imageStore(img_new_genome, uv_i, nextGenome);
    imageStore(img_new_state, uv_i, vec4(0.0, totalVelocity, age + p.u_dt));
}

#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Params {
    vec2 u_res;
    vec2 u_mouse_world;
    
    float u_dt;
    float u_seed;
    float u_density;
    float u_init_grid;
    
    float u_R;
    float u_k_w1;
    float u_k_w2;
    float u_k_w3;
    
    float u_g_mu_base;
    float u_g_mu_range;
    float u_g_sigma;
    float u_force_flow;
    
    float u_force_rep;
    float u_decay;
    float u_eat_rate;
    float u_chemotaxis;
    
    float u_mutation_rate;
    float u_inertia;
    float u_brush_size;
    float u_brush_hue;
    
    float u_brush_mode;
    float u_show_waste;
    float u_mouse_click; // 1.0 or 0.0
    float _pad0;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_living;
layout(set = 0, binding = 2) uniform sampler2D tex_waste;
layout(set = 0, binding = 3) uniform sampler2D tex_potential;

layout(set = 0, binding = 4, rgba32f) uniform image2D img_new_living;
layout(set = 0, binding = 5, rgba32f) uniform image2D img_new_waste;

float hash(vec2 pt) {
    return fract(sin(dot(pt, vec2(12.9898, 78.233))) * 43758.5453);
}

float getEffectiveMu(float gene) { 
    return p.u_g_mu_base + gene * p.u_g_mu_range; 
}

float G(float u, float mu, float sigma) {
    float effMu = getEffectiveMu(mu);
    return 2.0 * exp(-0.5 * pow((u - effMu) / sigma, 2.0)) - 1.0;
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 px = 1.0 / p.u_res;
    vec2 uv = (vec2(uv_i) + 0.5) * px;

    // --- 1. ADVECTION (MOVEMENT) ---
    float newMass = 0.0;
    float winnerMass = -1.0;
    vec3 winnerTraits = vec3(0.0); // g=Struct, b=Diet, a=Sigma
    
    int searchR = 2;
    
    for(int y = -searchR; y <= searchR; y++) {
        for(int x = -searchR; x <= searchR; x++) {
            vec2 offset = vec2(float(x), float(y));
            vec2 srcUV = uv + offset * px;
            
            vec4 srcState = texture(tex_living, srcUV);
            float m = srcState.r;
            
            if(m > 0.001) {
                float K_C = texture(tex_potential, srcUV).r;
                float K_L = texture(tex_potential, srcUV - vec2(px.x, 0)).r; 
                float K_R = texture(tex_potential, srcUV + vec2(px.x, 0)).r;
                float K_B = texture(tex_potential, srcUV - vec2(0, px.y)).r; 
                float K_T = texture(tex_potential, srcUV + vec2(0, px.y)).r;
                
                float muStr = srcState.g; 
                float muDiet = srcState.b; 
                float effectiveSigma = p.u_g_sigma + (srcState.a * 0.02); 

                // Potential Gradient
                vec2 grad_U = vec2(
                    G(K_R, muStr, effectiveSigma) - G(K_L, muStr, effectiveSigma), 
                    G(K_T, muStr, effectiveSigma) - G(K_B, muStr, effectiveSigma)
                ) * 0.5;
                
                float M_L = texture(tex_living, srcUV - vec2(px.x, 0)).r; 
                float M_R = texture(tex_living, srcUV + vec2(px.x, 0)).r;
                float M_B = texture(tex_living, srcUV - vec2(0, px.y)).r; 
                float M_T = texture(tex_living, srcUV + vec2(0, px.y)).r;
                
                // Mass Gradient
                vec2 grad_M = vec2(M_R - M_L, M_T - M_B) * 0.5;

                vec2 grad_Food = vec2(0.0);
                if(p.u_chemotaxis > 0.0) {
                    vec4 wL = texture(tex_waste, srcUV - vec2(px.x, 0)); 
                    vec4 wR = texture(tex_waste, srcUV + vec2(px.x, 0));
                    vec4 wB = texture(tex_waste, srcUV - vec2(0, px.y)); 
                    vec4 wT = texture(tex_waste, srcUV + vec2(0, px.y));
                    
                    float tType = fract(muDiet);
                    float affL = (1.0-abs(wL.g-tType))*wL.r; 
                    float affR = (1.0-abs(wR.g-tType))*wR.r;
                    float affB = (1.0-abs(wB.g-tType))*wB.r; 
                    float affT = (1.0-abs(wT.g-tType))*wT.r;
                    grad_Food = vec2(affR - affL, affT - affB);
                }

                vec2 flow = grad_U * p.u_force_flow - grad_M * p.u_force_rep + grad_Food * p.u_chemotaxis;
                
                if(length(flow) > 4.0) flow = normalize(flow) * 4.0;

                vec2 dest = srcUV + flow * p.u_dt * px;
                vec2 distVec = (dest - uv) * p.u_res;
                
                float weight = max(0.0, 1.0 - abs(distVec.x)) * max(0.0, 1.0 - abs(distVec.y));
                
                if(weight > 0.0) {
                    float incomingMass = m * weight;
                    newMass += incomingMass;
                    if(incomingMass > winnerMass) {
                        winnerMass = incomingMass;
                        winnerTraits = srcState.gba;
                    }
                }
            }
        }
    }

    vec3 finalTraits = vec3(0.0, 0.0, 0.03);
    if(newMass > 0.001) finalTraits = winnerTraits;

    // Mutation
    if(newMass > 0.01 && p.u_mutation_rate > 0.0) {
        if(hash(uv + p.u_seed) < 0.2) {
             float drift = (hash(uv * 1.1 + p.u_seed) - 0.5) * p.u_mutation_rate;
             finalTraits.y = clamp(finalTraits.y + drift * 4.0, 0.0, 1.0); // Diet
             finalTraits.x = clamp(finalTraits.x + drift, 0.0, 1.0); // Struct
             finalTraits.z = clamp(finalTraits.z + drift * 0.5, 0.0, 1.0); // Sigma
        }
    }
    
    // Inertia
    vec4 oldState = texture(tex_living, uv);
    if(oldState.r > 0.01) finalTraits = mix(finalTraits, oldState.gba, p.u_inertia);

    // --- 2. WASTE MANAGEMENT ---
    vec4 wCenter = texture(tex_waste, uv);
    vec4 wL = texture(tex_waste, uv - vec2(px.x, 0)); 
    vec4 wR = texture(tex_waste, uv + vec2(px.x, 0));
    vec4 wT = texture(tex_waste, uv + vec2(0, px.y)); 
    vec4 wB = texture(tex_waste, uv - vec2(0, px.y));
    
    vec4 wasteDiffusion = (wL + wR + wT + wB) * 0.25 - wCenter;
    vec4 diffusedWaste = wCenter + wasteDiffusion * 0.5; // Fixed diffusion rate of 0.5 implied by prototype
    
    float wMass = max(0.0, diffusedWaste.r);
    float wType = diffusedWaste.g;

    // Digestion
    float targetType = fract(finalTraits.y);
    float enzyme = getEffectiveMu(targetType);
    float wasteStruct = getEffectiveMu(wType);
    float similarity = 1.0 - abs(enzyme - wasteStruct);
    float efficiency = smoothstep(0.5, 1.0, similarity);
    
    float eaten = 0.0;
    if(newMass > 0.001 && wMass > 0.001) {
        eaten = min(wMass, newMass * p.u_eat_rate * efficiency * p.u_dt);
        newMass += eaten;
        wMass -= eaten;
    }

    // Death
    float crowdPenalty = max(0.0, newMass - 1.2) * 1.5;
    float deathRate = p.u_decay + crowdPenalty;
    float deadMass = min(newMass, newMass * deathRate);
    
    newMass -= deadMass;
    float newWasteMass = wMass + deadMass;
    
    float newWasteType = wType;
    if(newWasteMass > 0.0001 && deadMass > 0.0) {
        newWasteType = mix(wType, finalTraits.x, deadMass / newWasteMass);
    }
    
    // Write outputs
    imageStore(img_new_living, uv_i, vec4(newMass, finalTraits));
    imageStore(img_new_waste, uv_i, vec4(newWasteMass, newWasteType, 0.0, 1.0));
}

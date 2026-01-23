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
    float u_mouse_click;
    float _pad0;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_living;
layout(set = 0, binding = 2, rgba32f) uniform image2D img_potential;

float gaussian(float r, float mu, float sigma) {
    return exp(-0.5 * ((r - mu) / sigma) * ((r - mu) / sigma));
}

float K(float r, float mu) {
    // Anillo 1: Interno (fijo)
    float k1 = gaussian(r, 0.15, 0.08) * p.u_k_w1;
    
    // Anillo 2: Cuerpo (variable con genética 'mu')
    // El radio se mueve entre 0.2 y 0.5 según la especie
    float k2 = gaussian(r, 0.2 + mu * 0.3, 0.12) * p.u_k_w2;
    
    // Anillo 3: Externo (fijo)
    float k3 = gaussian(r, 0.85, 0.08) * p.u_k_w3;
    
    // Mezclamos la morfología según la especie (mu)
    return mix(k1 * 0.7 + k2, k2 + k3 * 0.7, mu);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    
    // Read center Mu (Green channel)
    float centerMu = texture(tex_living, uv).g;
    
    float sum = 0.0;
    float totalWeight = 0.0;
    
    int radius = int(p.u_R);
    
    for(int y = -radius; y <= radius; y++) {
        for(int x = -radius; x <= radius; x++) {
            vec2 offset = vec2(float(x), float(y));
            float r = length(offset) / float(radius);
            
            if(r <= 1.0) {
                float w = K(r, centerMu);
                // Sample texture with offset. Texture is set to REPEAT in sampler, so simple addition works.
                vec2 sampleUV = uv + offset / p.u_res;
                sum += texture(tex_living, sampleUV).r * w;
                totalWeight += w;
            }
        }
    }
    
    if(totalWeight > 0.0) sum /= totalWeight;
    
    // Write Potential (Red channel)
    imageStore(img_potential, uv_i, vec4(sum, 0.0, 0.0, 1.0));
}

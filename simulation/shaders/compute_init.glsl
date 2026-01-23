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

layout(set = 0, binding = 1, rgba32f) uniform image2D img_living;
layout(set = 0, binding = 2, rgba32f) uniform image2D img_waste;

// Hash function from prototype
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    ivec2 uv_i = ivec2(gl_GlobalInvocationID.xy);
    if (uv_i.x >= int(p.u_res.x) || uv_i.y >= int(p.u_res.y)) return;
    
    vec2 uv = (vec2(uv_i) + 0.5) / p.u_res;
    vec2 coord = vec2(uv_i);
    
    float rnd = hash(coord + p.u_seed * 123.0);
    float mass = step(1.0 - p.u_density, rnd);
    
    if(mass > 0.0) {
        mass = 0.5 + 0.5 * hash(coord + p.u_seed * 99.0);
    }
    
    vec2 cellID = floor(uv * p.u_init_grid);
    float cellSeed = hash(cellID + p.u_seed * 77.7);
    
    float muStruct = cellSeed;
    float drift = (hash(cellID * 1.1) - 0.5) * 0.2;
    float muDiet = fract(muStruct + drift);
    float sig = 0.03;
    
    // Living: Mass, MuStruct, MuDiet, Sigma
    imageStore(img_living, uv_i, vec4(mass, muStruct, muDiet, sig));
    
    float wasteNoise = hash(uv * 10.0 + p.u_seed);
    // Waste: Mass, Type, 0, 1
    imageStore(img_waste, uv_i, vec4(wasteNoise * 0.2, hash(uv * 5.0), 0.0, 1.0));
}

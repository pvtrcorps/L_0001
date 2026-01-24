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
    float _pad1, _pad2, _pad3;
} p;

layout(set = 0, binding = 1) uniform sampler2D tex_state;
layout(set = 0, binding = 2) uniform sampler2D tex_genome;

// Analysis Buffer: 64x64 grid = 4096 cells
// Each cell stores average properties of a 16x16 pixel block
struct CellData {
    float mass;      // Average mass
    float mu;        // Average genes
    float sigma;
    float radius;
    float flow;
    float affinity;
    float lambda;
    float _pad;      // Align to 32 bytes (8 floats)
};

layout(set = 0, binding = 3, std430) buffer Analysis {
    CellData cells[4096];
} analysis;

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed);
    float a = float(bits >> 16) / 65535.0;
    float b = float(bits & 0xFFFFu) / 65535.0;
    return vec2(a, b);
}

void main() {
    // Dispatch size should be 64x64/8x8 = 8x8 groups? 
    // Wait, 64x64 output grid.
    // If local size is 8x8, we need 64/8 = 8 groups in X and Y.
    // Total global invocations = 64x64.
    
    ivec2 grid_pos = ivec2(gl_GlobalInvocationID.xy);
    if (grid_pos.x >= 64 || grid_pos.y >= 64) return;
    
    // Each grid cell covers a 16x16 pixel block in the 1024x1024 texture
    vec2 block_size = p.u_res / 64.0; // Should be 16.0
    vec2 base_uv = (vec2(grid_pos) * block_size) / p.u_res; // Normalized start UV
    vec2 px = 1.0 / p.u_res;
    
    float sum_mass = 0.0;
    
    // Weighted genes sums
    float sum_mu = 0.0;
    float sum_sigma = 0.0;
    float sum_radius = 0.0;
    float sum_flow = 0.0;
    float sum_affinity = 0.0;
    float sum_lambda = 0.0;
    
    // Iterate 16x16 pixels within the block
    int block_w = int(block_size.x);
    int block_h = int(block_size.y);
    
    // Optimization: sampling every pixel in 16x16 (256 samples) per thread is heavy.
    // Maybe sample 4x4 stride (16 samples)?
    // 256 samples is fine for a 10Hz analysis pass.
    
    int stride = 2; // Sample every 2nd pixel (8x8 = 64 samples) for speed
    
    for (int y = 0; y < block_h; y += stride) {
        for (int x = 0; x < block_w; x += stride) {
            vec2 offset = vec2(float(x) + 0.5, float(y) + 0.5) * px;
            vec2 uv = base_uv + offset;
            
            vec4 state = texture(tex_state, uv);
            float m = state.r;
            
            if (m > 0.001) {
                sum_mass += m;
                
                vec4 genome = texture(tex_genome, uv);
                vec2 rf = unpack2(genome.b);
                vec2 al = unpack2(genome.a);
                
                sum_mu += genome.r * m;
                sum_sigma += genome.g * m;
                sum_radius += rf.x * m;
                sum_flow += rf.y * m;
                sum_affinity += al.x * m;
                sum_lambda += al.y * m;
            }
        }
    }
    
    // Normalize
    uint idx = uint(grid_pos.y * 64 + grid_pos.x);
    
    if (sum_mass > 0.0) {
        analysis.cells[idx].mass = sum_mass / (64.0); // Avg mass per sampled pixel (64 samples)
        // Note: dividing by samples gives avg mass density.
        // sum_mass is weighted sum for genes.
        
        analysis.cells[idx].mu = sum_mu / sum_mass;
        analysis.cells[idx].sigma = sum_sigma / sum_mass;
        analysis.cells[idx].radius = sum_radius / sum_mass;
        analysis.cells[idx].flow = sum_flow / sum_mass;
        analysis.cells[idx].affinity = sum_affinity / sum_mass;
        analysis.cells[idx].lambda = sum_lambda / sum_mass;
    } else {
        analysis.cells[idx].mass = 0.0;
        analysis.cells[idx].mu = 0.0;
        analysis.cells[idx].sigma = 0.0;
        analysis.cells[idx].radius = 0.0;
        analysis.cells[idx].flow = 0.0;
        analysis.cells[idx].affinity = 0.0;
        analysis.cells[idx].lambda = 0.0;
    }
}

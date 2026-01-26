shader_type canvas_item;

uniform sampler2D tex_state : repeat_enable, filter_linear;
uniform sampler2D tex_genome : repeat_enable, filter_nearest;
uniform sampler2D tex_signal; // [NEW]
uniform vec2 camera_pos = vec2(0.0, 0.0);
uniform float camera_zoom = 1.0;
uniform bool u_show_select = false;
uniform vec4 u_select_vector = vec4(0.0);

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Unpack two [0,1] floats from packed representation
vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u; // Clear normalization bit
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

// Helper to get color from a specific integer grid coordinate
vec3 get_pixel_color(ivec2 coords, vec2 tex_size) {
    // Clamp coordinates
    coords = clamp(coords, ivec2(0), ivec2(tex_size) - 1);
    vec2 uv = (vec2(coords) + 0.5) / tex_size;
    
    vec4 genome = texture(tex_genome, uv);
    
    // Decode genes
    vec2 mu_sigma = unpack2(genome.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    
    vec2 affinity_lambda = unpack2(genome.b);
    float g_affinity = affinity_lambda.x;
    
    // Color mapping logic - UNIFIED with Signal Vector
    // Signal Vector was: vec3(g_mu, g_flow, g_affinity)
    vec2 radius_flow = unpack2(genome.g);
    float g_flow = radius_flow.y;
    
    // Direct RGB Mapping to match the "Scent"
    // R = Mu (Growth/Physiology)
    // G = Flow (Velocity/Movement)
    // B = Affinity (Structure/Social)
    vec3 col = vec3(g_mu, g_flow, g_affinity);
    
    // Boost saturation slightly so they don't look too dark/gray
    col = pow(col, vec3(0.8)); 
    col *= 1.2;
    
    return col;
}

void fragment() {
    // Calculate aspect ratio correction
    vec2 resolution = 1.0 / SCREEN_PIXEL_SIZE;
    float aspect = resolution.x / resolution.y;
    
    // Apply camera transform
    vec2 uv = UV;
    uv -= 0.5;
    
    // Correct aspect ratio to keep simulation square
    if (aspect > 1.0) {
        uv.x *= aspect;
    } else {
        uv.y /= aspect;
    }
    
    // Apply zoom and pan
    uv /= camera_zoom;
    uv += camera_pos;
    uv += 0.5;
    
    // Vignette
    float vignette = 1.0;
    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        vignette = 0.1;
    }
    
    vec4 state = texture(tex_state, uv);
    float mass = state.r;
    vec3 signal = texture(tex_signal, uv).rgb;
    
    // Manual Bilinear Interpolation for Genome Color
    vec2 tex_size = vec2(textureSize(tex_genome, 0));
    vec2 pixel_pos = uv * tex_size - 0.5;
    ivec2 i_pos = ivec2(floor(pixel_pos));
    vec2 f_pos = fract(pixel_pos);
    
    vec3 c00 = get_pixel_color(i_pos, tex_size);
    vec3 c10 = get_pixel_color(i_pos + ivec2(1, 0), tex_size);
    vec3 c01 = get_pixel_color(i_pos + ivec2(0, 1), tex_size);
    vec3 c11 = get_pixel_color(i_pos + ivec2(1, 1), tex_size);
    
    vec3 top = mix(c00, c10, f_pos.x);
    vec3 bottom = mix(c01, c11, f_pos.x);
    vec3 col_base = mix(top, bottom, f_pos.y);
    
    // Apply Mass and Brightness
    float brightness = mass * 5.0; 
    vec3 col = col_base * clamp(brightness, 0.0, 1.0);
    col *= smoothstep(0.0, 0.05, mass);
    
    // Signal layer visualization (Full RGB)
    // Add signal using LIGHTEN blend mode (max) instead of Additive
    vec3 signal_col = signal * 0.5; // Boost slightly as Lighten doesn't accumulate brightness
    col = max(col, signal_col);
    
    // Highlight Logic
    // Re-sample central genome for selection logic (Nearest Neighbor)
    vec4 genome = texture(tex_genome, uv);
    vec2 mu_sigma = unpack2(genome.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    vec2 affinity_lambda = unpack2(genome.b);
    float g_affinity = affinity_lambda.x;

    if (u_show_select) {
        vec2 radius_flow = unpack2(genome.g);
        float g_radius = radius_flow.x;
        
        float d = 0.0;
        d += abs(g_mu - u_select_vector.x) * 1.5;
        d += abs(g_sigma - u_select_vector.y) * 1.5;
        d += abs(g_radius - u_select_vector.z) * 1.0;
        d += abs(g_affinity - u_select_vector.w) * 1.0;
        
        if (d < 0.15) { // Keeping slightly higher in shader for smoother visual "neighborhood" effect
             float pulse = 0.5 + 0.5 * sin(TIME * 6.0);
             col = mix(col, vec3(1.0, 1.0, 1.0), 0.4 * pulse);
        } else {
             col *= 0.4; // Dim others
             col = vec3(dot(col, vec3(0.3, 0.59, 0.11))); 
        }
    }

    col *= vignette;
    
    // Tone mapping and gamma
    col = col / (1.0 + col);
    col = pow(col, vec3(0.85));
    
    if (mass < 0.01 && length(signal) < 0.05) {
        col = vec3(0.0);
    }
    
    COLOR = vec4(col, 1.0);
}

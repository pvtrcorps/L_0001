shader_type canvas_item;

uniform sampler2D tex_state : repeat_enable, filter_linear;
uniform sampler2D tex_genome : repeat_enable, filter_linear;
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
    uint bits = floatBitsToUint(packed);
    float a = float(bits >> 16u) / 65535.0;
    float b = float(bits & 0xFFFFu) / 65535.0;
    return vec2(a, b);
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
    
    // Darken out-of-bounds areas
    float vignette = 1.0;
    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        vignette = 0.1;
    }
    
    vec4 state = texture(tex_state, uv);
    vec4 genome = texture(tex_genome, uv);
    
    float mass = state.r;
    
    // Decode genes for visualization
    float g_mu = genome.r;
    float g_sigma = genome.g;
    vec2 radius_flow = unpack2(genome.b);
    float g_radius = radius_flow.x;
    vec2 affinity_lambda = unpack2(genome.a);
    float g_affinity = affinity_lambda.x;
    
    // Color mapping:
    // Hue = mu gene (species identity) with slight affinity shift for variety
    // Saturation = sigma gene (specialist=vivid, generalist=pastel)
    // Value = ONLY mass (so all species are equally visible)
    float hue = g_mu + g_affinity * 0.15; // Slight shift based on affinity
    float saturation = 0.5 + g_sigma * 0.5; // [0.5, 1.0]
    float brightness = 1.0 - exp(-mass * 2.0); // Pure mass-based, soft curve
    
    vec3 col = hsv2rgb(vec3(hue, saturation, clamp(brightness, 0.0, 1.0)));
    
    // Smoothly fade dead areas (but don't make living cells transparent)
    col *= smoothstep(0.0, 0.05, mass);
    
    // Highlight Logic
    if (u_show_select) {
        float d_mu = abs(g_mu - u_select_vector.x);
        float d_sigma = abs(g_sigma - u_select_vector.y);
        // Using gene similarity threshold
        if (d_mu + d_sigma < 0.15) {
             // Selected Species: Pulse
             float pulse = 0.5 + 0.5 * sin(TIME * 6.0);
             col = mix(col, vec3(1.0, 1.0, 1.0), 0.4 * pulse);
        } else {
             // Other Species: Dim
             col *= 0.2;
             col = vec3(dot(col, vec3(0.3, 0.59, 0.11))); // Grayscale
        }
    }

    // Apply vignette
    col *= vignette;
    
    // Tone mapping and gamma
    col = col / (1.0 + col);
    col = pow(col, vec3(0.85));
    
    // Strict Cleanup for latent species visibility
    if (mass < 0.01) {
        col = vec3(0.0);
    }
    
    COLOR = vec4(col, 1.0);
}

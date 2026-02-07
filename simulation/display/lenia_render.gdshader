shader_type canvas_item;

uniform sampler2D tex_state : repeat_enable, filter_linear;
uniform sampler2D tex_genome : repeat_enable, filter_linear;
uniform sampler2D tex_genome_ext : repeat_enable, filter_linear; // [NEW] Valid data
uniform sampler2D tex_signal : repeat_enable, filter_linear; // [NEW]
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
    // Decode genes matching compute_init.glsl packing
    // A: (mu, sigma) -> Stored in R
    vec2 mu_sigma = unpack2(genome.r);
    float g_mu = mu_sigma.x;
    float g_sigma = mu_sigma.y;
    
    // B: (radius, viscosity) -> Stored in G
    vec2 rad_visc = unpack2(genome.g);
    float g_flow = rad_visc.y; // Viscosity as proxy for flow properties
    float g_radius = rad_visc.x;
    
    // Affinity is not physically stored in genome currently (Phantom Gene)
    // We use a neutral value or we could map sigma to it for visualization
    float g_affinity = 0.5; 
    
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
    

    
    // 4. Calculate 3D Normals from the Mass Gradient
    vec2 gradient = vec2(dFdx(mass), dFdy(mass));
    vec3 normal = normalize(vec3(-gradient * 50.0, 1.0 + mass * 10.0));
    
    // === SPECTRAL IDENTITY ===
    // 1. Get Genes
    vec4 g_ext = texture(tex_genome_ext, uv);
    vec2 hues = unpack2(g_ext.a);
    float h_emit = hues.x; // Identity
    float h_detect = hues.y; // Target
    
    // 2. Core Color (Who I am)
    // Saturation Logic: 
    // If I like myself (Diff ~ 0), I am Pure (Sat 1.0). 
    // If I like others, I might be slightly desaturated? 
    // User requested "Detection Hue Difference" mapping for Saturation.
    // Let's try: Sat = 0.5 + 0.5 * (1.0 - 2.0 * abs(h_emit - h_detect)); // Simple mapping?
    // Actually, let's keep it vibrant for now (Sat 1.0) to ensure the Hues are legible.
    // The "Fresnel" will do the heavy lifting of showing the difference.
    vec3 col_core = hsv2rgb(vec3(h_emit, 0.9, 1.0));
    
    // 3. Detection Aura (Who I want)
    vec3 col_detect = hsv2rgb(vec3(h_detect, 1.0, 1.0));
    
    // 4. Fresnel / Rim Mixing
    // View vector is effectively (0,0,1) in 2D ortho, but normal tilts.
    // Fresnel = (1 - N.V)^k
    float fresnel = pow(1.0 - normal.z, 2.0); // Power 3.0 was a bit sharp, 2.0 shows more aura
    
    // 5. Synthesis
    // Core body is Identity. Edges are Desire.
    vec3 col_base = mix(col_core, col_detect, fresnel * 0.8);
    
    // Enhance "Glow" of the aura
    col_base += col_detect * fresnel * 0.5 * mass;

    // Apply Mass Brightness (Organic gel look)
    float brightness = mass * 2.0; 
    vec3 col = col_base * brightness;

    // 6. Lighting (Specular)
    vec3 light_dir = normalize(vec3(-0.5, -0.5, 1.0));
    float specular = pow(max(0.0, dot(reflect(-light_dir, normal), vec3(0,0,1))), 15.0);
    col += vec3(1.0) * specular * 0.8 * smoothstep(0.1, 0.3, mass);
    
    // Signal layer visualization (Full RGB)
    // Additive blending for glow
    vec3 signal_col = signal * 0.1; 
    col += signal_col;

    // Highlight Logic
    // Re-sample central genome for selection logic (Nearest Neighbor) is done via `g_ext` check
    
    if (u_show_select) {
         // Check match against u_select_vector (using unpacked genes if we passed them?)
         // Currently u_select_vector passes (mu, sigma, radius, affinity).
         // We might need to update Inspector to pass Hues if we want to filter by Hue.
         // For now, let's keep the logic but maybe we should filter by Hue?
         // Let's assume u_select_vector is still old genes for now.
         // Actually, let's skip the complex filter update for this step to focus on Rendering.
         
         // Using the OLD genes for selection filter for now as they are still reliable for "Species ID"
         // (Mu/Sigma are still the primary species differentiators physically).
         vec4 genome = texture(tex_genome, uv);
         vec2 mu_sigma = unpack2(genome.r);
         vec2 rad_visc = unpack2(genome.g);
         vec2 aff_rep = unpack2(g_ext.r);
         
         float d = 0.0;
         d += abs(mu_sigma.x - u_select_vector.x) * 1.5;
         d += abs(mu_sigma.y - u_select_vector.y) * 1.5;
         d += abs(rad_visc.x - u_select_vector.z) * 1.0;
         d += abs(aff_rep.x - u_select_vector.w) * 1.0;
         
         if (d < 0.04) {
             float pulse = 0.5 + 0.5 * sin(TIME * 6.0);
             if (mass > 0.0001) {
                 col = col_base * 1.5; 
                 // Show pure Core Color for clarity during selection?
                 col = mix(col_core * 2.0, vec3(1.0), 0.2 * pulse);
                 col *= mass * 4.0;
             } else {
                 col = vec3(0.0);
             }
         } else {
             col *= 0.1;
             col = vec3(dot(col, vec3(0.3, 0.59, 0.11))); 
         }
    }

    col *= vignette;
    
    // Tone mapping and gamma
    // Reinhard tone mapping
    col = col / (1.0 + col);
    // Higher gamma (close to 1.0) = Softer contrast. 
    // Lower gamma (< 1.0) = Higher contrast.
    col = pow(col, vec3(0.9));
    
    //if (mass < 0.01 && length(signal) < 0.01) {
    //    col = vec3(0.0);
    //}
    
    COLOR = vec4(col, 1.0);
}

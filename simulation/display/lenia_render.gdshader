shader_type canvas_item;

uniform sampler2D tex_state : repeat_enable, filter_linear;
uniform sampler2D tex_genome : repeat_enable, filter_linear;
uniform sampler2D tex_genome_ext : repeat_enable, filter_linear;
uniform sampler2D tex_signal : repeat_enable, filter_linear;
uniform vec2 camera_pos = vec2(0.0, 0.0);
uniform float camera_zoom = 1.0;
uniform bool u_show_select = false;
uniform vec4 u_select_vector = vec4(0.0);

const float PI = 3.14159265359;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 unpack2(float packed) {
    uint bits = floatBitsToUint(packed) & ~0x40000000u;
    float a = float((bits >> 15u) & 0x7FFFu) / 32767.0;
    float b = float(bits & 0x7FFFu) / 32767.0;
    return vec2(a, b);
}

void fragment() {
    vec2 resolution = 1.0 / SCREEN_PIXEL_SIZE;
    float aspect = resolution.x / resolution.y;

    vec2 uv = UV;
    uv -= 0.5;

    if (aspect > 1.0) {
        uv.x *= aspect;
    } else {
        uv.y /= aspect;
    }

    uv /= camera_zoom;
    uv += camera_pos;
    uv += 0.5;

    float vignette = 1.0;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        vignette = 0.1;
    }

    vec4 state = texture(tex_state, uv);
    float mass = state.r;
    vec3 signal = texture(tex_signal, uv).rgb;

    vec4 g1 = texture(tex_genome, uv);
    vec4 g2 = texture(tex_genome_ext, uv);

    vec2 shape_c_aniso = unpack2(g1.a);
    float anisotropy = shape_c_aniso.y;

    vec2 comp_rep = unpack2(g2.r);
    float compactness = comp_rep.x;

    vec2 plast_mob = unpack2(g2.g);
    float plasticity = plast_mob.x;

    vec2 hues = unpack2(g2.a);
    float h_emit = hues.x;
    float h_detect = hues.y;

    vec2 gradient = vec2(dFdx(mass), dFdy(mass));
    vec3 normal = normalize(vec3(-gradient * 50.0, 1.0 + mass * 10.0));

    vec3 col_core = hsv2rgb(vec3(h_emit, 0.9, 1.0));
    vec3 col_detect = hsv2rgb(vec3(h_detect, 1.0, 1.0));

    float fresnel = pow(1.0 - normal.z, mix(2.3, 1.3, anisotropy));
    vec3 col_base = mix(col_core, col_detect, fresnel * 0.8);

    // Morphology tint: compact forms are warmer, plastic forms are cooler.
    vec3 morph_tint = mix(vec3(0.8, 0.9, 1.2), vec3(1.2, 1.0, 0.8), compactness);
    col_base *= mix(vec3(1.0), morph_tint, 0.25 + 0.25 * plasticity);

    float brightness = mass * (1.8 + 0.4 * anisotropy);
    vec3 col = col_base * brightness;

    vec3 light_dir = normalize(vec3(-0.5, -0.5, 1.0));
    float specular = pow(max(0.0, dot(reflect(-light_dir, normal), vec3(0, 0, 1))), 15.0);
    col += vec3(1.0) * specular * 0.8 * smoothstep(0.1, 0.3, mass);

    // Polarity stripe
    float angle = state.a * (2.0 * PI);
    vec2 axis = vec2(cos(angle), sin(angle));
    vec2 p = (uv - floor(uv * 64.0) / 64.0 - 0.5) * 2.0;
    float stripe = 0.5 + 0.5 * sin(dot(p, axis) * 8.0);
    col += col_detect * stripe * 0.15 * anisotropy * mass;

    col += signal * 0.1;

    if (u_show_select) {
        vec2 mu_sigma = unpack2(g1.r);
        vec2 rad_visc = unpack2(g1.g);
        vec2 ext_r = unpack2(g2.r);

        float d = 0.0;
        d += abs(mu_sigma.x - u_select_vector.x) * 1.5;
        d += abs(mu_sigma.y - u_select_vector.y) * 1.5;
        d += abs(rad_visc.x - u_select_vector.z) * 1.0;
        d += abs(ext_r.y - u_select_vector.w) * 1.0;

        if (d < 0.04) {
            float pulse = 0.5 + 0.5 * sin(TIME * 6.0);
            if (mass > 0.0001) {
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
    col = col / (1.0 + col);
    col = pow(col, vec3(0.9));

    COLOR = vec4(col, 1.0);
}

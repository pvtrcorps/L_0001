shader_type canvas_item;

uniform sampler2D tex_living : repeat_enable, filter_linear;
uniform sampler2D tex_waste : repeat_enable, filter_linear;
uniform float show_waste : hint_range(0.0, 1.0) = 1.0;
uniform vec2 camera_pos = vec2(0.0, 0.0);
uniform float camera_zoom = 1.0;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
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
    
    vec4 lData = texture(tex_living, uv);
    vec4 wData = texture(tex_waste, uv);
    
    float lMass = lData.r;
    float lStruct = lData.g;
    float lDiet = lData.b;
    
    float wMass = wData.r;
    float wType = wData.g;
    
    float saturation = 0.1 + lDiet * 0.9;
    float visibleMass = 1.0 - exp(-lMass * 1.5);
    vec3 livingCol = hsv2rgb(vec3(lStruct, saturation, visibleMass));
    
    vec3 wasteCol = vec3(0.0);
    if(show_waste > 0.5) {
        float visibleWaste = 1.0 - exp(-wMass * 0.8);
        wasteCol = hsv2rgb(vec3(wType, 0.3, visibleWaste * 0.5));
    }
    
    vec3 col = (livingCol + wasteCol) * vignette;
    col = col / (1.0 + col); // Tone mapping
    col = pow(col, vec3(0.8)); // Gamma
    
    COLOR = vec4(col, 1.0);
}

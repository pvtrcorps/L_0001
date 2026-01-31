shader_type canvas_item;

uniform float shape_a = 0.5;
uniform float shape_b = 1.0;
uniform float shape_c = 0.5;
uniform float ring_width = 0.5;

float gaussian(float x, float mu, float sigma) {
	float d = (x - mu) / max(sigma, 0.001);
	return exp(-0.5 * d * d);
}

void fragment() {
	vec2 uv = UV - 0.5;
	float r = length(uv) * 2.0; // Normalized Radius (0 at center, 1 at edge)
	
	// Kernel Logic mirrored from compute_convolution.glsl
	
	// Weights
	float b1 = 0.1 + shape_a * 0.9;
	float b3 = 0.1 + (1.0 - shape_a) * 0.9;
	float b2 = shape_b;
	
	// Positions
	float a1 = 0.15;
	float a2 = 0.35 + shape_c * 0.3;
	float a3 = 0.85;
	
	// Widths (Modulated by Ring Width)
	// Base: 0.15, 0.20, 0.15
	// Modulator: (0.5 + ring_width * 1.5) -> [0.5, 2.0]x Base
	float w_mod = 0.5 + ring_width * 1.5;
	float w1 = 0.15 * w_mod;
	float w2 = 0.20 * w_mod;
	float w3 = 0.15 * w_mod;
	
	float k = 0.0;
	if (r < 1.0) {
		k += b1 * gaussian(r, a1, w1);
		k += b2 * gaussian(r, a2, w2);
		k += b3 * gaussian(r, a3, w3);
	}
	
	// Visual Mapping (Heatmap-ish)
	// Map intensity to a nice color ramp
	vec3 col = vec3(0.0);
	
	// Simple Cyan/Green/Yellow ramp
	col = mix(vec3(0.0, 0.0, 0.2), vec3(0.0, 1.0, 1.0), smoothstep(0.0, 0.5, k));
	col = mix(col, vec3(1.0, 1.0, 0.0), smoothstep(0.5, 1.0, k));
	
	// Clamp just in case
	col = clamp(col, 0.0, 1.0);
	
	COLOR = vec4(col, 1.0);
}

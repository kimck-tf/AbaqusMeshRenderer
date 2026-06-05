#version 330

in vec3 v_normal;
in vec3 v_color;

uniform vec3 light_dir;     // direction TO the light, in view space
uniform float ambient;
uniform bool use_flat_color; // wireframe pass draws a single flat colour
uniform vec3 flat_color;

out vec4 f_color;

void main() {
    if (use_flat_color) {
        f_color = vec4(flat_color, 1.0);
        return;
    }
    vec3 n = normalize(v_normal);
    vec3 l = normalize(light_dir);
    // Two-sided lighting so exterior faces read well from any angle.
    float diff = max(abs(dot(n, l)), 0.0);
    float intensity = ambient + (1.0 - ambient) * diff;
    f_color = vec4(v_color * intensity, 1.0);
}

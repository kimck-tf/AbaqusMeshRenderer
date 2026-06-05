#version 330

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

uniform mat4 mvp;
uniform mat3 normal_matrix;

out vec3 v_normal;
out vec3 v_color;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_normal = normal_matrix * in_normal;
    v_color = in_color;
}

#version 430 core
out vec4 FragColor;

void main()
{
    vec2 p = gl_PointCoord * 2.0 - 1.0;   // -1..1 across the point
    if (dot(p, p) > 1.0)
        discard;                          // round, not square
    FragColor = vec4(1.0);
}

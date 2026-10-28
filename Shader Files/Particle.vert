#version 430 core
// One point per particle, read straight from the simulation's buffers.
layout (location = 0) in vec4 aPosMass;     // xyz = position, w = mass
layout (location = 1) in vec4 aVelRadius;   // xyz = velocity, w = radius

uniform mat4 view;
uniform mat4 projection;
uniform float viewportHeight;   // in pixels

void main()
{
    vec4 viewPos = view * vec4(aPosMass.xyz, 1.0);
    gl_Position = projection * viewPos;

    // size on screen shrinks with distance, like a real sphere: projection[1][1] = 1 / tan(fov / 2)
    float pixelsPerUnit = projection[1][1] * viewportHeight * 0.5 / max(-viewPos.z, 0.001);
    gl_PointSize = max(2.0 * aVelRadius.w * pixelsPerUnit, 1.0);   // never below 1 px, so far-away particles don't vanish
}

#ifndef BARNES_HUT_H
#define BARNES_HUT_H

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "compute_shader.h"

// ---------------------------------------------------------------------------------------------
// N-body gravity on the GPU: Barnes-Hut force calculation + velocity Verlet integration, all in
// compute shaders. Particle data never leaves the GPU; the same buffers are drawn as points.
//
// Each physics step:
//   1. Integrate.comp (stage 0)  x += v dt + 1/2 a dt^2,  v += 1/2 a dt
//   2. Bounds.comp               bounding box of all particles
//   3. Morton.comp               a Z-order (Morton) code per particle: its octree cell, as bits
//   4. SortLocal/SortGlobal.comp bitonic sort of the particles by Morton code
//   5. BuildTree.comp            Barnes-Hut tree from the sorted codes
//   6. TreeMass.comp             each tree node's mass, centre of mass and box, leaves to root,
//                                plus a compact copy of each node for the force walk
//   7. Force.comp                walk the tree (64 neighbouring particles at a time): F = G * m1 * m2 / r^2, plus the halo
//   8. Integrate.comp (stage 1)  v += 1/2 a dt
// ---------------------------------------------------------------------------------------------

// One particle as stored on the GPU: two vec4s, matching the std430 layout in the shaders.
struct Particle
{
    glm::vec3 position;
    float mass;
    glm::vec3 velocity;
    float radius;   // only used for drawing
};

const float G = 1.0f;           // gravitational constant, in simulation units
const float THETA = 0.7f;       // Barnes-Hut opening angle: 0 = exact (every pair), bigger = faster but rougher
const float SOFTENING = 0.5f;   // distance below which gravity stops growing like 1/r^2 (see Force.comp)

// Invisible dark matter halo: a fixed, round cloud of mass centred on the origin that pulls on every
// particle but is never drawn and never moves. Its density falls off as 1 / (1 + (r / core)^2)
// (a "pseudo-isothermal" halo), so the mass inside radius r keeps growing roughly in step with r,
// which makes orbital speeds level off far from the centre the way they do in real galaxies.
// By the shell theorem only the halo mass inside a particle's radius pulls on it:
//   F = G * m * M_halo(< r) / r^2, towards the centre
struct DarkMatterHalo
{
    float massScale = 0.0f;    // 4 * pi * (central density) * core^3; 0 = no halo
    float coreRadius = 1.0f;   // inside this the density is roughly constant

    // a halo with `mass` of dark matter inside `radius`
    static DarkMatterHalo withMassInside(float mass, float radius, float coreRadius)
    {
        return { mass / shape(radius / coreRadius), coreRadius };
    }

    // dark matter mass inside radius r
    float massInside(float r) const
    {
        return massScale * shape(r / coreRadius);
    }

    // x - atan(x): how the enclosed mass grows with x = r / core (same formula as Force.comp)
    static float shape(float x)
    {
        if (x < 0.1f)   // near the centre: series form, avoids float round-off
            return x * x * x * (1.0f / 3.0f - x * x * (0.2f - x * x / 7.0f));
        return x - std::atan(x);
    }
};

class BarnesHut
{
public:
    float theta = THETA;
    DarkMatterHalo halo;

    explicit BarnesHut(const std::vector<Particle>& particles, DarkMatterHalo darkMatterHalo = DarkMatterHalo())
        : halo(darkMatterHalo),
          n((unsigned int)particles.size()),
          integrateShader("Shader Files/Integrate.comp"),
          boundsShader("Shader Files/Bounds.comp"),
          mortonShader("Shader Files/Morton.comp"),
          sortLocalShader("Shader Files/SortLocal.comp"),
          sortGlobalShader("Shader Files/SortGlobal.comp"),
          buildTreeShader("Shader Files/BuildTree.comp"),
          treeMassShader("Shader Files/TreeMass.comp"),
          forceShader("Shader Files/Force.comp")
    {
        if (n < 2)
            std::cout << "BarnesHut needs at least 2 particles" << std::endl;

        // the sort works on a power-of-two number of keys, at least one full sort block
        paddedCount = SORT_BLOCK;
        while (paddedCount < n)
            paddedCount *= 2;

        std::vector<glm::vec4> positions(n), velocities(n);
        for (unsigned int i = 0; i < n; ++i)
        {
            positions[i] = glm::vec4(particles[i].position, particles[i].mass);
            velocities[i] = glm::vec4(particles[i].velocity, particles[i].radius);
        }

        positionBuffer     = createBuffer(n * sizeof(glm::vec4), positions.data());
        velocityBuffer     = createBuffer(n * sizeof(glm::vec4), velocities.data());
        accelerationBuffer = createBuffer(n * sizeof(glm::vec4), nullptr);
        keyBuffer          = createBuffer(paddedCount * sizeof(unsigned int), nullptr);
        idBuffer           = createBuffer(paddedCount * sizeof(unsigned int), nullptr);
        nodeBuffer         = createBuffer((2 * n - 1) * NODE_SIZE, nullptr);
        forceNodeBuffer    = createBuffer((2 * n - 1) * FORCE_NODE_SIZE, nullptr);
        boundsBuffer       = createBuffer(6 * sizeof(int), nullptr);

        // drawing: the position and velocity buffers double as vertex attributes, one point per particle
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);   // xyz = position, w = mass
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, velocityBuffer);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);   // xyz = velocity, w = radius
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);

        glGenQueries(TIMER_STAMPS, timerQueries);
        computeAccelerations();   // velocity Verlet needs a(t) before the first step
    }

    unsigned int count() const { return n; }

    // Advances every particle by dt with velocity Verlet:
    //   x(t+dt) = x(t) + v(t) dt + 1/2 a(t) dt^2
    //   a(t+dt) = F(x(t+dt)) / m           (Barnes-Hut)
    //   v(t+dt) = v(t) + 1/2 (a(t) + a(t+dt)) dt
    void step(float dt)
    {
        timingThisStep = timingRequested;
        timingRequested = false;
        stamp(0);

        bindBuffers();
        integrateShader.use();
        integrateShader.setUInt("numParticles", n);
        integrateShader.setFloat("dt", dt);
        integrateShader.setInt("stage", 0);
        integrateShader.dispatch(n, GROUP_SIZE);
        barrier();
        stamp(1);

        computeAccelerations();

        integrateShader.use();
        integrateShader.setInt("stage", 1);
        integrateShader.dispatch(n, GROUP_SIZE);
        barrier();
        stamp(6);

        if (timingThisStep)
            finishTiming();
        timingThisStep = false;
    }

    // GPU timing, to see which stage is slow: call requestTiming(), and after the next step()
    // takeTimingReport() returns how long that step took on the GPU, split by stage ("" if none waiting).
    void requestTiming() { timingRequested = true; }
    std::string takeTimingReport()
    {
        std::string report = timingReport;
        timingReport.clear();
        return report;
    }

    // Binds the particles for drawing; follow with glDrawArrays(GL_POINTS, 0, count()).
    void bindForDrawing() const
    {
        glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);   // finish the compute writes first
        glBindVertexArray(vao);
    }

    // Recomputes every particle's acceleration from the current positions (rebuilds the tree).
    void computeAccelerations()
    {
        bindBuffers();

        // 1. bounding box: reset to "empty", then every particle widens it
        int emptyBox[6] = { INT_MAX, INT_MAX, INT_MAX, INT_MIN, INT_MIN, INT_MIN };
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, boundsBuffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(emptyBox), emptyBox);
        boundsShader.use();
        boundsShader.setUInt("numParticles", n);
        boundsShader.dispatch(n, GROUP_SIZE);
        barrier();

        // 2. Morton code per particle
        mortonShader.use();
        mortonShader.setUInt("numParticles", n);
        mortonShader.setUInt("paddedCount", paddedCount);
        mortonShader.dispatch(paddedCount, GROUP_SIZE);
        barrier();
        stamp(2);

        // 3. sort particles by Morton code
        sortByKey();
        stamp(3);

        // 4. tree structure: one thread per internal node
        buildTreeShader.use();
        buildTreeShader.setUInt("numParticles", n);
        buildTreeShader.dispatch(n - 1, GROUP_SIZE);
        barrier();

        // 5. node masses, centres of mass and boxes: one thread per leaf, climbing to the root
        treeMassShader.use();
        treeMassShader.setUInt("numParticles", n);
        treeMassShader.setFloat("theta", theta);
        treeMassShader.dispatch(n, GROUP_SIZE);
        barrier();
        stamp(4);

        // 6. gravity: one thread per particle, each group of 64 neighbours walking the tree together
        forceShader.use();
        forceShader.setUInt("numParticles", n);
        forceShader.setFloat("G", G);
        forceShader.setFloat("softening", SOFTENING);
        forceShader.setFloat("haloMassScale", halo.massScale);
        forceShader.setFloat("haloCoreRadius", halo.coreRadius);
        forceShader.dispatch(n, FORCE_GROUP_SIZE);
        barrier();
        stamp(5);
    }

    // Buffer handles, e.g. for reading results back while debugging
    GLuint positionBuffer = 0, velocityBuffer = 0, accelerationBuffer = 0;
    GLuint keyBuffer = 0, idBuffer = 0, nodeBuffer = 0, forceNodeBuffer = 0, boundsBuffer = 0;

private:
    static const unsigned int GROUP_SIZE = 256;   // local_size_x of most shaders
    static const unsigned int FORCE_GROUP_SIZE = 64;   // local_size_x of Force.comp (particles that walk the tree together)
    static const unsigned int SORT_BLOCK = 1024;  // keys sorted per work group in SortLocal.comp
    static const unsigned int NODE_SIZE = 64;     // bytes per tree node (4 x vec4, see BuildTree.comp)
    static const unsigned int FORCE_NODE_SIZE = 32;   // bytes per compact node for the force walk (see TreeMass.comp)

    static const int TIMER_STAMPS = 7;            // GPU timestamps taken through one step
    GLuint timerQueries[TIMER_STAMPS] = {};
    bool timingRequested = false, timingThisStep = false;
    std::string timingReport;

    unsigned int n;             // number of particles
    unsigned int paddedCount;   // n rounded up to a power of two (>= SORT_BLOCK) for the sort
    GLuint vao = 0;

    ComputeShader integrateShader, boundsShader, mortonShader, sortLocalShader, sortGlobalShader,
                  buildTreeShader, treeMassShader, forceShader;

    static GLuint createBuffer(size_t bytes, const void* data)
    {
        GLuint buffer;
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, data, GL_DYNAMIC_DRAW);
        return buffer;
    }

    // binding points used by every shader (layout(binding = ...) in the .comp files)
    void bindBuffers() const
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, positionBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velocityBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, accelerationBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, keyBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, idBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, nodeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, boundsBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, forceNodeBuffer);
    }

    // records when the GPU reaches this point (only on a step being timed)
    void stamp(int i)
    {
        if (timingThisStep)
            glQueryCounter(timerQueries[i], GL_TIMESTAMP);
    }

    void finishTiming()
    {
        GLuint64 t[TIMER_STAMPS];
        for (int i = 0; i < TIMER_STAMPS; ++i)
            glGetQueryObjectui64v(timerQueries[i], GL_QUERY_RESULT, &t[i]);   // waits for the GPU to get there
        auto ms = [&](int a, int b) { return (double)(t[b] - t[a]) / 1.0e6; };
        char text[256];
        snprintf(text, sizeof(text),
                 "physics step %.2f ms on the GPU = force %.2f + sort %.2f + tree %.2f + box/codes %.2f + integrate %.2f",
                 ms(0, 6), ms(4, 5), ms(2, 3), ms(3, 4), ms(1, 2), ms(0, 1) + ms(5, 6));
        timingReport = text;
    }

    // makes one pass's buffer writes visible to the next pass
    static void barrier() { glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); }

    // Bitonic sort of (Morton code, particle index) pairs. Stages whose compare distance fits inside
    // one block of SORT_BLOCK keys run in shared memory (SortLocal); longer distances use SortGlobal.
    void sortByKey()
    {
        unsigned int blocks = paddedCount / SORT_BLOCK;

        sortLocalShader.use();
        sortLocalShader.setUInt("stageSize", 0);          // 0 = fully sort each block
        glDispatchCompute(blocks, 1, 1);
        barrier();

        for (unsigned int k = 2 * SORT_BLOCK; k <= paddedCount; k *= 2)
        {
            sortGlobalShader.use();
            sortGlobalShader.setUInt("paddedCount", paddedCount);
            sortGlobalShader.setUInt("k", k);
            for (unsigned int j = k / 2; j >= SORT_BLOCK; j /= 2)
            {
                sortGlobalShader.setUInt("j", j);
                sortGlobalShader.dispatch(paddedCount / 2, GROUP_SIZE);
                barrier();
            }
            sortLocalShader.use();
            sortLocalShader.setUInt("stageSize", k);      // finish stage k inside each block
            glDispatchCompute(blocks, 1, 1);
            barrier();
        }
    }
};

#endif

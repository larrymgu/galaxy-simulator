#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"
#include "barnes_hut.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

// Test for GitHub (placeholder line changed)

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera: above the disk, looking down at it at an angle
Camera camera(glm::vec3(0.0f, 70.0f, 100.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -35.0f);
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// physics
const float PHYSICS_DT = 1.0f / 30.0f;    // fixed simulation step in seconds, independent of frame rate
const int MAX_STEPS_PER_FRAME = 2;        // if the GPU can't keep up, run in slow motion instead of the frame rate collapsing

// the disk
const int NUM_PARTICLES = 100000;         // particles in the disk (plus one heavy body in the middle)
const float DISK_INNER_RADIUS = 5.0f;
const float DISK_OUTER_RADIUS = 50.0f;
const float CENTER_MASS = 50.0f;          // body in the middle (think central black hole). Kept light next to the
                                          // dark matter halo so the inner disk doesn't spin far faster than the outer
                                          // disk, which is what winds the spiral arms up (was 350 before the halo)
const float DISK_MASS = 35.0f;            // all the disk particles together
const float CENTER_RADIUS = 1.5f;         // drawn sizes
const float PARTICLE_RADIUS = 0.05f;      // small, so 100,000 of them don't blur into solid white
const float ELLIPSE_FLATTENING = 0.5f;    // how squashed the orbit ellipses get (0 = circles); most squashed halfway out
const float SPIRAL_TWIST = 1.5f * std::numbers::pi_v<float>;   // how far the ellipses turn from inner to outer edge (radians)
const float ARM_SOFTNESS = 0.08f;         // how far particles stray off their ellipse, as a fraction of its size (0 = razor-sharp arms)
const float DISK_SCALE_HEIGHT = 1.0f;     // thickness: particles thin out exponentially above and below the middle plane

// invisible dark matter halo around the galaxy (see DarkMatterHalo in barnes_hut.h)
const float HALO_MASS = 700.0f;           // dark matter inside the disk's outer radius
const float HALO_CORE_RADIUS = 10.0f;     // the halo's density is roughly even inside this radius, then falls off
const DarkMatterHalo HALO = DarkMatterHalo::withMassInside(HALO_MASS, DISK_OUTER_RADIUS, HALO_CORE_RADIUS);

// Builds a spiral disk around a heavy body at the origin: every particle sits on one of a set of
// concentric ellipses, each turned a little further than the one inside it, and is spread a little
// above and below the middle plane.
std::vector<Particle> createDisk()
{
    std::mt19937 rng(1);   // fixed seed: the same disk every run
    std::uniform_real_distribution<float> random01(0.0f, 1.0f);
    std::exponential_distribution<float> height(1.0f / DISK_SCALE_HEIGHT);   // mostly small, rarely large
    std::normal_distribution<float> gaussian(0.0f, 1.0f);

    std::vector<Particle> particles;
    particles.push_back({ glm::vec3(0.0f), CENTER_MASS, glm::vec3(0.0f), CENTER_RADIUS });

    const float particleMass = DISK_MASS / NUM_PARTICLES;
    const float inner2 = DISK_INNER_RADIUS * DISK_INNER_RADIUS;
    const float outer2 = DISK_OUTER_RADIUS * DISK_OUTER_RADIUS;
    const float softening2 = SOFTENING * SOFTENING;

    for (int i = 0; i < NUM_PARTICLES; ++i)
    {
        // size of this particle's ellipse (its semi-major axis a): spread evenly over the disk's area,
        // working outwards, so particle i has about i disk particles inside its orbit
        float fraction = (i + random01(rng)) / NUM_PARTICLES;
        float a = std::sqrt(inner2 + (outer2 - inner2) * fraction);

        // the ellipse: squashed most halfway out and round at the centre and edge (so the galaxy's outline
        // stays round), and turned further the bigger it is. Where neighbouring ellipses crowd together,
        // the particles bunch up into spiral arms.
        float across = (a - DISK_INNER_RADIUS) / (DISK_OUTER_RADIUS - DISK_INNER_RADIUS);   // 0 = inner edge, 1 = outer edge
        float b = a * (1.0f - ELLIPSE_FLATTENING * std::sin(std::numbers::pi_v<float> * across));   // semi-minor axis
        float turn = SPIRAL_TWIST * across;

        // a random point on the ellipse, and the direction along the ellipse at that point.
        // `stray` moves the point slightly in or out of its ellipse, so the arms come out soft instead of
        // razor-thin lines (with every particle exactly on its ellipse, the crowding is perfectly sharp)
        float t = random01(rng) * 2.0f * std::numbers::pi_v<float>;
        float stray = std::max(1.0f + ARM_SOFTNESS * gaussian(rng), 0.1f);
        glm::vec2 point(stray * a * std::cos(t), stray * b * std::sin(t));
        glm::vec2 tangent = glm::normalize(glm::vec2(a * std::sin(t), -b * std::cos(t)));   // same way round as before

        // turn both by the ellipse's angle (x-z plane)
        float c = std::cos(turn), sn = std::sin(turn);
        glm::vec2 p(c * point.x - sn * point.y, sn * point.x + c * point.y);
        glm::vec2 along(c * tangent.x - sn * tangent.y, sn * tangent.x + c * tangent.y);

        // height: above or below the middle plane at random, with exponentially fewer particles further out
        float y = height(rng) * (random01(rng) < 0.5f ? -1.0f : 1.0f);

        // speed: circular-orbit speed for its distance r from the centre, moving along its ellipse.
        // gravity from everything inside = centripetal force: G * M * m / r^2 = m * v^2 / r  ->  v^2 = G * M / r,
        // where M = centre body + disk inside the orbit (r^2 softened the same way as Force.comp) + dark matter inside
        float r = glm::length(p);
        float visibleMassInside = CENTER_MASS + i * particleMass;
        float speed = std::sqrt(G * visibleMassInside * r * r / std::pow(r * r + softening2, 1.5f)
                              + G * HALO.massInside(r) / r);

        particles.push_back({ glm::vec3(p.x, y, p.y), particleMass, speed * glm::vec3(along.x, 0.0f, along.y), PARTICLE_RADIUS });
    }

    // shift to the centre-of-mass frame so the whole disk doesn't slowly drift off
    float totalMass = 0.0f;
    glm::vec3 center(0.0f), momentum(0.0f);
    for (const Particle& p : particles)
    {
        totalMass += p.mass;
        center += p.mass * p.position;
        momentum += p.mass * p.velocity;
    }
    for (Particle& p : particles)
    {
        p.position -= center / totalMass;
        p.velocity -= momentum / totalMass;
    }
    return particles;
}

int main()
{
    // glfw: initialize and configure (compute shaders need OpenGL 4.3)
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Galaxy Simulator", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window (needs a GPU and driver with OpenGL 4.3 or newer)" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);   // let the vertex shader set each point's size

    Shader particleShader("Shader Files/Particle.vert", "Shader Files/Particle.frag");

    BarnesHut simulation(createDisk(), HALO);   // uploads the particles to the GPU
    float physicsAccumulator = 0.0f;      // simulated time owed to the physics, carried between frames

    lastFrame = static_cast<float>(glfwGetTime());   // don't count start-up time as the first frame
    float lastTimingPrint = lastFrame;                // when the GPU timing was last printed

    double prevTime = 0.0;
    double curTime = 0.0;
    double timeDiff;
    unsigned int counter = 0;

    // render loop
    while (!glfwWindowShouldClose(window))
    {
        curTime = glfwGetTime();
        timeDiff = curTime - prevTime;
        counter++;

        if (timeDiff >= 1.0 / 30.0)
        {
            std::string FPS = std::to_string((1.0 / timeDiff) * counter);
            std::string newTitle = "Galaxy Simulator: " + FPS + " FPS";
            glfwSetWindowTitle(window, newTitle.c_str());
            prevTime = curTime;
            counter = 0;
        }
        // per-frame time logic
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        processInput(window);

        // every 2 seconds, time one physics step on the GPU and print where the time went
        if (currentFrame - lastTimingPrint >= 2.0f)
        {
            simulation.requestTiming();
            lastTimingPrint = currentFrame;
        }

        // physics: step gravity forward in fixed increments so the result doesn't depend on frame rate.
        // At most MAX_STEPS_PER_FRAME per frame; time the GPU couldn't catch up on is dropped, so a slow
        // GPU makes the galaxy turn in slow motion rather than piling up more and more work every frame.
        physicsAccumulator += deltaTime;
        int stepsThisFrame = 0;
        while (physicsAccumulator >= PHYSICS_DT && stepsThisFrame < MAX_STEPS_PER_FRAME)
        {
            simulation.step(PHYSICS_DT);
            physicsAccumulator -= PHYSICS_DT;
            stepsThisFrame++;
        }
        physicsAccumulator = std::min(physicsAccumulator, PHYSICS_DT);

        std::string timing = simulation.takeTimingReport();
        if (!timing.empty())
            std::cout << timing << std::endl;

        // render
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);
        glm::mat4 view = camera.GetViewMatrix();
        particleShader.use();
        particleShader.setMat4("projection", projection);
        particleShader.setMat4("view", view);
        particleShader.setFloat("viewportHeight", (float)SCR_HEIGHT);

        simulation.bindForDrawing();
        glDrawArrays(GL_POINTS, 0, simulation.count());   // one draw call for every particle

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
}


// glfw: whenever the mouse moves, this callback is called
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
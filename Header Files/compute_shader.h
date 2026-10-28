#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <glad/glad.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Like Shader, but for a single compute shader (needs OpenGL 4.3 or newer).
class ComputeShader
{
public:
    unsigned int ID = 0;

    explicit ComputeShader(const char* computePath)
    {
        // 1. read the source file
        std::string code;
        std::ifstream file;
        file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try
        {
            file.open(computePath);
            std::stringstream stream;
            stream << file.rdbuf();
            file.close();
            code = stream.str();
        }
        catch (std::ifstream::failure& e)
        {
            std::cout << "ERROR::COMPUTE_SHADER::FILE_NOT_SUCCESSFULLY_READ: " << computePath << ": " << e.what() << std::endl;
        }

        // 2. compile and link
        const char* source = code.c_str();
        unsigned int shader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);
        checkErrors(shader, computePath, false);

        ID = glCreateProgram();
        glAttachShader(ID, shader);
        glLinkProgram(ID);
        checkErrors(ID, computePath, true);
        glDeleteShader(shader);
    }

    void use() const { glUseProgram(ID); }

    void setInt(const std::string& name, int value) const { glUniform1i(glGetUniformLocation(ID, name.c_str()), value); }
    void setUInt(const std::string& name, unsigned int value) const { glUniform1ui(glGetUniformLocation(ID, name.c_str()), value); }
    void setFloat(const std::string& name, float value) const { glUniform1f(glGetUniformLocation(ID, name.c_str()), value); }

    // Runs the shader on at least `threads` threads, in work groups of `groupSize`
    // (must match local_size_x in the shader).
    void dispatch(unsigned int threads, unsigned int groupSize) const
    {
        glDispatchCompute((threads + groupSize - 1) / groupSize, 1, 1);
    }

private:
    static void checkErrors(GLuint object, const char* path, bool isProgram)
    {
        GLint success;
        GLchar infoLog[1024];
        if (!isProgram)
        {
            glGetShaderiv(object, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(object, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: COMPUTE (" << path << ")\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else
        {
            glGetProgramiv(object, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetProgramInfoLog(object, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: COMPUTE (" << path << ")\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }
};

#endif

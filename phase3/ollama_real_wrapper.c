/*
 * C binary wrapper for /usr/local/bin/ollama.real.
 * Exec'd by the main ollama wrapper; execve()s ollama.real.bin with vGPU env.
 * Used so "ollama serve" and "ollama.real runner" both get LD_PRELOAD without
 * running a shell under LD_PRELOAD (which caused SEGV).
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OLLAMA_REAL_BIN "/usr/local/bin/ollama.real.bin"
#define LD_PRELOAD_VAL "/opt/vgpu/lib/libnvidia-ml.so.1:/opt/vgpu/lib/libcuda.so.1:/opt/vgpu/lib/libcudart.so.12"
#define LD_LIBRARY_PATH_VAL "/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama"
#define OLLAMA_LLM_LIBRARY_VAL "cuda_v12"
#define OLLAMA_NUM_GPU_VAL "999"

#define MAX_ENV 512

static int env_starts_with(const char *e, const char *name) {
    size_t n = strlen(name);
    return strncmp(e, name, n) == 0 && e[n] == '=';
}

int main(int argc, char **argv) {
    extern char **environ;
    char *new_env[MAX_ENV];
    int n = 0;

    for (char **e = environ; *e && n < MAX_ENV - 4; e++) {
        if (env_starts_with(*e, "LD_PRELOAD") ||
            env_starts_with(*e, "LD_LIBRARY_PATH") ||
            env_starts_with(*e, "OLLAMA_LLM_LIBRARY") ||
            env_starts_with(*e, "OLLAMA_NUM_GPU"))
            continue;
        new_env[n++] = *e;
    }

    new_env[n++] = "LD_PRELOAD=" LD_PRELOAD_VAL;
    new_env[n++] = "LD_LIBRARY_PATH=" LD_LIBRARY_PATH_VAL;
    new_env[n++] = "OLLAMA_LLM_LIBRARY=" OLLAMA_LLM_LIBRARY_VAL;
    new_env[n++] = "OLLAMA_NUM_GPU=" OLLAMA_NUM_GPU_VAL;
    new_env[n] = NULL;

    execve(OLLAMA_REAL_BIN, argv, new_env);
    return 126;
}

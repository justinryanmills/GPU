/*
 * Minimal C wrapper for /usr/local/bin/ollama.
 * Sets vGPU env (LD_PRELOAD, LD_LIBRARY_PATH, etc.) and execve()s ollama.real.
 * No shell, no script - avoids SEGV that occurred with the bash wrapper.
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OLLAMA_REAL "/usr/local/bin/ollama.real"
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

    /* Copy current environment, but we will override our four vars */
    for (char **e = environ; *e && n < MAX_ENV - 4; e++) {
        if (env_starts_with(*e, "LD_PRELOAD") ||
            env_starts_with(*e, "LD_LIBRARY_PATH") ||
            env_starts_with(*e, "OLLAMA_LLM_LIBRARY") ||
            env_starts_with(*e, "OLLAMA_NUM_GPU"))
            continue;
        new_env[n++] = *e;
    }

    /* Add our vGPU variables */
    new_env[n++] = "LD_PRELOAD=" LD_PRELOAD_VAL;
    new_env[n++] = "LD_LIBRARY_PATH=" LD_LIBRARY_PATH_VAL;
    new_env[n++] = "OLLAMA_LLM_LIBRARY=" OLLAMA_LLM_LIBRARY_VAL;
    new_env[n++] = "OLLAMA_NUM_GPU=" OLLAMA_NUM_GPU_VAL;
    new_env[n] = NULL;

    /* argv: use /usr/local/bin/ollama as argv[0] so when main spawns runner it
     * invokes "ollama runner" (our wrapper) and the runner gets LD_PRELOAD. */
    char **new_argv = malloc((argc + 2) * sizeof(char *));
    if (!new_argv) return 127;
    new_argv[0] = (char *)"/usr/local/bin/ollama";
    for (int i = 1; i < argc; i++)
        new_argv[i] = argv[i];
    new_argv[argc] = NULL;

    execve(OLLAMA_REAL, new_argv, new_env);
    free(new_argv);
    return 126;
}

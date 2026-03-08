/*
 * libvgpu_exec_inject.c — inject LD_PRELOAD into Ollama runner subprocess
 *
 * Ollama server spawns the runner with exec() but does not pass LD_PRELOAD
 * in the child environment, so the runner never loads our vGPU shims and
 * discovery reports CPU. This library is LD_PRELOADed first. It intercepts
 * execve() and when argv contains "runner", adds our vGPU env vars to the
 * child so the runner loads libvgpu-nvml and libvgpu-cuda and sees the GPU.
 *
 * Build: gcc -shared -fPIC -o libvgpu-exec-inject.so libvgpu_exec_inject.c -ldl
 * Use: LD_PRELOAD=libvgpu-exec-inject.so:libvgpu-nvml.so:libvgpu-cuda.so.1 ...
 */
#define _GNU_SOURCE
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

static const char *LD_PRELOAD_VAL = "/opt/vgpu/lib/libvgpu-nvml.so:/opt/vgpu/lib/libvgpu-cuda.so.1";
static const char *LD_LIBRARY_PATH_VAL = "/opt/vgpu/lib:/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12";
static const char *OLLAMA_NUM_GPU_VAL = "1";
static const char *OLLAMA_LLM_LIBRARY_VAL = "cuda_v12";

#define MAX_ENV 512
#define VGPU_ENV_COUNT 4

static int is_runner_argv(char **argv) {
    if (!argv) return 0;
    for (char **a = argv; *a; a++)
        if (strcmp(*a, "runner") == 0) return 1;
    return 0;
}

static int env_starts_with(const char *e, const char *name) {
    size_t n = strlen(name);
    return strncmp(e, name, n) == 0 && e[n] == '=';
}

int execve(const char *pathname, char *const argv[], char *const envp[]) {
    static int (*real_execve)(const char *, char *const [], char *const []) = NULL;
    if (!real_execve)
        real_execve = (int (*)(const char *, char *const [], char *const []))dlsym(RTLD_NEXT, "execve");

    if (!real_execve || !is_runner_argv((char **)argv))
        return real_execve ? real_execve(pathname, argv, envp) : -1;

    /* Build new env: copy existing, add/override vGPU vars */
    const char *new_env[MAX_ENV];
    int n = 0;
    if (envp) {
        for (const char **e = (const char **)envp; *e && n < MAX_ENV - VGPU_ENV_COUNT; e++) {
            if (env_starts_with(*e, "LD_PRELOAD") || env_starts_with(*e, "LD_LIBRARY_PATH") ||
                env_starts_with(*e, "OLLAMA_NUM_GPU") || env_starts_with(*e, "OLLAMA_LLM_LIBRARY"))
                continue;
            new_env[n++] = *e;
        }
    }
    static char ld_preload_buf[256];
    static char ld_library_path_buf[512];
    static char num_gpu_buf[16];
    static char llm_lib_buf[32];
    snprintf(ld_preload_buf, sizeof(ld_preload_buf), "LD_PRELOAD=%s", LD_PRELOAD_VAL);
    snprintf(ld_library_path_buf, sizeof(ld_library_path_buf), "LD_LIBRARY_PATH=%s", LD_LIBRARY_PATH_VAL);
    snprintf(num_gpu_buf, sizeof(num_gpu_buf), "OLLAMA_NUM_GPU=%s", OLLAMA_NUM_GPU_VAL);
    snprintf(llm_lib_buf, sizeof(llm_lib_buf), "OLLAMA_LLM_LIBRARY=%s", OLLAMA_LLM_LIBRARY_VAL);
    new_env[n++] = ld_preload_buf;
    new_env[n++] = ld_library_path_buf;
    new_env[n++] = num_gpu_buf;
    new_env[n++] = llm_lib_buf;
    new_env[n] = NULL;

    return real_execve(pathname, argv, (char *const *)new_env);
}

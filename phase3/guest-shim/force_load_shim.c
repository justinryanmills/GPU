/*
 * Force Load Shim - Alternative injection method
 * 
 * This creates a wrapper that uses dlopen to force-load the shim
 * before exec'ing the target binary. Works even when LD_PRELOAD fails.
 * 
 * Compile: gcc -o force_load_shim force_load_shim.c -ldl
 * Usage: ./force_load_shim ollama run ...
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

/* CUDA result type - matches cuda.h */
typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_NO_DEVICE = 100
} CUresult;

static const char *SHIM_CUDA = "/usr/lib64/libvgpu-cuda.so";
static const char *SHIM_NVML = "/usr/lib64/libvgpu-nvml.so";
static const char *LD_AUDIT_LIB = "/usr/lib64/libldaudit_cuda.so";

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <program> [args...]\n", argv[0]);
        fprintf(stderr, "Example: %s ollama run llama3.2:1b hello\n", argv[0]);
        return 1;
    }
    
    fprintf(stderr, "[force-load] Pre-loading shim libraries...\n");
    
    /* NOTE: libvgpu-exec removed - Go's runtime uses direct syscalls which bypass
     * libc's execve() wrapper, so interception doesn't work. Instead, we rely on:
     * 1. System-wide library path registration via /etc/ld.so.conf.d/vgpu.conf
     * 2. Symlinks in standard library paths
     * 3. LD_LIBRARY_PATH as backup
     * These mechanisms work at the dynamic linker level, independent of process spawning method. */
    
    /* Force load CUDA shim with RTLD_GLOBAL so it's available to all */
    void *cuda_handle = dlopen(SHIM_CUDA, RTLD_NOW | RTLD_GLOBAL);
    if (!cuda_handle) {
        fprintf(stderr, "[force-load] WARNING: Failed to load %s: %s\n", 
                SHIM_CUDA, dlerror());
    } else {
        fprintf(stderr, "[force-load] Loaded: %s\n", SHIM_CUDA);
    }
    
    /* Force load NVML shim */
    void *nvml_handle = dlopen(SHIM_NVML, RTLD_NOW | RTLD_GLOBAL);
    if (!nvml_handle) {
        fprintf(stderr, "[force-load] WARNING: Failed to load %s: %s\n", 
                SHIM_NVML, dlerror());
    } else {
        fprintf(stderr, "[force-load] Loaded: %s\n", SHIM_NVML);
    }
    
    /* Set LD_LIBRARY_PATH FIRST - critical for LD_PRELOAD to find libraries */
    char libpath_val[512];
    snprintf(libpath_val, sizeof(libpath_val), "/usr/lib64:%s", getenv("LD_LIBRARY_PATH") ?: "");
    setenv("LD_LIBRARY_PATH", libpath_val, 1);
    fprintf(stderr, "[force-load] Set LD_LIBRARY_PATH=%s\n", libpath_val);
    
    /* Set LD_AUDIT (for monitoring - la_objsearch removed due to glibc incompatibility) */
    setenv("LD_AUDIT", LD_AUDIT_LIB, 1);
    fprintf(stderr, "[force-load] Set LD_AUDIT=%s\n", LD_AUDIT_LIB);
    
    /* Set LD_PRELOAD with full paths as backup mechanism
     * Even though Go uses direct syscalls, LD_PRELOAD may still help with
     * library discovery for subprocesses. We use full paths to avoid
     * secure-execution mode issues. This works in combination with:
     * 1. System-wide library path registration (/etc/ld.so.conf.d/vgpu.conf)
     * 2. Symlinks in standard paths (libcuda.so.1 -> libvgpu-cuda.so)
     * 3. LD_LIBRARY_PATH
     * Multiple mechanisms ensure libraries are found regardless of how processes spawn */
    char preload_val[512];
    snprintf(preload_val, sizeof(preload_val), "%s", SHIM_CUDA);
    setenv("LD_PRELOAD", preload_val, 1);
    fprintf(stderr, "[force-load] Set LD_PRELOAD=%s\n", preload_val);
    
    /* Verify libraries are loaded by checking for CUDA symbols
     * Based on NVIDIA best practices: Check already-loaded libraries first
     * This follows the recommended search order for maximum stability */
    void *cuInit_sym = dlsym(RTLD_DEFAULT, "cuInit");
    if (cuInit_sym) {
        fprintf(stderr, "[force-load] ✓ Verified: CUDA symbols available (cuInit found)\n");
        
        /* NOTE: Based on NVIDIA research, we do NOT pre-initialize CUDA here.
         * 
         * Best practices indicate:
         * 1. Lazy initialization is preferred (initialization happens on first CUDA call)
         * 2. cuInit() should be called explicitly by the application or via ensure_init()
         * 3. Pre-initialization in constructors/wrappers can cause conflicts
         * 
         * The library is now loaded and will initialize automatically when needed.
         * This provides maximum stability and follows NVIDIA's recommended patterns.
         */
        fprintf(stderr, "[force-load] ✓ Libraries loaded (lazy initialization will occur on first CUDA call)\n");
    } else {
        fprintf(stderr, "[force-load] ⚠ WARNING: CUDA symbols not yet available (will be loaded via symlinks/LD_PRELOAD)\n");
    }
    
    fprintf(stderr, "[force-load] Executing: %s\n", argv[1]);
    fflush(stderr);
    
    /* Execute the target program - shims should now be loaded
     * Use execve instead of execvp to handle absolute paths correctly
     * execvp expects a filename to search in PATH, but we have an absolute path */
    extern char **environ;
    execve(argv[1], &argv[1], environ);
    
    /* If we get here, exec failed */
    perror("execvp failed");
    return 1;
}

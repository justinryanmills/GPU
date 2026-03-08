/*
 * libvgpu_nvml.c  —  NVML shim library for guest VM
 *
 * This shared library (libvgpu-nvml.so) replaces libnvidia-ml.so.1
 * in the guest VM.  It provides a minimal NVML implementation so
 * that nvidia-smi and Ollama's GPU detection code work correctly.
 *
 * Device discovery functions are answered locally using the same
 * GPU properties that the CUDA shim uses.  Monitoring functions
 * (temperature, utilisation) are forwarded to the host.
 *
 * Build:
 *   gcc -shared -fPIC -o libvgpu-nvml.so libvgpu_nvml.c cuda_transport.c \
 *       -I../include -I.
 *
 * Symlink:
 *   ln -sf /usr/lib64/libvgpu-nvml.so /usr/lib64/libnvidia-ml.so.1
 *   ln -sf /usr/lib64/libvgpu-nvml.so /usr/lib64/libnvidia-ml.so
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#include <time.h>      /* For clock_gettime, timestamps */
#include <sys/syscall.h>  /* For syscall() */
#include <fcntl.h>        /* For O_RDONLY */

#include "cuda_protocol.h"
#include "cuda_transport.h"
#include "gpu_properties.h"

/* ================================================================
 * Stub for libvgpu_set_skip_interception
 * 
 * This function is defined in libvgpu-cuda.so, but when libggml-cuda.so
 * loads libnvidia-ml.so.1 (our NVML shim) as a dependency, the CUDA shim
 * might not be loaded yet. We provide a stub here so the symbol resolves.
 * 
 * If the CUDA shim is loaded later, cuda_transport.c will use dlsym to
 * find the real implementation and use that instead.
 * ================================================================ */
void libvgpu_set_skip_interception(int skip)
{
    /* Stub implementation - does nothing */
    /* cuda_transport.c will use dlsym to find the real implementation
     * from libvgpu-cuda.so if it's available */
    (void)skip;  /* Suppress unused parameter warning */
}

/* ================================================================
 * dlsym interception for NVML discovery
 *
 * Ollama uses dlsym to find NVML functions. If dlsym fails to find
 * a function, Ollama might skip NVML discovery entirely. We intercept
 * dlsym to log what functions Ollama is looking for and ensure they
 * are found.
 * ================================================================ */

/* NOTE: dlsym interception is complex and requires __libc_dlsym which may not
 * be available. Instead, we ensure all NVML functions are properly exported so
 * dlsym can find them. We'll add logging via a different mechanism if needed.
 * For now, we don't intercept dlsym to avoid bootstrap issues. */

/* ================================================================
 * NVML types — minimal ABI-compatible definitions
 * ================================================================ */

typedef int nvmlReturn_t;

#define NVML_SUCCESS                      0
#define NVML_ERROR_UNINITIALIZED          1
#define NVML_ERROR_INVALID_ARGUMENT       2
#define NVML_ERROR_NOT_SUPPORTED          3
#define NVML_ERROR_NO_PERMISSION          4
#define NVML_ERROR_NOT_FOUND              6
#define NVML_ERROR_INSUFFICIENT_SIZE      7
#define NVML_ERROR_DRIVER_NOT_LOADED      9
#define NVML_ERROR_UNKNOWN                999

typedef struct {
    uint32_t handle_id;       /* internal — always 0 */
} nvmlDevice_st;

typedef nvmlDevice_st* nvmlDevice_t;

typedef enum {
    NVML_TEMPERATURE_GPU = 0,
} nvmlTemperatureSensors_t;

typedef struct {
    unsigned long long total;     /* bytes */
    unsigned long long free;      /* bytes */
    unsigned long long used;      /* bytes */
} nvmlMemory_t;

typedef struct {
    unsigned int gpu;             /* percent */
    unsigned int memory;          /* percent */
} nvmlUtilization_t;

typedef struct {
    unsigned int power;           /* milliwatts */
} nvmlPowerUsage_t;

typedef enum {
    NVML_PCIE_UTIL_TX_BYTES = 0,
    NVML_PCIE_UTIL_RX_BYTES = 1,
} nvmlPcieUtilCounter_t;

typedef enum {
    NVML_CLOCK_GRAPHICS = 0,
    NVML_CLOCK_SM       = 1,
    NVML_CLOCK_MEM      = 2,
    NVML_CLOCK_VIDEO    = 3,
} nvmlClockType_t;

typedef enum {
    NVML_CLOCK_ID_CURRENT   = 0,
    NVML_CLOCK_ID_APP_CLOCK_TARGET = 1,
    NVML_CLOCK_ID_APP_CLOCK_DEFAULT = 2,
    NVML_CLOCK_ID_CUSTOMER_BOOST_MAX = 3,
} nvmlClockId_t;

typedef enum {
    NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0,
    NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1,
    NVML_TEMPERATURE_THRESHOLD_MEM_MAX  = 2,
    NVML_TEMPERATURE_THRESHOLD_GPU_MAX  = 3,
} nvmlTemperatureThresholds_t;

typedef struct {
    char                 busId[32];
    unsigned int         domain;
    unsigned int         bus;
    unsigned int         device;
    unsigned int         pciDeviceId;
    unsigned int         pciSubSystemId;
} nvmlPciInfo_t;

typedef enum {
    NVML_COMPUTEMODE_DEFAULT          = 0,
    NVML_COMPUTEMODE_EXCLUSIVE_THREAD = 1,
    NVML_COMPUTEMODE_PROHIBITED       = 2,
    NVML_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,
} nvmlComputeMode_t;

/* Forward declarations — versioned API wrappers */
nvmlReturn_t nvmlInit_v2(void);
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci);
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor);
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length);

/* ================================================================
 * Load-time diagnostic: confirm the NVML shim is actually loaded.
 * Also writes to /tmp/vgpu-shim-nvml-<pid>.log so piped discovery
 * subprocess activity is visible even when stderr is not journald.
 * ================================================================ */
/* Helper function: Check if this is an Ollama process (robust, safe check)
 * Uses the same defensive logic as is_application_process() in libvgpu_cuda.c
 * CRITICAL: Check for system processes FIRST, then check for ollama
 * Default to 0 (safe) if anything goes wrong
 * NOTE: Currently unused because constructor is empty, but kept for future use */
__attribute__((unused)) static int is_ollama_process_safe(void)
{
    /* CRITICAL: Default to NOT intercepting (safe default) */
    /* If ANYTHING goes wrong at ANY point, we return 0 and don't intercept */
    char comm[256] = {0};
    int fd = -1;
    ssize_t n = 0;
    
    /* CRITICAL: Be EXTREMELY defensive - use ONLY syscalls, catch ALL errors */
    /* If ANY step fails, return 0 immediately (don't intercept) */
    fd = syscall(__NR_open, "/proc/self/comm", O_RDONLY);
    if (fd < 0) {
        /* Can't read comm - default to not intercepting (safe default) */
        return 0;
    }
    
    n = syscall(__NR_read, fd, comm, sizeof(comm) - 1);
    syscall(__NR_close, fd);  /* Close immediately after read */
    
    if (n <= 0 || n >= (ssize_t)sizeof(comm)) {
        /* Read failed or too large - default to not intercepting (safe default) */
        return 0;
    }
    
    comm[n] = '\0';
    /* Remove newline if present - be very careful with bounds */
    if (n > 0 && comm[n-1] == '\n') {
        comm[n-1] = '\0';
        n--;
    }
    
    /* CRITICAL: Check for system processes FIRST using ONLY direct character comparisons */
    /* Use minimal checks - just first few characters to catch common system tools */
    /* All early returns are safe - we don't intercept system processes */
    
    /* Check lspci FIRST (most common crash source) - check first 2 chars immediately */
    if (n >= 2 && comm[0] == 'l' && comm[1] == 's') {
        return 0;  /* Could be lspci, ls, or other 'ls*' tools - don't intercept */
    }
    /* Check cat - common system tool */
    if (n >= 3 && comm[0] == 'c' && comm[1] == 'a' && comm[2] == 't') {
        return 0;  /* Don't intercept cat */
    }
    /* Check bash/sh - shell processes */
    if (n >= 4 && comm[0] == 'b' && comm[1] == 'a' && comm[2] == 's' && comm[3] == 'h') {
        return 0;  /* Don't intercept bash */
    }
    if (n >= 2 && comm[0] == 's' && comm[1] == 'h') {
        return 0;  /* Don't intercept sh */
    }
    /* Check sshd/systemd - first char 's' */
    if (n >= 1 && comm[0] == 's') {
        /* Could be sshd, systemd, or other 's*' tools - be more specific */
        if (n >= 4 && comm[1] == 's' && comm[2] == 'h' && comm[3] == 'd') {
            return 0;  /* sshd */
        }
        if (n >= 7 && comm[1] == 'y' && comm[2] == 's' && comm[3] == 't' && 
            comm[4] == 'e' && comm[5] == 'm' && comm[6] == 'd') {
            return 0;  /* systemd */
        }
    }
    /* Check init */
    if (n >= 4 && comm[0] == 'i' && comm[1] == 'n' && comm[2] == 'i' && comm[3] == 't') {
        return 0;  /* Don't intercept init */
    }
    /* Check echo */
    if (n >= 4 && comm[0] == 'e' && comm[1] == 'c' && comm[2] == 'h' && comm[3] == 'o') {
        return 0;  /* Don't intercept echo */
    }
    /* Check pwd */
    if (n >= 3 && comm[0] == 'p' && comm[1] == 'w' && comm[2] == 'd') {
        return 0;  /* Don't intercept pwd */
    }
    /* Check systemctl */
    if (n >= 9 && comm[0] == 's' && comm[1] == 'y' && comm[2] == 's' && 
        comm[3] == 't' && comm[4] == 'e' && comm[5] == 'm' && 
        comm[6] == 'c' && comm[7] == 't' && comm[8] == 'l') {
        return 0;  /* Don't intercept systemctl */
    }
    
    /* WHITELIST: Only intercept for ollama processes */
    /* Use direct character comparison (6 chars) - NO string functions */
    /* If it's not ollama, we don't intercept (return 0) */
    if (n >= 6 && comm[0] == 'o' && comm[1] == 'l' && comm[2] == 'l' && 
        comm[3] == 'a' && comm[4] == 'm' && comm[5] == 'a') {
        return 1;  /* Only ollama gets intercepted */
    }
    
    /* CRITICAL: If we reach here, it's not ollama and not a known system process */
    /* Default to not intercepting (safe default) */
    return 0;
}

/* Constructor - do nothing. Init is lazy on first nvmlDeviceGetCount_v2() / nvmlInit_v2().
 * Even a single syscall write here can trigger SEGV after other constructors. */
__attribute__((constructor(101)))
static void libvgpu_nvml_on_load(void)
{
    (void)0;  /* no-op; avoid any syscall/write */
    return;

    /* OLD COMMENT - kept for reference:
     * When deployed via /etc/ld.so.preload, this library loads into ALL processes
     * (sshd, systemd, etc.) during VERY EARLY initialization, BEFORE libc/pthreads
     * are fully initialized. Even syscalls may not be safe at this point.
     * 
     * UNSAFE operations that MUST NOT be in constructor:
     * - getenv() - libc may not be initialized
     * - getpid() - libc may not be initialized
     * - fprintf() - libc may not be initialized
     * - fopen() - filesystem may not be ready
     * - nvmlInit_v2() - calls pthread_mutex_lock, filesystem I/O, fprintf
     * - Any syscalls - can fail during very early init
     * - Any libc calls - libc may not be ready
     * - Any file I/O - filesystem may not be ready
     * - Any mutex operations - pthreads may not be initialized
     * - Process detection checks - even syscalls may crash during early init
     * 
     * SOLUTION: Do absolutely nothing in constructor.
     * 
     * Initialization happens lazily via nvmlInit_v2() when NVML functions are
     * actually called. This is safe because:
     * - System processes never call NVML functions, so they're unaffected
     * - Application processes are fully initialized by the time they call NVML
     * - nvmlInit_v2() can check is_ollama_process_safe() at that point (when safe)
     * 
     * For Ollama processes that need early initialization, they should:
     * - Call nvmlInit_v2() explicitly or via first NVML call
     * - Use a wrapper script or systemd service to handle initialization
     */
}

/* ================================================================
 * Global state
 * ================================================================ */
static int               g_nvml_initialized = 0;
static pthread_mutex_t   g_nvml_mutex;  /* Lazy-initialized - do NOT use PTHREAD_MUTEX_INITIALIZER */
static int               g_nvml_mutex_initialized = 0;  /* Track if mutex is initialized */

/* Helper: Ensure mutex is initialized (lazy initialization)
 * CRITICAL: Do NOT use PTHREAD_MUTEX_INITIALIZER - it runs at library load time
 * and can crash during early initialization via /etc/ld.so.preload */
static void ensure_nvml_mutex_init(void)
{
    if (!g_nvml_mutex_initialized) {
        /* Simple check - during early library loading, we're typically single-threaded */
        /* If there's a race, pthread_mutex_init will fail on second call, which is OK */
        int rc = pthread_mutex_init(&g_nvml_mutex, NULL);
        if (rc == 0) {
            g_nvml_mutex_initialized = 1;
        }
        /* If rc != 0, mutex was already initialized by another thread - that's OK */
    }
}
static cuda_transport_t *g_nvml_transport = NULL;
static CUDAGpuInfo       g_nvml_gpu_info;
static int               g_nvml_gpu_info_valid = 0;

/* Singleton device handle */
static nvmlDevice_st     g_device = { .handle_id = 0 };

/* ================================================================
 * Internal: fetch GPU info from host
 * NOTE: Currently unused but kept for future use
 * ================================================================ */
__attribute__((unused)) static void nvml_fetch_gpu_info(void)
{
    CUDACallResult result;
    uint32_t recv_len = 0;

    memset(&g_nvml_gpu_info, 0, sizeof(g_nvml_gpu_info));

    int rc = cuda_transport_call(g_nvml_transport,
                                 CUDA_CALL_GET_GPU_INFO,
                                 NULL, 0, NULL, 0,
                                 &result,
                                 &g_nvml_gpu_info, sizeof(g_nvml_gpu_info),
                                 &recv_len);

    if (rc == 0 && recv_len >= sizeof(g_nvml_gpu_info)) {
        g_nvml_gpu_info_valid = 1;
    } else {
        /* Use defaults */
        strncpy(g_nvml_gpu_info.name, GPU_DEFAULT_NAME,
                sizeof(g_nvml_gpu_info.name) - 1);
        g_nvml_gpu_info.total_mem = GPU_DEFAULT_TOTAL_MEM;
        g_nvml_gpu_info.free_mem  = GPU_DEFAULT_FREE_MEM;
        g_nvml_gpu_info.compute_cap_major = GPU_DEFAULT_CC_MAJOR;
        g_nvml_gpu_info.compute_cap_minor = GPU_DEFAULT_CC_MINOR;
        g_nvml_gpu_info.driver_version = GPU_DEFAULT_DRIVER_VERSION;
        g_nvml_gpu_info_valid = 1;
    }
}

/* ================================================================
 * NVML API — Initialisation
 * ================================================================ */

nvmlReturn_t nvmlInit(void)
{
    return nvmlInit_v2();
}

nvmlReturn_t nvmlInit_v2(void)
{
    /* TEMPORARY: Remove fprintf to test if it's causing segfault */
    /*
    fprintf(stderr, "[libvgpu-nvml] nvmlInit_v2() CALLED (pid=%d)\n", (int)getpid());
    */
    ensure_nvml_mutex_init();
    pthread_mutex_lock(&g_nvml_mutex);

    if (g_nvml_initialized) {
        pthread_mutex_unlock(&g_nvml_mutex);
        /*
        fprintf(stderr, "[libvgpu-nvml] nvmlInit_v2() already initialized\n");
        */
        return NVML_SUCCESS;
    }

    /*
     * Phase 1: lightweight device scan.
     *
     * Call cuda_transport_discover() first so that g_discovered_bdf is
     * populated even if the full transport init below fails (e.g. because
     * resource0 is not yet writable inside a systemd sandbox).
     * nvmlDeviceGetPciInfo_v3() reads the BDF via cuda_transport_pci_bdf()
     * which falls back to g_discovered_bdf when g_nvml_transport is NULL.
     */
    cuda_transport_discover();   /* sets g_discovered_bdf; ignore error here */

    /*
     * Phase 2: Defer full transport init (similar to cuInit()).
     * 
     * During discovery phase, we don't need the full transport (BAR0 mmap).
     * We only need device information which we can provide from defaults.
     * Full transport will be established later when compute operations start.
     * 
     * This makes initialization fast and prevents timeouts during discovery.
     */
    /* Use defaults immediately - no blocking transport init */
    strncpy(g_nvml_gpu_info.name, GPU_DEFAULT_NAME,
            sizeof(g_nvml_gpu_info.name) - 1);
    g_nvml_gpu_info.total_mem = GPU_DEFAULT_TOTAL_MEM;
    g_nvml_gpu_info.free_mem  = GPU_DEFAULT_FREE_MEM;
    g_nvml_gpu_info.driver_version = GPU_DEFAULT_DRIVER_VERSION;
    g_nvml_gpu_info.compute_cap_major = GPU_DEFAULT_CC_MAJOR;
    g_nvml_gpu_info.compute_cap_minor = GPU_DEFAULT_CC_MINOR;
    g_nvml_gpu_info_valid = 1;
    g_nvml_initialized = 1;
    pthread_mutex_unlock(&g_nvml_mutex);
    
    /* FIX: Store cuda_transport_pci_bdf() result in a local variable first
     * This avoids potential issues with calling it directly in fprintf() format string */
    const char *bdf = cuda_transport_pci_bdf(NULL);
    fprintf(stderr,
            "[libvgpu-nvml] nvmlInit() succeeded with defaults (transport deferred, bdf=%s)\n",
            bdf ? bdf : "unknown");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlShutdown(void)
{
    ensure_nvml_mutex_init();
    pthread_mutex_lock(&g_nvml_mutex);
    if (g_nvml_transport) {
        cuda_transport_destroy(g_nvml_transport);
        g_nvml_transport = NULL;
    }
    g_nvml_initialized = 0;
    pthread_mutex_unlock(&g_nvml_mutex);
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — System queries
 * ================================================================ */

nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)
{
    fprintf(stderr, "[libvgpu-nvml] nvmlSystemGetDriverVersion() CALLED (pid=%d)\n", (int)getpid());
    fflush(stderr);
    if (!version || length == 0) {
        fprintf(stderr, "[libvgpu-nvml] nvmlSystemGetDriverVersion() invalid arguments\n");
        fflush(stderr);
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    
    /* Ensure NVML is initialized */
    if (!g_nvml_initialized) {
        fprintf(stderr, "[libvgpu-nvml] nvmlSystemGetDriverVersion() auto-initializing NVML\n");
        fflush(stderr);
        nvmlInit_v2();
    }
    
    /* Format: "535.129.03" style */
    int dv = g_nvml_gpu_info_valid ? g_nvml_gpu_info.driver_version
                                    : GPU_DEFAULT_DRIVER_VERSION;
    snprintf(version, length, "%d.%d.%02d",
             dv / 1000, (dv % 1000) / 10, dv % 10);
    fprintf(stderr, "[libvgpu-nvml] nvmlSystemGetDriverVersion() returning: %s (pid=%d)\n",
            version, (int)getpid());
    fflush(stderr);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion)
{
    if (!cudaDriverVersion) return NVML_ERROR_INVALID_ARGUMENT;
    *cudaDriverVersion = g_nvml_gpu_info_valid
                          ? g_nvml_gpu_info.driver_version
                          : GPU_DEFAULT_DRIVER_VERSION;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion)
{
    return nvmlSystemGetCudaDriverVersion(cudaDriverVersion);
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length)
{
    if (!version || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(version, length, "12.8.0");
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Device enumeration
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount)
{
    return nvmlDeviceGetCount_v2(deviceCount);
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount)
{
    /* Debug: write to file so we can confirm discovery path even if stderr is redirected */
    {
        char buf[80];
        int n = snprintf(buf, sizeof(buf), "nvmlDeviceGetCount_v2 called pid=%d\n", (int)getpid());
        if (n > 0 && n < (int)sizeof(buf)) {
            int fd = (int)syscall(__NR_open, "/tmp/nvml_get_count_called.txt",
                O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (fd >= 0) {
                syscall(__NR_write, fd, buf, (size_t)n);
                syscall(__NR_close, fd);
            }
        }
    }
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCount_v2() CALLED (pid=%d)\n", (int)getpid());
    fflush(stderr);
    
    if (!deviceCount) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCount_v2() invalid pointer\n");
        fflush(stderr);
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    
    /* CRITICAL: Return count=1 immediately if we have LD_PRELOAD (shims loaded).
     * This ensures ggml_cuda_init() can proceed even if NVML isn't fully initialized yet.
     * Similar to cuDeviceGetCount(), we return immediately to avoid any delays. */
    const char *ld_preload = getenv("LD_PRELOAD");
    int has_shims = (ld_preload && strstr(ld_preload, "libvgpu"));
    
    if (has_shims) {
        *deviceCount = 1;
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCount_v2() returning count=1 immediately (has_shims=1, pid=%d)\n", (int)getpid());
        fflush(stderr);
        return NVML_SUCCESS;
    }
    
    /* If we don't have shims, ensure NVML is initialized */
    if (!g_nvml_initialized) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCount_v2() auto-initializing NVML\n");
        fflush(stderr);
        nvmlInit_v2();
    }
    *deviceCount = 1;
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCount_v2() returning count=1\n");
    fflush(stderr);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index,
                                         nvmlDevice_t *device)
{
    return nvmlDeviceGetHandleByIndex_v2(index, device);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                            nvmlDevice_t *device)
{
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByIndex_v2() CALLED (pid=%d, index=%u)\n", 
            (int)getpid(), index);
    if (!device) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByIndex_v2() invalid device pointer\n");
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (index != 0) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByIndex_v2() index %u not found\n", index);
        return NVML_ERROR_NOT_FOUND;
    }
    /* Ensure NVML is initialized before returning device handle */
    if (!g_nvml_initialized) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByIndex_v2() auto-initializing NVML\n");
        nvmlInit_v2();
    }
    *device = &g_device;
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByIndex_v2() returning device handle\n");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid,
                                        nvmlDevice_t *device)
{
    (void)uuid;
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    *device = &g_device;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId,
                                            nvmlDevice_t *device)
{
    return nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId,
                                               nvmlDevice_t *device)
{
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByPciBusId_v2() CALLED (pid=%d, pciBusId=%s)\n", 
            (int)getpid(), pciBusId ? pciBusId : "NULL");
    fflush(stderr);
    if (!device) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByPciBusId_v2() invalid device pointer\n");
        fflush(stderr);
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    /* Ensure NVML is initialized */
    if (!g_nvml_initialized) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByPciBusId_v2() auto-initializing NVML\n");
        fflush(stderr);
        nvmlInit_v2();
    }
    *device = &g_device;
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetHandleByPciBusId_v2() returning device handle\n");
    fflush(stderr);
    (void)pciBusId;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index)
{
    if (!index) return NVML_ERROR_INVALID_ARGUMENT;
    *index = 0;
    (void)device;
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Device properties
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name,
                                unsigned int length)
{
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetName() CALLED (pid=%d)\n", (int)getpid());
    if (!name || length == 0) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetName() invalid arguments\n");
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    (void)device;

    /* Ensure NVML is initialized */
    if (!g_nvml_initialized) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetName() auto-initializing NVML\n");
        nvmlInit_v2();
    }

    const char *src = g_nvml_gpu_info_valid ? g_nvml_gpu_info.name
                                             : GPU_DEFAULT_NAME;
    strncpy(name, src, length - 1);
    name[length - 1] = '\0';
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetName() returning: %s\n", name);
    fflush(stderr);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid,
                                unsigned int length)
{
    if (!uuid || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;

    /* Format: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx */
    const uint8_t *u = g_nvml_gpu_info.uuid;
    snprintf(uuid, length,
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-"
             "%02x%02x-%02x%02x%02x%02x%02x%02x",
             u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
             u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)
{
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetMemoryInfo() CALLED (pid=%d)\n", (int)getpid());
    fflush(stderr);
    if (!memory) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetMemoryInfo() invalid pointer\n");
        fflush(stderr);
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    (void)device;

    /* Ensure NVML is initialized */
    if (!g_nvml_initialized) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetMemoryInfo() auto-initializing NVML\n");
        fflush(stderr);
        nvmlInit_v2();
    }

    memory->total = g_nvml_gpu_info_valid ? g_nvml_gpu_info.total_mem
                                           : GPU_DEFAULT_TOTAL_MEM;
    memory->free  = g_nvml_gpu_info_valid ? g_nvml_gpu_info.free_mem
                                           : GPU_DEFAULT_FREE_MEM;
    memory->used  = memory->total - memory->free;
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetMemoryInfo() returning: total=%llu MB, free=%llu MB (pid=%d)\n",
            (unsigned long long)(memory->total / (1024 * 1024)),
            (unsigned long long)(memory->free / (1024 * 1024)),
            (int)getpid());
    fflush(stderr);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device,
                                         nvmlMemory_t *memory)
{
    return nvmlDeviceGetMemoryInfo(device, memory);
}

nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci)
{
    return nvmlDeviceGetPciInfo_v3(device, pci);
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci)
{
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetPciInfo_v3() CALLED (pid=%d)\n", (int)getpid());
    if (!pci) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetPciInfo_v3() invalid pointer\n");
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    (void)device;
    memset(pci, 0, sizeof(*pci));

    /* Ensure NVML is initialized */
    if (!g_nvml_initialized) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetPciInfo_v3() auto-initializing NVML\n");
        nvmlInit_v2();
    }

    /* Use the actual PCI BDF of the VGPU-STUB device so that Ollama's
     * GPU-discovery code can match the NVML device with the CUDA device
     * by PCI bus ID (previously this was hardcoded "00000000:00:00.0",
     * which caused the match to fail and GPU discovery to fall back to CPU). */
    const char *bdf = cuda_transport_pci_bdf(g_nvml_transport);
    if (!bdf || bdf[0] == '\0') {
        bdf = cuda_transport_pci_bdf(NULL);  /* Use discovered BDF */
        if (!bdf || bdf[0] == '\0') {
            bdf = "0000:00:05.0";  /* Final fallback */
        }
    }
    
    /* CRITICAL FIX: Format BDF to match filesystem format (4-digit domain) for Ollama's matching logic.
     * The filesystem provides "0000:00:05.0" (4-digit domain), and Ollama matches PCI devices
     * with NVML devices by comparing the bus ID from filesystem with the bus ID from NVML.
     * They must match exactly, so we use 4-digit format to match the filesystem. */
    unsigned int dom = 0, bus = 0, dev = 0, fn = 0;
    if (sscanf(bdf, "%x:%x:%x.%x", &dom, &bus, &dev, &fn) == 4) {
        /* Format as 4-digit domain to match filesystem: "0000:00:05.0" */
        snprintf(pci->busId, sizeof(pci->busId), "%04x:%02x:%02x.%x", dom, bus, dev, fn);
    } else {
        /* Fallback: use as-is */
        snprintf(pci->busId, sizeof(pci->busId), "%s", bdf);
    }
    
    pci->domain = dom;
    pci->bus    = bus;
    pci->device = dev;
    pci->pciDeviceId    = (GPU_DEFAULT_PCI_DEVICE_ID << 16) | GPU_DEFAULT_PCI_VENDOR_ID;
    pci->pciSubSystemId = 0;
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetPciInfo_v3() returning busId=\"%s\" (pid=%d)\n", 
            pci->busId, (int)getpid());
    fflush(stderr);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device,
                                                 int *major, int *minor)
{
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCudaComputeCapability() CALLED (pid=%d)\n", (int)getpid());
    fflush(stderr);
    if (!major || !minor) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCudaComputeCapability() invalid arguments\n");
        fflush(stderr);
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    (void)device;
    
    /* Ensure NVML is initialized */
    if (!g_nvml_initialized) {
        fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCudaComputeCapability() auto-initializing NVML\n");
        fflush(stderr);
        nvmlInit_v2();
    }
    
    /* CRITICAL: Always return 9.0 regardless of initialization state */
    *major = GPU_DEFAULT_CC_MAJOR;  /* Force 9 */
    *minor = GPU_DEFAULT_CC_MINOR;  /* Force 0 */
    
    /* CRITICAL: Enhanced logging to verify this is called during discovery */
    fprintf(stderr, "[libvgpu-nvml] nvmlDeviceGetCudaComputeCapability() returning: %d.%d (FORCED, pid=%d)\n",
            *major, *minor, (int)getpid());
    fflush(stderr);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device,
                                       nvmlComputeMode_t *mode)
{
    if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *mode = NVML_COMPUTEMODE_DEFAULT;
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Monitoring (temperature, utilization, power, clocks)
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                       nvmlTemperatureSensors_t sensorType,
                                       unsigned int *temp)
{
    if (!temp) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device; (void)sensorType;
    /* Return a reasonable default */
    *temp = 45;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device,
                                                nvmlTemperatureThresholds_t thresholdType,
                                                unsigned int *temp)
{
    if (!temp) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    switch (thresholdType) {
    case NVML_TEMPERATURE_THRESHOLD_SHUTDOWN:  *temp = 95; break;
    case NVML_TEMPERATURE_THRESHOLD_SLOWDOWN:  *temp = 90; break;
    case NVML_TEMPERATURE_THRESHOLD_MEM_MAX:   *temp = 95; break;
    case NVML_TEMPERATURE_THRESHOLD_GPU_MAX:   *temp = 83; break;
    default: *temp = 90; break;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device,
                                            nvmlUtilization_t *utilization)
{
    if (!utilization) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    /* Return idle as default */
    utilization->gpu = 0;
    utilization->memory = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device,
                                      unsigned int *power)
{
    if (!power) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *power = 50000;  /* 50 W in milliwatts (idle H100) */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device,
                                                unsigned int *limit)
{
    if (!limit) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *limit = 350000;  /* 350 W (H100 TDP) */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device,
                                     nvmlClockType_t type,
                                     unsigned int *clock)
{
    if (!clock) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    switch (type) {
    case NVML_CLOCK_GRAPHICS:
    case NVML_CLOCK_SM:
        *clock = GPU_DEFAULT_CLOCK_RATE_KHZ / 1000; break;
    case NVML_CLOCK_MEM:
        *clock = GPU_DEFAULT_MEM_CLOCK_RATE_KHZ / 1000; break;
    case NVML_CLOCK_VIDEO:
        *clock = GPU_DEFAULT_CLOCK_RATE_KHZ / 1000; break;
    default: *clock = 0; break;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device,
                                        nvmlClockType_t type,
                                        unsigned int *clock)
{
    return nvmlDeviceGetClockInfo(device, type, clock);
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed)
{
    if (!speed) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *speed = 30;  /* 30% */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan,
                                       unsigned int *speed)
{
    (void)fan;
    return nvmlDeviceGetFanSpeed(device, speed);
}

/* ================================================================
 * NVML API — ECC
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device,
                                   unsigned int *current,
                                   unsigned int *pending)
{
    if (!current || !pending) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *current = GPU_DEFAULT_ECC_ENABLED;
    *pending = GPU_DEFAULT_ECC_ENABLED;
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Process info
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device,
                                                    unsigned int *infoCount,
                                                    void *infos)
{
    if (!infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device; (void)infos;
    *infoCount = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device,
                                                      unsigned int *infoCount,
                                                      void *infos)
{
    return nvmlDeviceGetComputeRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device,
                                                      unsigned int *infoCount,
                                                      void *infos)
{
    return nvmlDeviceGetComputeRunningProcesses(device, infoCount, infos);
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t device,
                                                     unsigned int *infoCount,
                                                     void *infos)
{
    if (!infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device; (void)infos;
    *infoCount = 0;
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Persistence mode
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device,
                                           unsigned int *mode)
{
    if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *mode = 0;  /* disabled */
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Performance state
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device,
                                            unsigned int *pState)
{
    if (!pState) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *pState = 0;  /* P0 = max performance */
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Serial / board info
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial,
                                  unsigned int length)
{
    if (!serial || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    snprintf(serial, length, "VGPU0000000001");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId)
{
    if (!boardId) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *boardId = 1;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version,
                                        unsigned int length)
{
    if (!version || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    snprintf(version, length, "96.00.74.00.02");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber,
                                           unsigned int length)
{
    if (!partNumber || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    snprintf(partNumber, length, "VGPU-H100-80GB");
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Miscellaneous stubs
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device,
                                       unsigned int *minorNumber)
{
    if (!minorNumber) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *minorNumber = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, unsigned int *type)
{
    if (!type) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *type = 12;  /* NVML_BRAND_NVIDIA_H100 (approximate) */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device,
                                        unsigned int *arch)
{
    if (!arch) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *arch = 9;  /* NVML_DEVICE_ARCH_HOPPER */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device,
                                         unsigned int *multiGpuBool)
{
    if (!multiGpuBool) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *multiGpuBool = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device,
                                       unsigned int *display)
{
    if (!display) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *display = 0;  /* disabled */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device,
                                         unsigned int *isActive)
{
    if (!isActive) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *isActive = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device,
                                       unsigned int *numCores)
{
    if (!numCores) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *numCores = GPU_DEFAULT_SM_COUNT * GPU_DEFAULT_CORES_PER_SM;
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Error string
 * ================================================================ */

const char* nvmlErrorString(nvmlReturn_t result)
{
    switch (result) {
    case NVML_SUCCESS:             return "Success";
    case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
    case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
    case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
    case NVML_ERROR_NOT_FOUND:     return "Not Found";
    case NVML_ERROR_DRIVER_NOT_LOADED: return "Driver Not Loaded";
    default:                       return "Unknown Error";
    }
}

/* ================================================================
 * NVML API — Stubs that return NOT_SUPPORTED
 *
 * nvidia-smi and some tools call these but they're not critical.
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device,
                                              unsigned int *utilization,
                                              unsigned int *samplingPeriodUs)
{
    if (utilization) *utilization = 0;
    if (samplingPeriodUs) *samplingPeriodUs = 0;
    (void)device;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device,
                                              unsigned int *utilization,
                                              unsigned int *samplingPeriodUs)
{
    if (utilization) *utilization = 0;
    if (samplingPeriodUs) *samplingPeriodUs = 0;
    (void)device;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device,
                                          void *bar1Memory)
{
    (void)device; (void)bar1Memory;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device,
                                                   unsigned int *currLinkGen)
{
    if (!currLinkGen) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *currLinkGen = 5;  /* PCIe Gen5 */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device,
                                              unsigned int *currLinkWidth)
{
    if (!currLinkWidth) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *currLinkWidth = 16;  /* x16 */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device,
                                                  unsigned int *maxLinkGen)
{
    if (!maxLinkGen) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *maxLinkGen = 5;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device,
                                             unsigned int *maxLinkWidth)
{
    if (!maxLinkWidth) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *maxLinkWidth = 16;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device,
                                          nvmlPcieUtilCounter_t counter,
                                          unsigned int *value)
{
    if (!value) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device; (void)counter;
    *value = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device,
                                          unsigned int object,
                                          char *version,
                                          unsigned int length)
{
    if (!version || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device; (void)object;
    snprintf(version, length, "N/A");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device,
                                               char *version,
                                               unsigned int length)
{
    if (!version || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    snprintf(version, length, "H100-VGPU");
    return NVML_SUCCESS;
}

/* ================================================================
 * NVML API — Virtualization
 * ================================================================ */

nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device,
                                              unsigned int *pVirtualMode)
{
    if (!pVirtualMode) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *pVirtualMode = 0;  /* NVML_GPU_VIRTUALIZATION_MODE_NONE */
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device,
                                        unsigned int *pHostVgpuMode)
{
    if (!pHostVgpuMode) return NVML_ERROR_INVALID_ARGUMENT;
    (void)device;
    *pHostVgpuMode = 0;
    return NVML_SUCCESS;
}

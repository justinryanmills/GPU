/*
 * Phase 3: NVML GPU Health Monitor
 *
 * Uses dlopen("libnvidia-ml.so.1") at runtime to avoid hard compile-time
 * dependency.  If the library is missing (e.g. dev machine without GPU),
 * all functions gracefully return "unavailable" without crashing.
 */

#include "nvml_monitor.h"
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

/* NVML type definitions (enough to call the functions we need) */
typedef int nvmlReturn_t;
typedef void *nvmlDevice_t;

typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

/* Function pointer types */
typedef nvmlReturn_t (*nvmlInit_v2_t)(void);
typedef nvmlReturn_t (*nvmlShutdown_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2_t)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*nvmlDeviceGetTemperature_t)(nvmlDevice_t, int, unsigned int *);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t *);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*nvmlDeviceGetPowerUsage_t)(nvmlDevice_t, unsigned int *);
typedef nvmlReturn_t (*nvmlDeviceGetTotalEccErrors_t)(nvmlDevice_t, int, int, unsigned long long *);

/* Global state */
static void *g_nvml_lib = NULL;
static nvmlDevice_t g_device = NULL;
static int g_available = 0;

/* Function pointers */
static nvmlInit_v2_t                     fp_init = NULL;
static nvmlShutdown_t                    fp_shutdown = NULL;
static nvmlDeviceGetHandleByIndex_v2_t   fp_getHandle = NULL;
static nvmlDeviceGetTemperature_t        fp_getTemp = NULL;
static nvmlDeviceGetUtilizationRates_t   fp_getUtil = NULL;
static nvmlDeviceGetMemoryInfo_t         fp_getMemInfo = NULL;
static nvmlDeviceGetPowerUsage_t         fp_getPower = NULL;
static nvmlDeviceGetTotalEccErrors_t     fp_getEcc = NULL;

/* Temperature sensor type */
#define NVML_TEMPERATURE_GPU 0

/* ECC error types */
#define NVML_MEMORY_ERROR_TYPE_UNCORRECTED 1
#define NVML_VOLATILE_ECC 0

/* ---- Public API -------------------------------------------------------- */

int nvml_init(void)
{
    if (g_available) return 0;  /* Already initialized */

    g_nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!g_nvml_lib) {
        g_nvml_lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
    }

    if (!g_nvml_lib) {
        fprintf(stderr, "[NVML] libnvidia-ml.so not found — GPU health monitoring disabled\n");
        return -1;
    }

    fp_init       = (nvmlInit_v2_t)dlsym(g_nvml_lib, "nvmlInit_v2");
    fp_shutdown   = (nvmlShutdown_t)dlsym(g_nvml_lib, "nvmlShutdown");
    fp_getHandle  = (nvmlDeviceGetHandleByIndex_v2_t)dlsym(g_nvml_lib, "nvmlDeviceGetHandleByIndex_v2");
    fp_getTemp    = (nvmlDeviceGetTemperature_t)dlsym(g_nvml_lib, "nvmlDeviceGetTemperature");
    fp_getUtil    = (nvmlDeviceGetUtilizationRates_t)dlsym(g_nvml_lib, "nvmlDeviceGetUtilizationRates");
    fp_getMemInfo = (nvmlDeviceGetMemoryInfo_t)dlsym(g_nvml_lib, "nvmlDeviceGetMemoryInfo");
    fp_getPower   = (nvmlDeviceGetPowerUsage_t)dlsym(g_nvml_lib, "nvmlDeviceGetPowerUsage");
    fp_getEcc     = (nvmlDeviceGetTotalEccErrors_t)dlsym(g_nvml_lib, "nvmlDeviceGetTotalEccErrors");

    if (!fp_init || !fp_shutdown || !fp_getHandle) {
        fprintf(stderr, "[NVML] Missing required symbols in libnvidia-ml.so\n");
        dlclose(g_nvml_lib);
        g_nvml_lib = NULL;
        return -1;
    }

    nvmlReturn_t rc = fp_init();
    if (rc != 0) {
        fprintf(stderr, "[NVML] nvmlInit_v2 failed (rc=%d)\n", rc);
        dlclose(g_nvml_lib);
        g_nvml_lib = NULL;
        return -1;
    }

    rc = fp_getHandle(0, &g_device);
    if (rc != 0) {
        fprintf(stderr, "[NVML] nvmlDeviceGetHandleByIndex(0) failed (rc=%d)\n", rc);
        fp_shutdown();
        dlclose(g_nvml_lib);
        g_nvml_lib = NULL;
        return -1;
    }

    g_available = 1;
    printf("[NVML] GPU health monitoring enabled\n");
    return 0;
}

void nvml_shutdown(void)
{
    if (g_nvml_lib) {
        if (fp_shutdown) fp_shutdown();
        dlclose(g_nvml_lib);
        g_nvml_lib = NULL;
    }
    g_available = 0;
    g_device = NULL;
}

void nvml_poll(nvml_health_t *health)
{
    memset(health, 0, sizeof(*health));

    if (!g_available || !g_device) {
        health->available = 0;
        return;
    }

    health->available = 1;

    if (fp_getTemp) {
        fp_getTemp(g_device, NVML_TEMPERATURE_GPU, &health->temperature_c);
    }

    if (fp_getUtil) {
        nvmlUtilization_t util;
        if (fp_getUtil(g_device, &util) == 0) {
            health->gpu_utilization = util.gpu;
            health->memory_utilization = util.memory;
        }
    }

    if (fp_getMemInfo) {
        nvmlMemory_t mem;
        if (fp_getMemInfo(g_device, &mem) == 0) {
            health->memory_used_mb = mem.used / (1024 * 1024);
            health->memory_total_mb = mem.total / (1024 * 1024);
        }
    }

    if (fp_getPower) {
        unsigned int power_mw = 0;
        if (fp_getPower(g_device, &power_mw) == 0) {
            health->power_watts = power_mw / 1000;
        }
    }

    if (fp_getEcc) {
        unsigned long long ecc = 0;
        fp_getEcc(g_device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                  NVML_VOLATILE_ECC, &ecc);
        health->ecc_errors = ecc;
    }

    /* Heuristic: flag GPU as needing reset if temp > 95°C or high ECC errors */
    if (health->temperature_c > 95 || health->ecc_errors > 100) {
        health->needs_reset = 1;
    }
}

int nvml_is_available(void)
{
    return g_available;
}

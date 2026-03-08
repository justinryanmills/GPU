/*
 * Phase 3: NVML GPU Health Monitor
 *
 * Uses dlopen() to load libnvidia-ml.so at runtime.
 * If the library is not available, all functions are no-ops.
 * Polls GPU temperature, utilization, ECC errors, and power usage.
 */

#ifndef NVML_MONITOR_H
#define NVML_MONITOR_H

#include <stdint.h>

/* GPU health snapshot */
typedef struct {
    int      available;          /* 1 if NVML is loaded and working           */
    uint32_t temperature_c;      /* GPU temperature in Celsius                */
    uint32_t gpu_utilization;    /* GPU utilization 0-100%                    */
    uint32_t memory_utilization; /* Memory utilization 0-100%                 */
    uint64_t memory_used_mb;     /* Memory used in MB                         */
    uint64_t memory_total_mb;    /* Total memory in MB                        */
    uint32_t power_watts;        /* Current power draw in watts               */
    uint64_t ecc_errors;         /* Cumulative ECC errors                     */
    int      needs_reset;        /* 1 if GPU shows signs of needing reset     */
} nvml_health_t;

/* Initialize NVML (dlopen-based).  Returns 0 on success, -1 if unavailable. */
int nvml_init(void);

/* Shutdown NVML and unload library */
void nvml_shutdown(void);

/* Poll current GPU health.  Always succeeds; if NVML unavailable, available=0. */
void nvml_poll(nvml_health_t *health);

/* Check if NVML is loaded */
int nvml_is_available(void);

#endif /* NVML_MONITOR_H */

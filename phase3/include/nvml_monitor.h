#ifndef NVML_MONITOR_H
#define NVML_MONITOR_H

#include <stdint.h>

typedef struct {
    int      available;
    uint32_t temperature_c;
    uint32_t gpu_utilization;
    uint32_t memory_utilization;
    uint64_t memory_used_mb;
    uint64_t memory_total_mb;
    uint32_t power_watts;
    uint64_t ecc_errors;
    int      needs_reset;
} nvml_health_t;

int nvml_init(void);
void nvml_shutdown(void);
void nvml_poll(nvml_health_t *health);
int nvml_is_available(void);

#endif /* NVML_MONITOR_H */

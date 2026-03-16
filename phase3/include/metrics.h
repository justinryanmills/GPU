#ifndef METRICS_H
#define METRICS_H

#include <stdint.h>
#include <pthread.h>
#include <time.h>

#define METRICS_MAX_VMS       64
#define METRICS_WINDOW_SIZE   1000

typedef struct {
    uint32_t vm_id;
    int      active;
    uint64_t latency_samples[METRICS_WINDOW_SIZE];
    int      sample_count;
    int      sample_head;
    uint64_t total_jobs;
    uint64_t total_errors;
    uint64_t total_rejected;
    uint64_t total_exec_time_us;
} metrics_vm_t;

struct metrics {
    metrics_vm_t vms[METRICS_MAX_VMS];
    pthread_mutex_t lock;
    uint64_t global_jobs;
    uint64_t global_errors;
    uint64_t global_rejected;
    uint64_t context_switches;
    uint64_t gpu_resets;
    struct timespec start_time;
};
typedef struct metrics metrics_t;

void metrics_init(metrics_t *m);
void metrics_destroy(metrics_t *m);
void metrics_record_job(metrics_t *m, uint32_t vm_id, uint64_t latency_us,
                        uint64_t exec_time_us);
void metrics_record_error(metrics_t *m, uint32_t vm_id);
void metrics_record_rejection(metrics_t *m, uint32_t vm_id);
void metrics_record_context_switch(metrics_t *m);
void metrics_record_gpu_reset(metrics_t *m);
uint64_t metrics_percentile(metrics_t *m, uint32_t vm_id, int percentile);
uint64_t metrics_global_percentile(metrics_t *m, int percentile);

int metrics_export_prometheus(metrics_t *m, char *buf, int buf_size);
int metrics_export_summary(metrics_t *m, char *buf, int buf_size);

#endif /* METRICS_H */

/*
 * Phase 3: Metrics (latency, percentiles, throughput, Prometheus export).
 */
#ifndef METRICS_H
#define METRICS_H

#include <stdint.h>
#include <pthread.h>
#include <time.h>

#define METRICS_MAX_VMS       64
#define METRICS_WINDOW_SIZE   1000  /* Last N samples for percentile calc */

/* Per-VM metrics */
typedef struct {
    uint32_t vm_id;
    int      active;

    /* Sliding-window latency samples (microseconds) */
    uint64_t latency_samples[METRICS_WINDOW_SIZE];
    int      sample_count;      /* Total samples written (wraps)             */
    int      sample_head;       /* Next write position (circular buffer)     */

    /* Counters */
    uint64_t total_jobs;
    uint64_t total_errors;
    uint64_t total_rejected;     /* Rate-limited or quarantined rejections   */
    uint64_t total_exec_time_us; /* Cumulative CUDA execution time           */
} metrics_vm_t;

/* Global metrics */
struct metrics {
    metrics_vm_t vms[METRICS_MAX_VMS];
    pthread_mutex_t lock;

    /* Global counters */
    uint64_t global_jobs;
    uint64_t global_errors;
    uint64_t global_rejected;
    uint64_t context_switches;
    uint64_t gpu_resets;

    struct timespec start_time;  /* When metrics collection started           */
};
typedef struct metrics metrics_t;

/* Initialize metrics collector */
void metrics_init(metrics_t *m);

/* Destroy */
void metrics_destroy(metrics_t *m);

/* Record a completed job latency (total time from enqueue to response, in us) */
void metrics_record_job(metrics_t *m, uint32_t vm_id, uint64_t latency_us,
                        uint64_t exec_time_us);

/* Record an error */
void metrics_record_error(metrics_t *m, uint32_t vm_id);

/* Record a rejection (rate limit or quarantine) */
void metrics_record_rejection(metrics_t *m, uint32_t vm_id);

/* Record a context switch */
void metrics_record_context_switch(metrics_t *m);

/* Record a GPU reset */
void metrics_record_gpu_reset(metrics_t *m);

/* Get percentile latency for a VM (0-100, e.g. 50, 95, 99) in microseconds */
uint64_t metrics_percentile(metrics_t *m, uint32_t vm_id, int percentile);

/* Get global percentile latency */
uint64_t metrics_global_percentile(metrics_t *m, int percentile);

/*
 * Export metrics in Prometheus text format.
 * Writes to buf (max buf_size bytes).  Returns bytes written.
 */
int metrics_export_prometheus(metrics_t *m, char *buf, int buf_size);

/*
 * Export human-readable metrics summary.
 * Writes to buf (max buf_size bytes).  Returns bytes written.
 */
int metrics_export_summary(metrics_t *m, char *buf, int buf_size);

#endif /* METRICS_H */

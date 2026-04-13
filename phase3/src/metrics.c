/*
 * Phase 3: Metrics Collector
 *
 * Tracks per-VM and global latency histograms (sliding-window circular buffer),
 * computes p50/p95/p99 percentiles on demand, and exports in Prometheus format.
 */

#include "metrics.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ---- Internal ---------------------------------------------------------- */

static metrics_vm_t *find_or_create_vm(metrics_t *m, uint32_t vm_id)
{
    for (int i = 0; i < METRICS_MAX_VMS; i++) {
        if (m->vms[i].active && m->vms[i].vm_id == vm_id)
            return &m->vms[i];
    }
    /* Allocate new */
    for (int i = 0; i < METRICS_MAX_VMS; i++) {
        if (!m->vms[i].active) {
            memset(&m->vms[i], 0, sizeof(m->vms[i]));
            m->vms[i].vm_id = vm_id;
            m->vms[i].active = 1;
            return &m->vms[i];
        }
    }
    return NULL;
}

/* Comparison function for qsort */
static int cmp_uint64(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

/*
 * Compute percentile from a circular buffer of samples.
 * Returns 0 if no samples available.
 */
static uint64_t compute_percentile(const uint64_t *samples, int sample_count,
                                    int window_size, int percentile)
{
    int n = (sample_count < window_size) ? sample_count : window_size;
    if (n == 0) return 0;

    /* Copy to temp array and sort */
    uint64_t *tmp = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tmp) return 0;

    if (sample_count <= window_size) {
        memcpy(tmp, samples, n * sizeof(uint64_t));
    } else {
        /* Buffer has wrapped; just use all window_size entries */
        memcpy(tmp, samples, window_size * sizeof(uint64_t));
        n = window_size;
    }

    qsort(tmp, n, sizeof(uint64_t), cmp_uint64);

    int idx = (percentile * n) / 100;
    if (idx >= n) idx = n - 1;
    uint64_t result = tmp[idx];

    free(tmp);
    return result;
}

/* ---- Public API -------------------------------------------------------- */

void metrics_init(metrics_t *m)
{
    memset(m, 0, sizeof(*m));
    pthread_mutex_init(&m->lock, NULL);
    clock_gettime(CLOCK_MONOTONIC, &m->start_time);
}

void metrics_destroy(metrics_t *m)
{
    pthread_mutex_destroy(&m->lock);
}

void metrics_record_job(metrics_t *m, uint32_t vm_id, uint64_t latency_us,
                        uint64_t exec_time_us)
{
    pthread_mutex_lock(&m->lock);

    m->global_jobs++;

    metrics_vm_t *vm = find_or_create_vm(m, vm_id);
    if (vm) {
        vm->total_jobs++;
        vm->total_exec_time_us += exec_time_us;

        /* Record latency sample in circular buffer */
        vm->latency_samples[vm->sample_head] = latency_us;
        vm->sample_head = (vm->sample_head + 1) % METRICS_WINDOW_SIZE;
        vm->sample_count++;
    }

    pthread_mutex_unlock(&m->lock);
}

void metrics_record_error(metrics_t *m, uint32_t vm_id)
{
    pthread_mutex_lock(&m->lock);
    m->global_errors++;
    metrics_vm_t *vm = find_or_create_vm(m, vm_id);
    if (vm) vm->total_errors++;
    pthread_mutex_unlock(&m->lock);
}

void metrics_record_rejection(metrics_t *m, uint32_t vm_id)
{
    pthread_mutex_lock(&m->lock);
    m->global_rejected++;
    metrics_vm_t *vm = find_or_create_vm(m, vm_id);
    if (vm) vm->total_rejected++;
    pthread_mutex_unlock(&m->lock);
}

void metrics_record_context_switch(metrics_t *m)
{
    pthread_mutex_lock(&m->lock);
    m->context_switches++;
    pthread_mutex_unlock(&m->lock);
}

void metrics_record_gpu_reset(metrics_t *m)
{
    pthread_mutex_lock(&m->lock);
    m->gpu_resets++;
    pthread_mutex_unlock(&m->lock);
}

uint64_t metrics_percentile(metrics_t *m, uint32_t vm_id, int percentile)
{
    pthread_mutex_lock(&m->lock);
    uint64_t result = 0;
    for (int i = 0; i < METRICS_MAX_VMS; i++) {
        if (m->vms[i].active && m->vms[i].vm_id == vm_id) {
            result = compute_percentile(m->vms[i].latency_samples,
                                         m->vms[i].sample_count,
                                         METRICS_WINDOW_SIZE, percentile);
            break;
        }
    }
    pthread_mutex_unlock(&m->lock);
    return result;
}

uint64_t metrics_global_percentile(metrics_t *m, int percentile)
{
    pthread_mutex_lock(&m->lock);

    /* Merge all VM samples into a single temp buffer */
    int total = 0;
    uint64_t tmp[METRICS_MAX_VMS * METRICS_WINDOW_SIZE];

    for (int i = 0; i < METRICS_MAX_VMS; i++) {
        if (!m->vms[i].active) continue;
        int n = (m->vms[i].sample_count < METRICS_WINDOW_SIZE)
              ? m->vms[i].sample_count : METRICS_WINDOW_SIZE;
        for (int j = 0; j < n && total < (int)(sizeof(tmp)/sizeof(tmp[0])); j++) {
            tmp[total++] = m->vms[i].latency_samples[j];
        }
    }

    pthread_mutex_unlock(&m->lock);

    if (total == 0) return 0;

    qsort(tmp, total, sizeof(uint64_t), cmp_uint64);
    int idx = (percentile * total) / 100;
    if (idx >= total) idx = total - 1;
    return tmp[idx];
}

int metrics_export_prometheus(metrics_t *m, char *buf, int buf_size)
{
    pthread_mutex_lock(&m->lock);

    int written = 0;
    int n;

    n = snprintf(buf + written, buf_size - written,
        "# HELP vgpu_jobs_total Total jobs processed\n"
        "# TYPE vgpu_jobs_total counter\n"
        "vgpu_jobs_total %lu\n\n"
        "# HELP vgpu_errors_total Total errors\n"
        "# TYPE vgpu_errors_total counter\n"
        "vgpu_errors_total %lu\n\n"
        "# HELP vgpu_rejected_total Total rejected requests\n"
        "# TYPE vgpu_rejected_total counter\n"
        "vgpu_rejected_total %lu\n\n"
        "# HELP vgpu_context_switches_total Context switches\n"
        "# TYPE vgpu_context_switches_total counter\n"
        "vgpu_context_switches_total %lu\n\n"
        "# HELP vgpu_gpu_resets_total GPU resets detected\n"
        "# TYPE vgpu_gpu_resets_total counter\n"
        "vgpu_gpu_resets_total %lu\n\n",
        (unsigned long)m->global_jobs,
        (unsigned long)m->global_errors,
        (unsigned long)m->global_rejected,
        (unsigned long)m->context_switches,
        (unsigned long)m->gpu_resets);
    if (n > 0) written += n;

    /* Per-VM metrics */
    n = snprintf(buf + written, buf_size - written,
        "# HELP vgpu_vm_jobs_total Per-VM total jobs\n"
        "# TYPE vgpu_vm_jobs_total counter\n");
    if (n > 0) written += n;

    for (int i = 0; i < METRICS_MAX_VMS && written < buf_size; i++) {
        if (!m->vms[i].active) continue;
        n = snprintf(buf + written, buf_size - written,
            "vgpu_vm_jobs_total{vm_id=\"%u\"} %lu\n",
            m->vms[i].vm_id, (unsigned long)m->vms[i].total_jobs);
        if (n > 0) written += n;
    }

    n = snprintf(buf + written, buf_size - written,
        "\n# HELP vgpu_vm_latency_p95_us Per-VM p95 latency in microseconds\n"
        "# TYPE vgpu_vm_latency_p95_us gauge\n");
    if (n > 0) written += n;

    for (int i = 0; i < METRICS_MAX_VMS && written < buf_size; i++) {
        if (!m->vms[i].active) continue;
        uint64_t p95 = compute_percentile(m->vms[i].latency_samples,
                                           m->vms[i].sample_count,
                                           METRICS_WINDOW_SIZE, 95);
        n = snprintf(buf + written, buf_size - written,
            "vgpu_vm_latency_p95_us{vm_id=\"%u\"} %lu\n",
            m->vms[i].vm_id, (unsigned long)p95);
        if (n > 0) written += n;
    }

    n = snprintf(buf + written, buf_size - written,
        "\n# HELP vgpu_vm_latency_p99_us Per-VM p99 latency in microseconds\n"
        "# TYPE vgpu_vm_latency_p99_us gauge\n");
    if (n > 0) written += n;

    for (int i = 0; i < METRICS_MAX_VMS && written < buf_size; i++) {
        if (!m->vms[i].active) continue;
        uint64_t p99 = compute_percentile(m->vms[i].latency_samples,
                                           m->vms[i].sample_count,
                                           METRICS_WINDOW_SIZE, 99);
        n = snprintf(buf + written, buf_size - written,
            "vgpu_vm_latency_p99_us{vm_id=\"%u\"} %lu\n",
            m->vms[i].vm_id, (unsigned long)p99);
        if (n > 0) written += n;
    }

    pthread_mutex_unlock(&m->lock);
    return written;
}

int metrics_export_summary(metrics_t *m, char *buf, int buf_size)
{
    pthread_mutex_lock(&m->lock);

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double uptime = (now.tv_sec - m->start_time.tv_sec)
                  + (now.tv_nsec - m->start_time.tv_nsec) / 1e9;

    int written = 0;
    int n;

    n = snprintf(buf + written, buf_size - written,
        "=== vGPU Mediator Metrics ===\n"
        "Uptime:            %.1f seconds\n"
        "Total jobs:        %lu\n"
        "Total errors:      %lu\n"
        "Total rejected:    %lu\n"
        "Context switches:  %lu\n"
        "GPU resets:        %lu\n\n",
        uptime,
        (unsigned long)m->global_jobs,
        (unsigned long)m->global_errors,
        (unsigned long)m->global_rejected,
        (unsigned long)m->context_switches,
        (unsigned long)m->gpu_resets);
    if (n > 0) written += n;

    /* Per-VM breakdown */
    for (int i = 0; i < METRICS_MAX_VMS && written < buf_size; i++) {
        if (!m->vms[i].active) continue;

        uint64_t p50 = compute_percentile(m->vms[i].latency_samples,
                                           m->vms[i].sample_count,
                                           METRICS_WINDOW_SIZE, 50);
        uint64_t p95 = compute_percentile(m->vms[i].latency_samples,
                                           m->vms[i].sample_count,
                                           METRICS_WINDOW_SIZE, 95);
        uint64_t p99 = compute_percentile(m->vms[i].latency_samples,
                                           m->vms[i].sample_count,
                                           METRICS_WINDOW_SIZE, 99);

        double avg_exec = (m->vms[i].total_jobs > 0)
            ? (double)m->vms[i].total_exec_time_us / m->vms[i].total_jobs
            : 0;

        n = snprintf(buf + written, buf_size - written,
            "--- VM %u ---\n"
            "  Jobs: %lu  Errors: %lu  Rejected: %lu\n"
            "  Latency (us): p50=%lu  p95=%lu  p99=%lu\n"
            "  Avg exec time: %.0f us\n\n",
            m->vms[i].vm_id,
            (unsigned long)m->vms[i].total_jobs,
            (unsigned long)m->vms[i].total_errors,
            (unsigned long)m->vms[i].total_rejected,
            (unsigned long)p50, (unsigned long)p95, (unsigned long)p99,
            avg_exec);
        if (n > 0) written += n;
    }

    pthread_mutex_unlock(&m->lock);
    return written;
}

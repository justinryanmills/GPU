/*
 * Phase 3: Watchdog & Error Recovery
 *
 * Runs a background thread that:
 *   1. Monitors the currently executing CUDA job for timeout
 *   2. Tracks per-VM fault counters
 *   3. Auto-quarantines VMs that exceed the fault threshold
 *   4. Polls NVML for GPU health (via nvml_monitor.h)
 */

#include "watchdog.h"
#include "nvml_monitor.h"
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

/* ---- Internal ---------------------------------------------------------- */

static wd_vm_state_t *find_or_create_vm(watchdog_t *wd, uint32_t vm_id)
{
    for (int i = 0; i < wd->num_vms; i++) {
        if (wd->vm_states[i].active && wd->vm_states[i].vm_id == vm_id)
            return &wd->vm_states[i];
    }
    if (wd->num_vms >= WD_MAX_VMS) return NULL;
    wd_vm_state_t *vs = &wd->vm_states[wd->num_vms++];
    memset(vs, 0, sizeof(*vs));
    vs->vm_id = vm_id;
    vs->active = 1;
    return vs;
}

static void *watchdog_thread(void *arg)
{
    watchdog_t *wd = (watchdog_t *)arg;
    nvml_health_t health;

    printf("[WATCHDOG] Started (job_timeout=%ds, fault_threshold=%d)\n",
           wd->job_timeout_sec, wd->fault_threshold);

    while (wd->running) {
        usleep(WD_POLL_INTERVAL_MS * 1000);

        pthread_mutex_lock(&wd->lock);

        /* 1. Check for job timeout */
        if (wd->active_job.active) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - wd->active_job.start_time.tv_sec)
                           + (now.tv_nsec - wd->active_job.start_time.tv_nsec) / 1e9;

            if (elapsed > wd->job_timeout_sec) {
                printf("[WATCHDOG] Job timeout: vm=%u req=%u (%.1fs > %ds)\n",
                       wd->active_job.vm_id, wd->active_job.request_id,
                       elapsed, wd->job_timeout_sec);

                /* Record fault for this VM */
                wd_vm_state_t *vs = find_or_create_vm(wd, wd->active_job.vm_id);
                if (vs) {
                    vs->error_count++;
                    clock_gettime(CLOCK_MONOTONIC, &vs->last_error_time);

                    if (vs->error_count >= wd->fault_threshold && !vs->quarantined) {
                        vs->quarantined = 1;
                        printf("[WATCHDOG] Auto-quarantine: vm=%u (errors=%d)\n",
                               vs->vm_id, vs->error_count);
                    }
                }

                /* Mark job as no longer active (mediator should handle the timeout) */
                wd->active_job.active = 0;
            }
        }

        /* 2. Poll NVML for GPU health */
        nvml_poll(&health);
        if (health.available && health.needs_reset) {
            if (!wd->gpu_reset_detected) {
                wd->gpu_reset_detected = 1;
                wd->total_resets++;
                printf("[WATCHDOG] GPU needs reset! (temp=%u°C, ecc=%lu)\n",
                       health.temperature_c, (unsigned long)health.ecc_errors);
            }
        } else {
            wd->gpu_reset_detected = 0;
        }

        pthread_mutex_unlock(&wd->lock);
    }

    printf("[WATCHDOG] Stopped\n");
    return NULL;
}

/* ---- Public API -------------------------------------------------------- */

void wd_init(watchdog_t *wd)
{
    memset(wd, 0, sizeof(*wd));
    pthread_mutex_init(&wd->lock, NULL);
    wd->job_timeout_sec = WD_DEFAULT_JOB_TIMEOUT;
    wd->fault_threshold = WD_FAULT_THRESHOLD;
}

void wd_destroy(watchdog_t *wd)
{
    if (wd->running) wd_stop(wd);
    pthread_mutex_destroy(&wd->lock);
}

int wd_start(watchdog_t *wd)
{
    if (wd->running) return 0;
    wd->running = 1;

    if (pthread_create(&wd->thread, NULL, watchdog_thread, wd) != 0) {
        wd->running = 0;
        return -1;
    }

    return 0;
}

void wd_stop(watchdog_t *wd)
{
    wd->running = 0;
    pthread_join(wd->thread, NULL);
}

void wd_job_started(watchdog_t *wd, uint32_t vm_id, uint32_t request_id)
{
    pthread_mutex_lock(&wd->lock);
    wd->active_job.vm_id = vm_id;
    wd->active_job.request_id = request_id;
    wd->active_job.active = 1;
    clock_gettime(CLOCK_MONOTONIC, &wd->active_job.start_time);
    pthread_mutex_unlock(&wd->lock);
}

void wd_job_completed(watchdog_t *wd, uint32_t vm_id, uint32_t request_id)
{
    (void)vm_id;
    (void)request_id;
    pthread_mutex_lock(&wd->lock);
    wd->active_job.active = 0;
    pthread_mutex_unlock(&wd->lock);
}

void wd_job_failed(watchdog_t *wd, uint32_t vm_id, uint32_t request_id)
{
    (void)request_id;
    pthread_mutex_lock(&wd->lock);
    wd->active_job.active = 0;

    wd_vm_state_t *vs = find_or_create_vm(wd, vm_id);
    if (vs) {
        vs->error_count++;
        clock_gettime(CLOCK_MONOTONIC, &vs->last_error_time);

        if (vs->error_count >= wd->fault_threshold && !vs->quarantined) {
            vs->quarantined = 1;
            printf("[WATCHDOG] Auto-quarantine: vm=%u (errors=%d)\n",
                   vs->vm_id, vs->error_count);
        }
    }

    pthread_mutex_unlock(&wd->lock);
}

int wd_is_quarantined(watchdog_t *wd, uint32_t vm_id)
{
    pthread_mutex_lock(&wd->lock);
    int result = 0;
    for (int i = 0; i < wd->num_vms; i++) {
        if (wd->vm_states[i].active && wd->vm_states[i].vm_id == vm_id) {
            result = wd->vm_states[i].quarantined;
            break;
        }
    }
    pthread_mutex_unlock(&wd->lock);
    return result;
}

void wd_clear_quarantine(watchdog_t *wd, uint32_t vm_id)
{
    pthread_mutex_lock(&wd->lock);
    for (int i = 0; i < wd->num_vms; i++) {
        if (wd->vm_states[i].active && wd->vm_states[i].vm_id == vm_id) {
            wd->vm_states[i].quarantined = 0;
            wd->vm_states[i].error_count = 0;
            printf("[WATCHDOG] Quarantine cleared: vm=%u\n", vm_id);
            break;
        }
    }
    pthread_mutex_unlock(&wd->lock);
}

int wd_job_timed_out(watchdog_t *wd)
{
    pthread_mutex_lock(&wd->lock);
    int result = 0;
    if (wd->active_job.active) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - wd->active_job.start_time.tv_sec)
                       + (now.tv_nsec - wd->active_job.start_time.tv_nsec) / 1e9;
        result = (elapsed > wd->job_timeout_sec);
    }
    pthread_mutex_unlock(&wd->lock);
    return result;
}

uint64_t wd_total_resets(watchdog_t *wd)
{
    pthread_mutex_lock(&wd->lock);
    uint64_t r = wd->total_resets;
    pthread_mutex_unlock(&wd->lock);
    return r;
}

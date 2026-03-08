/*
 * Phase 3: Watchdog & error recovery (timeouts, quarantine, GPU reset).
 */
#ifndef WATCHDOG_H
#define WATCHDOG_H

#include <stdint.h>
#include <pthread.h>
#include <time.h>

/* Forward declarations (the watchdog references these opaquely) */
typedef struct wfq_scheduler wfq_scheduler_t;
typedef struct metrics metrics_t;

#define WD_MAX_VMS              64
#define WD_DEFAULT_JOB_TIMEOUT  30    /* seconds per job before considered stuck */
#define WD_FAULT_THRESHOLD      5     /* errors before auto-quarantine           */
#define WD_POLL_INTERVAL_MS     1000  /* watchdog tick interval                  */

/* Per-VM fault tracking */
typedef struct {
    uint32_t vm_id;
    int      active;
    int      error_count;           /* Errors since last reset                 */
    int      quarantined;           /* 1 if auto-quarantined                   */
    struct timespec last_error_time;
} wd_vm_state_t;

/* Active job tracking (for timeout detection) */
typedef struct {
    uint32_t vm_id;
    uint32_t request_id;
    struct timespec start_time;
    int      active;
} wd_active_job_t;

/* Watchdog state */
typedef struct {
    pthread_t        thread;
    int              running;
    pthread_mutex_t  lock;

    /* Job timeout tracking */
    wd_active_job_t  active_job;   /* Currently executing CUDA job             */
    int              job_timeout_sec;

    /* Per-VM fault tracking */
    wd_vm_state_t    vm_states[WD_MAX_VMS];
    int              num_vms;
    int              fault_threshold;

    /* GPU reset detection */
    int              gpu_reset_detected;
    uint64_t         total_resets;
} watchdog_t;

/* Initialize watchdog (does not start thread) */
void wd_init(watchdog_t *wd);

/* Destroy */
void wd_destroy(watchdog_t *wd);

/* Start watchdog background thread */
int wd_start(watchdog_t *wd);

/* Stop watchdog background thread */
void wd_stop(watchdog_t *wd);

/* Notify watchdog that a job has started executing */
void wd_job_started(watchdog_t *wd, uint32_t vm_id, uint32_t request_id);

/* Notify watchdog that a job has completed */
void wd_job_completed(watchdog_t *wd, uint32_t vm_id, uint32_t request_id);

/* Notify watchdog that a job has failed */
void wd_job_failed(watchdog_t *wd, uint32_t vm_id, uint32_t request_id);

/* Check if a VM is quarantined by the watchdog */
int wd_is_quarantined(watchdog_t *wd, uint32_t vm_id);

/* Manually clear quarantine for a VM */
void wd_clear_quarantine(watchdog_t *wd, uint32_t vm_id);

/* Check if the current job has timed out */
int wd_job_timed_out(watchdog_t *wd);

/* Get total GPU resets detected */
uint64_t wd_total_resets(watchdog_t *wd);

#endif /* WATCHDOG_H */

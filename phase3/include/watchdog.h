#ifndef WATCHDOG_H
#define WATCHDOG_H

#include <stdint.h>
#include <pthread.h>
#include <time.h>

typedef struct wfq_scheduler wfq_scheduler_t;
typedef struct metrics metrics_t;

#define WD_MAX_VMS              64
#define WD_DEFAULT_JOB_TIMEOUT  30
#define WD_FAULT_THRESHOLD      5
#define WD_POLL_INTERVAL_MS     1000

typedef struct {
    uint32_t vm_id;
    int      active;
    int      error_count;
    int      quarantined;
    struct timespec last_error_time;
} wd_vm_state_t;

typedef struct {
    uint32_t vm_id;
    uint32_t request_id;
    struct timespec start_time;
    int      active;
} wd_active_job_t;

typedef struct {
    pthread_t        thread;
    int              running;
    pthread_mutex_t  lock;
    wd_active_job_t  active_job;
    int              job_timeout_sec;
    wd_vm_state_t    vm_states[WD_MAX_VMS];
    int              num_vms;
    int              fault_threshold;
    int              gpu_reset_detected;
    uint64_t         total_resets;
} watchdog_t;

void wd_init(watchdog_t *wd);
void wd_destroy(watchdog_t *wd);
int wd_start(watchdog_t *wd);
void wd_stop(watchdog_t *wd);
void wd_job_started(watchdog_t *wd, uint32_t vm_id, uint32_t request_id);
void wd_job_completed(watchdog_t *wd, uint32_t vm_id, uint32_t request_id);
void wd_job_failed(watchdog_t *wd, uint32_t vm_id, uint32_t request_id);
int wd_is_quarantined(watchdog_t *wd, uint32_t vm_id);
void wd_clear_quarantine(watchdog_t *wd, uint32_t vm_id);
int wd_job_timed_out(watchdog_t *wd);
uint64_t wd_total_resets(watchdog_t *wd);

#endif /* WATCHDOG_H */

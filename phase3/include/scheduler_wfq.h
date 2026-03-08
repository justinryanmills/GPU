/*
 * Phase 3: Demand-Aware Weighted Fair Queuing Scheduler
 *
 * Replaces the simple priority-sorted linked list from Phase 2 with a
 * scheduler that computes an urgency score per request:
 *
 *   urgency = base_priority_mult
 *           * (weight / 50.0)
 *           * (1.0 + queue_pressure)
 *           * (1.0 + wait_time_sec / 10.0)
 *
 * Higher urgency → dequeued first.  Ties broken by arrival order (FIFO).
 */

#ifndef SCHEDULER_WFQ_H
#define SCHEDULER_WFQ_H

#include <stdint.h>
#include <time.h>
#include <pthread.h>

/* Maximum VMs the scheduler tracks simultaneously */
#define WFQ_MAX_VMS         64
#define WFQ_MAX_QUEUE_SIZE  1024

/* Per-request entry stored in the scheduler */
typedef struct wfq_entry {
    /* Identification */
    uint32_t vm_id;
    uint32_t request_id;
    char     pool_id;        /* 'A' or 'B' */
    uint8_t  priority;       /* 0=low, 1=med, 2=high */

    /* Scheduling metadata */
    int      weight;         /* VM scheduler weight (1-100)       */
    double   urgency;        /* Computed urgency score            */
    struct timespec enqueue_time;  /* When the request was enqueued */

    /* Request data (copied from the incoming VGPURequest) */
    int      num1, num2;
    int      client_fd;      /* Socket fd to send response to     */

    /* Payload pass-through */
    uint8_t  payload[1024];
    uint16_t payload_len;
} wfq_entry_t;

/* Per-VM statistics used by the scheduler to measure demand */
typedef struct {
    uint32_t vm_id;
    int      current_queue_depth; /* How many jobs from this VM are queued    */
    double   recent_submit_rate;  /* Jobs/sec over last window                */
    double   avg_exec_time_us;    /* Rolling average execution time           */
    uint64_t total_submitted;
    uint64_t total_completed;
    struct timespec last_submit_time;
} wfq_vm_stats_t;

/* Scheduler state */
struct wfq_scheduler {
    wfq_entry_t   queue[WFQ_MAX_QUEUE_SIZE];
    int           queue_len;
    pthread_mutex_t lock;

    wfq_vm_stats_t vm_stats[WFQ_MAX_VMS];
    int            num_vms;

    uint64_t context_switches;     /* How many times the "active VM" changed */
    uint32_t last_dispatched_vm;   /* VM ID of last dispatched request       */
};
typedef struct wfq_scheduler wfq_scheduler_t;

/* Initialize the WFQ scheduler */
void wfq_init(wfq_scheduler_t *sched);

/* Destroy (cleanup) */
void wfq_destroy(wfq_scheduler_t *sched);

/*
 * Enqueue a request.  The scheduler computes the urgency score internally.
 * Returns 0 on success, -1 if the queue is full.
 */
int wfq_enqueue(wfq_scheduler_t *sched, const wfq_entry_t *entry);

/*
 * Dequeue the highest-urgency request.
 * Copies the entry into *out and returns 0, or returns -1 if the queue is empty.
 */
int wfq_dequeue(wfq_scheduler_t *sched, wfq_entry_t *out);

/*
 * Notify the scheduler that a request completed (updates per-VM stats).
 * exec_time_us is the CUDA execution time in microseconds.
 */
void wfq_complete(wfq_scheduler_t *sched, uint32_t vm_id, uint32_t exec_time_us);

/* Get queue depth for a specific VM */
int wfq_vm_queue_depth(wfq_scheduler_t *sched, uint32_t vm_id);

/* Get total queue length */
int wfq_queue_len(wfq_scheduler_t *sched);

/* Get context switch count */
uint64_t wfq_context_switches(wfq_scheduler_t *sched);

/* Get per-VM stats (returns pointer, or NULL if vm_id not found) */
const wfq_vm_stats_t *wfq_get_vm_stats(wfq_scheduler_t *sched, uint32_t vm_id);

#endif /* SCHEDULER_WFQ_H */

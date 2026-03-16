#ifndef SCHEDULER_WFQ_H
#define SCHEDULER_WFQ_H

#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define WFQ_MAX_VMS         64
#define WFQ_MAX_QUEUE_SIZE  1024

typedef struct wfq_entry {
    uint32_t vm_id;
    uint32_t request_id;
    char     pool_id;
    uint8_t  priority;
    int      weight;
    double   urgency;
    struct timespec enqueue_time;
    int      num1, num2;
    int      client_fd;
    uint8_t  payload[1024];
    uint16_t payload_len;
} wfq_entry_t;

typedef struct {
    uint32_t vm_id;
    int      current_queue_depth;
    double   recent_submit_rate;
    double   avg_exec_time_us;
    uint64_t total_submitted;
    uint64_t total_completed;
    struct timespec last_submit_time;
} wfq_vm_stats_t;

struct wfq_scheduler {
    wfq_entry_t   queue[WFQ_MAX_QUEUE_SIZE];
    int           queue_len;
    pthread_mutex_t lock;
    wfq_vm_stats_t vm_stats[WFQ_MAX_VMS];
    int            num_vms;
    uint64_t context_switches;
    uint32_t last_dispatched_vm;
};
typedef struct wfq_scheduler wfq_scheduler_t;

void wfq_init(wfq_scheduler_t *sched);
void wfq_destroy(wfq_scheduler_t *sched);
int wfq_enqueue(wfq_scheduler_t *sched, const wfq_entry_t *entry);
int wfq_dequeue(wfq_scheduler_t *sched, wfq_entry_t *out);
void wfq_complete(wfq_scheduler_t *sched, uint32_t vm_id, uint32_t exec_time_us);
int wfq_vm_queue_depth(wfq_scheduler_t *sched, uint32_t vm_id);
int wfq_queue_len(wfq_scheduler_t *sched);
uint64_t wfq_context_switches(wfq_scheduler_t *sched);
const wfq_vm_stats_t *wfq_get_vm_stats(wfq_scheduler_t *sched, uint32_t vm_id);

#endif /* SCHEDULER_WFQ_H */

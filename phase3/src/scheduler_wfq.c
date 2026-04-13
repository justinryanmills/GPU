/*
 * Phase 3: Demand-Aware Weighted Fair Queuing Scheduler
 *
 * Urgency score formula:
 *   base_mult = priority==2 ? 4.0 : priority==1 ? 2.0 : 1.0
 *   weight_mult = weight / 50.0       (weight 1-100, default 50)
 *   pressure    = vm_queue_depth / 10.0
 *   wait_bonus  = elapsed_sec / 10.0
 *   urgency     = base_mult * weight_mult * (1.0 + pressure) * (1.0 + wait_bonus)
 *
 * The queue is a simple array sorted on dequeue (selection sort of max).
 * For the expected queue sizes (tens to low hundreds), this is efficient
 * enough and avoids heap complexity.
 */

#include "scheduler_wfq.h"
#include <string.h>
#include <stdio.h>

/* ---- Internal helpers -------------------------------------------------- */

static wfq_vm_stats_t *find_or_create_vm(wfq_scheduler_t *sched, uint32_t vm_id)
{
    for (int i = 0; i < sched->num_vms; i++) {
        if (sched->vm_stats[i].vm_id == vm_id)
            return &sched->vm_stats[i];
    }
    if (sched->num_vms >= WFQ_MAX_VMS)
        return NULL;
    wfq_vm_stats_t *vs = &sched->vm_stats[sched->num_vms++];
    memset(vs, 0, sizeof(*vs));
    vs->vm_id = vm_id;
    return vs;
}

static double compute_urgency(const wfq_entry_t *e, int vm_queue_depth)
{
    double base;
    switch (e->priority) {
        case 2:  base = 4.0; break;
        case 1:  base = 2.0; break;
        default: base = 1.0; break;
    }

    double weight_mult = (e->weight > 0) ? (e->weight / 50.0) : 1.0;

    double pressure = vm_queue_depth / 10.0;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (now.tv_sec - e->enqueue_time.tv_sec)
                   + (now.tv_nsec - e->enqueue_time.tv_nsec) / 1e9;
    double wait_bonus = elapsed / 10.0;

    return base * weight_mult * (1.0 + pressure) * (1.0 + wait_bonus);
}

/* ---- Public API -------------------------------------------------------- */

void wfq_init(wfq_scheduler_t *sched)
{
    memset(sched, 0, sizeof(*sched));
    pthread_mutex_init(&sched->lock, NULL);
}

void wfq_destroy(wfq_scheduler_t *sched)
{
    pthread_mutex_destroy(&sched->lock);
}

int wfq_enqueue(wfq_scheduler_t *sched, const wfq_entry_t *entry)
{
    pthread_mutex_lock(&sched->lock);

    if (sched->queue_len >= WFQ_MAX_QUEUE_SIZE) {
        pthread_mutex_unlock(&sched->lock);
        return -1;
    }

    /* Copy entry into queue */
    wfq_entry_t *slot = &sched->queue[sched->queue_len];
    memcpy(slot, entry, sizeof(*slot));
    clock_gettime(CLOCK_MONOTONIC, &slot->enqueue_time);

    sched->queue_len++;

    /* Update per-VM stats */
    wfq_vm_stats_t *vs = find_or_create_vm(sched, entry->vm_id);
    if (vs) {
        vs->current_queue_depth++;
        vs->total_submitted++;

        /* Update submit rate (simple exponential moving average) */
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double dt = (now.tv_sec - vs->last_submit_time.tv_sec)
                  + (now.tv_nsec - vs->last_submit_time.tv_nsec) / 1e9;
        if (dt > 0.001) {
            double instant_rate = 1.0 / dt;
            vs->recent_submit_rate = 0.7 * vs->recent_submit_rate + 0.3 * instant_rate;
        }
        vs->last_submit_time = now;
    }

    pthread_mutex_unlock(&sched->lock);

    printf("[WFQ] Enqueue: vm=%u req=%u prio=%u weight=%d (queue=%d)\n",
           entry->vm_id, entry->request_id, entry->priority,
           entry->weight, sched->queue_len);
    return 0;
}

int wfq_dequeue(wfq_scheduler_t *sched, wfq_entry_t *out)
{
    pthread_mutex_lock(&sched->lock);

    if (sched->queue_len == 0) {
        pthread_mutex_unlock(&sched->lock);
        return -1;
    }

    /* Recompute urgency for every entry and pick the highest */
    int best_idx = 0;
    double best_urgency = -1.0;

    for (int i = 0; i < sched->queue_len; i++) {
        wfq_entry_t *e = &sched->queue[i];

        /* Find VM queue depth */
        int vqd = 0;
        for (int j = 0; j < sched->num_vms; j++) {
            if (sched->vm_stats[j].vm_id == e->vm_id) {
                vqd = sched->vm_stats[j].current_queue_depth;
                break;
            }
        }

        double u = compute_urgency(e, vqd);
        e->urgency = u;

        if (u > best_urgency) {
            best_urgency = u;
            best_idx = i;
        }
    }

    /* Copy best entry out */
    memcpy(out, &sched->queue[best_idx], sizeof(*out));
    out->urgency = best_urgency;

    /* Track context switches */
    if (sched->last_dispatched_vm != 0 &&
        sched->last_dispatched_vm != out->vm_id) {
        sched->context_switches++;
    }
    sched->last_dispatched_vm = out->vm_id;

    /* Remove from queue (swap with last) */
    sched->queue[best_idx] = sched->queue[sched->queue_len - 1];
    sched->queue_len--;

    /* Update per-VM stats */
    wfq_vm_stats_t *vs = find_or_create_vm(sched, out->vm_id);
    if (vs && vs->current_queue_depth > 0) {
        vs->current_queue_depth--;
    }

    pthread_mutex_unlock(&sched->lock);

    printf("[WFQ] Dequeue: vm=%u req=%u urgency=%.2f (queue=%d)\n",
           out->vm_id, out->request_id, best_urgency, sched->queue_len);
    return 0;
}

void wfq_complete(wfq_scheduler_t *sched, uint32_t vm_id, uint32_t exec_time_us)
{
    pthread_mutex_lock(&sched->lock);

    wfq_vm_stats_t *vs = find_or_create_vm(sched, vm_id);
    if (vs) {
        vs->total_completed++;
        /* Rolling average execution time */
        if (vs->total_completed == 1) {
            vs->avg_exec_time_us = (double)exec_time_us;
        } else {
            vs->avg_exec_time_us = 0.9 * vs->avg_exec_time_us + 0.1 * exec_time_us;
        }
    }

    pthread_mutex_unlock(&sched->lock);
}

int wfq_vm_queue_depth(wfq_scheduler_t *sched, uint32_t vm_id)
{
    pthread_mutex_lock(&sched->lock);
    int depth = 0;
    for (int i = 0; i < sched->num_vms; i++) {
        if (sched->vm_stats[i].vm_id == vm_id) {
            depth = sched->vm_stats[i].current_queue_depth;
            break;
        }
    }
    pthread_mutex_unlock(&sched->lock);
    return depth;
}

int wfq_queue_len(wfq_scheduler_t *sched)
{
    pthread_mutex_lock(&sched->lock);
    int len = sched->queue_len;
    pthread_mutex_unlock(&sched->lock);
    return len;
}

uint64_t wfq_context_switches(wfq_scheduler_t *sched)
{
    pthread_mutex_lock(&sched->lock);
    uint64_t cs = sched->context_switches;
    pthread_mutex_unlock(&sched->lock);
    return cs;
}

const wfq_vm_stats_t *wfq_get_vm_stats(wfq_scheduler_t *sched, uint32_t vm_id)
{
    for (int i = 0; i < sched->num_vms; i++) {
        if (sched->vm_stats[i].vm_id == vm_id)
            return &sched->vm_stats[i];
    }
    return NULL;
}

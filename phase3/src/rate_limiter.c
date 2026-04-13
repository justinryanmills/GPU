/*
 * Phase 3: Per-VM Token Bucket Rate Limiter
 *
 * Each VM has a bucket that refills at max_jobs_per_sec tokens/second.
 * Tokens accumulate up to max_jobs_per_sec (1-second burst capacity).
 * If the bucket is empty, the request is rejected with RL_REJECT_RATE.
 * If the VM's current queue depth exceeds max_queue_depth, rejected
 * with RL_REJECT_QUEUE.
 *
 * Rate = 0 means unlimited (always allowed).
 */

#include "rate_limiter.h"
#include <string.h>
#include <stdio.h>

/* ---- Internal ---------------------------------------------------------- */

static rl_bucket_t *find_bucket(rate_limiter_t *rl, uint32_t vm_id)
{
    for (int i = 0; i < RL_MAX_VMS; i++) {
        if (rl->buckets[i].active && rl->buckets[i].vm_id == vm_id)
            return &rl->buckets[i];
    }
    return NULL;
}

static void refill_tokens(rl_bucket_t *b)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    double elapsed = (now.tv_sec - b->last_refill.tv_sec)
                   + (now.tv_nsec - b->last_refill.tv_nsec) / 1e9;

    if (elapsed > 0) {
        b->tokens += elapsed * b->refill_rate;
        if (b->tokens > b->max_tokens)
            b->tokens = b->max_tokens;
        b->last_refill = now;
    }
}

/* ---- Public API -------------------------------------------------------- */

void rl_init(rate_limiter_t *rl)
{
    memset(rl, 0, sizeof(*rl));
    pthread_mutex_init(&rl->lock, NULL);
}

void rl_destroy(rate_limiter_t *rl)
{
    pthread_mutex_destroy(&rl->lock);
}

void rl_configure_vm(rate_limiter_t *rl, uint32_t vm_id,
                     int max_jobs_per_sec, int max_queue_depth)
{
    pthread_mutex_lock(&rl->lock);

    /* Find existing or allocate new slot */
    rl_bucket_t *b = find_bucket(rl, vm_id);
    if (!b) {
        for (int i = 0; i < RL_MAX_VMS; i++) {
            if (!rl->buckets[i].active) {
                b = &rl->buckets[i];
                break;
            }
        }
    }

    if (!b) {
        fprintf(stderr, "[RATE-LIMIT] No free bucket slots for vm_id=%u\n", vm_id);
        pthread_mutex_unlock(&rl->lock);
        return;
    }

    b->vm_id = vm_id;
    b->active = 1;
    b->refill_rate = (max_jobs_per_sec > 0) ? (double)max_jobs_per_sec : 0;
    b->max_tokens  = (max_jobs_per_sec > 0) ? (double)max_jobs_per_sec : 0;
    b->tokens      = b->max_tokens;  /* Start full */
    b->max_queue_depth = max_queue_depth;
    clock_gettime(CLOCK_MONOTONIC, &b->last_refill);

    pthread_mutex_unlock(&rl->lock);

    printf("[RATE-LIMIT] vm=%u: rate=%d jobs/sec, max_queue=%d\n",
           vm_id, max_jobs_per_sec, max_queue_depth);
}

int rl_check(rate_limiter_t *rl, uint32_t vm_id, int current_queue_depth)
{
    pthread_mutex_lock(&rl->lock);

    rl_bucket_t *b = find_bucket(rl, vm_id);
    if (!b) {
        /* VM not configured → allow (no limit) */
        pthread_mutex_unlock(&rl->lock);
        return RL_ALLOW;
    }

    /* Check queue depth limit */
    if (b->max_queue_depth > 0 && current_queue_depth >= b->max_queue_depth) {
        pthread_mutex_unlock(&rl->lock);
        printf("[RATE-LIMIT] vm=%u: REJECTED (queue depth %d >= %d)\n",
               vm_id, current_queue_depth, b->max_queue_depth);
        return RL_REJECT_QUEUE;
    }

    /* If no rate limit configured, allow */
    if (b->refill_rate <= 0) {
        pthread_mutex_unlock(&rl->lock);
        return RL_ALLOW;
    }

    /* Refill tokens based on elapsed time */
    refill_tokens(b);

    /* Try to consume one token */
    if (b->tokens >= 1.0) {
        b->tokens -= 1.0;
        pthread_mutex_unlock(&rl->lock);
        return RL_ALLOW;
    }

    /* No tokens available → reject */
    pthread_mutex_unlock(&rl->lock);
    printf("[RATE-LIMIT] vm=%u: REJECTED (rate limit, tokens=%.2f)\n",
           vm_id, b->tokens);
    return RL_REJECT_RATE;
}

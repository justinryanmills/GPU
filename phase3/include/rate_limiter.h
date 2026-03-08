/*
 * Phase 3: Per-VM token bucket rate limiter (0 = unlimited).
 */
#ifndef RATE_LIMITER_H
#define RATE_LIMITER_H

#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define RL_MAX_VMS  64

/* Per-VM bucket */
typedef struct {
    uint32_t vm_id;
    int      active;             /* 1 if this slot is in use                   */

    /* Token bucket */
    double   tokens;             /* Current tokens available                   */
    double   max_tokens;         /* Bucket capacity (= max_jobs_per_sec)      */
    double   refill_rate;        /* Tokens/second (= max_jobs_per_sec)        */
    struct timespec last_refill; /* Last time tokens were refilled             */

    /* Queue depth cap */
    int      max_queue_depth;    /* 0 = unlimited                             */
} rl_bucket_t;

typedef struct {
    rl_bucket_t buckets[RL_MAX_VMS];
    pthread_mutex_t lock;
} rate_limiter_t;

/* Return codes */
#define RL_ALLOW            0
#define RL_REJECT_RATE      1   /* Token bucket empty                        */
#define RL_REJECT_QUEUE     2   /* Queue depth exceeded                      */

/* Initialize rate limiter */
void rl_init(rate_limiter_t *rl);

/* Destroy */
void rl_destroy(rate_limiter_t *rl);

void rl_configure_vm(rate_limiter_t *rl, uint32_t vm_id,
                     int max_jobs_per_sec, int max_queue_depth);

int rl_check(rate_limiter_t *rl, uint32_t vm_id, int current_queue_depth);

#endif /* RATE_LIMITER_H */

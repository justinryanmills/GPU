#ifndef RATE_LIMITER_H
#define RATE_LIMITER_H

#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define RL_MAX_VMS  64

typedef struct {
    uint32_t vm_id;
    int      active;
    double   tokens;
    double   max_tokens;
    double   refill_rate;
    struct timespec last_refill;
    int      max_queue_depth;
} rl_bucket_t;

typedef struct {
    rl_bucket_t buckets[RL_MAX_VMS];
    pthread_mutex_t lock;
} rate_limiter_t;

#define RL_ALLOW            0
#define RL_REJECT_RATE      1
#define RL_REJECT_QUEUE     2

void rl_init(rate_limiter_t *rl);
void rl_destroy(rate_limiter_t *rl);
void rl_configure_vm(rate_limiter_t *rl, uint32_t vm_id,
                     int max_jobs_per_sec, int max_queue_depth);
int rl_check(rate_limiter_t *rl, uint32_t vm_id, int current_queue_depth);

#endif /* RATE_LIMITER_H */

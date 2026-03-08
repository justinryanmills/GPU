/*
 * MEDIATOR Daemon: priority queue, async CUDA, NFS /var/vgpu.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/stat.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include "cuda_vector_add.h"

#define NFS_BASE_DIR "/var/vgpu"
#define MAX_VM_ID 20
#define POLL_INTERVAL 1  // seconds
#define MAX_LINE_LEN 512

typedef struct Request {
    char pool_id;           // 'A' or 'B' (metadata only)
    uint32_t priority;      // 2=high, 1=medium, 0=low
    uint32_t vm_id;
    int num1, num2;         // Numbers to add
    time_t timestamp;       // For FIFO ordering
    void *user_data;        // For callback (points to Request itself)
    struct Request *next;
} Request;

typedef struct {
    Request *queue_head;     // Single priority queue (spans both pools)
    pthread_mutex_t queue_lock;
    int running;            // Control flag
    int cuda_busy;          // CUDA processing flag
    Request *current_request; // Currently processing request
    uint64_t total_processed;
    uint64_t pool_a_processed;
    uint64_t pool_b_processed;
} MediatorState;

// Global state
static MediatorState g_state;
static int g_shutdown = 0;

// Forward declarations
static void process_request(MediatorState *state, Request *req);

static void signal_handler(int sig) {
    printf("\n[SHUTDOWN] Received signal %d, shutting down gracefully...\n", sig);
    g_shutdown = 1;
    g_state.running = 0;
}

static void init_mediator(MediatorState *state) {
    state->queue_head = NULL;
    pthread_mutex_init(&state->queue_lock, NULL);
    state->running = 1;
    state->cuda_busy = 0;
    state->current_request = NULL;
    state->total_processed = 0;
    state->pool_a_processed = 0;
    state->pool_b_processed = 0;
    printf("[MEDIATOR] Initialized\n");
}

static void enqueue_request(MediatorState *state, Request *new_req) {
    pthread_mutex_lock(&state->queue_lock);
    
    // Empty queue
    if (state->queue_head == NULL) {
        state->queue_head = new_req;
        new_req->next = NULL;
        pthread_mutex_unlock(&state->queue_lock);
        printf("[ENQUEUE] Pool %c: vm=%u, prio=%u, %d+%d (queue empty)\n",
               new_req->pool_id, new_req->vm_id, new_req->priority,
               new_req->num1, new_req->num2);
        return;
    }
    
    // Find insertion point: higher priority first, then earlier timestamp
    Request *curr = state->queue_head;
    Request *prev = NULL;
    
    while (curr != NULL) {
        // New request has HIGHER priority - insert before current
        if (new_req->priority > curr->priority) {
            break;
        }
        
        // Same priority - FIFO (earlier timestamp first)
        if (new_req->priority == curr->priority) {
            if (new_req->timestamp < curr->timestamp) {
                break;  // New request is earlier
            }
        }
        
        prev = curr;
        curr = curr->next;
    }
    
    // Insert
    if (prev == NULL) {
        // Insert at head
        new_req->next = state->queue_head;
        state->queue_head = new_req;
    } else {
        // Insert in middle or end
        new_req->next = curr;
        prev->next = new_req;
    }
    
    pthread_mutex_unlock(&state->queue_lock);
    
    printf("[ENQUEUE] Pool %c: vm=%u, prio=%u, %d+%d\n",
           new_req->pool_id, new_req->vm_id, new_req->priority,
           new_req->num1, new_req->num2);
}

static Request* dequeue_request(MediatorState *state) {
    pthread_mutex_lock(&state->queue_lock);
    
    if (state->queue_head == NULL) {
        pthread_mutex_unlock(&state->queue_lock);
        return NULL;
    }
    
    Request *req = state->queue_head;
    state->queue_head = req->next;
    req->next = NULL;
    
    pthread_mutex_unlock(&state->queue_lock);
    
    return req;
}

static int get_queue_size(MediatorState *state) {
    pthread_mutex_lock(&state->queue_lock);
    int count = 0;
    Request *curr = state->queue_head;
    while (curr) {
        count++;
        curr = curr->next;
    }
    pthread_mutex_unlock(&state->queue_lock);
    return count;
}

static void cuda_result_callback(int result, void *user_data) {
    Request *req = (Request *)user_data;
    MediatorState *state = &g_state;
    
    printf("[RESULT] Pool %c: vm=%u, result=%d\n",
           req->pool_id, req->vm_id, result);
    
    // Write response to VM with proper NFS synchronization
    char response_file[256];
    snprintf(response_file, sizeof(response_file), "%s/vm%u/response.txt", NFS_BASE_DIR, req->vm_id);
    
    FILE *fp = fopen(response_file, "w");
    if (fp) {
        fprintf(fp, "%d\n", result);
        fflush(fp);  // Flush to ensure data is written
        fsync(fileno(fp));  // Force NFS synchronization
        fclose(fp);
        printf("[RESPONSE] Sent to vm%u: %d\n", req->vm_id, result);
    } else {
        fprintf(stderr, "[ERROR] Failed to write response to %s: %s\n",
                response_file, strerror(errno));
    }
    
    // Clear request file (response will be cleared by VM after reading, or by MEDIATOR when new request arrives)
    char request_file[256];
    snprintf(request_file, sizeof(request_file), "%s/vm%u/request.txt", NFS_BASE_DIR, req->vm_id);
    
    fp = fopen(request_file, "w");
    if (fp) {
        fclose(fp);  // Truncate to zero
        printf("[INIT] Cleared request file for vm%u\n", req->vm_id);
    }
    
    // Update statistics
    state->total_processed++;
    if (req->pool_id == 'A') {
        state->pool_a_processed++;
    } else {
        state->pool_b_processed++;
    }
    
    // Mark CUDA as idle
    state->cuda_busy = 0;
    state->current_request = NULL;
    
    // Free request
    free(req);
    
    // Process next request if available
    Request *next_req = dequeue_request(state);
    if (next_req) {
        process_request(state, next_req);
    }
}

static void process_request(MediatorState *state, Request *req) {
    if (state->cuda_busy) {
        // Should not happen, but re-queue if it does
        fprintf(stderr, "[WARNING] CUDA busy, re-queuing request\n");
        enqueue_request(state, req);
        return;
    }
    
    state->cuda_busy = 1;
    state->current_request = req;
    
    printf("[PROCESS] Pool %c: vm=%u, prio=%u, %d+%d\n",
           req->pool_id, req->vm_id, req->priority,
           req->num1, req->num2);
    
    // Set user_data to request for callback
    req->user_data = req;
    
    // Send to CUDA asynchronously
    if (cuda_vector_add_async(req->num1, req->num2, cuda_result_callback, req) != 0) {
        fprintf(stderr, "[ERROR] Failed to start CUDA operation\n");
        state->cuda_busy = 0;
        state->current_request = NULL;
        free(req);
    }
}

static int parse_request(const char *line, char *pool_id, uint32_t *priority,
                        uint32_t *vm_id, int *num1, int *num2) {
    char pool;
    uint32_t prio, vm;
    int n1, n2;
    
    if (sscanf(line, "%c:%u:%u:%d:%d", &pool, &prio, &vm, &n1, &n2) != 5) {
        return -1;
    }
    
    // Validate pool_id
    if (pool != 'A' && pool != 'B') {
        return -1;
    }
    
    // Validate priority
    if (prio > 2) {
        return -1;
    }
    
    *pool_id = pool;
    *priority = prio;
    *vm_id = vm;
    *num1 = n1;
    *num2 = n2;
    
    return 0;
}

static void poll_requests(MediatorState *state) {
    struct dirent *entry;
    DIR *dir;
    
    dir = opendir(NFS_BASE_DIR);
    if (!dir) {
        fprintf(stderr, "[ERROR] Failed to open %s: %s\n", NFS_BASE_DIR, strerror(errno));
        return;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (entry->d_name[0] == '.') {
            continue;
        }
        
        // Check if it's a VM directory (vm1, vm2, etc.)
        if (strncmp(entry->d_name, "vm", 2) != 0) {
            continue;
        }
        
        // Construct request file path
        char request_file[512];
        snprintf(request_file, sizeof(request_file), "%s/%s/request.txt", NFS_BASE_DIR, entry->d_name);
        
        // Extract VM ID from directory name for response file clearing
        uint32_t dir_vm_id;
        if (sscanf(entry->d_name, "vm%u", &dir_vm_id) != 1) {
            continue;
        }
        
        // Clear old response file if it exists (indicates previous response was not read, or new request)
        char response_file[512];
        snprintf(response_file, sizeof(response_file), "%s/%s/response.txt", NFS_BASE_DIR, entry->d_name);
        FILE *fp = fopen(response_file, "w");
        if (fp) {
            fclose(fp);  // Clear old response
        }
        
        // Check if request file exists and is readable
        fp = fopen(request_file, "r");
        if (!fp) {
            continue;
        }
        
        // Read request line
        char line[MAX_LINE_LEN];
        if (fgets(line, sizeof(line), fp) == NULL) {
            fclose(fp);
            continue;
        }
        fclose(fp);
        
        // Trim newline
        line[strcspn(line, "\n")] = '\0';
        
        // Skip empty lines
        if (strlen(line) == 0) {
            continue;
        }
        
        // Parse request
        char pool_id;
        uint32_t priority, vm_id;
        int num1, num2;
        
        if (parse_request(line, &pool_id, &priority, &vm_id, &num1, &num2) != 0) {
            fprintf(stderr, "[ERROR] Invalid request format: %s\n", line);
            continue;
        }
        
        // Verify VM ID matches (dir_vm_id already extracted above)
        if (dir_vm_id != vm_id) {
            fprintf(stderr, "[WARNING] VM ID mismatch: dir=%u, request=%u\n",
                    dir_vm_id, vm_id);
        }
        
        // Create request
        Request *req = (Request *)malloc(sizeof(Request));
        if (!req) {
            fprintf(stderr, "[ERROR] Failed to allocate request\n");
            continue;
        }
        
        req->pool_id = pool_id;
        req->priority = priority;
        req->vm_id = vm_id;
        req->num1 = num1;
        req->num2 = num2;
        req->timestamp = time(NULL);
        req->next = NULL;
        req->user_data = NULL;
        
        // Enqueue request
        enqueue_request(state, req);
        
        // Clear request file (so it's not processed again)
        fp = fopen(request_file, "w");
        if (fp) {
            fclose(fp);  // Truncate
        }
    }
    
    closedir(dir);
}

static void process_queue(MediatorState *state) {
    // If CUDA is busy, don't process more
    if (state->cuda_busy) {
        return;
    }
    
    // Get next request
    Request *req = dequeue_request(state);
    if (req) {
        process_request(state, req);
    }
}

static void print_stats(MediatorState *state) {
    int queue_size = get_queue_size(state);
    printf("\n[MEDIATOR STATS]\n");
    printf("  Total processed: %lu\n", state->total_processed);
    printf("  Pool A processed: %lu\n", state->pool_a_processed);
    printf("  Pool B processed: %lu\n", state->pool_b_processed);
    printf("  Queue size: %d\n", queue_size);
    printf("  CUDA busy: %s\n", state->cuda_busy ? "yes" : "no");
    printf("\n");
}

static void run_mediator(MediatorState *state) {
    time_t last_stats = time(NULL);
    
    printf("[MEDIATOR] Starting main loop...\n");
    printf("[MEDIATOR] Polling %s every %d seconds\n", NFS_BASE_DIR, POLL_INTERVAL);
    
    while (state->running && !g_shutdown) {
        // Poll for new requests
        poll_requests(state);
        
        // Process queue (send to CUDA if idle)
        process_queue(state);
        
        // Print statistics every 60 seconds
        time_t now = time(NULL);
        if (now - last_stats >= 60) {
            print_stats(state);
            last_stats = now;
        }
        
        // Sleep before next poll
        sleep(POLL_INTERVAL);
    }
    
    printf("[MEDIATOR] Shutting down...\n");
    
    // Wait for CUDA to finish
    if (state->cuda_busy) {
        printf("[MEDIATOR] Waiting for CUDA to finish...\n");
        cuda_sync();
    }
    
    // Cleanup remaining requests
    Request *req;
    while ((req = dequeue_request(state)) != NULL) {
        free(req);
    }
}

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
    printf("================================================================================\n");
    printf("                    MEDIATOR DAEMON - CUDA Vector Addition\n");
    printf("================================================================================\n\n");
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Initialize mediator
    init_mediator(&g_state);
    
    // Initialize CUDA
    if (cuda_init() != 0) {
        fprintf(stderr, "[ERROR] Failed to initialize CUDA\n");
        return 1;
    }
    
    // Verify NFS directory exists
    struct stat st;
    if (stat(NFS_BASE_DIR, &st) != 0) {
        fprintf(stderr, "[ERROR] NFS directory %s does not exist\n", NFS_BASE_DIR);
        fprintf(stderr, "        Create it and set up per-VM directories\n");
        cuda_cleanup();
        return 1;
    }
    
    // Run mediator
    run_mediator(&g_state);
    
    // Cleanup
    cuda_cleanup();
    pthread_mutex_destroy(&g_state.queue_lock);
    
    printf("[MEDIATOR] Exited\n");
    return 0;
}

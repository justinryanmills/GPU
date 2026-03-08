/*
 * GPU Mediation Daemon
 * 
 * Purpose: Manage GPU request queues with pool separation and priority ordering
 * 
 * Features:
 * - Two independent queues (Pool A and Pool B)
 * - Priority-based ordering within each pool (high > medium > low)
 * - FIFO tie-breaking within same priority
 * - Per-VM request/response file handling
 * - Thread-safe queue operations
 * 
 * Usage: sudo ./mediator
 * 
 * Requirements:
 * - NFS export at /var/vgpu with per-VM directories
 * - Write access to /var/vgpu/vm<id>/response.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>      /* For size_t */
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/stat.h>
#include <signal.h>
#include <errno.h>

/*
 * Request Structure
 * Represents a single GPU request from a VM
 */
typedef struct Request {
    char pool_id;           // 'A' or 'B'
    uint32_t priority;      // 2=high, 1=medium, 0=low
    uint32_t vm_id;         // Unique VM ID
    char command[256];      // Command to execute
    time_t timestamp;       // For FIFO tie-breaking
    struct Request *next;   // Linked list pointer
} Request;

/*
 * Pool Queue Structure
 * Maintains priority-sorted queue for one pool
 */
typedef struct {
    char pool_id;           // 'A' or 'B'
    Request *head;          // Head of priority queue
    int count;              // Number of requests in queue
    pthread_mutex_t lock;   // Thread safety
} PoolQueue;

/*
 * Mediator State
 * Global state for mediation daemon
 */
typedef struct {
    PoolQueue pool_a;       // Queue for Pool A
    PoolQueue pool_b;       // Queue for Pool B
    int running;            // Control flag
    uint64_t total_processed; // Statistics
    int test_mode;          // Test mode: wait for multiple requests
    int wait_count;         // Number of requests to wait for (test mode)
    uint32_t preferred_vm; // VM ID to prefer in tie-breaking (0 = none)
    int processing_paused;  // Pause processing until test mode questions answered
    int test_mode_pending;  // Test mode detected but questions not answered yet
    // Round-robin tracking: last VM processed per priority level [low, medium, high]
    uint32_t last_vm_per_priority[3];  // Index 0=low, 1=medium, 2=high
} MediatorState;

// Global state pointer for signal handlers
static MediatorState *g_state = NULL;

/*
 * Initialize a pool queue
 */
void init_pool_queue(PoolQueue *queue, char pool_id) {
    queue->pool_id = pool_id;
    queue->head = NULL;
    queue->count = 0;
    pthread_mutex_init(&queue->lock, NULL);
    printf("[INIT] Pool %c queue initialized\n", pool_id);
}

/*
 * Initialize mediator state
 */
void init_mediator(MediatorState *state) {
    init_pool_queue(&state->pool_a, 'A');
    init_pool_queue(&state->pool_b, 'B');
    state->running = 1;
    state->total_processed = 0;
    state->test_mode = 0;
    state->wait_count = 0;
    state->preferred_vm = 0;
    state->processing_paused = 0;
    state->test_mode_pending = 0;
    // Initialize round-robin tracking (0 means no VM processed yet)
    state->last_vm_per_priority[0] = 0;  // Low priority
    state->last_vm_per_priority[1] = 0;  // Medium priority
    state->last_vm_per_priority[2] = 0;  // High priority
}

/*
 * Insert request into queue with priority sorting
 * Higher priority processed first, FIFO within same priority
 * If preferred_vm is set and priorities/pools are equal, prefer that VM
 */
void insert_request(PoolQueue *queue, Request *new_req, MediatorState *state) {
    pthread_mutex_lock(&queue->lock);
    
    // Empty queue - insert at head
    if (queue->head == NULL) {
        queue->head = new_req;
        new_req->next = NULL;
        queue->count++;
        pthread_mutex_unlock(&queue->lock);
        return;
    }
    
    // Find insertion point based on priority and timestamp
    // Higher priority first (2 > 1 > 0)
    // Within same priority, check preferred VM or FIFO (earlier timestamp first)
    Request *curr = queue->head;
    Request *prev = NULL;
    
    while (curr != NULL) {
        // New request has HIGHER priority - insert before current
        if (new_req->priority > curr->priority) {
            break;
        }
        
        // Same priority - check tie-breaking rules
        if (new_req->priority == curr->priority) {
            // If preferred VM is set, check if either request is from preferred VM
            if (state->preferred_vm != 0) {
                int new_is_preferred = (new_req->vm_id == state->preferred_vm);
                int curr_is_preferred = (curr->vm_id == state->preferred_vm);
                
                // Preferred VM always goes first
                if (new_is_preferred && !curr_is_preferred) {
                    break;
                }
                if (!new_is_preferred && curr_is_preferred) {
                    prev = curr;
                    curr = curr->next;
                    continue;
                }
            }
            
            // No preferred VM or both/neither are preferred - use FIFO (earlier timestamp first)
            if (new_req->timestamp < curr->timestamp) {
                break;
            }
        }
        
        prev = curr;
        curr = curr->next;
    }
    
    // Insert at head
    if (prev == NULL) {
        new_req->next = queue->head;
        queue->head = new_req;
    }
    // Insert in middle or end
    else {
        prev->next = new_req;
        new_req->next = curr;
    }
    
    queue->count++;
    pthread_mutex_unlock(&queue->lock);
}

/*
 * Find the highest priority level that has requests in the queue
 * Returns: 2=high, 1=medium, 0=low, or -1 if queue is empty
 */
static int find_highest_priority_in_queue(PoolQueue *queue) {
    Request *req = queue->head;
    uint32_t highest_prio = 0;  // Use uint32_t to match priority type
    int found_any = 0;
    
    printf("[DEBUG] find_highest_priority: Starting scan, head=%p\n", (void*)req);
    fflush(stdout);
    
    while (req != NULL) {
        printf("[DEBUG] find_highest_priority: req=%p, req->priority=%u, req->vm_id=%u\n",
               (void*)req, req->priority, req->vm_id);
        fflush(stdout);
        
        if (!found_any || req->priority > highest_prio) {
            highest_prio = req->priority;
            found_any = 1;
        }
        req = req->next;
    }
    
    printf("[DEBUG] find_highest_priority: Returning %u (found_any=%d)\n", highest_prio, found_any);
    fflush(stdout);
    
    if (!found_any) {
        return -1;  // No requests found
    }
    
    return (int)highest_prio;  // Cast to int for return value
}

/*
 * Find the next VM in round-robin order for a given priority level
 * Returns the VM ID of the next VM to process, or 0 if none found
 * 
 * Algorithm:
 * 1. Find all VMs with requests at this priority level
 * 2. If last_vm is 0, return first VM found
 * 3. Otherwise, find VM that comes after last_vm in round-robin
 * 4. If last_vm was last, wrap around to first VM
 */
static uint32_t find_next_vm_round_robin(PoolQueue *queue, int priority, uint32_t last_vm) {
    Request *req = queue->head;
    uint32_t first_vm = 0;
    uint32_t next_vm = 0;
    int found_last = 0;
    
    // First pass: find first VM and check if last_vm exists
    while (req != NULL) {
        if (req->priority == priority) {
            if (first_vm == 0) {
                first_vm = req->vm_id;  // Remember first VM for wrap-around
            }
            if (req->vm_id == last_vm) {
                found_last = 1;
            }
        }
        req = req->next;
    }
    
    // If no requests at this priority, return 0
    if (first_vm == 0) {
        return 0;
    }
    
    // If last_vm is 0 or not found, return first VM
    if (last_vm == 0 || !found_last) {
        return first_vm;
    }
    
    // Second pass: find VM that comes after last_vm
    req = queue->head;
    while (req != NULL) {
        if (req->priority == priority && req->vm_id > last_vm) {
            // Found a VM with higher ID after last_vm
            if (next_vm == 0 || req->vm_id < next_vm) {
                next_vm = req->vm_id;
            }
        }
        req = req->next;
    }
    
    // If no VM found after last_vm, wrap around to first VM
    if (next_vm == 0) {
        return first_vm;
    }
    
    return next_vm;
}

/*
 * Pop a specific request from the queue (by VM ID and priority)
 * Removes the first matching request and returns it
 * Returns NULL if not found
 */
static Request* pop_request_from_vm(PoolQueue *queue, uint32_t vm_id, int priority) {
    Request *req = queue->head;
    Request *prev = NULL;
    
    while (req != NULL) {
        if (req->vm_id == vm_id && req->priority == priority) {
            // Found the request to pop
            if (prev == NULL) {
                // Removing head
                queue->head = req->next;
            } else {
                // Removing from middle or end
                prev->next = req->next;
            }
            req->next = NULL;  // Disconnect from list
            return req;
        }
        prev = req;
        req = req->next;
    }
    
    return NULL;  // Not found
}

/*
 * Pop highest priority request from queue with VM round-robin
 * Returns NULL if queue is empty
 * 
 * Round-robin logic:
 * - Find highest priority level with requests
 * - Select next VM in round-robin order for that priority
 * - Pop request from that VM
 * - Update tracking for next round-robin cycle
 */
Request* pop_request(PoolQueue *queue, MediatorState *state) {
    pthread_mutex_lock(&queue->lock);
    
    printf("[DEBUG] pop_request: Pool %c, head=%p, count=%d\n",
           queue->pool_id, (void*)queue->head, queue->count);
    fflush(stdout);
    
    if (queue->head == NULL) {
        printf("[DEBUG] pop_request: Queue head is NULL for Pool %c\n", queue->pool_id);
        fflush(stdout);
        pthread_mutex_unlock(&queue->lock);
        return NULL;
    }
    
    // Find highest priority level with requests
    int highest_prio = find_highest_priority_in_queue(queue);
    printf("[DEBUG] pop_request: Highest priority = %d for Pool %c\n", highest_prio, queue->pool_id);
    fflush(stdout);
    
    if (highest_prio < 0) {
        printf("[DEBUG] pop_request: No requests found in Pool %c\n", queue->pool_id);
        fflush(stdout);
        pthread_mutex_unlock(&queue->lock);
        return NULL;
    }
    
    // Find next VM in round-robin order for this priority
    uint32_t last_vm = state->last_vm_per_priority[highest_prio];
    printf("[DEBUG] pop_request: Looking for next VM, priority=%d, last_vm=%u\n",
           highest_prio, last_vm);
    fflush(stdout);
    
    uint32_t next_vm = find_next_vm_round_robin(queue, highest_prio, last_vm);
    
    printf("[DEBUG] pop_request: next_vm = %u for Pool %c\n", next_vm, queue->pool_id);
    fflush(stdout);
    
    if (next_vm == 0) {
        // Debug: This should not happen - log for troubleshooting
        printf("[DEBUG] pop_request: next_vm is 0 for priority %d, pool %c\n", 
               highest_prio, queue->pool_id);
        fflush(stdout);
        pthread_mutex_unlock(&queue->lock);
        return NULL;
    }
    
    // Pop request from that VM
    Request *req = pop_request_from_vm(queue, next_vm, highest_prio);
    
    if (!req) {
        // Debug: Request not found - log for troubleshooting
        printf("[DEBUG] pop_request: Request not found for vm=%u, prio=%d, pool=%c\n",
               next_vm, highest_prio, queue->pool_id);
        fflush(stdout);
        pthread_mutex_unlock(&queue->lock);
        return NULL;
    }
    
    printf("[DEBUG] pop_request: Successfully popped request vm=%u, prio=%d from Pool %c\n",
           req->vm_id, req->priority, queue->pool_id);
    fflush(stdout);
    
    // Update round-robin tracking for this priority level
    state->last_vm_per_priority[highest_prio] = next_vm;
    queue->count--;
    
    pthread_mutex_unlock(&queue->lock);
    
    return req;
}

/*
 * Pop highest priority request from queue (legacy version - for backward compatibility)
 * This version always pops the head (no round-robin)
 * Use pop_request(queue, state) for round-robin behavior
 */
Request* pop_request_legacy(PoolQueue *queue) {
    pthread_mutex_lock(&queue->lock);
    
    if (queue->head == NULL) {
        pthread_mutex_unlock(&queue->lock);
        return NULL;
    }
    
    // Head is always highest priority (sorted insert)
    Request *req = queue->head;
    queue->head = req->next;
    queue->count--;
    
    pthread_mutex_unlock(&queue->lock);
    
    return req;
}

/*
 * Parse incoming request from file data
 * Format: "pool_id:priority:vm_id:command"
 * Example: "A:2:100:VECTOR_ADD"
 */
Request* parse_request(const char *data) {
    Request *req = malloc(sizeof(Request));
    if (!req) return NULL;
    
    char pool[2];
    unsigned int prio, vmid;
    
    // Parse format: pool_id:priority:vm_id:command
    if (sscanf(data, "%1[AB]:%u:%u:%255s", 
               pool, &prio, &vmid, req->command) != 4) {
        free(req);
        return NULL;
    }
    
    req->pool_id = pool[0];
    req->priority = prio;
    req->vm_id = vmid;
    req->timestamp = time(NULL);
    req->next = NULL;
    
    return req;
}

/*
 * Check if a request from this VM is already in the queues
 */
int is_vm_request_queued(MediatorState *state, uint32_t vm_id) {
    Request *req;
    
    // Check Pool A
    req = state->pool_a.head;
    while (req != NULL) {
        if (req->vm_id == vm_id) {
            return 1;  // Found
        }
        req = req->next;
    }
    
    // Check Pool B
    req = state->pool_b.head;
    while (req != NULL) {
        if (req->vm_id == vm_id) {
            return 1;  // Found
        }
        req = req->next;
    }
    
    return 0;  // Not found
}

/*
 * Poll for new requests from all VM directories
 * Dynamically scans /var/vgpu for vm* directories
 */
void poll_requests(MediatorState *state) {
    char request_file[512];
    char buffer[512];
    FILE *fp;
    DIR *dir;
    struct dirent *entry;
    
    // Open /var/vgpu directory
    dir = opendir("/var/vgpu");
    if (!dir) {
        // Directory doesn't exist - not an error, just no VMs yet
        return;
    }
    
    // Scan for vm* directories
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (entry->d_name[0] == '.') continue;
        
        // Check if it's a vm* directory (vm1, vm2, vm200, etc.)
        if (strncmp(entry->d_name, "vm", 2) != 0) continue;
        
        // Build request file path
        snprintf(request_file, sizeof(request_file),
                 "/var/vgpu/%s/request.txt", entry->d_name);
        
        fp = fopen(request_file, "r");
        if (!fp) continue;  // VM directory doesn't exist or no request
        
        if (fgets(buffer, sizeof(buffer), fp)) {
            // Check if it's a new request (not empty, not just whitespace)
            if (strlen(buffer) > 3 && buffer[0] != '\n') {
                Request *req = parse_request(buffer);
                if (req) {
                    // Check if this VM already has a request queued (prevent duplicates)
                    if (is_vm_request_queued(state, req->vm_id)) {
                        // Request already queued, skip to avoid duplicates
                        free(req);
                        fclose(fp);
                        continue;
                    }
                    
                    // Insert into correct pool queue
                    if (req->pool_id == 'A') {
                        insert_request(&state->pool_a, req, state);
                        printf("[ENQUEUE] Pool A: vm=%u, prio=%u (%s), cmd=%s, queue_size=%d\n",
                               req->vm_id, req->priority,
                               req->priority == 2 ? "high" : 
                               req->priority == 1 ? "medium" : "low",
                               req->command, state->pool_a.count);
                    } else if (req->pool_id == 'B') {
                        insert_request(&state->pool_b, req, state);
                        printf("[ENQUEUE] Pool B: vm=%u, prio=%u (%s), cmd=%s, queue_size=%d\n",
                               req->vm_id, req->priority,
                               req->priority == 2 ? "high" : 
                               req->priority == 1 ? "medium" : "low",
                               req->command, state->pool_b.count);
                    } else {
                        printf("[WARN] Invalid pool_id: %c, ignoring request\n", req->pool_id);
                        free(req);
                    }
                    
                    // Don't clear request file yet - wait until request is processed
                    // This prevents VM from thinking request was sent when processing is paused
                    fclose(fp);
                } else {
                    printf("[WARN] Failed to parse request from %s: %s\n", entry->d_name, buffer);
                    fclose(fp);
                }
            } else {
                fclose(fp);
            }
        } else {
            fclose(fp);
        }
    }
    
    closedir(dir);
}

/*
 * Execute GPU workload (placeholder - CUDA integration later)
 */
void execute_gpu_workload(Request *req, char *result, size_t result_len) {
    // Placeholder: Simulate GPU work
    // In real implementation, this would call CUDA kernel
    
    if (strcmp(req->command, "VECTOR_ADD") == 0) {
        snprintf(result, result_len,
                 "1:Vector add completed on Pool %c (vm=%u, prio=%u)",
                 req->pool_id, req->vm_id, req->priority);
    } else if (strcmp(req->command, "MATRIX_MUL") == 0) {
        snprintf(result, result_len,
                 "1:Matrix multiply completed on Pool %c (vm=%u, prio=%u)",
                 req->pool_id, req->vm_id, req->priority);
    } else {
        snprintf(result, result_len,
                 "1:Command '%s' executed on Pool %c (vm=%u, prio=%u)",
                 req->command, req->pool_id, req->vm_id, req->priority);
    }
    
    // Simulate some GPU work (remove this in production)
    usleep(500000);  // 500ms
}

/*
 * Cleanup function: Clear all request and response files
 * Called on termination to prevent stale data on restart
 */
void cleanup_files(void) {
    DIR *dir;
    struct dirent *entry;
    char file_path[512];
    
    printf("\n[CLEANUP] Clearing all request/response files...\n");
    
    dir = opendir("/var/vgpu");
    if (!dir) {
        printf("[CLEANUP] /var/vgpu directory not found\n");
        return;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        if (strncmp(entry->d_name, "vm", 2) != 0) continue;
        
        // Clear request file
        snprintf(file_path, sizeof(file_path), "/var/vgpu/%s/request.txt", entry->d_name);
        FILE *fp = fopen(file_path, "w");
        if (fp) {
            fclose(fp);
            printf("[CLEANUP] Cleared: %s\n", file_path);
        }
        
        // Clear response file
        snprintf(file_path, sizeof(file_path), "/var/vgpu/%s/response.txt", entry->d_name);
        fp = fopen(file_path, "w");
        if (fp) {
            fclose(fp);
            printf("[CLEANUP] Cleared: %s\n", file_path);
        }
    }
    
    closedir(dir);
    printf("[CLEANUP] Cleanup complete\n");
}

/*
 * Signal handler for cleanup on termination
 */
void signal_handler(int sig) {
    if (g_state) {
        g_state->running = 0;
    }
    cleanup_files();
    printf("\n[STOP] Mediator stopped by signal %d\n", sig);
    exit(0);
}

/*
 * Count total requests in both queues
 */
int count_total_requests(MediatorState *state) {
    return state->pool_a.count + state->pool_b.count;
}

/*
 * Check which pool contains the preferred VM (if set)
 * Returns 'A', 'B', or 0 if not found
 * Thread-safe: locks queues during search
 */
char find_preferred_vm_pool(MediatorState *state) {
    if (state->preferred_vm == 0) {
        return 0;  // No preferred VM set
    }
    
    Request *req;
    char result = 0;
    
    // Check Pool A (with mutex lock)
    pthread_mutex_lock(&state->pool_a.lock);
    req = state->pool_a.head;
    while (req != NULL) {
        if (req->vm_id == state->preferred_vm) {
            result = 'A';
            break;
        }
        req = req->next;
    }
    pthread_mutex_unlock(&state->pool_a.lock);
    
    if (result != 0) {
        return result;  // Found in Pool A
    }
    
    // Check Pool B (with mutex lock)
    pthread_mutex_lock(&state->pool_b.lock);
    req = state->pool_b.head;
    while (req != NULL) {
        if (req->vm_id == state->preferred_vm) {
            result = 'B';
            break;
        }
        req = req->next;
    }
    pthread_mutex_unlock(&state->pool_b.lock);
    
    return result;  // Returns 'B' if found, 0 if not found
}

/*
 * Get list of VM IDs that have requests
 */
void get_requesting_vms(MediatorState *state, uint32_t *vm_list, int max_vms, int *count) {
    Request *req;
    int found = 0;
    int i;
    
    *count = 0;
    
    // Check Pool A
    req = state->pool_a.head;
    while (req && found < max_vms) {
        // Check if VM ID already in list
        int exists = 0;
        for (i = 0; i < *count; i++) {
            if (vm_list[i] == req->vm_id) {
                exists = 1;
                break;
            }
        }
        if (!exists) {
            vm_list[*count] = req->vm_id;
            (*count)++;
            found++;
        }
        req = req->next;
    }
    
    // Check Pool B
    req = state->pool_b.head;
    while (req && found < max_vms) {
        // Check if VM ID already in list
        int exists = 0;
        for (i = 0; i < *count; i++) {
            if (vm_list[i] == req->vm_id) {
                exists = 1;
                break;
            }
        }
        if (!exists) {
            vm_list[*count] = req->vm_id;
            (*count)++;
            found++;
        }
        req = req->next;
    }
}

/*
 * Reset timestamps for all requests in queues to same value (simulate concurrent arrival)
 */
void reset_request_timestamps(MediatorState *state) {
    time_t concurrent_time = time(NULL);
    Request *req;
    
    // Reset Pool A timestamps
    req = state->pool_a.head;
    while (req != NULL) {
        req->timestamp = concurrent_time;
        req = req->next;
    }
    
    // Reset Pool B timestamps
    req = state->pool_b.head;
    while (req != NULL) {
        req->timestamp = concurrent_time;
        req = req->next;
    }
}

/*
 * Re-sort queues after timestamp reset (to apply preferred VM logic)
 * Must be called with queues already locked or when no processing is happening
 */
void resort_queues(MediatorState *state) {
    Request *req;
    Request *temp_a_head = NULL;
    Request *temp_b_head = NULL;
    int temp_a_count = 0;
    int temp_b_count = 0;
    
    // Lock both queues to safely extract requests
    pthread_mutex_lock(&state->pool_a.lock);
    pthread_mutex_lock(&state->pool_b.lock);
    
    // Extract all requests from queues
    temp_a_head = state->pool_a.head;
    temp_a_count = state->pool_a.count;
    temp_b_head = state->pool_b.head;
    temp_b_count = state->pool_b.count;
    
    // Clear queues
    state->pool_a.head = NULL;
    state->pool_a.count = 0;
    state->pool_b.head = NULL;
    state->pool_b.count = 0;
    
    // Unlock before re-inserting (insert_request will lock again)
    pthread_mutex_unlock(&state->pool_b.lock);
    pthread_mutex_unlock(&state->pool_a.lock);
    
    // Re-insert Pool A requests
    req = temp_a_head;
    while (req != NULL) {
        Request *next = req->next;
        req->next = NULL;  // Clear next pointer before re-inserting
        insert_request(&state->pool_a, req, state);
        req = next;
    }
    
    // Re-insert Pool B requests
    req = temp_b_head;
    while (req != NULL) {
        Request *next = req->next;
        req->next = NULL;  // Clear next pointer before re-inserting
        insert_request(&state->pool_b, req, state);
        req = next;
    }
}

/*
 * Ask user if they want to enable test mode at startup
 */
int ask_test_mode_at_startup(void) {
    char answer[16];
    
    printf("\n");
    printf("================================================================================\n");
    printf("                    TEST MODE CONFIGURATION\n");
    printf("================================================================================\n");
    printf("\n");
    printf("[STARTUP] Do you want to run in test mode?\n");
    printf("          Test mode will wait for requests from 2+ VMs before processing.\n");
    printf("          (YES/NO): ");
    fflush(stdout);
    
    if (fgets(answer, sizeof(answer), stdin) == NULL) {
        printf("[STARTUP] Invalid input, defaulting to NO\n");
        return 0;
    }
    
    // Remove newline
    answer[strcspn(answer, "\n")] = 0;
    
    if (strcasecmp(answer, "YES") == 0 || strcasecmp(answer, "Y") == 0) {
        printf("[STARTUP] Test mode enabled - waiting for 2+ VM requests...\n");
        printf("\n");
        return 1;
    } else {
        printf("[STARTUP] Normal mode - processing requests as they arrive\n");
        printf("\n");
        return 0;
    }
}

/*
 * Interactive test mode: Ask user for scheduling preferences when 2+ VMs detected
 */
void interactive_test_mode(MediatorState *state) {
    char answer[16];
    uint32_t vm_list[10];
    int vm_count;
    int i;
    
    printf("\n");
    printf("================================================================================\n");
    printf("                    TEST MODE: Concurrent Request Scheduling\n");
    printf("================================================================================\n");
    printf("\n");
    
    // Get list of requesting VMs
    get_requesting_vms(state, vm_list, 10, &vm_count);
    
    printf("[TEST] Found requests from %d VM(s): ", vm_count);
    for (i = 0; i < vm_count; i++) {
        printf("VM%u ", vm_list[i]);
    }
    printf("\n\n");
    
    // Question 1: Proceed as if requests arrived simultaneously?
    printf("[TEST] Question 1: Do you want to proceed as if requests from %d VM(s) arrived\n", vm_count);
    printf("        at the same time? (YES/NO): ");
    fflush(stdout);
    
    if (fgets(answer, sizeof(answer), stdin) == NULL) {
        printf("[TEST] Invalid input, defaulting to NO\n");
        state->test_mode = 0;
        state->processing_paused = 0;  // Resume processing
        return;
    }
    
    // Remove newline
    answer[strcspn(answer, "\n")] = 0;
    
    if (strcasecmp(answer, "YES") == 0 || strcasecmp(answer, "Y") == 0) {
        state->test_mode = 1;
        printf("[TEST] Test mode enabled - requests will be processed as concurrent\n");
        
        // Reset all timestamps to same value (simulate concurrent arrival)
        reset_request_timestamps(state);
        
        // Question 2: Which VM should be processed first in case of ties?
        printf("\n[TEST] Question 2: If priority and pool are the same, which VM should be\n");
        printf("        processed first? (Enter VM ID, e.g., 1, 200, or press Enter for FIFO): ");
        fflush(stdout);
        
        if (fgets(answer, sizeof(answer), stdin) == NULL) {
            printf("[TEST] Invalid input, using FIFO (timestamp order)\n");
            state->preferred_vm = 0;
        } else {
            // Remove newline
            answer[strcspn(answer, "\n")] = 0;
            
            if (strlen(answer) == 0 || strcasecmp(answer, "FIFO") == 0) {
                state->preferred_vm = 0;
                printf("[TEST] Using FIFO (timestamp order) for tie-breaking\n");
            } else {
                unsigned int preferred = 0;
                if (sscanf(answer, "%u", &preferred) == 1) {
                    state->preferred_vm = preferred;
                    printf("[TEST] VM%u will be preferred in tie-breaking\n", preferred);
                } else {
                    state->preferred_vm = 0;
                    printf("[TEST] Invalid VM ID, using FIFO (timestamp order)\n");
                }
            }
        }
        
        // Re-sort queues with new preferred VM setting
        resort_queues(state);
        
        printf("\n[TEST] Starting concurrent request processing...\n");
        printf("================================================================================\n");
        printf("\n");
        
        // Resume processing after test mode is configured
        state->processing_paused = 0;
    } else {
        state->test_mode = 0;
        state->processing_paused = 0;  // Resume normal processing
        printf("[TEST] Normal mode - processing requests as they arrive\n");
    }
}

/*
 * Process one request from pool queue
 */
void process_pool(PoolQueue *queue, MediatorState *state) {
    printf("[DEBUG] process_pool: Pool %c, queue->count=%d, queue->head=%p\n",
           queue->pool_id, queue->count, (void*)queue->head);
    fflush(stdout);
    
    Request *req = pop_request(queue, state);  // Use round-robin version
    if (!req) {
        printf("[DEBUG] process_pool: pop_request returned NULL for Pool %c\n", queue->pool_id);
        fflush(stdout);
        return;  // Queue empty
    }
    
    printf("[PROCESS] Pool %c: vm=%u, prio=%u (%s), cmd=%s\n",
           queue->pool_id, req->vm_id, req->priority,
           req->priority == 2 ? "high" : 
           req->priority == 1 ? "medium" : "low",
           req->command);
    fflush(stdout);
    
    // Execute GPU workload
    char result[512];
    execute_gpu_workload(req, result, sizeof(result));
    
    // Write response to VM
    char response_file[256];
    snprintf(response_file, sizeof(response_file),
             "/var/vgpu/vm%u/response.txt", req->vm_id);
    
    // Clear response file first to ensure clean state
    FILE *fp = fopen(response_file, "w");
    if (fp) {
        fclose(fp);
    }
    
    // Write new response
    fp = fopen(response_file, "w");
    if (fp) {
        fprintf(fp, "%s\n", result);
        fflush(fp);  // Force write to disk
        fsync(fileno(fp));  // Ensure NFS sync
        fclose(fp);
        
        printf("[RESPONSE] Sent to vm%u: %s\n", req->vm_id, result);
    } else {
        printf("[ERROR] Failed to write response for vm%u\n", req->vm_id);
    }
    
    // Clear request file now that request is processed
    char request_file[256];
    snprintf(request_file, sizeof(request_file),
             "/var/vgpu/vm%u/request.txt", req->vm_id);
    fp = fopen(request_file, "w");
    if (fp) {
        fflush(fp);
        fsync(fileno(fp));  // Ensure NFS sync
        fclose(fp);
    }
    
    state->total_processed++;
    free(req);
}

/*
 * Main daemon loop
 */
int main() {
    MediatorState state;
    time_t last_stats = time(NULL);
    int last_request_count = 0;
    
    // Set up global state pointer for signal handlers
    g_state = &state;
    
    // Register signal handlers for cleanup
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("================================================================================\n");
    printf("                    GPU Mediation Daemon\n");
    printf("                    (Dynamic Fair Scheduling with Test Mode)\n");
    printf("================================================================================\n");
    printf("\n");
    
    init_mediator(&state);
    
    printf("[START] Daemon started\n");
    printf("[START] Monitoring: /var/vgpu/vm*/request.txt\n");
    printf("[START] Priority order: High (2) > Medium (1) > Low (0) within each pool\n");
    printf("[START] Pool scheduling: Round-robin (fair, no starvation)\n");
    printf("[START] Pool separation: A and B are independent resources\n");
    printf("\n");
    printf("[INFO] Pools represent separate GPU resources, not priority levels\n");
    printf("[INFO] Both pools get fair access - no pool starvation\n");
    printf("[INFO] Press Ctrl+C to stop (will cleanup all request/response files)\n");
    printf("\n");
    
    // Ask if user wants test mode at startup
    int enable_test_mode = ask_test_mode_at_startup();
    if (enable_test_mode) {
        state.test_mode = 1;  // Enable test mode flag
        state.processing_paused = 1;  // Pause processing until 2+ VMs detected
        printf("[TEST] Test mode active - waiting for requests from 2+ VMs...\n");
        printf("[TEST] Processing will pause when 2+ VMs detected, then ask scheduling questions\n");
        printf("\n");
    }
    
    while (state.running) {
        // Step 1: Poll for new requests from all VMs (always poll, but may pause processing)
        poll_requests(&state);
        
        int current_request_count = count_total_requests(&state);
        
        // Step 2: If test mode is enabled, check for 2+ VMs before processing
        if (state.test_mode && state.processing_paused && !state.test_mode_pending) {
            // Check if we have requests from at least 2 different VMs
            uint32_t vm_list[10];
            int vm_count;
            get_requesting_vms(&state, vm_list, 10, &vm_count);
            
            if (vm_count >= 2 && current_request_count >= 2) {
                // We have 2+ VMs with requests - ask scheduling questions
                printf("\n[TEST] Detected requests from %d VM(s) with %d total requests!\n", 
                       vm_count, current_request_count);
                state.test_mode_pending = 1;   // Mark that we're asking questions
                interactive_test_mode(&state);
                state.test_mode_pending = 0;   // Questions answered, clear pending flag
                // Note: interactive_test_mode() will set processing_paused = 0 when done
            }
        }
        
        last_request_count = current_request_count;
        
        // Step 3: Process requests ONLY if not paused
        // If processing is paused (waiting for test mode questions), skip processing
        if (state.processing_paused) {
            // Still poll for new requests, but don't process until user answers
            usleep(100000);  // Short sleep while paused
            continue;
        }
        
        int processed = 0;
        
        // Step 3: Process pools - if preferred VM is set in test mode, process that pool first
        char preferred_pool = 0;
        if (state.test_mode && state.preferred_vm != 0) {
            preferred_pool = find_preferred_vm_pool(&state);
        }
        
        // Process preferred pool first (if set and has requests)
        if (preferred_pool == 'A' && state.pool_a.count > 0) {
            process_pool(&state.pool_a, &state);
            processed++;
        } else if (preferred_pool == 'B' && state.pool_b.count > 0) {
            process_pool(&state.pool_b, &state);
            processed++;
        }
        
        // Then process the other pool (if has requests and not already processed)
        if (preferred_pool != 'A' && state.pool_a.count > 0) {
            process_pool(&state.pool_a, &state);
            processed++;
        }
        if (preferred_pool != 'B' && state.pool_b.count > 0) {
            process_pool(&state.pool_b, &state);
            processed++;
        }
        
        // Step 4: Print statistics periodically
        time_t now = time(NULL);
        if (now - last_stats >= 60) {  // Every 60 seconds
            printf("\n[STATS] Total processed: %lu\n", state.total_processed);
            printf("[STATS] Pool A: queue=%d\n", state.pool_a.count);
            printf("[STATS] Pool B: queue=%d\n", state.pool_b.count);
            if (state.preferred_vm != 0) {
                printf("[STATS] Preferred VM for tie-breaking: VM%u\n", state.preferred_vm);
            }
            printf("\n");
            last_stats = now;
        }
        
        // Step 5: Adaptive sleep based on workload
        // Longer sleep when idle, shorter when busy
        if (processed == 0) {
            usleep(500000);  // 500ms when both pools idle
        } else {
            usleep(100000);  // 100ms when processing
        }
    }
    
    cleanup_files();
    printf("[STOP] Daemon stopped\n");
    return 0;
}

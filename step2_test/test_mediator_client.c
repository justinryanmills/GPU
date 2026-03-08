/*
 * Test MEDIATOR Client - Testing and Visualization Tool
 * 
 * Purpose: Simulate multiple VMs sending requests to demonstrate:
 *   - CUDA progress and response to simultaneous requests
 *   - Scheduling behavior when VMs arrive sequentially
 *   - Queue state visualization
 *   - Real-time operation experience
 * 
 * Usage: 
 *   ./test_mediator_client simultaneous --vms "1:A:2:100:200,4:B:2:150:250,2:A:1:50:75"
 *   ./test_mediator_client sequential --vms "1:A:2,2:A:2,3:A:2" --delay 0.5 --nums "100:200,150:250,50:75"
 *   ./test_mediator_client preset1
 * 
 * Requirements:
 *   - NFS share must be mounted at /mnt/vgpu
 *   - MEDIATOR daemon must be running on host
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <sys/time.h>
#include <math.h>

#define NFS_MOUNT "/mnt/vgpu"
#define RESPONSE_TIMEOUT 30  // seconds
#define POLL_INTERVAL 100000  // microseconds (0.1 seconds)
#define MAX_VMS 20
#define MAX_LINE_LEN 512

/*
 * Test Request Structure
 */
typedef struct {
    uint32_t vm_id;
    char pool_id;
    uint32_t priority;
    int num1, num2;
    struct timespec submit_time;
    struct timespec response_time;
    int result;
    int status;  // 0=PENDING, 1=SUBMITTED, 2=PROCESSING, 3=COMPLETED, -1=ERROR
    pthread_t thread_id;
} TestRequest;

/*
 * Test State
 */
typedef struct {
    TestRequest *requests;
    int count;
    pthread_mutex_t lock;
    int running;
    struct timespec start_time;
} TestState;

// Global test state
static TestState g_test_state;

/*
 * Get current time as double (seconds since start)
 */
static double get_elapsed_time(struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start->tv_sec) + (now.tv_nsec - start->tv_nsec) / 1e9;
}

/*
 * Send request to MEDIATOR (same as vm_client_vector.c)
 */
static int send_test_request(uint32_t vm_id, char pool_id, uint32_t priority, int num1, int num2) {
    char request_file[512];
    char request_data[256];
    FILE *fp;
    
    // Construct per-VM request file path
    snprintf(request_file, sizeof(request_file), 
             "%s/vm%u/request.txt", NFS_MOUNT, vm_id);
    
    // Format: "pool_id:priority:vm_id:num1:num2"
    snprintf(request_data, sizeof(request_data),
             "%c:%u:%u:%d:%d",
             pool_id, priority, vm_id, num1, num2);
    
    // Write request (explicit I/O to ensure NFS propagation)
    fp = fopen(request_file, "w");
    if (!fp) {
        perror("Failed to open request file");
        return -1;
    }
    
    fprintf(fp, "%s\n", request_data);
    fflush(fp);
    fsync(fileno(fp));
    fclose(fp);
    
    return 0;
}

/*
 * Wait for response from MEDIATOR (same as vm_client_vector.c)
 */
static int wait_for_test_response(uint32_t vm_id, int *result) {
    char response_file[512];
    FILE *fp;
    struct timespec start_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Construct response file path
    snprintf(response_file, sizeof(response_file),
             "%s/vm%u/response.txt", NFS_MOUNT, vm_id);
    
    // Poll for response
    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
        
        // Check timeout
        if (elapsed >= RESPONSE_TIMEOUT) {
            return -1;
        }
        
        // Try to read response file
        fp = fopen(response_file, "r");
        if (fp) {
            char line[256];
            if (fgets(line, sizeof(line), fp) != NULL) {
                if (sscanf(line, "%d", result) == 1) {
                    fclose(fp);
                    
                    // Clear response file after reading
                    fp = fopen(response_file, "w");
                    if (fp) {
                        fclose(fp);
                    }
                    
                    return 0;
                }
            }
            fclose(fp);
        }
        
        // Wait before next poll
        usleep(POLL_INTERVAL);
    }
    
    return -1;
}

/*
 * VM simulation thread
 */
static void* vm_simulation_thread(void *arg) {
    TestRequest *req = (TestRequest *)arg;
    int result;
    
    // Update status: SUBMITTED
    pthread_mutex_lock(&g_test_state.lock);
    clock_gettime(CLOCK_MONOTONIC, &req->submit_time);
    req->status = 1;
    pthread_mutex_unlock(&g_test_state.lock);
    
    // Send request
    if (send_test_request(req->vm_id, req->pool_id, req->priority, 
                          req->num1, req->num2) != 0) {
        pthread_mutex_lock(&g_test_state.lock);
        req->status = -1;
        pthread_mutex_unlock(&g_test_state.lock);
        return NULL;
    }
    
    // Update status: PROCESSING (waiting for response)
    pthread_mutex_lock(&g_test_state.lock);
    req->status = 2;
    pthread_mutex_unlock(&g_test_state.lock);
    
    // Wait for response
    if (wait_for_test_response(req->vm_id, &result) == 0) {
        pthread_mutex_lock(&g_test_state.lock);
        clock_gettime(CLOCK_MONOTONIC, &req->response_time);
        req->result = result;
        req->status = 3;  // COMPLETED
        pthread_mutex_unlock(&g_test_state.lock);
    } else {
        pthread_mutex_lock(&g_test_state.lock);
        req->status = -1;  // ERROR
        pthread_mutex_unlock(&g_test_state.lock);
    }
    
    return NULL;
}

/*
 * Display timeline
 */
static void display_timeline(void) {
    int i;
    double elapsed;
    const char *priority_str[] = {"Low", "Medium", "High"};
    
    printf("\n=== Timeline ===\n");
    
    pthread_mutex_lock(&g_test_state.lock);
    for (i = 0; i < g_test_state.count; i++) {
        TestRequest *req = &g_test_state.requests[i];
        elapsed = get_elapsed_time(&g_test_state.start_time);
        
        printf("T=%-6.2f  [VM-%u] Pool %c, %-6s -> Request: %d+%d",
               elapsed, req->vm_id, req->pool_id, 
               priority_str[req->priority], req->num1, req->num2);
        
        if (req->status >= 1) {
            double submit_elapsed = get_elapsed_time(&req->submit_time);
            printf(" (submitted at T=%.2f)", submit_elapsed);
        }
        
        if (req->status == 3) {
            double response_elapsed = get_elapsed_time(&req->response_time);
            double total_time = response_elapsed - get_elapsed_time(&req->submit_time);
            printf("\n            -> Response: %d (received at T=%.2f, total=%.2fs)",
                   req->result, response_elapsed, total_time);
        } else if (req->status == -1) {
            printf("\n            -> ERROR");
        } else if (req->status == 2) {
            printf(" [waiting...]");
        }
        
        printf("\n");
    }
    pthread_mutex_unlock(&g_test_state.lock);
    
    printf("\n");
}

/*
 * Display queue state (inferred from request order and timing)
 */
static void display_queue_state(void) {
    int i, j;
    TestRequest *sorted[MAX_VMS];
    int sorted_count = 0;
    
    pthread_mutex_lock(&g_test_state.lock);
    
    // Create sorted array (by priority DESC, then submit time ASC)
    for (i = 0; i < g_test_state.count; i++) {
        if (g_test_state.requests[i].status >= 1) {  // Submitted
            sorted[sorted_count++] = &g_test_state.requests[i];
        }
    }
    
    // Simple bubble sort by priority (high to low), then time (early to late)
    for (i = 0; i < sorted_count - 1; i++) {
        for (j = 0; j < sorted_count - i - 1; j++) {
            int swap = 0;
            
            // Higher priority first
            if (sorted[j]->priority < sorted[j+1]->priority) {
                swap = 1;
            }
            // Same priority: earlier timestamp first
            else if (sorted[j]->priority == sorted[j+1]->priority) {
                double time_j = get_elapsed_time(&sorted[j]->submit_time);
                double time_j1 = get_elapsed_time(&sorted[j+1]->submit_time);
                if (time_j > time_j1) {
                    swap = 1;
                }
            }
            
            if (swap) {
                TestRequest *tmp = sorted[j];
                sorted[j] = sorted[j+1];
                sorted[j+1] = tmp;
            }
        }
    }
    
    pthread_mutex_unlock(&g_test_state.lock);
    
    if (sorted_count == 0) {
        return;
    }
    
    printf("\n=== Inferred Queue State (Priority -> FIFO) ===\n");
    
    for (i = 0; i < sorted_count; i++) {
        TestRequest *req = sorted[i];
        const char *priority_str[] = {"Low", "Medium", "High"};
        double submit_elapsed = get_elapsed_time(&req->submit_time);
        
        printf("%d. VM-%u (Pool %c, %s, T=%.2f) -> %d+%d",
               i + 1, req->vm_id, req->pool_id, priority_str[req->priority],
               submit_elapsed, req->num1, req->num2);
        
        if (req->status == 3) {
            printf(" [Result: %d]", req->result);
        } else if (req->status == 2) {
            printf(" [processing...]");
        }
        
        printf("\n");
    }
    
    printf("\n");
}

/*
 * Display statistics
 */
static void display_statistics(void) {
    int i;
    int completed = 0, errors = 0;
    double total_time = 0.0, min_time = 999.0, max_time = 0.0;
    int priority_count[3] = {0, 0, 0};
    int pool_a = 0, pool_b = 0;
    
    pthread_mutex_lock(&g_test_state.lock);
    
    for (i = 0; i < g_test_state.count; i++) {
        TestRequest *req = &g_test_state.requests[i];
        
        if (req->status == 3) {
            completed++;
            double submit_elapsed = get_elapsed_time(&req->submit_time);
            double response_elapsed = get_elapsed_time(&req->response_time);
            double req_time = response_elapsed - submit_elapsed;
            
            total_time += req_time;
            if (req_time < min_time) min_time = req_time;
            if (req_time > max_time) max_time = req_time;
        } else if (req->status == -1) {
            errors++;
        }
        
        priority_count[req->priority]++;
        if (req->pool_id == 'A') pool_a++;
        else if (req->pool_id == 'B') pool_b++;
    }
    
    pthread_mutex_unlock(&g_test_state.lock);
    
    printf("\n=== Statistics ===\n");
    printf("Total Requests:    %d\n", g_test_state.count);
    printf("Completed:         %d\n", completed);
    printf("Errors:            %d\n", errors);
    
    if (completed > 0) {
        printf("Average Response:  %.2fs\n", total_time / completed);
        printf("Min Response:      %.2fs\n", min_time);
        printf("Max Response:      %.2fs\n", max_time);
    }
    
    printf("\nPriority Distribution:\n");
    printf("  High:   %d requests\n", priority_count[2]);
    printf("  Medium: %d requests\n", priority_count[1]);
    printf("  Low:    %d requests\n", priority_count[0]);
    printf("\nPool Distribution:\n");
    printf("  Pool A: %d requests\n", pool_a);
    printf("  Pool B: %d requests\n", pool_b);
    printf("\n");
}

/*
 * Display thread (updates display periodically)
 */
static void* display_thread(void *arg) {
    (void)arg;
    
    while (g_test_state.running) {
        usleep(500000);  // Update every 0.5 seconds
        
        // Clear screen (simple approach - just print newlines)
        printf("\n\n");
        
        display_timeline();
        display_queue_state();
        
        // Check if all done
        int all_done = 1;
        pthread_mutex_lock(&g_test_state.lock);
        for (int i = 0; i < g_test_state.count; i++) {
            if (g_test_state.requests[i].status == 0 || 
                g_test_state.requests[i].status == 1 ||
                g_test_state.requests[i].status == 2) {
                all_done = 0;
                break;
            }
        }
        pthread_mutex_unlock(&g_test_state.lock);
        
        if (all_done) {
            break;
        }
    }
    
    return NULL;
}

/*
 * Parse VM specification: "vm_id:pool:priority" or "vm_id:pool:priority:num1:num2"
 */
static int parse_vm_spec(const char *spec, uint32_t *vm_id, char *pool_id, 
                        uint32_t *priority, int *num1, int *num2) {
    char pool;
    uint32_t vm, prio;
    int n1 = 0, n2 = 0;
    int fields = sscanf(spec, "%u:%c:%u:%d:%d", &vm, &pool, &prio, &n1, &n2);
    
    if (fields < 3) {
        return -1;
    }
    
    if (pool != 'A' && pool != 'B') {
        return -1;
    }
    
    if (prio > 2) {
        return -1;
    }
    
    *vm_id = vm;
    *pool_id = pool;
    *priority = prio;
    
    // If numbers not provided, use defaults
    if (fields >= 5) {
        *num1 = n1;
        *num2 = n2;
    } else {
        *num1 = 100 + vm;  // Default values
        *num2 = 200 + vm;
    }
    
    return 0;
}

/*
 * Simultaneous test scenario
 */
static int test_simultaneous(int argc, char *argv[]) {
    int i;
    pthread_t display_tid;
    
    if (argc < 3) {
        fprintf(stderr, "Usage: %s simultaneous --vms \"1:A:2:100:200,4:B:2:150:250,...\"\n", argv[0]);
        return 1;
    }
    
    // Parse VM list
    char *vm_list = NULL;
    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--vms") == 0 && i + 1 < argc) {
            vm_list = argv[i + 1];
            break;
        }
    }
    
    if (!vm_list) {
        fprintf(stderr, "Error: --vms required\n");
        return 1;
    }
    
    // Count VMs
    int vm_count = 1;
    for (i = 0; vm_list[i]; i++) {
        if (vm_list[i] == ',') vm_count++;
    }
    
    if (vm_count > MAX_VMS) {
        fprintf(stderr, "Error: Maximum %d VMs supported\n", MAX_VMS);
        return 1;
    }
    
    // Allocate requests
    g_test_state.requests = (TestRequest *)calloc(vm_count, sizeof(TestRequest));
    g_test_state.count = vm_count;
    pthread_mutex_init(&g_test_state.lock, NULL);
    g_test_state.running = 1;
    clock_gettime(CLOCK_MONOTONIC, &g_test_state.start_time);
    
    // Parse and create requests
    char *vm_list_copy = strdup(vm_list);
    char *token = strtok(vm_list_copy, ",");
    i = 0;
    
    while (token && i < vm_count) {
        if (parse_vm_spec(token, &g_test_state.requests[i].vm_id,
                         &g_test_state.requests[i].pool_id,
                         &g_test_state.requests[i].priority,
                         &g_test_state.requests[i].num1,
                         &g_test_state.requests[i].num2) != 0) {
            fprintf(stderr, "Error: Invalid VM spec: %s\n", token);
            free(vm_list_copy);
            return 1;
        }
        g_test_state.requests[i].status = 0;
        i++;
        token = strtok(NULL, ",");
    }
    free(vm_list_copy);
    
    printf("================================================================================\n");
    printf("                    TEST MEDIATOR CLIENT - Simultaneous Requests\n");
    printf("================================================================================\n\n");
    printf("Test Configuration:\n");
    printf("  VMs: %d\n", vm_count);
    printf("  Timing: Simultaneous (all at T=0)\n\n");
    
    // Start display thread
    pthread_create(&display_tid, NULL, display_thread, NULL);
    
    // Start all VM threads simultaneously
    for (i = 0; i < vm_count; i++) {
        pthread_create(&g_test_state.requests[i].thread_id, NULL, 
                       vm_simulation_thread, &g_test_state.requests[i]);
    }
    
    // Wait for all threads
    for (i = 0; i < vm_count; i++) {
        pthread_join(g_test_state.requests[i].thread_id, NULL);
    }
    
    // Stop display
    g_test_state.running = 0;
    pthread_join(display_tid, NULL);
    
    // Final display
    display_timeline();
    display_queue_state();
    display_statistics();
    
    return 0;
}

/*
 * Sequential test scenario
 */
static int test_sequential(int argc, char *argv[]) {
    int i;
    double delay = 0.5;  // Default delay
    pthread_t display_tid;
    
    if (argc < 3) {
        fprintf(stderr, "Usage: %s sequential --vms \"1:A:2,2:A:2,...\" --delay 0.5 [--nums \"100:200,150:250,...\"]\n", argv[0]);
        return 1;
    }
    
    // Parse arguments
    char *vm_list = NULL;
    char *num_list = NULL;
    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--vms") == 0 && i + 1 < argc) {
            vm_list = argv[i + 1];
        } else if (strcmp(argv[i], "--delay") == 0 && i + 1 < argc) {
            delay = atof(argv[i + 1]);
        } else if (strcmp(argv[i], "--nums") == 0 && i + 1 < argc) {
            num_list = argv[i + 1];
        }
    }
    
    if (!vm_list) {
        fprintf(stderr, "Error: --vms required\n");
        return 1;
    }
    
    // Count VMs
    int vm_count = 1;
    for (i = 0; vm_list[i]; i++) {
        if (vm_list[i] == ',') vm_count++;
    }
    
    if (vm_count > MAX_VMS) {
        fprintf(stderr, "Error: Maximum %d VMs supported\n", MAX_VMS);
        return 1;
    }
    
    // Parse numbers if provided
    int nums[MAX_VMS][2];
    int num_count = 0;
    if (num_list) {
        char *num_list_copy = strdup(num_list);
        char *token = strtok(num_list_copy, ",");
        while (token && num_count < MAX_VMS) {
            if (sscanf(token, "%d:%d", &nums[num_count][0], &nums[num_count][1]) == 2) {
                num_count++;
            }
            token = strtok(NULL, ",");
        }
        free(num_list_copy);
    }
    
    // Allocate requests
    g_test_state.requests = (TestRequest *)calloc(vm_count, sizeof(TestRequest));
    g_test_state.count = vm_count;
    pthread_mutex_init(&g_test_state.lock, NULL);
    g_test_state.running = 1;
    clock_gettime(CLOCK_MONOTONIC, &g_test_state.start_time);
    
    // Parse and create requests
    char *vm_list_copy = strdup(vm_list);
    char *token = strtok(vm_list_copy, ",");
    i = 0;
    
    while (token && i < vm_count) {
        if (parse_vm_spec(token, &g_test_state.requests[i].vm_id,
                         &g_test_state.requests[i].pool_id,
                         &g_test_state.requests[i].priority,
                         &g_test_state.requests[i].num1,
                         &g_test_state.requests[i].num2) != 0) {
            fprintf(stderr, "Error: Invalid VM spec: %s\n", token);
            free(vm_list_copy);
            return 1;
        }
        
        // Override numbers if provided
        if (i < num_count) {
            g_test_state.requests[i].num1 = nums[i][0];
            g_test_state.requests[i].num2 = nums[i][1];
        }
        
        g_test_state.requests[i].status = 0;
        i++;
        token = strtok(NULL, ",");
    }
    free(vm_list_copy);
    
    printf("================================================================================\n");
    printf("                    TEST MEDIATOR CLIENT - Sequential Requests\n");
    printf("================================================================================\n\n");
    printf("Test Configuration:\n");
    printf("  VMs: %d\n", vm_count);
    printf("  Timing: Sequential (delay=%.2fs between requests)\n\n", delay);
    
    // Start display thread
    pthread_create(&display_tid, NULL, display_thread, NULL);
    
    // Start VM threads sequentially with delay
    for (i = 0; i < vm_count; i++) {
        pthread_create(&g_test_state.requests[i].thread_id, NULL, 
                       vm_simulation_thread, &g_test_state.requests[i]);
        if (i < vm_count - 1) {
            usleep((int)(delay * 1000000));  // Delay in microseconds
        }
    }
    
    // Wait for all threads
    for (i = 0; i < vm_count; i++) {
        pthread_join(g_test_state.requests[i].thread_id, NULL);
    }
    
    // Stop display
    g_test_state.running = 0;
    pthread_join(display_tid, NULL);
    
    // Final display
    display_timeline();
    display_queue_state();
    display_statistics();
    
    return 0;
}

/*
 * Preset test scenario 1: Mixed priorities, simultaneous
 */
static int test_preset1(void) {
    printf("Running Preset Test 1: Mixed Priorities (Simultaneous)\n");
    char *argv[] = {
        "test_mediator_client",
        "simultaneous",
        "--vms",
        "1:A:2:100:200,4:B:2:150:250,2:A:1:50:75,5:B:1:80:120,3:A:0:200:300"
    };
    return test_simultaneous(5, argv);
}

/*
 * Main function
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <scenario> [options]\n", argv[0]);
        fprintf(stderr, "\nScenarios:\n");
        fprintf(stderr, "  simultaneous --vms \"1:A:2:100:200,4:B:2:150:250,...\"\n");
        fprintf(stderr, "  sequential --vms \"1:A:2,2:A:2,...\" --delay 0.5 [--nums \"100:200,150:250,...\"]\n");
        fprintf(stderr, "  preset1\n");
        return 1;
    }
    
    if (strcmp(argv[1], "simultaneous") == 0) {
        return test_simultaneous(argc, argv);
    } else if (strcmp(argv[1], "sequential") == 0) {
        return test_sequential(argc, argv);
    } else if (strcmp(argv[1], "preset1") == 0) {
        return test_preset1();
    } else {
        fprintf(stderr, "Unknown scenario: %s\n", argv[1]);
        return 1;
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/select.h>
#include <dirent.h>
#include "cuda_vector_add.h"
#include "vgpu_protocol.h"

// VGPU_SOCKET_PATH is defined in vgpu_protocol.h
#define MAX_CONNECTIONS 32
#define SOCKET_BACKLOG 10
#define MAX_SERVER_SOCKETS 16

/* Multiple server sockets — one per QEMU chroot */
static int    g_server_fds[MAX_SERVER_SOCKETS];
static char   g_socket_paths[MAX_SERVER_SOCKETS][512];
static int    g_num_servers = 0;

typedef struct Request {
    char pool_id;           // 'A' or 'B'
    uint32_t priority;      // 2=high, 1=medium, 0=low
    uint32_t vm_id;
    uint32_t request_id;    // Request tracking ID
    int num1, num2;         // Numbers to add
    int client_fd;           // Socket file descriptor for response
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
static int setup_socket_server(const char *socket_path);
static void handle_client_connection(int client_fd);

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
        printf("[ENQUEUE] Pool %c: vm=%u, req_id=%u, prio=%u, %d+%d (queue empty)\n",
               new_req->pool_id, new_req->vm_id, new_req->request_id,
               new_req->priority, new_req->num1, new_req->num2);
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
    
    printf("[ENQUEUE] Pool %c: vm=%u, req_id=%u, prio=%u, %d+%d\n",
           new_req->pool_id, new_req->vm_id, new_req->request_id,
           new_req->priority, new_req->num1, new_req->num2);
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

static int send_response(int client_fd, uint32_t vm_id, uint32_t request_id,
                        char pool_id, uint8_t priority, int result, uint32_t exec_time_us) {
    VGPUSocketHeader hdr;
    VGPUResponse resp;
    uint32_t result_value = (uint32_t)result;
    struct iovec iov[3];
    struct msghdr msg;
    ssize_t sent;

    // Build response header
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = VGPU_SOCKET_MAGIC;
    hdr.msg_type = VGPU_MSG_RESPONSE;
    hdr.vm_id = vm_id;
    hdr.request_id = request_id;
    hdr.pool_id = pool_id;
    hdr.priority = priority;
    hdr.payload_len = VGPU_RESPONSE_HEADER_SIZE + sizeof(uint32_t);  // header + 1 result

    // Build response payload
    memset(&resp, 0, sizeof(resp));
    resp.version = VGPU_PROTOCOL_VERSION;
    resp.status = 0;  // success
    resp.result_count = 1;
    resp.data_offset = VGPU_RESPONSE_HEADER_SIZE + sizeof(uint32_t);  // after results
    resp.data_length = 0;  // no variable data
    resp.exec_time_us = exec_time_us;
    resp.reserved[0] = 0;
    resp.reserved[1] = 0;

    // Send header + response struct + result value
    iov[0].iov_base = &hdr;
    iov[0].iov_len = VGPU_SOCKET_HDR_SIZE;
    iov[1].iov_base = &resp;
    iov[1].iov_len = VGPU_RESPONSE_HEADER_SIZE;
    iov[2].iov_base = &result_value;
    iov[2].iov_len = sizeof(uint32_t);

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = iov;
    msg.msg_iovlen = 3;

    sent = sendmsg(client_fd, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
        fprintf(stderr, "[ERROR] sendmsg failed: %s\n", strerror(errno));
        return -1;
    }

    return 0;
}

static void cuda_result_callback(int result, void *user_data) {
    Request *req = (Request *)user_data;
    MediatorState *state = &g_state;
    
    printf("[RESULT] Pool %c: vm=%u, req_id=%u, result=%d\n",
           req->pool_id, req->vm_id, req->request_id, result);
    
    // Calculate execution time (simplified - use 0 for now)
    uint32_t exec_time_us = 0;
    
    // Send response via socket
    if (send_response(req->client_fd, req->vm_id, req->request_id,
                     req->pool_id, (uint8_t)req->priority, result, exec_time_us) == 0) {
        printf("[RESPONSE] Sent to vm%u (req_id=%u): %d\n",
               req->vm_id, req->request_id, result);
    } else {
        fprintf(stderr, "[ERROR] Failed to send response to vm%u\n", req->vm_id);
    }
    
    // Close client socket
    close(req->client_fd);
    
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
    
    printf("[PROCESS] Pool %c: vm=%u, req_id=%u, prio=%u, %d+%d\n",
           req->pool_id, req->vm_id, req->request_id,
           req->priority, req->num1, req->num2);
    
    // Set user_data to request for callback
    req->user_data = req;
    
    // Send to CUDA asynchronously
    if (cuda_vector_add_async(req->num1, req->num2, cuda_result_callback, req) != 0) {
        fprintf(stderr, "[ERROR] Failed to start CUDA operation\n");
        state->cuda_busy = 0;
        state->current_request = NULL;
        close(req->client_fd);
        free(req);
    }
}

static int parse_vgpu_request(const uint8_t *payload, uint16_t payload_len,
                              int *num1, int *num2) {
    if (payload_len < VGPU_REQUEST_HEADER_SIZE) {
        return -1;
    }

    VGPURequest *req = (VGPURequest *)payload;

    // Validate protocol version
    if (req->version != VGPU_PROTOCOL_VERSION) {
        fprintf(stderr, "[ERROR] Invalid protocol version: 0x%08x\n", req->version);
        return -1;
    }

    // Check opcode
    if (req->opcode != VGPU_OP_CUDA_KERNEL) {
        fprintf(stderr, "[ERROR] Unsupported opcode: 0x%04x\n", req->opcode);
        return -1;
    }

    // For vector addition, we expect 2 parameters (num1, num2)
    if (req->param_count != 2) {
        fprintf(stderr, "[ERROR] Expected 2 parameters, got %u\n", req->param_count);
        return -1;
    }

    // Check payload has enough space for params
    if (payload_len < VGPU_REQUEST_HEADER_SIZE + 2 * sizeof(uint32_t)) {
        return -1;
    }

    // Extract parameters (they follow the header)
    uint32_t *params = (uint32_t *)(payload + VGPU_REQUEST_HEADER_SIZE);
    *num1 = (int)params[0];
    *num2 = (int)params[1];

    return 0;
}

static void handle_client_connection(int client_fd) {
    uint8_t rx_buf[VGPU_SOCKET_HDR_SIZE + VGPU_SOCKET_MAX_PAYLOAD];
    ssize_t n;
    VGPUSocketHeader *hdr;
    uint32_t total_len;

    // Read socket header
    n = read(client_fd, rx_buf, VGPU_SOCKET_HDR_SIZE);
    if (n < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            fprintf(stderr, "[ERROR] read header failed: %s\n", strerror(errno));
        }
        close(client_fd);
        return;
    }

    if (n < VGPU_SOCKET_HDR_SIZE) {
        fprintf(stderr, "[ERROR] Incomplete header (got %zd bytes)\n", n);
        close(client_fd);
        return;
    }

    hdr = (VGPUSocketHeader *)rx_buf;

    // Validate magic
    if (hdr->magic != VGPU_SOCKET_MAGIC) {
        fprintf(stderr, "[ERROR] Invalid magic: 0x%08x\n", hdr->magic);
        close(client_fd);
        return;
    }

    // Check message type
    if (hdr->msg_type != VGPU_MSG_REQUEST) {
        if (hdr->msg_type == VGPU_MSG_PING) {
            // Reply with PONG
            VGPUSocketHeader pong;
            memset(&pong, 0, sizeof(pong));
            pong.magic = VGPU_SOCKET_MAGIC;
            pong.msg_type = VGPU_MSG_PONG;
            pong.vm_id = hdr->vm_id;
            write(client_fd, &pong, VGPU_SOCKET_HDR_SIZE);
        }
        close(client_fd);
        return;
    }

    // Read payload
    total_len = VGPU_SOCKET_HDR_SIZE + hdr->payload_len;
    if (hdr->payload_len > 0) {
        n = read(client_fd, rx_buf + VGPU_SOCKET_HDR_SIZE, hdr->payload_len);
        if (n < 0) {
            fprintf(stderr, "[ERROR] read payload failed: %s\n", strerror(errno));
            close(client_fd);
            return;
        }
        if ((size_t)n < hdr->payload_len) {
            fprintf(stderr, "[ERROR] Incomplete payload (got %zd, expected %u)\n",
                    n, hdr->payload_len);
            close(client_fd);
            return;
        }
    }

    // Parse request payload
    int num1, num2;
    if (parse_vgpu_request(rx_buf + VGPU_SOCKET_HDR_SIZE, hdr->payload_len,
                          &num1, &num2) != 0) {
        fprintf(stderr, "[ERROR] Failed to parse request from vm%u\n", hdr->vm_id);
        close(client_fd);
        return;
    }

    // Validate pool_id
    if (hdr->pool_id != 'A' && hdr->pool_id != 'B') {
        fprintf(stderr, "[ERROR] Invalid pool_id: '%c'\n", hdr->pool_id);
        close(client_fd);
        return;
    }

    // Validate priority
    if (hdr->priority > VGPU_PRIORITY_HIGH) {
        fprintf(stderr, "[ERROR] Invalid priority: %u\n", hdr->priority);
        close(client_fd);
        return;
    }

    // Create request
    Request *req = (Request *)malloc(sizeof(Request));
    if (!req) {
        fprintf(stderr, "[ERROR] Failed to allocate request\n");
        close(client_fd);
        return;
    }

    req->pool_id = hdr->pool_id;
    req->priority = hdr->priority;
    req->vm_id = hdr->vm_id;
    req->request_id = hdr->request_id;
    req->num1 = num1;
    req->num2 = num2;
    req->client_fd = client_fd;  // Keep socket open for response
    req->timestamp = time(NULL);
    req->next = NULL;
    req->user_data = NULL;

    // Enqueue request
    enqueue_request(&g_state, req);
}

static int discover_all_qemu_chroots(char *chroots[], int max_chroots) {
    DIR *proc_dir;
    struct dirent *entry;
    char cmdline_path[256];
    char cmdline[4096];
    int count = 0;

    proc_dir = opendir("/proc");
    if (!proc_dir) {
        return 0;
    }

    while ((entry = readdir(proc_dir)) != NULL && count < max_chroots) {
        // Only look at numeric directories (PIDs)
        if (entry->d_name[0] < '0' || entry->d_name[0] > '9') {
            continue;
        }

        snprintf(cmdline_path, sizeof(cmdline_path), "/proc/%s/cmdline", entry->d_name);
        FILE *f = fopen(cmdline_path, "r");
        if (!f) continue;

        // cmdline is NUL-separated args; read the whole thing
        size_t len = fread(cmdline, 1, sizeof(cmdline) - 1, f);
        fclose(f);
        if (len == 0) continue;
        cmdline[len] = '\0';

        // Replace NULs with spaces so strstr works
        for (size_t i = 0; i < len; i++) {
            if (cmdline[i] == '\0') cmdline[i] = ' ';
        }

        // Check if this process has "vgpu-stub" in its command line
        if (strstr(cmdline, "vgpu-stub") == NULL) {
            continue;
        }

        // Found a QEMU process with vgpu-stub — look for -chroot argument
        char *chroot_arg = strstr(cmdline, "-chroot ");
        if (chroot_arg) {
            chroot_arg += strlen("-chroot ");
            while (*chroot_arg == ' ') chroot_arg++;
            char *end = strchr(chroot_arg, ' ');
            size_t path_len = end ? (size_t)(end - chroot_arg) : strlen(chroot_arg);

            // Check for duplicates
            int duplicate = 0;
            for (int i = 0; i < count; i++) {
                if (strlen(chroots[i]) == path_len &&
                    memcmp(chroots[i], chroot_arg, path_len) == 0) {
                    duplicate = 1;
                    break;
                }
            }
            if (duplicate) continue;

            chroots[count] = malloc(path_len + 1);
            if (chroots[count]) {
                memcpy(chroots[count], chroot_arg, path_len);
                chroots[count][path_len] = '\0';
                count++;
            }
        }
    }

    closedir(proc_dir);
    return count;
}

static int setup_socket_server(const char *socket_path) {
    struct sockaddr_un addr;
    int fd;
    int reuse = 1;

    // Remove existing socket file
    unlink(socket_path);

    // Create socket
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return -1;
    }

    // Set socket options
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        perror("setsockopt");
        close(fd);
        return -1;
    }

    // Make non-blocking
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl");
        close(fd);
        return -1;
    }

    // Bind to filesystem Unix socket
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(fd);
        return -1;
    }

    // Make socket accessible to QEMU (which runs as non-root user)
    chmod(socket_path, 0777);

    // Listen
    if (listen(fd, SOCKET_BACKLOG) < 0) {
        perror("listen");
        close(fd);
        return -1;
    }

    printf("[SOCKET] Listening on %s\n", socket_path);
    return fd;
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
    fd_set read_fds;
    int max_fd;
    struct timeval timeout;

    printf("[MEDIATOR] Starting main loop...\n");
    printf("[MEDIATOR] Listening on %d socket(s) for VM connections\n", g_num_servers);

    while (state->running && !g_shutdown) {
        // Setup select() for ALL server sockets
        FD_ZERO(&read_fds);
        max_fd = -1;
        for (int i = 0; i < g_num_servers; i++) {
            FD_SET(g_server_fds[i], &read_fds);
            if (g_server_fds[i] > max_fd) {
                max_fd = g_server_fds[i];
            }
        }

        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int ret = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);
        if (ret < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted by signal
            }
            perror("select");
            break;
        }

        // Accept new connections on ANY server socket
        for (int i = 0; i < g_num_servers; i++) {
            if (FD_ISSET(g_server_fds[i], &read_fds)) {
                struct sockaddr_un client_addr;
                socklen_t client_len = sizeof(client_addr);
                int client_fd = accept(g_server_fds[i],
                                       (struct sockaddr *)&client_addr, &client_len);
                if (client_fd >= 0) {
                    // Make client socket non-blocking
                    int flags = fcntl(client_fd, F_GETFL, 0);
                    if (flags >= 0) {
                        fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
                    }
                    printf("[SOCKET] New connection on %s (fd=%d)\n",
                           g_socket_paths[i], client_fd);
                    handle_client_connection(client_fd);
                }
            }
        }

        // Process queue (send to CUDA if idle)
        process_queue(state);

        // Print statistics every 60 seconds
        time_t now = time(NULL);
        if (now - last_stats >= 60) {
            print_stats(state);
            last_stats = now;
        }
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
        if (req->client_fd >= 0) {
            close(req->client_fd);
        }
        free(req);
    }

    // Close all server sockets and remove socket files
    for (int i = 0; i < g_num_servers; i++) {
        close(g_server_fds[i]);
        if (g_socket_paths[i][0]) {
            unlink(g_socket_paths[i]);
        }
    }
}

static void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --socket-path <path>   Override socket path (host filesystem)\n");
    printf("  --help                 Show this help\n");
    printf("\nIf --socket-path is not given, the mediator auto-discovers\n");
    printf("the QEMU chroot directory and creates the socket inside it.\n");
    printf("Fallback: %s\n", VGPU_SOCKET_PATH);
}

int main(int argc, char *argv[]) {
    const char *override_path = NULL;

    printf("================================================================================\n");
    printf("          MEDIATOR DAEMON v2 - MMIO/Socket Communication\n");
    printf("================================================================================\n\n");

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--socket-path") == 0 && i + 1 < argc) {
            override_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

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

    // Determine socket paths and create server sockets
    g_num_servers = 0;

    if (override_path) {
        // User provided explicit path — single socket
        snprintf(g_socket_paths[0], sizeof(g_socket_paths[0]), "%s", override_path);
        g_server_fds[0] = setup_socket_server(g_socket_paths[0]);
        if (g_server_fds[0] < 0) {
            fprintf(stderr, "[ERROR] Failed to setup socket at %s\n", override_path);
            cuda_cleanup();
            return 1;
        }
        g_num_servers = 1;
        printf("[CONFIG] Using user-specified socket: %s\n", g_socket_paths[0]);
    } else {
        // Auto-discover ALL QEMU chroots
        char *chroots[MAX_SERVER_SOCKETS];
        int num_chroots = discover_all_qemu_chroots(chroots, MAX_SERVER_SOCKETS);

        if (num_chroots > 0) {
            printf("[CONFIG] Found %d QEMU VM(s) with vgpu-stub:\n", num_chroots);
            for (int i = 0; i < num_chroots; i++) {
                // Create /tmp inside the chroot if needed
                char tmp_dir[512];
                snprintf(tmp_dir, sizeof(tmp_dir), "%s/tmp", chroots[i]);
                mkdir(tmp_dir, 0755);

                // Socket path = <chroot>/tmp/vgpu-mediator.sock
                snprintf(g_socket_paths[g_num_servers],
                         sizeof(g_socket_paths[g_num_servers]),
                         "%s%s", chroots[i], VGPU_SOCKET_PATH);

                g_server_fds[g_num_servers] = setup_socket_server(g_socket_paths[g_num_servers]);
                if (g_server_fds[g_num_servers] >= 0) {
                    printf("  [%d] %s -> %s\n", g_num_servers + 1, chroots[i],
                           g_socket_paths[g_num_servers]);
                    g_num_servers++;
                } else {
                    fprintf(stderr, "[WARN] Failed to setup socket in %s\n", chroots[i]);
                }
                free(chroots[i]);
            }
        } else {
            // Fallback: no QEMU found, use default path
            snprintf(g_socket_paths[0], sizeof(g_socket_paths[0]), "%s", VGPU_SOCKET_PATH);
            g_server_fds[0] = setup_socket_server(g_socket_paths[0]);
            if (g_server_fds[0] >= 0) {
                g_num_servers = 1;
            }
            printf("[CONFIG] No QEMU chroot found, using fallback: %s\n", g_socket_paths[0]);
        }
    }

    if (g_num_servers == 0) {
        fprintf(stderr, "[ERROR] No server sockets created\n");
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

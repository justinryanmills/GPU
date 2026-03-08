/* Phase 3 mediator: WFQ, rate limiter, watchdog, metrics, admin socket. */
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
#include <sys/uio.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/select.h>
#include <dirent.h>

#include "vgpu_protocol.h"
#include "cuda_vector_add.h"

#include "scheduler_wfq.h"
#include "rate_limiter.h"
#include "metrics.h"
#include "watchdog.h"
#include "nvml_monitor.h"

#include "vgpu_config.h"

#include "cuda_protocol.h"
#include "cuda_executor.h"

#define MAX_CONNECTIONS      32
#define SOCKET_BACKLOG       10
#define MAX_SERVER_SOCKETS   16
#define ADMIN_BUF_SIZE       (64 * 1024)

static int    g_server_fds[MAX_SERVER_SOCKETS];
static char   g_socket_paths[MAX_SERVER_SOCKETS][512];
static int    g_num_servers = 0;
static int    g_admin_fd = -1;
static volatile int g_shutdown = 0;
static wfq_scheduler_t g_scheduler;
static rate_limiter_t  g_rate_limiter;
static metrics_t       g_metrics;
static watchdog_t      g_watchdog;
static uint64_t g_total_processed = 0;
static uint64_t g_pool_a_processed = 0;
static uint64_t g_pool_b_processed = 0;
static int  g_cuda_busy = 0;
static pthread_mutex_t g_cuda_lock = PTHREAD_MUTEX_INITIALIZER;
static cuda_executor_t *g_cuda_executor = NULL;
static wfq_entry_t g_current_job;
static int g_has_current_job = 0;
static sqlite3 *g_db = NULL;
#define MAX_TRACKED_CONNECTIONS 128
typedef struct {
    uint32_t vm_id;
    int      fd;
    time_t   connect_time;
    uint64_t messages_sent;
    uint64_t messages_received;
    int      is_active;
} connection_info_t;

static connection_info_t g_connections[MAX_TRACKED_CONNECTIONS];
static int g_num_connections = 0;
static pthread_mutex_t g_connections_lock = PTHREAD_MUTEX_INITIALIZER;

static int  setup_socket_server(const char *socket_path);
static int  setup_admin_socket(void);
static void track_connection(uint32_t vm_id, int fd);
static void untrack_connection(int fd);
static void handle_client_connection(int client_fd);
static int  handle_persistent_message(int client_fd);
static void handle_cuda_call(int client_fd, VGPUSocketHeader *sock_hdr,
                              const uint8_t *payload, uint32_t payload_len);
static void handle_admin_connection(int client_fd);
static void dispatch_next_job(void);
static void execute_job(wfq_entry_t *entry);

static void signal_handler(int sig)
{
    printf("\n[SHUTDOWN] Received signal %d, shutting down gracefully...\n", sig);
    g_shutdown = 1;
}

static int discover_all_qemu_chroots(char *chroots[], int max_chroots, int verbose)
{
    DIR *proc_dir;
    struct dirent *entry;
    char path_buf[512];
    char cmdline[4096];
    char chroot_path[512];
    int count = 0;
    int scanned = 0;
    int vgpu_found = 0;

    proc_dir = opendir("/proc");
    if (!proc_dir) {
        fprintf(stderr, "[DISCOVERY] ERROR: Cannot open /proc: %s\n", strerror(errno));
        return 0;
    }

    while ((entry = readdir(proc_dir)) != NULL && count < max_chroots) {
        /* Only look at numeric entries (PIDs) */
        if (entry->d_name[0] < '0' || entry->d_name[0] > '9')
            continue;

        scanned++;

        /* Read /proc/<pid>/cmdline */
        snprintf(path_buf, sizeof(path_buf), "/proc/%s/cmdline", entry->d_name);
        FILE *f = fopen(path_buf, "r");
        if (!f) continue;

        size_t len = fread(cmdline, 1, sizeof(cmdline) - 1, f);
        fclose(f);
        if (len == 0) continue;
        cmdline[len] = '\0';

        for (size_t i = 0; i < len; i++) {
            if (cmdline[i] == '\0') cmdline[i] = ' ';
        }

        if (strstr(cmdline, "vgpu-cuda") == NULL)
            continue;

        vgpu_found++;

        snprintf(path_buf, sizeof(path_buf), "/proc/%s/root", entry->d_name);
        ssize_t rl = readlink(path_buf, chroot_path, sizeof(chroot_path) - 1);
        if (rl < 0) {
            fprintf(stderr, "[DISCOVERY] WARNING: readlink(%s) failed: %s\n",
                    path_buf, strerror(errno));
            continue;
        }
        chroot_path[rl] = '\0';

        if (strcmp(chroot_path, "/") == 0) {
            fprintf(stderr, "[DISCOVERY] INFO: vgpu-cuda process pid=%s "
                    "is not chrooted (root=/), skipping\n", entry->d_name);
            continue;
        }

        int duplicate = 0;
        for (int i = 0; i < count; i++) {
            if (strcmp(chroots[i], chroot_path) == 0) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) continue;

        chroots[count] = strdup(chroot_path);
        if (chroots[count]) {
            count++;
        } else {
            fprintf(stderr, "[DISCOVERY] WARNING: strdup failed for chroot path\n");
        }
    }

    closedir(proc_dir);

    if (verbose) {
        if (count > 0) {
            printf("[DISCOVERY] Scanned %d processes, found %d vgpu-cuda process(es), "
                   "discovered %d unique chroot(s)\n", scanned, vgpu_found, count);
        } else if (vgpu_found > 0) {
            fprintf(stderr, "[DISCOVERY] WARNING: Found %d vgpu-cuda process(es) but "
                    "none were chrooted — socket will be created at %s (fallback)\n",
                    vgpu_found, VGPU_SOCKET_PATH);
        } else {
            printf("[DISCOVERY] Scanned %d processes, no vgpu-cuda QEMU process found yet\n",
                   scanned);
        }
    }

    return count;
}

static int setup_socket_server(const char *socket_path)
{
    struct sockaddr_un addr;
    int fd;
    struct stat st;

    if (unlink(socket_path) < 0 && errno != ENOENT) {
        fprintf(stderr, "[SOCKET] WARNING: Failed to unlink existing socket %s: %s\n",
                socket_path, strerror(errno));
    }

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "[SOCKET] ERROR: socket() failed for %s: %s (errno=%d)\n",
                socket_path, strerror(errno), errno);
        return -1;
    }

    int reuse = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        fprintf(stderr, "[SOCKET] WARNING: setsockopt(SO_REUSEADDR) failed: %s\n",
                strerror(errno));
    }

    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        fprintf(stderr, "[SOCKET] ERROR: fcntl(O_NONBLOCK) failed for %s: %s (errno=%d)\n",
                socket_path, strerror(errno), errno);
        close(fd);
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    char *dir_path = strdup(socket_path);
    char *last_slash = strrchr(dir_path, '/');
    if (last_slash) {
        *last_slash = '\0';
        if (mkdir(dir_path, 0755) < 0 && errno != EEXIST) {
            fprintf(stderr, "[SOCKET] WARNING: Failed to create directory %s: %s\n",
                    dir_path, strerror(errno));
        }
    }
    free(dir_path);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[SOCKET] ERROR: bind() failed for %s: %s (errno=%d)\n",
                socket_path, strerror(errno), errno);
        close(fd);
        return -1;
    }

    if (chmod(socket_path, 0666) < 0) {
        fprintf(stderr, "[SOCKET] WARNING: chmod(0666) failed for %s: %s\n",
                socket_path, strerror(errno));
    } else {
        if (stat(socket_path, &st) == 0) {
            mode_t mode = st.st_mode & 0777;
            if ((mode & 0666) != 0666) {
                fprintf(stderr, "[SOCKET] WARNING: Socket %s has permissions %03o, expected 0666\n",
                        socket_path, mode);
            }
        }
    }

    if (listen(fd, SOCKET_BACKLOG) < 0) {
        fprintf(stderr, "[SOCKET] ERROR: listen() failed for %s: %s (errno=%d)\n",
                socket_path, strerror(errno), errno);
        close(fd);
        return -1;
    }

    printf("[SOCKET] Successfully listening on %s (fd=%d, permissions=0666)\n",
           socket_path, fd);
    return fd;
}

static int setup_admin_socket(void)
{
    mkdir("/var/vgpu", 0755);

    int fd = setup_socket_server(VGPU_ADMIN_SOCKET_PATH);
    if (fd >= 0) {
        printf("[ADMIN] Admin socket listening on %s\n", VGPU_ADMIN_SOCKET_PATH);
    }
    return fd;
}

static int parse_vgpu_request(const uint8_t *payload, uint32_t payload_len,
                              int *num1, int *num2)
{
    if (payload_len < VGPU_REQUEST_HEADER_SIZE)
        return -1;

    VGPURequest *req = (VGPURequest *)payload;

    if (req->version != VGPU_PROTOCOL_VERSION) {
        fprintf(stderr, "[ERROR] Invalid protocol version: 0x%08x\n",
                req->version);
        return -1;
    }
    if (req->opcode != VGPU_OP_CUDA_KERNEL) {
        fprintf(stderr, "[ERROR] Unsupported opcode: 0x%04x\n", req->opcode);
        return -1;
    }
    if (req->param_count != 2) {
        fprintf(stderr, "[ERROR] Expected 2 parameters, got %u\n",
                req->param_count);
        return -1;
    }
    if (payload_len < VGPU_REQUEST_HEADER_SIZE + 2 * sizeof(uint32_t))
        return -1;

    uint32_t *params = (uint32_t *)(payload + VGPU_REQUEST_HEADER_SIZE);
    *num1 = (int)params[0];
    *num2 = (int)params[1];
    return 0;
}

static int send_response(int client_fd, uint32_t vm_id, uint32_t request_id,
                         char pool_id, uint8_t priority, int result,
                         uint32_t exec_time_us)
{
    VGPUSocketHeader hdr;
    VGPUResponse resp;
    uint32_t result_value = (uint32_t)result;
    struct iovec iov[3];
    struct msghdr msg;

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = VGPU_SOCKET_MAGIC;
    hdr.msg_type    = VGPU_MSG_RESPONSE;
    hdr.vm_id       = vm_id;
    hdr.request_id  = request_id;
    hdr.pool_id     = pool_id;
    hdr.priority    = priority;
    hdr.payload_len = VGPU_RESPONSE_HEADER_SIZE + sizeof(uint32_t);

    memset(&resp, 0, sizeof(resp));
    resp.version      = VGPU_PROTOCOL_VERSION;
    resp.status       = 0;
    resp.result_count = 1;
    resp.data_offset  = VGPU_RESPONSE_HEADER_SIZE + sizeof(uint32_t);
    resp.data_length  = 0;
    resp.exec_time_us = exec_time_us;

    iov[0].iov_base = &hdr;
    iov[0].iov_len  = VGPU_SOCKET_HDR_SIZE;
    iov[1].iov_base = &resp;
    iov[1].iov_len  = VGPU_RESPONSE_HEADER_SIZE;
    iov[2].iov_base = &result_value;
    iov[2].iov_len  = sizeof(uint32_t);

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov    = iov;
    msg.msg_iovlen = 3;

    ssize_t sent = sendmsg(client_fd, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
        fprintf(stderr, "[ERROR] sendmsg failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static void send_rejection(int client_fd, uint32_t vm_id, uint32_t request_id,
                           char pool_id, uint8_t priority, uint32_t msg_type)
{
    VGPUSocketHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = VGPU_SOCKET_MAGIC;
    hdr.msg_type    = msg_type;
    hdr.vm_id       = vm_id;
    hdr.request_id  = request_id;
    hdr.pool_id     = pool_id;
    hdr.priority    = priority;
    hdr.payload_len = 0;

    write(client_fd, &hdr, VGPU_SOCKET_HDR_SIZE);
    close(client_fd);
}

static void cuda_result_callback(int result, void *user_data)
{
    wfq_entry_t *entry = (wfq_entry_t *)user_data;

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (now.tv_sec - entry->enqueue_time.tv_sec)
                   + (now.tv_nsec - entry->enqueue_time.tv_nsec) / 1e9;
    uint32_t exec_time_us = (uint32_t)(elapsed * 1e6);

    printf("[RESULT] Pool %c: vm=%u, req=%u, result=%d, time=%u us\n",
           entry->pool_id, entry->vm_id, entry->request_id,
           result, exec_time_us);

    if (send_response(entry->client_fd, entry->vm_id, entry->request_id,
                      entry->pool_id, entry->priority, result,
                      exec_time_us) == 0) {
        printf("[RESPONSE] Sent to vm%u (req=%u): %d\n",
               entry->vm_id, entry->request_id, result);
    } else {
        fprintf(stderr, "[ERROR] Failed to send response to vm%u\n",
                entry->vm_id);
    }

    pthread_mutex_lock(&g_connections_lock);
    for (int i = 0; i < g_num_connections; i++) {
        if (g_connections[i].fd == entry->client_fd && g_connections[i].is_active) {
            g_connections[i].messages_sent++;
            break;
        }
    }
    pthread_mutex_unlock(&g_connections_lock);

    untrack_connection(entry->client_fd);
    close(entry->client_fd);

    uint64_t latency_us = (uint64_t)exec_time_us;
    metrics_record_job(&g_metrics, entry->vm_id, latency_us, latency_us);

    wfq_complete(&g_scheduler, entry->vm_id, exec_time_us);

    wd_job_completed(&g_watchdog, entry->vm_id, entry->request_id);

    g_total_processed++;
    if (entry->pool_id == 'A') g_pool_a_processed++;
    else                        g_pool_b_processed++;

    pthread_mutex_lock(&g_cuda_lock);
    g_cuda_busy = 0;
    g_has_current_job = 0;
    pthread_mutex_unlock(&g_cuda_lock);

    free(entry);

    dispatch_next_job();
}

static void execute_job(wfq_entry_t *entry)
{
    pthread_mutex_lock(&g_cuda_lock);
    if (g_cuda_busy) {
        fprintf(stderr, "[WARNING] CUDA busy, cannot execute job\n");
        pthread_mutex_unlock(&g_cuda_lock);
        return;
    }
    g_cuda_busy = 1;
    memcpy(&g_current_job, entry, sizeof(g_current_job));
    g_has_current_job = 1;
    pthread_mutex_unlock(&g_cuda_lock);

    printf("[PROCESS] Pool %c: vm=%u, req=%u, prio=%u, w=%d, %d+%d\n",
           entry->pool_id, entry->vm_id, entry->request_id,
           entry->priority, entry->weight, entry->num1, entry->num2);

    uint64_t cs = wfq_context_switches(&g_scheduler);
    static uint64_t last_cs = 0;
    if (cs > last_cs) {
        for (uint64_t i = last_cs; i < cs; i++)
            metrics_record_context_switch(&g_metrics);
        last_cs = cs;
    }

    wd_job_started(&g_watchdog, entry->vm_id, entry->request_id);

    wfq_entry_t *cb_entry = (wfq_entry_t *)malloc(sizeof(wfq_entry_t));
    if (!cb_entry) {
        fprintf(stderr, "[ERROR] malloc failed for cb_entry\n");
        pthread_mutex_lock(&g_cuda_lock);
        g_cuda_busy = 0;
        g_has_current_job = 0;
        pthread_mutex_unlock(&g_cuda_lock);
        close(entry->client_fd);
        return;
    }
    memcpy(cb_entry, entry, sizeof(*cb_entry));

    if (cuda_vector_add_async(entry->num1, entry->num2,
                              cuda_result_callback, cb_entry) != 0) {
        fprintf(stderr, "[ERROR] cuda_vector_add_async failed for vm%u\n",
                entry->vm_id);
        wd_job_failed(&g_watchdog, entry->vm_id, entry->request_id);
        metrics_record_error(&g_metrics, entry->vm_id);

        pthread_mutex_lock(&g_cuda_lock);
        g_cuda_busy = 0;
        g_has_current_job = 0;
        pthread_mutex_unlock(&g_cuda_lock);

        close(entry->client_fd);
        free(cb_entry);
    }
}

static void dispatch_next_job(void)
{
    pthread_mutex_lock(&g_cuda_lock);
    if (g_cuda_busy) {
        pthread_mutex_unlock(&g_cuda_lock);
        return;
    }
    pthread_mutex_unlock(&g_cuda_lock);

    wfq_entry_t entry;
    if (wfq_dequeue(&g_scheduler, &entry) == 0) {
        execute_job(&entry);
    }
}

static void track_connection(uint32_t vm_id, int fd)
{
    pthread_mutex_lock(&g_connections_lock);
    
    int idx = -1;
    for (int i = 0; i < g_num_connections; i++) {
        if (g_connections[i].vm_id == vm_id && !g_connections[i].is_active) {
            idx = i;
            break;
        }
    }
    if (idx < 0 && g_num_connections < MAX_TRACKED_CONNECTIONS) {
        idx = g_num_connections++;
    }
    
    if (idx >= 0) {
        g_connections[idx].vm_id = vm_id;
        g_connections[idx].fd = fd;
        g_connections[idx].connect_time = time(NULL);
        g_connections[idx].messages_sent = 0;
        g_connections[idx].messages_received = 0;
        g_connections[idx].is_active = 1;
    }
    
    pthread_mutex_unlock(&g_connections_lock);
}

/* Remove connection tracking */
static void untrack_connection(int fd)
{
    pthread_mutex_lock(&g_connections_lock);
    for (int i = 0; i < g_num_connections; i++) {
        if (g_connections[i].fd == fd && g_connections[i].is_active) {
            g_connections[i].is_active = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_connections_lock);
}

static void handle_client_connection(int client_fd)
{
    uint8_t rx_buf[VGPU_SOCKET_HDR_SIZE + VGPU_SOCKET_MAX_PAYLOAD];
    ssize_t n;
    struct sockaddr_un peer_addr;
    socklen_t peer_len = sizeof(peer_addr);

    /* Get peer address for logging */
    if (getpeername(client_fd, (struct sockaddr *)&peer_addr, &peer_len) == 0) {
        printf("[CONNECTION] New connection from %s (fd=%d)\n",
               peer_addr.sun_path[0] ? peer_addr.sun_path : "<abstract>", client_fd);
    } else {
        printf("[CONNECTION] New connection (fd=%d, peer info unavailable: %s)\n",
               client_fd, strerror(errno));
    }
    fflush(stdout);

    /* Read socket header */
    n = read(client_fd, rx_buf, VGPU_SOCKET_HDR_SIZE);
    if (n < (ssize_t)VGPU_SOCKET_HDR_SIZE) {
        if (n > 0) {
            fprintf(stderr, "[ERROR] Incomplete header from fd=%d (%zd/%zu bytes): %s\n",
                    client_fd, n, VGPU_SOCKET_HDR_SIZE, strerror(errno));
        } else if (n == 0) {
            fprintf(stderr, "[ERROR] Connection closed by peer (fd=%d) before header\n",
                    client_fd);
        } else {
            fprintf(stderr, "[ERROR] Read error from fd=%d: %s (errno=%d)\n",
                    client_fd, strerror(errno), errno);
        }
        untrack_connection(client_fd);
        close(client_fd);
        return;
    }

    VGPUSocketHeader *hdr = (VGPUSocketHeader *)rx_buf;

    /* Validate magic */
    if (hdr->magic != VGPU_SOCKET_MAGIC) {
        fprintf(stderr, "[ERROR] Invalid magic: 0x%08x from fd=%d\n", hdr->magic, client_fd);
        untrack_connection(client_fd);
        close(client_fd);
        return;
    }

    track_connection(hdr->vm_id, client_fd);
    
    /* Update message count */
    pthread_mutex_lock(&g_connections_lock);
    for (int i = 0; i < g_num_connections; i++) {
        if (g_connections[i].fd == client_fd && g_connections[i].is_active) {
            g_connections[i].messages_received++;
            break;
        }
    }
    pthread_mutex_unlock(&g_connections_lock);

    /* Handle PING */
    if (hdr->msg_type == VGPU_MSG_PING) {
        VGPUSocketHeader pong;
        memset(&pong, 0, sizeof(pong));
        pong.magic    = VGPU_SOCKET_MAGIC;
        pong.msg_type = VGPU_MSG_PONG;
        pong.vm_id    = hdr->vm_id;
        ssize_t sent = write(client_fd, &pong, VGPU_SOCKET_HDR_SIZE);
        if (sent < 0) {
            fprintf(stderr, "[PING] Failed to send PONG to vm_id=%u (fd=%d): %s\n",
                    hdr->vm_id, client_fd, strerror(errno));
            untrack_connection(client_fd);
            close(client_fd);
        } else {
            /* Update sent count for PONG */
            pthread_mutex_lock(&g_connections_lock);
            for (int i = 0; i < g_num_connections; i++) {
                if (g_connections[i].fd == client_fd && g_connections[i].is_active) {
                    g_connections[i].messages_sent++;
                    break;
                }
            }
            pthread_mutex_unlock(&g_connections_lock);
        }
        close(client_fd);
        return;
    }

    if (hdr->msg_type != VGPU_MSG_REQUEST &&
        hdr->msg_type != VGPU_MSG_CUDA_CALL) {
        untrack_connection(client_fd);
        close(client_fd);
        return;
    }

    uint8_t *payload_buf = rx_buf + VGPU_SOCKET_HDR_SIZE;
    uint8_t *alloc_buf = NULL;
    uint32_t total_payload = hdr->payload_len;

    if (total_payload > 0) {
        if (total_payload > VGPU_SOCKET_MAX_PAYLOAD) {
            alloc_buf = (uint8_t *)malloc(total_payload);
            if (!alloc_buf) {
                fprintf(stderr, "[ERROR] malloc failed for %u byte payload\n",
                        total_payload);
                untrack_connection(client_fd);
                close(client_fd);
                return;
            }
            payload_buf = alloc_buf;
        }

        uint32_t total_read = 0;
        while (total_read < total_payload) {
            n = read(client_fd, payload_buf + total_read,
                     total_payload - total_read);
            if (n <= 0) {
                fprintf(stderr, "[ERROR] Incomplete payload (%u/%u)\n",
                        total_read, total_payload);
                if (alloc_buf) free(alloc_buf);
                untrack_connection(client_fd);
                close(client_fd);
                return;
            }
            total_read += (uint32_t)n;
        }
    }

    /* Handle CUDA API call */
    if (hdr->msg_type == VGPU_MSG_CUDA_CALL) {
        handle_cuda_call(client_fd, hdr, payload_buf, total_payload);
        if (alloc_buf) free(alloc_buf);
        return;
    }

    /* Parse request (legacy vector-add path) */
    int num1, num2;
    if (parse_vgpu_request(payload_buf, hdr->payload_len, &num1, &num2) != 0) {
        fprintf(stderr, "[ERROR] Failed to parse request from vm%u\n",
                hdr->vm_id);
        if (alloc_buf) free(alloc_buf);
        untrack_connection(client_fd);
        close(client_fd);
        return;
    }

    /* Validate pool and priority */
    if (hdr->pool_id != 'A' && hdr->pool_id != 'B') {
        fprintf(stderr, "[ERROR] Invalid pool_id: '%c'\n", hdr->pool_id);
        untrack_connection(client_fd);
        close(client_fd);
        return;
    }
    if (hdr->priority > VGPU_PRIORITY_HIGH) {
        fprintf(stderr, "[ERROR] Invalid priority: %u\n", hdr->priority);
        untrack_connection(client_fd);
        close(client_fd);
        return;
    }


    /* 1. Check quarantine */
    if (wd_is_quarantined(&g_watchdog, hdr->vm_id)) {
        printf("[REJECT] vm=%u quarantined\n", hdr->vm_id);
        metrics_record_rejection(&g_metrics, hdr->vm_id);
        send_rejection(client_fd, hdr->vm_id, hdr->request_id,
                       hdr->pool_id, hdr->priority, VGPU_MSG_QUARANTINED);
        return;
    }

    /* 2. Check rate limiter */
    int vm_qd = wfq_vm_queue_depth(&g_scheduler, hdr->vm_id);
    int rl_result = rl_check(&g_rate_limiter, hdr->vm_id, vm_qd);
    if (rl_result != RL_ALLOW) {
        printf("[REJECT] vm=%u rate-limited (code=%d)\n",
               hdr->vm_id, rl_result);
        metrics_record_rejection(&g_metrics, hdr->vm_id);
        send_rejection(client_fd, hdr->vm_id, hdr->request_id,
                       hdr->pool_id, hdr->priority, VGPU_MSG_BUSY);
        return;
    }

    /* 3. Look up VM config from DB (for weight) */
    int weight = 50;  /* default */
    if (g_db) {
        vgpu_vm_config_t vm_cfg;
        if (vgpu_get_vm_config_by_id(g_db, (int)hdr->vm_id, &vm_cfg) == VGPU_OK) {
            weight = (vm_cfg.weight > 0) ? vm_cfg.weight : 50;
        }
    }

    /* 4. Enqueue into WFQ scheduler */
    wfq_entry_t entry;
    memset(&entry, 0, sizeof(entry));
    entry.vm_id      = hdr->vm_id;
    entry.request_id = hdr->request_id;
    entry.pool_id    = hdr->pool_id;
    entry.priority   = hdr->priority;
    entry.weight     = weight;
    entry.num1       = num1;
    entry.num2       = num2;
    entry.client_fd  = client_fd;  /* Keep open until response */

    if (hdr->payload_len > 0 && hdr->payload_len <= sizeof(entry.payload)) {
        memcpy(entry.payload, payload_buf, hdr->payload_len);
        entry.payload_len = hdr->payload_len;
    }

    /* Free allocated payload buffer if any */
    if (alloc_buf) free(alloc_buf);

    if (wfq_enqueue(&g_scheduler, &entry) != 0) {
        fprintf(stderr, "[ERROR] WFQ queue full for vm%u\n", hdr->vm_id);
        metrics_record_rejection(&g_metrics, hdr->vm_id);
        send_rejection(client_fd, hdr->vm_id, hdr->request_id,
                       hdr->pool_id, hdr->priority, VGPU_MSG_BUSY);
        return;
    }

    dispatch_next_job();
}

static void handle_cuda_call(int client_fd, VGPUSocketHeader *sock_hdr,
                              const uint8_t *payload, uint32_t payload_len)
{
    if (!g_cuda_executor) {
        fprintf(stderr, "[ERROR] CUDA executor not initialized\n");
        untrack_connection(client_fd);
        close(client_fd);
        return;
    }

    if (payload_len < sizeof(CUDACallHeader)) {
        fprintf(stderr, "[ERROR] CUDA call payload too short (%u < %zu)\n",
                payload_len, sizeof(CUDACallHeader));
        untrack_connection(client_fd);
        close(client_fd);
        return;
    }


    /* Check quarantine */
    if (wd_is_quarantined(&g_watchdog, sock_hdr->vm_id)) {
        printf("[REJECT] vm=%u quarantined (CUDA call)\n", sock_hdr->vm_id);
        metrics_record_rejection(&g_metrics, sock_hdr->vm_id);
        send_rejection(client_fd, sock_hdr->vm_id, sock_hdr->request_id,
                       sock_hdr->pool_id, sock_hdr->priority,
                       VGPU_MSG_QUARANTINED);
        return;
    }

    /* Check rate limiter */
    int vm_qd = wfq_vm_queue_depth(&g_scheduler, sock_hdr->vm_id);
    int rl_result = rl_check(&g_rate_limiter, sock_hdr->vm_id, vm_qd);
    if (rl_result != RL_ALLOW) {
        printf("[REJECT] vm=%u rate-limited (CUDA call, code=%d)\n",
               sock_hdr->vm_id, rl_result);
        metrics_record_rejection(&g_metrics, sock_hdr->vm_id);
        send_rejection(client_fd, sock_hdr->vm_id, sock_hdr->request_id,
                       sock_hdr->pool_id, sock_hdr->priority, VGPU_MSG_BUSY);
        return;
    }

    /* Parse the CUDA call header */
    const CUDACallHeader *cuda_hdr = (const CUDACallHeader *)payload;
    const void *bulk_data = NULL;
    uint32_t bulk_len = 0;

    if (payload_len > sizeof(CUDACallHeader)) {
        bulk_data = payload + sizeof(CUDACallHeader);
        bulk_len = payload_len - sizeof(CUDACallHeader);
    }

    /* Execute the CUDA call */
    CUDACallResult result;
    uint8_t *result_data = NULL;
    uint32_t result_cap = 0;
    uint32_t result_len = 0;

    if (cuda_hdr->call_id == CUDA_CALL_MEMCPY_DTOH ||
        cuda_hdr->call_id == CUDA_CALL_MEMCPY_DTOH_ASYNC ||
        cuda_hdr->call_id == CUDA_CALL_GET_GPU_INFO) {
        result_cap = 8 * 1024 * 1024;  /* 8 MB max */
        result_data = (uint8_t *)malloc(result_cap);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int rc = cuda_executor_call(g_cuda_executor,
                                 cuda_hdr, bulk_data, bulk_len,
                                 &result, result_data, result_cap,
                                 &result_len);

    clock_gettime(CLOCK_MONOTONIC, &end);
    uint64_t elapsed_us = (uint64_t)(end.tv_sec - start.tv_sec) * 1000000
                        + (uint64_t)(end.tv_nsec - start.tv_nsec) / 1000;

    /* Record metrics */
    metrics_record_job(&g_metrics, sock_hdr->vm_id, elapsed_us, elapsed_us);

    /* Send result back to vgpu-stub */
    VGPUSocketHeader resp_hdr;
    memset(&resp_hdr, 0, sizeof(resp_hdr));
    resp_hdr.magic       = VGPU_SOCKET_MAGIC;
    resp_hdr.msg_type    = VGPU_MSG_CUDA_RESULT;
    resp_hdr.vm_id       = sock_hdr->vm_id;
    resp_hdr.request_id  = sock_hdr->request_id;
    resp_hdr.pool_id     = sock_hdr->pool_id;
    resp_hdr.priority    = sock_hdr->priority;
    resp_hdr.payload_len = (uint32_t)(sizeof(CUDACallResult) + result_len);

    struct iovec iov[3];
    struct msghdr msg;
    int iov_cnt = 2;

    iov[0].iov_base = &resp_hdr;
    iov[0].iov_len  = VGPU_SOCKET_HDR_SIZE;
    iov[1].iov_base = &result;
    iov[1].iov_len  = sizeof(CUDACallResult);

    if (result_len > 0 && result_data) {
        iov[2].iov_base = result_data;
        iov[2].iov_len  = result_len;
        iov_cnt = 3;
    }

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov    = iov;
    msg.msg_iovlen = iov_cnt;

    ssize_t sent = sendmsg(client_fd, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
        fprintf(stderr, "[ERROR] Failed to send CUDA result: %s\n",
                strerror(errno));
    } else {
        /* Update sent message count */
        pthread_mutex_lock(&g_connections_lock);
        for (int i = 0; i < g_num_connections; i++) {
            if (g_connections[i].fd == client_fd && g_connections[i].is_active) {
                g_connections[i].messages_sent++;
                break;
            }
        }
        pthread_mutex_unlock(&g_connections_lock);
    }

    if (result_data) free(result_data);

    /* Update stats */
    g_total_processed++;
    if (sock_hdr->pool_id == 'A') g_pool_a_processed++;
    else                           g_pool_b_processed++;

    (void)rc;
}

static void handle_admin_connection(int client_fd)
{
    VGPUAdminRequest req;
    ssize_t n = read(client_fd, &req, sizeof(req));
    if (n < (ssize_t)sizeof(req)) {
        close(client_fd);
        return;
    }

    if (req.magic != VGPU_SOCKET_MAGIC) {
        close(client_fd);
        return;
    }

    char *buf = (char *)malloc(ADMIN_BUF_SIZE);
    if (!buf) { close(client_fd); return; }

    int data_len = 0;
    VGPUAdminResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.magic  = VGPU_SOCKET_MAGIC;
    resp.status = 0;

    switch (req.command) {
    case VGPU_ADMIN_SHOW_METRICS:
        if (req.param1 == 0) {
            /* Full summary */
            data_len = metrics_export_summary(&g_metrics, buf, ADMIN_BUF_SIZE);
        } else {
            /* Prometheus format */
            data_len = metrics_export_prometheus(&g_metrics, buf, ADMIN_BUF_SIZE);
        }
        break;

    case VGPU_ADMIN_SHOW_HEALTH: {
        nvml_health_t health;
        nvml_poll(&health);
        if (health.available) {
            data_len = snprintf(buf, ADMIN_BUF_SIZE,
                "=== GPU Health ===\n"
                "Temperature:   %u °C\n"
                "GPU Util:      %u%%\n"
                "Memory Util:   %u%%\n"
                "Memory Used:   %lu / %lu MB\n"
                "Power Draw:    %u W\n"
                "ECC Errors:    %lu\n"
                "Needs Reset:   %s\n"
                "\n"
                "=== Watchdog ===\n"
                "GPU Resets:    %lu\n"
                "Scheduler Queue: %d\n"
                "Context Switches: %lu\n",
                health.temperature_c,
                health.gpu_utilization,
                health.memory_utilization,
                (unsigned long)health.memory_used_mb,
                (unsigned long)health.memory_total_mb,
                health.power_watts,
                (unsigned long)health.ecc_errors,
                health.needs_reset ? "YES" : "no",
                (unsigned long)wd_total_resets(&g_watchdog),
                wfq_queue_len(&g_scheduler),
                (unsigned long)wfq_context_switches(&g_scheduler));
        } else {
            data_len = snprintf(buf, ADMIN_BUF_SIZE,
                "=== GPU Health ===\n"
                "NVML: not available (GPU monitoring disabled)\n"
                "\n"
                "=== Watchdog ===\n"
                "GPU Resets:    %lu\n"
                "Scheduler Queue: %d\n"
                "Context Switches: %lu\n",
                (unsigned long)wd_total_resets(&g_watchdog),
                wfq_queue_len(&g_scheduler),
                (unsigned long)wfq_context_switches(&g_scheduler));
        }
        break;
    }

    case VGPU_ADMIN_RELOAD_CONFIG:
        if (g_db) {
            vgpu_vm_config_t configs[64];
            int count = 0;
            if (vgpu_list_vms(g_db, 0, -1, configs, &count, 64) == VGPU_OK) {
                for (int i = 0; i < count; i++) {
                    rl_configure_vm(&g_rate_limiter,
                                    (uint32_t)configs[i].vm_id,
                                    configs[i].max_jobs_per_sec,
                                    configs[i].max_queue_depth);

                    if (configs[i].quarantined) {
                        /* Sync quarantine state from DB */
                        /* (watchdog tracks its own state, but DB is source of truth
                           for admin-set quarantines) */
                    }
                }
                data_len = snprintf(buf, ADMIN_BUF_SIZE,
                    "Config reloaded: %d VM(s) updated\n", count);
            } else {
                data_len = snprintf(buf, ADMIN_BUF_SIZE,
                    "ERROR: Failed to read VM configs from DB\n");
                resp.status = 1;
            }
        } else {
            data_len = snprintf(buf, ADMIN_BUF_SIZE,
                "ERROR: Database not connected\n");
            resp.status = 1;
        }
        break;

    case VGPU_ADMIN_SHOW_CONNECTIONS: {
        pthread_mutex_lock(&g_connections_lock);
        data_len = snprintf(buf, ADMIN_BUF_SIZE,
            "=== Active VM Connections ===\n"
            "Total tracked: %d\n"
            "Active: ", g_num_connections);
        
        int active_count = 0;
        for (int i = 0; i < g_num_connections; i++) {
            if (g_connections[i].is_active) {
                active_count++;
            }
        }
        data_len += snprintf(buf + data_len, ADMIN_BUF_SIZE - data_len,
            "%d\n\n", active_count);
        
        if (active_count > 0) {
            data_len += snprintf(buf + data_len, ADMIN_BUF_SIZE - data_len,
                "VM ID | FD  | Connected Since    | Messages (Rx/Tx)\n"
                "------+-----+--------------------+------------------\n");
            
            for (int i = 0; i < g_num_connections; i++) {
                if (g_connections[i].is_active) {
                    time_t now = time(NULL);
                    time_t elapsed = now - g_connections[i].connect_time;
                    int hours = elapsed / 3600;
                    int mins = (elapsed % 3600) / 60;
                    int secs = elapsed % 60;
                    
                    data_len += snprintf(buf + data_len, ADMIN_BUF_SIZE - data_len,
                        "  %3u | %3d | %02d:%02d:%02d ago      | %10lu / %10lu\n",
                        g_connections[i].vm_id,
                        g_connections[i].fd,
                        hours, mins, secs,
                        (unsigned long)g_connections[i].messages_received,
                        (unsigned long)g_connections[i].messages_sent);
                }
            }
        } else {
            data_len += snprintf(buf + data_len, ADMIN_BUF_SIZE - data_len,
                "No active connections.\n");
        }
        
        data_len += snprintf(buf + data_len, ADMIN_BUF_SIZE - data_len,
            "\n=== Server Sockets ===\n"
            "Active servers: %d\n", g_num_servers);
        
        for (int i = 0; i < g_num_servers; i++) {
            data_len += snprintf(buf + data_len, ADMIN_BUF_SIZE - data_len,
                "  [%d] %s (fd=%d)\n", i + 1, g_socket_paths[i], g_server_fds[i]);
        }
        
        pthread_mutex_unlock(&g_connections_lock);
        break;
    }

    default:
        data_len = snprintf(buf, ADMIN_BUF_SIZE,
            "ERROR: Unknown admin command 0x%02x\n", req.command);
        resp.status = 1;
        break;
    }

    resp.data_len = (uint32_t)data_len;

    /* Send response header + text */
    write(client_fd, &resp, sizeof(resp));
    if (data_len > 0) {
        write(client_fd, buf, data_len);
    }

    free(buf);
    close(client_fd);
}

static void load_vm_configs(void)
{
    if (!g_db) return;

    vgpu_vm_config_t configs[64];
    int count = 0;
    if (vgpu_list_vms(g_db, 0, -1, configs, &count, 64) != VGPU_OK) {
        fprintf(stderr, "[CONFIG] WARNING: Could not read VM configs from DB\n");
        return;
    }

    printf("[CONFIG] Loaded %d VM(s) from database\n", count);
    for (int i = 0; i < count; i++) {
        rl_configure_vm(&g_rate_limiter,
                        (uint32_t)configs[i].vm_id,
                        configs[i].max_jobs_per_sec,
                        configs[i].max_queue_depth);

        printf("[CONFIG]   vm_id=%d uuid=%s weight=%d rate=%d/s queue=%d %s\n",
               configs[i].vm_id, configs[i].vm_uuid,
               configs[i].weight, configs[i].max_jobs_per_sec,
               configs[i].max_queue_depth,
               configs[i].quarantined ? "(quarantined)" : "");
    }
}

static void print_stats(void)
{
    printf("\n[MEDIATOR STATS]\n");
    printf("  Total processed:  %lu\n", (unsigned long)g_total_processed);
    printf("  Pool A processed: %lu\n", (unsigned long)g_pool_a_processed);
    printf("  Pool B processed: %lu\n", (unsigned long)g_pool_b_processed);
    printf("  WFQ queue depth:  %d\n",  wfq_queue_len(&g_scheduler));
    printf("  Context switches: %lu\n",
           (unsigned long)wfq_context_switches(&g_scheduler));
    printf("  CUDA busy:        %s\n",  g_cuda_busy ? "yes" : "no");

    /* Quick NVML summary if available */
    nvml_health_t health;
    nvml_poll(&health);
    if (health.available) {
        printf("  GPU temp:         %u°C\n", health.temperature_c);
        printf("  GPU util:         %u%%\n",  health.gpu_utilization);
    }
    printf("\n");
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --socket-path <path>   Override socket path (host filesystem)\n");
    printf("  --no-db                Run without database\n");
    printf("  --no-nvml              Disable NVML GPU health monitoring\n");
    printf("  --help                 Show this help\n");
    printf("\nIf --socket-path is not given, the mediator auto-discovers\n");
    printf("the QEMU chroot directory and creates the socket inside it.\n");
    printf("Fallback: %s\n", VGPU_SOCKET_PATH);
}

#define REDISCOVERY_INTERVAL_SEC  10

static void rediscover_and_setup_sockets(void)
{
    char *chroots[MAX_SERVER_SOCKETS];
    /* verbose=0: suppress per-scan noise; only log truly new sockets below */
    int num_chroots = discover_all_qemu_chroots(chroots, MAX_SERVER_SOCKETS, 0);

    for (int i = 0; i < num_chroots; i++) {
        char candidate[512];
        snprintf(candidate, sizeof(candidate), "%s%s", chroots[i], VGPU_SOCKET_PATH);

        int already_have = 0;
        for (int j = 0; j < g_num_servers; j++) {
            if (strcmp(g_socket_paths[j], candidate) == 0) {
                already_have = 1;
                break;
            }
        }

        if (!already_have && g_num_servers < MAX_SERVER_SOCKETS) {
            /* Ensure /tmp exists inside the chroot */
            char tmp_dir[512];
            snprintf(tmp_dir, sizeof(tmp_dir), "%s/tmp", chroots[i]);
            mkdir(tmp_dir, 0755);

            snprintf(g_socket_paths[g_num_servers],
                     sizeof(g_socket_paths[g_num_servers]),
                     "%s", candidate);

            g_server_fds[g_num_servers] =
                setup_socket_server(g_socket_paths[g_num_servers]);

            if (g_server_fds[g_num_servers] >= 0) {
                printf("[REDISCOVERY] New VM socket: %s (fd=%d)\n",
                       g_socket_paths[g_num_servers],
                       g_server_fds[g_num_servers]);
                g_num_servers++;
            } else {
                fprintf(stderr, "[REDISCOVERY] Failed to setup socket %s: %s\n",
                        candidate, strerror(errno));
                g_socket_paths[g_num_servers][0] = '\0';
            }
        }

        free(chroots[i]);
    }
}

static int handle_persistent_message(int client_fd)
{
    uint8_t hdr_buf[VGPU_SOCKET_HDR_SIZE];
    ssize_t n;

    /* Read the socket header */
    n = read(client_fd, hdr_buf, VGPU_SOCKET_HDR_SIZE);
    if (n == 0) {
        printf("[PERSIST] fd=%d: connection closed by peer\n", client_fd);
        return 0;
    }
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
            return 1;  /* spurious wake-up, nothing to read */
        fprintf(stderr, "[PERSIST] fd=%d: read error: %s\n",
                client_fd, strerror(errno));
        return 0;
    }
    if (n < (ssize_t)VGPU_SOCKET_HDR_SIZE) {
        fprintf(stderr, "[PERSIST] fd=%d: incomplete header (%zd/%zu bytes)\n",
                client_fd, n, VGPU_SOCKET_HDR_SIZE);
        return 0;
    }

    VGPUSocketHeader *hdr = (VGPUSocketHeader *)hdr_buf;

    if (hdr->magic != VGPU_SOCKET_MAGIC) {
        fprintf(stderr, "[PERSIST] fd=%d: bad magic 0x%08x — closing\n",
                client_fd, hdr->magic);
        return 0;
    }

    if (hdr->msg_type != VGPU_MSG_CUDA_CALL) {
        fprintf(stderr, "[PERSIST] fd=%d: unexpected msg_type=0x%x — closing\n",
                client_fd, hdr->msg_type);
        return 0;
    }

    /* Update per-connection receive counter */
    pthread_mutex_lock(&g_connections_lock);
    for (int i = 0; i < g_num_connections; i++) {
        if (g_connections[i].fd == client_fd && g_connections[i].is_active) {
            g_connections[i].messages_received++;
            break;
        }
    }
    pthread_mutex_unlock(&g_connections_lock);

    uint32_t total_payload = hdr->payload_len;
    uint8_t  inline_buf[VGPU_SOCKET_HDR_SIZE + VGPU_SOCKET_MAX_PAYLOAD];
    uint8_t *payload_buf = NULL;
    uint8_t *alloc_buf   = NULL;

    if (total_payload > 0) {
        if (total_payload <= VGPU_SOCKET_MAX_PAYLOAD) {
            payload_buf = inline_buf;
        } else {
            alloc_buf = (uint8_t *)malloc(total_payload);
            if (!alloc_buf) {
                fprintf(stderr, "[PERSIST] fd=%d: malloc(%u) failed\n",
                        client_fd, total_payload);
                return 0;
            }
            payload_buf = alloc_buf;
        }

        uint32_t total_read = 0;
        while (total_read < total_payload) {
            n = read(client_fd, payload_buf + total_read,
                     total_payload - total_read);
            if (n <= 0) {
                fprintf(stderr, "[PERSIST] fd=%d: incomplete payload "
                        "(%u/%u bytes)\n", client_fd, total_read, total_payload);
                if (alloc_buf) free(alloc_buf);
                return 0;
            }
            total_read += (uint32_t)n;
        }
    }

    handle_cuda_call(client_fd, hdr, payload_buf, total_payload);

    if (alloc_buf) free(alloc_buf);

    int keep = (fcntl(client_fd, F_GETFD) != -1);
    return keep;
}

static void run_mediator(void)
{
    time_t last_stats     = time(NULL);
    time_t last_discovery = time(NULL);
    time_t last_heartbeat = time(NULL);
    fd_set read_fds;
    int max_fd;
    struct timeval timeout;

#define MAX_PERSISTENT_CLIENTS 64
    int persistent_fds[MAX_PERSISTENT_CLIENTS];
    for (int i = 0; i < MAX_PERSISTENT_CLIENTS; i++) persistent_fds[i] = -1;

    printf("[MEDIATOR] Starting main loop...\n");
    printf("[MEDIATOR] Listening on %d socket(s) for VM connections\n",
           g_num_servers);
    if (g_admin_fd >= 0)
        printf("[MEDIATOR] Admin socket active on %s\n",
               VGPU_ADMIN_SOCKET_PATH);
        for (int i = 0; i < g_num_servers; i++) {
        printf("[MEDIATOR] Server socket[%d]: %s  fd=%d\n",
               i, g_socket_paths[i], g_server_fds[i]);
    }
    fflush(stdout);

    while (!g_shutdown) {
        FD_ZERO(&read_fds);
        max_fd = -1;

        /* Add all QEMU chroot sockets */
        for (int i = 0; i < g_num_servers; i++) {
            if (g_server_fds[i] < 0) continue;
            FD_SET(g_server_fds[i], &read_fds);
            if (g_server_fds[i] > max_fd) max_fd = g_server_fds[i];
        }

        for (int i = 0; i < MAX_PERSISTENT_CLIENTS; i++) {
            if (persistent_fds[i] >= 0) {
                FD_SET(persistent_fds[i], &read_fds);
                if (persistent_fds[i] > max_fd) max_fd = persistent_fds[i];
            }
        }

        /* Add admin socket */
        if (g_admin_fd >= 0) {
            FD_SET(g_admin_fd, &read_fds);
            if (g_admin_fd > max_fd) max_fd = g_admin_fd;
        }

        timeout.tv_sec  = 1;
        timeout.tv_usec = 0;

        int ret = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);
        if (ret < 0) {
            if (errno == EINTR) continue;
            perror("select");
            break;
        }

        for (int i = 0; i < g_num_servers; i++) {
            if (g_server_fds[i] < 0) continue;
            if (FD_ISSET(g_server_fds[i], &read_fds)) {
                struct sockaddr_un client_addr;
                socklen_t client_len = sizeof(client_addr);
                int client_fd = accept(g_server_fds[i],
                                       (struct sockaddr *)&client_addr,
                                       &client_len);
                if (client_fd >= 0) {
                    printf("[SOCKET] New connection on %s (fd=%d, server_idx=%d)\n",
                           g_socket_paths[i], client_fd, i);
                    fflush(stdout);
                    handle_client_connection(client_fd);

                    /* handle_client_connection() closes client_fd for most
                     * message types (PING, errors, legacy requests).  For CUDA
                     * calls it intentionally leaves the fd open so the vgpu-stub
                     * can reuse the connection.  Detect the open case with fcntl
                     * and register the fd for persistent polling. */
                    if (fcntl(client_fd, F_GETFD) != -1) {
                        int slot = -1;
                        for (int j = 0; j < MAX_PERSISTENT_CLIENTS; j++) {
                            if (persistent_fds[j] == -1) { slot = j; break; }
                        }
                        if (slot >= 0) {
                            persistent_fds[slot] = client_fd;
                            printf("[PERSIST] fd=%d registered for persistent "
                                   "polling (slot=%d)\n", client_fd, slot);
                            fflush(stdout);
                        } else {
                            fprintf(stderr, "[PERSIST] too many persistent "
                                    "clients, closing fd=%d\n", client_fd);
                            untrack_connection(client_fd);
                            close(client_fd);
                        }
                    }
                } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    fprintf(stderr, "[SOCKET] accept() failed on %s: %s (errno=%d)\n",
                            g_socket_paths[i], strerror(errno), errno);
                }
            }
        }

        /* Accept connections on admin socket */
        if (g_admin_fd >= 0 && FD_ISSET(g_admin_fd, &read_fds)) {
            struct sockaddr_un client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(g_admin_fd,
                                   (struct sockaddr *)&client_addr,
                                   &client_len);
            if (client_fd >= 0) {
                handle_admin_connection(client_fd);
            }
        }

        for (int i = 0; i < MAX_PERSISTENT_CLIENTS; i++) {
            if (persistent_fds[i] < 0) continue;
            if (!FD_ISSET(persistent_fds[i], &read_fds)) continue;

            int pfd  = persistent_fds[i];
            int keep = handle_persistent_message(pfd);
            if (!keep) {
                printf("[PERSIST] fd=%d removed from persistent set\n", pfd);
                fflush(stdout);
                untrack_connection(pfd);
                close(pfd);
                persistent_fds[i] = -1;
            }
        }

        dispatch_next_job();

        time_t now = time(NULL);

        /* Periodic stats */
        if (now - last_stats >= 60) {
            print_stats();
            last_stats = now;
        }

        if (now - last_heartbeat >= 10) {
            printf("[HEARTBEAT] alive — %d server socket(s), admin_fd=%d\n",
                   g_num_servers, g_admin_fd);
            for (int i = 0; i < g_num_servers; i++) {
                printf("[HEARTBEAT]   [%d] fd=%d  %s\n",
                       i, g_server_fds[i], g_socket_paths[i]);
            }
            fflush(stdout);
            last_heartbeat = now;
        }

        if (now - last_discovery >= REDISCOVERY_INTERVAL_SEC) {
            rediscover_and_setup_sockets();
            last_discovery = now;
        }
    }

    /* ---- Shutdown ---- */
    printf("[MEDIATOR] Shutting down...\n");

    /* Close all persistent client connections */
    for (int i = 0; i < MAX_PERSISTENT_CLIENTS; i++) {
        if (persistent_fds[i] >= 0) {
            untrack_connection(persistent_fds[i]);
            close(persistent_fds[i]);
            persistent_fds[i] = -1;
        }
    }

    /* Stop watchdog thread */
    wd_stop(&g_watchdog);

    /* Wait for CUDA to finish */
    if (g_cuda_busy) {
        printf("[MEDIATOR] Waiting for CUDA to finish...\n");
        cuda_sync();
    }

    wfq_entry_t entry;
    while (wfq_dequeue(&g_scheduler, &entry) == 0) {
        if (entry.client_fd >= 0) close(entry.client_fd);
    }

    /* Close server sockets */
    for (int i = 0; i < g_num_servers; i++) {
        close(g_server_fds[i]);
        if (g_socket_paths[i][0]) unlink(g_socket_paths[i]);
    }

    /* Close admin socket */
    if (g_admin_fd >= 0) {
        close(g_admin_fd);
        unlink(VGPU_ADMIN_SOCKET_PATH);
    }
}

int main(int argc, char *argv[])
{
    const char *override_path = NULL;
    int no_db   = 0;
    int no_nvml = 0;

    /* Force unbuffered I/O so log messages are visible immediately even when
     * stdout/stderr are pipes (e.g. under sudo or nohup redirection).
     * Without this, libc switches to 8 KB block-buffering when the fd is not
     * a TTY, and messages like "[SOCKET] New connection on..." never appear
     * until the buffer fills up. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("===========================================================\n");
    printf("  MEDIATOR DAEMON v3.1 — Phase 3+: CUDA API Remoting\n");
    printf("  Communication: MMIO PCI BAR0/BAR1 + Unix domain socket\n");
    printf("===========================================================\n\n");

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--socket-path") == 0 && i + 1 < argc) {
            override_path = argv[++i];
        } else if (strcmp(argv[i], "--no-db") == 0) {
            no_db = 1;
        } else if (strcmp(argv[i], "--no-nvml") == 0) {
            no_nvml = 1;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* ---- Initialize Phase 3 subsystems ---- */

    /* 1. WFQ scheduler */
    wfq_init(&g_scheduler);
    printf("[INIT] WFQ scheduler ready\n");

    /* 2. Rate limiter */
    rl_init(&g_rate_limiter);
    printf("[INIT] Rate limiter ready\n");

    /* 3. Metrics */
    metrics_init(&g_metrics);
    printf("[INIT] Metrics collector ready\n");

    /* 4. Watchdog (start thread) */
    wd_init(&g_watchdog);
    if (wd_start(&g_watchdog) != 0) {
        fprintf(stderr, "[ERROR] Failed to start watchdog thread\n");
        return 1;
    }
    printf("[INIT] Watchdog started (timeout=%ds, threshold=%d)\n",
           g_watchdog.job_timeout_sec, g_watchdog.fault_threshold);

    /* 5. NVML (optional) */
    if (!no_nvml) {
        if (nvml_init() == 0) {
            printf("[INIT] NVML GPU health monitoring active\n");
        } else {
            printf("[INIT] NVML not available — running without GPU health monitoring\n");
        }
    }

    /* 6. Database (optional) */
    if (!no_db) {
        if (vgpu_db_init(&g_db) == VGPU_OK) {
            vgpu_db_init_schema(g_db);
            printf("[INIT] Database connected (%s)\n", VGPU_DB_PATH);
            load_vm_configs();
        } else {
            fprintf(stderr, "[WARN] Could not open database — "
                    "running without persistent config\n");
            g_db = NULL;
        }
    }

    /* 7. CUDA (legacy vector-add) */
    if (cuda_init() != 0) {
        fprintf(stderr, "[ERROR] Failed to initialize CUDA\n");
        return 1;
    }
    printf("[INIT] CUDA ready\n");

    /* 8. CUDA Executor (Phase 3+: API remoting) */
    if (cuda_executor_init(&g_cuda_executor) == 0) {
        printf("[INIT] CUDA API executor ready (remoting enabled)\n");
    } else {
        fprintf(stderr, "[WARN] CUDA executor init failed — "
                "CUDA remoting disabled, legacy mode only\n");
        g_cuda_executor = NULL;
    }

    /* ---- Setup sockets ---- */
    g_num_servers = 0;

    if (override_path) {
        snprintf(g_socket_paths[0], sizeof(g_socket_paths[0]),
                 "%s", override_path);
        g_server_fds[0] = setup_socket_server(g_socket_paths[0]);
        if (g_server_fds[0] < 0) {
            fprintf(stderr, "[ERROR] Failed to setup socket at %s\n",
                    override_path);
            cuda_cleanup();
            return 1;
        }
        g_num_servers = 1;
        printf("[CONFIG] Using user-specified socket: %s\n",
               g_socket_paths[0]);
    } else {
        char *chroots[MAX_SERVER_SOCKETS];
        int num_chroots = discover_all_qemu_chroots(chroots,
                                                     MAX_SERVER_SOCKETS, 1);

        if (num_chroots > 0) {
            printf("[CONFIG] Found %d QEMU VM(s) with vgpu-cuda device:\n",
                   num_chroots);
            for (int i = 0; i < num_chroots; i++) {
                char tmp_dir[512];
                snprintf(tmp_dir, sizeof(tmp_dir), "%s/tmp", chroots[i]);
                mkdir(tmp_dir, 0755);

                snprintf(g_socket_paths[g_num_servers],
                         sizeof(g_socket_paths[g_num_servers]),
                         "%s%s", chroots[i], VGPU_SOCKET_PATH);

                g_server_fds[g_num_servers] =
                    setup_socket_server(g_socket_paths[g_num_servers]);
                if (g_server_fds[g_num_servers] >= 0) {
                    printf("  [%d] %s -> %s (socket created successfully)\n",
                           g_num_servers + 1,
                           chroots[i],
                           g_socket_paths[g_num_servers]);
                    g_num_servers++;
                } else {
                    fprintf(stderr, "[WARN] Failed to setup socket in %s -> %s: %s\n",
                            chroots[i], g_socket_paths[g_num_servers],
                            strerror(errno));
                }
                free(chroots[i]);
            }
        } else {
            snprintf(g_socket_paths[0], sizeof(g_socket_paths[0]),
                     "%s", VGPU_SOCKET_PATH);
            g_server_fds[0] = setup_socket_server(g_socket_paths[0]);
            if (g_server_fds[0] >= 0) g_num_servers = 1;
            printf("[CONFIG] No QEMU chroot found, using fallback: %s\n",
                   g_socket_paths[0]);
        }
    }

    if (g_num_servers == 0) {
        fprintf(stderr, "[ERROR] No server sockets created\n");
        cuda_cleanup();
        return 1;
    }

    /* Admin socket */
    g_admin_fd = setup_admin_socket();

    /* ---- Run main loop ---- */
    run_mediator();

    /* ---- Cleanup ---- */
    nvml_shutdown();
    wd_destroy(&g_watchdog);
    wfq_destroy(&g_scheduler);
    rl_destroy(&g_rate_limiter);
    metrics_destroy(&g_metrics);
    if (g_cuda_executor) {
        cuda_executor_destroy(g_cuda_executor);
        g_cuda_executor = NULL;
    }
    cuda_cleanup();
    if (g_db) vgpu_db_close(g_db);

    printf("[MEDIATOR] Exited cleanly\n");
    return 0;
}

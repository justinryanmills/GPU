#ifndef _GNU_SOURCE
#define _GNU_SOURCE  /* Required for RTLD_DEFAULT */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <pthread.h>

#include "cuda_transport.h"
#include "cuda_protocol.h"

#ifndef CUDA_CHUNK_FLAG_MIDDLE
#define CUDA_CHUNK_FLAG_MIDDLE 0x04
#endif

/* Error codes for debug reporting */
#ifndef VGPU_ERR_INVALID_REQUEST
#define VGPU_ERR_INVALID_REQUEST     0x01
#define VGPU_ERR_REQUEST_TOO_LARGE   0x02
#define VGPU_ERR_CUDA_ERROR          0x05
#define VGPU_ERR_INVALID_POOL        0x06
#define VGPU_ERR_UNSUPPORTED_OP      0x08
#define VGPU_ERR_INVALID_LENGTH      0x09
#endif

#ifndef VGPU_ERR_MEDIATOR_UNAVAIL
#define VGPU_ERR_MEDIATOR_UNAVAIL  0x03
#define VGPU_ERR_TIMEOUT           0x04
#define VGPU_ERR_QUEUE_FULL        0x07
#define VGPU_ERR_RATE_LIMITED      0x0A
#define VGPU_ERR_VM_QUARANTINED    0x0B
#endif

/* CUDA_ERROR_UNKNOWN — avoid using 2 here; it collides with CUDA_ERROR_OUT_OF_MEMORY. */
#ifndef CUDA_TRANSPORT_FALLBACK_CURESULT
#define CUDA_TRANSPORT_FALLBACK_CURESULT 999
#endif

/*
 * Debug section — accurate error tracking with call history and context.
 * Writes /tmp/vgpu_debug.txt with full report when any error occurs.
 */

#define DEBUG_CALL_HISTORY_SIZE 24

typedef struct {
    uint32_t call_id;
    uint32_t seq;
    int      status;   /* 0=OK, 2=transport err, else CUDA status */
    uint32_t transport_err;
} debug_call_entry_t;

static debug_call_entry_t g_call_history[DEBUG_CALL_HISTORY_SIZE];
static int g_call_history_head = 0;
static int g_call_history_count = 0;
static char g_checkpoint_trail[256] = "";

static void write_probe_file(const char *fmt, ...)
{
    int fd = open("/tmp/vgpu_shmem_probe.txt", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        return;
    }

    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (len > 0) {
        if ((size_t)len > sizeof(buf)) {
            len = (int)sizeof(buf);
        }
        ssize_t wrote = write(fd, buf, (size_t)len);
        (void)wrote;
    }
    close(fd);
}

static const char *call_id_to_name(uint32_t call_id)
{
    if (call_id == 0) return "(init/connect)";
    switch (call_id) {
        case CUDA_CALL_INIT: return "cuInit";
        case CUDA_CALL_DEVICE_GET_COUNT: return "cuDeviceGetCount";
        case CUDA_CALL_DEVICE_GET: return "cuDeviceGet";
        case CUDA_CALL_DEVICE_GET_NAME: return "cuDeviceGetName";
        case CUDA_CALL_DEVICE_GET_ATTRIBUTE: return "cuDeviceGetAttribute";
        case CUDA_CALL_DEVICE_TOTAL_MEM: return "cuDeviceTotalMem";
        case CUDA_CALL_DEVICE_PRIMARY_CTX_RETAIN: return "cuDevicePrimaryCtxRetain";
        case CUDA_CALL_DEVICE_PRIMARY_CTX_RELEASE: return "cuDevicePrimaryCtxRelease";
        case CUDA_CALL_CTX_CREATE: return "cuCtxCreate_v2";
        case CUDA_CALL_CTX_DESTROY: return "cuCtxDestroy_v2";
        case CUDA_CALL_CTX_SET_CURRENT: return "cuCtxSetCurrent";
        case CUDA_CALL_CTX_GET_CURRENT: return "cuCtxGetCurrent";
        case CUDA_CALL_CTX_SYNCHRONIZE: return "cuCtxSynchronize";
        case CUDA_CALL_MEM_ALLOC: return "cuMemAlloc_v2";
        case CUDA_CALL_MEM_FREE: return "cuMemFree_v2";
        case CUDA_CALL_MEMCPY_HTOD: return "cuMemcpyHtoD_v2";
        case CUDA_CALL_MEMCPY_DTOH: return "cuMemcpyDtoH_v2";
        case CUDA_CALL_MEMCPY_DTOD: return "cuMemcpyDtoD_v2";
        case CUDA_CALL_MEMSET_D8: return "cuMemsetD8_v2";
        case CUDA_CALL_MEMSET_D16: return "cuMemsetD16_v2";
        case CUDA_CALL_MEMSET_D32: return "cuMemsetD32_v2";
        case CUDA_CALL_MODULE_LOAD_DATA: return "cuModuleLoadData";
        case CUDA_CALL_MODULE_LOAD_DATA_EX: return "cuModuleLoadDataEx";
        case CUDA_CALL_MODULE_LOAD_FAT_BINARY: return "cuModuleLoadFatBinary";
        case CUDA_CALL_MODULE_UNLOAD: return "cuModuleUnload";
        case CUDA_CALL_MODULE_GET_FUNCTION: return "cuModuleGetFunction";
        case CUDA_CALL_MODULE_GET_GLOBAL: return "cuModuleGetGlobal";
        case CUDA_CALL_LAUNCH_KERNEL: return "cuLaunchKernel";
        case CUDA_CALL_LAUNCH_COOPERATIVE_KERNEL: return "cuLaunchCooperativeKernel";
        case CUDA_CALL_STREAM_CREATE: return "cuStreamCreate";
        case CUDA_CALL_STREAM_CREATE_WITH_FLAGS: return "cuStreamCreateWithFlags";
        case CUDA_CALL_STREAM_CREATE_WITH_PRIORITY: return "cuStreamCreateWithPriority";
        case CUDA_CALL_STREAM_DESTROY: return "cuStreamDestroy";
        case CUDA_CALL_STREAM_SYNCHRONIZE: return "cuStreamSynchronize";
        case CUDA_CALL_STREAM_QUERY: return "cuStreamQuery";
        case CUDA_CALL_STREAM_WAIT_EVENT: return "cuStreamWaitEvent";
        case CUDA_CALL_EVENT_CREATE: return "cuEventCreate";
        case CUDA_CALL_EVENT_CREATE_WITH_FLAGS: return "cuEventCreateWithFlags";
        case CUDA_CALL_EVENT_DESTROY: return "cuEventDestroy";
        case CUDA_CALL_EVENT_RECORD: return "cuEventRecord";
        case CUDA_CALL_EVENT_SYNCHRONIZE: return "cuEventSynchronize";
        case CUDA_CALL_EVENT_QUERY: return "cuEventQuery";
        case CUDA_CALL_EVENT_ELAPSED_TIME: return "cuEventElapsedTime";
        case CUDA_CALL_FUNC_GET_PARAM_INFO: return "cuFuncGetParamInfo";
        case CUDA_CALL_GET_GPU_INFO: return "cuGetGpuInfo";
        default: return "?(call_id)";
    }
}

static void debug_record_call(uint32_t call_id, uint32_t seq, int status, uint32_t transport_err)
{
    int i = g_call_history_head % DEBUG_CALL_HISTORY_SIZE;
    g_call_history[i].call_id = call_id;
    g_call_history[i].seq = seq;
    g_call_history[i].status = status;
    g_call_history[i].transport_err = transport_err;
    g_call_history_head++;
    if (g_call_history_count < DEBUG_CALL_HISTORY_SIZE) g_call_history_count++;
}

static void debug_write_report(const char *code, uint32_t failing_call_id,
                               uint32_t transport_err, const char *detail)
{
    char buf[4096];
    size_t off = 0;
    time_t now = time(NULL);
    const char *call_name = call_id_to_name(failing_call_id);
    const char *err_str = (transport_err == VGPU_ERR_MEDIATOR_UNAVAIL) ? "MEDIATOR_UNAVAIL" :
                          (transport_err == VGPU_ERR_TIMEOUT) ? "TIMEOUT" :
                          (transport_err == VGPU_ERR_QUEUE_FULL) ? "QUEUE_FULL" :
                          (transport_err == VGPU_ERR_RATE_LIMITED) ? "RATE_LIMITED" :
                          (transport_err == VGPU_ERR_VM_QUARANTINED) ? "VM_QUARANTINED" : "OTHER";

    off += (size_t)snprintf(buf + off, sizeof(buf) - off,
        "===============================================================================\n"
        "VGPU DEBUG REPORT — Exact error diagnosis\n"
        "===============================================================================\n"
        "timestamp: %ld  pid: %d\n\n"
        ">>> WHAT FAILED: %s\n"
        ">>> FAILING CALL: %s (call_id=0x%04x)\n"
        ">>> TRANSPORT ERR: 0x%08x (%s)\n"
        ">>> DETAIL: %s\n\n",
        (long)now, (int)getpid(),
        code ? code : "UNKNOWN",
        call_name, failing_call_id,
        transport_err, err_str,
        detail ? detail : "(none)");

    off += (size_t)snprintf(buf + off, sizeof(buf) - off,
        "--- RECENT CALL HISTORY (oldest first) ---\n");
    for (int i = 0; i < g_call_history_count; i++) {
        int idx = (g_call_history_head - g_call_history_count + i + DEBUG_CALL_HISTORY_SIZE)
                  % DEBUG_CALL_HISTORY_SIZE;
        const debug_call_entry_t *e = &g_call_history[idx];
        const char *r = (e->status == 0) ? "OK" : (e->status == 2) ? "ERR" : "CUDA_ERR";
        off += (size_t)snprintf(buf + off, sizeof(buf) - off,
            "  %2d. 0x%04x %-30s seq=%u  -> %s\n",
            i + 1, e->call_id, call_id_to_name(e->call_id), e->seq, r);
    }
    off += (size_t)snprintf(buf + off, sizeof(buf) - off,
        "\n--- CHECKPOINT TRAIL ---\n%s\n\n",
        g_checkpoint_trail);

    if (strcmp(code, "MEDIATOR_UNAVAIL") == 0)
        off += (size_t)snprintf(buf + off, sizeof(buf) - off,
            ">>> LIKELY CAUSE: Host mediator not running. Start mediator_phase3 on host.\n");
    else if (strcmp(code, "TRANSPORT_TIMEOUT") == 0)
        off += (size_t)snprintf(buf + off, sizeof(buf) - off,
            ">>> LIKELY CAUSE: Host hung or slow. Check mediator logs, increase CUDA_TRANSPORT_TIMEOUT_SEC.\n");
    else if (strcmp(code, "DEVICE_NOT_FOUND") == 0)
        off += (size_t)snprintf(buf + off, sizeof(buf) - off,
            ">>> LIKELY CAUSE: VGPU-STUB PCI device missing. Add -device vgpu-cuda to QEMU.\n");
    else if (strcmp(code, "BAR0_OPEN_FAILED") == 0)
        off += (size_t)snprintf(buf + off, sizeof(buf) - off,
            ">>> LIKELY CAUSE: Permission or sandbox blocking resource0. Check systemd ReadWritePaths.\n");
    else if (strcmp(code, "CUDA_CALL_FAILED") == 0)
        off += (size_t)snprintf(buf + off, sizeof(buf) - off,
            ">>> LIKELY CAUSE: Host GPU returned CUDA error. Check mediator/host CUDA logs.\n");

    int fd = (int)syscall(__NR_open, "/tmp/vgpu_debug.txt",
                          (O_WRONLY | O_CREAT | O_TRUNC), 0666);
    if (fd >= 0) {
        syscall(__NR_write, fd, buf, off);
        syscall(__NR_close, fd);
    }
    /* Also keep short canonical line for scripts */
    fd = (int)syscall(__NR_open, "/tmp/vgpu_last_error",
                      (O_WRONLY | O_CREAT | O_TRUNC), 0666);
    if (fd >= 0) {
        int n = snprintf(buf, sizeof(buf), "VGPU_ERR|%ld|%s|0x%04x|0x%08x|%s\n",
                         (long)now, code ? code : "UNKNOWN", failing_call_id, transport_err,
                         detail ? detail : "");
        if (n > 0 && n < (int)sizeof(buf))
            syscall(__NR_write, fd, buf, (size_t)n);
        syscall(__NR_close, fd);
    }
}

void cuda_transport_write_error(const char *code, uint32_t call_id,
                                uint32_t transport_err, const char *detail)
{
    debug_write_report(code, call_id, transport_err, detail);
}

void cuda_transport_clear_debug_state(void)
{
    g_call_history_head = 0;
    g_call_history_count = 0;
    g_checkpoint_trail[0] = '\0';
}

void cuda_transport_write_checkpoint(const char *phase)
{
    size_t len = strlen(g_checkpoint_trail);
    if (len > 0) {
        snprintf(g_checkpoint_trail + len, sizeof(g_checkpoint_trail) - len, " -> %s", phase);
    } else {
        snprintf(g_checkpoint_trail, sizeof(g_checkpoint_trail), "%s", phase);
    }
    char buf[256];
    time_t now = time(NULL);
    int n = snprintf(buf, sizeof(buf), "VGPU_CHECKPOINT|%ld|%s\n", (long)now, phase);
    if (n <= 0 || n >= (int)sizeof(buf)) return;
    int fd = (int)syscall(__NR_open, "/tmp/vgpu_checkpoint",
                         (O_WRONLY | O_CREAT | O_TRUNC), 0666);
    if (fd >= 0) {
        syscall(__NR_write, fd, buf, (size_t)n);
        syscall(__NR_close, fd);
    }
}

/*
 * CUDA Transport — guest-side communication layer
 *
 * Finds the VGPU-STUB PCI device, maps BAR0, and provides a blocking
 * RPC interface for the CUDA shim library.
 *
 * Data path (preferred — VHOST-style shared memory):
 *   During init, a large anonymous region is mmap'd + mlock'd and its
 *   guest physical address (GPA) is registered with the vgpu-stub via
 *   new MMIO registers (VGPU_REG_SHMEM_*).  The vgpu-stub then maps
 *   that guest memory directly via cpu_physical_memory_map(), so data
 *   flows without going through the 8 MB BAR1 MMIO window:
 *
 *     guest shim                         vgpu-stub (QEMU)
 *       memcpy to shmem_g2h  ─────────▶  reads host ptr to same RAM
 *       write MMIO doorbell  ──MMIO──▶   (one VM exit, tiny payload)
 *       poll BAR0 STATUS     ◀──MMIO──   STATUS = DONE/ERROR
 *       memcpy from shmem_h2g ◀─────────  vgpu-stub wrote result there
 *
 * Data path (fallback — BAR1 MMIO, for guests that cannot mlock):
 *   The original 8 MB BAR1 window is used.  For transfers larger than
 *   CUDA_MAX_CHUNK_SIZE (4 MB) the existing chunked helpers are used.
 *
 * Control registers (BAR0) are always used for call metadata, status,
 * inline args, and small data regardless of which data path is active.
 *
 * Large transfers in shmem mode:
 *   When the data is larger than shmem_g2h_size, the existing chunked
 *   helpers are called with shmem_g2h_size as the chunk limit, giving
 *   far fewer round-trips than the 4 MB BAR1 chunks.
 */

/* When VGPU_DEBUG is unset, skip per-call and verbose discovery logging to avoid inference delay. */
static int vgpu_debug_logging(void) {
    static int cached = -1;
    if (cached < 0) cached = (getenv("VGPU_DEBUG") != NULL) ? 1 : 0;
    return cached;
}

/* Bulk-transfer tracing is much more expensive than regular debug logs.
 * Keep it off by default even when VGPU_DEBUG is set. */
static int vgpu_bulk_trace_logging(void) {
    static int cached = -1;
    if (cached < 0) cached = (getenv("VGPU_TRACE_BULK_IO") != NULL) ? 1 : 0;
    return cached;
}

/* Serialize all transport round-trips so one thread cannot read another's result from BAR0. */
static pthread_mutex_t g_transport_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * The guest VM exposes one shared MMIO transport device per VM, so requests
 * from different processes can collide even though each process has its own
 * in-process mutex. Serialize across processes too, otherwise duplicate small
 * seq numbers (1,2,3,...) from separate runners can cause stub-side stale
 * reply drops (`CUDA result IGNORED ... pending_seq=...`).
 */
static int acquire_transport_process_lock(void)
{
    int fd = open("/var/tmp/vgpu_transport.lock",
                  O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (fd < 0) {
        return -1;
    }
    if (flock(fd, LOCK_EX) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void release_transport_process_lock(int fd)
{
    if (fd < 0) {
        return;
    }
    (void)flock(fd, LOCK_UN);
    close(fd);
}

static int acquire_shmem_owner_lock(void)
{
    const char *allow_multi = getenv("VGPU_ALLOW_MULTI_PROCESS_SHMEM");
    int fd;

    if (allow_multi && allow_multi[0] && strcmp(allow_multi, "0") != 0) {
        return -2;
    }

    fd = open("/var/tmp/vgpu_shmem_owner.lock",
              O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (fd < 0) {
        return -1;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void release_shmem_owner_lock(int fd)
{
    if (fd < 0) {
        return;
    }
    (void)flock(fd, LOCK_UN);
    close(fd);
}

/* Forward declaration */
static void call_libvgpu_set_skip_interception(int skip);

/* ---- PCI scan constants ---------------------------------------- */
#define VGPU_VENDOR_ID   0x10DE
#define VGPU_DEVICE_ID   0x2331
#define VGPU_CLASS        0x030200
#define VGPU_CLASS_MASK   0xFFFF00

/* ---- BAR0 register offsets (must match vgpu_protocol.h) -------- */
#define REG_DOORBELL        0x000
#define REG_STATUS          0x004
#define REG_VM_ID           0x010
#define REG_ERROR_CODE      0x014
#define REG_REQUEST_LEN     0x018
#define REG_RESPONSE_LEN    0x01C

/* CUDA-specific registers */
#define REG_CUDA_OP         0x080
#define REG_CUDA_SEQ        0x084
#define REG_CUDA_NUM_ARGS   0x088
#define REG_CUDA_DATA_LEN   0x08C
#define REG_CUDA_DOORBELL   0x0A8
#define REG_CUDA_ARGS_BASE  0x0B0

/* BAR0 CUDA request/response data region */
#define CUDA_REQ_DATA_OFFSET   0x100
#define CUDA_RESP_DATA_OFFSET  0x500
#define CUDA_SMALL_DATA_MAX    1024

/* BAR0 CUDA result registers */
#define REG_CUDA_RESULT_STATUS   0x0F0
#define REG_CUDA_RESULT_NUM      0x0F4
#define REG_CUDA_RESULT_DATA_LEN 0x0F8
#define REG_CUDA_RESULT_BASE     0x900

/* Shared-memory registration registers (new, match vgpu_protocol.h 0x940-0x94C)
 * These are placed AFTER the CUDA result-value block to avoid overlap with
 * the CUDA control registers (0x080-0x0FF) and CUDA arg registers (0x0B0-0x0EF). */
#define REG_SHMEM_GPA_LO    0x940
#define REG_SHMEM_GPA_HI    0x944
#define REG_SHMEM_SIZE      0x948
#define REG_SHMEM_CTRL      0x94C

/* Capabilities register (BAR0 0x024) */
#define REG_CAPABILITIES    0x024
#define VGPU_CAP_SHMEM      (1u << 6)
#define VGPU_CAP_BAR1_DATA  (1u << 5)
#define REG_PROTOCOL_VER    0x020

/* BAR sizes */
#define BAR0_SIZE  4096
#define BAR1_SIZE  (16 * 1024 * 1024)
/* Workaround: BAR0 status read returns stale value on some Xen/qemu-dm; poll BAR1 tail instead */
#define BAR1_STATUS_MIRROR_OFFSET  (BAR1_SIZE - 4)

/* Status register values */
#define STATUS_IDLE   0x00
#define STATUS_BUSY   0x01
#define STATUS_DONE   0x02
#define STATUS_ERROR  0x03

#define VGPU_ERR_NONE              0x00
#define VGPU_ERR_MEDIATOR_UNAVAIL  0x03
#define VGPU_ERR_TIMEOUT           0x04
#define VGPU_ERR_QUEUE_FULL        0x07
#define VGPU_ERR_RATE_LIMITED      0x0A
#define VGPU_ERR_VM_QUARANTINED    0x0B

/* Polling */
/* 10 ms so QEMU main loop / fd handler can run and set DONE (avoid tight poll starving iothread) */
#define POLL_INTERVAL_US  2000
#define POLL_TIMEOUT_SEC_DEFAULT  60
#define POLL_TIMEOUT_SEC_MIN      120  /* model load can be slow; avoid ~10s failure */
/* Override via CUDA_TRANSPORT_TIMEOUT_SEC when mediator is slow. */
static int poll_timeout_sec(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("CUDA_TRANSPORT_TIMEOUT_SEC");
        int v = (e && *e) ? (int)strtol(e, NULL, 10) : POLL_TIMEOUT_SEC_DEFAULT;
        if (v <= 0) v = POLL_TIMEOUT_SEC_DEFAULT;
        cached = (v < POLL_TIMEOUT_SEC_MIN) ? POLL_TIMEOUT_SEC_MIN : v;
    }
    return cached;
}

static size_t shmem_min_span_bytes(void)
{
    static size_t cached = 0;
    if (cached == 0) {
        const char *e = getenv("VGPU_SHMEM_MIN_SPAN_KB");
        size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        size_t min_allowed;
        size_t v = 64u * 1024u;

        if (page_size == 0) {
            page_size = 4096;
        }
        min_allowed = 2u * page_size;

        if (e && e[0]) {
            char *end = NULL;
            unsigned long long kb = strtoull(e, &end, 10);
            if (end && *end == '\0' && kb > 0) {
                v = (size_t)kb * 1024u;
            }
        }

        if (v < min_allowed) {
            v = min_allowed;
        }
        if ((v % min_allowed) != 0) {
            v = ((v + min_allowed - 1) / min_allowed) * min_allowed;
        }
        cached = v;
    }
    return cached;
}

/* BAR1 legacy regions */
#define BAR1_GUEST_TO_HOST_OFFSET  0x000000
#define BAR1_GUEST_TO_HOST_SIZE    (8 * 1024 * 1024)
#define BAR1_HOST_TO_GUEST_OFFSET  0x800000
#define BAR1_HOST_TO_GUEST_SIZE    (8 * 1024 * 1024)

/* Default shared-memory region (must match VGPU_SHMEM_DEFAULT_SIZE) */
#define SHMEM_DEFAULT_SIZE   (256u * 1024u * 1024u)
#define SHMEM_MIN_SIZE       (  1u * 1024u * 1024u)

/* Register access */
#define REG32(base, off)  (*(volatile uint32_t *)((volatile char *)(base) + (off)))
#define REG64(base, off)  (*(volatile uint64_t *)((volatile char *)(base) + (off)))

/* Module-level PCI BDF populated by find_vgpu_device().
 * Accessible even when the transport struct has not been allocated yet
 * (e.g. when the caller only needs the address for cuDeviceGetPCIBusId). */
static char g_discovered_bdf[256] = "";  /* Increased from 64 to 256 to match dirent->d_name max size */

static const char *vgpu_err_to_str(uint32_t err)
{
    switch (err) {
        case VGPU_ERR_NONE: return "NONE";
        case VGPU_ERR_INVALID_REQUEST: return "INVALID_REQUEST";
        case VGPU_ERR_REQUEST_TOO_LARGE: return "REQUEST_TOO_LARGE";
        case VGPU_ERR_MEDIATOR_UNAVAIL: return "MEDIATOR_UNAVAIL";
        case VGPU_ERR_TIMEOUT: return "TIMEOUT";
        case VGPU_ERR_CUDA_ERROR: return "CUDA_ERROR";
        case VGPU_ERR_INVALID_LENGTH: return "INVALID_LENGTH";
        case VGPU_ERR_QUEUE_FULL: return "QUEUE_FULL";
        case VGPU_ERR_RATE_LIMITED: return "RATE_LIMITED";
        case VGPU_ERR_VM_QUARANTINED: return "VM_QUARANTINED";
        case VGPU_ERR_UNSUPPORTED_OP: return "UNSUPPORTED_OP";
        case VGPU_ERR_INVALID_POOL: return "INVALID_POOL";
        default: return "UNKNOWN";
    }
}

/* ---------------------------------------------------------------- */
struct cuda_transport {
    volatile void *bar0;         /* BAR0 MMIO mapping                    */
    volatile void *bar1;         /* BAR1 MMIO mapping (legacy, may NULL) */
    int            bar0_fd;
    int            bar1_fd;
    uint32_t       vm_id;
    uint32_t       seq_counter;
    int            has_bar1;     /* 1 if BAR1 is mapped (legacy path)    */

    /* PCI bus/device/function identifier, e.g. "0000:00:05.0" */
    char           pci_bdf[64];

    /* VHOST-style shared memory */
    void          *shmem;        /* mmap base (may be larger than window) */
    size_t         shmem_alloc_size; /* total mmap+mlock span            */
    size_t         shmem_size;   /* registered contiguous GPA window     */
    void          *shmem_g2h;   /* first half: guest → host data        */
    void          *shmem_h2g;   /* second half: host → guest data       */
    size_t         shmem_half;  /* shmem_size / 2                       */
    int            has_shmem;   /* 1 if shared-memory path is active    */
    uint64_t       shmem_registered_gpa; /* GPA passed to stub at REG_SHMEM_* */
    int            shmem_owner_lock_fd;  /* VM-global SHMEM ownership lock */
};

/* ================================================================
 * PCI device scanner
 * ================================================================ */
static int find_vgpu_device(char *res0_path, size_t res0_sz,
                            char *res1_path, size_t res1_sz,
                            char *bdf_out,   size_t bdf_sz)
{
    DIR *dir;
    struct dirent *entry;
    char pci_path[288];
    char attr_path[512];
    char line[256];
    FILE *fp;
    unsigned int vendor, device, cls;

    dir = opendir("/sys/bus/pci/devices");
    if (!dir) {
        fprintf(stderr, "[cuda-transport] Cannot open /sys/bus/pci/devices: %s\n",
                strerror(errno));
        return -1;
    }

    /* CRITICAL: Set skip flag at the VERY START of find_vgpu_device()
     * This ensures files are read with real values, not intercepted values
     * This is needed because find_vgpu_device() might be called directly
     * without going through cuda_transport_init() or cuda_transport_discover() */
    call_libvgpu_set_skip_interception(1);
    if (vgpu_debug_logging()) {
        (void)syscall(__NR_write, 2,
                      "[cuda-transport] FORCE: find_vgpu_device() STARTED - setting skip flag\n",
                      sizeof("[cuda-transport] FORCE: find_vgpu_device() STARTED - setting skip flag\n") - 1);
        (void)syscall(__NR_write, 2,
                      "[cuda-transport] FORCE: Skip flag set to 1 in find_vgpu_device()\n",
                      sizeof("[cuda-transport] FORCE: Skip flag set to 1 in find_vgpu_device()\n") - 1);
    }
    int device_count = 0;
    if (vgpu_debug_logging()) { fprintf(stderr, "[cuda-transport] DEBUG: Starting device scan...\n"); fflush(stderr); }
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        device_count++;
        if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Scanning device %d: %s\n", device_count, entry->d_name);

        snprintf(pci_path, sizeof(pci_path),
                 "/sys/bus/pci/devices/%s", entry->d_name);

        vendor = device = cls = 0;

        snprintf(attr_path, sizeof(attr_path), "%s/vendor", pci_path);
        fp = fopen(attr_path, "r");
        if (!fp) {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Cannot open vendor: %s\n", entry->d_name, strerror(errno));
            continue;
        }
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &vendor);
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Read vendor: 0x%04x (line: %s)", entry->d_name, vendor, line);
        } else {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Failed to read vendor line\n", entry->d_name);
        }
        fclose(fp);

        snprintf(attr_path, sizeof(attr_path), "%s/device", pci_path);
        fp = fopen(attr_path, "r");
        if (!fp) {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Cannot open device: %s\n", entry->d_name, strerror(errno));
            continue;
        }
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &device);
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Read device: 0x%04x (line: %s)", entry->d_name, device, line);
        } else {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Failed to read device line\n", entry->d_name);
        }
        fclose(fp);

        snprintf(attr_path, sizeof(attr_path), "%s/class", pci_path);
        fp = fopen(attr_path, "r");
        if (!fp) {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Cannot open class: %s\n", entry->d_name, strerror(errno));
            continue;
        }
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &cls);
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Read class: 0x%06x (line: %s)", entry->d_name, cls, line);
        } else {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Failed to read class line\n", entry->d_name);
        }
        fclose(fp);

        if (vgpu_debug_logging()) {
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] Final values: vendor=0x%04x device=0x%04x class=0x%06x\n",
                    entry->d_name, vendor, device, cls);
            fflush(stderr);
        }
        int class_ok = ((cls & VGPU_CLASS_MASK) == VGPU_CLASS);
        if (vgpu_debug_logging()) {
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] class_ok=%d (cls & mask=0x%06x, expected=0x%06x)\n",
                    entry->d_name, class_ok, (cls & VGPU_CLASS_MASK), VGPU_CLASS);
            fflush(stderr);
        }

        /*
         * Accept the device if:
         *   (a) Exact match: vendor=0x10DE device=0x2331 class=0x030200
         *       — QEMU built with our vgpu_protocol.h IDs.
         *   (b) Class match with a QEMU/Red-Hat generic vendor
         *       (0x1234 = Red Hat QEMU, 0x1AF4 = VirtIO/Red Hat)
         *       — Handles QEMU builds that use the legacy stub vendor IDs.
         *
         * In either case the PCI class must be 0x030200 (3D controller /
         * VGA-compatible GPU) to avoid accidentally matching non-GPU devices.
         */
        int exact  = class_ok && (vendor == VGPU_VENDOR_ID) &&
                                  (device == VGPU_DEVICE_ID);
        int legacy = class_ok && (vendor == 0x1234 || vendor == 0x1AF4);
        
        if (vgpu_debug_logging()) {
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] Matching: exact=%d (vendor_match=%d device_match=%d) legacy=%d\n",
                    entry->d_name, exact, (vendor == VGPU_VENDOR_ID), (device == VGPU_DEVICE_ID), legacy);
            fflush(stderr);
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] Expected: vendor=0x%04x device=0x%04x, got: vendor=0x%04x device=0x%04x\n",
                    entry->d_name, VGPU_VENDOR_ID, VGPU_DEVICE_ID, vendor, device);
            fflush(stderr);
        }
        if (!exact && !legacy) {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] No match, continuing...\n", entry->d_name);
            continue;
        }
        if (vgpu_debug_logging()) {
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] *** MATCH FOUND! exact=%d legacy=%d ***\n",
                    entry->d_name, exact, legacy);
            fprintf(stderr,
                    "[cuda-transport] Found VGPU-STUB at %s "
                    "(vendor=0x%04x device=0x%04x class=0x%06x match=%s)\n",
                    entry->d_name, vendor, device, cls,
                    exact ? "exact" : "legacy-qemu");
        }

        snprintf(res0_path, res0_sz, "%s/resource0", pci_path);
        snprintf(res1_path, res1_sz, "%s/resource1", pci_path);
        if (bdf_out && bdf_sz > 0) {
            /* Copy BDF, ensuring null termination */
            size_t copy_len = (bdf_sz - 1 < strlen(entry->d_name)) ? (bdf_sz - 1) : strlen(entry->d_name);
            memcpy(bdf_out, entry->d_name, copy_len);
            bdf_out[copy_len] = '\0';
        }
        /* Always record the BDF globally so callers that don't need the
         * full transport (e.g. the lightweight discover path) can still
         * return a correct PCI address via cuda_transport_pci_bdf(NULL). */
        strncpy(g_discovered_bdf, entry->d_name, sizeof(g_discovered_bdf) - 1);
        g_discovered_bdf[sizeof(g_discovered_bdf) - 1] = '\0';
        closedir(dir);
        /* Re-enable interception after successful discovery */
        call_libvgpu_set_skip_interception(0);
        return 0;
    }

    fprintf(stderr,
            "[cuda-transport] VGPU-STUB not found in /sys/bus/pci/devices "
            "(scanned %d devices, want vendor=0x%04x device=0x%04x OR QEMU-vendor with "
            "class=0x%06x)\n",
            device_count, VGPU_VENDOR_ID, VGPU_DEVICE_ID, VGPU_CLASS);
    closedir(dir);
    /* Re-enable interception after discovery */
    call_libvgpu_set_skip_interception(0);
    return -1;
}

/* ================================================================
 * Guest Physical Address resolution via /proc/self/pagemap
 *
 * Returns the physical address of the virtual page containing vaddr,
 * or 0 on failure.  The page must already be faulted in (e.g. by
 * mlock or a dummy write).
 * ================================================================ */
enum vgpu_pagemap_probe_status {
    VGPU_PAGEMAP_OK = 0,
    VGPU_PAGEMAP_OPEN_FAILED,
    VGPU_PAGEMAP_READ_FAILED,
    VGPU_PAGEMAP_NOT_PRESENT,
    VGPU_PAGEMAP_PFN_HIDDEN,
    VGPU_PAGEMAP_NONCONTIGUOUS,
};

static const char *vgpu_pagemap_probe_status_str(int status)
{
    switch (status) {
        case VGPU_PAGEMAP_OK: return "ok";
        case VGPU_PAGEMAP_OPEN_FAILED: return "open_failed";
        case VGPU_PAGEMAP_READ_FAILED: return "read_failed";
        case VGPU_PAGEMAP_NOT_PRESENT: return "not_present";
        case VGPU_PAGEMAP_PFN_HIDDEN: return "pfn_hidden";
        case VGPU_PAGEMAP_NONCONTIGUOUS: return "noncontiguous";
        default: return "unknown";
    }
}

static uint64_t current_effective_caps(void)
{
    FILE *fp = fopen("/proc/self/status", "r");
    char line[256];
    uint64_t caps = 0;

    if (!fp) return 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "CapEff:", 7) == 0) {
            unsigned long long parsed = 0;
            if (sscanf(line + 7, "%llx", &parsed) == 1) {
                caps = (uint64_t)parsed;
            }
            break;
        }
    }
    fclose(fp);
    return caps;
}

static int process_cmdline_contains(const char *needle)
{
    int fd;
    char buf[4096];
    ssize_t n;
    size_t needle_len;

    if (!needle || !needle[0]) {
        return 0;
    }

    fd = open("/proc/self/cmdline", O_RDONLY);
    if (fd < 0) {
        return 0;
    }
    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) {
        return 0;
    }
    buf[n] = '\0';
    for (ssize_t i = 0; i < n; i++) {
        if (buf[i] == '\0') {
            buf[i] = ' ';
        }
    }
    needle_len = strlen(needle);
    if (needle_len == 0 || needle_len >= sizeof(buf)) {
        return 0;
    }
    return strstr(buf, needle) != NULL ? 1 : 0;
}

static int skip_shmem_for_ollama_engine(void)
{
    const char *allow = getenv("VGPU_SHMEM_ALLOW_OLLAMA_ENGINE");
    if (allow && allow[0] && strcmp(allow, "0") != 0) {
        return 0;
    }
    if (!process_cmdline_contains("--ollama-engine")) {
        return 0;
    }
    fprintf(stderr,
            "[cuda-transport] SHMEM disabled for --ollama-engine helper pid=%d; using BAR1/inline only\n",
            (int)getpid());
    return 1;
}

static int pagemap_entry_for(const void *vaddr, uint64_t *pme_out)
{
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return VGPU_PAGEMAP_OPEN_FAILED;

    uint64_t page_size = (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t vfn       = (uintptr_t)vaddr / page_size;
    uint64_t pme       = 0;

    if (pread(fd, &pme, sizeof(pme), (off_t)(vfn * sizeof(pme))) != (ssize_t)sizeof(pme)) {
        close(fd);
        return VGPU_PAGEMAP_READ_FAILED;
    }
    close(fd);
    *pme_out = pme;

    /* Bit 63 = present; bits 54:0 = PFN */
    if (!(pme & (1ULL << 63))) return VGPU_PAGEMAP_NOT_PRESENT;
    if ((pme & 0x007FFFFFFFFFFFFFULL) == 0) return VGPU_PAGEMAP_PFN_HIDDEN;
    return VGPU_PAGEMAP_OK;
}

static void shmem_registration_trace(const char *stage,
                                     cuda_transport_t *tp,
                                     uint64_t requested_gpa,
                                     size_t shmem_size,
                                     const void *g2h_ptr,
                                     const void *h2g_ptr)
{
    uint64_t pme = 0;
    uint64_t pagemap_gpa = 0;
    int pmst = VGPU_PAGEMAP_OPEN_FAILED;
    char line[640];
    int n;

    if (!stage) {
        return;
    }

    if (g2h_ptr) {
        pmst = pagemap_entry_for(g2h_ptr, &pme);
        if (pmst == VGPU_PAGEMAP_OK) {
            uint64_t pfn = pme & 0x007FFFFFFFFFFFFFULL;
            uint64_t ps = (uint64_t)sysconf(_SC_PAGESIZE);
            pagemap_gpa = pfn * ps;
        }
    }

    n = snprintf(line, sizeof(line),
                 "[cuda-transport] SHMEM_REG stage=%s pid=%d tp=%p req_gpa=0x%016llx "
                 "stored_gpa=0x%016llx size=%zu g2h=%p h2g=%p pagemap_st=%s "
                 "pagemap_gpa=0x%016llx\n",
                 stage, (int)getpid(), (void *)tp,
                 (unsigned long long)requested_gpa,
                 (unsigned long long)(tp ? tp->shmem_registered_gpa : 0),
                 shmem_size, g2h_ptr, h2g_ptr,
                 vgpu_pagemap_probe_status_str(pmst),
                 (unsigned long long)pagemap_gpa);
    if (n <= 0) {
        return;
    }

    (void)fwrite(line, 1, (size_t)n, stderr);
    fflush(stderr);

    {
        int fd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_shmem_registration.log",
                              O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            (void)syscall(__NR_write, fd, line, (size_t)n);
            (void)syscall(__NR_close, fd);
        }
    }
}

static void refresh_shmem_registration_for_request(cuda_transport_t *tp,
                                                   uint32_t call_id,
                                                   uint32_t send_len)
{
    uint64_t pme = 0;
    uint64_t gpa = 0;
    uint64_t ps = 0;
    int pmst;
    uint32_t st;

    if (!tp || !tp->has_shmem || !tp->bar0 || !tp->shmem_g2h ||
        send_len <= CUDA_SMALL_DATA_MAX) {
        return;
    }

    switch (call_id) {
    case CUDA_CALL_MEMCPY_HTOD:
    case CUDA_CALL_MEMCPY_HTOD_ASYNC:
    case CUDA_CALL_MODULE_LOAD_DATA:
    case CUDA_CALL_MODULE_LOAD_DATA_EX:
    case CUDA_CALL_MODULE_LOAD_FAT_BINARY:
    case CUDA_CALL_LIBRARY_LOAD_DATA:
        break;
    default:
        return;
    }

    pmst = pagemap_entry_for(tp->shmem_g2h, &pme);
    if (pmst != VGPU_PAGEMAP_OK) {
        return;
    }

    ps = (uint64_t)sysconf(_SC_PAGESIZE);
    if (ps == 0) {
        ps = 4096;
    }
    gpa = (pme & 0x007FFFFFFFFFFFFFULL) * ps;
    if (gpa == 0) {
        return;
    }

    tp->shmem_registered_gpa = gpa;
    REG32(tp->bar0, REG_SHMEM_GPA_LO) = (uint32_t)(gpa & 0xFFFFFFFFu);
    REG32(tp->bar0, REG_SHMEM_GPA_HI) = (uint32_t)(gpa >> 32);
    REG32(tp->bar0, REG_SHMEM_SIZE)   = (uint32_t)tp->shmem_size;
    REG32(tp->bar0, REG_SHMEM_CTRL)   = 1;
    __sync_synchronize();
    st = REG32(tp->bar0, REG_STATUS);

    if (vgpu_bulk_trace_logging() && call_id == CUDA_CALL_LIBRARY_LOAD_DATA) {
        char line[320];
        int n = snprintf(line, sizeof(line),
                         "[cuda-transport] REFRESH_SHMEM_REG pid=%d call_id=0x%04x len=%u gpa=0x%016llx status=0x%02x\n",
                         (int)getpid(), call_id, send_len,
                         (unsigned long long)gpa, st);
        if (n > 0) {
            (void)fwrite(line, 1, (size_t)n, stderr);
            fflush(stderr);
        }
    }
}

static int find_contiguous_gpa_span(void *base,
                                    size_t span_len,
                                    size_t min_len,
                                    void **best_virt_out,
                                    uint64_t *best_gpa_out,
                                    size_t *best_len_out,
                                    uint64_t *detail_out)
{
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    size_t total_pages;
    size_t cur_pages = 0;
    size_t best_pages = 0;
    size_t ok_pages = 0;
    uint64_t prev_phys = 0;
    uint64_t cur_start_phys = 0;
    void *cur_start_virt = NULL;
    void *best_start_virt = NULL;
    uint64_t best_start_phys = 0;
    uint64_t last_pme = 0;
    int last_status = VGPU_PAGEMAP_OK;
    enum { PFN_SAMPLE_MAX = 8 };
    size_t sample_count = 0;
    size_t sample_page_idx[PFN_SAMPLE_MAX];
    uint64_t sample_pfn[PFN_SAMPLE_MAX];
    memset(sample_page_idx, 0, sizeof(sample_page_idx));
    memset(sample_pfn, 0, sizeof(sample_pfn));

    if (page_size == 0 || span_len < min_len || min_len < page_size) {
        if (best_virt_out) *best_virt_out = NULL;
        if (best_gpa_out) *best_gpa_out = 0;
        if (best_len_out) *best_len_out = 0;
        if (detail_out) *detail_out = 0;
        return VGPU_PAGEMAP_NONCONTIGUOUS;
    }

    total_pages = span_len / page_size;

    for (size_t i = 0; i < total_pages; ++i) {
        char *page = (char *)base + (i * page_size);
        uint64_t pme = 0;
        int st = pagemap_entry_for(page, &pme);

        if (st != VGPU_PAGEMAP_OK) {
            cur_pages = 0;
            cur_start_virt = NULL;
            cur_start_phys = 0;
            last_status = st;
            last_pme = pme;
            continue;
        }

        uint64_t phys = (pme & 0x007FFFFFFFFFFFFFULL) * (uint64_t)page_size;
        ok_pages++;
        if (sample_count < PFN_SAMPLE_MAX) {
            sample_page_idx[sample_count] = i;
            sample_pfn[sample_count] = pme & 0x007FFFFFFFFFFFFFULL;
            sample_count++;
        }
        if (cur_pages == 0 || phys != prev_phys + page_size) {
            cur_pages = 1;
            cur_start_virt = page;
            cur_start_phys = phys;
        } else {
            cur_pages++;
        }
        prev_phys = phys;

        if (cur_pages > best_pages) {
            best_pages = cur_pages;
            best_start_virt = cur_start_virt;
            best_start_phys = cur_start_phys;
        }
        /* Do not stop at the first qualifying run. The live Phase 1 blocker is
         * now the largest libload payloads still falling back to BAR1 because
         * the registered SHMEM aperture is too small. Scan the full mapping and
         * return the largest contiguous GPA run we can find. */
    }

    if (detail_out) {
        if (best_pages > 0) {
            *detail_out = best_pages * page_size;
        } else {
            *detail_out = last_pme;
        }
    }
    if (best_pages > 0) {
        size_t best_len = best_pages * page_size;
        best_len &= ~(size_t)((2 * page_size) - 1);
        if (best_len >= min_len) {
            write_probe_file("probe_v1 success span_len=%zu min_len=%zu ok_pages=%zu best_pages=%zu best_len=%zu best_gpa=0x%016llx best_virt=%p samples=",
                             span_len,
                             min_len,
                             ok_pages,
                             best_pages,
                             best_len,
                             (unsigned long long)best_start_phys,
                             best_start_virt);
            for (size_t i = 0; i < sample_count; ++i) {
                write_probe_file("%s%zu:0x%llx",
                                 (i == 0) ? "" : ",",
                                 sample_page_idx[i],
                                 (unsigned long long)sample_pfn[i]);
            }
            write_probe_file("\n");
            *best_virt_out = best_start_virt;
            *best_gpa_out = best_start_phys;
            *best_len_out = best_len;
            return VGPU_PAGEMAP_OK;
        }
    }
    write_probe_file("probe_v1 sample status=%s last_pme=0x%016llx ok_pages=%zu best_pages=%zu samples=",
                     vgpu_pagemap_probe_status_str(last_status),
                     (unsigned long long)last_pme,
                     ok_pages,
                     best_pages);
    for (size_t i = 0; i < sample_count; ++i) {
        write_probe_file("%s%zu:0x%llx",
                         (i == 0) ? "" : ",",
                         sample_page_idx[i],
                         (unsigned long long)sample_pfn[i]);
    }
    write_probe_file("\n");
    return (last_status == VGPU_PAGEMAP_OK) ? VGPU_PAGEMAP_NONCONTIGUOUS : last_status;
}

#define SHMEM_HUGEPAGE_ALIGN (2u * 1024u * 1024u)

static void *alloc_aligned_anon_mapping(size_t size, int mmap_flags, size_t align)
{
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    if (page_size == 0) {
        page_size = 4096;
    }
    if (align < page_size) {
        align = page_size;
    }
    if ((align & (align - 1)) != 0) {
        errno = EINVAL;
        return MAP_FAILED;
    }

    size_t reserve = size;
    if (align > page_size) {
        reserve += align;
    }

    void *base = mmap(NULL, reserve, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
    if (base == MAP_FAILED) {
        return MAP_FAILED;
    }

    uintptr_t raw = (uintptr_t)base;
    uintptr_t aligned = (raw + align - 1) & ~((uintptr_t)align - 1);
    size_t prefix = (size_t)(aligned - raw);
    size_t suffix = reserve - prefix - size;

    if (prefix > 0) {
        munmap((void *)raw, prefix);
    }
    if (suffix > 0) {
        munmap((void *)(aligned + size), suffix);
    }

    return (void *)aligned;
}

/* ================================================================
 * Try to allocate and register the shared-memory region.
 * Returns 1 on success, 0 if the feature should fall back to BAR1.
 * ================================================================ */
static int setup_shmem(cuda_transport_t *t)
{
    const char *disable_shmem = getenv("VGPU_DISABLE_SHMEM");
    int shmem_owner_lock_fd = -1;
    if (disable_shmem && disable_shmem[0] && strcmp(disable_shmem, "0") != 0) {
        fprintf(stderr, "[cuda-transport] SHMEM disabled by VGPU_DISABLE_SHMEM=%s — using BAR1\n",
                disable_shmem);
        return 0;
    }
    if (skip_shmem_for_ollama_engine()) {
        return 0;
    }
    shmem_owner_lock_fd = acquire_shmem_owner_lock();
    if (shmem_owner_lock_fd == -1) {
        fprintf(stderr,
                "[cuda-transport] SHMEM owner already held by another process; pid=%d using BAR1/inline only\n",
                (int)getpid());
        return 0;
    }

    uint32_t caps = REG32(t->bar0, REG_CAPABILITIES);
    if (!(caps & VGPU_CAP_SHMEM)) {
        fprintf(stderr, "[cuda-transport] vgpu-stub does not support "
                "shared-memory data path (caps=0x%x), using BAR1\n", caps);
        release_shmem_owner_lock(shmem_owner_lock_fd);
        return 0;
    }

    const size_t shmem_min_span = shmem_min_span_bytes();
    const char *env_min_kb = getenv("VGPU_SHMEM_MIN_SPAN_KB");

    fprintf(stderr,
            "[cuda-transport] shmem_marker_20260402a pid=%d env_min_kb=%s "
            "resolved_min_span_bytes=%zu resolved_min_span_kb=%zu\n",
            (int)getpid(),
            (env_min_kb && env_min_kb[0]) ? env_min_kb : "<unset>",
            shmem_min_span,
            shmem_min_span / 1024u);
    write_probe_file("marker_20260402a pid=%d env_min_kb=%s resolved_min_span_bytes=%zu resolved_min_span_kb=%zu\n",
                     (int)getpid(),
                     (env_min_kb && env_min_kb[0]) ? env_min_kb : "<unset>",
                     shmem_min_span,
                     shmem_min_span / 1024u);

    fprintf(stderr,
            "[cuda-transport] shmem probe config: min_span_bytes=%zu min_span_mb=%zu "
            "hugepage_align_bytes=%u default_size_mb=%u caps=0x%x\n",
            shmem_min_span,
            shmem_min_span >> 20,
            (unsigned)SHMEM_HUGEPAGE_ALIGN,
            (unsigned)(SHMEM_DEFAULT_SIZE >> 20),
            caps);
    write_probe_file("probe_v1 config min_span_bytes=%zu min_span_mb=%zu hugepage_align_bytes=%u default_size_mb=%u caps=0x%x\n",
                     shmem_min_span,
                     shmem_min_span >> 20,
                     (unsigned)SHMEM_HUGEPAGE_ALIGN,
                     (unsigned)(SHMEM_DEFAULT_SIZE >> 20),
                     caps);

    /* Try a large span first, but allow smaller windows if the guest cannot
     * lock or expose a large enough contiguous GPA run. */
    size_t try_sizes[] = {
        SHMEM_DEFAULT_SIZE,
        64u * 1024u * 1024u,
        16u * 1024u * 1024u,
        shmem_min_span,
        0
    };

    for (int size_idx = 0; try_sizes[size_idx] != 0; size_idx++) {
        size_t req_size = try_sizes[size_idx];
        int max_attempts = (req_size == SHMEM_DEFAULT_SIZE) ? 6 : 3;

        for (int attempt = 0; attempt < max_attempts; attempt++) {
            /* MAP_SHARED so guest PFNs for this mapping stay tied to the buffer
             * QEMU maps via registered GPA. MAP_PRIVATE + later bulk writes has
             * produced shmem_g2h that stayed all-zero when read back (see
             * vgpu_htod_transport.log HTOD written vs source). */
            int mmap_flags = MAP_SHARED | MAP_ANONYMOUS;
#ifdef MAP_32BIT
            int use_map32 = (attempt & 1);
            if (use_map32) {
                mmap_flags |= MAP_32BIT;
            }
#else
            int use_map32 = 0;
#endif
            void *shmem = alloc_aligned_anon_mapping(req_size, mmap_flags,
                                                     SHMEM_HUGEPAGE_ALIGN);
            size_t shmem_alloc_size = req_size;
            if (shmem == MAP_FAILED) {
                fprintf(stderr, "[cuda-transport] mmap shmem %zu MB failed "
                        "(attempt=%d map32=%d): %s\n",
                        req_size >> 20, attempt + 1, use_map32, strerror(errno));
                continue;
            }

#ifdef MADV_HUGEPAGE
            /* Give THP the best chance to back the minimum SHMEM window
             * with a single contiguous PFN run. */
            (void)madvise(shmem, shmem_alloc_size, MADV_HUGEPAGE);
#endif
            memset(shmem, 0, shmem_alloc_size);

#ifdef MADV_COLLAPSE
            if (getenv("VGPU_SHMEM_TRY_COLLAPSE") &&
                strcmp(getenv("VGPU_SHMEM_TRY_COLLAPSE"), "0") != 0) {
                if (madvise(shmem, shmem_alloc_size, MADV_COLLAPSE) == 0) {
                    fprintf(stderr,
                            "[cuda-transport] madvise(MADV_COLLAPSE) succeeded "
                            "(size=%zu MB attempt=%d map32=%d)\n",
                            shmem_alloc_size >> 20, attempt + 1, use_map32);
                    write_probe_file("probe_v1 collapse ok size_mb=%zu attempt=%d map32=%d\n",
                                     shmem_alloc_size >> 20, attempt + 1, use_map32);
                } else {
                    fprintf(stderr,
                            "[cuda-transport] madvise(MADV_COLLAPSE) failed "
                            "(size=%zu MB attempt=%d map32=%d): %s\n",
                            shmem_alloc_size >> 20, attempt + 1, use_map32,
                            strerror(errno));
                    write_probe_file("probe_v1 collapse fail size_mb=%zu attempt=%d map32=%d errno=%d\n",
                                     shmem_alloc_size >> 20, attempt + 1, use_map32, errno);
                }
            }
#endif

            if (mlock(shmem, shmem_alloc_size) != 0) {
                fprintf(stderr, "[cuda-transport] mlock(%zu MB) failed "
                        "(attempt=%d map32=%d): %s\n",
                        shmem_alloc_size >> 20, attempt + 1, use_map32, strerror(errno));
                munmap(shmem, shmem_alloc_size);
                continue;
            }

            void *shmem_window = NULL;
            uint64_t gpa = 0;
            size_t shmem_size = 0;
            uint64_t detail = 0;
            int gpa_rc = find_contiguous_gpa_span(shmem, shmem_alloc_size, shmem_min_span,
                                                  &shmem_window, &gpa, &shmem_size, &detail);
            if (gpa_rc != VGPU_PAGEMAP_OK) {
                uint64_t caps_eff = current_effective_caps();
                if (gpa_rc == VGPU_PAGEMAP_NONCONTIGUOUS) {
                    fprintf(stderr, "[cuda-transport] runtime1m_v2 No contiguous GPA span >= %zu KB "
                            "(min_span_bytes=%zu) inside %zu MB shmem mapping "
                            "(attempt=%d map32=%d best=%zu MB capeff=0x%llx)\n",
                            shmem_min_span,
                            shmem_min_span,
                            shmem_alloc_size >> 20,
                            attempt + 1, use_map32, detail >> 20,
                            (unsigned long long)caps_eff);
                    write_probe_file("probe_v1 noncontig min_span_bytes=%zu req_size_mb=%zu attempt=%d map32=%d best_mb=%zu caps=0x%llx\n",
                                     shmem_min_span,
                                     shmem_alloc_size >> 20,
                                     attempt + 1,
                                     use_map32,
                                     detail >> 20,
                                     (unsigned long long)caps_eff);
                } else {
                    fprintf(stderr, "[cuda-transport] Cannot resolve GPA for shmem "
                            "(attempt=%d map32=%d reason=%s detail=0x%llx capeff=0x%llx)\n",
                            attempt + 1, use_map32,
                            vgpu_pagemap_probe_status_str(gpa_rc),
                            (unsigned long long)detail,
                            (unsigned long long)caps_eff);
                }
                munlock(shmem, shmem_alloc_size);
                munmap(shmem, shmem_alloc_size);
                continue;
            }

            REG32(t->bar0, REG_SHMEM_GPA_LO) = (uint32_t)(gpa & 0xFFFFFFFFu);
            REG32(t->bar0, REG_SHMEM_GPA_HI) = (uint32_t)(gpa >> 32);
            REG32(t->bar0, REG_SHMEM_SIZE)   = (uint32_t)shmem_size;
            REG32(t->bar0, REG_SHMEM_CTRL)   = 1;  /* register */

            time_t start = time(NULL);
            uint32_t st;
            while (1) {
                st = REG32(t->bar0, REG_STATUS);
                if (st == STATUS_DONE || st == STATUS_ERROR) break;
                if (time(NULL) - start >= 5) { st = 0xFF; break; }
                usleep(1000);
            }

            if (st == STATUS_DONE) {
                t->shmem      = shmem;
                t->shmem_alloc_size = shmem_alloc_size;
                t->shmem_size = shmem_size;
                t->shmem_half = shmem_size / 2;
                t->shmem_g2h  = shmem_window;
                t->shmem_h2g  = (char *)shmem_window + shmem_size / 2;
                t->has_shmem  = 1;
                t->shmem_registered_gpa = gpa;
                t->shmem_owner_lock_fd = shmem_owner_lock_fd;
                shmem_registration_trace("register", t, gpa, shmem_size,
                                         t->shmem_g2h, t->shmem_h2g);

                if (vgpu_debug_logging()) {
                    fprintf(stderr, "[cuda-transport] Shared-memory registered: "
                            "gpa=0x%016llx size=%zu MB (mapped=%zu MB window_off=%zu MB "
                            "G2H=%zu MB H2G=%zu MB, attempt=%d map32=%d)\n",
                            (unsigned long long)gpa,
                            shmem_size >> 20, shmem_alloc_size >> 20,
                            ((size_t)((char *)shmem_window - (char *)shmem)) >> 20,
                            (shmem_size / 2) >> 20, (shmem_size / 2) >> 20,
                            attempt + 1, use_map32);
                }
                return 1;
            }

            {
                uint32_t err = REG32(t->bar0, REG_ERROR_CODE);
                fprintf(stderr, "[cuda-transport] vgpu-stub rejected shmem registration "
                        "(gpa=0x%016llx size=%zu MB attempt=%d map32=%d status=0x%02x err=0x%08x:%s)\n",
                        (unsigned long long)gpa, shmem_size >> 20, attempt + 1, use_map32,
                        st, err, vgpu_err_to_str(err));
                REG32(t->bar0, REG_SHMEM_CTRL) = 0;
                munlock(shmem, shmem_alloc_size);
                munmap(shmem, shmem_alloc_size);

                if (err != VGPU_ERR_INVALID_REQUEST) {
                    fprintf(stderr, "[cuda-transport] Non-retryable shmem registration failure "
                            "— using BAR1\n");
                    release_shmem_owner_lock(shmem_owner_lock_fd);
                    return 0;
                }
            }
        }
    }

    release_shmem_owner_lock(shmem_owner_lock_fd);
    fprintf(stderr, "[cuda-transport] Exhausted shmem registration retries — using BAR1\n");
    return 0;
}

static const char *cuda_transport_data_path_name(cuda_transport_t *tp)
{
    if (tp->has_shmem) return "shmem";
    if (tp->has_bar1) return "BAR1";
    return "BAR0-inline";
}

static const char *cuda_transport_status_path_name(cuda_transport_t *tp)
{
    if (tp->has_bar1) return "BAR1-status-mirror";
    return "BAR0";
}

/* ================================================================
 * Initialise transport
 * ================================================================ */
int cuda_transport_init(cuda_transport_t **tp)
{
    if (vgpu_debug_logging())
        (void)syscall(__NR_write, 2,
                      "[cuda-transport] FORCE: cuda_transport_init() STARTED\n",
                      sizeof("[cuda-transport] FORCE: cuda_transport_init() STARTED\n") - 1);
    char res0_path[512], res1_path[512];
    char pci_bdf[64] = {0};
    cuda_transport_t *t;

    /* CRITICAL: Disable PCI file interception BEFORE calling find_vgpu_device()
     * This ensures we read real values from /sys, not intercepted values
     * Same fix as in cuda_transport_discover() */
    call_libvgpu_set_skip_interception(1);
    if (vgpu_debug_logging()) {
        (void)syscall(__NR_write, 2,
                      "[cuda-transport] FORCE: About to set skip flag to 1\n",
                      sizeof("[cuda-transport] FORCE: About to set skip flag to 1\n") - 1);
        (void)syscall(__NR_write, 2,
                      "[cuda-transport] FORCE: Skip flag SET to 1 (pid=",
                      sizeof("[cuda-transport] FORCE: Skip flag SET to 1 (pid=") - 1);
        char pid_str[32];
        snprintf(pid_str, sizeof(pid_str), "%d)\n", (int)getpid());
        (void)syscall(__NR_write, 2, pid_str, strlen(pid_str));
        fprintf(stderr, "[cuda-transport] DEBUG: cuda_transport_init() - Skip flag SET to 1 (pid=%d)\n", (int)getpid());
        fflush(stderr);
    }

    cuda_transport_write_checkpoint("INIT_START");
    if (find_vgpu_device(res0_path, sizeof(res0_path),
                         res1_path, sizeof(res1_path),
                         pci_bdf,   sizeof(pci_bdf)) != 0) {
        cuda_transport_write_error("DEVICE_NOT_FOUND", 0, 0,
            "VGPU-STUB not in /sys/bus/pci/devices");
        fprintf(stderr, "[cuda-transport] VGPU-STUB device not found\n");
        /* Re-enable interception after discovery */
        call_libvgpu_set_skip_interception(0);
        return -1;
    }
    cuda_transport_write_checkpoint("DEVICE_FOUND");

    t = (cuda_transport_t *)calloc(1, sizeof(cuda_transport_t));
    if (!t) return -1;

    t->bar0_fd  = -1;
    t->bar1_fd  = -1;
    t->has_bar1 = 0;
    t->has_shmem = 0;
    t->shmem_owner_lock_fd = -1;
    t->seq_counter = 1;
    /* Use snprintf to avoid truncation warning */
    snprintf(t->pci_bdf, sizeof(t->pci_bdf), "%.*s", (int)(sizeof(t->pci_bdf) - 1), pci_bdf);

    /* Map BAR0 (always required) */
    t->bar0_fd = open(res0_path, O_RDWR | O_SYNC);
    if (t->bar0_fd < 0) {
        char errdetail[640];
        (void)snprintf(errdetail, sizeof(errdetail), "open(%s) %s", res0_path, strerror(errno));
        cuda_transport_write_error("BAR0_OPEN_FAILED", 0, 0, errdetail);
        fprintf(stderr, "[cuda-transport] Cannot open BAR0: %s (%s)\n",
                res0_path, strerror(errno));
        free(t);
        return -1;
    }

    cuda_transport_write_checkpoint("BAR0_OPENED");
    t->bar0 = mmap(NULL, BAR0_SIZE, PROT_READ | PROT_WRITE,
                    MAP_SHARED, t->bar0_fd, 0);
    if (t->bar0 == MAP_FAILED) {
        char errdetail[64];
        snprintf(errdetail, sizeof(errdetail), "mmap BAR0 %s", strerror(errno));
        cuda_transport_write_error("BAR0_MMAP_FAILED", 0, 0, errdetail);
        fprintf(stderr, "[cuda-transport] Cannot mmap BAR0: %s\n",
                strerror(errno));
        close(t->bar0_fd);
        free(t);
        return -1;
    }

    t->vm_id = REG32(t->bar0, REG_VM_ID);

    /* --- Preferred path: VHOST-style shared memory --- */
    fprintf(stderr,
            "[cuda-transport] build-config shmem_default_mb=%zu shmem_min_mb=%zu\n",
            (size_t)(SHMEM_DEFAULT_SIZE >> 20),
            (size_t)(SHMEM_MIN_SIZE >> 20));
    {
        int cfgfd = open("/var/tmp/vgpu_transport_build_cfg.log",
                         O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (cfgfd >= 0) {
            char cfgbuf[128];
            int cfglen = snprintf(cfgbuf, sizeof(cfgbuf),
                                  "shmem_default_mb=%zu shmem_min_mb=%zu\n",
                                  (size_t)(SHMEM_DEFAULT_SIZE >> 20),
                                  (size_t)(SHMEM_MIN_SIZE >> 20));
            if (cfglen > 0) {
                ssize_t wrote = write(cfgfd, cfgbuf, (size_t)cfglen);
                (void)wrote;
            }
            close(cfgfd);
        }
    }
    if (!setup_shmem(t)) {
        /* --- Fallback: map BAR1 (legacy data region) --- */
        t->bar1_fd = open(res1_path, O_RDWR | O_SYNC);
        if (t->bar1_fd >= 0) {
            t->bar1 = mmap(NULL, BAR1_SIZE, PROT_READ | PROT_WRITE,
                            MAP_SHARED, t->bar1_fd, 0);
            if (t->bar1 != MAP_FAILED) {
                t->has_bar1 = 1;
                if (vgpu_debug_logging())
                    fprintf(stderr, "[cuda-transport] BAR1 mapped "
                            "(16 MB legacy data region)\n");
            } else {
                t->bar1 = NULL;
                close(t->bar1_fd);
                t->bar1_fd = -1;
            }
        }
    }
    /* --- Always map BAR1 for status mirror (avoids broken BAR0 status path) --- */
    if (!t->has_bar1) {
        t->bar1_fd = open(res1_path, O_RDWR | O_SYNC);
        if (t->bar1_fd >= 0) {
            t->bar1 = mmap(NULL, BAR1_SIZE, PROT_READ | PROT_WRITE,
                            MAP_SHARED, t->bar1_fd, 0);
            if (t->bar1 != MAP_FAILED) {
                t->has_bar1 = 1;
                fprintf(stderr, "[cuda-transport] BAR1 mapped for status mirror\n");
            } else {
                t->bar1 = NULL;
                close(t->bar1_fd);
                t->bar1_fd = -1;
                fprintf(stderr, "[cuda-transport] BAR1 mmap failed - status will be read from BAR0\n");
            }
        } else {
            fprintf(stderr, "[cuda-transport] BAR1 open failed (errno=%d) - status will be read from BAR0\n", errno);
        }
    }

    /* Re-enable interception after successful discovery */
    call_libvgpu_set_skip_interception(0);
    cuda_transport_write_checkpoint("TRANSPORT_READY");
    fprintf(stderr, "[cuda-transport] Connected (vm_id=%u) data_path=%s status_from=%s\n",
            t->vm_id,
            cuda_transport_data_path_name(t),
            cuda_transport_status_path_name(t));
    if (vgpu_debug_logging())
        fprintf(stderr, "[cuda-transport] (debug logging on)\n");
    *tp = t;
    return 0;
}

/* ================================================================
 * Destroy transport
 * ================================================================ */
void cuda_transport_destroy(cuda_transport_t *tp)
{
    if (!tp) return;

    /* Release shared memory */
    if (tp->has_shmem && tp->shmem) {
        shmem_registration_trace("unregister", tp, tp->shmem_registered_gpa,
                                 tp->shmem_size, tp->shmem_g2h, tp->shmem_h2g);
        REG32(tp->bar0, REG_SHMEM_CTRL) = 0;  /* unregister */
        munlock(tp->shmem, tp->shmem_alloc_size);
        munmap(tp->shmem, tp->shmem_alloc_size);
        tp->shmem    = NULL;
        tp->shmem_alloc_size = 0;
        tp->has_shmem = 0;
    }
    release_shmem_owner_lock(tp->shmem_owner_lock_fd);
    tp->shmem_owner_lock_fd = -1;

    if (tp->bar0 && tp->bar0 != MAP_FAILED)
        munmap((void *)tp->bar0, BAR0_SIZE);
    if (tp->bar1 && tp->bar1 != MAP_FAILED)
        munmap((void *)tp->bar1, BAR1_SIZE);
    if (tp->bar0_fd >= 0) close(tp->bar0_fd);
    if (tp->bar1_fd >= 0) close(tp->bar1_fd);
    free(tp);
}

/* ================================================================
 * Write bulk data to the active data region.
 *
 * Shared-memory path: plain memcpy into the G2H half — no VM exit.
 * BAR1 fallback:      memcpy into MMIO window (generates VM exits).
 * BAR0 inline:        for small data when neither shmem nor BAR1 exists.
 *
 * len MUST be <= the active window size (caller's responsibility).
 * ================================================================ */

/* MAP_SHARED anonymous G2H: push dirty pages so the host (QEMU cpu_physical_memory_map
 * of the registered GPA) observes memmove() before the CUDA doorbell. */
static void msync_shmem_g2h_range(void *addr, size_t len)
{
    if (!addr || len == 0)
        return;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0)
        ps = 4096;
    uintptr_t a = (uintptr_t)addr;
    uintptr_t start = a & ~((uintptr_t)ps - 1u);
    uintptr_t end = a + len;
    if (end < start)
        return;
    size_t mlen = (size_t)(end - start);
    (void)msync((void *)start, mlen, MS_SYNC);
    __sync_synchronize();
}

static void write_bar1_data_words_mode(cuda_transport_t *tp,
                                       const void *data, uint32_t len,
                                       int use_u64_writes)
{
    const uint8_t *src = (const uint8_t *)data;
    uint32_t off = 0;

    /* Prefer mmap + MMIO stores so QEMU's vgpu_bar1_write() updates the stub's
     * bar1_data backing store. pwrite(resource1) can update the BAR without
     * going through the emulated MMIO path — mediator then reads zeros from the
     * stub (module-chunk first8=0, INVALID_IMAGE). */
    if (tp->bar1 && tp->bar1 != MAP_FAILED) {
        if (use_u64_writes) {
            while (off + 8u <= len) {
                uint64_t qword = 0;
                memcpy(&qword, src + off, sizeof(qword));
                *(volatile uint64_t *)((volatile char *)tp->bar1 +
                                       BAR1_GUEST_TO_HOST_OFFSET + off) = qword;
                off += 8u;
            }
        }
        while (off < len) {
            uint32_t word = 0;
            uint32_t chunk = len - off;
            if (chunk > 4) chunk = 4;
            memcpy(&word, src + off, chunk);
            *(volatile uint32_t *)((volatile char *)tp->bar1 +
                                   BAR1_GUEST_TO_HOST_OFFSET + off) = word;
            off += 4;
        }
        __sync_synchronize();
        return;
    }

    if (tp->bar1_fd >= 0) {
        static int pwrite_warn_once;
        if (!pwrite_warn_once) {
            pwrite_warn_once = 1;
            fprintf(stderr,
                    "[cuda-transport] WARN: BAR1 G2H via pwrite (mmap missing); "
                    "QEMU vgpu_bar1_write MMIO counter may stay 0\n");
        }
        while (off < len) {
            ssize_t n = pwrite(tp->bar1_fd, src + off, len - off,
                               (off_t)(BAR1_GUEST_TO_HOST_OFFSET + off));
            if (n <= 0) {
                break;
            }
            off += (uint32_t)n;
        }
    }
}

static void write_bar1_data_words(cuda_transport_t *tp,
                                  const void *data, uint32_t len)
{
    write_bar1_data_words_mode(tp, data, len, 0);
}

static int cuda_transport_use_bar1_for_htod(cuda_transport_t *tp,
                                            uint32_t call_id,
                                            uint32_t len)
{
    return tp && tp->has_bar1 &&
           len > CUDA_SMALL_DATA_MAX &&
           (call_id == CUDA_CALL_MEMCPY_HTOD ||
            call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC);
}

static int env_not_zero(const char *name)
{
    const char *e = getenv(name);
    return e && e[0] && strcmp(e, "0") != 0;
}

/* Master: when set, large HtoD and module/fatbin bulk both use BAR1 (see also below). */
static int bulk_all_bar1_from_env(void)
{
    return env_not_zero("VGPU_BULK_BAR1") ? 1 : 0;
}

/* When set (non-empty, not "0"), large HtoD uses BAR1 MMIO even if shmem is
 * registered. Also implied by VGPU_BULK_BAR1. */
static int htod_env_force_bar1(void)
{
    return (bulk_all_bar1_from_env() || env_not_zero("VGPU_HTOD_BAR1")) ? 1 : 0;
}

/* When set, MODULE_LOAD_* / FAT_BINARY bulk uses BAR1 instead of shmem memmove.
 * Also implied by VGPU_BULK_BAR1. Mediator INVALID_IMAGE often tracks shmem zeros. */
static int module_env_force_bar1(void)
{
    return (bulk_all_bar1_from_env() || env_not_zero("VGPU_MODULE_BAR1")) ? 1 : 0;
}

/* Default-on correctness backstop for HtoD BAR1 shadowing.
 * Set VGPU_HTOD_BAR1_SHADOW=0 to A/B the clean SHMEM path without the extra
 * BAR1 mirror when host fallback is no longer expected. */
static int htod_bar1_shadow_enabled(void)
{
    static int cached = -1;
    static int enabled = 1;
    if (cached < 0) {
        const char *e = getenv("VGPU_HTOD_BAR1_SHADOW");
        enabled = !(e && (!e[0] || strcmp(e, "0") == 0));
        cached = 1;
    }
    return enabled;
}

/* Optional A/B for large cuLibraryLoadData BAR1 shadow cost.
 * Default is unlimited (preserve current behavior). When set to a positive byte
 * count, library-load BAR1 shadowing is skipped only for payloads larger than
 * that threshold, while HtoD shadowing and smaller module/library shadows keep
 * the current semantics. */
static uint32_t library_bar1_shadow_max_bytes(void)
{
    static int cached = -1;
    static uint32_t value = 0;
    if (cached < 0) {
        const char *e = getenv("VGPU_LIBRARY_BAR1_SHADOW_MAX_BYTES");
        value = UINT32_MAX;
        if (e && e[0]) {
            char *end = NULL;
            errno = 0;
            unsigned long long parsed = strtoull(e, &end, 10);
            if (errno == 0 && end && *end == '\0' &&
                parsed > 0ull && parsed <= (unsigned long long)UINT32_MAX) {
                value = (uint32_t)parsed;
            }
        }
        cached = 1;
    }
    return value;
}

static int bulk_shadow_bar1_enabled(cuda_transport_t *tp,
                                    uint32_t call_id,
                                    uint32_t len,
                                    int is_htod_bulk,
                                    int is_mod_bulk)
{
    if (!tp || !tp->has_bar1 || len > BAR1_GUEST_TO_HOST_SIZE) {
        return 0;
    }
    if (is_htod_bulk) {
        return htod_bar1_shadow_enabled();
    }
    if (!is_mod_bulk) {
        return 0;
    }
    if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA) {
        if (tp->has_shmem && tp->shmem_half > 0 &&
            len <= (uint32_t)tp->shmem_half) {
            return 0;
        }
        uint32_t max_bytes = library_bar1_shadow_max_bytes();
        if (len > max_bytes) {
            return 0;
        }
    }
    return 1;
}

typedef enum bulk_primary_path_e {
    BULK_PRIMARY_BAR0 = 0,
    BULK_PRIMARY_SHMEM,
    BULK_PRIMARY_BAR1,
} bulk_primary_path_t;

static bulk_primary_path_t bulk_primary_path_for_call(cuda_transport_t *tp,
                                                      uint32_t call_id,
                                                      uint32_t len)
{
    const int is_htod_bulk = (call_id == CUDA_CALL_MEMCPY_HTOD ||
                              call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC);
    const int is_mod_bulk = (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
                             call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
                             call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY ||
                             call_id == CUDA_CALL_LIBRARY_LOAD_DATA);
    const int force_htod_bar1 = htod_env_force_bar1();
    const int force_mod_bar1 = module_env_force_bar1();

    if (tp->has_shmem && len > CUDA_SMALL_DATA_MAX &&
        ((is_mod_bulk && !force_mod_bar1) ||
         (is_htod_bulk && !force_htod_bar1))) {
        return BULK_PRIMARY_SHMEM;
    }
    if (cuda_transport_use_bar1_for_htod(tp, call_id, len)) {
        return BULK_PRIMARY_BAR1;
    }
    if (force_mod_bar1 && tp->has_bar1 && len > CUDA_SMALL_DATA_MAX &&
        is_mod_bulk) {
        return BULK_PRIMARY_BAR1;
    }
    if (tp->has_bar1 && len > CUDA_SMALL_DATA_MAX && is_mod_bulk &&
        !tp->has_shmem) {
        return BULK_PRIMARY_BAR1;
    }
    if (tp->has_shmem && len > CUDA_SMALL_DATA_MAX) {
        return BULK_PRIMARY_SHMEM;
    }
    if (tp->has_bar1 && len > CUDA_SMALL_DATA_MAX) {
        return BULK_PRIMARY_BAR1;
    }
    return BULK_PRIMARY_BAR0;
}

static const char *bulk_primary_path_name(cuda_transport_t *tp,
                                          uint32_t call_id,
                                          uint32_t len)
{
    switch (bulk_primary_path_for_call(tp, call_id, len)) {
    case BULK_PRIMARY_SHMEM:
        return "shmem";
    case BULK_PRIMARY_BAR1:
        return "bar1";
    default:
        return "bar0";
    }
}

static const uint8_t *bulk_primary_written_ptr(cuda_transport_t *tp,
                                               uint32_t call_id,
                                               uint32_t len)
{
    switch (bulk_primary_path_for_call(tp, call_id, len)) {
    case BULK_PRIMARY_SHMEM:
        return (const uint8_t *)tp->shmem_g2h;
    case BULK_PRIMARY_BAR1:
        return (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
    default:
        return (const uint8_t *)tp->bar0 + CUDA_REQ_DATA_OFFSET;
    }
}

static int library_bar1_u64_writes_enabled(void)
{
    return env_not_zero("VGPU_LIBRARY_BAR1_U64_WRITES") ? 1 : 0;
}

static void write_bar1_data_u64_words(cuda_transport_t *tp,
                                      const void *data, uint32_t len)
{
    const uint8_t *src = (const uint8_t *)data;
    uint32_t off = 0;

    if (tp->bar1 && tp->bar1 != MAP_FAILED) {
        while (off + 8u <= len) {
            uint64_t qword = 0;
            memcpy(&qword, src + off, sizeof(qword));
            *(volatile uint64_t *)((volatile char *)tp->bar1 +
                                   BAR1_GUEST_TO_HOST_OFFSET + off) = qword;
            off += 8u;
        }
        while (off < len) {
            uint32_t word = 0;
            uint32_t chunk = len - off;
            if (chunk > 4) {
                chunk = 4;
            }
            memcpy(&word, src + off, chunk);
            *(volatile uint32_t *)((volatile char *)tp->bar1 +
                                   BAR1_GUEST_TO_HOST_OFFSET + off) = word;
            off += 4u;
        }
        __sync_synchronize();
        return;
    }

    write_bar1_data_words(tp, data, len);
}

/* After memcpy into shmem_g2h: log src vs destination bytes. If src/shmem_first8
 * are non-zero here but mediator sees zeros, the bug is GPA mapping or QEMU read.
 * First 12 large chunks only (>= 4 KiB) — avoids unbounded logs; no env required. */
static void bulk_guest_payload_trace(uint32_t call_id, uint32_t len,
                                    const void *src, const void *dst_shmem)
{
    static unsigned traces_left = 12u;
    if (traces_left == 0u || !src || !dst_shmem || len < 4096u)
        return;
    traces_left--;
    const uint8_t *s = (const uint8_t *)src;
    const uint8_t *d = (const uint8_t *)dst_shmem;
    char buf[384];
    int n = snprintf(buf, sizeof(buf),
                     "[cuda-transport] BULK_GUEST call_id=0x%04x len=%u "
                     "src_first8=%02x%02x%02x%02x%02x%02x%02x%02x "
                     "shmem_first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                     call_id, len,
                     s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
                     d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
    if (n <= 0)
        return;
    (void)fwrite(buf, 1, (size_t)n, stderr);
    fflush(stderr);
    /* /dev/shm is writable under typical Ollama PrivateTmp; /var/tmp often is not. */
    int fd = (int)syscall(__NR_openat, -100, "/dev/shm/vgpu_bulk_guest.log",
                          O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd >= 0) {
        (void)syscall(__NR_write, fd, buf, (size_t)n);
        (void)syscall(__NR_close, fd);
    }
}

/* Immediate post-memmove: pointer compare, memcmp, pagemap GPA vs stub registration. */
static void diag_shmem_after_memmove(cuda_transport_t *tp, uint32_t call_id,
                                     const void *data, uint32_t len)
{
    static pid_t diag_pid;
    static unsigned diag_done;
    pid_t p = getpid();
    if (diag_pid != p) {
        diag_pid = p;
        diag_done = 0;
    }
    if (diag_done >= 8u || !tp || !tp->has_shmem || !tp->shmem_g2h || len < 8u)
        return;
    diag_done++;

    int same_ptr = (data == (const void *)tp->shmem_g2h);
    size_t ncmp = len < 64u ? (size_t)len : 64u;
    int diff = memcmp(data, tp->shmem_g2h, ncmp);

    uint64_t pme = 0;
    int pmst = pagemap_entry_for(tp->shmem_g2h, &pme);
    uint64_t pagemap_gpa = 0;
    if (pmst == VGPU_PAGEMAP_OK) {
        uint64_t pfn = pme & 0x007FFFFFFFFFFFFFULL;
        uint64_t ps = (uint64_t)sysconf(_SC_PAGESIZE);
        pagemap_gpa = pfn * ps;
    }

    char line[512];
    int nw = snprintf(line, sizeof(line),
                      "[cuda-transport] DIAG_POST_MOVE call_id=0x%04x len=%u "
                      "same_ptr=%d memcmp64=%d data=%p g2h=%p "
                      "reg_gpa=0x%016llx pagemap_st=%s pagemap_gpa=0x%016llx "
                      "volatile_g2h0=%02x\n",
                      call_id, len, same_ptr, diff, data, (void *)tp->shmem_g2h,
                      (unsigned long long)tp->shmem_registered_gpa,
                      vgpu_pagemap_probe_status_str(pmst),
                      (unsigned long long)pagemap_gpa,
                      *(volatile const uint8_t *)tp->shmem_g2h);
    if (nw > 0) {
        (void)fwrite(line, 1, (size_t)nw, stderr);
        fflush(stderr);
        int fd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_transport.log",
                              O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            (void)syscall(__NR_write, fd, line, (size_t)nw);
            (void)syscall(__NR_close, fd);
        }
    }
}

static void library_write_site_trace(const char *stage,
                                     cuda_transport_t *tp,
                                     const void *src,
                                     uint32_t len,
                                     const char *branch)
{
    if (!stage || !tp || !tp->shmem_g2h || !src || len == 0)
        return;

    const uint8_t *s = (const uint8_t *)src;
    const uint8_t *d = (const uint8_t *)tp->shmem_g2h;
    char buf[512];
    int n = snprintf(buf, sizeof(buf),
                     "LIBRARY_WRITE_SITE stage=%s branch=%s pid=%d len=%u "
                     "src=%p g2h=%p src_first8=%02x%02x%02x%02x%02x%02x%02x%02x "
                     "g2h_first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                     stage, branch ? branch : "unknown", (int)getpid(), len,
                     src, tp->shmem_g2h,
                     s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
                     d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
    if (n <= 0)
        return;

    int fd = (int)syscall(__NR_openat, -100,
                          "/var/tmp/vgpu_library_load_fingerprint.log",
                          O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd >= 0) {
        (void)syscall(__NR_write, fd, buf, (size_t)n);
        (void)syscall(__NR_close, fd);
    }
}

static void library_transport_stage_trace(const char *stage,
                                          cuda_transport_t *tp,
                                          const void *src,
                                          uint32_t len)
{
    if (!stage || !tp || !tp->shmem_g2h || !src || len < 8u) {
        return;
    }

    const uint8_t *s = (const uint8_t *)src;
    const uint8_t *g = (const uint8_t *)tp->shmem_g2h;
    const uint8_t *b = NULL;
    if (tp->bar1 && tp->bar1 != MAP_FAILED &&
        len <= BAR1_GUEST_TO_HOST_SIZE) {
        b = (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
    }

    char line[512];
    int n = snprintf(line, sizeof(line),
                     "[cuda-transport] LIBLOAD_STAGE stage=%s len=%u "
                     "src_first8=%02x%02x%02x%02x%02x%02x%02x%02x "
                     "shmem_first8=%02x%02x%02x%02x%02x%02x%02x%02x "
                     "bar1_first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                     stage, len,
                     s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
                     g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
                     b ? b[0] : 0, b ? b[1] : 0, b ? b[2] : 0, b ? b[3] : 0,
                     b ? b[4] : 0, b ? b[5] : 0, b ? b[6] : 0, b ? b[7] : 0);
    if (n <= 0) {
        return;
    }

    int fd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_transport.log",
                          O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        (void)syscall(__NR_write, fd, line, (size_t)n);
        (void)syscall(__NR_close, fd);
    }
}

static void library_chunk_trace(uint32_t seq,
                                uint32_t offset,
                                uint32_t chunk_len,
                                uint32_t total_len,
                                uint32_t flags,
                                const void *src)
{
    if (!src || chunk_len == 0) {
        return;
    }

    const uint8_t *s = (const uint8_t *)src;
    const uint8_t *tail = s + ((chunk_len >= 8u) ? (chunk_len - 8u) : 0u);
    char line[512];
    int n = snprintf(line, sizeof(line),
                     "LIBRARY_CHUNK seq=%u offset=%u chunk=%u total=%u flags=0x%x "
                     "head8=%02x%02x%02x%02x%02x%02x%02x%02x "
                     "tail8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                     seq, offset, chunk_len, total_len, flags,
                     s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
                     tail[0], tail[1], tail[2], tail[3],
                     tail[4], tail[5], tail[6], tail[7]);
    if (n <= 0) {
        return;
    }

    int fd = (int)syscall(__NR_openat, -100,
                          "/var/tmp/vgpu_library_load_fingerprint.log",
                          O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd >= 0) {
        (void)syscall(__NR_write, fd, line, (size_t)n);
        (void)syscall(__NR_close, fd);
    }
}

static void htod_bar1_shadow_trace(cuda_transport_t *tp,
                                   uint32_t seq,
                                   uint32_t len,
                                   const char *branch)
{
    if (!tp || !branch || len == 0) {
        return;
    }

    const uint8_t *b = NULL;
    if (tp->bar1 && tp->bar1 != MAP_FAILED &&
        len <= BAR1_GUEST_TO_HOST_SIZE) {
        b = (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
    }

    char line[384];
    int n = snprintf(line, sizeof(line),
                     "[cuda-transport] HTOD_BAR1_SHADOW seq=%u len=%u branch=%s "
                     "has_bar1=%d bar1_ptr=%p first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                     seq, len, branch, tp->has_bar1 ? 1 : 0,
                     (const void *)b,
                     b ? b[0] : 0, b ? b[1] : 0, b ? b[2] : 0, b ? b[3] : 0,
                     b ? b[4] : 0, b ? b[5] : 0, b ? b[6] : 0, b ? b[7] : 0);
    if (n <= 0) {
        return;
    }

    int fd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_transport.log",
                          O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        (void)syscall(__NR_write, fd, line, (size_t)n);
        (void)syscall(__NR_close, fd);
    }
}

static void htod_bar1_shadow_decision_trace(cuda_transport_t *tp,
                                            uint32_t call_id,
                                            uint32_t seq,
                                            uint32_t len,
                                            const char *branch,
                                            int shadow_enabled)
{
    if (!tp || !branch) {
        return;
    }

    const uint8_t *b = NULL;
    if (tp->bar1 && tp->bar1 != MAP_FAILED &&
        len <= BAR1_GUEST_TO_HOST_SIZE) {
        b = (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
    }

    char line[416];
    int n = snprintf(line, sizeof(line),
                     "[cuda-transport] HTOD_BAR1_DECISION seq=%u call_id=0x%04x "
                     "len=%u branch=%s has_bar1=%d over_limit=%d enabled=%d "
                     "bar1_ptr=%p first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                     seq, call_id, len, branch, tp->has_bar1 ? 1 : 0,
                     len > BAR1_GUEST_TO_HOST_SIZE ? 1 : 0, shadow_enabled ? 1 : 0,
                     (const void *)b,
                     b ? b[0] : 0, b ? b[1] : 0, b ? b[2] : 0, b ? b[3] : 0,
                     b ? b[4] : 0, b ? b[5] : 0, b ? b[6] : 0, b ? b[7] : 0);
    if (n <= 0) {
        return;
    }

    int fd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_transport.log",
                          O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        (void)syscall(__NR_write, fd, line, (size_t)n);
        (void)syscall(__NR_close, fd);
    }
}

static uint64_t transport_fnv1a64(const void *data, uint32_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ull;
    uint32_t i;

    if (!p || len == 0) {
        return h;
    }
    for (i = 0; i < len; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t monotonic_ns_now(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void library_timing_trace(const char *stage,
                                 uint32_t seq,
                                 uint32_t len,
                                 uint64_t elapsed_ns,
                                 const char *detail)
{
    if (!stage) {
        return;
    }

    char line[320];
    unsigned long long us = (unsigned long long)((elapsed_ns + 500ull) / 1000ull);
    int n;
    if (detail && detail[0]) {
        n = snprintf(line, sizeof(line),
                     "LIBLOAD_TIMING stage=%s seq=%u len=%u us=%llu %s\n",
                     stage, seq, len, us, detail);
    } else {
        n = snprintf(line, sizeof(line),
                     "LIBLOAD_TIMING stage=%s seq=%u len=%u us=%llu\n",
                     stage, seq, len, us);
    }
    if (n <= 0) {
        return;
    }

    int fd = (int)syscall(__NR_openat, -100,
                          "/var/tmp/vgpu_library_load_timing.log",
                          O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd >= 0) {
        (void)syscall(__NR_write, fd, line, (size_t)n);
        (void)syscall(__NR_close, fd);
    }
}

static void write_bulk_data(cuda_transport_t *tp,
                            uint32_t call_id,
                            uint32_t seq,
                            const void *data, uint32_t len)
{
    static pid_t wb_pid;
    static int wb_logged;
    pid_t cur = getpid();
    if (wb_pid != cur) {
        wb_pid = cur;
        wb_logged = 0;
    }
    if (!wb_logged) {
        wb_logged = 1;
        char eb[192];
        int en = snprintf(eb, sizeof(eb),
                          "write_bulk_enter pid=%d call_id=0x%04x len=%u data=%p has_shmem=%d\n",
                          (int)cur, call_id, len, data, tp->has_shmem ? 1 : 0);
        if (en > 0) {
            int efd = (int)syscall(__NR_openat, -100,
                                   "/var/tmp/vgpu_htod_transport.log",
                                   O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (efd >= 0) {
                (void)syscall(__NR_write, efd, eb, (size_t)en);
                (void)syscall(__NR_close, efd);
            }
        }
    }

    if (len == 0 || !data) return;

    const int is_htod_bulk = (call_id == CUDA_CALL_MEMCPY_HTOD ||
                              call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC);
    const int is_mod_bulk = (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
                             call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
                             call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY ||
                             call_id == CUDA_CALL_LIBRARY_LOAD_DATA);
    const int force_htod_bar1 = htod_env_force_bar1();
    const int force_mod_bar1 = module_env_force_bar1();
    const int trace_large_bulk = (len > CUDA_SMALL_DATA_MAX) &&
                                 (is_htod_bulk || is_mod_bulk);
    const int trace_library_timing =
        (call_id == CUDA_CALL_LIBRARY_LOAD_DATA && len > CUDA_SMALL_DATA_MAX);

#define TRACE_BULK_BRANCH(BRANCH_NAME)                                                   \
    do {                                                                                 \
        if (trace_large_bulk) {                                                          \
            const uint8_t *src8 = (const uint8_t *)data;                                 \
            char line[320];                                                              \
            int nw = snprintf(line, sizeof(line),                                        \
                              "[cuda-transport] BULK_BRANCH call_id=0x%04x len=%u "      \
                              "branch=%s has_shmem=%d has_bar1=%d "                      \
                              "force_htod_bar1=%d force_mod_bar1=%d "                    \
                              "src_first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",          \
                              call_id, len, BRANCH_NAME,                                 \
                              tp->has_shmem ? 1 : 0, tp->has_bar1 ? 1 : 0,              \
                              force_htod_bar1, force_mod_bar1,                           \
                              len > 0 ? src8[0] : 0, len > 1 ? src8[1] : 0,             \
                              len > 2 ? src8[2] : 0, len > 3 ? src8[3] : 0,             \
                              len > 4 ? src8[4] : 0, len > 5 ? src8[5] : 0,             \
                              len > 6 ? src8[6] : 0, len > 7 ? src8[7] : 0);            \
            if (nw > 0) {                                                                \
                int fd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_transport.log", \
                                      O_WRONLY | O_CREAT | O_APPEND, 0644);              \
                if (fd >= 0) {                                                           \
                    (void)syscall(__NR_write, fd, line, (size_t)nw);                     \
                    (void)syscall(__NR_close, fd);                                       \
                }                                                                        \
            }                                                                            \
        }                                                                                \
    } while (0)

    /* Large HtoD + large module images: when shmem is active, memcpy to G2H so
     * the stub reads s->shmem_g2h (same half as guest RAM). BAR1-only module
     * while shmem was registered left the stub preferring BAR1 and the guest
     * here using BAR1 — mediator still saw zeros on some stacks; shmem matches
     * HtoD and the stub's copy_from_fresh_shmem path. */
    if (tp->has_shmem && len > CUDA_SMALL_DATA_MAX &&
        ((is_mod_bulk && !force_mod_bar1) ||
         (is_htod_bulk && !force_htod_bar1))) {
        TRACE_BULK_BRANCH("shmem-preferred");
        if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA)
            library_write_site_trace("pre", tp, data, len, "shmem-preferred");
        /* memmove: host source buffer may overlap shmem_g2h (same mmap or
         * GGML views); memcpy has undefined behavior when regions overlap,
         * which produced all-zero G2H after copy on Ollama/llama.cpp. */
        uint64_t shmem_copy_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
        memmove(tp->shmem_g2h, data, len);
        msync_shmem_g2h_range(tp->shmem_g2h, (size_t)len);
        if (trace_library_timing) {
            library_timing_trace("shmem_copy", seq, len,
                                 monotonic_ns_now() - shmem_copy_start_ns,
                                 "branch=shmem-preferred");
        }
        if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA)
            library_write_site_trace("post", tp, data, len, "shmem-preferred");
        diag_shmem_after_memmove(tp, call_id, data, len);
        if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA)
            library_transport_stage_trace("after_memmove", tp, data, len);
        bulk_guest_payload_trace(call_id, len, data, tp->shmem_g2h);
        /* Current HTOD SHMEM readback can stay zero while the stub falls back
         * to BAR1. Mirror the current HTOD bytes into BAR1 as a correctness
         * backstop so the fallback path does not consume stale payloads. */
        {
            const int shadow_bar1 =
                bulk_shadow_bar1_enabled(tp, call_id, len, is_htod_bulk, is_mod_bulk);
            if (is_htod_bulk) {
                htod_bar1_shadow_decision_trace(tp, call_id, seq, len,
                                               "shmem-preferred", shadow_bar1);
            }
            if (shadow_bar1) {
            const void *bar1_src = is_htod_bulk ? data : tp->shmem_g2h;
            uint64_t bar1_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
            if (trace_library_timing && library_bar1_u64_writes_enabled()) {
                write_bar1_data_u64_words(tp, bar1_src, len);
            } else {
                write_bar1_data_words(tp, bar1_src, len);
            }
            if (trace_library_timing) {
                library_timing_trace("bar1_mirror", seq, len,
                                     monotonic_ns_now() - bar1_start_ns,
                                     library_bar1_u64_writes_enabled()
                                         ? "branch=shmem-preferred mode=u64"
                                         : "branch=shmem-preferred mode=u32");
            }
            if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA)
                library_transport_stage_trace("after_bar1_mirror", tp, data, len);
            if (is_htod_bulk) {
                htod_bar1_shadow_trace(tp, seq, len, "shmem-preferred");
            }
            } else if (trace_library_timing && is_mod_bulk && tp->has_bar1 &&
                       len <= BAR1_GUEST_TO_HOST_SIZE) {
            char detail[96];
            snprintf(detail, sizeof(detail), "branch=shmem-preferred max=%u",
                     (unsigned)library_bar1_shadow_max_bytes());
            library_timing_trace("bar1_mirror_skipped", seq, len, 0, detail);
            } else if (is_mod_bulk && !tp->has_bar1) {
            static int mod_no_bar1_once;
            if (!mod_no_bar1_once) {
                mod_no_bar1_once = 1;
                fprintf(stderr,
                        "[cuda-transport] WARN: module bulk with shmem but BAR1 not "
                        "mapped — no MMIO mirror; dom0 BAR1_MMIO delta stays 0; "
                        "stub uses shmem only\n");
            }
        }
        }
    } else if (cuda_transport_use_bar1_for_htod(tp, call_id, len)) {
        TRACE_BULK_BRANCH("htod-bar1");
        /* BAR1 MMIO when guest has no shmem or small shmem window unavailable */
        write_bar1_data_words(tp, data, len);
    } else if (force_mod_bar1 && tp->has_bar1 && len > CUDA_SMALL_DATA_MAX &&
               is_mod_bulk) {
        TRACE_BULK_BRANCH("module-bar1");
        uint64_t bar1_only_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
        write_bar1_data_words(tp, data, len);
        if (trace_library_timing) {
            library_timing_trace("bar1_only", seq, len,
                                 monotonic_ns_now() - bar1_only_start_ns,
                                 "branch=module-bar1");
        }
    } else if (tp->has_bar1 && len > CUDA_SMALL_DATA_MAX &&
               is_mod_bulk &&
               !tp->has_shmem) {
        TRACE_BULK_BRANCH("module-bar1-no-shmem");
        /* No shmem registered: module bulk must use 32-bit MMIO stores (see
         * write_bar1_data_words header). */
        uint64_t bar1_only_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
        write_bar1_data_words(tp, data, len);
        if (trace_library_timing) {
            library_timing_trace("bar1_only", seq, len,
                                 monotonic_ns_now() - bar1_only_start_ns,
                                 "branch=module-bar1-no-shmem");
        }
    } else if (tp->has_shmem && len > CUDA_SMALL_DATA_MAX) {
        TRACE_BULK_BRANCH("shmem-fallback");
        if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA)
            library_write_site_trace("pre", tp, data, len, "shmem-fallback");
        /* See memmove note above (overlap-safe). */
        uint64_t shmem_copy_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
        memmove(tp->shmem_g2h, data, len);
        msync_shmem_g2h_range(tp->shmem_g2h, (size_t)len);
        if (trace_library_timing) {
            library_timing_trace("shmem_copy", seq, len,
                                 monotonic_ns_now() - shmem_copy_start_ns,
                                 "branch=shmem-fallback");
        }
        if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA)
            library_write_site_trace("post", tp, data, len, "shmem-fallback");
        diag_shmem_after_memmove(tp, call_id, data, len);
        if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA)
            library_transport_stage_trace("after_memmove", tp, data, len);
        bulk_guest_payload_trace(call_id, len, data, tp->shmem_g2h);
        {
            const int shadow_bar1 =
                bulk_shadow_bar1_enabled(tp, call_id, len, is_htod_bulk, is_mod_bulk);
            if (is_htod_bulk) {
                htod_bar1_shadow_decision_trace(tp, call_id, seq, len,
                                               "shmem-fallback", shadow_bar1);
            }
            if (shadow_bar1) {
            const void *bar1_src = is_htod_bulk ? data : tp->shmem_g2h;
            uint64_t bar1_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
            if (trace_library_timing && library_bar1_u64_writes_enabled()) {
                write_bar1_data_u64_words(tp, bar1_src, len);
            } else {
                write_bar1_data_words(tp, bar1_src, len);
            }
            if (trace_library_timing) {
                library_timing_trace("bar1_mirror", seq, len,
                                     monotonic_ns_now() - bar1_start_ns,
                                     library_bar1_u64_writes_enabled()
                                         ? "branch=shmem-fallback mode=u64"
                                         : "branch=shmem-fallback mode=u32");
            }
            if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA)
                library_transport_stage_trace("after_bar1_mirror", tp, data, len);
            if (is_htod_bulk) {
                htod_bar1_shadow_trace(tp, seq, len, "shmem-fallback");
            }
            } else if (trace_library_timing && is_mod_bulk && tp->has_bar1 &&
                       len <= BAR1_GUEST_TO_HOST_SIZE) {
            char detail[96];
            snprintf(detail, sizeof(detail), "branch=shmem-fallback max=%u",
                     (unsigned)library_bar1_shadow_max_bytes());
            library_timing_trace("bar1_mirror_skipped", seq, len, 0, detail);
            }
        }
    } else if (tp->has_bar1 && len > CUDA_SMALL_DATA_MAX) {
        TRACE_BULK_BRANCH("bar1-fallback");
        volatile uint8_t *dst = (volatile uint8_t *)tp->bar1
                                + BAR1_GUEST_TO_HOST_OFFSET;
        memcpy((void *)dst, data, len);
    } else {
        TRACE_BULK_BRANCH("bar0-inline");
        const uint8_t *src = (const uint8_t *)data;
        uint32_t to_copy = (len > CUDA_SMALL_DATA_MAX) ? CUDA_SMALL_DATA_MAX : len;
        uint32_t off = 0;

        /* BAR0 inline data must be written as 32-bit MMIO stores. A plain
         * memcpy() may issue wider stores that the stub does not preserve. */
        while (off < to_copy) {
            uint32_t word = 0;
            uint32_t chunk = to_copy - off;
            if (chunk > 4) chunk = 4;
            memcpy(&word, src + off, chunk);
            REG32(tp->bar0, CUDA_REQ_DATA_OFFSET + off) = word;
            off += 4;
        }
    }

#undef TRACE_BULK_BRANCH
}

/* ================================================================
 * Read bulk response data from the active data region.
 * ================================================================ */
static void read_bulk_data(cuda_transport_t *tp,
                           void *buf, uint32_t len)
{
    if (len == 0 || !buf) return;

    if (tp->has_shmem && len > CUDA_SMALL_DATA_MAX) {
        memcpy(buf, tp->shmem_h2g, len);
    } else if (tp->has_bar1 && len > CUDA_SMALL_DATA_MAX) {
        volatile uint8_t *src = (volatile uint8_t *)tp->bar1
                                + BAR1_HOST_TO_GUEST_OFFSET;
        memcpy(buf, (void *)src, len);
    } else {
        uint8_t *dst = (uint8_t *)buf;
        uint32_t to_copy = (len > CUDA_SMALL_DATA_MAX) ? CUDA_SMALL_DATA_MAX : len;
        uint32_t off = 0;

        while (off < to_copy) {
            uint32_t word = REG32(tp->bar0, CUDA_RESP_DATA_OFFSET + off);
            uint32_t chunk = to_copy - off;
            if (chunk > 4) chunk = 4;
            memcpy(dst + off, &word, chunk);
            off += 4;
        }
    }
}

/* ================================================================
 * Maximum payload per single round-trip.
 * In shmem mode this is the half-window size (128 MB by default).
 * In BAR1 mode this is 8 MB.
 * In inline mode this is 1 KB.
 *
 * When BOTH shmem and BAR1 are mapped, cap at BAR1 G2H size so each chunk
 * fits in the MMIO window. We duplicate the same bytes into shmem_g2h and
 * BAR1 (word stores); the QEMU stub then reads bar1_data reliably. Shmem-only
 * multi-MiB chunks matched stub shmem_g2h reads that still showed all-zero
 * prefixes on the mediator for HtoD/module.
 * ================================================================ */
static uint32_t max_single_payload(cuda_transport_t *tp)
{
    if (tp->has_shmem && tp->has_bar1) {
        uint32_t sh = (uint32_t)tp->shmem_half;
        return sh < BAR1_GUEST_TO_HOST_SIZE ? sh : BAR1_GUEST_TO_HOST_SIZE;
    }
    if (tp->has_shmem) return (uint32_t)tp->shmem_half;
    if (tp->has_bar1)  return BAR1_GUEST_TO_HOST_SIZE;
    return CUDA_SMALL_DATA_MAX;
}

/*
 * Payload (BAR1 / shmem / BAR0 inline) must be visible before we write BAR0
 * metadata: the stub may read combined state when processing the CUDA doorbell.
 * After metadata is written, flush_cuda_metadata_visible() readbacks REG_CUDA_DATA_LEN
 * before the doorbell.
 */
static inline void flush_cuda_payload_writes(cuda_transport_t *tp,
                                             uint32_t call_id,
                                             uint32_t send_len)
{
    __sync_synchronize();
    if (send_len > 0 && send_len <= CUDA_SMALL_DATA_MAX) {
        uint32_t tail_off = CUDA_REQ_DATA_OFFSET + ((send_len - 1u) & ~3u);
        (void)REG32(tp->bar0, tail_off);
    } else if (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX &&
               (call_id == CUDA_CALL_MEMCPY_HTOD ||
                call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC ||
                call_id == CUDA_CALL_MODULE_LOAD_DATA ||
                call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
                call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY ||
                call_id == CUDA_CALL_LIBRARY_LOAD_DATA)) {
        volatile uint8_t *tail =
            (volatile uint8_t *)tp->shmem_g2h + (send_len - 1u);
        (void)*tail;
        if (bulk_shadow_bar1_enabled(tp, call_id, send_len,
                                     call_id == CUDA_CALL_MEMCPY_HTOD ||
                                     call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC,
                                     call_id == CUDA_CALL_MODULE_LOAD_DATA ||
                                     call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
                                     call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY ||
                                     call_id == CUDA_CALL_LIBRARY_LOAD_DATA)) {
            volatile uint32_t *btail =
                (volatile uint32_t *)((volatile char *)tp->bar1 +
                                      BAR1_GUEST_TO_HOST_OFFSET +
                                      ((send_len - 1u) & ~3u));
            (void)*btail;
        }
    } else if (cuda_transport_use_bar1_for_htod(tp, call_id, send_len)) {
        volatile uint32_t *tail =
            (volatile uint32_t *)((volatile char *)tp->bar1 +
                                  BAR1_GUEST_TO_HOST_OFFSET +
                                  ((send_len - 1u) & ~3u));
        (void)*tail;
    } else if (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX) {
        /* Flush the active high-throughput payload path first. When SHMEM is
         * active, large HtoD writes land in shmem_g2h rather than BAR1. */
        volatile uint8_t *tail =
            (volatile uint8_t *)tp->shmem_g2h + (send_len - 1u);
        (void)*tail;
    } else if (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX) {
        volatile uint32_t *tail =
            (volatile uint32_t *)((volatile char *)tp->bar1 +
                                  BAR1_GUEST_TO_HOST_OFFSET +
                                  ((send_len - 1u) & ~3u));
        (void)*tail;
    }
    __sync_synchronize();
}

static inline void flush_cuda_metadata_visible(cuda_transport_t *tp)
{
    (void)REG32(tp->bar0, REG_CUDA_DATA_LEN);
    __sync_synchronize();
}

/* One line per module send: pre = caller buffer, post = transport destination
 * after write_bulk_data (see SYSTEMATIC funnel steps). /var/tmp survives Ollama PrivateTmp. */
static void module_funnel_line(const cuda_transport_t *tp, const char *stage,
                               uint32_t call_id, uint32_t seq, uint32_t send_len,
                               const uint8_t *bytes8)
{
    (void)tp;
    if (!bytes8 || send_len == 0) return;
    char b[256];
    int n = snprintf(b, sizeof(b),
                     "[cuda-transport] FUNNEL %s call_id=0x%04x seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                     stage, call_id, seq, send_len,
                     bytes8[0], bytes8[1], bytes8[2], bytes8[3],
                     bytes8[4], bytes8[5], bytes8[6], bytes8[7]);
    if (n > 0) {
        (void)fwrite(b, 1, (size_t)n, stderr);
        fflush(stderr);
        int fd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_module_funnel.log",
                              O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd >= 0) {
            (void)syscall(__NR_write, fd, b, (size_t)n);
            (void)syscall(__NR_close, fd);
        }
    }
}

static const uint8_t *module_payload_after_ptr(const cuda_transport_t *tp,
                                                uint32_t call_id,
                                                uint32_t send_len)
{
    if (send_len == 0) return NULL;
    if (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX &&
        (call_id == CUDA_CALL_MEMCPY_HTOD ||
         call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC ||
         call_id == CUDA_CALL_MODULE_LOAD_DATA ||
         call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
         call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY ||
         call_id == CUDA_CALL_LIBRARY_LOAD_DATA))
        return (const uint8_t *)tp->shmem_g2h;
    if (cuda_transport_use_bar1_for_htod((cuda_transport_t *)tp, call_id, send_len))
        return (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
    if (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX &&
        (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
         call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
         call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY ||
         call_id == CUDA_CALL_LIBRARY_LOAD_DATA))
        return (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
    if (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX)
        return (const uint8_t *)tp->shmem_g2h;
    if (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX)
        return (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
    return (const uint8_t *)tp->bar0 + CUDA_REQ_DATA_OFFSET;
}

/* ================================================================
 * Core MMIO round-trip: write registers → ring doorbell → poll → read.
 *
 * send_len MUST fit in one pass (≤ max_single_payload()).
 * ================================================================ */
static int g_first_call_done = 0;

static void runtime_build_marker_once(cuda_transport_t *tp)
{
    static pid_t marker_pid;
    pid_t cur = getpid();
    if (marker_pid == cur) {
        return;
    }
    marker_pid = cur;

    {
        const char *tag = "phase3-runtime-marker-20260331b";
        char line[256];
        int n = snprintf(line, sizeof(line),
                         "[cuda-transport] RUNTIME_BUILD %s pid=%d has_shmem=%d has_bar1=%d g2h=%p bar1=%p\n",
                         tag, (int)cur, tp && tp->has_shmem ? 1 : 0,
                         tp && tp->has_bar1 ? 1 : 0,
                         tp ? tp->shmem_g2h : NULL,
                         tp ? tp->bar1 : NULL);
        if (n > 0) {
            int fd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_runtime_build.log",
                                  O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (fd >= 0) {
                (void)syscall(__NR_write, fd, line, (size_t)n);
                (void)syscall(__NR_close, fd);
            }
        }
    }
}

static int do_single_cuda_call(cuda_transport_t *tp,
                               uint32_t call_id,
                               const uint32_t *args, uint32_t num_args,
                               const void *send_data, uint32_t send_len,
                               CUDACallResult *result,
                               void *recv_data, uint32_t recv_cap,
                               uint32_t *recv_len)
{
    runtime_build_marker_once(tp);
    int debug_enabled = vgpu_debug_logging();
    int bulk_trace_enabled = vgpu_bulk_trace_logging();
    int log_module_payload = (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
                              call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
                              call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY) &&
                             send_data && send_len > 0 &&
                             debug_enabled;
    int log_htod_payload = (call_id == CUDA_CALL_MEMCPY_HTOD ||
                            call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC) &&
                           send_data && send_len > CUDA_SMALL_DATA_MAX &&
                           bulk_trace_enabled;
    int log_library_payload = (call_id == CUDA_CALL_LIBRARY_LOAD_DATA) &&
                              send_data && send_len > 0 &&
                              bulk_trace_enabled;
    int trace_library_timing = (call_id == CUDA_CALL_LIBRARY_LOAD_DATA) &&
                               send_data && send_len > CUDA_SMALL_DATA_MAX;
    if (!g_first_call_done) {
        g_first_call_done = 1;
        cuda_transport_write_checkpoint("FIRST_CALL");
    }
    uint32_t seq = tp->seq_counter++;
    time_t start;
    uint64_t poll_start_ns = 0;
    uint32_t status;

    if (bulk_trace_enabled && call_id == CUDA_CALL_LIBRARY_LOAD_DATA) {
        char lbuf[256];
        int n = snprintf(lbuf, sizeof(lbuf),
                         "SINGLECALL_00A8 pre_write_bulk seq=%u len=%u pid=%d send_data=%p g2h=%p has_shmem=%d\n",
                         seq, send_len, (int)getpid(), send_data,
                         tp ? tp->shmem_g2h : NULL, tp && tp->has_shmem ? 1 : 0);
        if (n > 0) {
            int lfd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_library_load_fingerprint.log",
                                   O_WRONLY | O_CREAT | O_APPEND, 0666);
            if (lfd >= 0) {
                if (n > (int)sizeof(lbuf)) n = (int)sizeof(lbuf);
                (void)syscall(__NR_write, lfd, lbuf, (size_t)n);
                (void)syscall(__NR_close, lfd);
            }
        }
    }

    if (result) {
        memset(result, 0, sizeof(*result));
    }

    if (debug_enabled) {
        fprintf(stderr, "[cuda-transport] SENDING to VGPU-STUB: call_id=0x%04x seq=%u args=%u data_len=%u (pid=%d)\n",
                call_id, seq, num_args, send_len, (int)getpid());
        fflush(stderr);
    }
    if (log_module_payload) {
        const uint8_t *src = (const uint8_t *)send_data;
        fprintf(stderr,
                "[cuda-transport] MODULE_LOAD source seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x path=%s\n",
                seq, send_len,
                send_len > 0 ? src[0] : 0, send_len > 1 ? src[1] : 0,
                send_len > 2 ? src[2] : 0, send_len > 3 ? src[3] : 0,
                send_len > 4 ? src[4] : 0, send_len > 5 ? src[5] : 0,
                send_len > 6 ? src[6] : 0, send_len > 7 ? src[7] : 0,
                (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX &&
                 (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
                  call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
                  call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY)) ? "shmem" :
                (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX &&
                 (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
                  call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
                  call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY)) ? "bar1" :
                (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX) ? "shmem" :
                (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX ? "bar1" : "bar0"));
        fflush(stderr);
    }
    if (log_library_payload) {
        const uint8_t *src = (const uint8_t *)send_data;
        char lbuf[768];
        int n = snprintf(lbuf, sizeof(lbuf),
                "LIBRARY_LOAD source seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x path=%s\n"
                "LIBRARY_LOAD pre_write_bulk seq=%u len=%u pid=%d g2h=%p src=%p has_shmem=%d\n",
                seq, send_len,
                send_len > 0 ? src[0] : 0, send_len > 1 ? src[1] : 0,
                send_len > 2 ? src[2] : 0, send_len > 3 ? src[3] : 0,
                send_len > 4 ? src[4] : 0, send_len > 5 ? src[5] : 0,
                send_len > 6 ? src[6] : 0, send_len > 7 ? src[7] : 0,
                (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX) ? "shmem" :
                (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX ? "bar1" : "bar0"),
                seq, send_len, (int)getpid(),
                (void *)tp->shmem_g2h, (void *)send_data, tp->has_shmem ? 1 : 0);
        fprintf(stderr,
                "[cuda-transport] LIBRARY_LOAD source seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x path=%s\n",
                seq, send_len,
                send_len > 0 ? src[0] : 0, send_len > 1 ? src[1] : 0,
                send_len > 2 ? src[2] : 0, send_len > 3 ? src[3] : 0,
                send_len > 4 ? src[4] : 0, send_len > 5 ? src[5] : 0,
                send_len > 6 ? src[6] : 0, send_len > 7 ? src[7] : 0,
                (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX) ? "shmem" :
                (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX ? "bar1" : "bar0"));
        fprintf(stderr,
                "[cuda-transport] LIBRARY_LOAD pre_write_bulk seq=%u len=%u pid=%d g2h=%p src=%p has_shmem=%d\n",
                seq, send_len, (int)getpid(),
                (void *)tp->shmem_g2h, (void *)send_data, tp->has_shmem ? 1 : 0);
        fflush(stderr);
        if (n > 0) {
            int lfd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_library_load_fingerprint.log",
                                   O_WRONLY | O_CREAT | O_APPEND, 0666);
            if (lfd >= 0) {
                if (n > (int)sizeof(lbuf)) n = (int)sizeof(lbuf);
                (void)syscall(__NR_write, lfd, lbuf, (size_t)n);
                (void)syscall(__NR_close, lfd);
            }
        }
    }
    if (call_id == CUDA_CALL_MEMCPY_HTOD ||
        call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC) {
        if (send_len > CUDA_SMALL_DATA_MAX) {
            const uint8_t *handoff = (const uint8_t *)send_data;
            uint64_t handoff_hash = transport_fnv1a64(send_data, send_len);
            int lfd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_handoff.log",
                                   O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (lfd >= 0) {
                char lbuf[384];
                int n = snprintf(lbuf, sizeof(lbuf),
                                 "HTOD handoff seq=%u len=%u src=%p g2h=%p has_shmem=%d "
                                 "fnv1a64=0x%016llx first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                                 seq, send_len, (void *)send_data, (void *)tp->shmem_g2h,
                                 tp->has_shmem ? 1 : 0,
                                 (unsigned long long)handoff_hash,
                                 (handoff && send_len > 0) ? handoff[0] : 0,
                                 (handoff && send_len > 1) ? handoff[1] : 0,
                                 (handoff && send_len > 2) ? handoff[2] : 0,
                                 (handoff && send_len > 3) ? handoff[3] : 0,
                                 (handoff && send_len > 4) ? handoff[4] : 0,
                                 (handoff && send_len > 5) ? handoff[5] : 0,
                                 (handoff && send_len > 6) ? handoff[6] : 0,
                                 (handoff && send_len > 7) ? handoff[7] : 0);
                if (n > 0 && n < (int)sizeof(lbuf)) {
                    (void)syscall(__NR_write, lfd, lbuf, (size_t)n);
                }
                (void)syscall(__NR_close, lfd);
            }
        }
    }
    if (log_htod_payload) {
        const uint8_t *src = (const uint8_t *)send_data;
        uint64_t htod_hash = transport_fnv1a64(send_data, send_len);
        const int force_htod_bar1_live = htod_env_force_bar1();
        const int shadow_enabled_live =
            bulk_shadow_bar1_enabled(tp, call_id, send_len, 1, 0);
        const char *primary_path = bulk_primary_path_name(tp, call_id, send_len);
        {
            const int shadow_enabled =
                bulk_shadow_bar1_enabled(tp, call_id, send_len, 1, 0);
            const uint8_t *bar1_cur = NULL;
            if (tp->bar1 && tp->bar1 != MAP_FAILED &&
                send_len <= BAR1_GUEST_TO_HOST_SIZE) {
                bar1_cur = (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
            }
            char dbuf[384];
            int dn = snprintf(dbuf, sizeof(dbuf),
                              "HTOD_OUTER_DECISION seq=%u len=%u shadow_enabled=%d "
                              "has_bar1=%d bar1_ptr=%p bar1_first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                              seq, send_len, shadow_enabled ? 1 : 0,
                              tp->has_bar1 ? 1 : 0, (const void *)bar1_cur,
                              bar1_cur ? bar1_cur[0] : 0, bar1_cur ? bar1_cur[1] : 0,
                              bar1_cur ? bar1_cur[2] : 0, bar1_cur ? bar1_cur[3] : 0,
                              bar1_cur ? bar1_cur[4] : 0, bar1_cur ? bar1_cur[5] : 0,
                              bar1_cur ? bar1_cur[6] : 0, bar1_cur ? bar1_cur[7] : 0);
            if (dn > 0) {
                int dfd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_transport.log",
                                       O_WRONLY | O_CREAT | O_APPEND, 0644);
                if (dfd >= 0) {
                    (void)syscall(__NR_write, dfd, dbuf, (size_t)dn);
                    (void)syscall(__NR_close, dfd);
                }
            }
        }
        fprintf(stderr,
                "[cuda-transport] HTOD source marker=phase3-htod-marker-20260331c seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x path=%s force_htod_bar1=%d shadow=%d\n",
                seq, send_len,
                send_len > 0 ? src[0] : 0, send_len > 1 ? src[1] : 0,
                send_len > 2 ? src[2] : 0, send_len > 3 ? src[3] : 0,
                send_len > 4 ? src[4] : 0, send_len > 5 ? src[5] : 0,
                send_len > 6 ? src[6] : 0, send_len > 7 ? src[7] : 0,
                primary_path, force_htod_bar1_live, shadow_enabled_live);
        fprintf(stderr,
                "[cuda-transport] pre_write_bulk seq=%u len=%u pid=%d g2h=%p src=%p has_shmem=%d\n",
                seq, send_len, (int)getpid(),
                (void *)tp->shmem_g2h, (void *)send_data, tp->has_shmem ? 1 : 0);
        fflush(stderr);
        int lfd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_transport.log",
                               O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (lfd >= 0) {
            /* One snprintf + one write: same text as stderr; file may be ollama-only (ACL). */
            char lbuf[768];
            int n = snprintf(lbuf, sizeof(lbuf),
                            "HTOD source marker=phase3-htod-marker-20260331c seq=%u len=%u "
                            "fnv1a64=0x%016llx first8=%02x%02x%02x%02x%02x%02x%02x%02x "
                            "path=%s force_htod_bar1=%d shadow=%d\n"
                            "pre_write_bulk seq=%u len=%u pid=%d g2h=%p src=%p has_shmem=%d\n",
                             seq, send_len,
                             (unsigned long long)htod_hash,
                             send_len > 0 ? src[0] : 0, send_len > 1 ? src[1] : 0,
                             send_len > 2 ? src[2] : 0, send_len > 3 ? src[3] : 0,
                             send_len > 4 ? src[4] : 0, send_len > 5 ? src[5] : 0,
                             send_len > 6 ? src[6] : 0, send_len > 7 ? src[7] : 0,
                            primary_path, force_htod_bar1_live, shadow_enabled_live,
                             seq, send_len, (int)getpid(),
                             (void *)tp->shmem_g2h, (void *)send_data,
                             tp->has_shmem ? 1 : 0);
            if (n > (int)sizeof(lbuf)) n = (int)sizeof(lbuf);
            if (n > 0) (void)syscall(__NR_write, lfd, lbuf, (size_t)n);
            (void)syscall(__NR_close, lfd);
        }
    }
    if (send_len > 0 && send_data &&
        (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
         call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
         call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY ||
         call_id == CUDA_CALL_LIBRARY_LOAD_DATA)) {
        module_funnel_line(tp, "pre", call_id, seq, send_len, (const uint8_t *)send_data);
    }
    refresh_shmem_registration_for_request(tp, call_id, send_len);
    /* Write bulk data before writing metadata registers */
    {
        uint64_t bulk_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
        write_bulk_data(tp, call_id, seq, send_data, send_len);
        if (trace_library_timing) {
            library_timing_trace("write_bulk_total", seq, send_len,
                                 monotonic_ns_now() - bulk_start_ns, "");
        }
    }
    if (send_len > 0 &&
        (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
         call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
         call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY ||
         call_id == CUDA_CALL_LIBRARY_LOAD_DATA)) {
        const uint8_t *pw = module_payload_after_ptr(tp, call_id, send_len);
        if (pw) module_funnel_line(tp, "post", call_id, seq, send_len, pw);
    }
    if (bulk_trace_enabled && call_id == CUDA_CALL_LIBRARY_LOAD_DATA) {
        const uint8_t *written = NULL;
        char lbuf[256];
        int n;
        if (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX) {
            written = (const uint8_t *)tp->shmem_g2h;
        } else if (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX) {
            written = (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
        } else {
            written = (const uint8_t *)tp->bar0 + CUDA_REQ_DATA_OFFSET;
        }
        n = snprintf(lbuf, sizeof(lbuf),
                     "SINGLECALL_00A8 post_write_bulk seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                     seq, send_len,
                     send_len > 0 ? written[0] : 0, send_len > 1 ? written[1] : 0,
                     send_len > 2 ? written[2] : 0, send_len > 3 ? written[3] : 0,
                     send_len > 4 ? written[4] : 0, send_len > 5 ? written[5] : 0,
                     send_len > 6 ? written[6] : 0, send_len > 7 ? written[7] : 0);
        if (n > 0) {
            int lfd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_library_load_fingerprint.log",
                                   O_WRONLY | O_CREAT | O_APPEND, 0666);
            if (lfd >= 0) {
                if (n > (int)sizeof(lbuf)) n = (int)sizeof(lbuf);
                (void)syscall(__NR_write, lfd, lbuf, (size_t)n);
                (void)syscall(__NR_close, lfd);
            }
        }
    }
    if (log_module_payload) {
        const uint8_t *written = NULL;
        if (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX &&
            (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
             call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
             call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY)) {
            written = (const uint8_t *)tp->shmem_g2h;
        } else if (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX &&
                   (call_id == CUDA_CALL_MODULE_LOAD_DATA ||
                    call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
                    call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY)) {
            written = (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
        } else if (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX) {
            written = (const uint8_t *)tp->shmem_g2h;
        } else if (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX) {
            written = (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
        } else {
            written = (const uint8_t *)tp->bar0 + CUDA_REQ_DATA_OFFSET;
        }
        fprintf(stderr,
                "[cuda-transport] MODULE_LOAD written seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                seq, send_len,
                send_len > 0 ? written[0] : 0, send_len > 1 ? written[1] : 0,
                send_len > 2 ? written[2] : 0, send_len > 3 ? written[3] : 0,
                send_len > 4 ? written[4] : 0, send_len > 5 ? written[5] : 0,
                send_len > 6 ? written[6] : 0, send_len > 7 ? written[7] : 0);
        fflush(stderr);
    }
    if (log_library_payload) {
        const uint8_t *written = NULL;
        if (tp->has_shmem && send_len > CUDA_SMALL_DATA_MAX) {
            written = (const uint8_t *)tp->shmem_g2h;
        } else if (tp->has_bar1 && send_len > CUDA_SMALL_DATA_MAX) {
            written = (const uint8_t *)tp->bar1 + BAR1_GUEST_TO_HOST_OFFSET;
        } else {
            written = (const uint8_t *)tp->bar0 + CUDA_REQ_DATA_OFFSET;
        }
        fprintf(stderr,
                "[cuda-transport] LIBRARY_LOAD written seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                seq, send_len,
                send_len > 0 ? written[0] : 0, send_len > 1 ? written[1] : 0,
                send_len > 2 ? written[2] : 0, send_len > 3 ? written[3] : 0,
                send_len > 4 ? written[4] : 0, send_len > 5 ? written[5] : 0,
                send_len > 6 ? written[6] : 0, send_len > 7 ? written[7] : 0);
        fflush(stderr);
        {
            char lbuf[256];
            int n = snprintf(lbuf, sizeof(lbuf),
                             "LIBRARY_LOAD written seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                             seq, send_len,
                             send_len > 0 ? written[0] : 0, send_len > 1 ? written[1] : 0,
                             send_len > 2 ? written[2] : 0, send_len > 3 ? written[3] : 0,
                             send_len > 4 ? written[4] : 0, send_len > 5 ? written[5] : 0,
                             send_len > 6 ? written[6] : 0, send_len > 7 ? written[7] : 0);
            if (n > 0) {
                int lfd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_library_load_fingerprint.log",
                                       O_WRONLY | O_CREAT | O_APPEND, 0666);
                if (lfd >= 0) {
                    if (n > (int)sizeof(lbuf)) n = (int)sizeof(lbuf);
                    (void)syscall(__NR_write, lfd, lbuf, (size_t)n);
                    (void)syscall(__NR_close, lfd);
                }
            }
        }
    }
    if (log_htod_payload) {
        const uint8_t *written = bulk_primary_written_ptr(tp, call_id, send_len);
        /* memmove writes through void*; volatile load so the log sees RAM bytes. */
        __sync_synchronize();
        {
            volatile const uint8_t *vw = (volatile const uint8_t *)written;
            unsigned w0 = send_len > 0 ? vw[0] : 0, w1 = send_len > 1 ? vw[1] : 0;
            unsigned w2 = send_len > 2 ? vw[2] : 0, w3 = send_len > 3 ? vw[3] : 0;
            unsigned w4 = send_len > 4 ? vw[4] : 0, w5 = send_len > 5 ? vw[5] : 0;
            unsigned w6 = send_len > 6 ? vw[6] : 0, w7 = send_len > 7 ? vw[7] : 0;
        fprintf(stderr,
                "[cuda-transport] HTOD written seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                seq, send_len, w0, w1, w2, w3, w4, w5, w6, w7);
        fflush(stderr);
        int lfd = (int)syscall(__NR_openat, -100, "/var/tmp/vgpu_htod_transport.log",
                               O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (lfd >= 0) {
            char lbuf[256];
            int n = snprintf(lbuf, sizeof(lbuf),
                             "HTOD written seq=%u len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
                             seq, send_len, w0, w1, w2, w3, w4, w5, w6, w7);
            if (n > 0) (void)syscall(__NR_write, lfd, lbuf, (size_t)n);
            (void)syscall(__NR_close, lfd);
        }
        }
    }

    /* Ensure BAR1/shmem/BAR0-inline payload is visible before BAR0 metadata. */
    {
        uint64_t flush_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
        flush_cuda_payload_writes(tp, call_id, send_len);
        if (trace_library_timing) {
            library_timing_trace("flush_payload", seq, send_len,
                                 monotonic_ns_now() - flush_start_ns, "");
        }
    }
    if (bulk_trace_enabled &&
        call_id == CUDA_CALL_LIBRARY_LOAD_DATA &&
        send_data && send_len > CUDA_SMALL_DATA_MAX &&
        tp->has_shmem && tp->shmem_g2h) {
        library_transport_stage_trace("after_flush", tp, send_data, send_len);
    }

    /* Write call metadata */
    REG32(tp->bar0, REG_CUDA_OP)       = call_id;
    REG32(tp->bar0, REG_CUDA_SEQ)      = seq;
    REG32(tp->bar0, REG_CUDA_NUM_ARGS) = (num_args > CUDA_MAX_INLINE_ARGS)
                                          ? CUDA_MAX_INLINE_ARGS : num_args;
    REG32(tp->bar0, REG_CUDA_DATA_LEN) = send_len;

    uint32_t n = (num_args > CUDA_MAX_INLINE_ARGS) ? CUDA_MAX_INLINE_ARGS : num_args;
    for (uint32_t i = 0; i < n; i++)
        REG32(tp->bar0, REG_CUDA_ARGS_BASE + i * 4) = args[i];

    flush_cuda_metadata_visible(tp);

    if (debug_enabled) {
        fprintf(stderr, "[cuda-transport] RINGING DOORBELL: MMIO write to VGPU-STUB (call_id=0x%04x, pid=%d)\n",
                call_id, (int)getpid());
        fflush(stderr);
    }
    /* Lightweight trace: last line = last call sent before crash (for runner exit status 2 diagnosis) */
    if (bulk_trace_enabled) {
        int tfd = (int)syscall(__NR_open, "/tmp/vgpu_call_sequence.log",
                               O_WRONLY | O_CREAT | O_APPEND, 0666);
        if (tfd >= 0) {
            char tbuf[32];
            int tn = snprintf(tbuf, sizeof(tbuf), "0x%04x %s\n", call_id, call_id_to_name(call_id));
            if (tn > 0) (void)syscall(__NR_write, tfd, tbuf, (size_t)tn);
            (void)syscall(__NR_close, tfd);
        }
    }
    /* Verification: record that we are about to poll for this call (alloc/HtoD/DtoH) */
    if (bulk_trace_enabled && (call_id == 0x0030u || call_id == 0x0032u || call_id == 0x0033u)) {
        int vfd = (int)syscall(__NR_open, "/tmp/vgpu_host_response_verify.log",
                O_WRONLY | O_CREAT | O_APPEND, 0666);
        if (vfd >= 0) {
            char vbuf[80];
            int vn = snprintf(vbuf, sizeof(vbuf), "SUBMIT call_id=0x%04x seq=%u (about to poll)\n", call_id, seq);
            if (vn > 0) (void)syscall(__NR_write, vfd, vbuf, (size_t)vn);
            (void)syscall(__NR_close, vfd);
        }
    }
    REG32(tp->bar0, REG_CUDA_DOORBELL) = 1;

    /* Stuck detector: overwrite current call so we can read it while blocking (e.g. after 40+ min) */
    if (bulk_trace_enabled) {
        int cfd = (int)syscall(__NR_open, "/tmp/vgpu_current_call.txt",
                               O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (cfd >= 0) {
            char cbuf[128];
            int cn = snprintf(cbuf, sizeof(cbuf), "call_id=0x%04x %s seq=%u pid=%d\n",
                              call_id, call_id_to_name(call_id), seq, (int)getpid());
            if (cn > 0) (void)syscall(__NR_write, cfd, cbuf, (size_t)cn);
            (void)syscall(__NR_close, cfd);
        }
    }

    /* Poll for completion */
    start = time(NULL);
    poll_start_ns = trace_library_timing ? monotonic_ns_now() : 0;
    unsigned poll_iter = 0;
    while (1) {
        const char *status_src = cuda_transport_status_path_name(tp);
        __asm__ __volatile__ ("" ::: "memory");
        if (tp->has_bar1) {
            status = *(volatile uint32_t *)((volatile char *)tp->bar1 + BAR1_STATUS_MIRROR_OFFSET);
            if (status != STATUS_DONE && status != STATUS_ERROR) {
                /* BAR1 mirror can occasionally lag on some boots/races.
                 * Fall back to BAR0 status for completion so calls do not
                 * spin forever when host has already written DONE/ERROR. */
                uint32_t bar0_status = REG32(tp->bar0, REG_STATUS);
                if (bar0_status == STATUS_DONE || bar0_status == STATUS_ERROR) {
                    status = bar0_status;
                    status_src = "BAR0-fallback";
                }
            }
        } else {
            status = REG32(tp->bar0, REG_STATUS);
        }
        if (status == STATUS_DONE || status == STATUS_ERROR) {
            /* Verification: log why we broke (guest received host response via status register) */
            if (bulk_trace_enabled) {
                int vfd = (int)syscall(__NR_open, "/tmp/vgpu_host_response_verify.log",
                        O_WRONLY | O_CREAT | O_APPEND, 0666);
                if (vfd >= 0) {
                    char vbuf[128];
                    int vn = snprintf(vbuf, sizeof(vbuf), "BREAK reason=STATUS call_id=0x%04x seq=%u status=0x%02x iter=%u\n",
                            call_id, seq, (unsigned)status, poll_iter);
                    if (vn > 0) (void)syscall(__NR_write, vfd, vbuf, (size_t)vn);
                    (void)syscall(__NR_close, vfd);
                }
            }
            break;
        }
        if (poll_iter >= 30) {
            uint32_t rlen = REG32(tp->bar0, REG_RESPONSE_LEN);
            if (rlen != 0) {
                /* Verification: log break due to response_len (guest received host response via BAR0+0x01C) */
                if (bulk_trace_enabled) {
                    int vfd = (int)syscall(__NR_open, "/tmp/vgpu_host_response_verify.log",
                            O_WRONLY | O_CREAT | O_APPEND, 0666);
                    if (vfd >= 0) {
                        char vbuf[128];
                        int vn = snprintf(vbuf, sizeof(vbuf), "BREAK reason=RESPONSE_LEN call_id=0x%04x seq=%u status=0x%02x rlen=%u iter=%u\n",
                                call_id, seq, (unsigned)status, (unsigned)rlen, poll_iter);
                        if (vn > 0) (void)syscall(__NR_write, vfd, vbuf, (size_t)vn);
                        (void)syscall(__NR_close, vfd);
                    }
                }
                usleep(100000);
                if (call_id == 0x0030u) {
                    uint32_t rstat = REG32(tp->bar0, REG_CUDA_RESULT_STATUS);
                    uint64_t rptr  = REG64(tp->bar0, REG_CUDA_RESULT_BASE);
                    if (rstat != 0 || rptr == 0) continue;
                }
                status = STATUS_DONE;
                break;
            }
        }
        poll_iter++;
        /* Verification: log what guest reads (status, response_len) — throttle: first 100 iters every 5, then every 50 */
        if (bulk_trace_enabled &&
            (poll_iter <= 100 ? (poll_iter % 5 == 0 || poll_iter == 1) : (poll_iter % 50 == 0))) {
            uint32_t rlen_log = (poll_iter >= 30) ? REG32(tp->bar0, REG_RESPONSE_LEN) : 0xFFFFu;
            int vfd = (int)syscall(__NR_open, "/tmp/vgpu_host_response_verify.log",
                    O_WRONLY | O_CREAT | O_APPEND, 0666);
            if (vfd >= 0) {
                char vbuf[96];
                int vn = (rlen_log != 0xFFFFu)
                    ? snprintf(vbuf, sizeof(vbuf), "iter=%u call_id=0x%04x seq=%u status=0x%02x rlen=%u\n",
                            poll_iter, call_id, seq, (unsigned)status, (unsigned)rlen_log)
                    : snprintf(vbuf, sizeof(vbuf), "iter=%u call_id=0x%04x seq=%u status=0x%02x\n",
                            poll_iter, call_id, seq, (unsigned)status);
                if (vn > 0) (void)syscall(__NR_write, vfd, vbuf, (size_t)vn);
                (void)syscall(__NR_close, vfd);
            }
        }
        if (bulk_trace_enabled) {
            /* Log at start and every 50 iters (~500ms at 10ms sleep) so we see status without relying on time() */
            int should_log = (poll_iter == 1) || (poll_iter % 50 == 0);
            if (should_log) {
                fprintf(stderr, "[cuda-transport] poll call_id=0x%04x seq=%u iter=%u status=0x%02x from=%s\n",
                        call_id, seq, poll_iter, (unsigned)status, status_src);
                fflush(stderr);
                /* Write to file (runner may not have stderr drained by server) */
                {
                    static char status_log_path[256];
                    if (status_log_path[0] == '\0') {
                        const char *home = getenv("HOME");
                        (void)snprintf(status_log_path, sizeof(status_log_path), "%s/vgpu_status_poll.log",
                                (home && home[0]) ? home : "/tmp");
                    }
                    int lfd = (int)syscall(__NR_open, status_log_path,
                            O_WRONLY | O_CREAT | O_APPEND, 0600);
                    if (lfd >= 0) {
                        char lbuf[96];
                        int n = snprintf(lbuf, sizeof(lbuf), "iter=%u seq=%u status=0x%02x from=%s\n",
                                poll_iter, seq, (unsigned)status, status_src);
                        if (n > 0) (void)syscall(__NR_write, lfd, lbuf, (size_t)n);
                        (void)syscall(__NR_close, lfd);
                    }
                }
            }
        }
        if (time(NULL) - start >= poll_timeout_sec()) {
            if (trace_library_timing) {
                char timing_detail[96];
                snprintf(timing_detail, sizeof(timing_detail),
                         "iter=%u status=0x%02x", poll_iter, (unsigned)status);
                library_timing_trace("poll_timeout", seq, send_len,
                                     monotonic_ns_now() - poll_start_ns,
                                     timing_detail);
            }
            /* Verification: log timeout (guest never saw DONE or response_len) */
            if (bulk_trace_enabled) {
                int vfd = (int)syscall(__NR_open, "/tmp/vgpu_host_response_verify.log",
                        O_WRONLY | O_CREAT | O_APPEND, 0666);
                if (vfd >= 0) {
                    char vbuf[128];
                    int vn = snprintf(vbuf, sizeof(vbuf), "BREAK reason=TIMEOUT call_id=0x%04x seq=%u status=0x%02x iter=%u\n",
                            call_id, seq, (unsigned)status, poll_iter);
                    if (vn > 0) (void)syscall(__NR_write, vfd, vbuf, (size_t)vn);
                    (void)syscall(__NR_close, vfd);
                }
            }
            char detail[64];
            snprintf(detail, sizeof(detail), "call_id=0x%04x seq=%u after %ds",
                     call_id, seq, poll_timeout_sec());
            debug_record_call(call_id, seq, 2, VGPU_ERR_TIMEOUT);
            cuda_transport_write_error("TRANSPORT_TIMEOUT", call_id,
                                       VGPU_ERR_TIMEOUT, detail);
            fprintf(stderr, "[cuda-transport] Timeout on call 0x%04x (seq=%u) after %ds\n",
                    call_id, seq, poll_timeout_sec());
            if (result) {
                memset(result, 0, sizeof(*result));
                result->magic = 0x56475055;
                result->seq_num = seq;
                result->status = CUDA_TRANSPORT_FALLBACK_CURESULT;
            }
            if (recv_len) *recv_len = 0;
            return CUDA_TRANSPORT_FALLBACK_CURESULT;
        }
        usleep(POLL_INTERVAL_US);
    }

    if (trace_library_timing) {
        char timing_detail[96];
        snprintf(timing_detail, sizeof(timing_detail),
                 "iter=%u status=0x%02x", poll_iter, (unsigned)status);
        library_timing_trace("poll_wait", seq, send_len,
                             monotonic_ns_now() - poll_start_ns,
                             timing_detail);
    }

    /* BAR-level ERROR is authoritative; don't reinterpret stale CUDA result regs as success. */
    if (status == STATUS_ERROR) {
        uint32_t err = REG32(tp->bar0, REG_ERROR_CODE);
        const char *err_name = vgpu_err_to_str(err);
        char detail[128];
        snprintf(detail, sizeof(detail), "seq=%u vm_id=%u %s", seq, tp->vm_id, err_name);
        cuda_transport_write_error(err_name, call_id, err, detail);
        fprintf(stderr,
                "[cuda-transport] STATUS_ERROR: call=%s(0x%04x) seq=%u err=0x%08x(%s) vm_id=%u\n",
                call_id_to_name(call_id), call_id, seq, err, err_name, tp->vm_id);
        fflush(stderr);
        if (result) {
            memset(result, 0, sizeof(*result));
            result->magic = 0x56475055;
            result->seq_num = seq;
            /* Host sets cuda_result_* MMIO before flipping to ERROR for driver failures. */
            if (err == VGPU_ERR_CUDA_ERROR) {
                uint32_t cst = REG32(tp->bar0, REG_CUDA_RESULT_STATUS);
                uint32_t nr = REG32(tp->bar0, REG_CUDA_RESULT_NUM);
                result->data_len = REG32(tp->bar0, REG_CUDA_RESULT_DATA_LEN);
                if (cst == 0 && nr == 0) {
                    result->status = CUDA_TRANSPORT_FALLBACK_CURESULT;
                    debug_record_call(call_id, seq,
                                      (int)CUDA_TRANSPORT_FALLBACK_CURESULT,
                                      VGPU_ERR_CUDA_ERROR);
                } else {
                    result->status = cst;
                    result->num_results = nr;
                    if (nr > CUDA_MAX_INLINE_RESULTS)
                        nr = CUDA_MAX_INLINE_RESULTS;
                    for (uint32_t i = 0; i < nr; i++)
                        result->results[i] = REG64(tp->bar0, REG_CUDA_RESULT_BASE + i * 8);
                    debug_record_call(call_id, seq, (int)cst, cst);
                    if (cst != 0) {
                        char cuda_detail[64];
                        snprintf(cuda_detail, sizeof(cuda_detail), "host_cuda_status=0x%x seq=%u",
                                 (unsigned)cst, seq);
                        cuda_transport_write_error("CUDA_CALL_FAILED", call_id, cst, cuda_detail);
                        fprintf(stderr,
                                "[cuda-transport] STATUS_ERROR host-cuda: call=%s(0x%04x) seq=%u "
                                "host_status=0x%08x\n",
                                call_id_to_name(call_id), call_id, seq, (unsigned)cst);
                    }
                }
            } else {
                result->status = CUDA_TRANSPORT_FALLBACK_CURESULT;
                debug_record_call(call_id, seq, (int)CUDA_TRANSPORT_FALLBACK_CURESULT, err);
            }
        } else {
            debug_record_call(call_id, seq, (int)CUDA_TRANSPORT_FALLBACK_CURESULT, err);
        }
        if (trace_library_timing) {
            char timing_detail[128];
            uint32_t host_status = (result && err == VGPU_ERR_CUDA_ERROR) ? result->status : 0;
            snprintf(timing_detail, sizeof(timing_detail),
                     "vm_err=0x%08x host_status=0x%08x", err, host_status);
            library_timing_trace("status_error", seq, send_len, 0, timing_detail);
        }
        if (recv_len) *recv_len = 0;
        return result ? (int)result->status : CUDA_TRANSPORT_FALLBACK_CURESULT;
    }

    /* Successful HtoD/HtoDAsync calls do not require inline results or bulk return
     * data. We already observed STATUS_DONE, and repeated BAR result-register reads
     * after large BAR1 copies are the only remaining gap in this path. Return early
     * with synthetic success to avoid stalling after host completion. */
    if (call_id == CUDA_CALL_MEMCPY_HTOD || call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC) {
        if (result) {
            memset(result, 0, sizeof(*result));
            result->magic   = 0x56475055;
            result->seq_num = seq;
            result->status  = 0;
        }
        if (recv_len) *recv_len = 0;
        debug_record_call(call_id, seq, 0, 0);
        return 0;
    }

    if (vgpu_debug_logging()) {
        if (status == STATUS_DONE)
            fprintf(stderr, "[cuda-transport] RECEIVED from VGPU-STUB: call_id=0x%04x seq=%u status=DONE (pid=%d)\n",
                    call_id, seq, (int)getpid());
        else
            fprintf(stderr, "[cuda-transport] RECEIVED from VGPU-STUB: call_id=0x%04x seq=%u status=ERROR (pid=%d)\n",
                    call_id, seq, (int)getpid());
        fflush(stderr);
    }

    /* Read result registers */
    if (result) {
        result->magic       = 0x56475055;
        result->seq_num     = seq;
        result->status      = REG32(tp->bar0, REG_CUDA_RESULT_STATUS);
        result->num_results = REG32(tp->bar0, REG_CUDA_RESULT_NUM);
        result->data_len    = REG32(tp->bar0, REG_CUDA_RESULT_DATA_LEN);
        result->reserved    = 0;
        uint32_t nr = result->num_results;
        if (nr > CUDA_MAX_INLINE_RESULTS) nr = CUDA_MAX_INLINE_RESULTS;
        for (uint32_t i = 0; i < nr; i++)
            result->results[i] = REG64(tp->bar0, REG_CUDA_RESULT_BASE + i * 8);
        /* Host returned CUDA error — debug report for exact determination */
        if (result->status != 0) {
            char detail[64];
            snprintf(detail, sizeof(detail), "host_cuda_status=0x%x seq=%u",
                     (unsigned)result->status, seq);
            debug_record_call(call_id, seq, (int)result->status, (uint32_t)result->status);
            cuda_transport_write_error("CUDA_CALL_FAILED", call_id,
                                       result->status, detail);
        } else {
            debug_record_call(call_id, seq, 0, 0);
        }
    } else {
        debug_record_call(call_id, seq, 0, 0);
    }

    /* Read bulk response */
    uint32_t resp_len = REG32(tp->bar0, REG_CUDA_RESULT_DATA_LEN);
    if (recv_data && resp_len > 0) {
        uint32_t copy_len = (resp_len > recv_cap) ? recv_cap : resp_len;
        read_bulk_data(tp, recv_data, copy_len);
        if (recv_len) *recv_len = copy_len;
    } else {
        if (recv_len) *recv_len = 0;
    }

    {
        int ret = result ? (int)result->status : 0;
        if (trace_library_timing) {
            char timing_detail[96];
            snprintf(timing_detail, sizeof(timing_detail),
                     "ret=%d resp_len=%u", ret, resp_len);
            library_timing_trace("return", seq, send_len, 0, timing_detail);
        }
        if (ret != 0) {
            int fd = (int)syscall(__NR_open, "/tmp/vgpu_transport_returned_nonzero",
                    O_WRONLY | O_CREAT | O_TRUNC, 0666);
            if (fd >= 0) {
                char buf[64];
                int n = snprintf(buf, sizeof(buf), "ret=%d call_id=0x%04x seq=%u\n", ret, call_id, seq);
                if (n > 0) (void)syscall(__NR_write, fd, buf, (size_t)n);
                (void)syscall(__NR_close, fd);
            }
        }
        return ret;
    }
}

/* ================================================================
 * Chunked host-to-device copy (cuMemcpyHtoD / cuMemcpyHtoDAsync)
 *
 * Sends the data in max_single_payload()-sized pieces, each with an
 * adjusted device pointer (dst + offset).
 * ================================================================ */
static int cuda_transport_call_htod_chunked(cuda_transport_t *tp,
                                            uint32_t call_id,
                                            const uint32_t *args,
                                            uint32_t num_args,
                                            const void *send_data,
                                            uint32_t send_len,
                                            CUDACallResult *result)
{
    uint64_t base_dst = CUDA_UNPACK_U64(args, 0);
    uint32_t chunk_args[CUDA_MAX_INLINE_ARGS];
    memcpy(chunk_args, args, num_args * sizeof(uint32_t));

    uint32_t limit  = max_single_payload(tp);
    uint32_t offset = 0;
    int rc = 0;
    CUDACallResult chunk_result;

    while (offset < send_len) {
        uint32_t chunk = send_len - offset;
        if (chunk > limit) chunk = limit;

        CUDA_PACK_U64(chunk_args, 0, base_dst + offset);
        CUDA_PACK_U64(chunk_args, 2, (uint64_t)chunk);

        if (send_len <= limit) {
            chunk_args[14] = CUDA_CHUNK_FLAG_SINGLE;
        } else if (offset == 0) {
            chunk_args[14] = CUDA_CHUNK_FLAG_FIRST;
        } else if (offset + chunk >= send_len) {
            chunk_args[14] = CUDA_CHUNK_FLAG_LAST;
        } else {
            chunk_args[14] = CUDA_CHUNK_FLAG_MIDDLE;
        }

        if (vgpu_debug_logging()) {
            char msg[192];
            int len = snprintf(msg, sizeof(msg),
                               "[cuda-transport] HTOD chunk send offset=%u chunk=%u total=%u flags=0x%x limit=%u\n",
                               offset, chunk, send_len, chunk_args[14], limit);
            if (len > 0 && len < (int)sizeof(msg)) {
                syscall(__NR_write, 2, msg, (size_t)len);
            }
        }

        memset(&chunk_result, 0, sizeof(chunk_result));
        rc = do_single_cuda_call(tp, call_id,
                                 chunk_args, num_args,
                                 (const char *)send_data + offset, chunk,
                                 &chunk_result, NULL, 0, NULL);
        if (rc != 0) {
            fprintf(stderr, "[cuda-transport] HTOD chunk failed "
                    "at offset=%u chunk=%u total=%u rc=%d\n",
                    offset, chunk, send_len, rc);
            if (result) *result = chunk_result;
            return rc;
        }
        offset += chunk;
    }
    if (result) *result = chunk_result;
    return rc;
}

/* ================================================================
 * Chunked device-to-host copy (cuMemcpyDtoH / cuMemcpyDtoHAsync)
 * ================================================================ */
static int cuda_transport_call_dtoh_chunked(cuda_transport_t *tp,
                                            uint32_t call_id,
                                            const uint32_t *args,
                                            uint32_t num_args,
                                            void *recv_data,
                                            uint32_t total_recv,
                                            uint32_t *recv_len_out,
                                            CUDACallResult *result)
{
    uint64_t base_src = CUDA_UNPACK_U64(args, 0);
    uint32_t chunk_args[CUDA_MAX_INLINE_ARGS];
    memcpy(chunk_args, args, num_args * sizeof(uint32_t));

    uint32_t limit  = max_single_payload(tp);
    uint32_t offset = 0;
    int rc = 0;
    CUDACallResult chunk_result;
    uint32_t total_received = 0;

    while (offset < total_recv) {
        uint32_t chunk = total_recv - offset;
        if (chunk > limit) chunk = limit;

        CUDA_PACK_U64(chunk_args, 0, base_src + offset);
        CUDA_PACK_U64(chunk_args, 2, (uint64_t)chunk);

        if (total_recv <= limit) {
            chunk_args[14] = CUDA_CHUNK_FLAG_SINGLE;
        } else if (offset == 0) {
            chunk_args[14] = CUDA_CHUNK_FLAG_FIRST;
        } else if (offset + chunk >= total_recv) {
            chunk_args[14] = CUDA_CHUNK_FLAG_LAST;
        } else {
            chunk_args[14] = 0;
        }

        uint32_t chunk_recv = 0;
        memset(&chunk_result, 0, sizeof(chunk_result));
        rc = do_single_cuda_call(tp, call_id,
                                 chunk_args, num_args,
                                 NULL, 0,
                                 &chunk_result,
                                 (char *)recv_data + offset, chunk,
                                 &chunk_recv);
        if (rc != 0) {
            fprintf(stderr, "[cuda-transport] DTOH chunk failed "
                    "at offset=%u chunk=%u total=%u rc=%d\n",
                    offset, chunk, total_recv, rc);
            if (result) *result = chunk_result;
            if (recv_len_out) *recv_len_out = total_received;
            return rc;
        }
        total_received += chunk_recv;
        offset += chunk;
    }
    if (result) *result = chunk_result;
    if (recv_len_out) *recv_len_out = total_received;
    return rc;
}

/* ================================================================
 * Chunked image upload (module/library load payloads)
 *
 * Carries CUDA_CHUNK_FLAG_* in args[14]; the executor accumulates
 * on the host side and performs the real load on the LAST chunk.
 * ================================================================ */
static int cuda_transport_call_image_load_chunked(
    cuda_transport_t *tp,
    uint32_t call_id,
    const void *send_data, uint32_t send_len,
    CUDACallResult *result)
{
    uint32_t chunk_args[CUDA_MAX_INLINE_ARGS];
    uint32_t limit  = max_single_payload(tp);
    uint32_t offset = 0;
    int rc = 0;
    CUDACallResult chunk_result;
    uint8_t *chunk_copy = NULL;

    /* Keep small module images on conservative BAR0 chunking for correctness,
     * but allow larger images to use the active high-throughput path. */
    if (send_len <= (64u * 1024u)) {
        limit = CUDA_SMALL_DATA_MAX;
    } else if (limit < CUDA_SMALL_DATA_MAX) {
        limit = CUDA_SMALL_DATA_MAX;
    }

    /* Large cuModuleLoadFatBinary payloads (e.g. GGML ~400 KiB) previously used a
     * single chunk with CUDA_CHUNK_FLAG_SINGLE and one large BAR1/shmem copy.
     * The host executor path for SINGLE vs chunked FIRST/.../LAST differs; the
     * smaller module load succeeded via multi-chunk accumulation. Cap chunk size
     * for FAT_BINARY so we always use FIRST/MIDDLE/LAST and mod_chunk_buf. */
    if (call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY && send_len > (64u * 1024u)) {
        uint32_t cap = 64u * 1024u;
        if (limit > cap) limit = cap;
    }

    while (offset < send_len) {
        uint32_t chunk = send_len - offset;
        if (chunk > limit) chunk = limit;

        memset(chunk_args, 0, sizeof(chunk_args));
        /* SINGLE only when this chunk carries the *entire* payload in one RPC.
         * Do NOT use (send_len <= limit): with a large BAR1/shmem limit, that is
         * true for ~400 KiB images and wrongly emits SINGLE instead of FIRST/.../LAST. */
        if (chunk == send_len) {
            chunk_args[14] = CUDA_CHUNK_FLAG_SINGLE;
        } else if (offset == 0) {
            chunk_args[14] = CUDA_CHUNK_FLAG_FIRST;
        } else if (offset + chunk >= send_len) {
            chunk_args[14] = CUDA_CHUNK_FLAG_LAST;
        } else {
            chunk_args[14] = CUDA_CHUNK_FLAG_MIDDLE;
        }

        memset(&chunk_result, 0, sizeof(chunk_result));
        chunk_copy = NULL;
        chunk_copy = (uint8_t *)malloc(chunk);
        if (!chunk_copy) {
            if (result) memset(result, 0, sizeof(*result));
            return 1;
        }
        memcpy(chunk_copy, (const uint8_t *)send_data + offset, chunk);
        if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA) {
            library_chunk_trace(tp->seq_counter, offset, chunk, send_len,
                                chunk_args[14], chunk_copy);
        }
        rc = do_single_cuda_call(tp, call_id,
                                 chunk_args, CUDA_MAX_INLINE_ARGS,
                                 chunk_copy, chunk,
                                 &chunk_result, NULL, 0, NULL);
        free(chunk_copy);
        if (rc != 0) {
            fprintf(stderr, "[cuda-transport] IMAGE_LOAD chunk failed "
                    "at offset=%u chunk=%u total=%u rc=%d\n",
                    offset, chunk, send_len, rc);
            if (result) *result = chunk_result;
            return rc;
        }
        offset += chunk;
    }
    if (result) *result = chunk_result;
    return rc;
}

/* ================================================================
 * Public API: execute one CUDA call (blocking)
 *
 * Dispatches to the appropriate chunked helper for large transfers,
 * or falls through to do_single_cuda_call for everything else.
 * ================================================================ */
static int cuda_transport_call_impl(cuda_transport_t *tp,
                                    uint32_t call_id,
                                    const uint32_t *args, uint32_t num_args,
                                    const void *send_data, uint32_t send_len,
                                    CUDACallResult *result,
                                    void *recv_data, uint32_t recv_cap,
                                    uint32_t *recv_len)
{
    int process_lock_fd = -1;
    pthread_mutex_lock(&g_transport_mutex);
    process_lock_fd = acquire_transport_process_lock();
    if (process_lock_fd < 0 && vgpu_debug_logging()) {
        fprintf(stderr,
                "[cuda-transport] WARN: cross-process transport lock unavailable: %s\n",
                strerror(errno));
        fflush(stderr);
    }
    if (vgpu_debug_logging()) {
        char inv_msg[256];
        int inv_len = snprintf(inv_msg, sizeof(inv_msg),
                              "[cuda-transport] cuda_transport_call() INVOKED: call_id=0x%04x data_len=%u tp=%p bar0=%p (pid=%d)\n",
                              call_id, send_len, (void*)tp, tp ? (void*)tp->bar0 : NULL, (int)getpid());
        if (inv_len > 0 && inv_len < (int)sizeof(inv_msg))
            syscall(__NR_write, 2, inv_msg, inv_len);
    }
    if (!tp || !tp->bar0) {
        if (vgpu_debug_logging()) {
            char err_msg[128];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                                  "[cuda-transport] ERROR: tp=%p bar0=%p (pid=%d)\n",
                                  (void*)tp, tp ? (void*)tp->bar0 : NULL, (int)getpid());
            if (err_len > 0 && err_len < (int)sizeof(err_msg))
                syscall(__NR_write, 2, err_msg, err_len);
        }
        release_transport_process_lock(process_lock_fd);
        pthread_mutex_unlock(&g_transport_mutex);
        return 1;
    }

    uint32_t limit = max_single_payload(tp);
    {
    int rc;

    /* ---- Chunked host-to-device copy ---- */
    if ((call_id == CUDA_CALL_MEMCPY_HTOD ||
         call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC) &&
        send_data && send_len > limit)
    {
        rc = cuda_transport_call_htod_chunked(tp, call_id,
                                                args, num_args,
                                                send_data, send_len,
                                                result);
        release_transport_process_lock(process_lock_fd);
        pthread_mutex_unlock(&g_transport_mutex);
        return rc;
    }

    /* ---- Chunked device-to-host copy ---- */
    if ((call_id == CUDA_CALL_MEMCPY_DTOH ||
         call_id == CUDA_CALL_MEMCPY_DTOH_ASYNC) &&
        recv_data && recv_cap > limit)
    {
        rc = cuda_transport_call_dtoh_chunked(tp, call_id,
                                                args, num_args,
                                                recv_data, recv_cap,
                                                recv_len, result);
        release_transport_process_lock(process_lock_fd);
        pthread_mutex_unlock(&g_transport_mutex);
        return rc;
    }

    /* ---- Chunked module image upload ---- */
    if ((call_id == CUDA_CALL_MODULE_LOAD_DATA    ||
         call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
         call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY) &&
        send_data && send_len > CUDA_SMALL_DATA_MAX)
    {
        rc = cuda_transport_call_image_load_chunked(tp, call_id,
                                                       send_data, send_len,
                                                       result);
        release_transport_process_lock(process_lock_fd);
        pthread_mutex_unlock(&g_transport_mutex);
        return rc;
    }

    /* ---- Chunked library image upload ----
     * Only chunk the oversized libloads that still exceed the active SHMEM
     * half-window. Smaller libloads now complete quickly via single-call SHMEM. */
    if (call_id == CUDA_CALL_LIBRARY_LOAD_DATA &&
        send_data && send_len > limit)
    {
        rc = cuda_transport_call_image_load_chunked(tp, call_id,
                                                    send_data, send_len,
                                                    result);
        release_transport_process_lock(process_lock_fd);
        pthread_mutex_unlock(&g_transport_mutex);
        return rc;
    }

    /* ---- Normal single-call path ---- */
    rc = do_single_cuda_call(tp, call_id,
                             args, num_args,
                             send_data, send_len,
                             result,
                             recv_data, recv_cap,
                             recv_len);
    release_transport_process_lock(process_lock_fd);
    pthread_mutex_unlock(&g_transport_mutex);
    return rc;
    }
}

int cuda_transport_call(cuda_transport_t *tp,
                        uint32_t call_id,
                        const uint32_t *args, uint32_t num_args,
                        const void *send_data, uint32_t send_len,
                        CUDACallResult *result,
                        void *recv_data, uint32_t recv_cap,
                        uint32_t *recv_len)
{
    return cuda_transport_call_impl(tp, call_id, args, num_args,
                                    send_data, send_len,
                                    result, recv_data, recv_cap, recv_len);
}

__attribute__((visibility("hidden")))
int cuda_transport_call_internal(cuda_transport_t *tp,
                                 uint32_t call_id,
                                 const uint32_t *args, uint32_t num_args,
                                 const void *send_data, uint32_t send_len,
                                 CUDACallResult *result,
                                 void *recv_data, uint32_t recv_cap,
                                 uint32_t *recv_len)
{
    return cuda_transport_call_impl(tp, call_id, args, num_args,
                                    send_data, send_len,
                                    result, recv_data, recv_cap, recv_len);
}

/* ================================================================
 * Accessors
 * ================================================================ */
uint32_t cuda_transport_vm_id(cuda_transport_t *tp)
{
    return tp ? tp->vm_id : 0;
}

int cuda_transport_is_connected(cuda_transport_t *tp)
{
    if (!tp || !tp->bar0) return 0;
    uint32_t ver = REG32(tp->bar0, REG_PROTOCOL_VER);
    return (ver == 0x00010000) ? 1 : 0;
}

/*
 * cuda_transport_discover — lightweight device scan.
 *
 * Scans /sys/bus/pci/devices for the VGPU-STUB PCI device and records its
 * BDF in g_discovered_bdf.  Does NOT open resource0 or map any BARs, so it
 * succeeds even inside a systemd sandbox where /sys is read-only or resource0
 * is not yet accessible.
 *
 * Returns 0 if the device was found, -1 otherwise.
 * Side-effect: sets g_discovered_bdf so cuda_transport_pci_bdf(NULL) works.
 * 
 * This function is idempotent: if g_discovered_bdf is already set and the
 * device still exists, it returns success without re-scanning.
 */
/* Forward declaration for skip interception function */
/* Use runtime resolution via dlsym to avoid linking dependency */
static void (*libvgpu_set_skip_interception_ptr)(int skip) = NULL;

/* Forward declaration - the actual function is defined in libvgpu-cuda.so or libvgpu-nvml.so
 * We use a wrapper to avoid static/non-static conflicts */

/* Wrapper that uses dlsym to find the real implementation */
static void call_libvgpu_set_skip_interception(int skip)
{
    if (!libvgpu_set_skip_interception_ptr) {
        /* Resolve symbol at runtime - it's in libvgpu-cuda.so or libvgpu-nvml.so */
        libvgpu_set_skip_interception_ptr = (void (*)(int))dlsym(RTLD_DEFAULT, "libvgpu_set_skip_interception");
        if (!libvgpu_set_skip_interception_ptr) {
            /* Symbol not found - this is OK if CUDA shim isn't loaded */
            return;
        }
    }
    libvgpu_set_skip_interception_ptr(skip);
}

int cuda_transport_discover(void)
{
    /* CRITICAL: Disable PCI file interception FIRST, before ANY operations
     * This ensures we read real values from /sys, not intercepted values
     * Based on documentation: when working, cuda_transport.c read real values directly */
    call_libvgpu_set_skip_interception(1);
    if (vgpu_debug_logging()) {
        fprintf(stderr, "[cuda-transport] DEBUG: Skip flag SET to 1 (pid=%d)\n", (int)getpid());
        fprintf(stderr, "[cuda-transport] DEBUG: cuda_transport_discover() called, g_discovered_bdf='%s'\n", g_discovered_bdf);
        fflush(stderr);
    }
    /* CRITICAL: Clear g_discovered_bdf to ensure fresh scan every time
     * This prevents issues with stale values from previous calls */
    g_discovered_bdf[0] = '\0';
    /* Fast path: if we already discovered a device, verify it still exists */
    if (g_discovered_bdf[0] != '\0') {
        if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Fast path: g_discovered_bdf='%s', verifying...\n", g_discovered_bdf);
        char verify_path[512];
        snprintf(verify_path, sizeof(verify_path),
                 "/sys/bus/pci/devices/%s/vendor", g_discovered_bdf);
        FILE *fp = fopen(verify_path, "r");
        if (fp) {
            fclose(fp);
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Fast path: Device '%s' verified, returning success (SKIPPING SCAN!)\n", g_discovered_bdf);
            return 0;
        }
        if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Fast path: Device '%s' not found, clearing cache\n", g_discovered_bdf);
        g_discovered_bdf[0] = '\0';
    }
    if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Slow path: Starting device scan...\n");
    char res0[512], res1[512], bdf[64];
    int rc = find_vgpu_device(res0, sizeof(res0), res1, sizeof(res1),
                              bdf,  sizeof(bdf));
    if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Slow path: find_vgpu_device() returned %d, g_discovered_bdf='%s'\n", rc, g_discovered_bdf);
    
    /* Re-enable interception after discovery */
    call_libvgpu_set_skip_interception(0);
    
    return rc;
}

const char *cuda_transport_pci_bdf(cuda_transport_t *tp)
{
    if (tp && tp->pci_bdf[0])
        return tp->pci_bdf;
    /* Fall back to the module-level BDF stored by find_vgpu_device()
     * during cuda_transport_discover() or cuda_transport_init(). */
    if (g_discovered_bdf[0])
        return g_discovered_bdf;
    return "0000:00:00.0";
}

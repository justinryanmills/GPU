/*
 * libvgpu_cublas.c - CUBLAS API shim library
 * 
 * This library replaces libcublas.so.12 and provides stub implementations
 * of CUBLAS functions required by GGML's CUDA backend.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <sys/syscall.h>

#include "cuda_protocol.h"
#include "cuda_transport.h"

#define __NR_write  1
#ifndef __NR_open
#define __NR_open   2
#endif
#ifndef __NR_close
#define __NR_close  3
#endif

/* Write mode to /tmp/vgpu_status for 100% outcome determination (STUB vs REAL) */
static void cublas_write_mode(const char *mode) {
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "VGPU_CUBLAS_MODE=%s\n", mode);
    if (n <= 0 || n >= (int)sizeof(buf)) return;
    int fd = (int)syscall(__NR_open, "/tmp/vgpu_status",
                          (0x41 | 0x100 | 0x200), 0666); /* O_WRONLY|O_CREAT|O_TRUNC */
    if (fd >= 0) {
        syscall(__NR_write, fd, buf, (size_t)n);
        syscall(__NR_close, fd);
    }
}

/* CUBLAS handle type */
typedef void* cublasHandle_t;
typedef int cublasStatus_t;

#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_NOT_INITIALIZED 1
#define CUBLAS_STATUS_ALLOC_FAILED 3
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_STATUS_ARCH_MISMATCH 8
#define CUBLAS_STATUS_MAPPING_ERROR 11
#define CUBLAS_STATUS_EXECUTION_FAILED 13
#define CUBLAS_STATUS_INTERNAL_ERROR 14
#define CUBLAS_STATUS_NOT_SUPPORTED 15
#define CUBLAS_STATUS_LICENSE_ERROR 16

/* Stub handle when transport is unavailable — VM-only fallback, no host. */
#define CUBLAS_STUB_MAGIC 0xCB1A5FAC
typedef struct { unsigned int magic; } cublas_stub_handle_t;
static int is_stub_handle(cublasHandle_t h) {
    if (!h) return 0;
    return ((cublas_stub_handle_t *)h)->magic == CUBLAS_STUB_MAGIC;
}

#define CUBLAS_REMOTE_MAGIC 0xCB1A5FAD
typedef struct {
    unsigned int magic;
    uint64_t remote_handle;
    uint64_t stream_handle;
} cublas_remote_handle_t;

static int is_remote_handle(cublasHandle_t h) {
    if (!h) return 0;
    return ((cublas_remote_handle_t *)h)->magic == CUBLAS_REMOTE_MAGIC;
}

static cublas_remote_handle_t *as_remote_handle(cublasHandle_t h) {
    return (cublas_remote_handle_t *)h;
}

static cuda_transport_t *g_cublas_transport = NULL;
static pthread_mutex_t g_cublas_transport_mutex = PTHREAD_MUTEX_INITIALIZER;

static int cublas_diag_logging(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = (getenv("VGPU_DEBUG") || getenv("CUBLAS_DEBUG")) ? 1 : 0;
    }
    return cached;
}

/* Guest-side sync after mediated BLAS: executor already cuCtxSynchronizes on the host;
 * this runs the shimmed cuCtxSynchronize RPC so guest/driver state stays ordered. */
typedef unsigned int CUresult;
typedef unsigned long long CUdeviceptr;
typedef unsigned int (*cublas_pfn_cuCtxSynchronize)(void);
typedef CUresult (*cublas_pfn_cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
typedef CUdeviceptr (*cublas_pfn_resolve_mapped_ptr)(CUdeviceptr);
static cublas_pfn_cuCtxSynchronize g_cuCtxSynchronize;
static cublas_pfn_cuMemcpyDtoH_v2 g_cuMemcpyDtoH_v2;
static cublas_pfn_resolve_mapped_ptr g_resolve_mapped_ptr;
static void *g_cuda_driver_dl;

static void cublas_log_batched(const char *fmt, ...)
{
    if (!cublas_diag_logging()) {
        return;
    }
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if (n >= (int)sizeof(buf)) n = (int)sizeof(buf) - 1;
    int fd = (int)syscall(__NR_open, "/tmp/vgpu_cublas_batched.log",
                          O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd >= 0) {
        syscall(__NR_write, fd, buf, (size_t)n);
        syscall(__NR_close, fd);
    }
}

static void cublas_log_sgemm_sample(int m, int n, int k, int lda, int ldb, int ldc)
{
    static unsigned long call_count = 0;
    unsigned long idx = ++call_count;

    if (idx > 64 && (idx % 256ul) != 0) {
        return;
    }

    cublas_log_batched("[libvgpu-cublas] sgemm sample idx=%lu m=%d n=%d k=%d lda=%d ldb=%d ldc=%d pid=%d\n",
                       idx, m, n, k, lda, ldb, ldc, (int)getpid());
}

static void cublas_ensure_cuCtxSynchronize(void)
{
    if (g_cuCtxSynchronize)
        return;
    if (!g_cuda_driver_dl)
        g_cuda_driver_dl = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_cuda_driver_dl)
        return;
    g_cuCtxSynchronize = (cublas_pfn_cuCtxSynchronize)dlsym(g_cuda_driver_dl, "cuCtxSynchronize");
    g_cuMemcpyDtoH_v2 = (cublas_pfn_cuMemcpyDtoH_v2)dlsym(g_cuda_driver_dl, "cuMemcpyDtoH_v2");
    g_resolve_mapped_ptr = (cublas_pfn_resolve_mapped_ptr)dlsym(g_cuda_driver_dl, "libvgpu_cuda_resolve_mapped_ptr");
    if (!g_resolve_mapped_ptr) {
        g_resolve_mapped_ptr = (cublas_pfn_resolve_mapped_ptr)dlsym(RTLD_DEFAULT, "libvgpu_cuda_resolve_mapped_ptr");
    }
}

static void cublas_guest_ctx_sync_after_rpc(void)
{
    cublas_ensure_cuCtxSynchronize();
    if (g_cuCtxSynchronize)
        (void)g_cuCtxSynchronize();
}

static uint64_t cublas_resolve_device_ptr(const void *ptr)
{
    CUdeviceptr raw = (CUdeviceptr)(uintptr_t)ptr;
    cublas_ensure_cuCtxSynchronize();
    if (raw == 0 || !g_resolve_mapped_ptr) {
        return (uint64_t)raw;
    }
    return (uint64_t)g_resolve_mapped_ptr(raw);
}

#define CUBLAS_COMPUTE_16F          64
#define CUBLAS_COMPUTE_16F_PEDANTIC 65

static float cublas_half_bits_to_float(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t frac = (uint32_t)h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (frac << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (frac << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static float cublas_scalar_to_f32(const void *scalar, int computeType)
{
    float out = 0.0f;

    if (!scalar) {
        return out;
    }

    if (computeType == CUBLAS_COMPUTE_16F || computeType == CUBLAS_COMPUTE_16F_PEDANTIC) {
        uint16_t half_bits = 0;
        memcpy(&half_bits, scalar, sizeof(half_bits));
        return cublas_half_bits_to_float(half_bits);
    }

    memcpy(&out, scalar, sizeof(out));
    return out;
}

static int cublas_ptr_table_looks_empty(const uint64_t *dst, int batchCount)
{
    int limit = batchCount < 4 ? batchCount : 4;
    for (int i = 0; i < limit; ++i) {
        if (dst[i] != 0) {
            return 0;
        }
    }
    return limit > 0;
}

static int cublas_copy_ptr_table(uint64_t *dst, const void *const *src, int batchCount,
                                 const char *label)
{
    size_t bytes = (size_t)batchCount * sizeof(uint64_t);

    cublas_ensure_cuCtxSynchronize();
    if (g_cuMemcpyDtoH_v2) {
        CUresult rc = g_cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, bytes);
        if (rc == 0) {
            if (cublas_ptr_table_looks_empty(dst, batchCount) && g_cuCtxSynchronize) {
                (void)g_cuCtxSynchronize();
                rc = g_cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, bytes);
                cublas_log_batched("[libvgpu-cublas] ptr_table=%s mode=dtoh_retry_after_sync rc=%u src=0x%llx bytes=%zu first=0x%llx pid=%d\n",
                                   label ? label : "?", rc,
                                   (unsigned long long)(uintptr_t)src, bytes,
                                   batchCount > 0 ? (unsigned long long)dst[0] : 0ull, (int)getpid());
                if (rc != 0) {
                    cublas_log_batched("[libvgpu-cublas] ptr_table=%s retry_failed rc=%u src=0x%llx bytes=%zu pid=%d\n",
                                       label ? label : "?", rc,
                                       (unsigned long long)(uintptr_t)src, bytes, (int)getpid());
                }
            }
            cublas_log_batched("[libvgpu-cublas] ptr_table=%s mode=dtoh src=0x%llx bytes=%zu first=0x%llx pid=%d\n",
                               label ? label : "?", (unsigned long long)(uintptr_t)src, bytes,
                               batchCount > 0 ? (unsigned long long)dst[0] : 0ull, (int)getpid());
            for (int i = 0; i < batchCount; i++) {
                dst[i] = cublas_resolve_device_ptr((const void *)(uintptr_t)dst[i]);
            }
            return 1;
        }
        cublas_log_batched("[libvgpu-cublas] ptr_table=%s mode=dtoh_failed rc=%u src=0x%llx bytes=%zu pid=%d\n",
                           label ? label : "?", rc, (unsigned long long)(uintptr_t)src, bytes, (int)getpid());
    }

    for (int i = 0; i < batchCount; i++) {
        dst[i] = cublas_resolve_device_ptr(src[i]);
    }
    cublas_log_batched("[libvgpu-cublas] ptr_table=%s mode=fallback_deref src=0x%llx bytes=%zu first=0x%llx pid=%d\n",
                       label ? label : "?", (unsigned long long)(uintptr_t)src, bytes,
                       batchCount > 0 ? (unsigned long long)dst[0] : 0ull, (int)getpid());
    return 0;
}

static int cublas_ensure_connected(void) {
    CUDACallResult init_result = {0};
    uint32_t recv_len = 0;

    if (g_cublas_transport && cuda_transport_is_connected(g_cublas_transport)) {
        return 0;
    }

    if (cuda_transport_discover() != 0) {
        return -1;
    }

    pthread_mutex_lock(&g_cublas_transport_mutex);
    if (g_cublas_transport && cuda_transport_is_connected(g_cublas_transport)) {
        pthread_mutex_unlock(&g_cublas_transport_mutex);
        return 0;
    }

    if (g_cublas_transport) {
        cuda_transport_destroy(g_cublas_transport);
        g_cublas_transport = NULL;
    }

    if (cuda_transport_init(&g_cublas_transport) != 0 || !g_cublas_transport) {
        pthread_mutex_unlock(&g_cublas_transport_mutex);
        return -1;
    }

    if (cuda_transport_call(g_cublas_transport, CUDA_CALL_INIT,
                            NULL, 0, NULL, 0,
                            &init_result, NULL, 0, &recv_len) != 0) {
        cuda_transport_destroy(g_cublas_transport);
        g_cublas_transport = NULL;
        pthread_mutex_unlock(&g_cublas_transport_mutex);
        return -1;
    }

    pthread_mutex_unlock(&g_cublas_transport_mutex);
    return 0;
}

static int cublas_rpc_simple(uint32_t call_id,
                             const uint32_t *args, uint32_t num_args,
                             CUDACallResult *result) {
    if (cublas_ensure_connected() != 0) {
        return -1;
    }
    return cuda_transport_call(g_cublas_transport, call_id,
                               args, num_args,
                               NULL, 0,
                               result, NULL, 0, NULL);
}

/* Resolve and forward to the real CUDA cublas runtime so math is executed.
 * CRITICAL: Load our vGPU libcuda (libvgpu-cuda) FIRST with RTLD_GLOBAL so
 * real CUBLAS binds to OUR shim for all CUDA calls. Otherwise CUBLAS may load
 * a different libcuda (stub/driver) and cuCtxGetCurrent returns NULL -> NOT_INITIALIZED. */
static void *g_real_cublas = NULL;
static int g_real_init_done = 0;
static int g_has_real_context = 0;  /* 1 only when transport gave real ctx (no host dependency for stub path) */
static const char *g_cublas_chosen_path = NULL;   /* for diagnostic */
static char g_cublas_dlerror_buf[256] = {0};      /* copy of last dlerror (dlerror ptr is volatile) */

static void *ensure_vgpu_cuda_loaded(void) {
    /* Load vGPU CUDA shim first with GLOBAL so CUBLAS sees it for cuCtxGetCurrent etc. */
    static void *vgpu_cuda = NULL;
    if (vgpu_cuda) return vgpu_cuda;
    const char *cuda_candidates[] = {
        "/opt/vgpu/lib/libvgpu-cuda.so.1",
        "/opt/vgpu/lib/libcuda.so.1",
        "libvgpu-cuda.so.1",
        "libcuda.so.1",
    };
    for (size_t i = 0; i < sizeof(cuda_candidates) / sizeof(cuda_candidates[0]); i++) {
        vgpu_cuda = dlopen(cuda_candidates[i], RTLD_NOW | RTLD_GLOBAL);
        if (vgpu_cuda) return vgpu_cuda;
    }
    return NULL;
}

/* Write diagnostic block to /tmp for post-crash analysis (handoff Mar 10) */
static void write_cublas_init_diag(const char *chosen_path, int dlopen_ok,
                                   const char *dlerror_last, const char *dlsym_result,
                                   int real_create_ret) {
    char buf[1024];
    size_t off = 0;
    time_t now = time(NULL);
    off += (size_t)snprintf(buf + off, sizeof(buf) - off,
        "===============================================================================\n"
        "VGPU CUBLAS INIT DIAGNOSTIC (pid=%d ts=%ld)\n"
        "===============================================================================\n"
        "dlopen chosen path: %s\n"
        "dlopen success: %d (1=ok 0=fail)\n"
        "dlerror() (last failed candidate): %s\n"
        "dlsym(cublasCreate_v2) result: %s\n"
        "real cublasCreate_v2 return: %d\n"
        "===============================================================================\n",
        (int)getpid(), (long)now,
        chosen_path ? chosen_path : "(null)",
        dlopen_ok,
        dlerror_last ? dlerror_last : "(none)",
        dlsym_result ? dlsym_result : "(null)",
        real_create_ret);
    int fd = (int)syscall(__NR_open, "/tmp/vgpu_cublas_init_diag.txt",
                          (0x41 | 0x100 | 0x200), 0666);
    if (fd >= 0) {
        syscall(__NR_write, fd, buf, off);
        syscall(__NR_close, fd);
    }
}

static void init_real_cublas(void) {
    if (g_real_init_done) return;
    g_real_init_done = 1;

    /* MUST load vGPU CUDA first so CUBLAS uses our shim for cuCtx* */
    (void)ensure_vgpu_cuda_loaded();

    const char *candidates[] = {
        /* Prefer versioned vendor file first: when libcublas.so.12 is a symlink to
         * libvgpu-cublas.so.12, dlopen(".../libcublas.so.12") would load THIS shim
         * again and break dlsym. */
        "/usr/local/lib/ollama/cuda_v12/libcublas.so.12.3.2.9",
        "/usr/local/lib/ollama/cuda_v12/libcublas.so.12",
        "/usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.12",
        "/usr/local/cuda/lib64/libcublas.so.12",
        "/usr/lib/x86_64-linux-gnu/libcublas.so.12",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        dlerror();
        g_real_cublas = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        if (g_real_cublas) {
            g_cublas_chosen_path = candidates[i];
            g_cublas_dlerror_buf[0] = '\0';
            break;
        }
        {
            const char *err = dlerror();
            if (err)
                snprintf(g_cublas_dlerror_buf, sizeof(g_cublas_dlerror_buf), "%s", err);
            else
                g_cublas_dlerror_buf[0] = '\0';
        }
    }
    /* Log init result for handoff debugging (gated; single write per cublasCreate) */
    if (cublas_diag_logging()) {
        char diag[384];
        int n = snprintf(diag, sizeof(diag),
            "[libvgpu-cublas] init_real_cublas: chosen=%s ok=%d dlerror=%s\n",
            g_cublas_chosen_path ? g_cublas_chosen_path : "(none)", g_real_cublas ? 1 : 0,
            g_cublas_dlerror_buf[0] ? g_cublas_dlerror_buf : "(none)");
        if (n > 0 && n < (int)sizeof(diag))
            syscall(__NR_write, 2, diag, (size_t)n);
    }
}

static void ensure_cuda_primary_context(void) {
    g_has_real_context = 0;
    void *cuda = ensure_vgpu_cuda_loaded();
    if (!cuda) return;

    typedef int (*cuInit_t)(unsigned int);
    typedef int (*cuDevicePrimaryCtxRetain_t)(void **, int);
    typedef int (*cuCtxSetCurrent_t)(void *);
    typedef int (*cuMemAlloc_v2_t)(unsigned long long *, size_t);
    typedef int (*cuMemFree_v2_t)(unsigned long long);

    cuInit_t p_cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t p_retain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t p_set_current =
        (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuMemAlloc_v2_t p_mem_alloc =
        (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t p_mem_free =
        (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");

    if (!p_cuInit || !p_retain || !p_set_current) return;

    if (p_cuInit(0) != 0) return;

    /* Retry: transport may not be ready. Real CUBLAS rejects dummy ctx (0xDEADBEEF). */
    void *ctx = NULL;
    for (int attempt = 0; attempt < 60; attempt++) {
        if (p_retain(&ctx, 0) == 0 && ctx != NULL) {
            if ((uintptr_t)ctx != 0xDEADBEEF) {
                g_has_real_context = 1;
                break;
            }
        }
        ctx = NULL;
        usleep(500000);  /* 500ms between retries, 30s total */
    }
    if (!ctx) (void)p_retain(&ctx, 0);
    if (ctx && (uintptr_t)ctx != 0xDEADBEEF) g_has_real_context = 1;

    if (!ctx || (uintptr_t)ctx == 0xDEADBEEF) return;  /* VM stub path: no real ctx */
    (void)p_set_current(ctx);

    if (p_mem_alloc && p_mem_free) {
        unsigned long long devptr = 0;
        if (p_mem_alloc(&devptr, 4096) == 0 && devptr != 0)
            (void)p_mem_free(devptr);
    }
}

static void log_cuda_context_snapshot(const char *stage)
{
    if (!cublas_diag_logging()) {
        return;
    }
    void *cuda = ensure_vgpu_cuda_loaded();
    char log_msg[512];
    void *ctx = NULL;
    int init_rc = -1;
    int get_current_rc = -1;
    int set_current_rc = -1;
    int device_rc = -1;
    int device = -1;

    if (!stage) stage = "(null)";
    if (!cuda) {
        int log_len = snprintf(log_msg, sizeof(log_msg),
                              "[libvgpu-cublas] ctx-snapshot[%s]: cuda handle unavailable (pid=%d)\n",
                              stage, (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg))
            syscall(__NR_write, 2, log_msg, (size_t)log_len);
        return;
    }

    typedef int (*cuInit_t)(unsigned int);
    typedef int (*cuCtxGetCurrent_t)(void **);
    typedef int (*cuCtxSetCurrent_t)(void *);
    typedef int (*cuCtxGetDevice_t)(int *);

    cuInit_t p_cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuCtxGetCurrent_t p_get_current = (cuCtxGetCurrent_t)dlsym(cuda, "cuCtxGetCurrent");
    cuCtxSetCurrent_t p_set_current = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxGetDevice_t p_get_device = (cuCtxGetDevice_t)dlsym(cuda, "cuCtxGetDevice");

    if (p_cuInit) init_rc = p_cuInit(0);
    if (p_get_current) get_current_rc = p_get_current(&ctx);
    if (p_set_current && ctx) set_current_rc = p_set_current(ctx);
    if (p_get_device) device_rc = p_get_device(&device);

    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] ctx-snapshot[%s]: init_rc=%d get_current_rc=%d ctx=%p set_current_rc=%d get_device_rc=%d device=%d has_real_context=%d (pid=%d)\n",
                          stage, init_rc, get_current_rc, ctx, set_current_rc,
                          device_rc, device, g_has_real_context, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, (size_t)log_len);
        int fd = (int)syscall(__NR_open, "/tmp/vgpu_cublas_context_diag.txt",
                              O_WRONLY | O_CREAT | O_APPEND, 0666);
        if (fd >= 0) {
            syscall(__NR_write, fd, log_msg, (size_t)log_len);
            syscall(__NR_close, fd);
        }
    }
}

#define RESOLVE_OR_FALLBACK(fn_name, fn_type, fallback_ret) \
    init_real_cublas(); \
    fn_type real_fn = NULL; \
    if (g_real_cublas) real_fn = (fn_type)dlsym(g_real_cublas, fn_name); \
    if (!real_fn) return (fallback_ret)

/* CUBLAS create handle */
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    /* Debug: which call is reached after the 6 allocs (for runner exit 2) */
    if (cublas_diag_logging()) {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666); /* O_WRONLY|O_CREAT|O_APPEND */
        if (nfd >= 0) {
            const char *msg = "cublas_create\n";
            syscall(__NR_write, nfd, msg, 14);
            syscall(__NR_close, nfd);
        }
    }
    CUDACallResult result = {0};
    cublas_remote_handle_t *remote = NULL;

    /* CRITICAL: Log this call - GGML requires CUBLAS for matrix operations */
    char log_msg[128];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasCreate_v2() CALLED (pid=%d)\n",
                          (int)getpid());
    if (cublas_diag_logging() && log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    /* Diagnostic: if this file appears after generate, inference path is CUBLAS-first (B1) */
    if (cublas_diag_logging()) {
        int mfd = (int)syscall(__NR_open, "/tmp/vgpu_cublas_called", 0x41 | 0x100 | 0x200, 0666); /* O_WRONLY|O_CREAT|O_TRUNC */
        if (mfd >= 0) {
            char buf[64];
            int n = snprintf(buf, sizeof(buf), "pid=%d\n", (int)getpid());
            if (n > 0) syscall(__NR_write, mfd, buf, (size_t)n);
            syscall(__NR_close, mfd);
        }
    }
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;

    int rpc_rc = cublas_rpc_simple(CUDA_CALL_CUBLAS_CREATE, NULL, 0, &result);
    if (cublas_diag_logging()) {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) {
            char msg[64];
            int n = snprintf(msg, sizeof(msg), "cublas_rpc_rc=%d num_results=%u\n", rpc_rc, (unsigned)result.num_results);
            if (n > 0) syscall(__NR_write, nfd, msg, (size_t)n);
            syscall(__NR_close, nfd);
        }
    }
    if (rpc_rc == 0 && result.num_results >= 1) {
        cublasStatus_t status = (cublasStatus_t)result.results[0];
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
        if (result.num_results < 2) {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        remote = (cublas_remote_handle_t *)calloc(1, sizeof(*remote));
        if (!remote) {
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
        remote->magic = CUBLAS_REMOTE_MAGIC;
        remote->remote_handle = result.results[1];
        remote->stream_handle = 0;
        *handle = (cublasHandle_t)remote;
        cublas_write_mode("REMOTE");
        return CUBLAS_STATUS_SUCCESS;
    }

    init_real_cublas();  /* Load vGPU cuda first, then cublas - order critical */
    ensure_cuda_primary_context();
    log_cuda_context_snapshot("before-real-create");

    /* VM-only fallback: no real context (transport unavailable). Return stub handle. */
    if (!g_has_real_context) {
        cublas_write_mode("STUB");  /* Deterministic: no host used */
        write_cublas_init_diag(g_cublas_chosen_path, g_real_cublas ? 1 : 0,
                               g_cublas_dlerror_buf[0] ? g_cublas_dlerror_buf : NULL,
                               "skipped(stub)", -1);
        cublas_stub_handle_t *stub = (cublas_stub_handle_t *)malloc(sizeof(cublas_stub_handle_t));
        if (!stub) return CUBLAS_STATUS_ALLOC_FAILED;
        stub->magic = CUBLAS_STUB_MAGIC;
        *handle = (cublasHandle_t)stub;
        return CUBLAS_STATUS_SUCCESS;
    }
    {
        typedef cublasStatus_t (*fn_t)(cublasHandle_t *);
        fn_t real_fn = NULL;
        dlerror();
        if (g_real_cublas) real_fn = (fn_t)dlsym(g_real_cublas, "cublasCreate_v2");
        const char *dlsym_err = dlerror();
        if (!real_fn) {
            cublas_write_mode("REAL_DLSYM_FAILED");  /* Cannot find cublasCreate_v2 */
            write_cublas_init_diag(g_cublas_chosen_path, g_real_cublas ? 1 : 0,
                                   g_cublas_dlerror_buf[0] ? g_cublas_dlerror_buf : NULL,
                                   dlsym_err ? dlsym_err : "dlsym returned NULL",
                                   -1);
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        cublasStatus_t ret = real_fn(handle);
        log_cuda_context_snapshot("after-real-create");
        write_cublas_init_diag(g_cublas_chosen_path, g_real_cublas ? 1 : 0,
                               g_cublas_dlerror_buf[0] ? g_cublas_dlerror_buf : NULL,
                               "ok", ret);
        if (ret == CUBLAS_STATUS_SUCCESS) {
            cublas_write_mode("REAL");  /* Deterministic: host path used */
        } else {
            /* Debug report for CUBLAS failure */
            char buf[512];
            size_t off = 0;
            time_t now = time(NULL);
            off += (size_t)snprintf(buf + off, sizeof(buf) - off,
                "===============================================================================\n"
                "VGPU DEBUG REPORT — CUBLAS error\n"
                "===============================================================================\n"
                "timestamp: %ld  pid: %d\n\n"
                ">>> WHAT FAILED: CUBLAS_CREATE_FAILED\n"
                ">>> FAILING CALL: cublasCreate_v2\n"
                ">>> CUBLAS STATUS: %d\n"
                ">>> DETAIL: real CUBLAS library rejected context (invalid or 0xDEADBEEF)\n\n"
                ">>> LIKELY CAUSE: Transport/mediator not connected; real CUBLAS needs valid cuCtxGetCurrent.\n"
                ">>> See /tmp/vgpu_cublas_init_diag.txt for dlopen/dlsym/real_create details.\n",
                (long)now, (int)getpid(), ret);
            int fd = (int)syscall(__NR_open, "/tmp/vgpu_debug.txt",
                                  (0x41 | 0x100 | 0x200), 0666);
            if (fd >= 0) {
                syscall(__NR_write, fd, buf, off);
                syscall(__NR_close, fd);
            }
            fd = (int)syscall(__NR_open, "/tmp/vgpu_last_error",
                              (0x41 | 0x100 | 0x200), 0666);
            if (fd >= 0) {
                int n = snprintf(buf, sizeof(buf), "VGPU_ERR|%ld|CUBLAS_CREATE_FAILED|0|%d|real_cublasCreate_v2=%d\n",
                                 (long)now, ret, ret);
                if (n > 0 && n < (int)sizeof(buf))
                    syscall(__NR_write, fd, buf, (size_t)n);
                syscall(__NR_close, fd);
            }
        }
        return ret;
    }
}

/* CUBLAS create handle (non-v2 version) */
cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    return cublasCreate_v2(handle);
}

/* CUBLAS destroy handle */
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    if (cublas_diag_logging()) {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) { const char *msg = "cublas_destroy\n"; syscall(__NR_write, nfd, msg, 16); syscall(__NR_close, nfd); }
    }
    CUDACallResult result = {0};
    uint32_t args[2];

    if (is_stub_handle(handle)) {
        free(handle);
        return CUBLAS_STATUS_SUCCESS;
    }
    if (is_remote_handle(handle)) {
        CUDA_PACK_U64(args, 0, as_remote_handle(handle)->remote_handle);
        if (cublas_rpc_simple(CUDA_CALL_CUBLAS_DESTROY, args, 2, &result) != 0 ||
            result.num_results < 1) {
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        cublasStatus_t status = (cublasStatus_t)result.results[0];
        if (status == CUBLAS_STATUS_SUCCESS) {
            free(handle);
        }
        return status;
    }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t);
    RESOLVE_OR_FALLBACK("cublasDestroy_v2", fn_t, CUBLAS_STATUS_NOT_INITIALIZED);
    return real_fn(handle);
}

/* CUBLAS destroy handle (non-v2 version) */
cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return cublasDestroy_v2(handle);
}

/* CUBLAS set stream */
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void *stream) {
    if (cublas_diag_logging()) {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) { const char *msg = "set_stream\n"; syscall(__NR_write, nfd, msg, 12); syscall(__NR_close, nfd); }
    }
    CUDACallResult result = {0};
    uint32_t args[4];

    if (is_stub_handle(handle)) return CUBLAS_STATUS_SUCCESS;
    if (is_remote_handle(handle)) {
        if (as_remote_handle(handle)->stream_handle == (uint64_t)(uintptr_t)stream) {
            return CUBLAS_STATUS_SUCCESS;
        }
        CUDA_PACK_U64(args, 0, as_remote_handle(handle)->remote_handle);
        CUDA_PACK_U64(args, 2, (uint64_t)(uintptr_t)stream);
        if (cublas_rpc_simple(CUDA_CALL_CUBLAS_SET_STREAM, args, 4, &result) != 0 ||
            result.num_results < 1) {
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        cublasStatus_t status = (cublasStatus_t)result.results[0];
        if (status == CUBLAS_STATUS_SUCCESS) {
            as_remote_handle(handle)->stream_handle = (uint64_t)(uintptr_t)stream;
        }
        if (cublas_diag_logging()) {
            int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
            if (nfd >= 0) { const char *msg = "set_stream_done\n"; syscall(__NR_write, nfd, msg, 17); syscall(__NR_close, nfd); }
        }
        return status;
    }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, void *);
    RESOLVE_OR_FALLBACK("cublasSetStream_v2", fn_t, CUBLAS_STATUS_NOT_INITIALIZED);
    return real_fn(handle, stream);
}

/* CUBLAS set stream (non-v2 version) */
cublasStatus_t cublasSetStream(cublasHandle_t handle, void *stream) {
    return cublasSetStream_v2(handle, stream);
}

/* CUBLAS get stream */
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, void **stream) {
    if (!stream) return CUBLAS_STATUS_INVALID_VALUE;
    if (is_stub_handle(handle)) { *stream = NULL; return CUBLAS_STATUS_SUCCESS; }
    if (is_remote_handle(handle)) {
        *stream = (void *)(uintptr_t)as_remote_handle(handle)->stream_handle;
        return CUBLAS_STATUS_SUCCESS;
    }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, void **);
    RESOLVE_OR_FALLBACK("cublasGetStream_v2", fn_t, CUBLAS_STATUS_NOT_INITIALIZED);
    return real_fn(handle, stream);
}

/* CUBLAS get stream (non-v2 version) */
cublasStatus_t cublasGetStream(cublasHandle_t handle, void **stream) {
    return cublasGetStream_v2(handle, stream);
}

/* CUBLAS set math mode */
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode) {
    if (is_stub_handle(handle)) return CUBLAS_STATUS_SUCCESS;
    if (is_remote_handle(handle)) return CUBLAS_STATUS_SUCCESS;
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, int);
    RESOLVE_OR_FALLBACK("cublasSetMathMode", fn_t, CUBLAS_STATUS_NOT_SUPPORTED);
    return real_fn(handle, mode);
}

/* CUBLAS get math mode */
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, int *mode) {
    if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
    if (is_stub_handle(handle)) { *mode = 0; return CUBLAS_STATUS_SUCCESS; }
    if (is_remote_handle(handle)) { *mode = 0; return CUBLAS_STATUS_SUCCESS; }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, int *);
    RESOLVE_OR_FALLBACK("cublasGetMathMode", fn_t, CUBLAS_STATUS_NOT_SUPPORTED);
    return real_fn(handle, mode);
}

/* cublasGetStatusString - get status string */
/* CRITICAL: Must be exported with correct version for GGML */
__attribute__((visibility("default")))
const char* cublasGetStatusString(cublasStatus_t status) {
    init_real_cublas();
    if (g_real_cublas) {
        typedef const char *(*fn_t)(cublasStatus_t);
        fn_t real_fn = (fn_t)dlsym(g_real_cublas, "cublasGetStatusString");
        if (real_fn) {
            return real_fn(status);
        }
    }

    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

/* cublasSgemm_v2 - single precision matrix multiply */
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, int transa, int transb,
                              int m, int n, int k,
                              const float *alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              const float *beta,
                              float *C, int ldc) {
    if (cublas_diag_logging()) {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) { const char *msg = "sgemm\n"; syscall(__NR_write, nfd, msg, 7); syscall(__NR_close, nfd); }
    }
    CUDACallResult result = {0};

    if (is_stub_handle(handle)) return CUBLAS_STATUS_SUCCESS;
    if (is_remote_handle(handle)) {
        CublasSgemmCall payload;

        if (!alpha || !beta || !A || !B || !C) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        cublas_log_sgemm_sample(m, n, k, lda, ldb, ldc);
        memset(&payload, 0, sizeof(payload));
        payload.handle = as_remote_handle(handle)->remote_handle;
        payload.a = cublas_resolve_device_ptr(A);
        payload.b = cublas_resolve_device_ptr(B);
        payload.c = cublas_resolve_device_ptr(C);
        payload.transa = transa;
        payload.transb = transb;
        payload.m = m;
        payload.n = n;
        payload.k = k;
        payload.lda = lda;
        payload.ldb = ldb;
        payload.ldc = ldc;
        payload.alpha = *alpha;
        payload.beta = *beta;

        if (cublas_ensure_connected() != 0) {
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        if (cuda_transport_call(g_cublas_transport, CUDA_CALL_CUBLAS_SGEMM,
                                NULL, 0,
                                &payload, (uint32_t)sizeof(payload),
                                &result, NULL, 0, NULL) != 0 ||
            result.num_results < 1) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        cublas_guest_ctx_sync_after_rpc();
        return (cublasStatus_t)result.results[0];
    }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, int, int, int, int,
                                   const float *, const float *, int,
                                   const float *, int, const float *,
                                   float *, int);
    RESOLVE_OR_FALLBACK("cublasSgemm_v2", fn_t, CUBLAS_STATUS_EXECUTION_FAILED);
    return real_fn(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

/* cublasStrsmBatched - batched triangular solve */
cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, int side, int uplo,
                                  int trans, int diag,
                                  int m, int n,
                                  const float *alpha,
                                  float *const A[], int lda,
                                  float *const B[], int ldb,
                                  int batchCount) {
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasStrsmBatched() CALLED (m=%d, n=%d, batch=%d, pid=%d)\n",
                          m, n, batchCount, (int)getpid());
    if (cublas_diag_logging() && log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, int, int, int, int, int,
                                   const float *, float *const [], int,
                                   float *const [], int, int);
    RESOLVE_OR_FALLBACK("cublasStrsmBatched", fn_t, CUBLAS_STATUS_EXECUTION_FAILED);
    return real_fn(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

/* cublasGemmEx - extended GEMM with type support */
cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                            int transa, int transb,
                            int m, int n, int k,
                            const void *alpha,
                            const void *A, int Atype, int lda,
                            const void *B, int Btype, int ldb,
                            const void *beta,
                            void *C, int Ctype, int ldc,
                            int computeType, int algo) {
    if (cublas_diag_logging()) {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) { const char *msg = "gemm_ex\n"; syscall(__NR_write, nfd, msg, 9); syscall(__NR_close, nfd); }
    }
    CUDACallResult result = {0};
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasGemmEx() CALLED (m=%d, n=%d, k=%d, pid=%d)\n",
                          m, n, k, (int)getpid());
    if (cublas_diag_logging() && log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    if (is_stub_handle(handle)) return CUBLAS_STATUS_SUCCESS;
    if (is_remote_handle(handle)) {
        CublasGemmExCall payload;
        if (!alpha || !beta || !A || !B || !C) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        memset(&payload, 0, sizeof(payload));
        payload.handle = as_remote_handle(handle)->remote_handle;
        payload.a = cublas_resolve_device_ptr(A);
        payload.b = cublas_resolve_device_ptr(B);
        payload.c = cublas_resolve_device_ptr(C);
        payload.transa = transa;
        payload.transb = transb;
        payload.m = m;
        payload.n = n;
        payload.k = k;
        payload.Atype = Atype;
        payload.Btype = Btype;
        payload.Ctype = Ctype;
        payload.lda = lda;
        payload.ldb = ldb;
        payload.ldc = ldc;
        payload.computeType = computeType;
        payload.algo = algo;
        payload.alpha_f32 = cublas_scalar_to_f32(alpha, computeType);
        payload.beta_f32 = cublas_scalar_to_f32(beta, computeType);

        if (cublas_ensure_connected() != 0) {
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        if (cublas_diag_logging()) {
            int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
            if (nfd >= 0) { const char *msg = "gemm_ex_before_send\n"; syscall(__NR_write, nfd, msg, 20); syscall(__NR_close, nfd); }
        }
        int tc_rc = cuda_transport_call(g_cublas_transport, CUDA_CALL_CUBLAS_GEMM_EX,
                                       NULL, 0,
                                       &payload, (uint32_t)sizeof(payload),
                                       &result, NULL, 0, NULL);
        if (cublas_diag_logging()) {
            int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
            if (nfd >= 0) {
                char msg[80];
                int n = snprintf(msg, sizeof(msg), "gemm_ex tc_rc=%d num_results=%u status=%u\n",
                                tc_rc, (unsigned)result.num_results, (unsigned)result.status);
                if (n > 0) syscall(__NR_write, nfd, msg, (size_t)n);
                syscall(__NR_close, nfd);
            }
        }
        if (tc_rc != 0 || result.num_results < 1) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        cublas_guest_ctx_sync_after_rpc();
        if (cublas_diag_logging()) {
            int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
            if (nfd >= 0) { const char *msg = "gemm_ex_return\n"; syscall(__NR_write, nfd, msg, 15); syscall(__NR_close, nfd); }
        }
        if (cublas_diag_logging()) {
            char ret_msg[288];
            int rn = snprintf(ret_msg, sizeof(ret_msg),
                              "[libvgpu-cublas] cublasGemmEx() RETURN ok tc_rc=%d cublas_status=%u (m=%d n=%d k=%d pid=%d)\n",
                              tc_rc, (unsigned)result.results[0], m, n, k, (int)getpid());
            if (rn > 0 && rn < (int)sizeof(ret_msg)) {
                syscall(__NR_write, 2, ret_msg, (size_t)rn);
            }
        }
        return (cublasStatus_t)result.results[0];
    }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, int, int, int, int,
                                   const void *, const void *, int, int,
                                   const void *, int, int, const void *,
                                   void *, int, int, int, int);
    RESOLVE_OR_FALLBACK("cublasGemmEx", fn_t, CUBLAS_STATUS_EXECUTION_FAILED);
    return real_fn(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
                   beta, C, Ctype, ldc, computeType, algo);
}

/* cublasGemmStridedBatchedEx - strided batched GEMM with type support */
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                         int transa, int transb,
                                         int m, int n, int k,
                                         const void *alpha,
                                         const void *A, int Atype, int lda,
                                         long long int strideA,
                                         const void *B, int Btype, int ldb,
                                         long long int strideB,
                                         const void *beta,
                                         void *C, int Ctype, int ldc,
                                         long long int strideC,
                                         int batchCount,
                                         int computeType, int algo) {
    if (cublas_diag_logging()) {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) { const char *msg = "gemm_strided_batched\n"; syscall(__NR_write, nfd, msg, 22); syscall(__NR_close, nfd); }
    }
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasGemmStridedBatchedEx() CALLED (m=%d, n=%d, k=%d, batch=%d, pid=%d)\n",
                          m, n, k, batchCount, (int)getpid());
    if (cublas_diag_logging() && log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    if (is_stub_handle(handle)) return CUBLAS_STATUS_SUCCESS;
    if (is_remote_handle(handle)) {
        CUDACallResult result = {0};
        CublasGemmStridedBatchedExCall payload;
        if (!alpha || !beta || !A || !B || !C) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        memset(&payload, 0, sizeof(payload));
        payload.handle = as_remote_handle(handle)->remote_handle;
        payload.a = cublas_resolve_device_ptr(A);
        payload.b = cublas_resolve_device_ptr(B);
        payload.c = cublas_resolve_device_ptr(C);
        payload.strideA = (int64_t)strideA;
        payload.strideB = (int64_t)strideB;
        payload.strideC = (int64_t)strideC;
        payload.transa = transa;
        payload.transb = transb;
        payload.m = m;
        payload.n = n;
        payload.k = k;
        payload.Atype = Atype;
        payload.Btype = Btype;
        payload.Ctype = Ctype;
        payload.lda = lda;
        payload.ldb = ldb;
        payload.ldc = ldc;
        payload.batchCount = batchCount;
        payload.computeType = computeType;
        payload.algo = algo;
        payload.alpha_f32 = cublas_scalar_to_f32(alpha, computeType);
        payload.beta_f32 = cublas_scalar_to_f32(beta, computeType);
        if (cublas_ensure_connected() != 0) {
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        if (cuda_transport_call(g_cublas_transport, CUDA_CALL_CUBLAS_GEMM_STRIDED_BATCHED_EX,
                                NULL, 0,
                                &payload, (uint32_t)sizeof(payload),
                                &result, NULL, 0, NULL) != 0 ||
            result.num_results < 1) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        cublas_guest_ctx_sync_after_rpc();
        return (cublasStatus_t)result.results[0];
    }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, int, int, int, int,
                                   const void *, const void *, int, int, long long int,
                                   const void *, int, int, long long int,
                                   const void *, void *, int, int, long long int,
                                   int, int, int);
    RESOLVE_OR_FALLBACK("cublasGemmStridedBatchedEx", fn_t, CUBLAS_STATUS_EXECUTION_FAILED);
    return real_fn(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA,
                   B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount,
                   computeType, algo);
}

/* cublasGemmBatchedEx - batched GEMM with type support */
cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
                                   int transa, int transb,
                                   int m, int n, int k,
                                   const void *alpha,
                                   const void *const Aarray[], int Atype, int lda,
                                   const void *const Barray[], int Btype, int ldb,
                                   const void *beta,
                                   void *const Carray[], int Ctype, int ldc,
                                   int batchCount,
                                   int computeType, int algo) {
    if (cublas_diag_logging()) {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) { const char *msg = "gemm_batched\n"; syscall(__NR_write, nfd, msg, 13); syscall(__NR_close, nfd); }
    }
    if (is_stub_handle(handle)) return CUBLAS_STATUS_SUCCESS;
    if (is_remote_handle(handle)) {
        CUDACallResult result = {0};
        if (!alpha || !beta || !Aarray || !Barray || !Carray || batchCount < 1) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        size_t ptr_bytes = (size_t)batchCount * 3u * sizeof(uint64_t);
        size_t total = sizeof(CublasGemmBatchedExCallHdr) + ptr_bytes;
        uint8_t *payload_buf = (uint8_t *)malloc(total);
        if (!payload_buf) {
            return CUBLAS_STATUS_ALLOC_FAILED;
        }
        CublasGemmBatchedExCallHdr *hdr = (CublasGemmBatchedExCallHdr *)payload_buf;
        memset(hdr, 0, sizeof(*hdr));
        hdr->handle = as_remote_handle(handle)->remote_handle;
        hdr->transa = transa;
        hdr->transb = transb;
        hdr->m = m;
        hdr->n = n;
        hdr->k = k;
        hdr->Atype = Atype;
        hdr->Btype = Btype;
        hdr->Ctype = Ctype;
        hdr->lda = lda;
        hdr->ldb = ldb;
        hdr->ldc = ldc;
        hdr->batchCount = batchCount;
        hdr->computeType = computeType;
        hdr->algo = algo;
        hdr->alpha_f32 = cublas_scalar_to_f32(alpha, computeType);
        hdr->beta_f32 = cublas_scalar_to_f32(beta, computeType);
        uint64_t *tr = (uint64_t *)(payload_buf + sizeof(CublasGemmBatchedExCallHdr));
        int a_dtoh = cublas_copy_ptr_table(tr, Aarray, batchCount, "A");
        int b_dtoh = cublas_copy_ptr_table(tr + (size_t)batchCount, Barray, batchCount, "B");
        int c_dtoh = cublas_copy_ptr_table(tr + 2u * (size_t)batchCount, (const void *const *)Carray, batchCount, "C");
        cublas_log_batched("[libvgpu-cublas] gemm_batched submit batch=%d m=%d n=%d k=%d Atype=%d Btype=%d Ctype=%d computeType=%d algo=%d lda=%d ldb=%d ldc=%d dtoh=%d/%d/%d firstA=0x%llx firstB=0x%llx firstC=0x%llx pid=%d\n",
                           batchCount, m, n, k, Atype, Btype, Ctype, computeType, algo, lda, ldb, ldc,
                           a_dtoh, b_dtoh, c_dtoh,
                           batchCount > 0 ? (unsigned long long)tr[0] : 0ull,
                           batchCount > 0 ? (unsigned long long)tr[(size_t)batchCount] : 0ull,
                           batchCount > 0 ? (unsigned long long)tr[2u * (size_t)batchCount] : 0ull,
                           (int)getpid());
        if (cublas_ensure_connected() != 0) {
            free(payload_buf);
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        int tc_rc = cuda_transport_call(g_cublas_transport, CUDA_CALL_CUBLAS_GEMM_BATCHED_EX,
                                        NULL, 0,
                                        payload_buf, (uint32_t)total,
                                        &result, NULL, 0, NULL);
        free(payload_buf);
        if (tc_rc != 0 || result.num_results < 1) {
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        cublas_guest_ctx_sync_after_rpc();
        cublas_log_batched("[libvgpu-cublas] gemm_batched done rc=%d transport_rc=%d pid=%d\n",
                           (int)result.results[0], tc_rc, (int)getpid());
        return (cublasStatus_t)result.results[0];
    }
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, int, int, int, int, int,
                                   const void *, const void *const [], int, int,
                                   const void *const [], int, int, const void *,
                                   void *const [], int, int, int, int, int);
    RESOLVE_OR_FALLBACK("cublasGemmBatchedEx", fn_t, CUBLAS_STATUS_EXECUTION_FAILED);
    return real_fn(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda,
                   Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount,
                   computeType, algo);
}

/* Constructor */
__attribute__((constructor))
static void libvgpu_cublas_on_load(void) {
    if (cublas_diag_logging()) {
        const char *msg = "[libvgpu-cublas] Library loaded - CUBLAS shim initialized\n";
        syscall(__NR_write, 2, msg, 60);
        int fd = (int)syscall(__NR_open, "/tmp/vgpu_cublas_loaded",
                              O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd >= 0) {
            char buf[96];
            int n = snprintf(buf, sizeof(buf), "pid=%d\n", (int)getpid());
            if (n > 0 && n < (int)sizeof(buf)) {
                syscall(__NR_write, fd, buf, (size_t)n);
            }
            syscall(__NR_close, fd);
        }
    }
}

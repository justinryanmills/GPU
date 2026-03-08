/*
 * libvgpu_cublas.c - CUBLAS API shim library
 * 
 * This library replaces libcublas.so.12 and provides stub implementations
 * of CUBLAS functions required by GGML's CUDA backend.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/syscall.h>

#define __NR_write  1
#ifndef __NR_open
#define __NR_open   2
#endif
#ifndef __NR_close
#define __NR_close  3
#endif

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

/* CUBLAS create handle */
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    /* CRITICAL: Log this call - GGML requires CUBLAS for matrix operations */
    char log_msg[128];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasCreate_v2() CALLED (pid=%d)\n",
                          (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    /* Diagnostic: if this file appears after generate, inference path is CUBLAS-first (B1) */
    {
        int mfd = (int)syscall(__NR_open, "/tmp/vgpu_cublas_called", 0x41 | 0x100 | 0x200, 0666); /* O_WRONLY|O_CREAT|O_TRUNC */
        if (mfd >= 0) {
            char buf[64];
            int n = snprintf(buf, sizeof(buf), "pid=%d\n", (int)getpid());
            if (n > 0) syscall(__NR_write, mfd, buf, (size_t)n);
            syscall(__NR_close, mfd);
        }
    }
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    
    /* Allocate a dummy handle - just use a static pointer */
    static void *dummy_handle = (void *)0x1000;
    *handle = (cublasHandle_t)dummy_handle;
    
    char success_msg[128];
    int success_len = snprintf(success_msg, sizeof(success_msg),
                              "[libvgpu-cublas] cublasCreate_v2() SUCCESS: handle=%p (pid=%d)\n",
                              *handle, (int)getpid());
    if (success_len > 0 && success_len < (int)sizeof(success_msg)) {
        syscall(__NR_write, 2, success_msg, success_len);
    }
    
    return CUBLAS_STATUS_SUCCESS;
}

/* CUBLAS create handle (non-v2 version) */
cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    return cublasCreate_v2(handle);
}

/* CUBLAS destroy handle */
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    char log_msg[128];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasDestroy_v2() CALLED (handle=%p, pid=%d)\n",
                          handle, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    /* No-op - just succeed */
    return CUBLAS_STATUS_SUCCESS;
}

/* CUBLAS destroy handle (non-v2 version) */
cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return cublasDestroy_v2(handle);
}

/* CUBLAS set stream */
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void *stream) {
    char log_msg[128];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasSetStream_v2() CALLED (handle=%p, stream=%p, pid=%d)\n",
                          handle, stream, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    /* No-op - just succeed */
    return CUBLAS_STATUS_SUCCESS;
}

/* CUBLAS set stream (non-v2 version) */
cublasStatus_t cublasSetStream(cublasHandle_t handle, void *stream) {
    return cublasSetStream_v2(handle, stream);
}

/* CUBLAS get stream */
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, void **stream) {
    if (!stream) return CUBLAS_STATUS_INVALID_VALUE;
    
    /* Return NULL stream */
    *stream = NULL;
    return CUBLAS_STATUS_SUCCESS;
}

/* CUBLAS get stream (non-v2 version) */
cublasStatus_t cublasGetStream(cublasHandle_t handle, void **stream) {
    return cublasGetStream_v2(handle, stream);
}

/* CUBLAS set math mode */
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode) {
    /* No-op - just succeed */
    return CUBLAS_STATUS_SUCCESS;
}

/* CUBLAS get math mode */
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, int *mode) {
    if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = 0; /* Default math mode */
    return CUBLAS_STATUS_SUCCESS;
}

/* cublasGetStatusString - get status string */
/* CRITICAL: Must be exported with correct version for GGML */
__attribute__((visibility("default")))
const char* cublasGetStatusString(cublasStatus_t status) {
    return "CUBLAS_STATUS_SUCCESS";
}

/* cublasSgemm_v2 - single precision matrix multiply */
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, int transa, int transb,
                              int m, int n, int k,
                              const float *alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              const float *beta,
                              float *C, int ldc) {
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasSgemm_v2() CALLED (m=%d, n=%d, k=%d, pid=%d)\n",
                          m, n, k, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUBLAS_STATUS_SUCCESS;
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
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUBLAS_STATUS_SUCCESS;
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
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasGemmEx() CALLED (m=%d, n=%d, k=%d, pid=%d)\n",
                          m, n, k, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUBLAS_STATUS_SUCCESS;
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
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasGemmStridedBatchedEx() CALLED (m=%d, n=%d, k=%d, batch=%d, pid=%d)\n",
                          m, n, k, batchCount, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUBLAS_STATUS_SUCCESS;
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
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cublas] cublasGemmBatchedEx() CALLED (m=%d, n=%d, k=%d, batch=%d, pid=%d)\n",
                          m, n, k, batchCount, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUBLAS_STATUS_SUCCESS;
}

/* Constructor */
__attribute__((constructor))
static void libvgpu_cublas_on_load(void) {
    const char *msg = "[libvgpu-cublas] Library loaded - CUBLAS shim initialized\n";
    syscall(__NR_write, 2, msg, 60);
}

/*
 * libvgpu_cublasLt.c - CUBLAS LT (Light) API shim library
 *
 * This library provides stub implementations for CUBLAS LT functions
 * that GGML might call during CUDA backend initialization.
 *
 * Build:
 *   gcc -shared -fPIC -o libvgpu-cublasLt.so.12 libvgpu_cublasLt.c -ldl
 *
 * Symlink:
 *   ln -sf /opt/vgpu/lib/libvgpu-cublasLt.so.12 /usr/lib64/libcublasLt.so.12
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/syscall.h>

#ifndef __NR_write
#define __NR_write 1
#endif

/* CUBLAS LT types */
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulAlgo_t;
typedef void* cublasLtMatmulPreference_t;

/* CUBLAS LT status */
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

/* Stub implementations - all return success */
cublasStatus_t cublasLtCreate(cublasLtHandle_t *handle) {
    const char *msg = "[libvgpu-cublasLt] cublasLtCreate() CALLED\n";
    syscall(__NR_write, 2, msg, 48);
    if (handle) *handle = (cublasLtHandle_t)0x1;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t handle) {
    const char *msg = "[libvgpu-cublasLt] cublasLtDestroy() CALLED\n";
    syscall(__NR_write, 2, msg, 49);
    return CUBLAS_STATUS_SUCCESS;
}

/* Generic stub for any other CUBLAS LT function */
static cublasStatus_t cublasLt_stub(const char *func_name) {
    char msg[128];
    int len = snprintf(msg, sizeof(msg), "[libvgpu-cublasLt] %s() CALLED (stub)\n", func_name);
    if (len > 0 && len < (int)sizeof(msg)) {
        syscall(__NR_write, 2, msg, len);
    }
    return CUBLAS_STATUS_SUCCESS;
}

/* Export common CUBLAS LT functions as stubs */
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, int computeType, int scaleType) {
    return cublasLt_stub("cublasLtMatmulDescCreate");
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    return cublasLt_stub("cublasLtMatmulDescDestroy");
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, int type, uint64_t rows, uint64_t cols, int64_t ld) {
    return cublasLt_stub("cublasLtMatrixLayoutCreate");
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    return cublasLt_stub("cublasLtMatrixLayoutDestroy");
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    return cublasLt_stub("cublasLtMatmulPreferenceCreate");
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    return cublasLt_stub("cublasLtMatmulPreferenceDestroy");
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                              cublasLtMatmulDesc_t operationDesc,
                                              cublasLtMatrixLayout_t Adesc,
                                              cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc,
                                              cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t preference,
                                              int requestedAlgoCount,
                                              cublasLtMatmulAlgo_t *heuristicResults,
                                              int *returnedAlgoCount) {
    return cublasLt_stub("cublasLtMatmulAlgoGetHeuristic");
}

cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                              cublasLtMatmulDesc_t computeDesc,
                              const void *alpha,
                              const void *A,
                              cublasLtMatrixLayout_t Adesc,
                              const void *B,
                              cublasLtMatrixLayout_t Bdesc,
                              const void *beta,
                              const void *C,
                              cublasLtMatrixLayout_t Cdesc,
                              void *D,
                              cublasLtMatrixLayout_t Ddesc,
                              const cublasLtMatmulAlgo_t *algo,
                              void *workspace,
                              size_t workspaceSizeInBytes,
                              void *stream) {
    return cublasLt_stub("cublasLtMatmul");
}

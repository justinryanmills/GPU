/*
 * Preflight (no Ollama): cublasGemmEx at GGML-like sizes seen before SIGSEGV
 * (journal: m=2048 n=512 k=2048; then m=256 n=512 k=2048 ×2).
 *
 * Build on VM:
 *   gcc -O2 -std=c11 -Wall -o /tmp/test_gemm_ex_vm test_gemm_ex_vm.c -ldl
 * Run (wrapper-equivalent search path):
 *   LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:$LD_LIBRARY_PATH \
 *     /tmp/test_gemm_ex_vm
 *
 * Exit 0 only if cublasGemmEx + cuCtxSynchronize succeed (matches mediated stack).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *cublasHandle_t;

enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
};

enum { CUBLAS_OP_N = 0 };
#ifndef CUDA_R_32F
#define CUDA_R_32F 0
#endif
#define CUBLAS_COMPUTE_32F 68
#define CUBLAS_GEMM_DEFAULT 0

typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef int (*cuCtxSetCurrent_t)(CUcontext);
typedef int (*cuCtxSynchronize_t)(void);
typedef int (*cuMemAlloc_v2_t)(CUdeviceptr *, size_t);
typedef int (*cuMemFree_v2_t)(CUdeviceptr);
typedef int (*cuMemcpyHtoD_v2_t)(CUdeviceptr, const void *, size_t);

typedef int (*cublasCreate_v2_t)(cublasHandle_t *);
typedef int (*cublasDestroy_v2_t)(cublasHandle_t);
typedef int (*cublasGemmEx_t)(
    cublasHandle_t, int transa, int transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta,
    void *C, int Ctype, int ldc,
    int computeType, int algo);

static int run_gemm(cublasGemmEx_t fn, cublasHandle_t h, cuCtxSynchronize_t sync,
                    int m, int n, int k,
                    CUdeviceptr da, CUdeviceptr db, CUdeviceptr dc,
                    int lda, int ldb, int ldc)
{
    float alpha = 1.f;
    float beta = 0.f;
    int st = fn(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                &alpha,
                (const void *)(uintptr_t)da, CUDA_R_32F, lda,
                (const void *)(uintptr_t)db, CUDA_R_32F, ldb,
                &beta,
                (void *)(uintptr_t)dc, CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    printf("  cublasGemmEx m=%d n=%d k=%d lda=%d ldb=%d ldc=%d -> %d\n",
           m, n, k, lda, ldb, ldc, st);
    if (st != CUBLAS_STATUS_SUCCESS)
        return st;
    if (sync) {
        int e = sync();
        printf("  cuCtxSynchronize after GemmEx -> %d\n", e);
        if (e != 0)
            return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

int main(void)
{
    void *cuda = NULL, *cublas = NULL;
    cublasHandle_t handle = NULL;
    CUcontext ctx = NULL;

    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        fprintf(stderr, "FAIL dlopen: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 = (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 = (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cublasCreate_v2_t cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 = (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmEx_t cublasGemmEx = (cublasGemmEx_t)dlsym(cublas, "cublasGemmEx");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc_v2 ||
        !cuMemFree_v2 || !cuMemcpyHtoD_v2 || !cublasCreate_v2 || !cublasDestroy_v2 ||
        !cublasGemmEx) {
        fprintf(stderr, "FAIL dlsym\n");
        return 1;
    }

    printf("=== test_gemm_ex_vm (preflight, mediated cublasGemmEx) ===\n");

    if (cuInit(0) != 0) {
        fprintf(stderr, "FAIL cuInit\n");
        return 1;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        fprintf(stderr, "FAIL cuDevicePrimaryCtxRetain\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        fprintf(stderr, "FAIL cuCtxSetCurrent\n");
        return 1;
    }

    int st = cublasCreate_v2(&handle);
    printf("cublasCreate_v2 -> %d handle=%p\n", st, (void *)handle);
    if (st != CUBLAS_STATUS_SUCCESS)
        return 1;

    /* Problem shape from long-run journal (libvgpu-cublas lines). */
    const int shapes[][3] = {
        { 2048, 512, 2048 },
        { 256, 512, 2048 },
        { 256, 512, 2048 },
    };
    const int nshape = (int)(sizeof(shapes) / sizeof(shapes[0]));

    for (int si = 0; si < nshape; si++) {
        int m = shapes[si][0];
        int n = shapes[si][1];
        int k = shapes[si][2];
        int lda = m;
        int ldb = k;
        int ldc = m;
        size_t sz_a = (size_t)m * (size_t)k * sizeof(float);
        size_t sz_b = (size_t)k * (size_t)n * sizeof(float);
        size_t sz_c = (size_t)m * (size_t)n * sizeof(float);

        float *ha = (float *)calloc(1, sz_a);
        float *hb = (float *)calloc(1, sz_b);
        float *hc = (float *)calloc(1, sz_c);
        if (!ha || !hb || !hc) {
            fprintf(stderr, "FAIL calloc shape %d\n", si);
            free(ha);
            free(hb);
            free(hc);
            cublasDestroy_v2(handle);
            return 1;
        }

        CUdeviceptr da = 0, db = 0, dc = 0;
        if (cuMemAlloc_v2(&da, sz_a) != 0 || cuMemAlloc_v2(&db, sz_b) != 0 ||
            cuMemAlloc_v2(&dc, sz_c) != 0) {
            fprintf(stderr, "FAIL cuMemAlloc shape %d\n", si);
            free(ha);
            free(hb);
            free(hc);
            cublasDestroy_v2(handle);
            return 1;
        }
        cuMemcpyHtoD_v2(da, ha, sz_a);
        cuMemcpyHtoD_v2(db, hb, sz_b);

        printf("-- shape %d/%d --\n", si + 1, nshape);
        st = run_gemm(cublasGemmEx, handle, cuCtxSynchronize, m, n, k, da, db, dc,
                      lda, ldb, ldc);

        cuMemFree_v2(dc);
        cuMemFree_v2(db);
        cuMemFree_v2(da);
        free(ha);
        free(hb);
        free(hc);

        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("PREFLIGHT_FAIL status=%d\n", st);
            cublasDestroy_v2(handle);
            return 2;
        }
    }

    cublasDestroy_v2(handle);
    printf("PREFLIGHT_OK all GemmEx + sync passed\n");
    return 0;
}

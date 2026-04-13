/*
 * Quick VM test: cublasGemmBatchedEx + Tensor Op (no Ollama, no model load).
 *
 * Hypothesis: libvgpu-cublas.c forwards cublasGemmEx via RPC but cublasGemmBatchedEx
 * only uses RESOLVE_OR_FALLBACK → real libcublas on guest → "architectural feature" etc.
 *
 * Build: gcc -O2 -Wall -Wextra -o /tmp/test_gemm_batched_ex_vm test_gemm_batched_ex_vm.c -ldl
 * Run:   LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:$LD_LIBRARY_PATH /tmp/test_gemm_batched_ex_vm
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

enum { CUBLAS_STATUS_SUCCESS = 0,
       CUBLAS_STATUS_NOT_INITIALIZED = 1,
       CUBLAS_STATUS_ARCH_MISMATCH = 8,
       CUBLAS_STATUS_EXECUTION_FAILED = 13,
       CUBLAS_STATUS_INTERNAL_ERROR = 14 };

enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };

/* CUDA 12 / cuBLAS 12 typical values */
#ifndef CUDA_R_16F
#define CUDA_R_16F 2
#endif
#ifndef CUDA_R_32F
#define CUDA_R_32F 0
#endif
#define CUBLAS_COMPUTE_16F 64
#define CUBLAS_COMPUTE_32F 68
#define CUBLAS_GEMM_DEFAULT 0
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP 99

typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef int (*cuCtxSetCurrent_t)(CUcontext);
typedef int (*cuCtxSynchronize_t)(void);
typedef int (*cuMemAlloc_v2_t)(CUdeviceptr *, size_t);
typedef int (*cuMemFree_v2_t)(CUdeviceptr);
typedef int (*cuMemcpyHtoD_v2_t)(CUdeviceptr, const void *, size_t);
typedef int (*cuMemcpyDtoH_v2_t)(void *, CUdeviceptr, size_t);
typedef const char *(*cudaGetErrorString_t)(int);

typedef int (*cublasCreate_v2_t)(cublasHandle_t *);
typedef int (*cublasDestroy_v2_t)(cublasHandle_t);
typedef int (*cublasGemmBatchedEx_t)(
    cublasHandle_t, int, int, int, int, int,
    const void *alpha,
    const void *const Aarray[], int, int,
    const void *const Barray[], int, int,
    const void *beta,
    void *const Carray[], int, int,
    int batchCount,
    int computeType, int algo);

static const char *cublas_name(int s)
{
    if (s == CUBLAS_STATUS_SUCCESS) return "SUCCESS";
    if (s == CUBLAS_STATUS_NOT_INITIALIZED) return "NOT_INITIALIZED";
    if (s == CUBLAS_STATUS_ARCH_MISMATCH) return "ARCH_MISMATCH";
    if (s == CUBLAS_STATUS_EXECUTION_FAILED) return "EXECUTION_FAILED";
    if (s == CUBLAS_STATUS_INTERNAL_ERROR) return "INTERNAL_ERROR";
    return "OTHER";
}

int main(int argc, char **argv)
{
    void *cudart = NULL, *cuda = NULL, *cublas = NULL;
    cublasHandle_t handle = NULL;
    CUcontext ctx = NULL;
    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    CUdeviceptr d_Aa = 0, d_Ba = 0, d_Ca = 0;
    uint64_t *ptrA_host = NULL, *ptrB_host = NULL, *ptrC_host = NULL;
    CUdeviceptr *d_c_batch = NULL;
    int rc = 1;

    /* Match Ollama / INVESTIGATION_CUBLASCREATE_V2 load order */
    cudart = dlopen("/opt/vgpu/lib/libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart)
        cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        printf("FAIL: dlopen cuda/cublas: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize =
        (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 =
        (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 =
        (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 =
        (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_v2_t cuMemcpyDtoH_v2 =
        (cuMemcpyDtoH_v2_t)dlsym(cuda, "cuMemcpyDtoH_v2");
    cudaGetErrorString_t cudaGetErrorString = NULL;
    if (cudart)
        cudaGetErrorString =
            (cudaGetErrorString_t)dlsym(cudart, "cudaGetErrorString");

    cublasCreate_v2_t cublasCreate_v2 =
        (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 =
        (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmBatchedEx_t cublasGemmBatchedEx =
        (cublasGemmBatchedEx_t)dlsym(cublas, "cublasGemmBatchedEx");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent ||
        !cuMemAlloc_v2 || !cuMemFree_v2 || !cuMemcpyHtoD_v2 ||
        !cublasCreate_v2 || !cublasDestroy_v2 || !cublasGemmBatchedEx) {
        printf("FAIL: dlsym missing\n");
        goto done;
    }

    /* argv[1]: 0 = CUBLAS_GEMM_DEFAULT (no Tensor Op), 1 or omit = TENSOR_OP */
    int use_tensor_op = 1;
    if (argc > 1 && argv[1][0] == '0')
        use_tensor_op = 0;
    int algo = use_tensor_op ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    int m = 32;
    int n = 32;
    int k = 32;
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) n = atoi(argv[3]);
    if (argc > 4) k = atoi(argv[4]);
    const char *dtype = (argc > 5) ? argv[5] : "f32";
    const char *ptr_mode = (argc > 6) ? argv[6] : "host";
    int batch_count = (argc > 7) ? atoi(argv[7]) : 1;
    int use_f16 = strcmp(dtype, "f16") == 0;
    int device_ptr_tables = strcmp(ptr_mode, "device") == 0;
    if (m < 1 || n < 1 || k < 1 || batch_count < 1) {
        printf("FAIL: invalid dims m=%d n=%d k=%d batch=%d\n", m, n, k, batch_count);
        return 1;
    }

    printf("=== test_gemm_batched_ex_vm (batch=%d, algo=%s, m=%d, n=%d, k=%d, dtype=%s, ptrs=%s) ===\n",
           batch_count,
           use_tensor_op ? "CUBLAS_GEMM_DEFAULT_TENSOR_OP" : "CUBLAS_GEMM_DEFAULT",
           m, n, k, dtype, ptr_mode);

    if (cuInit(0) != 0) {
        printf("FAIL: cuInit\n");
        goto done;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        printf("FAIL: cuDevicePrimaryCtxRetain\n");
        goto done;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        printf("FAIL: cuCtxSetCurrent\n");
        goto done;
    }

    printf("  cublasCreate_v2 ...\n");
    int st = cublasCreate_v2(&handle);
    printf("  cublasCreate_v2 -> %d (%s) handle=%p\n", st, cublas_name(st),
           (void *)handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        goto done;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const uint16_t alpha_f16 = 0x3c00u;
    const uint16_t beta_f16 = 0x0000u;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;
    size_t elem_size = use_f16 ? sizeof(uint16_t) : sizeof(float);
    size_t sz_a = (size_t)lda * (size_t)k * elem_size;
    size_t sz_b = (size_t)ldb * (size_t)n * elem_size;
    size_t sz_c = (size_t)ldc * (size_t)n * elem_size;
    void *ha = malloc(sz_a);
    void *hb = malloc(sz_b);
    if (!ha || !hb) {
        printf("FAIL: malloc host matrices A/B\n");
        free(ha);
        free(hb);
        goto done;
    }
    if (use_f16) {
        uint16_t *ha16 = (uint16_t *)ha;
        uint16_t *hb16 = (uint16_t *)hb;
        for (int col = 0; col < k; col++)
            for (int row = 0; row < m; row++) {
                ha16[row + col * lda] = (row == (col % m)) ? 0x3c00u : 0x0000u;
            }
        for (int col = 0; col < n; col++)
            for (int row = 0; row < k; row++) {
                hb16[row + col * ldb] = (row == (col % k)) ? 0x3c00u : 0x0000u;
            }
    } else {
        float *haf = (float *)ha;
        float *hbf = (float *)hb;
        for (int col = 0; col < k; col++)
            for (int row = 0; row < m; row++) {
                haf[row + col * lda] = (row == (col % m)) ? 1.0f : 0.0f;
            }
        for (int col = 0; col < n; col++)
            for (int row = 0; row < k; row++) {
                hbf[row + col * ldb] = (row == (col % k)) ? 1.0f : 0.0f;
            }
    }

    d_c_batch = (CUdeviceptr *)calloc((size_t)batch_count, sizeof(CUdeviceptr));
    if (!d_c_batch) {
        printf("FAIL: calloc output batches\n");
        free(ha);
        free(hb);
        goto done;
    }
    if (cuMemAlloc_v2(&d_a, sz_a) != 0 || cuMemAlloc_v2(&d_b, sz_b) != 0) {
        printf("FAIL: cuMemAlloc\n");
        free(ha);
        free(hb);
        goto done;
    }
    for (int i = 0; i < batch_count; i++) {
        if (cuMemAlloc_v2(&d_c_batch[i], sz_c) != 0) {
            printf("FAIL: cuMemAlloc output batch %d\n", i);
            free(ha);
            free(hb);
            goto done;
        }
    }
    d_c = d_c_batch[0];
    if (cuMemcpyHtoD_v2(d_a, ha, sz_a) != 0 || cuMemcpyHtoD_v2(d_b, hb, sz_b) != 0) {
        printf("FAIL: HtoD\n");
        free(ha);
        free(hb);
        goto done;
    }
    free(ha);
    free(hb);

    ptrA_host = (uint64_t *)calloc((size_t)batch_count, sizeof(uint64_t));
    ptrB_host = (uint64_t *)calloc((size_t)batch_count, sizeof(uint64_t));
    ptrC_host = (uint64_t *)calloc((size_t)batch_count, sizeof(uint64_t));
    const void *const *Aa = (const void *const *)ptrA_host;
    const void *const *Ba = (const void *const *)ptrB_host;
    void *const *Ca = (void *const *)ptrC_host;
    if (!ptrA_host || !ptrB_host || !ptrC_host) {
        printf("FAIL: calloc pointer tables\n");
        free(ptrA_host);
        free(ptrB_host);
        free(ptrC_host);
        goto done;
    }
    for (int i = 0; i < batch_count; i++) {
        ptrA_host[i] = (uint64_t)d_a;
        ptrB_host[i] = (uint64_t)d_b;
        ptrC_host[i] = (uint64_t)d_c_batch[i];
    }

    if (device_ptr_tables) {
        size_t ptr_bytes = (size_t)batch_count * sizeof(uint64_t);
        if (cuMemAlloc_v2(&d_Aa, ptr_bytes) != 0 ||
            cuMemAlloc_v2(&d_Ba, ptr_bytes) != 0 ||
            cuMemAlloc_v2(&d_Ca, ptr_bytes) != 0) {
            printf("FAIL: cuMemAlloc pointer tables\n");
            goto done;
        }
        if (cuMemcpyHtoD_v2(d_Aa, ptrA_host, ptr_bytes) != 0 ||
            cuMemcpyHtoD_v2(d_Ba, ptrB_host, ptr_bytes) != 0 ||
            cuMemcpyHtoD_v2(d_Ca, ptrC_host, ptr_bytes) != 0) {
            printf("FAIL: HtoD pointer tables\n");
            goto done;
        }
        if (cuMemcpyDtoH_v2) {
            uint64_t chkA = 0, chkB = 0, chkC = 0;
            (void)cuMemcpyDtoH_v2(&chkA, d_Aa, sizeof(chkA));
            (void)cuMemcpyDtoH_v2(&chkB, d_Ba, sizeof(chkB));
            (void)cuMemcpyDtoH_v2(&chkC, d_Ca, sizeof(chkC));
            printf("  pointer tables on device[0]: A=0x%llx B=0x%llx C=0x%llx\n",
                   (unsigned long long)chkA,
                   (unsigned long long)chkB,
                   (unsigned long long)chkC);
        }
        Aa = (const void *const *)(uintptr_t)d_Aa;
        Ba = (const void *const *)(uintptr_t)d_Ba;
        Ca = (void *const *)(uintptr_t)d_Ca;
    }

    printf("  cublasGemmBatchedEx (m=%d, n=%d, k=%d) ...\n", m, n, k);
    st = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             use_f16 ? (const void *)&alpha_f16 : (const void *)&alpha,
                             Aa, use_f16 ? CUDA_R_16F : CUDA_R_32F, lda,
                             Ba, use_f16 ? CUDA_R_16F : CUDA_R_32F, ldb,
                             use_f16 ? (const void *)&beta_f16 : (const void *)&beta,
                             Ca, use_f16 ? CUDA_R_16F : CUDA_R_32F, ldc,
                             batch_count, use_f16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F, algo);
    printf("  cublasGemmBatchedEx -> %d (%s)\n", st, cublas_name(st));
    if (cuCtxSynchronize)
        (void)cuCtxSynchronize();
    if (cudaGetErrorString) {
        typedef int (*cudaGetLastError_t)(void);
        cudaGetLastError_t gl = (cudaGetLastError_t)dlsym(cudart, "cudaGetLastError");
        if (gl) {
            int e = gl();
            printf("  cudaGetLastError -> %d (%s)\n", e,
                   cudaGetErrorString(e));
        }
    }
    /* If we see "architectural feature" or similar in the string, hypothesis matches GGML failure. */
    rc = (st == CUBLAS_STATUS_SUCCESS) ? 0 : 2;

done:
    if (d_c_batch && cuMemFree_v2) {
        for (int i = 0; i < batch_count; i++) {
            if (d_c_batch[i]) {
                cuMemFree_v2(d_c_batch[i]);
            }
        }
    }
    free(d_c_batch);
    free(ptrA_host);
    free(ptrB_host);
    free(ptrC_host);
    if (d_Ca && cuMemFree_v2)
        cuMemFree_v2(d_Ca);
    if (d_Ba && cuMemFree_v2)
        cuMemFree_v2(d_Ba);
    if (d_Aa && cuMemFree_v2)
        cuMemFree_v2(d_Aa);
    if (d_b && cuMemFree_v2)
        cuMemFree_v2(d_b);
    if (d_a && cuMemFree_v2)
        cuMemFree_v2(d_a);
    if (handle && cublasDestroy_v2)
        cublasDestroy_v2(handle);
    if (cublas)
        dlclose(cublas);
    if (cuda)
        dlclose(cuda);
    if (cudart)
        dlclose(cudart);
    printf("  exit_code=%d (0=GEMM ok, 2=GEMM failed)\n", rc);
    return rc;
}

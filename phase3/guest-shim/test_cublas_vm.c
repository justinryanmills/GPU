/*
 * VM-local CUBLAS test — run on the VM only, no host transfer.
 * Build: gcc -O2 -Wall -Wextra -o test_cublas_vm test_cublas_vm.c -ldl
 * Run:   LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH ./test_cublas_vm
 *
 * This test loads the real libcublas.so.12 together with the active libcuda
 * shim, creates a real CUDA context, performs two tiny SGEMM calls, and copies
 * the result back to host memory. The second call mirrors GGML's `T,N` usage.
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
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15
};

enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1
};

static const char *cublas_status_name(int status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    default: return "OTHER";
    }
}

static void print_matrix_2x2(const char *label, const float *m)
{
    /* cublas uses column-major layout */
    printf("%s = [[%.3f, %.3f], [%.3f, %.3f]]\n",
           label, m[0], m[2], m[1], m[3]);
}

int main(void)
{
    void *cudart = NULL;
    void *cuda = NULL;
    void *cublas = NULL;
    cublasHandle_t handle = NULL;
    CUcontext ctx = NULL;
    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    int exit_code = 1;

    typedef int (*cuInit_t)(unsigned int);
    typedef int (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
    typedef int (*cuCtxSetCurrent_t)(CUcontext);
    typedef int (*cuCtxSynchronize_t)(void);
    typedef int (*cuMemAlloc_v2_t)(CUdeviceptr *, size_t);
    typedef int (*cuMemFree_v2_t)(CUdeviceptr);
    typedef int (*cuMemcpyHtoD_v2_t)(CUdeviceptr, const void *, size_t);
    typedef int (*cuMemcpyDtoH_v2_t)(void *, CUdeviceptr, size_t);

    typedef int (*cublasCreate_v2_t)(cublasHandle_t *);
    typedef int (*cublasDestroy_v2_t)(cublasHandle_t);
    typedef int (*cublasSgemm_v2_t)(cublasHandle_t, int, int,
                                    int, int, int,
                                    const float *, const float *, int,
                                    const float *, int,
                                    const float *, float *, int);

    printf("=== CUBLAS VM-local SGEMM test (no host transfer) ===\n");

    /* Step 1 (Mar 2026): Match Ollama wrapper order — load vGPU libcudart FIRST with
     * RTLD_GLOBAL so NVIDIA libcublas (loaded later) resolves runtime symbols against
     * the shim, not a different libcudart. See INVESTIGATION_CUBLASCREATE_V2.md. */
    cudart = dlopen("/opt/vgpu/lib/libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart)
        cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (cudart)
        printf("  libcudart.so.12 loaded (prefer /opt/vgpu/lib)\n");
    else
        printf("  WARN: libcudart not preloaded: %s\n", dlerror());

    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda) {
        printf("FAIL: dlopen(libcuda.so.1) = %s\n", dlerror());
        goto cleanup;
    }
    printf("  libcuda.so.1 loaded\n");

    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cublas) {
        printf("FAIL: dlopen(libcublas.so.12) = %s\n", dlerror());
        goto cleanup;
    }
    printf("  libcublas.so.12 loaded\n");

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent =
        (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
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

    cublasCreate_v2_t cublasCreate_v2 =
        (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 =
        (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasSgemm_v2_t cublasSgemm_v2 =
        (cublasSgemm_v2_t)dlsym(cublas, "cublasSgemm_v2");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent ||
        !cuCtxSynchronize || !cuMemAlloc_v2 || !cuMemFree_v2 ||
        !cuMemcpyHtoD_v2 || !cuMemcpyDtoH_v2 ||
        !cublasCreate_v2 || !cublasDestroy_v2 || !cublasSgemm_v2) {
        printf("FAIL: missing symbol via dlsym\n");
        goto cleanup;
    }

    if (cuInit(0) != 0) {
        printf("FAIL: cuInit(0)\n");
        goto cleanup;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        printf("FAIL: cuDevicePrimaryCtxRetain(dev=0) ctx=%p\n", (void *)ctx);
        goto cleanup;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        printf("FAIL: cuCtxSetCurrent(ctx=%p)\n", (void *)ctx);
        goto cleanup;
    }
    printf("  current CUDA context: %p\n", (void *)ctx);

    /* Step 2: Many stacks pair driver primary context with runtime device selection.
     * NVIDIA cuBLAS may assume cuda runtime state; try cudaSetDevice(0) from shim. */
    if (cudart) {
        typedef int (*cudaSetDevice_t)(int);
        typedef int (*cudaGetLastError_t)(void);
        cudaSetDevice_t p_set = (cudaSetDevice_t)dlsym(cudart, "cudaSetDevice");
        cudaGetLastError_t p_err = (cudaGetLastError_t)dlsym(cudart, "cudaGetLastError");
        if (p_set) {
            int r = p_set(0);
            printf("  cudaSetDevice(0) returned: %d\n", r);
        }
        if (p_err)
            printf("  cudaGetLastError after setDevice: %d\n", p_err());
    }

    {
        int status = cublasCreate_v2(&handle);
        printf("  cublasCreate_v2 returned: %d (%s)\n",
               status, cublas_status_name(status));
        printf("  handle: %p\n", (void *)handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            goto cleanup;
        }
    }

    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        /* Column-major 2x2 matrices:
         * A = [[1, 2], [3, 4]]
         * B = [[5, 6], [7, 8]]
         */
        const float a[4] = {1.0f, 3.0f, 2.0f, 4.0f};
        const float b[4] = {5.0f, 7.0f, 6.0f, 8.0f};
        float c_nn[4] = {0};
        float c_tn[4] = {0};

        size_t bytes = sizeof(a);
        if (cuMemAlloc_v2(&d_a, bytes) != 0 ||
            cuMemAlloc_v2(&d_b, bytes) != 0 ||
            cuMemAlloc_v2(&d_c, bytes) != 0) {
            printf("FAIL: device allocation\n");
            goto cleanup;
        }

        if (cuMemcpyHtoD_v2(d_a, a, bytes) != 0 ||
            cuMemcpyHtoD_v2(d_b, b, bytes) != 0) {
            printf("FAIL: host-to-device copy\n");
            goto cleanup;
        }

        print_matrix_2x2("  A", a);
        print_matrix_2x2("  B", b);

        {
            int status = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        2, 2, 2,
                                        &alpha,
                                        (const float *)(uintptr_t)d_a, 2,
                                        (const float *)(uintptr_t)d_b, 2,
                                        &beta,
                                        (float *)(uintptr_t)d_c, 2);
            printf("  cublasSgemm_v2(N,N) returned: %d (%s)\n",
                   status, cublas_status_name(status));
            if (status == CUBLAS_STATUS_SUCCESS) {
                if (cuCtxSynchronize() != 0 ||
                    cuMemcpyDtoH_v2(c_nn, d_c, sizeof(c_nn)) != 0) {
                    printf("FAIL: sync/copy after N,N GEMM\n");
                    goto cleanup;
                }
                print_matrix_2x2("  C(N,N)", c_nn);
            }
        }

        memset(c_tn, 0, sizeof(c_tn));
        if (cuMemcpyHtoD_v2(d_c, c_tn, sizeof(c_tn)) != 0) {
            printf("FAIL: clear output buffer for T,N GEMM\n");
            goto cleanup;
        }

        {
            int status = cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        2, 2, 2,
                                        &alpha,
                                        (const float *)(uintptr_t)d_a, 2,
                                        (const float *)(uintptr_t)d_b, 2,
                                        &beta,
                                        (float *)(uintptr_t)d_c, 2);
            printf("  cublasSgemm_v2(T,N) returned: %d (%s)\n",
                   status, cublas_status_name(status));
            if (status == CUBLAS_STATUS_SUCCESS) {
                if (cuCtxSynchronize() != 0 ||
                    cuMemcpyDtoH_v2(c_tn, d_c, sizeof(c_tn)) != 0) {
                    printf("FAIL: sync/copy after T,N GEMM\n");
                    goto cleanup;
                }
                print_matrix_2x2("  C(T,N)", c_tn);
            }
        }
    }

    exit_code = 0;

cleanup:
    if (d_c && cuMemFree_v2) (void)cuMemFree_v2(d_c);
    if (d_b && cuMemFree_v2) (void)cuMemFree_v2(d_b);
    if (d_a && cuMemFree_v2) (void)cuMemFree_v2(d_a);
    if (handle && cublasDestroy_v2) (void)cublasDestroy_v2(handle);
    if (cublas) dlclose(cublas);
    if (cuda) dlclose(cuda);
    if (cudart) dlclose(cudart);
    return exit_code;
}

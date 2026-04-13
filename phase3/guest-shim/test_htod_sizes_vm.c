#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef void *CUcontext;

typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef int (*cuCtxSetCurrent_t)(CUcontext);
typedef int (*cuMemAlloc_v2_t)(CUdeviceptr *, size_t);
typedef int (*cuMemFree_v2_t)(CUdeviceptr);
typedef int (*cuMemcpyHtoD_v2_t)(CUdeviceptr, const void *, size_t);
typedef int (*cuMemcpyDtoH_v2_t)(void *, CUdeviceptr, size_t);

static void fill_pattern(uint8_t *buf, size_t len, uint8_t seed)
{
    for (size_t i = 0; i < len; i++) {
        buf[i] = (uint8_t)(seed + (i % 251u));
    }
}

static int run_one(size_t len,
                   cuMemAlloc_v2_t cuMemAlloc_v2,
                   cuMemFree_v2_t cuMemFree_v2,
                   cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2,
                   cuMemcpyDtoH_v2_t cuMemcpyDtoH_v2)
{
    uint8_t *h_in = malloc(len);
    uint8_t *h_out = malloc(len);
    CUdeviceptr dptr = 0;
    int rc = 0;

    if (!h_in || !h_out) {
        free(h_in);
        free(h_out);
        return 100;
    }

    memset(h_out, 0, len);
    fill_pattern(h_in, len, (uint8_t)(len & 0xff));

    if (cuMemAlloc_v2(&dptr, len) != 0) {
        free(h_in);
        free(h_out);
        return 101;
    }

    rc = cuMemcpyHtoD_v2(dptr, h_in, len);
    if (rc != 0) {
        printf("SIZE=%zu HTOD_RC=%d\n", len, rc);
        cuMemFree_v2(dptr);
        free(h_in);
        free(h_out);
        return 102;
    }

    rc = cuMemcpyDtoH_v2(h_out, dptr, len);
    if (rc != 0) {
        printf("SIZE=%zu DTOH_RC=%d\n", len, rc);
        cuMemFree_v2(dptr);
        free(h_in);
        free(h_out);
        return 103;
    }

    printf("SIZE=%zu SAME=%d IN=%02x%02x%02x%02x OUT=%02x%02x%02x%02x\n",
           len, memcmp(h_in, h_out, len) == 0,
           h_in[0], h_in[1], h_in[2], h_in[3],
           h_out[0], h_out[1], h_out[2], h_out[3]);

    cuMemFree_v2(dptr);
    free(h_in);
    free(h_out);
    return 0;
}

int main(void)
{
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    CUcontext ctx = NULL;

    if (!cuda) {
        fprintf(stderr, "FAIL dlopen(libcuda.so.1): %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent =
        (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuMemAlloc_v2_t cuMemAlloc_v2 =
        (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 =
        (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 =
        (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_v2_t cuMemcpyDtoH_v2 =
        (cuMemcpyDtoH_v2_t)dlsym(cuda, "cuMemcpyDtoH_v2");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent ||
        !cuMemAlloc_v2 || !cuMemFree_v2 ||
        !cuMemcpyHtoD_v2 || !cuMemcpyDtoH_v2) {
        fprintf(stderr, "FAIL dlsym\n");
        return 2;
    }

    if (cuInit(0) != 0) return 3;
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) return 4;
    if (cuCtxSetCurrent(ctx) != 0) return 5;

    size_t sizes[] = {
        16u,
        1024u,
        8192u,
        131072u,
        294912u,
        1048576u,
        2097152u,
        4194304u,
        6291456u,
        8388608u,
    };

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        int rc = run_one(sizes[i],
                         cuMemAlloc_v2,
                         cuMemFree_v2,
                         cuMemcpyHtoD_v2,
                         cuMemcpyDtoH_v2);
        if (rc != 0) {
            return rc;
        }
    }

    return 0;
}

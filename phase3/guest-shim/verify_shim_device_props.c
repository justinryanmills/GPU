/*
 * Runtime check (no model / no Ollama): load libcudart with the same
 * LD_LIBRARY_PATH as Ollama and verify sharedMemPerBlockOptin values.
 *
 * Build on VM (link the shim directly; symlinks under cuda_v12 point at the shim):
 *   gcc -std=c11 -O -Wall -o verify_shim_device_props verify_shim_device_props.c \
 *     /opt/vgpu/lib/libvgpu-cudart.so -ldl -Wl,-rpath,/opt/vgpu/lib
 *   ./verify_shim_device_props
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaDevAttrMaxSharedMemoryPerBlockOptin 97

/* libvgpu_cudart.c cudaDeviceProp: sharedMemPerBlockOptin @ 0x1A8, maxSharedMemoryPerBlockOptin @ 0x1B0 */
#define OFF_SHARED_MEM_PER_BLOCK_OPTIN       0x1A8
#define OFF_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN  0x1B0
#define OFF_CC_NEW_MAJOR 0x148
#define OFF_CC_NEW_MINOR 0x14C
#define OFF_CC_OLD_MAJOR 0x15C
#define OFF_CC_OLD_MINOR 0x160
#define OFF_CC_ALT_MAJOR 0x168
#define OFF_CC_ALT_MINOR 0x16C

extern cudaError_t cudaGetDeviceProperties_v2(void *prop, int device);
extern cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device);

#ifndef EXPECT_OPTIN
#define EXPECT_OPTIN 227328
#endif

int main(void) {
    unsigned char raw[4096];
    memset(raw, 0, sizeof(raw));

    cudaError_t e = cudaGetDeviceProperties_v2(raw, 0);
    if (e != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaGetDeviceProperties_v2 -> %d\n", (int)e);
        return 2;
    }

    int optin = *(const int *)(raw + OFF_SHARED_MEM_PER_BLOCK_OPTIN);
    int maxopt = *(const int *)(raw + OFF_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    int cc_new_major = *(const int *)(raw + OFF_CC_NEW_MAJOR);
    int cc_new_minor = *(const int *)(raw + OFF_CC_NEW_MINOR);
    int cc_old_major = *(const int *)(raw + OFF_CC_OLD_MAJOR);
    int cc_old_minor = *(const int *)(raw + OFF_CC_OLD_MINOR);
    int cc_alt_major = *(const int *)(raw + OFF_CC_ALT_MAJOR);
    int cc_alt_minor = *(const int *)(raw + OFF_CC_ALT_MINOR);

    int attr97 = -1;
    cudaError_t e2 = cudaDeviceGetAttribute(&attr97, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (e2 != cudaSuccess) {
        fprintf(stderr, "FAIL: cudaDeviceGetAttribute(97) -> %d\n", (int)e2);
        return 3;
    }

    printf("cudaGetDeviceProperties_v2 (offsets 0x1A8 / 0x1B0):\n");
    printf("  sharedMemPerBlockOptin         = %d\n", optin);
    printf("  maxSharedMemoryPerBlockOptin   = %d\n", maxopt);
    printf("candidate compute-capability offsets:\n");
    printf("  0x148 / 0x14C                 = %d / %d\n", cc_new_major, cc_new_minor);
    printf("  0x15C / 0x160                 = %d / %d\n", cc_old_major, cc_old_minor);
    printf("  0x168 / 0x16C                 = %d / %d\n", cc_alt_major, cc_alt_minor);
    printf("cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin=%d):\n", cudaDevAttrMaxSharedMemoryPerBlockOptin);
    printf("  value                          = %d\n", attr97);
    printf("expected (Hopper MMQ fix)        = %d\n", EXPECT_OPTIN);

    if (optin != EXPECT_OPTIN || maxopt != EXPECT_OPTIN || attr97 != EXPECT_OPTIN) {
        fprintf(stderr, "FAIL: opt-in shared mem fields do not match expected %d\n", EXPECT_OPTIN);
        return 1;
    }

    printf("OK: shim reports Hopper-class opt-in shared memory (no model load required).\n");
    return 0;
}

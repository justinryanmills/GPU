#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("CUDA devices: %d\n", count);
    if (count < 1) return 1;

    cudaSetDevice(0);
    void *p = NULL;
    err = cudaMalloc(&p, 4);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaFree(p);
    printf("CUDA sanity OK\n");
    return 0;
}
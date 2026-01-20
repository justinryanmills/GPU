#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" int run_vector_add_test(char *result_msg, int max_len) {
    const int N = 1024;
    const int SIZE = N * sizeof(float);
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int errors = 0;
    cudaError_t err;
    
    h_a = (float*)malloc(SIZE);
    h_b = (float*)malloc(SIZE);
    h_c = (float*)malloc(SIZE);
    
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    cudaSetDevice(0);
    err = cudaMalloc(&d_a, SIZE);
    if (err != cudaSuccess) {
        snprintf(result_msg, max_len, "%s", cudaGetErrorString(err));
        free(h_a); free(h_b); free(h_c);
        return -1;
    }
    err = cudaMalloc(&d_b, SIZE);
    if (err != cudaSuccess) {
        snprintf(result_msg, max_len, "%s", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a); free(h_b); free(h_c);
        return -1;
    }
    err = cudaMalloc(&d_c, SIZE);
    if (err != cudaSuccess) {
        snprintf(result_msg, max_len, "%s", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b);
        free(h_a); free(h_b); free(h_c);
        return -1;
    }
    
    cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(result_msg, max_len, "%s", cudaGetErrorString(err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return -1;
    }
    
    cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        if (h_c[i] != (h_a[i] + h_b[i])) {
            errors++;
        }
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    if (errors == 0) {
        snprintf(result_msg, max_len, "");
        return 0;
    } else {
        snprintf(result_msg, max_len, "%d", errors);
        return -1;
    }
}
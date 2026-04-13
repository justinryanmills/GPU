/*
 * CUDA Vector Addition Implementation
 * 
 * Purpose: Asynchronous CUDA vector addition with callback mechanism
 * 
 * Features:
 * - CUDA kernel for vector addition
 * - Asynchronous execution (non-blocking)
 * - Callback when result ready
 * - Thread-safe operation
 * 
 * Usage:
 *   #include "cuda_vector_add.h"
 *   cuda_init();
 *   cuda_vector_add_async(100, 200, my_callback, user_data);
 *   cuda_cleanup();
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error checking macro (for use in non-void functions)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

// Callback function type
typedef void (*cuda_callback_t)(int result, void *user_data);

// Structure to hold async operation context
typedef struct {
    int num1;
    int num2;
    cuda_callback_t callback;
    void *user_data;
    pthread_t thread;
} AsyncContext;

// Global state
static int cuda_initialized = 0;
static cudaStream_t stream = NULL;

/*
 * CUDA kernel for vector addition
 * Simple addition: c[0] = a[0] + b[0]
 */
__global__ void vector_add_kernel(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {  // Only first thread computes
        c[0] = a[0] + b[0];
    }
}

/*
 * Worker thread function for async execution
 */
static void* async_worker(void *arg) {
    AsyncContext *ctx = (AsyncContext *)arg;
    int *d_a = NULL, *d_b = NULL, *d_c = NULL;  // Device pointers
    int h_a, h_b, h_c;     // Host values
    int result = 0;
    cudaError_t err;
    
    // Set host values
    h_a = ctx->num1;
    h_b = ctx->num2;
    h_c = 0;
    
    // Allocate device memory
    err = cudaMalloc((void**)&d_a, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaMalloc d_a: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&d_b, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaMalloc d_b: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&d_c, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaMalloc d_c: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Copy input data to device
    err = cudaMemcpyAsync(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaMemcpyAsync d_a: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    err = cudaMemcpyAsync(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaMemcpyAsync d_b: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Launch kernel (1 block, 1 thread)
    vector_add_kernel<<<1, 1, 0, stream>>>(d_a, d_b, d_c);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: kernel launch: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Copy result back to host
    err = cudaMemcpyAsync(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaMemcpyAsync result: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Synchronize stream to ensure completion
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: cudaStreamSynchronize: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    result = h_c;
    
    // Call callback with result
    if (ctx->callback) {
        ctx->callback(result, ctx->user_data);
    }
    
cleanup:
    // Free device memory
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    
    // Free context
    free(ctx);
    
    return NULL;
}

/*
 * Initialize CUDA
 * Returns 0 on success, -1 on failure
 */
int cuda_init(void) {
    if (cuda_initialized) {
        return 0;  // Already initialized
    }
    
    // Check for CUDA-capable device
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA-capable devices found\n");
        return -1;
    }
    
    // Set device 0
    CUDA_CHECK(cudaSetDevice(0));
    
    // Create CUDA stream for async operations
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    cuda_initialized = 1;
    printf("[CUDA] Initialized successfully (device count: %d)\n", device_count);
    
    return 0;
}

/*
 * Execute vector addition asynchronously
 * 
 * Parameters:
 *   num1, num2: Numbers to add
 *   callback: Function to call when result is ready
 *   user_data: User data passed to callback
 * 
 * Returns:
 *   0 on success (operation started)
 *   -1 on failure
 */
int cuda_vector_add_async(int num1, int num2, cuda_callback_t callback, void *user_data) {
    if (!cuda_initialized) {
        fprintf(stderr, "ERROR: CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }
    
    if (!callback) {
        fprintf(stderr, "ERROR: Callback function is required\n");
        return -1;
    }
    
    // Allocate context
    AsyncContext *ctx = (AsyncContext *)malloc(sizeof(AsyncContext));
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to allocate async context\n");
        return -1;
    }
    
    // Set context values
    ctx->num1 = num1;
    ctx->num2 = num2;
    ctx->callback = callback;
    ctx->user_data = user_data;
    
    // Create worker thread
    int ret = pthread_create(&ctx->thread, NULL, async_worker, ctx);
    if (ret != 0) {
        fprintf(stderr, "ERROR: Failed to create worker thread: %s\n", strerror(ret));
        free(ctx);
        return -1;
    }
    
    // Detach thread (we don't need to join)
    pthread_detach(ctx->thread);
    
    printf("[CUDA] Started async vector addition: %d + %d\n", num1, num2);
    
    return 0;
}

/*
 * Check if CUDA is busy
 * Note: This is a simple check. For production, track active operations.
 * 
 * Returns:
 *   1 if CUDA is busy
 *   0 if CUDA is idle
 */
int cuda_is_busy(void) {
    // Simple implementation: check if stream has pending operations
    // For a more accurate implementation, track active async contexts
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaErrorNotReady) {
        return 1;  // Stream has pending operations
    }
    return 0;  // Stream is idle
}

/*
 * Wait for all pending CUDA operations to complete
 */
int cuda_sync(void) {
    if (!cuda_initialized) {
        return -1;
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}

/*
 * Cleanup CUDA resources
 */
void cuda_cleanup(void) {
    if (!cuda_initialized) {
        return;
    }
    
    // Wait for all operations to complete
    cuda_sync();
    
    // Destroy stream
    if (stream) {
        cudaStreamDestroy(stream);
        stream = NULL;
    }
    
    cuda_initialized = 0;
    printf("[CUDA] Cleaned up\n");
}

#ifdef __cplusplus
}
#endif

/*
 * Test function (for standalone testing)
 */
#ifdef CUDA_VECTOR_ADD_TEST
static void test_callback(int result, void *user_data) {
    printf("[TEST] Result: %d\n", result);
    int *done = (int *)user_data;
    *done = 1;
}

int main(int argc, char *argv[]) {
    printf("CUDA Vector Addition Test\n");
    printf("=======================\n\n");
    
    // Initialize
    if (cuda_init() != 0) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }
    
    // Test 1: Simple addition
    printf("Test 1: 100 + 200\n");
    int done1 = 0;
    if (cuda_vector_add_async(100, 200, test_callback, &done1) != 0) {
        fprintf(stderr, "Failed to start async operation\n");
        cuda_cleanup();
        return 1;
    }
    
    // Wait for completion
    while (!done1) {
        usleep(10000);  // 10ms
    }
    
    // Test 2: Another addition
    printf("\nTest 2: 50 + 75\n");
    int done2 = 0;
    if (cuda_vector_add_async(50, 75, test_callback, &done2) != 0) {
        fprintf(stderr, "Failed to start async operation\n");
        cuda_cleanup();
        return 1;
    }
    
    // Wait for completion
    while (!done2) {
        usleep(10000);  // 10ms
    }
    
    // Cleanup
    cuda_cleanup();
    
    printf("\nAll tests completed successfully!\n");
    return 0;
}
#endif

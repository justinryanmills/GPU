/* Simple test to verify Runtime API shim is being used */
#include <stdio.h>
#include <dlfcn.h>

/* Declare CUDA Runtime API function */
typedef int cudaError_t;
#define cudaSuccess 0
cudaError_t cudaGetDeviceCount(int *count);

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    
    printf("cudaGetDeviceCount() returned: err=%d, count=%d\n", err, count);
    
    if (err == cudaSuccess && count == 1) {
        printf("SUCCESS: Runtime API shim is working!\n");
        return 0;
    } else {
        printf("FAILED: Runtime API shim might not be working\n");
        return 1;
    }
}

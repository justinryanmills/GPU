#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("=== Testing Shim Library Loading ===\n");
    
    // Test 1: Try dlopen
    printf("\n1. Testing dlopen(\"libcuda.so.1\"):\n");
    void *h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (h) {
        printf("   SUCCESS: libcuda.so.1 loaded\n");
        
        // Test 2: Try to find cuInit
        printf("\n2. Testing dlsym(h, \"cuInit\"):\n");
        void (*cuInit)(unsigned int) = (void (*)(unsigned int))dlsym(h, "cuInit");
        if (cuInit) {
            printf("   SUCCESS: cuInit symbol found\n");
            
            // Test 3: Try calling cuInit
            printf("\n3. Testing cuInit(0):\n");
            typedef int (*cuInit_t)(unsigned int);
            cuInit_t cuInit_func = (cuInit_t)cuInit;
            int result = cuInit_func(0);
            printf("   cuInit returned: %d (0 = success)\n", result);
        } else {
            printf("   FAILED: cuInit not found: %s\n", dlerror());
        }
        
        dlclose(h);
    } else {
        printf("   FAILED: %s\n", dlerror());
    }
    
    // Test 4: Check if shim is in preload
    printf("\n4. Checking /etc/ld.so.preload:\n");
    FILE *f = fopen("/etc/ld.so.preload", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "libvgpu")) {
                printf("   FOUND: %s", line);
            }
        }
        fclose(f);
    } else {
        printf("   No /etc/ld.so.preload file\n");
    }
    
    return 0;
}

#include <stdio.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;
typedef int CUdevice_attribute;

CUresult (*cuInit)(unsigned int) = NULL;
CUresult (*cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev) = NULL;

int main() {
    void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Cannot load libcuda.so.1\n");
        return 1;
    }
    
    cuInit = (CUresult (*)(unsigned int))dlsym(handle, "cuInit");
    cuDeviceGetAttribute = (CUresult (*)(int *, CUdevice_attribute, CUdevice))dlsym(handle, "cuDeviceGetAttribute");
    
    if (!cuInit || !cuDeviceGetAttribute) {
        fprintf(stderr, "Cannot find functions\n");
        return 1;
    }
    
    cuInit(0);
    int value = 0;
    CUresult res = cuDeviceGetAttribute(&value, 1, 0);  // 1 = MAX_THREADS_PER_BLOCK
    printf("MAX_THREADS_PER_BLOCK = %d (result=%d)\n", value, res);
    
    dlclose(handle);
    return 0;
}

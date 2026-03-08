/*
 * ggml_mmap_intercept.c - Intercept mmap to ensure 32-byte alignment
 * 
 * GGML may use mmap() for buffer allocation. This library ensures all
 * mmap allocations are 32-byte aligned to meet GGML's TENSOR_ALIGNMENT requirement.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <errno.h>

#define __NR_write 1
#define TENSOR_ALIGNMENT 32

/* Real mmap function */
static void *(*real_mmap)(void *, size_t, int, int, int, off_t) = NULL;

/* Initialize real function pointer */
static void init_mmap_hook(void) {
    if (real_mmap) return;
    real_mmap = (void *(*)(void *, size_t, int, int, int, off_t))
                dlsym(RTLD_NEXT, "mmap");
}

/* Intercept mmap to ensure 32-byte alignment */
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    init_mmap_hook();
    
    if (!real_mmap) {
        errno = ENOSYS;
        return MAP_FAILED;
    }
    
    /* If addr is specified, ensure it's aligned */
    if (addr != NULL && (uintptr_t)addr % TENSOR_ALIGNMENT != 0) {
        /* Align the address */
        uintptr_t aligned_addr = ((uintptr_t)addr + TENSOR_ALIGNMENT - 1) & ~(TENSOR_ALIGNMENT - 1);
        
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                              "[ggml-mmap-intercept] mmap() addr alignment fix: 0x%llx -> 0x%llx (pid=%d)\n",
                              (unsigned long long)addr, (unsigned long long)aligned_addr, (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
            syscall(__NR_write, 2, log_msg, log_len);
        }
        
        addr = (void *)aligned_addr;
    }
    
    /* Call real mmap */
    void *result = real_mmap(addr, length, prot, flags, fd, offset);
    
    /* If mmap succeeded but result is not aligned, we need to handle it */
    if (result != MAP_FAILED && (uintptr_t)result % TENSOR_ALIGNMENT != 0) {
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                              "[ggml-mmap-intercept] mmap() WARNING: result 0x%llx is not %d-byte aligned (pid=%d)\n",
                              (unsigned long long)result, TENSOR_ALIGNMENT, (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
            syscall(__NR_write, 2, log_msg, log_len);
        }
        
        /* Try to remap with aligned address if MAP_FIXED is not set */
        if (!(flags & MAP_FIXED)) {
            /* Unmap and remap with aligned address */
            munmap(result, length);
            
            /* Calculate aligned address */
            uintptr_t aligned_result = ((uintptr_t)result + TENSOR_ALIGNMENT - 1) & ~(TENSOR_ALIGNMENT - 1);
            size_t aligned_length = length + (aligned_result - (uintptr_t)result);
            
            result = real_mmap((void *)aligned_result, aligned_length, prot, flags, fd, offset);
            
            if (result != MAP_FAILED) {
                log_len = snprintf(log_msg, sizeof(log_msg),
                                  "[ggml-mmap-intercept] mmap() remapped to aligned address 0x%llx (pid=%d)\n",
                                  (unsigned long long)result, (int)getpid());
                if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
                    syscall(__NR_write, 2, log_msg, log_len);
                }
            }
        }
    }
    
    return result;
}

/* Intercept munmap */
int munmap(void *addr, size_t length) {
    static int (*real_munmap)(void *, size_t) = NULL;
    if (!real_munmap) {
        real_munmap = (int (*)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
    }
    if (real_munmap) {
        return real_munmap(addr, length);
    }
    return -1;
}

/* Constructor */
__attribute__((constructor))
static void ggml_mmap_intercept_on_load(void) {
    const char *msg = "[ggml-mmap-intercept] Library loaded - will ensure mmap() returns 32-byte aligned addresses\n";
    syscall(__NR_write, 2, msg, 88);
}

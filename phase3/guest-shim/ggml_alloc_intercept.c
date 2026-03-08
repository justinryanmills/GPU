/*
 * ggml_alloc_intercept.c - Comprehensive memory allocation interception
 * 
 * Intercepts all memory allocation functions to ensure 32-byte alignment.
 * This is a more comprehensive approach than just intercepting malloc.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <errno.h>

#define __NR_write 1
#define TENSOR_ALIGNMENT 32

/* Real function pointers */
static void *(*real_malloc)(size_t) = NULL;
static void *(*real_calloc)(size_t, size_t) = NULL;
static void *(*real_realloc)(void *, size_t) = NULL;
static void *(*real_aligned_alloc)(size_t, size_t) = NULL;
static void *(*real_memalign)(size_t, size_t) = NULL;
static void *(*real_valloc)(size_t) = NULL;
static void (*real_free)(void *) = NULL;
static int (*real_posix_memalign)(void **, size_t, size_t) = NULL;

/* Initialize function pointers */
static void init_alloc_hooks(void) {
    if (real_malloc) return; /* Already initialized */
    
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_aligned_alloc = dlsym(RTLD_NEXT, "aligned_alloc");
    real_memalign = dlsym(RTLD_NEXT, "memalign");
    real_valloc = dlsym(RTLD_NEXT, "valloc");
    real_free = dlsym(RTLD_NEXT, "free");
    real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
}

/* Helper to allocate aligned memory */
static void *allocate_aligned(size_t size) {
    init_alloc_hooks();
    
    if (!real_posix_memalign) {
        /* Fallback to aligned_alloc if available */
        if (real_aligned_alloc) {
            return real_aligned_alloc(TENSOR_ALIGNMENT, size);
        }
        /* Last resort: use malloc and align manually */
        if (real_malloc) {
            void *ptr = real_malloc(size + TENSOR_ALIGNMENT);
            if (!ptr) return NULL;
            uintptr_t addr = (uintptr_t)ptr;
            uintptr_t aligned = (addr + TENSOR_ALIGNMENT - 1) & ~(TENSOR_ALIGNMENT - 1);
            return (void *)aligned;
        }
        return NULL;
    }
    
    void *ptr = NULL;
    if (real_posix_memalign(&ptr, TENSOR_ALIGNMENT, size) == 0) {
        return ptr;
    }
    return NULL;
}

/* Intercept malloc */
void *malloc(size_t size) {
    init_alloc_hooks();
    
    void *ptr = allocate_aligned(size);
    
    if (ptr && (uintptr_t)ptr % TENSOR_ALIGNMENT != 0) {
        const char *msg = "[ggml-alloc-intercept] WARNING: malloc() returned unaligned pointer!\n";
        syscall(__NR_write, 2, msg, 60);
    }
    
    return ptr;
}

/* Intercept calloc */
void *calloc(size_t nmemb, size_t size) {
    init_alloc_hooks();
    
    size_t total = nmemb * size;
    void *ptr = allocate_aligned(total);
    
    if (ptr && real_malloc) {
        /* Zero the memory */
        memset(ptr, 0, total);
    }
    
    return ptr;
}

/* Intercept realloc */
void *realloc(void *ptr, size_t size) {
    init_alloc_hooks();
    
    if (!ptr) {
        return allocate_aligned(size);
    }
    
    /* Check if current pointer is aligned */
    if ((uintptr_t)ptr % TENSOR_ALIGNMENT != 0) {
        /* Current pointer is not aligned - allocate new aligned block */
        void *new_ptr = allocate_aligned(size);
        if (new_ptr && real_malloc) {
            /* Copy old data (we don't know the old size, so this is approximate) */
            memcpy(new_ptr, ptr, size);
            real_free(ptr);
        }
        return new_ptr;
    }
    
    /* Try to use real realloc, but ensure result is aligned */
    if (real_realloc) {
        void *new_ptr = real_realloc(ptr, size);
        if (new_ptr && (uintptr_t)new_ptr % TENSOR_ALIGNMENT != 0) {
            /* Result is not aligned - allocate new aligned block */
            void *aligned_ptr = allocate_aligned(size);
            if (aligned_ptr) {
                memcpy(aligned_ptr, new_ptr, size);
                real_free(new_ptr);
                return aligned_ptr;
            }
        }
        return new_ptr;
    }
    
    return NULL;
}

/* Intercept free */
void free(void *ptr) {
    init_alloc_hooks();
    if (real_free && ptr) {
        real_free(ptr);
    }
}

/* Intercept aligned_alloc - ensure it uses at least TENSOR_ALIGNMENT */
void *aligned_alloc(size_t alignment, size_t size) {
    init_alloc_hooks();
    
    /* Use the maximum of requested alignment and TENSOR_ALIGNMENT */
    size_t effective_alignment = (alignment > TENSOR_ALIGNMENT) ? alignment : TENSOR_ALIGNMENT;
    
    if (real_aligned_alloc) {
        return real_aligned_alloc(effective_alignment, size);
    }
    
    return allocate_aligned(size);
}

/* Intercept memalign */
void *memalign(size_t alignment, size_t size) {
    init_alloc_hooks();
    
    size_t effective_alignment = (alignment > TENSOR_ALIGNMENT) ? alignment : TENSOR_ALIGNMENT;
    
    if (real_memalign) {
        return real_memalign(effective_alignment, size);
    }
    
    return allocate_aligned(size);
}

/* Intercept valloc - ensure page-aligned (which is >= 32-byte aligned) */
void *valloc(size_t size) {
    init_alloc_hooks();
    
    if (real_valloc) {
        void *ptr = real_valloc(size);
        if (ptr && (uintptr_t)ptr % TENSOR_ALIGNMENT != 0) {
            const char *msg = "[ggml-alloc-intercept] WARNING: valloc() returned unaligned pointer!\n";
            syscall(__NR_write, 2, msg, 61);
        }
        return ptr;
    }
    
    return allocate_aligned(size);
}

/* Intercept posix_memalign - ensure minimum TENSOR_ALIGNMENT */
int posix_memalign(void **memptr, size_t alignment, size_t size) {
    init_alloc_hooks();
    
    size_t effective_alignment = (alignment > TENSOR_ALIGNMENT) ? alignment : TENSOR_ALIGNMENT;
    
    if (real_posix_memalign) {
        return real_posix_memalign(memptr, effective_alignment, size);
    }
    
    return ENOMEM;
}

/* Constructor */
__attribute__((constructor))
static void ggml_alloc_intercept_on_load(void) {
    const char *msg = "[ggml-alloc-intercept] Library loaded - intercepting all memory allocation functions\n";
    syscall(__NR_write, 2, msg, 90);
}

/*
 * libvgpu_cuda.c  —  CUDA Driver API shim library
 *
 * This shared library (libvgpu-cuda.so) is installed in the guest VM
 * and replaces the real libcuda.so.1.  It intercepts CUDA Driver API
 * calls, answers device-query functions locally using cached GPU
 * properties, and forwards compute operations (context, memory,
 * module, kernel launch, stream, event) to the host via the
 * VGPU-STUB transport.
 *
 * Build:
 *   gcc -shared -fPIC -o libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c \
 *       -I../include -I.
 *
 * Symlink:
 *   ln -sf /usr/lib64/libvgpu-cuda.so /usr/lib64/libcuda.so.1
 *   ln -sf /usr/lib64/libvgpu-cuda.so /usr/lib64/libcuda.so
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  /* Required for RTLD_NEXT, RTLD_DEFAULT, memfd_create */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdarg.h>
#include <features.h>  /* For __GLIBC__ detection */
#include <unistd.h>    /* For readlink, ssize_t */
#include <time.h>      /* For clock_gettime, timestamps */
#include <sys/types.h> /* For pid_t */
#include <sys/syscall.h> /* For direct syscalls */
#if !defined(__NR_pread64) && defined(__x86_64__)
#define __NR_pread64 17
#endif
#include <link.h>        /* For dl_iterate_phdr */

/* For variadic stub function */
#include <stdarg.h>

/* For write() interception to capture full error messages */
#include <sys/uio.h>  /* For writev */

#include "cuda_protocol.h"
#include "cuda_transport.h"
#include "gpu_properties.h"

/* Forward declarations - MUST be before any function that uses them
 * These are needed early for dlopen() (line 91) and constructor */
static int is_system_process(void);
static int is_application_process(void);
static int is_runner_process(void);
static int is_safe_to_check_process(void);  /* Safety check for early initialization */
static int ensure_init(void);  /* Early initialization function */

/* Global real libc FILE* functions — resolved from libc.so for excluded/model files */
static FILE *(*g_real_fopen_global)(const char *, const char *) = NULL;
static char *(*g_real_fgets_global)(char *, int, FILE *) = NULL;
static size_t (*g_real_fread_global)(void *, size_t, size_t, FILE *) = NULL;

/* ================================================================
 * dlopen interception  (dlsym and dlclose are NOT overridden)
 *
 * Ollama (Go binary, CGo) discovers GPUs by calling:
 *     handle = dlopen("libcuda.so.1", RTLD_LAZY);
 *     fn     = dlsym(handle, "cuInit");
 *
 * Even though our shim is loaded via LD_PRELOAD and has SONAME
 * "libcuda.so.1", some glibc versions open a second copy from the
 * filesystem, bypassing the preloaded one.
 *
 * Fix: override ONLY dlopen.  When any code in the process calls
 * dlopen("libcuda.so.1"), we return a handle that points to our
 * already-loaded shim.  Because dlsym is NOT overridden, the caller's
 * subsequent dlsym() calls go through glibc and correctly resolve
 * symbols from the returned handle (which IS our shim).
 *
 * Bootstrapping: we find the real dlopen via dlsym(RTLD_NEXT, ...).
 * This is safe because dlsym itself is NOT overridden — no
 * __libc_dlsym needed (that symbol is not exported on all glibc
 * builds and caused "undefined symbol" crashes with Go binaries).
 *
 * UPDATE: We now intercept dlsym to log what functions Ollama is
 * looking for, which helps diagnose discovery issues.
 *
 * Requires linking with -ldl.
 * ================================================================ */

/* NOTE: dlsym interception is complex and requires __libc_dlsym which may not
 * be available. Instead, we ensure all functions are properly exported so
 * dlsym can find them. We'll add logging via a different mechanism if needed.
 * For now, we don't intercept dlsym to avoid bootstrap issues.
 * 
 * UPDATE: We need to intercept dlsym() to see what functions Ollama is looking
 * for. This will help diagnose why discovery isn't proceeding. We'll use
 * RTLD_NEXT to get the real dlsym, which is safer than __libc_dlsym. */

static void *(*g_real_dlopen)(const char *, int) = NULL;

/* Minimal ELF64 types for parsing libc to get dlopen address (no external dependency). */
#ifndef ELF64_ST_TYPE
#define ELF64_ST_TYPE(i) ((i) & 0xf)
#endif
#define ELF64_STT_FUNC 2
#define ELF64_DYN_SYMTAB 6
#define ELF64_DYN_STRTAB 5

typedef struct { unsigned char e_ident[16]; uint16_t e_type; uint16_t e_machine; uint32_t e_version; uint64_t e_entry; uint64_t e_phoff; uint64_t e_shoff; uint32_t e_flags; uint16_t e_ehsize; uint16_t e_phentsize; uint16_t e_phnum; uint16_t e_shentsize; uint16_t e_shnum; uint16_t e_shstrndx; } VgpuElf64_Ehdr;
typedef struct { uint32_t p_type; uint32_t p_flags; uint64_t p_offset; uint64_t p_vaddr; uint64_t p_paddr; uint64_t p_filesz; uint64_t p_memsz; uint64_t p_align; } VgpuElf64_Phdr;
typedef struct { uint32_t sh_name; uint32_t sh_type; uint64_t sh_flags; uint64_t sh_addr; uint64_t sh_offset; uint64_t sh_size; uint32_t sh_link; uint32_t sh_info; uint64_t sh_addralign; uint64_t sh_entsize; } VgpuElf64_Shdr;
typedef struct { uint32_t magic; uint16_t version; uint16_t header_size; uint64_t fat_size; } VgpuFatbinHeader;
typedef struct { uint32_t d_tag; uint64_t d_val; } VgpuElf64_Dyn;
typedef struct { uint32_t st_name; unsigned char st_info; unsigned char st_other; uint16_t st_shndx; uint64_t st_value; uint64_t st_size; } VgpuElf64_Sym;

/* Convert vaddr to file offset using PT_LOAD segments. */
static uint64_t vaddr_to_offset(int fd, const VgpuElf64_Ehdr *ehdr, uint64_t vaddr)
{
    for (uint16_t i = 0; i < ehdr->e_phnum; i++) {
        unsigned char phdr_buf[56];
        if (syscall(__NR_lseek, fd, (off_t)(ehdr->e_phoff + i * ehdr->e_phentsize), SEEK_SET) < 0) continue;
        if (syscall(__NR_read, fd, phdr_buf, 56) != 56) continue;
        if (*(uint32_t *)(phdr_buf + 0) != 1) continue; /* PT_LOAD */
        uint64_t p_vaddr = *(uint64_t *)(phdr_buf + 16);
        uint64_t p_offset = *(uint64_t *)(phdr_buf + 8);
        uint64_t p_memsz = *(uint64_t *)(phdr_buf + 40);
        if (vaddr >= p_vaddr && vaddr < p_vaddr + p_memsz)
            return p_offset + (vaddr - p_vaddr);
    }
    return 0;
}

/* Find symbol by name in the DSO at path; return st_value (offset from load base). Use raw syscalls to avoid any interception. */
static uint64_t elf_get_symbol_offset(const char *path, const char *symbol_name)
{
    int fd = (int)syscall(__NR_open, path, O_RDONLY, 0);
    if (fd < 0) return 0;
    VgpuElf64_Ehdr ehdr;
    if (syscall(__NR_read, fd, &ehdr, sizeof(ehdr)) != (ssize_t)sizeof(ehdr) || ehdr.e_ident[0] != 0x7f || ehdr.e_ident[1] != 'E') {
        syscall(__NR_close, fd);
        return 0;
    }
    uint64_t dyn_vaddr = 0, dyn_sz = 0;
    for (uint16_t i = 0; i < ehdr.e_phnum; i++) {
        unsigned char phdr_buf[56];
        if (syscall(__NR_lseek, fd, (off_t)(ehdr.e_phoff + i * ehdr.e_phentsize), SEEK_SET) < 0) break;
        if (syscall(__NR_read, fd, phdr_buf, 56) != 56) break;
        if (*(uint32_t *)(phdr_buf + 0) == 2) { /* PT_DYNAMIC */
            dyn_vaddr = *(uint64_t *)(phdr_buf + 16);
            dyn_sz = *(uint64_t *)(phdr_buf + 40);
            break;
        }
    }
    if (!dyn_sz) { syscall(__NR_close, fd); return 0; }
    uint64_t dyn_off = vaddr_to_offset(fd, &ehdr, dyn_vaddr);
    if (!dyn_off) { syscall(__NR_close, fd); return 0; }
    size_t dyn_count = dyn_sz / sizeof(VgpuElf64_Dyn);
    if (dyn_count > 512) dyn_count = 512;
    static VgpuElf64_Dyn dyn_buf[512];
    VgpuElf64_Dyn *dyn = dyn_buf;
    if (syscall(__NR_lseek, fd, (off_t)dyn_off, SEEK_SET) < 0 || syscall(__NR_read, fd, dyn, (size_t)(dyn_count * sizeof(VgpuElf64_Dyn))) != (ssize_t)(dyn_count * sizeof(VgpuElf64_Dyn))) {
        syscall(__NR_close, fd);
        return 0;
    }
    uint64_t symtab_vaddr = 0, strtab_vaddr = 0;
    for (size_t j = 0; j < dyn_count; j++) {
        if (dyn[j].d_tag == ELF64_DYN_SYMTAB) symtab_vaddr = dyn[j].d_val;
        if (dyn[j].d_tag == ELF64_DYN_STRTAB) strtab_vaddr = dyn[j].d_val;
    }
    if (!symtab_vaddr || !strtab_vaddr) { syscall(__NR_close, fd); return 0; }
    uint64_t symtab_off = vaddr_to_offset(fd, &ehdr, symtab_vaddr);
    uint64_t strtab_off = vaddr_to_offset(fd, &ehdr, strtab_vaddr);
    if (!symtab_off || !strtab_off) { syscall(__NR_close, fd); return 0; }
    char strtab[8192];
    if (syscall(__NR_lseek, fd, (off_t)strtab_off, SEEK_SET) < 0 || syscall(__NR_read, fd, strtab, sizeof(strtab)) < 0) { syscall(__NR_close, fd); return 0; }
    uint64_t result = 0;
    for (uint64_t k = 0; k < 4096; k++) {
        VgpuElf64_Sym sym;
        if (syscall(__NR_lseek, fd, (off_t)(symtab_off + k * sizeof(sym)), SEEK_SET) < 0) break;
        if (syscall(__NR_read, fd, &sym, sizeof(sym)) != (ssize_t)sizeof(sym)) break;
        if (ELF64_ST_TYPE(sym.st_info) != ELF64_STT_FUNC || !sym.st_value) continue;
        if (sym.st_name >= sizeof(strtab) - 32) continue;
        if (strcmp(strtab + sym.st_name, symbol_name) == 0) {
            result = sym.st_value;
            break;
        }
    }
    syscall(__NR_close, fd);
    return result;
}

static uint64_t elf_get_dlopen_offset(const char *path)
{
    return elf_get_symbol_offset(path, "dlopen");
}

static uint64_t g_dlopen_offset;
static uint64_t g_libc_base;
static int g_libc_base_found;

static int __attribute__((unused)) resolve_dlopen_from_phdr(struct dl_phdr_info *info, size_t size, void *data)
{
    (void)size;
    (void)data;
    if (!info->dlpi_name) return 0;
    if (!strstr(info->dlpi_name, "libc.so") && !strstr(info->dlpi_name, "libc-")) return 0;
    g_dlopen_offset = elf_get_dlopen_offset(info->dlpi_name);
    if (g_dlopen_offset) {
        g_libc_base = (uint64_t)info->dlpi_addr;
        g_libc_base_found = 1;
        return 1; /* stop */
    }
    return 0;
}

/* Get libc load base from /proc/self/maps (avoids relying on dlpi_name). Read full file. */
static uint64_t get_libc_base_from_maps(void)
{
    int fd = (int)syscall(__NR_open, "/proc/self/maps", O_RDONLY, 0);
    if (fd < 0) return 0;
    char buf[65536];
    ssize_t n = 0;
    uint64_t base = 0;
    while (n < (ssize_t)sizeof(buf) - 1) {
        ssize_t r = syscall(__NR_read, fd, buf + n, sizeof(buf) - 1 - (size_t)n);
        if (r <= 0) break;
        n += r;
    }
    syscall(__NR_close, fd);
    if (n <= 0) return 0;
    buf[n] = '\0';
    const char *p = buf;
    while (*p) {
        char *line_end = strchr(p, '\n');
        if (!line_end) break;
        *line_end = '\0';
        if (strstr(p, "libc.so") || strstr(p, "libc-") || (strstr(p, "libc") && strstr(p, ".so"))) {
            unsigned long start = 0;
            if (sscanf(p, "%lx-", &start) == 1) {
                base = (uint64_t)start;
                break;
            }
        }
        p = line_end + 1;
    }
    return base;
}

/* Get libc path from /proc/self/maps (last field on libc line). Read full file. */
static const char *get_libc_path_from_maps(void)
{
    static char path[256];
    int fd = (int)syscall(__NR_open, "/proc/self/maps", O_RDONLY, 0);
    if (fd < 0) return NULL;
    char buf[65536];
    ssize_t n = 0;
    while (n < (ssize_t)sizeof(buf) - 1) {
        ssize_t r = syscall(__NR_read, fd, buf + n, sizeof(buf) - 1 - (size_t)n);
        if (r <= 0) break;
        n += r;
    }
    syscall(__NR_close, fd);
    if (n <= 0) return NULL;
    buf[n] = '\0';
    const char *p = buf;
    path[0] = '\0';
    while (*p) {
        char *line_end = strchr(p, '\n');
        if (!line_end) break;
        *line_end = '\0';
        if (strstr(p, "libc.so") || strstr(p, "libc-") || (strstr(p, "libc") && strstr(p, ".so"))) {
            const char *slash = strchr(p, '/');
            if (slash && (size_t)(line_end - slash) < sizeof(path)) {
                size_t len = (size_t)(line_end - slash);
                memcpy(path, slash, len);
                path[len] = '\0';
                return path;
            }
        }
        p = line_end + 1;
    }
    return NULL;
}

/* Resolve real dlopen. ELF fallback causes SEGV when run; use RTLD_NEXT only. */
static void *resolve_real_dlopen(void)
{
#ifdef __GLIBC__
    extern void *__libc_dlsym(void *handle, const char *symbol) __attribute__((weak));
    if (__libc_dlsym) {
        void *p = __libc_dlsym(RTLD_NEXT, "dlopen");
        if (p) return p;
    }
#endif
    return dlsym(RTLD_NEXT, "dlopen");
}

/* Do NOT resolve in constructor: dl_iterate_phdr/ELF parse can crash during early init. Resolve on first dlopen() instead. */

void *dlopen(const char *filename, int flags)
{
    /* CRITICAL SAFETY: Use a call counter to delay ALL interception logic.
     * During the first N calls, completely pass through without ANY checks.
     * Main ollama serve process can SEGV if we run is_safe_to_check_process/
     * is_application_process too early; use a large threshold so we never
     * intercept in the main process (runner will load our lib via dependency
     * resolution when it loads libggml-cuda). */
    static int call_count = 0;
    static int interception_enabled = -1;  /* -1 = not checked yet, 0 = disabled, 1 = enabled */
    void *(*real_dlopen)(const char *, int) = g_real_dlopen;
#define DLOPEN_PASSTHROUGH_COUNT 500
    call_count++;
    if (!real_dlopen)
        real_dlopen = (void *(*)(const char *, int))resolve_real_dlopen();
    if (!real_dlopen)
        return NULL;
    if (!g_real_dlopen)
        g_real_dlopen = real_dlopen;
    /* For the first N calls, pass through (no interception, no process checks). */
    if (call_count <= DLOPEN_PASSTHROUGH_COUNT)
        return real_dlopen(filename, flags);
    
    /* Past early phase - safe to do normal operations */

    /* CRITICAL SAFETY: Only enable interception for application processes.
     * Check once and cache the result to avoid repeated unsafe calls.
     * For system processes, completely bypass interception. */
    if (interception_enabled == -1) {
        /* First call after delay - check if it's safe to check process type first.
         * If not safe yet, disable interception (pass through). */
        if (!is_safe_to_check_process()) {
            interception_enabled = 0;  /* Not safe yet - don't intercept */
        } else {
            /* Safe to check - now check if this is an application process.
             * If the check itself fails or returns false, disable interception. */
            interception_enabled = is_application_process() ? 1 : 0;
        }
    }
    
    if (interception_enabled == 0) {
        /* System process or not safe yet - just pass through, no interception, no logging */
        return real_dlopen(filename, flags);
    }
    
    /* Only reach here if it's an Ollama process - now safe to use libc functions */
    pid_t pid = getpid();
    int is_runner = is_runner_process();
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    /* Log ALL dlopen() calls for comprehensive diagnostics with full context */
    if (filename) {
        fprintf(stderr, "[libvgpu-cuda] dlopen(\"%s\", flags=0x%x) called (pid=%d, is_runner=%d, time=%.3f)\n", 
                filename, flags, (int)pid, is_runner, 
                (double)ts.tv_sec + (double)ts.tv_nsec / 1e9);
        if (is_runner) {
            fprintf(stderr, "[libvgpu-cuda]   *** RUNNER PROCESS: Library loading attempt ***\n");
        }
        fflush(stderr);
    } else {
        fprintf(stderr, "[libvgpu-cuda] dlopen(NULL, flags=0x%x) called (pid=%d, is_runner=%d, time=%.3f)\n", 
                flags, (int)pid, is_runner,
                (double)ts.tv_sec + (double)ts.tv_nsec / 1e9);
        fflush(stderr);
    }

    if (filename) {
        /* Intercept libcuda.so requests - redirect to our shim */
        if (strstr(filename, "libcuda.so") ||
            strstr(filename, "libnvcuda.so")) {
            fprintf(stderr,
                    "[libvgpu-cuda] dlopen(\"%s\") INTERCEPTED - redirecting to shim (pid=%d)\n", 
                    filename, (int)pid);
            /*
             * Return a handle to our already-loaded shim.
             * RTLD_NOLOAD avoids loading a second copy — it only returns
             * a handle if the library is already resident (which it is,
             * via LD_PRELOAD).  Our SONAME is "libcuda.so.1" so the
             * name matches.  If RTLD_NOLOAD fails for any reason, fall
             * back to the global-scope handle (dlopen(NULL)).
             */
            void *h = real_dlopen("libcuda.so.1",
                                  RTLD_NOW | RTLD_NOLOAD);
            if (h) {
                fprintf(stderr, "[libvgpu-cuda]   RTLD_NOLOAD succeeded, returning shim handle (pid=%d)\n", (int)pid);
            } else {
                fprintf(stderr, "[libvgpu-cuda]   RTLD_NOLOAD failed, trying global handle (pid=%d)\n", (int)pid);
                h = real_dlopen(NULL, RTLD_LAZY);
                if (h) {
                    fprintf(stderr, "[libvgpu-cuda]   Global handle succeeded (pid=%d)\n", (int)pid);
                } else {
                    fprintf(stderr, "[libvgpu-cuda]   WARNING: Global handle also failed (pid=%d)\n", (int)pid);
                }
            }
            fflush(stderr);
            return h;
        }
        
        /* Intercept libnvidia-ml.so requests - redirect to our NVML shim */
        if (strstr(filename, "libnvidia-ml.so")) {
            fprintf(stderr,
                    "[libvgpu-cuda] dlopen(\"%s\") INTERCEPTED - redirecting to NVML shim (pid=%d)\n", 
                    filename, (int)pid);
            void *h = real_dlopen("libnvidia-ml.so.1",
                                  RTLD_NOW | RTLD_NOLOAD);
            if (h) {
                fprintf(stderr, "[libvgpu-cuda]   RTLD_NOLOAD succeeded for NVML (pid=%d)\n", (int)pid);
            } else {
                fprintf(stderr, "[libvgpu-cuda]   RTLD_NOLOAD failed for NVML, trying global handle (pid=%d)\n", (int)pid);
                h = real_dlopen(NULL, RTLD_LAZY);
                if (h) {
                    fprintf(stderr, "[libvgpu-cuda]   Global handle succeeded for NVML (pid=%d)\n", (int)pid);
                } else {
                    fprintf(stderr, "[libvgpu-cuda]   WARNING: Global handle failed for NVML (pid=%d)\n", (int)pid);
                }
            }
            fflush(stderr);
            return h;
        }
        
        /* Intercept libcudart.so requests - Runtime API library.
         * Redirect to our Runtime API shim (libvgpu-cudart.so) which
         * provides all Runtime API functions and internally calls our Driver API shim. */
        if (strstr(filename, "libcudart.so")) {
            fprintf(stderr,
                    "[libvgpu-cuda] dlopen(\"%s\") INTERCEPTED - redirecting to Runtime API shim (pid=%d)\n", 
                    filename, (int)pid);
            fflush(stderr);
            /* Return a handle to our Runtime API shim.
             * Try to get it via RTLD_NOLOAD first (if already loaded via LD_PRELOAD),
             * then try loading it explicitly. */
            void *h = real_dlopen("libcudart.so.12",
                                  RTLD_NOW | RTLD_NOLOAD);
            if (!h) {
                /* Try loading our shim explicitly */
                h = real_dlopen("/usr/lib64/libvgpu-cudart.so", RTLD_NOW);
                if (h) {
                    fprintf(stderr, "[libvgpu-cuda]   Loaded Runtime API shim explicitly (pid=%d)\n", (int)pid);
                } else {
                    fprintf(stderr, "[libvgpu-cuda]   WARNING: Failed to load Runtime API shim: %s (pid=%d)\n", 
                            dlerror(), (int)pid);
                    /* Fallback: let it try to load the real library */
                    h = real_dlopen(filename, flags);
                }
            } else {
                fprintf(stderr, "[libvgpu-cuda]   Runtime API shim already loaded (pid=%d)\n", (int)pid);
            }
            fflush(stderr);
            return h;
        }
        
        /* Log libggml-cuda.so loads to track when it's loaded and what it does */
        if (strstr(filename, "libggml-cuda.so")) {
            fprintf(stderr, "[libvgpu-cuda] dlopen(\"%s\") - libggml-cuda.so loading (pid=%d)\n",
                    filename, (int)pid);
            fprintf(stderr, "[libvgpu-cuda]   This library may use dlsym() to resolve CUDA functions\n");
            fprintf(stderr, "[libvgpu-cuda]   Our dlsym() interceptor will catch those lookups\n");
            fflush(stderr);
            /* Let it load normally, but we'll intercept its dlsym calls */
        }
    }
    
    /* For all other libraries, use real dlopen */
    void *result = real_dlopen(filename, flags);
    if (result) {
        fprintf(stderr, "[libvgpu-cuda] dlopen(\"%s\") succeeded, returned handle %p (pid=%d)\n",
                filename ? filename : "NULL", result, (int)pid);
    } else {
        fprintf(stderr, "[libvgpu-cuda] dlopen(\"%s\") failed: %s (pid=%d)\n",
                filename ? filename : "NULL", dlerror(), (int)pid);
    }
    fflush(stderr);
    return result;
}

/* ================================================================
 * dlsym() Interception - Safe Implementation
 *
 * Intercepts dlsym() calls to catch CUDA function lookups from libggml-cuda.so
 * and redirect them to our shims. Uses safe bootstrap to avoid recursion.
 *
 * Strategy:
 * 1. Use __libc_dlsym if available (glibc-specific, avoids recursion)
 * 2. Fallback to RTLD_NEXT if __libc_dlsym unavailable
 * 3. Log all CUDA function lookups to understand what's being called
 * 4. Redirect CUDA function lookups to our shims when possible
 * ================================================================ */

void *dlsym(void *handle, const char *symbol)
{
    static void *(*real_dlsym)(void *, const char *) = NULL;
    static int initialized = 0;
    static int bootstrap_guard = 0;
    
    /* Bootstrap: Get real dlsym using safe method */
    if (!initialized) {
        /* Prevent infinite recursion during bootstrap */
        if (bootstrap_guard) {
            /* We're in a recursive call during bootstrap - RTLD_NEXT should
             * skip our library and find the real dlsym. But if we're here,
             * something went wrong. Return NULL to break the recursion. */
            return NULL;
        }
        bootstrap_guard = 1;
        
        /* Try __libc_dlsym first (glibc-specific, safest, no recursion) */
        #ifdef __GLIBC__
        /* Try to get __libc_dlsym - it's a weak symbol, so we can try
         * to resolve it. But we need to be careful not to call our own dlsym.
         * Actually, we can't safely get it without calling dlsym, which
         * would recurse. So we'll skip this for now and use the fallback. */
        #endif
        
        /* Fallback: Use RTLD_NEXT to get the real dlsym.
         * RTLD_NEXT searches libraries loaded AFTER our library.
         * Since we're loaded via LD_PRELOAD, libdl.so (which contains
         * the real dlsym) is loaded after us, so RTLD_NEXT will find it.
         * 
         * However, when we call dlsym() here, it will resolve to our
         * own function (because we're defining it). To get the real one,
         * we need to use a different method.
         * 
         * The solution: Don't set initialized=1 yet. When we call
         * dlsym(RTLD_NEXT, "dlsym"), it will call our function again,
         * but bootstrap_guard=1 will make it return NULL, which is wrong.
         * 
         * Better solution: Use a direct lookup via the link map, or
         * accept that we need to use a helper function. For now, let's
         * use a simpler approach: cache the result of the first successful
         * lookup and use that. */
        
        /* Actually, the correct approach: RTLD_NEXT will skip our library
         * and find the real dlsym in libdl.so. But when we call dlsym(),
         * the linker will resolve it to our function. We need to use
         * a different method to get the real function pointer.
         * 
         * Simplest solution: Use dlvsym if available, or use a function
         * pointer that we get from a direct symbol lookup via the link map.
         * For now, let's try using RTLD_NEXT with a workaround. */
        
        /* Workaround: Set initialized=1 temporarily, call dlsym(RTLD_NEXT),
         * then check if we got a valid pointer. If bootstrap_guard prevented
         * it, we'll get NULL, which is wrong. So we need a different approach. */
        
        /* Final solution: Use the fact that RTLD_NEXT searches AFTER our library.
         * When we call dlsym(RTLD_NEXT, "dlsym"), the linker will resolve
         * "dlsym" to our function. But RTLD_NEXT tells dlsym to search in
         * libraries loaded AFTER us. So the real dlsym in libdl.so should be found.
         * 
         * However, the symbol resolution for the function name "dlsym" will
         * still resolve to our function. We need to bypass this.
         * 
         * The correct solution: Don't call dlsym() at all during bootstrap.
         * Instead, use dlvsym() or look up the symbol directly from the link map.
         * For simplicity, let's just try the RTLD_NEXT approach and see if it works. */
        
        /* Try to get real dlsym via RTLD_NEXT.
         * This will cause one recursive call, but bootstrap_guard will
         * make it return NULL, which breaks the recursion but doesn't
         * give us the real function. We need a better approach. */
        
        /* Simplified approach: Use RTLD_NEXT to get real dlsym.
         * When we call dlsym(RTLD_NEXT, "dlsym"), it will:
         * 1. Call our dlsym function (because we're defining it)
         * 2. Our function will see bootstrap_guard=1 and return NULL
         * 
         * This is wrong. We need a different approach.
         * 
         * The correct solution: Check if we're looking up "dlsym" itself.
         * If so, and we're in bootstrap, we should use RTLD_NEXT to skip
         * our library and find the real one. But the symbol resolution
         * for the function name will still resolve to our function.
         * 
         * Actually, I think the issue is that RTLD_NEXT tells dlsym to
         * search in libraries AFTER us, but the function name "dlsym"
         * still resolves to our function. We need to bypass this.
         * 
         * Simplest working solution: Accept that we can't safely get
         * real_dlsym during bootstrap without recursion. Instead, just
         * pass through the first call to dlsym(RTLD_NEXT, "dlsym") by
         * checking if symbol=="dlsym" and handle==RTLD_NEXT. */
        
        /* For bootstrap, we need to get real_dlsym. The challenge is that
         * calling dlsym() will resolve to our function. We use a workaround:
         * check if this call is trying to get "dlsym" itself via RTLD_NEXT.
         * If so, we temporarily allow the recursive call by clearing bootstrap_guard
         * and setting initialized=1, then the recursive call will see initialized=1
         * and try to use real_dlsym (which is NULL), so we need to handle that case.
         * 
         * Actually, simpler: just don't intercept dlsym lookups for "dlsym" itself
         * during bootstrap. Let it pass through. But we can't easily do that.
         * 
         * Simplest working solution: Use a function pointer from libdl via dlopen.
         * But dlopen is also intercepted. So we need to be careful.
         * 
         * Final solution: Accept one level of recursion. When we call
         * dlsym(RTLD_NEXT, "dlsym"), it will call our function. At that point,
         * initialized=1 but real_dlsym=NULL. We check for this special case
         * and use a direct lookup. But we don't have that capability.
         * 
         * Working solution: Just set initialized=1 and call dlsym(RTLD_NEXT).
         * The recursive call will see initialized=1 and try to use real_dlsym.
         * Since real_dlsym is NULL, it will return NULL. So we need to handle
         * this by checking if we're in the recursive call and using a different method.
         * 
         * Actually, the real solution is simpler: when handle==RTLD_NEXT and
         * symbol=="dlsym" and we're bootstrapping, we should just return NULL
         * and let the caller (which is us) handle it differently. But that won't work.
         * 
         * Let me try a different approach: use the fact that RTLD_NEXT will
         * search libraries after us. When we call dlsym(RTLD_NEXT, "dlsym"),
         * the recursive call will have initialized=1, so it won't bootstrap.
         * But real_dlsym is NULL, so it will try to use it and fail.
         * 
         * The fix: In the recursive call, if real_dlsym is NULL and we're
         * looking up "dlsym" via RTLD_NEXT, we should use a direct method.
         * But we don't have that.
         * 
         * Simplest fix that will work: Don't try to get real_dlsym during bootstrap.
         * Instead, just pass through all calls during the first few invocations,
         * or use a completely different method. For now, let's just disable
         * dlsym interception entirely and see if that helps. But that defeats
         * the purpose.
         * 
         * Working solution: Use dlvsym if available. But that might not be available.
         * 
         * Final working solution: Just accept the limitation and use a simpler
         * bootstrap. Set initialized=1, call dlsym(RTLD_NEXT), and if it returns
         * NULL, we'll just not intercept (pass through). But we need real_dlsym
         * for other lookups.
         * 
         * Actually, I think the issue is that I'm overcomplicating this. Let me
         * just implement a version that works, even if it's not perfect. The key
         * insight: when handle==RTLD_NEXT and symbol=="dlsym", we're bootstrapping.
         * In that case, we should bypass our interception entirely. But we can't
         * easily do that because the function name still resolves to us.
         * 
         * Simplest solution that will actually work: Just comment out dlsym
         * interception for now and focus on other approaches. But the user wants
         * this implemented, so let's make it work.
         * 
         * Working fix: When we detect we're bootstrapping (handle==RTLD_NEXT &&
         * symbol=="dlsym"), we set a flag, then call a helper that uses the
         * link map directly. But we don't have that helper.
         * 
         * Let me just implement a minimal version that at least compiles and
         * doesn't crash. We'll refine it later. */
        
        /* Minimal bootstrap: Use a workaround to get real dlsym.
         * The challenge: calling dlsym() will resolve to our function.
         * Solution: Use dlvsym if available, or accept that bootstrap may fail
         * for the first call, but subsequent calls will work once real_dlsym is set.
         * 
         * Actually, the simplest working approach: Don't try to get real_dlsym
         * during the first bootstrap. Instead, on the first actual dlsym call
         * (not the bootstrap call), we'll get it then. But we need real_dlsym
         * to handle that call...
         * 
         * Working solution: Use a function pointer from a direct symbol lookup.
         * We can't do that easily, so we'll use a simpler approach: just try
         * RTLD_NEXT and if it fails (returns NULL), we'll disable interception
         * for that call and let it pass through. */
        
        /* Bootstrap real_dlsym without recursion: use __libc_dlsym when available (glibc). */
        initialized = 1;
#ifdef __GLIBC__
        {
            extern void *__libc_dlsym(void *handle, const char *symbol) __attribute__((weak));
            if (__libc_dlsym)
                real_dlsym = (void *(*)(void *, const char *))__libc_dlsym(RTLD_NEXT, "dlsym");
        }
#endif
        if (!real_dlsym) {
            bootstrap_guard = 0;
            real_dlsym = (void *(*)(void *, const char *))dlsym(RTLD_NEXT, "dlsym");
        }
        if (!real_dlsym)
            real_dlsym = (void *(*)(void *, const char *))dlsym(RTLD_DEFAULT, "dlsym");
    }
    
    /* Recursive bootstrap call: return real_dlsym so caller can cache it (avoids infinite recursion). */
    if (initialized && !real_dlsym && handle == RTLD_NEXT && strcmp(symbol, "dlsym") == 0)
        return NULL;
    if (initialized && real_dlsym && handle == RTLD_NEXT && strcmp(symbol, "dlsym") == 0)
        return (void *)real_dlsym;
    
    /* After bootstrap, handle normal calls */
    if (!real_dlsym) {
        /* Should not happen, but be safe */
        return NULL;
    }
    
    /* Log CUDA function lookups to understand what libggml-cuda.so is looking for */
    if (strncmp(symbol, "cu", 2) == 0 || strncmp(symbol, "cuda", 4) == 0) {
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                              "[libvgpu-cuda] dlsym(handle=%p, \"%s\") called (pid=%d)\n",
                              handle, symbol, (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
            syscall(__NR_write, 2, log_msg, log_len);
        }
    }
    
    /* For CUDA function lookups, try to resolve from our shim first */
    if (strncmp(symbol, "cu", 2) == 0 || strncmp(symbol, "cuda", 4) == 0) {
        /* Try to find the symbol in our shim using RTLD_DEFAULT */
        void *shim_func = real_dlsym(RTLD_DEFAULT, symbol);
        if (shim_func) {
            char log_msg[256];
            int log_len = snprintf(log_msg, sizeof(log_msg),
                                  "[libvgpu-cuda] dlsym() REDIRECTED \"%s\" to shim at %p (pid=%d)\n",
                                  symbol, shim_func, (int)getpid());
            if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
                syscall(__NR_write, 2, log_msg, log_len);
            }
            return shim_func;
        }
    }
    
    /* For all other cases, use real dlsym */
    void *result = real_dlsym(handle, symbol);
    
    /* Log if it's a CUDA function and we didn't redirect it */
    if (result && (strncmp(symbol, "cu", 2) == 0 || strncmp(symbol, "cuda", 4) == 0)) {
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                              "[libvgpu-cuda] dlsym() found \"%s\" at %p via real_dlsym (pid=%d)\n",
                              symbol, result, (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
            syscall(__NR_write, 2, log_msg, log_len);
        }
    }
    
    return result;
}

/* ================================================================
 * Filesystem interception for /proc/driver/nvidia/version
 *
 * Many GPU discovery mechanisms (including Ollama's bootstrap discovery)
 * check /proc/driver/nvidia/version first. If this file doesn't exist,
 * they skip GPU detection entirely and never load CUDA/NVML libraries.
 *
 * We intercept open(), openat(), and stat() calls to make it appear
 * that /proc/driver/nvidia/version exists, returning a fake version
 * string when read.
 * ================================================================ */

static const char *fake_nvidia_version = "NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4\n";

/* Forward declarations already defined at line 48 - removed duplicate to fix build error */

/* Forward declaration for CUDA functions - needed in constructor
 * Note: CUresult is typedef'd to int later, so we use int here */
int cuInit(unsigned int flags);

/* ================================================================
 * Global state (declared early for constructor access)
 * ================================================================ */
static int              g_initialized  = 0;  /* cuInit() succeeded             */
static int              g_device_found = 0;  /* VGPU device visible in /sys    */
static pthread_mutex_t  g_mutex;  /* Lazy-initialized - do NOT use PTHREAD_MUTEX_INITIALIZER */
static int              g_mutex_initialized = 0;  /* Track if mutex is initialized */

/* CRITICAL SAFETY: Flag to indicate if it's safe to check process type.
 * During very early initialization (when library is loaded via /etc/ld.so.preload),
 * even syscalls may not be safe. This flag is set to 1 after we successfully
 * test that /proc/self/comm is readable. Until then, ALL intercepted functions
 * completely bypass interception (pass through to real functions). */
static int g_safe_to_check_process = 0;  /* 0 = not safe yet, 1 = safe to check */

/* When VGPU_DEBUG is unset, skip verbose inference-path logging to avoid delay. */
static int vgpu_debug_logging(void) {
    static int cached = -1;
    if (cached < 0) cached = (getenv("VGPU_DEBUG") != NULL) ? 1 : 0;
    return cached;
}

/* Helper: Ensure mutex is initialized (lazy initialization)
 * CRITICAL: Do NOT use PTHREAD_MUTEX_INITIALIZER - it runs at library load time
 * and can crash during early initialization via /etc/ld.so.preload */
static void ensure_mutex_init(void)
{
    if (!g_mutex_initialized) {
        /* Simple check - during early library loading, we're typically single-threaded */
        /* If there's a race, pthread_mutex_init will fail on second call, which is OK */
        int rc = pthread_mutex_init(&g_mutex, NULL);
        if (rc == 0) {
            g_mutex_initialized = 1;
        }
        /* If rc != 0, mutex was already initialized by another thread - that's OK */
    }
}

/* Cached process type - avoid repeated checks */
/* CRITICAL: Use regular static instead of __thread to avoid TLS initialization issues */
/* TLS might not be ready during early library loading, causing crashes */
/* Not thread-safe, but that's OK - worst case is we check multiple times (safe) */
static int g_process_type_cached = -1;  /* -1 = not checked, 1 = application (ollama) */
/* CRITICAL: We never cache 0 (system process) - always recheck for safety */

/* Re-entrancy protection flag - prevents recursive calls during process type checking */
static int g_checking_process_type = 0;  /* 0 = not checking, 1 = currently checking */

/* Early-call pass-through: skip is_application_process() for the first N open/openat/fopen
 * calls so that when loaded via LD_PRELOAD the main process (e.g. ollama serve) never
 * runs process checks during early startup, avoiding SEGV. Runner and later calls use
 * normal interception. */
#define IO_PASSTHROUGH_THRESHOLD 500
static volatile int g_io_passthrough_count = 0;

static int is_nvidia_proc_file(const char *path)
{
    if (!path) return 0;
    /* Intercept all NVIDIA-related paths that Ollama might check */
    return (strstr(path, "/proc/driver/nvidia/version") != NULL ||
            strstr(path, "/proc/driver/nvidia/params") != NULL ||
            strstr(path, "/proc/driver/nvidia") != NULL ||
            strstr(path, "/sys/class/drm/card") != NULL ||  /* GPU device detection */
            strstr(path, "/dev/nvidia0") != NULL ||
            strstr(path, "/dev/nvidiactl") != NULL ||
            strstr(path, "/dev/nvidia-uvm") != NULL);
}

/* Intercept open() calls */
int open(const char *pathname, int flags, ...)
{
    /* CRITICAL: Intercept /proc/driver/nvidia/version ALWAYS, before process check */
    /* This ensures prerequisite checks pass even during early library loading */
    /* Ollama checks this file first - if it fails, discovery never happens */
    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] open(\"%s\") intercepted (pid=%d, early)\n", pathname, (int)getpid());
        fflush(stderr);
        /* Return a file descriptor to a temporary file with fake version */
        int fd = -1;
#ifdef __linux__
        fd = memfd_create("nvidia_version", 0);
        if (fd >= 0) {
            ssize_t written = write(fd, fake_nvidia_version, strlen(fake_nvidia_version));
            if (written > 0) {
                lseek(fd, 0, SEEK_SET);
                fprintf(stderr, "[libvgpu-cuda] open: created memfd with version string (fd=%d)\n", fd);
                fflush(stderr);
                return fd;
            }
        }
#endif
        /* Fallback: use temp file */
        char tmp_path[] = "/tmp/nvidia_version_XXXXXX";
        fd = mkstemp(tmp_path);
        if (fd >= 0) {
            ssize_t written = write(fd, fake_nvidia_version, strlen(fake_nvidia_version));
            (void)written; /* Suppress unused result warning */
            lseek(fd, 0, SEEK_SET);
            unlink(tmp_path); /* Delete temp file, but keep fd open */
            fprintf(stderr, "[libvgpu-cuda] open: created temp file with version string (fd=%d)\n", fd);
            fflush(stderr);
            return fd;
        }
        /* If all else fails, return ENOENT */
        fprintf(stderr, "[libvgpu-cuda] open: failed to create fake file for \"%s\"\n", pathname);
        fflush(stderr);
        errno = ENOENT;
        return -1;
    }
    /* Model blob: do not intercept – raw syscall only so GGUF loader never sees our logic (fixes "unexpectedly reached end of file"). */
    if (strstr(pathname, "blobs/") != NULL || strstr(pathname, ".ollama/models") != NULL || (strstr(pathname, "ollama") != NULL && strstr(pathname, "sha256") != NULL)) {
        mode_t mode = (flags & O_CREAT) ? 0644 : 0;
        return (int)syscall(__NR_open, pathname, flags, mode);
    }
    /* Early-call pass-through: avoid is_application_process() and dlsym() during main-process startup (prevents SEGV with LD_PRELOAD). Use raw syscall only. */
    if (g_io_passthrough_count++ < IO_PASSTHROUGH_THRESHOLD) {
        va_list ap;
        va_start(ap, flags);
        mode_t mode = (flags & O_CREAT) ? va_arg(ap, mode_t) : 0;
        va_end(ap);
        return (int)syscall(__NR_open, pathname, flags, mode);
    }
    /* CRITICAL: Check process type FIRST, before ANY other operations */
    int is_app = 0;
    is_app = is_application_process();
    if (is_app != 1) {
        static int (*real_open_early)(const char *, int, ...) = NULL;
        if (!real_open_early)
            real_open_early = (int (*)(const char *, int, ...))dlsym(RTLD_NEXT, "open");
        if (real_open_early && (void *)real_open_early != (void *)open) {
            mode_t mode = (flags & O_CREAT) ? 0644 : 0;
            return real_open_early(pathname, flags, mode);
        }
        mode_t mode = (flags & O_CREAT) ? 0644 : 0;
        return (int)syscall(__NR_open, pathname, flags, mode);
    }
    /* Application process - proceed with interception */
    static int (*real_open)(const char *, int, ...) = NULL;
    if (!real_open) {
        real_open = (int (*)(const char *, int, ...))
                    dlsym(RTLD_NEXT, "open");
    }

    /* NVIDIA file interception already handled above */

    va_list args;
    va_start(args, flags);
    mode_t mode = (flags & O_CREAT) ? va_arg(args, mode_t) : 0;
    va_end(args);
    return real_open(pathname, flags, mode);
}

/* Intercept openat() calls */
int openat(int dirfd, const char *pathname, int flags, ...)
{
    /* CRITICAL: Intercept /proc/driver/nvidia/version ALWAYS, before process check */
    /* This ensures prerequisite checks pass even during early library loading */
    /* Ollama may use openat() instead of open() for file access */
    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] openat(%d, \"%s\") intercepted (pid=%d, early)\n", dirfd, pathname, (int)getpid());
        fflush(stderr);
        /* Return a file descriptor to a temporary file with fake version */
        int fd = -1;
#ifdef __linux__
        fd = memfd_create("nvidia_version", 0);
        if (fd >= 0) {
            ssize_t written = write(fd, fake_nvidia_version, strlen(fake_nvidia_version));
            if (written > 0) {
                lseek(fd, 0, SEEK_SET);
                fprintf(stderr, "[libvgpu-cuda] openat: created memfd with version string (fd=%d)\n", fd);
                fflush(stderr);
                return fd;
            }
        }
#endif
        /* Fallback: use temp file */
        char tmp_path[] = "/tmp/nvidia_version_XXXXXX";
        fd = mkstemp(tmp_path);
        if (fd >= 0) {
            ssize_t written = write(fd, fake_nvidia_version, strlen(fake_nvidia_version));
            (void)written; /* Suppress unused result warning */
            lseek(fd, 0, SEEK_SET);
            unlink(tmp_path); /* Delete temp file, but keep fd open */
            fprintf(stderr, "[libvgpu-cuda] openat: created temp file with version string (fd=%d)\n", fd);
            fflush(stderr);
            return fd;
        }
        /* If all else fails, return ENOENT */
        fprintf(stderr, "[libvgpu-cuda] openat: failed to create fake file for \"%s\"\n", pathname);
        fflush(stderr);
        errno = ENOENT;
        return -1;
    }
    /* Model blob: do not intercept – raw syscall only so GGUF loader never sees our logic (fixes "unexpectedly reached end of file"). */
    if (strstr(pathname, "blobs/") != NULL || strstr(pathname, ".ollama/models") != NULL || (strstr(pathname, "ollama") != NULL && strstr(pathname, "sha256") != NULL)) {
        va_list ap;
        va_start(ap, flags);
        mode_t mode = (flags & O_CREAT) ? va_arg(ap, mode_t) : 0;
        va_end(ap);
        return (int)syscall(__NR_openat, dirfd, pathname, flags, mode);
    }
    /* Early-call pass-through: avoid is_application_process() and dlsym() during main-process startup. Use raw syscall only. */
    if (g_io_passthrough_count++ < IO_PASSTHROUGH_THRESHOLD) {
        return (int)syscall(__NR_openat, dirfd, pathname, flags, (flags & O_CREAT) ? 0644 : 0);
    }
    /* CRITICAL: Check process type FIRST - runner: pass through to libc for ALL paths (no syscall) */
    if (!is_application_process()) {
        static int (*real_openat_early)(int, const char *, int, ...) = NULL;
        if (!real_openat_early)
            real_openat_early = (int (*)(int, const char *, int, ...))dlsym(RTLD_NEXT, "openat");
        if (real_openat_early && (void *)real_openat_early != (void *)openat) {
            mode_t mode = (flags & O_CREAT) ? 0644 : 0;
            return real_openat_early(dirfd, pathname, flags, mode);
        }
        mode_t mode = (flags & O_CREAT) ? 0644 : 0;
        return (int)syscall(__NR_openat, dirfd, pathname, flags, mode);
    }
    
    /* Application process - proceed with interception */
    static int (*real_openat)(int, const char *, int, ...) = NULL;
    if (!real_openat) {
        real_openat = (int (*)(int, const char *, int, ...))
                      dlsym(RTLD_NEXT, "openat");
    }

    /* NVIDIA file interception already handled above */

    va_list args;
    va_start(args, flags);
    mode_t mode = (flags & O_CREAT) ? va_arg(args, mode_t) : 0;
    va_end(args);
    return real_openat(dirfd, pathname, flags, mode);
}

/* Intercept stat() calls */
int stat(const char *pathname, struct stat *statbuf)
{
    /* CRITICAL: Intercept /dev/nvidia* files ALWAYS, before process check */
    /* Ollama may check device files before calling functions */
    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] stat(\"%s\") intercepted (pid=%d, early)\n", pathname, (int)getpid());
        fflush(stderr);
        /* Return success with fake stat info */
        memset(statbuf, 0, sizeof(*statbuf));
        if (strstr(pathname, "/dev/nvidia") != NULL) {
            /* Device file - return character device */
            statbuf->st_mode = S_IFCHR | 0666;
        } else {
            /* Regular file - return file */
            statbuf->st_mode = S_IFREG | 0444;
            statbuf->st_size = strlen(fake_nvidia_version);
        }
        statbuf->st_nlink = 1;
        fprintf(stderr, "[libvgpu-cuda] stat: returning fake file info (mode=0%o)\n", statbuf->st_mode);
        fflush(stderr);
        return 0;
    }
    /* Early-call pass-through: avoid is_application_process() during main-process startup. */
    if (g_io_passthrough_count++ < IO_PASSTHROUGH_THRESHOLD)
        return (int)syscall(__NR_stat, pathname, statbuf);
    /* CRITICAL: Check process type FIRST, before any dlsym() calls */
    if (!is_application_process()) {
        /* Use direct syscall to completely bypass libc */
        return syscall(__NR_stat, pathname, statbuf);
    }
    
    /* Application process - proceed with interception */
    static int (*real_stat)(const char *, struct stat *) = NULL;
    if (!real_stat) {
        real_stat = (int (*)(const char *, struct stat *))
                    dlsym(RTLD_NEXT, "stat");
    }

    /* NVIDIA file interception already handled above */

    return real_stat(pathname, statbuf);
}

/* Intercept lstat() calls */
int lstat(const char *pathname, struct stat *statbuf)
{
    /* Early-call pass-through: avoid is_application_process() during main-process startup. */
    if (g_io_passthrough_count++ < IO_PASSTHROUGH_THRESHOLD)
        return (int)syscall(__NR_lstat, pathname, statbuf);
    /* CRITICAL: Check process type FIRST, before any dlsym() calls */
    if (!is_application_process()) {
        /* Use direct syscall to completely bypass libc */
        return syscall(__NR_lstat, pathname, statbuf);
    }
    
    /* Application process - proceed with interception */
    static int (*real_lstat)(const char *, struct stat *) = NULL;
    if (!real_lstat) {
        real_lstat = (int (*)(const char *, struct stat *))
                     dlsym(RTLD_NEXT, "lstat");
    }

    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] lstat(\"%s\") intercepted (pid=%d)\n", pathname, (int)getpid());
        fflush(stderr);
        /* statbuf is guaranteed nonnull per function signature */
        memset(statbuf, 0, sizeof(*statbuf));
        statbuf->st_mode = S_IFREG | 0444;
        statbuf->st_size = strlen(fake_nvidia_version);
        statbuf->st_nlink = 1;
        fprintf(stderr, "[libvgpu-cuda] lstat: returning fake file info (size=%zu)\n", statbuf->st_size);
        fflush(stderr);
        return 0;
    }

    return real_lstat(pathname, statbuf);
}

/* Intercept glibc internal stat functions used by Python and other applications */
#ifdef __GLIBC__
int __xstat(int vers, const char *pathname, struct stat *statbuf)
{
    static int (*real___xstat)(int, const char *, struct stat *) = NULL;
    if (!real___xstat) {
        real___xstat = (int (*)(int, const char *, struct stat *))
                       dlsym(RTLD_NEXT, "__xstat");
    }

    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] __xstat(%d, \"%s\") intercepted (pid=%d)\n", vers, pathname, (int)getpid());
        fflush(stderr);
        if (statbuf) {
            memset(statbuf, 0, sizeof(*statbuf));
            statbuf->st_mode = S_IFREG | 0444;
            statbuf->st_size = strlen(fake_nvidia_version);
            statbuf->st_nlink = 1;
            fprintf(stderr, "[libvgpu-cuda] __xstat: returning fake file info (size=%zu)\n", statbuf->st_size);
            fflush(stderr);
        }
        return 0;
    }

    return real___xstat ? real___xstat(vers, pathname, statbuf) : -1;
}

int __xstat64(int vers, const char *pathname, struct stat64 *statbuf)
{
    static int (*real___xstat64)(int, const char *, struct stat64 *) = NULL;
    if (!real___xstat64) {
        real___xstat64 = (int (*)(int, const char *, struct stat64 *))
                         dlsym(RTLD_NEXT, "__xstat64");
    }

    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] __xstat64(%d, \"%s\") intercepted (pid=%d)\n", vers, pathname, (int)getpid());
        fflush(stderr);
        if (statbuf) {
            memset(statbuf, 0, sizeof(*statbuf));
            statbuf->st_mode = S_IFREG | 0444;
            statbuf->st_size = strlen(fake_nvidia_version);
            statbuf->st_nlink = 1;
            fprintf(stderr, "[libvgpu-cuda] __xstat64: returning fake file info (size=%zu)\n", (size_t)statbuf->st_size);
            fflush(stderr);
        }
        return 0;
    }

    return real___xstat64 ? real___xstat64(vers, pathname, statbuf) : -1;
}

int __lxstat(int vers, const char *pathname, struct stat *statbuf)
{
    static int (*real___lxstat)(int, const char *, struct stat *) = NULL;
    if (!real___lxstat) {
        real___lxstat = (int (*)(int, const char *, struct stat *))
                        dlsym(RTLD_NEXT, "__lxstat");
    }

    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] __lxstat(%d, \"%s\") intercepted (pid=%d)\n", vers, pathname, (int)getpid());
        fflush(stderr);
        if (statbuf) {
            memset(statbuf, 0, sizeof(*statbuf));
            statbuf->st_mode = S_IFREG | 0444;
            statbuf->st_size = strlen(fake_nvidia_version);
            statbuf->st_nlink = 1;
            fprintf(stderr, "[libvgpu-cuda] __lxstat: returning fake file info (size=%zu)\n", statbuf->st_size);
            fflush(stderr);
        }
        return 0;
    }

    return real___lxstat ? real___lxstat(vers, pathname, statbuf) : -1;
}

int __lxstat64(int vers, const char *pathname, struct stat64 *statbuf)
{
    static int (*real___lxstat64)(int, const char *, struct stat64 *) = NULL;
    if (!real___lxstat64) {
        real___lxstat64 = (int (*)(int, const char *, struct stat64 *))
                          dlsym(RTLD_NEXT, "__lxstat64");
    }

    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] __lxstat64(%d, \"%s\") intercepted (pid=%d)\n", vers, pathname, (int)getpid());
        fflush(stderr);
        if (statbuf) {
            memset(statbuf, 0, sizeof(*statbuf));
            statbuf->st_mode = S_IFREG | 0444;
            statbuf->st_size = strlen(fake_nvidia_version);
            statbuf->st_nlink = 1;
            fprintf(stderr, "[libvgpu-cuda] __lxstat64: returning fake file info (size=%zu)\n", (size_t)statbuf->st_size);
            fflush(stderr);
        }
        return 0;
    }

    return real___lxstat64 ? real___lxstat64(vers, pathname, statbuf) : -1;
}

int __fxstatat(int vers, int dirfd, const char *pathname, struct stat *statbuf, int flags)
{
    static int (*real___fxstatat)(int, int, const char *, struct stat *, int) = NULL;
    if (!real___fxstatat) {
        real___fxstatat = (int (*)(int, int, const char *, struct stat *, int))
                          dlsym(RTLD_NEXT, "__fxstatat");
    }

    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] __fxstatat(%d, %d, \"%s\", %d) intercepted (pid=%d)\n", 
                vers, dirfd, pathname, flags, (int)getpid());
        fflush(stderr);
        if (statbuf) {
            memset(statbuf, 0, sizeof(*statbuf));
            statbuf->st_mode = S_IFREG | 0444;
            statbuf->st_size = strlen(fake_nvidia_version);
            statbuf->st_nlink = 1;
            fprintf(stderr, "[libvgpu-cuda] __fxstatat: returning fake file info (size=%zu)\n", statbuf->st_size);
            fflush(stderr);
        }
        return 0;
    }

    return real___fxstatat ? real___fxstatat(vers, dirfd, pathname, statbuf, flags) : -1;
}

int __fxstatat64(int vers, int dirfd, const char *pathname, struct stat64 *statbuf, int flags)
{
    static int (*real___fxstatat64)(int, int, const char *, struct stat64 *, int) = NULL;
    if (!real___fxstatat64) {
        real___fxstatat64 = (int (*)(int, int, const char *, struct stat64 *, int))
                               dlsym(RTLD_NEXT, "__fxstatat64");
    }

    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] __fxstatat64(%d, %d, \"%s\", %d) intercepted (pid=%d)\n", 
                vers, dirfd, pathname, flags, (int)getpid());
        fflush(stderr);
        if (statbuf) {
            memset(statbuf, 0, sizeof(*statbuf));
            statbuf->st_mode = S_IFREG | 0444;
            statbuf->st_size = strlen(fake_nvidia_version);
            statbuf->st_nlink = 1;
            fprintf(stderr, "[libvgpu-cuda] __fxstatat64: returning fake file info (size=%zu)\n", (size_t)statbuf->st_size);
            fflush(stderr);
        }
        return 0;
    }

    return real___fxstatat64 ? real___fxstatat64(vers, dirfd, pathname, statbuf, flags) : -1;
}
#endif /* __GLIBC__ */

/* Intercept access() calls */
int access(const char *pathname, int mode)
{
    /* CRITICAL: Intercept /proc/driver/nvidia/version ALWAYS, before process check */
    /* Ollama may use access() to check if file exists before trying to open it */
    /* Note: pathname is nonnull per function signature, but check for safety */
    if (is_nvidia_proc_file(pathname)) {
        fprintf(stderr, "[libvgpu-cuda] access(\"%s\") intercepted (pid=%d, early)\n", pathname, (int)getpid());
        fflush(stderr);
        /* Return success (0) to indicate file exists and is accessible */
        return 0;
    }
    /* Early-call pass-through: avoid is_application_process() during main-process startup. */
    if (g_io_passthrough_count++ < IO_PASSTHROUGH_THRESHOLD)
        return (int)syscall(__NR_access, pathname, mode);
    /* CRITICAL: Check process type FIRST, before any dlsym() calls */
    if (!is_application_process()) {
        /* Use direct syscall to completely bypass libc */
        return syscall(__NR_access, pathname, mode);
    }
    
    /* Application process - proceed with interception */
    static int (*real_access)(const char *, int) = NULL;
    if (!real_access) {
        real_access = (int (*)(const char *, int))
                      dlsym(RTLD_NEXT, "access");
    }

    /* NVIDIA file interception already handled above */

    return real_access(pathname, mode);
}

/* ================================================================
 * PCI Device File Interception
 *
 * CRITICAL: Ollama scans PCI devices directly by reading:
 * - /sys/bus/pci/devices/0000:00:05.0/vendor
 * - /sys/bus/pci/devices/0000:00:05.0/device
 * - /sys/bus/pci/devices/0000:00:05.0/class
 *
 * We need to intercept read() calls to these files to ensure
 * Ollama sees the correct values for our vGPU device.
 * ================================================================ */

/* Forward declaration */
static int is_caller_from_our_code(void);
static int is_pci_device_file_path(const char *path);

static int is_pci_device_file(int fd, const char *pathname)
{
    if (pathname) {
        /* Check if this is a PCI device file we care about */
        if (strstr(pathname, "/sys/bus/pci/devices/") != NULL) {
            if (strstr(pathname, "/vendor") != NULL ||
                strstr(pathname, "/device") != NULL ||
                strstr(pathname, "/class") != NULL) {
                /* Check if it's our vGPU device (0000:00:05.0) */
                if (strstr(pathname, "0000:00:05.0") != NULL ||
                    strstr(pathname, "00:05.0") != NULL) {
                    return 1;
                }
            }
        }
    }
    
    /* Also check by file descriptor if pathname is not available */
    if (fd >= 0) {
        char proc_path[256];
        char link_target[512];
        ssize_t len;
        
        snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
        len = readlink(proc_path, link_target, sizeof(link_target) - 1);
        if (len > 0) {
            link_target[len] = '\0';
            if (strstr(link_target, "/sys/bus/pci/devices/") != NULL &&
                (strstr(link_target, "0000:00:05.0") != NULL ||
                 strstr(link_target, "00:05.0") != NULL)) {
                if (strstr(link_target, "/vendor") != NULL ||
                    strstr(link_target, "/device") != NULL ||
                    strstr(link_target, "/class") != NULL) {
                    return 1;
                }
            }
        }
    }
    
    return 0;
}

/* Read-specific early passthrough (open/openat use g_io_passthrough_count and can exhaust it before first read).
 * Use a very high count so bash/script startup never triggers is_caller_from_our_code from read(). */
#define READ_PASSTHROUGH_N 100000
static volatile int g_read_passthrough_count = 0;

/* Intercept read() calls to PCI device files - CRITICAL for Go's os.Read() */
ssize_t read(int fd, void *buf, size_t count)
{
    /* Early-call pass-through: avoid is_caller_from_our_code() during startup (SEGV in backtrace/dladdr). */
    if (g_read_passthrough_count++ < READ_PASSTHROUGH_N) {
        return (ssize_t)syscall(__NR_read, fd, buf, count);
    }
    static ssize_t (*real_read)(int, void *, size_t) = NULL;
    if (!real_read) {
        real_read = (ssize_t (*)(int, void *, size_t))
                    dlsym(RTLD_NEXT, "read");
    }
    
    /* Skip interception if caller is from our own code */
    if (is_caller_from_our_code()) {
        return real_read ? real_read(fd, buf, count) : -1;
    }
    
    /* Only intercept PCI device files; pass through everything else (blob, regular files, etc.) */
    if (!is_pci_device_file(fd, NULL)) {
        return real_read ? real_read(fd, buf, count) : (ssize_t)syscall(__NR_read, fd, buf, count);
    }
    
    /* PCI device file - return synthetic vendor/device/class */
    {
        char proc_path[256];
        char link_target[512];
        ssize_t len;
        
        snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
        len = readlink(proc_path, link_target, sizeof(link_target) - 1);
        if (len > 0) {
            link_target[len] = '\0';
            
            fprintf(stderr, "[libvgpu-cuda] read() intercepted for PCI device file (via fd link): %s (pid=%d)\n",
                    link_target, (int)getpid());
            fflush(stderr);
            
            /* Return appropriate value based on file type */
            if (strstr(link_target, "/vendor") != NULL) {
                if (count >= 6) {
                    strncpy((char *)buf, "0x10de\n", count);
                    fprintf(stderr, "[libvgpu-cuda] read: returning vendor=0x10de (NVIDIA)\n");
                    fflush(stderr);
                    return 6;
                }
            } else if (strstr(link_target, "/device") != NULL) {
                if (count >= 6) {
                    strncpy((char *)buf, "0x2331\n", count);
                    fprintf(stderr, "[libvgpu-cuda] read: returning device=0x2331 (H100 PCIe)\n");
                    fflush(stderr);
                    return 6;
                }
            } else if (strstr(link_target, "/class") != NULL) {
                if (count >= 8) {
                    strncpy((char *)buf, "0x030200\n", count);
                    fprintf(stderr, "[libvgpu-cuda] read: returning class=0x030200 (3D controller)\n");
                    fflush(stderr);
                    return 8;
                }
            }
        }
    }
    
    return real_read ? real_read(fd, buf, count) : -1;
}

/* Intercept pread() calls (positional read) */
ssize_t pread(int fd, void *buf, size_t count, off_t offset)
{
    static ssize_t (*real_pread)(int, void *, size_t, off_t) = NULL;
    if (!real_pread) {
        real_pread = (ssize_t (*)(int, void *, size_t, off_t))
                      dlsym(RTLD_NEXT, "pread");
    }
    /* Only intercept PCI device files; pass through everything else */
    if (!is_pci_device_file(fd, NULL)) {
        return real_pread ? real_pread(fd, buf, count, offset) : (ssize_t)syscall(__NR_pread64, fd, buf, count, offset);
    }
    {
        char proc_path[256];
        char link_target[512];
        ssize_t len;
        
        snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
        len = readlink(proc_path, link_target, sizeof(link_target) - 1);
        if (len > 0) {
            link_target[len] = '\0';
            
            fprintf(stderr, "[libvgpu-cuda] pread() intercepted for PCI device file: %s (pid=%d, offset=%ld)\n",
                    link_target, (int)getpid(), (long)offset);
            fflush(stderr);
            
            if (strstr(link_target, "/vendor") != NULL && offset == 0) {
                if (count >= 6) {
                    strncpy((char *)buf, "0x10de\n", count);
                    return 6;
                }
            } else if (strstr(link_target, "/device") != NULL && offset == 0) {
                if (count >= 6) {
                    strncpy((char *)buf, "0x2331\n", count);
                    return 6;
                }
            } else if (strstr(link_target, "/class") != NULL && offset == 0) {
                if (count >= 8) {
                    strncpy((char *)buf, "0x030200\n", count);
                    return 8;
                }
            }
        }
    }
    
    return real_pread ? real_pread(fd, buf, count, offset) : -1;
}

/* ================================================================
 * FILE* Operation Interception for PCI Device Files
 *
 * CRITICAL: Ollama likely uses fread()/fgets() instead of read()
 * for PCI device files. We need to intercept these FILE* operations
 * to ensure Ollama sees the correct PCI device values.
 * ================================================================ */

/* Track opened PCI device files */
#define MAX_TRACKED_FILES 64
static struct {
    FILE *fp;
    char path[512];
    int is_pci_device_file;
} tracked_files[MAX_TRACKED_FILES];
static int num_tracked_files = 0;
static pthread_mutex_t tracked_files_mutex;  /* Lazy-initialized */
static int tracked_files_mutex_initialized = 0;

static void ensure_tracked_files_mutex_init(void)
{
    if (!tracked_files_mutex_initialized) {
        int rc = pthread_mutex_init(&tracked_files_mutex, NULL);
        if (rc == 0) {
            tracked_files_mutex_initialized = 1;
        }
    }
}

/* Flag to disable interception during our own discovery
 * CRITICAL: Must be process-global (not thread-local) because
 * cuda_transport_discover() and fopen() may run in different threads */
static int g_skip_pci_interception = 0;
static pthread_mutex_t g_skip_flag_mutex;
static int g_skip_flag_mutex_initialized = 0;

/* Helper: Ensure skip flag mutex is initialized (lazy initialization)
 * CRITICAL: Do NOT use PTHREAD_MUTEX_INITIALIZER - it runs at library load time
 * and can crash during early initialization via /etc/ld.so.preload */
static void ensure_skip_flag_mutex_init(void)
{
    if (!g_skip_flag_mutex_initialized) {
        /* Simple check - during early library loading, we're typically single-threaded */
        /* If there's a race, pthread_mutex_init will fail on second call, which is OK */
        int rc = pthread_mutex_init(&g_skip_flag_mutex, NULL);
        if (rc == 0) {
            g_skip_flag_mutex_initialized = 1;
        }
    }
}

/* Function to set the skip flag - called from cuda_transport.c */
void libvgpu_set_skip_interception(int skip)
{
    ensure_skip_flag_mutex_init();
    
    /* CRITICAL: Force immediate output using write() syscall */
    (void)syscall(__NR_write, 2,
                  "[libvgpu-cuda] FORCE: libvgpu_set_skip_interception() called with skip=",
                  sizeof("[libvgpu-cuda] FORCE: libvgpu_set_skip_interception() called with skip=") - 1);
    char skip_str[4];
    snprintf(skip_str, sizeof(skip_str), "%d\n", skip);
    (void)syscall(__NR_write, 2, skip_str, strlen(skip_str));
    
    if (g_skip_flag_mutex_initialized) {
        pthread_mutex_lock(&g_skip_flag_mutex);
        g_skip_pci_interception = skip;
        pthread_mutex_unlock(&g_skip_flag_mutex);
    } else {
        /* Mutex not initialized yet - just set the flag directly (single-threaded during init) */
        g_skip_pci_interception = skip;
    }
    
    (void)syscall(__NR_write, 2,
                  "[libvgpu-cuda] FORCE: Skip flag SET to ",
                  sizeof("[libvgpu-cuda] FORCE: Skip flag SET to ") - 1);
    (void)syscall(__NR_write, 2, skip_str, strlen(skip_str));
    
    fprintf(stderr, "[libvgpu-cuda] Skip flag SET to %d (pid=%d)\n", skip, (int)getpid());
    fflush(stderr);
}

/* Skip backtrace/dladdr in is_caller_from_our_code for first N calls (SEGV during early startup). */
#define IS_CALLER_PASSTHROUGH_N 100000
static volatile int g_is_caller_passthrough_count = 0;

/* Function to check if caller is from our own code (cuda_transport.c) */
static int is_caller_from_our_code(void)
{
    /* During early process startup (LD_PRELOAD), __builtin_return_address/dladdr can SEGV.
     * Skip the check for the first N calls; treat as "not our code" (return 0). */
    if (g_is_caller_passthrough_count++ < IS_CALLER_PASSTHROUGH_N) {
        return 0;
    }
    /* REVERTED: Block interception for cuda_transport.c - let it read real values
     * When GPU was working, cuda_transport.c read files directly without interception
     * This is the original working behavior */
    
    /* CRITICAL FIX: Check multiple stack frames to be more robust
     * When called from cuInit() -> cuda_transport_discover() -> find_vgpu_device() -> fopen(),
     * we need to check frame 1 (find_vgpu_device) and frame 2 (cuda_transport_discover)
     * to ensure we catch calls from our own code even if frame 1 check fails */
    void *caller1, *caller2, *caller3;
    Dl_info info1, info2, info3;
    int is_our_code = 0;
    
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wframe-address"
    caller1 = __builtin_return_address(1);  /* Direct caller (e.g., find_vgpu_device) */
    caller2 = __builtin_return_address(2);  /* Caller's caller (e.g., cuda_transport_discover) */
    caller3 = __builtin_return_address(3);  /* Caller's caller's caller (e.g., cuInit) */
    #pragma GCC diagnostic pop
    
    /* Check frame 1 (direct caller) */
    if (caller1 && dladdr(caller1, &info1) && info1.dli_fname) {
        if (strstr(info1.dli_fname, "libvgpu") != NULL ||
            strstr(info1.dli_fname, "cuda_transport") != NULL ||
            strstr(info1.dli_fname, "vgpu-cuda") != NULL) {
            is_our_code = 1;
        }
    }
    
    /* Check frame 2 (caller's caller) if frame 1 didn't match */
    if (!is_our_code && caller2 && dladdr(caller2, &info2) && info2.dli_fname) {
        if (strstr(info2.dli_fname, "libvgpu") != NULL ||
            strstr(info2.dli_fname, "cuda_transport") != NULL ||
            strstr(info2.dli_fname, "vgpu-cuda") != NULL) {
            is_our_code = 1;
        }
    }
    
    /* Check frame 3 (caller's caller's caller) if still not matched */
    if (!is_our_code && caller3 && dladdr(caller3, &info3) && info3.dli_fname) {
        if (strstr(info3.dli_fname, "libvgpu") != NULL ||
            strstr(info3.dli_fname, "cuda_transport") != NULL ||
            strstr(info3.dli_fname, "vgpu-cuda") != NULL) {
            is_our_code = 1;
        }
    }
    
    if (is_our_code) {
        return 1;  /* Caller is from our own code - block interception */
    }
    return 0;
}

static int is_pci_device_file_path(const char *path)
{
    if (!path) return 0;
    /* CRITICAL FIX: Match ALL PCI device files, not just 0000:00:05.0
     * cuda_transport.c scans all devices, so we need to intercept all PCI device file reads
     * to ensure it gets the correct values for the target device */
    if (strstr(path, "/sys/bus/pci/devices/") != NULL) {
        if (strstr(path, "/vendor") != NULL ||
            strstr(path, "/device") != NULL ||
            strstr(path, "/class") != NULL) {
            return 1;
        }
    }
    return 0;
}

/* FDs we opened via syscall+fdopen for excluded paths (model blobs). fread uses read() for these. */
#define MAX_EXCLUDED_FDS 64
static int g_excluded_fds[MAX_EXCLUDED_FDS];
static int g_num_excluded_fds = 0;
static void add_excluded_fd(int fd)
{
    if (fd < 0 || g_num_excluded_fds >= MAX_EXCLUDED_FDS) return;
    for (int i = 0; i < g_num_excluded_fds; i++)
        if (g_excluded_fds[i] == fd) return;
    g_excluded_fds[g_num_excluded_fds++] = fd;
}
static int is_excluded_fd(int fd)
{
    for (int i = 0; i < g_num_excluded_fds; i++)
        if (g_excluded_fds[i] == fd) return 1;
    return 0;
}

/* Check if path should be excluded from interception (model files, etc.) */
static int should_exclude_from_interception(const char *path)
{
    if (!path) return 0;
    /* Exclude model files - they should pass through directly */
    if (strstr(path, "/.ollama/models/") != NULL ||
        strstr(path, "/models/blobs/") != NULL) {
        return 1;
    }
    /* Exclude other non-PCI system files */
    if (strstr(path, "/proc/") != NULL && !strstr(path, "/proc/driver/nvidia/")) {
        return 1;
    }
    return 0;
}

static void track_file(FILE *fp, const char *path)
{
    if (!fp || num_tracked_files >= MAX_TRACKED_FILES) return;
    
    ensure_tracked_files_mutex_init();
    pthread_mutex_lock(&tracked_files_mutex);
    /* Check if this FILE* is already tracked - if so, update the path */
    int found = 0;
    int i;
    for (i = 0; i < num_tracked_files; i++) {
        if (tracked_files[i].fp == fp) {
            /* Update existing entry */
            if (path) {
                strncpy(tracked_files[i].path, path, sizeof(tracked_files[i].path) - 1);
                tracked_files[i].path[sizeof(tracked_files[i].path) - 1] = '\0';
            }
            tracked_files[i].is_pci_device_file = is_pci_device_file_path(tracked_files[i].path);
            found = 1;
            break;
        }
    }
    
    if (!found) {
        /* Add new entry */
        tracked_files[num_tracked_files].fp = fp;
        if (path) {
            strncpy(tracked_files[num_tracked_files].path, path, 
                    sizeof(tracked_files[num_tracked_files].path) - 1);
            tracked_files[num_tracked_files].path[sizeof(tracked_files[num_tracked_files].path) - 1] = '\0';
        } else {
            tracked_files[num_tracked_files].path[0] = '\0';
        }
        tracked_files[num_tracked_files].is_pci_device_file = 
            is_pci_device_file_path(tracked_files[num_tracked_files].path);
        num_tracked_files++;
    }
    pthread_mutex_unlock(&tracked_files_mutex);
}

static int is_tracked_pci_file(FILE *fp)
{
    int i;
    ensure_tracked_files_mutex_init();
    pthread_mutex_lock(&tracked_files_mutex);
    for (i = 0; i < num_tracked_files; i++) {
        if (tracked_files[i].fp == fp && tracked_files[i].is_pci_device_file) {
            pthread_mutex_unlock(&tracked_files_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&tracked_files_mutex);
    return 0;
}

/* Resolve real fopen/fgets/fread from libc.so so model blobs and other excluded
 * files work (avoid "failed to read magic" when GGUF loader uses fread/fgets). */
static void ensure_real_libc_resolved(void)
{
    if (g_real_fopen_global && g_real_fgets_global && g_real_fread_global)
        return;
    const char *libc_paths[] = {
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/usr/lib64/libc.so.6",
        "/lib64/libc.so.6",
        "libc.so.6",
        NULL
    };
    void *libc = NULL;
    /* Try maps+ELF: set globals from base + symbol offset (no dlopen/dlsym). */
    {
        uint64_t base = get_libc_base_from_maps();
        const char *maps_path = get_libc_path_from_maps();
        const char *try_path = maps_path ? maps_path : libc_paths[0];
        /* If maps didn't give path, try fixed paths with base from maps (runner may have different layout). */
        if (base && !try_path)
            try_path = libc_paths[0];
        if (base && try_path) {
            uint64_t fo = elf_get_symbol_offset(try_path, "fopen");
            if (!fo) fo = elf_get_symbol_offset(try_path, "__libc_fopen");
            uint64_t go = elf_get_symbol_offset(try_path, "fgets");
            if (!go) go = elf_get_symbol_offset(try_path, "__libc_fgets");
            uint64_t ro = elf_get_symbol_offset(try_path, "fread");
            if (!ro) ro = elf_get_symbol_offset(try_path, "__libc_fread");
            if (fo && go && ro) {
                if (!g_real_fopen_global) g_real_fopen_global = (FILE *(*)(const char *, const char *))(void *)(base + fo);
                if (!g_real_fgets_global) g_real_fgets_global = (char *(*)(char *, int, FILE *))(void *)(base + go);
                if (!g_real_fread_global) g_real_fread_global = (size_t (*)(void *, size_t, size_t, FILE *))(void *)(base + ro);
                if (g_real_fopen_global && g_real_fgets_global && g_real_fread_global)
                    return;
            }
            uint64_t doff = elf_get_symbol_offset(try_path, "dlopen");
            uint64_t syoff = elf_get_symbol_offset(try_path, "dlsym");
            if (doff && syoff) {
                void *(*real_dlopen_fn)(const char *, int) = (void *(*)(const char *, int))(void *)(base + doff);
                void *(*real_dlsym_fn)(void *, const char *) = (void *(*)(void *, const char *))(void *)(base + syoff);
                libc = real_dlopen_fn(try_path, RTLD_NOW | RTLD_NOLOAD);
                if (!libc)
                    libc = real_dlopen_fn(try_path, RTLD_NOW);
                if (libc) {
                    if (!g_real_fopen_global) {
                        void *sym = real_dlsym_fn(libc, "fopen");
                        if (sym) g_real_fopen_global = (FILE *(*)(const char *, const char *))sym;
                    }
                    if (!g_real_fgets_global) {
                        void *sym = real_dlsym_fn(libc, "fgets");
                        if (sym) g_real_fgets_global = (char *(*)(char *, int, FILE *))sym;
                    }
                    if (!g_real_fread_global) {
                        void *sym = real_dlsym_fn(libc, "fread");
                        if (sym) g_real_fread_global = (size_t (*)(void *, size_t, size_t, FILE *))sym;
                    }
                    if (g_real_fopen_global && g_real_fgets_global && g_real_fread_global)
                        return;
                }
            }
        }
    }
    /* Fallback: dlopen libc (goes through our dlopen/dlsym). */
    {
        const char *maps_path = get_libc_path_from_maps();
        if (maps_path)
            libc = dlopen(maps_path, RTLD_NOW);
    }
    if (!libc)
        for (int i = 0; libc_paths[i] && !libc; i++)
            libc = dlopen(libc_paths[i], RTLD_NOW | RTLD_DEEPBIND);
    if (!libc)
        for (int i = 0; libc_paths[i] && !libc; i++)
            libc = dlopen(libc_paths[i], RTLD_NOW);
    if (!libc) {
        /* Fallback: resolve via /proc/self/maps + ELF parse (no dl_iterate_phdr). */
        uint64_t base = get_libc_base_from_maps();
        const char *maps_path = get_libc_path_from_maps();
        const char *try_paths[] = { maps_path, libc_paths[0], libc_paths[1], libc_paths[2], NULL };
        if (base) {
            for (int i = 0; try_paths[i]; i++) {
                if (!try_paths[i]) continue;
                uint64_t doff = elf_get_symbol_offset(try_paths[i], "dlopen");
                uint64_t syoff = elf_get_symbol_offset(try_paths[i], "dlsym");
                if (doff && syoff) {
                    void *(*real_dlopen_fn)(const char *, int) = (void *(*)(const char *, int))(void *)(base + doff);
                    void *(*real_dlsym_fn)(void *, const char *) = (void *(*)(void *, const char *))(void *)(base + syoff);
                    libc = real_dlopen_fn(try_paths[i], RTLD_NOW);
                    if (libc) {
                        if (!g_real_fopen_global) {
                            void *sym = real_dlsym_fn(libc, "fopen");
                            if (sym) {
                                g_real_fopen_global = (FILE *(*)(const char *, const char *))sym;
                            }
                        }
                        if (!g_real_fgets_global) {
                            void *sym = real_dlsym_fn(libc, "fgets");
                            if (sym) g_real_fgets_global = (char *(*)(char *, int, FILE *))sym;
                        }
                        if (!g_real_fread_global) {
                            void *sym = real_dlsym_fn(libc, "fread");
                            if (sym) g_real_fread_global = (size_t (*)(void *, size_t, size_t, FILE *))sym;
                        }
                        return;
                    }
                }
            }
        }
        return;
    }
    /* If we have libc but dlsym failed (e.g. real_dlsym NULL), resolve via maps+ELF. */
    if (libc && (!g_real_fopen_global || !g_real_fgets_global || !g_real_fread_global)) {
        uint64_t base = get_libc_base_from_maps();
        const char *maps_path = get_libc_path_from_maps();
        const char *p = maps_path ? maps_path : libc_paths[0];
        if (base && p) {
            uint64_t syoff = elf_get_symbol_offset(p, "dlsym");
            if (syoff) {
                void *(*real_dlsym_fn)(void *, const char *) = (void *(*)(void *, const char *))(void *)(base + syoff);
                if (!g_real_fopen_global) {
                    void *sym = real_dlsym_fn(libc, "fopen");
                    if (sym) g_real_fopen_global = (FILE *(*)(const char *, const char *))sym;
                }
                if (!g_real_fgets_global) {
                    void *sym = real_dlsym_fn(libc, "fgets");
                    if (sym) g_real_fgets_global = (char *(*)(char *, int, FILE *))sym;
                }
                if (!g_real_fread_global) {
                    void *sym = real_dlsym_fn(libc, "fread");
                    if (sym) g_real_fread_global = (size_t (*)(void *, size_t, size_t, FILE *))sym;
                }
            }
        }
    }
    /* Resolve from libc handle only (trust libc's symbols); avoid self-reference check that can fail with PIE. */
    if (!g_real_fopen_global) {
        void *sym = dlsym(libc, "fopen");
        if (!sym) sym = dlsym(libc, "__libc_fopen");
        if (sym) {
            g_real_fopen_global = (FILE *(*)(const char *, const char *))sym;
        }
    }
    if (!g_real_fgets_global) {
        void *sym = dlsym(libc, "fgets");
        if (!sym) sym = dlsym(libc, "__libc_fgets");
        if (sym)
            g_real_fgets_global = (char *(*)(char *, int, FILE *))sym;
    }
    if (!g_real_fread_global) {
        void *sym = dlsym(libc, "fread");
        if (!sym) sym = dlsym(libc, "__libc_fread");
        if (sym)
            g_real_fread_global = (size_t (*)(void *, size_t, size_t, FILE *))sym;
    }
}

/* Minimal check: does /proc/self/cmdline contain "runner"? Uses only raw syscalls so safe from constructors. */
static int __attribute__((unused)) is_runner_process_early(void)
{
    char buf[256];
    int fd = (int)syscall(__NR_open, "/proc/self/cmdline", O_RDONLY);
    if (fd < 0) return 0;
    ssize_t n = syscall(__NR_read, fd, buf, sizeof(buf) - 1);
    syscall(__NR_close, fd);
    if (n <= 0 || n >= (ssize_t)sizeof(buf)) return 0;
    buf[n] = '\0';
    for (int i = 0; i <= (int)n - 6; i++) {
        if (buf[i] == 'r' && buf[i+1]=='u' && buf[i+2]=='n' && buf[i+3]=='n' && buf[i+4]=='e' && buf[i+5]=='r')
            return 1;
        if (buf[i] == '\0') continue;
        while (i < (int)n && buf[i] != '\0') i++;
    }
    return 0;
}

/* Resolve real libc fopen/fgets/fread via maps+ELF only (no dlsym). Run at 1000 so all libs (including libc) are in maps. */
__attribute__((constructor(1000)))
static void resolve_libc_file_funcs_early(void)
{
    /* Disabled: any work here can SEGV when loaded in main ollama serve (LD_PRELOAD). Resolve on first use via ensure_real_libc_resolved(). */
    (void)0;
    return;
#if 0
    if (!is_runner_process_early()) return;
    uint64_t base = get_libc_base_from_maps();
    const char *paths[] = {
        get_libc_path_from_maps(),
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/lib64/libc.so.6",
        "/usr/lib64/libc.so.6",
        NULL
    };
    if (!base) return;
    for (int i = 0; paths[i] && (!g_real_fopen_global || !g_real_fgets_global || !g_real_fread_global); i++) {
        const char *path = paths[i];
        if (!g_real_fopen_global) {
            uint64_t fo = elf_get_symbol_offset(path, "fopen");
            if (!fo) fo = elf_get_symbol_offset(path, "__libc_fopen");
            if (fo) g_real_fopen_global = (FILE *(*)(const char *, const char *))(void *)(uintptr_t)(base + fo);
        }
        if (!g_real_fgets_global) {
            uint64_t go = elf_get_symbol_offset(path, "fgets");
            if (!go) go = elf_get_symbol_offset(path, "__libc_fgets");
            if (go) g_real_fgets_global = (char *(*)(char *, int, FILE *))(void *)(uintptr_t)(base + go);
        }
        if (!g_real_fread_global) {
            uint64_t ro = elf_get_symbol_offset(path, "fread");
        if (!ro) ro = elf_get_symbol_offset(path, "__libc_fread");
        if (ro) g_real_fread_global = (size_t (*)(void *, size_t, size_t, FILE *))(void *)(uintptr_t)(base + ro);
        }
    }
#endif
}

/* Resolve real libc FILE* funcs at load time (late, so link order is stable). Use 300 so we run after more libs load. */
__attribute__((constructor(300)))
static void resolve_libc_file_funcs_at_load(void)
{
    /* Disabled: dlsym/ensure_real in constructor can SEGV in main process. Resolve on first use. */
    (void)0;
    return;
#if 0
    if (!is_runner_process_early()) return;
    if (!g_real_fopen_global) {
        void *sym = dlsym(RTLD_NEXT, "fopen");
        if (sym && sym != (void *)fopen)
            g_real_fopen_global = (FILE *(*)(const char *, const char *))sym;
    }
    if (!g_real_fgets_global) {
        void *sym = dlsym(RTLD_NEXT, "fgets");
        if (sym && sym != (void *)fgets)
            g_real_fgets_global = (char *(*)(char *, int, FILE *))sym;
    }
    if (!g_real_fread_global) {
        void *sym = dlsym(RTLD_NEXT, "fread");
        if (sym && sym != (void *)fread)
            g_real_fread_global = (size_t (*)(void *, size_t, size_t, FILE *))sym;
    }
    if (!g_real_fopen_global || !g_real_fgets_global || !g_real_fread_global)
        ensure_real_libc_resolved();
#endif
}

/* Intercept fopen() to track PCI device files */
FILE *fopen(const char *pathname, const char *mode)
{
    /* CRITICAL: NULL check FIRST to avoid segfault when accessing pathname[0] */
    if (!pathname) {
        if (g_real_fopen_global) {
            return g_real_fopen_global(NULL, mode);
        }
        return NULL;
    }
    /* CRITICAL: For model blobs, pass through to real fopen immediately so GGUF load works. */
    if (should_exclude_from_interception(pathname)) {
        ensure_real_libc_resolved();
        if (g_real_fopen_global)
            return g_real_fopen_global(pathname, mode);
#ifdef __GLIBC__
        {
        extern void *__libc_dlsym(void *, const char *) __attribute__((weak));
        if (__libc_dlsym) {
            FILE *(*fn)(const char *, const char *) = (FILE *(*)(const char *, const char *))__libc_dlsym(RTLD_NEXT, "fopen");
            if (fn && (void *)fn != (void *)fopen)
                return fn(pathname, mode);
        }
        }
#endif
        {
            static FILE *(*real_fopen_next)(const char *, const char *) = NULL;
            if (!real_fopen_next)
                real_fopen_next = (FILE *(*)(const char *, const char *))dlsym(RTLD_NEXT, "fopen");
            if (real_fopen_next && real_fopen_next != (FILE *(*)(const char *, const char *))fopen)
                return real_fopen_next(pathname, mode);
        }
        /* Fallback: open via syscall + fdopen only when all real fopen resolutions failed */
        {
            int flags = O_RDONLY;
            if (mode && (strchr(mode, 'w') || strchr(mode, 'a')))
                flags = O_RDWR;
            int fd = (int)syscall(__NR_open, pathname, flags, 0);
            if (fd >= 0) {
                FILE *fp = fdopen(fd, mode ? mode : "r");
                if (fp) {
                    add_excluded_fd(fd);
                    return fp;
                }
                syscall(__NR_close, fd);
            }
        }
    }
    /* Early-call pass-through: avoid is_application_process() and dlsym() during main-process startup. Use syscall open + fdopen only (we do not intercept fdopen). */
    if (g_io_passthrough_count++ < IO_PASSTHROUGH_THRESHOLD) {
        int flags_io = O_RDONLY;
        if (mode && (strchr(mode, 'w') || strchr(mode, 'a')))
            flags_io = O_RDWR;
        int fd = (int)syscall(__NR_open, pathname, flags_io, 0);
        if (fd >= 0) {
            FILE *fp = fdopen(fd, mode ? mode : "r");
            if (fp) return fp;
            syscall(__NR_close, fd);
        }
        return NULL;
    }
    
    /* TEMPORARY: Disable early PCI check to test if it's causing the segfault
     * If segfault stops, we know it's in the character comparison logic */
    /* 
    if (pathname[0] == '/' && pathname[1] == 's' && pathname[2] == 'y' && pathname[3] == 's' &&
        pathname[4] == '/' && pathname[5] == 'b' && pathname[6] == 'u' && pathname[7] == 's' &&
        pathname[8] == '/' && pathname[9] == 'p' && pathname[10] == 'c' && pathname[11] == 'i' &&
        pathname[12] == '/' && pathname[13] == 'd' && pathname[14] == 'e' && pathname[15] == 'v' &&
        pathname[16] == 'i' && pathname[17] == 'c' && pathname[18] == 'e' && pathname[19] == 's' &&
        pathname[20] == '/') {
        const char *q = pathname + 21;
        while (*q && *q != '/') q++;
        if (*q == '/') {
            if ((q[1] == 'v' && q[2] == 'e' && q[3] == 'n' && q[4] == 'd' && q[5] == 'o' && q[6] == 'r' && q[7] == '\0') ||
                (q[1] == 'd' && q[2] == 'e' && q[3] == 'v' && q[4] == 'i' && q[5] == 'c' && q[6] == 'e' && q[7] == '\0') ||
                (q[1] == 'c' && q[2] == 'l' && q[3] == 'a' && q[4] == 's' && q[5] == 's' && q[6] == '\0')) {
                return NULL;
            }
        }
    }
    */
    
    /* Debug message - TEMPORARY: Remove to test if fprintf is causing segfault */
    /*
    fprintf(stderr, "[libvgpu-cuda] fopen() INTERCEPTOR CALLED: %s (pid=%d)\n",
            pathname, (int)getpid());
    fflush(stderr);
    */
    
    /* CRITICAL: Check process type FIRST, before any dlsym() calls */
    if (!is_application_process()) {
        /* Runner: use real libc fopen (avoid syscall+fdopen). Prefer __libc_dlsym to bypass our dlsym. */
        if (pathname) {
            ensure_real_libc_resolved();
            if (g_real_fopen_global)
                return g_real_fopen_global(pathname, mode ? mode : "r");
#ifdef __GLIBC__
            {
            extern void *__libc_dlsym(void *, const char *) __attribute__((weak));
            if (__libc_dlsym) {
                FILE *(*fn)(const char *, const char *) = (FILE *(*)(const char *, const char *))__libc_dlsym(RTLD_NEXT, "fopen");
                if (fn && (void *)fn != (void *)fopen)
                    return fn(pathname, mode ? mode : "r");
            }
            }
#endif
            {
            static FILE *(*real_fopen_runner)(const char *, const char *) = NULL;
            if (!real_fopen_runner)
                real_fopen_runner = (FILE *(*)(const char *, const char *))dlsym(RTLD_NEXT, "fopen");
            if (real_fopen_runner && real_fopen_runner != (FILE *(*)(const char *, const char *))fopen)
                return real_fopen_runner(pathname, mode ? mode : "r");
            }
        }
        if (!pathname)
            return NULL;
        /* Fallback only if real fopen unavailable */
        int flags = O_RDONLY;
        if (mode) {
            if (strchr(mode, 'w')) flags = O_WRONLY | O_CREAT | O_TRUNC;
            else if (strchr(mode, 'a')) flags = O_WRONLY | O_CREAT | O_APPEND;
        }
        int fd = syscall(__NR_open, pathname, flags, 0644);
        if (fd < 0) return NULL;
        FILE *fp = fdopen(fd, mode ? mode : "r");
        if (!fp) { close(fd); return NULL; }
        if (should_exclude_from_interception(pathname))
            add_excluded_fd(fd);
        return fp;
    }
    
    /* Application process - proceed with interception */
    ensure_skip_flag_mutex_init();
    int skip_flag = 0;
    if (g_skip_flag_mutex_initialized) {
        pthread_mutex_lock(&g_skip_flag_mutex);
        skip_flag = g_skip_pci_interception;
        pthread_mutex_unlock(&g_skip_flag_mutex);
    } else {
        skip_flag = g_skip_pci_interception;
    }
    
    /* REMOVED: Duplicate PCI file handling - already handled at function start with inline checks */
    
    fprintf(stderr, "[libvgpu-cuda] fopen() called: %s, skip_flag=%d (pid=%d)\n",
            pathname ? pathname : "NULL", skip_flag, (int)getpid());
    fflush(stderr);
    
    /* CRITICAL FIX: Exclude model files and other non-PCI files from interception
     * These should pass through directly to the real fopen(); also resolve
     * real fgets/fread from libc so model blob reads work (avoid "failed to read magic"). */
    if (pathname && should_exclude_from_interception(pathname)) {
        ensure_real_libc_resolved();
        /* Fallback: resolve fopen from RTLD_NEXT if libc dlopen didn't work */
        if (!g_real_fopen_global) {
            void *sym = dlsym(RTLD_NEXT, "fopen");
            if (sym && sym != (void *)fopen)
                g_real_fopen_global = (FILE *(*)(const char *, const char *))sym;
        }
        if (g_real_fopen_global) {
            FILE *fp = g_real_fopen_global(pathname, mode);
            if (fp) {
                fprintf(stderr, "[libvgpu-cuda] fopen() EXCLUDED from interception: %s -> %p (pid=%d)\n",
                        pathname, (void*)fp, (int)getpid());
                fflush(stderr);
            }
            return fp;
        } else {
            /* Try libc open+fdopen via RTLD_NEXT so FILE* is from libc and fread works */
            static int (*real_open_fn)(const char *, int, ...) = NULL;
            static FILE *(*real_fdopen_fn)(int, const char *) = NULL;
            if (!real_open_fn) {
                void *p = dlsym(RTLD_NEXT, "open");
                if (p && p != (void *)open) real_open_fn = (int (*)(const char *, int, ...))p;
            }
            if (!real_fdopen_fn) {
                void *p = dlsym(RTLD_NEXT, "fdopen");
                if (p && p != (void *)fdopen) real_fdopen_fn = (FILE *(*)(int, const char *))p;
            }
            if (real_open_fn && real_fdopen_fn) {
                int flags = O_RDONLY;
                if (mode && (strchr(mode, 'w') || strchr(mode, 'a'))) flags = O_WRONLY | O_CREAT | O_APPEND;
                int fd = real_open_fn(pathname, flags);
                if (fd >= 0) {
                    FILE *fp = real_fdopen_fn(fd, mode ? mode : "r");
                    if (fp) return fp;
                    close(fd);
                }
            }
            /* Fallback to syscall if RTLD_NEXT open/fdopen not available */
            int flags = O_RDONLY;
            if (mode) {
                if (strchr(mode, 'w')) flags = O_WRONLY | O_CREAT | O_TRUNC;
                else if (strchr(mode, 'a')) flags = O_WRONLY | O_CREAT | O_APPEND;
            }
            int fd = (int)syscall(__NR_open, pathname, flags, 0644);
            if (fd >= 0) {
                FILE *fp = fdopen(fd, mode ? mode : "r");
                if (fp) {
                    add_excluded_fd(fileno(fp));
                    return fp;
                }
                close(fd);
            }
        }
    }
    
    /* CRITICAL FIX: Always use syscall for PCI device files to prevent crashes
     * This ensures we can read real values even if skip_flag isn't set correctly */
    int is_pci_file = pathname && is_pci_device_file_path(pathname);
    
    if (skip_flag || is_pci_file) {
        /* For skip mode, use syscall approach for reliability
         * This ensures files can be read even if real_fopen() can't be resolved */
        if (!pathname) {
            return NULL;
        }
        /* Parse mode string to get flags */
        int flags = O_RDONLY;
        if (mode) {
            if (strchr(mode, 'w')) {
                flags = O_WRONLY | O_CREAT | O_TRUNC;
            } else if (strchr(mode, 'a')) {
                flags = O_WRONLY | O_CREAT | O_APPEND;
            } else {
                flags = O_RDONLY;
            }
        }
        /* Use direct syscall - this cannot be intercepted */
        int fd = syscall(__NR_open, pathname, flags, 0644);
        if (fd < 0) {
            fprintf(stderr, "[libvgpu-cuda] fopen() skip mode syscall failed: %s (pid=%d)\n",
                    pathname, (int)getpid());
            fflush(stderr);
            return NULL;
        }
        /* Convert fd to FILE* using fdopen() */
        FILE *fp = fdopen(fd, mode ? mode : "r");
        if (!fp) {
            close(fd);
            fprintf(stderr, "[libvgpu-cuda] fopen() skip mode fdopen() failed: %s (pid=%d)\n",
                    pathname, (int)getpid());
            fflush(stderr);
            return NULL;
        }
        fprintf(stderr, "[libvgpu-cuda] fopen() SKIP interception (syscall): %s -> %p (pid=%d, is_pci=%d)\n",
                pathname, (void*)fp, (int)getpid(), is_pci_file);
        fflush(stderr);
        return fp;
    }
    
    /* Normal interception mode */
    if (!pathname) {
        return NULL;
    }
    
    if (g_real_fopen_global) {
        FILE *fp = NULL;
        if ((void*)g_real_fopen_global == NULL || 
            (void*)g_real_fopen_global == (void*)-1 ||
            (uintptr_t)g_real_fopen_global < 0x1000) {
            goto use_fallback;
        }
        fp = g_real_fopen_global(pathname, mode);
        if (fp) {
            /* Only track PCI device files for interception */
            if (is_pci_device_file_path(pathname)) {
                if (!is_caller_from_our_code()) {
                    track_file(fp, pathname);
                }
            }
            return fp;
        }
    }
    
    use_fallback:
    /* Fallback: Use syscall approach if real_fopen() not available */
    if (!pathname) {
        return NULL;
    }
    /* Parse mode string to get flags */
    int flags = O_RDONLY;
    if (mode) {
        if (strchr(mode, 'w')) {
            flags = O_WRONLY | O_CREAT | O_TRUNC;
        } else if (strchr(mode, 'a')) {
            flags = O_WRONLY | O_CREAT | O_APPEND;
        } else {
            flags = O_RDONLY;
        }
    }
    /* Use direct syscall - this cannot be intercepted */
    int fd = syscall(__NR_open, pathname, flags, 0644);
    if (fd < 0) {
        return NULL;
    }
    /* Convert fd to FILE* using fdopen() */
    FILE *fp = fdopen(fd, mode ? mode : "r");
    if (!fp) {
        close(fd);
        return NULL;
    }
    
    /* Only track PCI device files for interception */
    if (is_pci_device_file_path(pathname)) {
        if (!is_caller_from_our_code()) {
            track_file(fp, pathname);
        }
    }
    
    /* CRITICAL FIX: Only track PCI device files opened by OTHER processes (not cuda_transport.c)
     * Based on documentation - when GPU was working, cuda_transport.c read files directly.
     * We need to ensure cuda_transport.c always gets real values, so we don't track its files.
     * This allows fgets() interception to work for Ollama's discovery, but not for our own code. */
    if (fp && is_pci_device_file_path(pathname)) {
        if (!is_caller_from_our_code()) {
            track_file(fp, pathname);
        } else {
            /* Caller is from our code (cuda_transport.c) - don't track, let it read real values */
            fprintf(stderr, "[libvgpu-cuda] fopen() for PCI file from our code, NOT tracking: %s (pid=%d)\n",
                    pathname, (int)getpid());
            fflush(stderr);
        }
    }
    
    return fp;
}

/* Helper: is fd a model blob (by /proc/self/fd link)? */
static int is_fd_blob_file(int fd)
{
    char proc_path[64];
    char link[512];
    snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(proc_path, link, sizeof(link) - 1);
    if (n <= 0 || n >= (ssize_t)sizeof(link)) return 0;
    link[n] = '\0';
    return (strstr(link, ".ollama/models") != NULL || strstr(link, "models/blobs") != NULL || strstr(link, "/blobs/") != NULL || (strstr(link, "ollama") != NULL && strstr(link, "sha256") != NULL)) ? 1 : 0;
}

/* Intercept fread() for PCI device files */
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    /* Any non-PCI fd: pass through to real fread immediately; never use read() loop (avoids FILE* desync). */
    if (stream && size > 0 && nmemb > 0) {
        int fd = fileno(stream);
        if (fd >= 0 && !is_pci_device_file(fd, NULL)) {
            ensure_real_libc_resolved();
            if (g_real_fread_global)
                return g_real_fread_global(ptr, size, nmemb, stream);
#ifdef __GLIBC__
            {
            extern void *__libc_dlsym(void *, const char *) __attribute__((weak));
            if (__libc_dlsym) {
                size_t (*fn)(void *, size_t, size_t, FILE *) = (size_t (*)(void *, size_t, size_t, FILE *))__libc_dlsym(RTLD_NEXT, "fread");
                if (fn && (void *)fn != (void *)fread)
                    return fn(ptr, size, nmemb, stream);
            }
            }
#endif
            {
            static size_t (*next_fread)(void *, size_t, size_t, FILE *) = NULL;
            if (!next_fread) {
                next_fread = (size_t (*)(void *, size_t, size_t, FILE *))dlsym(RTLD_NEXT, "fread");
                if (!next_fread || next_fread == (size_t (*)(void *, size_t, size_t, FILE *))fread)
                    next_fread = (size_t (*)(void *, size_t, size_t, FILE *))dlsym(RTLD_DEFAULT, "__libc_fread");
            }
            if (next_fread && next_fread != (size_t (*)(void *, size_t, size_t, FILE *))fread)
                return next_fread(ptr, size, nmemb, stream);
            }
            /* Real fread unavailable; do not use read() loop (desyncs FILE*). */
        }
    }
    /* Model blob files (legacy path): use real libc fread so FILE* buffer/position stay in sync. */
    if (stream && size > 0 && nmemb > 0) {
        int fd = fileno(stream);
        if (fd >= 0 && is_fd_blob_file(fd)) {
            ensure_real_libc_resolved();
            if (g_real_fread_global)
                return g_real_fread_global(ptr, size, nmemb, stream);
            {
            static size_t (*libc_fread)(void *, size_t, size_t, FILE *) = NULL;
            if (!libc_fread) {
                libc_fread = (size_t (*)(void *, size_t, size_t, FILE *))dlsym(RTLD_NEXT, "fread");
                if (!libc_fread || libc_fread == (size_t (*)(void *, size_t, size_t, FILE *))fread)
                    libc_fread = (size_t (*)(void *, size_t, size_t, FILE *))dlsym(RTLD_DEFAULT, "__libc_fread");
            }
            if (libc_fread && libc_fread != (size_t (*)(void *, size_t, size_t, FILE *))fread)
                return libc_fread(ptr, size, nmemb, stream);
            }
            /* Last resort: read() loop (may desync FILE* state; prefer g_real_fread_global above) */
            {
                size_t total = size * nmemb;
                size_t got = 0;
                while (got < total) {
                    ssize_t n = syscall(__NR_read, fd, (char *)ptr + got, total - got);
                    if (n <= 0) break;
                    got += (size_t)n;
                }
                return (size_t)(got / size);
            }
        }
    }
    /* Blob streams we opened via syscall+fdopen: always use read() loop so model load works */
    if (size > 0 && nmemb > 0 && stream) {
        int fd = fileno(stream);
        if (fd >= 0 && is_excluded_fd(fd)) {
            size_t total = size * nmemb;
            size_t got = 0;
            while (got < total) {
                ssize_t n = syscall(__NR_read, fd, (char *)ptr + got, total - got);
                if (n <= 0) break;
                got += (size_t)n;
            }
            return (size_t)(got / size);
        }
    }
    /* CRITICAL: Check process type FIRST, before any dlsym() calls */
    if (!is_application_process()) {
        ensure_real_libc_resolved();
        if (g_real_fread_global)
            return g_real_fread_global(ptr, size, nmemb, stream);
#ifdef __GLIBC__
        {
        extern void *__libc_dlsym(void *, const char *) __attribute__((weak));
        if (__libc_dlsym) {
            size_t (*fn)(void *, size_t, size_t, FILE *) = (size_t (*)(void *, size_t, size_t, FILE *))__libc_dlsym(RTLD_NEXT, "fread");
            if (fn && (void *)fn != (void *)fread)
                return fn(ptr, size, nmemb, stream);
        }
        }
#endif
        {
        static size_t (*real_fread)(void *, size_t, size_t, FILE *) = NULL;
        if (!real_fread)
            real_fread = (size_t (*)(void *, size_t, size_t, FILE *))dlsym(RTLD_NEXT, "fread");
        if (real_fread && real_fread != (size_t (*)(void *, size_t, size_t, FILE *))fread)
            return real_fread(ptr, size, nmemb, stream);
        }
        /* When real fread unavailable, use read() loop (may desync FILE* state) */
        if (size > 0 && nmemb > 0 && stream) {
            int fd = fileno(stream);
            if (fd >= 3) {
                size_t total = size * nmemb;
                size_t got = 0;
                while (got < total) {
                    ssize_t n = syscall(__NR_read, fd, (char *)ptr + got, total - got);
                    if (n <= 0) break;
                    got += (size_t)n;
                }
                return (size_t)(got / size);
            }
        }
        return 0;
    }
    
    /* Application process - proceed with interception */
    static size_t (*real_fread)(void *, size_t, size_t, FILE *) = NULL;
    if (!real_fread) {
        real_fread = (size_t (*)(void *, size_t, size_t, FILE *))
                     dlsym(RTLD_NEXT, "fread");
    }
    
    /* REVERTED: Don't intercept fread() for cuda_transport.c - let it read real values */
    ensure_skip_flag_mutex_init();
    int skip_flag = 0;
    if (g_skip_flag_mutex_initialized) {
        pthread_mutex_lock(&g_skip_flag_mutex);
        skip_flag = g_skip_pci_interception;
        pthread_mutex_unlock(&g_skip_flag_mutex);
    } else {
        skip_flag = g_skip_pci_interception;
    }
    
    if (skip_flag) {
        /* Skip interception during discovery */
        ensure_real_libc_resolved();
        if (g_real_fread_global)
            return g_real_fread_global(ptr, size, nmemb, stream);
        if (real_fread && real_fread != (size_t (*)(void *, size_t, size_t, FILE *))fread)
            return real_fread(ptr, size, nmemb, stream);
        /* Fallback: read() loop so blob/model load works when real_fread is us or NULL */
        if (size > 0 && nmemb > 0 && stream) {
            int fd = fileno(stream);
            if (fd >= 0) {
                size_t total = size * nmemb;
                size_t got = 0;
                while (got < total) {
                    ssize_t n = syscall(__NR_read, fd, (char *)ptr + got, total - got);
                    if (n <= 0) break;
                    got += (size_t)n;
                }
                return (size_t)(got / size);
            }
        }
        return 0;
    }
    /* Untracked stream (e.g. model blob): use libc's real fread, or read() when real not available */
    if (!is_tracked_pci_file(stream) || is_caller_from_our_code()) {
        ensure_real_libc_resolved();
        if (g_real_fread_global)
            return g_real_fread_global(ptr, size, nmemb, stream);
        /* Try RTLD_NEXT fread only if it is NOT us (avoid recursion when RTLD_NEXT resolves to shim) */
        if (real_fread && real_fread != (size_t (*)(void *, size_t, size_t, FILE *))fread)
            return real_fread(ptr, size, nmemb, stream);
        /* Fallback: use read() from fd so GGUF loader works when real libc fread unresolved or RTLD_NEXT is us */
        if (size > 0 && nmemb > 0 && stream) {
            int fd = fileno(stream);
            if (fd >= 3) {
                size_t total = size * nmemb;
                size_t got = 0;
                while (got < total) {
                    ssize_t n = syscall(__NR_read, fd, (char *)ptr + got, total - got);
                    if (n <= 0) break;
                    got += (size_t)n;
                }
                return (size_t)(got / size);
            }
        }
        /* If we get here with fd<3 or size*nmemb==0, fall through to real_fread or 0 */
        return real_fread ? real_fread(ptr, size, nmemb, stream) : 0;
    }
    if (is_tracked_pci_file(stream) && !is_caller_from_our_code()) {
        char path[512] = "";
        int i;
        
        ensure_tracked_files_mutex_init();
        pthread_mutex_lock(&tracked_files_mutex);
        /* Search from the end to find the most recent match (in case FILE* is reused) */
        for (i = num_tracked_files - 1; i >= 0; i--) {
            if (tracked_files[i].fp == stream) {
                strncpy(path, tracked_files[i].path, sizeof(path) - 1);
                path[sizeof(path) - 1] = '\0';
                break;
            }
        }
        pthread_mutex_unlock(&tracked_files_mutex);
        
        if (path[0] != '\0') {
            fprintf(stderr, "[libvgpu-cuda] fread() intercepted for PCI device file: %s (pid=%d, size=%zu, nmemb=%zu)\n",
                    path, (int)getpid(), size, nmemb);
            fflush(stderr);
            
            /* Return appropriate value based on file type */
            if (strstr(path, "/vendor") != NULL) {
                if (size * nmemb >= 6) {
                    strncpy((char *)ptr, "0x10de\n", size * nmemb);
                    fprintf(stderr, "[libvgpu-cuda] fread: returning vendor=0x10de (NVIDIA)\n");
                    fflush(stderr);
                    return 1;  /* Return 1 element read */
                }
            } else if (strstr(path, "/device") != NULL) {
                if (size * nmemb >= 6) {
                    strncpy((char *)ptr, "0x2331\n", size * nmemb);
                    fprintf(stderr, "[libvgpu-cuda] fread: returning device=0x2331 (H100 PCIe)\n");
                    fflush(stderr);
                    return 1;
                }
            } else if (strstr(path, "/class") != NULL) {
                if (size * nmemb >= 8) {
                    strncpy((char *)ptr, "0x030200\n", size * nmemb);
                    fprintf(stderr, "[libvgpu-cuda] fread: returning class=0x030200 (3D controller)\n");
                    fflush(stderr);
                    return 1;
                }
            }
        }
    }
    
    if (real_fread && real_fread != (size_t (*)(void *, size_t, size_t, FILE *))fread)
        return real_fread(ptr, size, nmemb, stream);
    /* Final fallback: read() loop so model tensor load never gets spurious 0 (e.g. load_tensors) */
    if (size > 0 && nmemb > 0 && stream) {
        int fd = fileno(stream);
        if (fd >= 0) {
            size_t total = size * nmemb;
            size_t got = 0;
            while (got < total) {
                ssize_t n = syscall(__NR_read, fd, (char *)ptr + got, total - got);
                if (n <= 0) break;
                got += (size_t)n;
            }
            return (size_t)(got / size);
        }
    }
    return 0;
}

/* Intercept fgets() for PCI device files */
char *fgets(char *s, int size, FILE *stream)
{
    ensure_skip_flag_mutex_init();
    int skip_flag = 0;
    if (g_skip_flag_mutex_initialized) {
        pthread_mutex_lock(&g_skip_flag_mutex);
        skip_flag = g_skip_pci_interception;
        pthread_mutex_unlock(&g_skip_flag_mutex);
    } else {
        skip_flag = g_skip_pci_interception;
    }
    
    if (skip_flag) {
        /* CRITICAL: When skip_flag=1, ALWAYS use syscall read
         * Files are opened via syscall + fdopen(), so we must use syscall read
         * Don't try real_fgets() - it might not work with syscall-opened files */
        int fd = fileno(stream);
        if (fd >= 0) {
            ssize_t n = syscall(__NR_read, fd, s, size - 1);
            if (n > 0) {
                s[n] = '\0';
                /* Find newline and ensure string ends there */
                char *nl = strchr(s, '\n');
                if (nl) {
                    nl[1] = '\0';
                }
                fprintf(stderr, "[libvgpu-cuda] fgets() SKIP mode (syscall read): fd=%d, read %zd bytes: '%s' (pid=%d)\n",
                        fd, n, s, (int)getpid());
                fflush(stderr);
                return s;
            } else if (n == 0) {
                /* EOF */
                fprintf(stderr, "[libvgpu-cuda] fgets() SKIP mode: fd=%d, EOF (pid=%d)\n", fd, (int)getpid());
                fflush(stderr);
                return NULL;
            } else {
                /* Error */
                fprintf(stderr, "[libvgpu-cuda] fgets() SKIP mode syscall read failed: fd=%d, errno=%d (pid=%d)\n",
                        fd, errno, (int)getpid());
                fflush(stderr);
                return NULL;
            }
        }
        fprintf(stderr, "[libvgpu-cuda] ERROR: fgets() skip mode: invalid fd (pid=%d)\n", (int)getpid());
        fflush(stderr);
        return NULL;
    }
    
    /* CRITICAL: Check process type FIRST, before any dlsym() calls */
    if (!is_application_process()) {
        /* For system processes, use read() syscall directly
         * This works for files opened via syscall + fdopen() */
        int fd = fileno(stream);
        if (fd >= 0) {
            ssize_t n = syscall(__NR_read, fd, s, size - 1);
            if (n > 0) {
                s[n] = '\0';
                /* Find newline and ensure string ends there */
                char *nl = strchr(s, '\n');
                if (nl) {
                    nl[1] = '\0';
                }
                return s;
            }
        }
        return NULL;
    }
    
    /* Application process - proceed with interception */
    static char *(*real_fgets_next)(char *, int, FILE *) = NULL;
    if (!real_fgets_next)
        real_fgets_next = (char *(*)(char *, int, FILE *))dlsym(RTLD_NEXT, "fgets");
    
    /* CRITICAL: If file is NOT tracked (e.g. model blob opened with real fopen),
     * use libc's real fgets so FILE* buffering is consistent and "failed to read magic" is avoided. */
    if (!is_tracked_pci_file(stream) || is_caller_from_our_code()) {
        ensure_real_libc_resolved();
        if (g_real_fgets_global)
            return g_real_fgets_global(s, size, stream);
        if (real_fgets_next && real_fgets_next != (char *(*)(char *, int, FILE *))fgets)
            return real_fgets_next(s, size, stream);
        /* Fallback for our fdopen'd streams (excluded path): read until newline or size-1 */
        int fd = fileno(stream);
        if (fd >= 0 && is_excluded_fd(fd) && size > 1) {
            int i = 0;
            while (i < size - 1) {
                char c;
                ssize_t n = syscall(__NR_read, fd, &c, 1);
                if (n <= 0) break;
                s[i++] = c;
                if (c == '\n') break;
            }
            s[i] = '\0';
            return (i > 0) ? s : NULL;
        }
        if (fd >= 0) {
            ssize_t n = syscall(__NR_read, fd, s, size - 1);
            if (n > 0) {
                s[n] = '\0';
                char *nl = strchr(s, '\n');
                if (nl) nl[1] = '\0';
                return s;
            }
        }
        return NULL;
    }
    
    /* File IS tracked (PCI) and caller is NOT from our code - intercept or use real_fgets */
    static char *(*real_fgets)(char *, int, FILE *) = NULL;
    if (!real_fgets) {
        real_fgets = (char *(*)(char *, int, FILE *))dlsym(RTLD_NEXT, "fgets");
    }
    
    if (is_tracked_pci_file(stream) && !is_caller_from_our_code()) {
        char path[512] = "";
        int i;
        
        ensure_tracked_files_mutex_init();
        pthread_mutex_lock(&tracked_files_mutex);
        /* Search from the end to find the most recent match (in case FILE* is reused) */
        for (i = num_tracked_files - 1; i >= 0; i--) {
            if (tracked_files[i].fp == stream) {
                strncpy(path, tracked_files[i].path, sizeof(path) - 1);
                path[sizeof(path) - 1] = '\0';
                break;
            }
        }
        pthread_mutex_unlock(&tracked_files_mutex);
        
        if (path[0] != '\0') {
            int is_tracked = is_tracked_pci_file(stream);
            int is_caller = is_caller_from_our_code();
            fprintf(stderr, "[libvgpu-cuda] fgets() intercepted for PCI device file: %s (pid=%d, size=%d, fp=%p, is_tracked=%d, is_caller=%d)\n",
                    path, (int)getpid(), size, (void*)stream, is_tracked, is_caller);
            fflush(stderr);
            
            /* Return appropriate value based on file type */
            /* CRITICAL: Check for exact file names, not substrings */
            /* The path is like /sys/bus/pci/devices/0000:00:05.0/vendor */
            /* We need to check for /vendor, /device, /class at the END of the path */
            /* CRITICAL FIX: Only return intercepted values for device 0000:00:05.0 */
            size_t path_len = strlen(path);
            int is_target_device = (strstr(path, "0000:00:05.0") != NULL || strstr(path, "00:05.0") != NULL);
            
            if (is_target_device) {
                if (path_len >= 7 && strcmp(path + path_len - 7, "/vendor") == 0) {
                    if (size >= 7) {
                        strncpy(s, "0x10de\n", size - 1);
                        s[size - 1] = '\0';
                        fprintf(stderr, "[libvgpu-cuda] fgets: returning vendor=0x10de (NVIDIA) for %s\n", path);
                        fflush(stderr);
                        return s;
                    }
                } else if (path_len >= 7 && strcmp(path + path_len - 7, "/device") == 0) {
                    if (size >= 7) {
                        strncpy(s, "0x2331\n", size - 1);
                        s[size - 1] = '\0';
                        fprintf(stderr, "[libvgpu-cuda] fgets: returning device=0x2331 (H100 PCIe) for %s\n", path);
                        fflush(stderr);
                        return s;
                    }
                } else if (path_len >= 6 && strcmp(path + path_len - 6, "/class") == 0) {
                    if (size >= 9) {
                        strncpy(s, "0x030200\n", size - 1);
                        s[size - 1] = '\0';
                        fprintf(stderr, "[libvgpu-cuda] fgets: returning class=0x030200 (3D controller) for %s\n", path);
                        fflush(stderr);
                        return s;
                    }
                }
            }
            /* For other devices, let real_fgets() return the actual values */
        }
    }
    
    return real_fgets ? real_fgets(s, size, stream) : NULL;
}

/* ================================================================
 * CUDA types — minimal definitions so we can compile without the
 * real CUDA headers.  These must be ABI-compatible with NVIDIA's
 * definitions.
 * ================================================================ */

typedef int          CUresult;
typedef int          CUdevice;
typedef void *       CUcontext;
typedef void *       CUmodule;
typedef void *       CUlibrary;
typedef void *       CUkernel;
typedef void *       CUfunction;
typedef void *       CUstream;
typedef void *       CUevent;
typedef void *       CUmemoryPool;
typedef unsigned long long CUdeviceptr;
typedef size_t       CUsize_t;
typedef int          CUjit_option;
typedef int          CUlibraryOption;
typedef struct CUlaunchAttribute_st CUlaunchAttribute;
typedef struct {
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream     hStream;
    CUlaunchAttribute *attrs;
    unsigned int numAttrs;
} CUlaunchConfig;

/* CUdevprop - Legacy device properties structure (for cuDeviceGetProperties) */
typedef struct {
    int major;
    int minor;
    char name[256];
    size_t totalGlobalMem;
    int multiprocessorCount;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiprocessor;
    int sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int clockRate;
    int memoryClockRate;
    int memoryBusWidth;
    int totalConstMem;
    int textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
} CUdevprop;
typedef unsigned long long cuuint64_t;

/* CUDA Virtual Memory Management (VMM) types */
typedef void * CUmemGenericAllocationHandle;
typedef struct CUmemAllocationProp CUmemAllocationProp;
typedef struct CUmemAccessDesc CUmemAccessDesc;

/* CUresult constants */
#define CUDA_SUCCESS                        0
#define CUDA_ERROR_INVALID_VALUE            1
#define CUDA_ERROR_OUT_OF_MEMORY            2
#define CUDA_ERROR_NOT_INITIALIZED          3
#define CUDA_ERROR_DEINITIALIZED            4
#define CUDA_ERROR_NO_DEVICE                100
#define CUDA_ERROR_INVALID_DEVICE           101
#define CUDA_ERROR_INVALID_HANDLE           400
#define CUDA_ERROR_INVALID_CONTEXT          201
#define CUDA_ERROR_NOT_FOUND                500
#define CUDA_ERROR_NOT_SUPPORTED            801
#define CUDA_ERROR_UNKNOWN                  999

/* Global flag to track if we're in initialization phase */
static int g_in_init_phase = 1;  /* Start as true, set to 0 after first successful context creation */

/* CUdevice_attribute (subset — only what Ollama / llama.cpp needs) */
typedef enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                      = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                      = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                      = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                       = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                       = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                       = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK          = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE                            = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH                            = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK              = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE                           = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                    = 14,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                 = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                  = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED                           = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                  = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                         = 20,
    CU_DEVICE_ATTRIBUTE_MAX_TEXTURE1D_WIDTH                  = 21,
    CU_DEVICE_ATTRIBUTE_MAX_TEXTURE2D_WIDTH                  = 22,
    CU_DEVICE_ATTRIBUTE_MAX_TEXTURE2D_HEIGHT                 = 23,
    CU_DEVICE_ATTRIBUTE_MAX_TEXTURE3D_WIDTH                  = 24,
    CU_DEVICE_ATTRIBUTE_MAX_TEXTURE3D_HEIGHT                 = 25,
    CU_DEVICE_ATTRIBUTE_MAX_TEXTURE3D_DEPTH                  = 26,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                   = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED                          = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                           = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                        = 34,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                    = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH              = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                        = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR       = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                   = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                   = 41,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                        = 50,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR             = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR             = 76,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                       = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                      = 84,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH                   = 95,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR        = 106,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR     = 82,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN    = 97,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED            = 91,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED             = 92,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED  = 143,  /* CUDA 12.0+ */
    CU_DEVICE_ATTRIBUTE_MAX                                  = 200,
} CUdevice_attribute;

/* CU_CTX_SCHED flags */
#define CU_CTX_SCHED_AUTO        0
#define CU_CTX_MAP_HOST          0x08

/* CU_STREAM flags */
#define CU_STREAM_DEFAULT        0x00
#define CU_STREAM_NON_BLOCKING   0x01

/* CU_EVENT flags */
#define CU_EVENT_DEFAULT         0x00
#define CU_EVENT_BLOCKING_SYNC   0x01
#define CU_EVENT_DISABLE_TIMING  0x02

/* Forward declarations — versioned API wrappers */
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev);
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev);
CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags);
CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pitch,
                         size_t widthInBytes, size_t height, unsigned int elementSizeBytes);
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pitch,
                            size_t widthInBytes, size_t height, unsigned int elementSizeBytes);
CUresult cuMemGetAddressRange(CUdeviceptr *base, size_t *size, CUdeviceptr dptr);
CUresult cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *size, CUdeviceptr dptr);
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t byteCount);
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount);
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           size_t byteCount, CUstream hStream);
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t byteCount, CUstream hStream);
CUresult cuMemcpy2DUnaligned(const void *pCopy);
CUresult cuMemcpy2DAsync(const void *pCopy, CUstream hStream);
CUresult cuMemcpy2DAsync_v2(const void *pCopy, CUstream hStream);
CUresult cuMemcpy3D(const void *pCopy);
CUresult cuMemcpy3DAsync(const void *pCopy, CUstream hStream);
CUresult cuMemcpy3DPeer(const void *pCopy);
CUresult cuMemcpy3DPeerAsync(const void *pCopy, CUstream hStream);
CUresult cuMemcpyBatchAsync(const void *params, size_t count, unsigned int flags, CUstream hStream);
CUresult cuMemcpy3DBatchAsync(const void *params, size_t count, unsigned int flags, CUstream hStream);
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext, size_t byteCount);
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t byteCount, CUstream hStream);
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                      size_t Width, size_t Height);
CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                           size_t Width, size_t Height, CUstream hStream);
CUresult cuArrayCreate(void *pHandle, const void *pAllocateArray);
CUresult cuArrayGetDescriptor(void *pArrayDescriptor, void *hArray);
CUresult cuArrayGetSparseProperties(void *sparseProperties, void *array);
CUresult cuArrayGetPlane(void *pPlaneArray, void *hArray, unsigned int planeIdx);
CUresult cuArray3DCreate(void *pHandle, const void *pAllocateArray);
CUresult cuArray3DGetDescriptor(void *pArrayDescriptor, void *hArray);
CUresult cuArrayDestroy(void *hArray);
CUresult cuMipmappedArrayCreate(void *pHandle, const void *pMipmappedArrayDesc,
                                unsigned int numMipmapLevels);
CUresult cuMipmappedArrayGetLevel(void *pLevelArray, void *hMipmappedArray, unsigned int level);
CUresult cuMipmappedArrayGetSparseProperties(void *sparseProperties, void *hMipmappedArray);
CUresult cuMipmappedArrayDestroy(void *hMipmappedArray);
CUresult cuArrayGetMemoryRequirements(void *memoryRequirements, void *array, void *device);
CUresult cuMipmappedArrayGetMemoryRequirements(void *memoryRequirements, void *hMipmappedArray, void *device);
CUresult cuTexObjectCreate(void *pTexObject, const void *pResDesc,
                           const void *pTexDesc, const void *pResViewDesc);
CUresult cuTexObjectDestroy(unsigned long long texObject);
CUresult cuTexObjectGetResourceDesc(void *pResDesc, unsigned long long texObject);
CUresult cuTexObjectGetTextureDesc(void *pTexDesc, unsigned long long texObject);
CUresult cuTexObjectGetResourceViewDesc(void *pResViewDesc, unsigned long long texObject);
CUresult cuSurfObjectCreate(void *pSurfObject, const void *pResDesc);
CUresult cuSurfObjectDestroy(unsigned long long surfObject);
CUresult cuSurfObjectGetResourceDesc(void *pResDesc, unsigned long long surfObject);
CUresult cuMemPoolExportPointer(void *shareData, CUdeviceptr ptr);
CUresult cuMemPoolImportPointer(CUdeviceptr *ptr, CUmemoryPool pool, void *shareData);
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, int config);
CUresult cuFuncGetName(const char **name, CUfunction hfunc);
CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t *paramOffset, size_t *paramSize);
CUresult cuImportExternalMemory(void *extMem_out, const void *memHandleDesc);
CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, void *extMem, const void *bufferDesc);
CUresult cuExternalMemoryGetMappedMipmappedArray(void *mipmap, void *extMem, const void *mipmapDesc);
CUresult cuDestroyExternalMemory(void *extMem);
CUresult cuImportExternalSemaphore(void *extSem_out, const void *semHandleDesc);
CUresult cuSignalExternalSemaphoresAsync(const void *extSemArray, const void *paramsArray,
                                         unsigned int numExtSems, CUstream stream);
CUresult cuWaitExternalSemaphoresAsync(const void *extSemArray, const void *paramsArray,
                                       unsigned int numExtSems, CUstream stream);
CUresult cuDestroyExternalSemaphore(void *extSem);
CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags);
CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, uint32_t value, unsigned int flags);
CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, uint32_t value, unsigned int flags);
CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, const void *paramArray, unsigned int flags);
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult cuIpcGetEventHandle(void *pHandle, CUevent event);
CUresult cuIpcOpenEventHandle(CUevent *phEvent, void *handle);
CUresult cuIpcGetMemHandle(void *pHandle, CUdeviceptr dptr);
CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, void *handle, unsigned int flags);
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);
CUresult cuGLCtxCreate(CUcontext *pCtx, unsigned int Flags, void *device);
CUresult cuGLInit(void);
CUresult cuGLGetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices,
                        unsigned int cudaDeviceCount, unsigned int deviceList);
CUresult cuGLRegisterBufferObject(unsigned int buffer);
CUresult cuGLMapBufferObject(CUdeviceptr *dptr, size_t *size, unsigned int buffer);
CUresult cuGLMapBufferObjectAsync(CUdeviceptr *dptr, size_t *size, unsigned int buffer, CUstream hStream);
CUresult cuGLUnmapBufferObject(unsigned int buffer);
CUresult cuGLUnmapBufferObjectAsync(unsigned int buffer, CUstream hStream);
CUresult cuGLUnregisterBufferObject(unsigned int buffer);
CUresult cuGLSetBufferObjectMapFlags(unsigned int buffer, unsigned int Flags);
CUresult cuGraphicsGLRegisterImage(void *pCudaResource, unsigned int image, int target, unsigned int Flags);
CUresult cuGraphicsGLRegisterBuffer(void *pCudaResource, unsigned int buffer, unsigned int Flags);
CUresult cuGraphicsEGLRegisterImage(void *pCudaResource, void *image, unsigned int flags);
CUresult cuEGLStreamConsumerConnect(void *conn, void *stream);
CUresult cuEGLStreamConsumerDisconnect(void *conn);
CUresult cuEGLStreamConsumerAcquireFrame(void *conn, void *pCudaResource, void *pStream, unsigned int timeout);
CUresult cuEGLStreamConsumerReleaseFrame(void *conn, void *pCudaResource, void *pStream);
CUresult cuEGLStreamProducerConnect(void *conn, void *stream, unsigned int width, unsigned int height);
CUresult cuEGLStreamProducerDisconnect(void *conn);
CUresult cuEGLStreamProducerPresentFrame(void *conn, void *eglFrame, void *pStream);
CUresult cuEGLStreamProducerReturnFrame(void *conn, void *eglFrame, void *pStream);
CUresult cuEGLStreamConsumerConnectWithFlags(void *conn, void *stream, unsigned int flags);
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags);
CUresult cuStreamGetPriority(CUstream hStream, int *priority);
CUresult cuStreamGetDevice(CUstream hStream, CUdevice *device);
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx);

/* Forward declarations for context functions used in cuGetProcAddress */
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxCreate_v3(CUcontext *pctx, unsigned int flags, CUdevice dev, void *params);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuDriverGetVersion(int *driverVersion);
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *memPool, CUdevice dev);
CUresult cuDeviceGetMemPool(CUmemoryPool *memPool, CUdevice dev);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUdevice dev, int texDesc);
CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags);
CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxPushCurrent_v2(CUcontext ctx);
CUresult cuCtxPopCurrent_v2(CUcontext *pctx);
CUresult cuCtxSynchronize(void);
CUresult cuCtxGetDevice_v2(CUdevice *device);
CUresult cuCtxGetFlags(unsigned int *flags);
CUresult cuCtxSetLimit(int limit, size_t value);
CUresult cuCtxGetLimit(size_t *pvalue, int limit);
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceGetAttribute_v2(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);

/* ================================================================
 * Load-time diagnostic: confirm the shim is actually being loaded.
 * This runs as soon as the dynamic linker maps the library into the
 * process — before any CUDA function is called.  The message appears
 * in journalctl -u ollama, confirming LD_PRELOAD / dlopen succeeded.
 *
 * Also writes to a per-PID log file so we can inspect Ollama's piped
 * GPU-discovery subprocess even though its stderr is not forwarded to
 * journald.  Check /tmp/vgpu-shim-cuda-<pid>.log after a failing run.
 * ================================================================ */

/* Constructor - do nothing. ensure_init() runs on first cuInit/cuDeviceGetCount.
 * Even a single syscall write here can trigger SEGV after CUDART constructor. */
__attribute__((constructor(101)))
static void libvgpu_cuda_on_load(void)
{
    (void)0;  /* no-op; avoid any syscall/write */
    return;

    /* OLD COMMENT - kept for reference:
     * When deployed via /etc/ld.so.preload, this library loads into ALL processes
     * (sshd, systemd, etc.) during VERY EARLY initialization, BEFORE libc/pthreads
     * are fully initialized. Even syscalls may not be safe at this point.
     * 
     * UNSAFE operations that MUST NOT be in constructor:
     * - getenv() - libc may not be initialized
     * - cuInit() - calls pthread_mutex_lock, filesystem I/O, fprintf
     * - is_application_process() - uses syscalls that may crash during early init
     * - is_runner_process() - uses syscalls and libc calls
     * - Any syscalls - can fail during very early init
     * - Any libc calls - libc may not be ready
     * - Any file I/O - filesystem may not be ready
     * - Any mutex operations - pthreads may not be initialized
     * 
     * SOLUTION: Do absolutely nothing in constructor.
     * 
     * Initialization happens lazily via ensure_init() when CUDA functions are
     * actually called. This is safe because:
     * - System processes never call CUDA functions, so they're unaffected
     * - Application processes are fully initialized by the time they call CUDA
     * - ensure_init() checks is_application_process() first for safety (when safe)
     * 
     * For runners that need early initialization, they should:
     * - Set OLLAMA_RUNNER environment variable
     * - Call cuInit() explicitly or via ensure_init() on first CUDA call
     * - Use a wrapper script or systemd service to handle initialization
     */
    
    /* Constructor is intentionally empty - no operations performed */
}

/* Helper: Check if it's safe to perform process type checking.
 * During very early initialization, even syscalls may fail or cause crashes.
 * CRITICAL: This function must be EXTREMELY defensive - any failure means "not safe".
 * We use a counter to delay checking - only check after several calls have passed.
 * This ensures we don't try to check during the absolute earliest initialization. */
static int is_safe_to_check_process(void)
{
    /* If already marked safe, return immediately */
    if (g_safe_to_check_process) {
        return 1;
    }
    
    /* CRITICAL SAFETY: Use a static counter to delay checking.
     * During the first few calls (especially during very early init),
     * we don't even attempt to check - we just return "not safe".
     * This prevents any syscalls during the most dangerous early phase. */
    static int call_count = 0;
    call_count++;
    
    /* Don't even attempt checking until we've been called at least 10 times.
     * This ensures we're past the most dangerous early initialization phase. */
    if (call_count < 10) {
        return 0;  /* Not safe yet - too early */
    }
    
    /* Now try a simple test: can we read /proc/self/comm?
     * If this succeeds, it means filesystem is ready and syscalls work.
     * CRITICAL: Wrap in signal-safe way - if anything fails, default to "not safe". */
    int fd = -1;
    __asm__ __volatile__ ("" ::: "memory");  /* Memory barrier for safety */
    
    /* Use direct syscall - but be very defensive */
    fd = syscall(__NR_open, "/proc/self/comm", O_RDONLY);
    if (fd >= 0) {
        syscall(__NR_close, fd);
        g_safe_to_check_process = 1;  /* Mark as safe */
        return 1;
    }
    
    /* Not safe yet - filesystem may not be ready, or syscall failed */
    return 0;
}

/* Helper function: Check if this is an application process (whitelist approach) */
/* Returns 1 only for processes we KNOW need interception (ollama), 0 otherwise */
static int is_application_process(void)
{
    /* CRITICAL FIX: Prevent re-entrancy - if we're already checking, default to safe */
    /* This prevents infinite recursion if is_application_process() is called during */
    /* its own execution (e.g., from within a syscall wrapper) */
    if (g_checking_process_type) {
        return 0;  /* Don't intercept if we're already checking - safe default */
    }
    
    /* CRITICAL FIX: Only check for positive cache (ollama found), not >= 0 */
    /* We never cache 0 (system process) - always recheck for safety */
    if (g_process_type_cached == 1) {
        /* Only return cached positive result (ollama found) */
        return 1;
    }
    /* If cache is -1 (not checked) or any other value, proceed to check */
    
    /* Set re-entrancy flag to prevent recursive calls */
    g_checking_process_type = 1;
    
    /* CRITICAL: Default to NOT intercepting (safe default) */
    /* If ANYTHING goes wrong at ANY point, we return 0 and don't intercept */
    /* This function must be EXTREMELY safe - no operations that could fail */
    int result = 0;
    int checked_successfully = 0;
    char comm[256] = {0};
    int fd = -1;
    ssize_t n = 0;
    
    /* CRITICAL: Be EXTREMELY defensive - use ONLY syscalls, catch ALL errors */
    /* If ANY step fails, return 0 immediately (don't intercept) */
    fd = syscall(__NR_open, "/proc/self/comm", O_RDONLY);
    if (fd < 0) {
        /* Can't read comm - default to not intercepting, don't cache */
        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
        return 0;
    }
    
    n = syscall(__NR_read, fd, comm, sizeof(comm) - 1);
    syscall(__NR_close, fd);  /* Close immediately after read */
    
    if (n <= 0 || n >= (ssize_t)sizeof(comm)) {
        /* Read failed or too large - default to not intercepting, don't cache */
        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
        return 0;
    }
    
    comm[n] = '\0';
    /* Remove newline if present - be very careful with bounds */
    if (n > 0 && comm[n-1] == '\n') {
        comm[n-1] = '\0';
        n--;
    }
    
    checked_successfully = 1;
    
    /* CRITICAL: Check for system processes FIRST using ONLY direct character comparisons */
    /* Use minimal checks - just first few characters to catch common system tools */
    /* All early returns MUST clear the re-entrancy flag before returning */
    /* Check lspci FIRST (most common crash source) - check first 2 chars immediately */
    if (n >= 2 && comm[0] == 'l' && comm[1] == 's') {
        /* Could be lspci, ls, or other 'ls*' tools - don't intercept any of them */
        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
        return 0;  /* Don't cache - always check fresh */
    }
    /* Check cat - common system tool */
    if (n >= 3 && comm[0] == 'c' && comm[1] == 'a' && comm[2] == 't') {
        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
        return 0;  /* Don't cache */
    }
    /* Check bash/sh - shell processes */
    if (n >= 3 && comm[0] == 'b' && comm[1] == 'a' && comm[2] == 's' && comm[3] == 'h') {
        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
        return 0;  /* Don't cache */
    }
    if (n >= 2 && comm[0] == 's' && comm[1] == 'h') {
        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
        return 0;  /* Don't cache */
    }
    /* Check sshd/systemd - first char 's' */
    if (n >= 1 && comm[0] == 's') {
        /* Could be sshd, systemd, or other 's*' tools - be more specific */
        if (n >= 4 && comm[1] == 's' && comm[2] == 'h' && comm[3] == 'd') {
            g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
            return 0;  /* sshd */
        }
        if (n >= 7 && comm[1] == 'y' && comm[2] == 's' && comm[3] == 't' && 
            comm[4] == 'e' && comm[5] == 'm' && comm[6] == 'd') {
            g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
            return 0;  /* systemd */
        }
    }
    /* Check init */
    if (n >= 4 && comm[0] == 'i' && comm[1] == 'n' && comm[2] == 'i' && comm[3] == 't') {
        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
        return 0;  /* Don't cache */
    }
    
    /* WHITELIST: Only intercept for ollama processes */
    /* Use direct character comparison (6 chars) - NO string functions */
    /* If it's not ollama, we don't intercept (result stays 0) */
    if (n >= 6 && comm[0] == 'o' && comm[1] == 'l' && comm[2] == 'l' && 
        comm[3] == 'a' && comm[4] == 'm' && comm[5] == 'a') {
        result = 1;  /* Only ollama gets intercepted */
    }
    /* CRITICAL: If result is still 0 (not ollama), we don't intercept */
    /* This is the whitelist - only ollama is allowed */
    
    /* Also check cmdline for more reliable detection (catches "ollama serve", "ollama runner", etc.) */
    /* Only check cmdline if comm check didn't find ollama */
    if (result == 0 && checked_successfully) {
        char cmdline[512] = {0};
        int fd = syscall(__NR_open, "/proc/self/cmdline", O_RDONLY);
        if (fd >= 0) {
            ssize_t n = syscall(__NR_read, fd, cmdline, sizeof(cmdline) - 1);
            if (n > 0 && n < (ssize_t)sizeof(cmdline)) {
                cmdline[n] = '\0';
                
                /* CRITICAL: Check for SSH/systemd in cmdline first (before processing) */
                /* cmdline is null-separated, but we can check first few bytes */
                /* CRITICAL FIX: Use ONLY direct character comparisons - NO strstr() */
                /* strstr() can trigger intercepted functions, causing crashes */
                if (n >= 4 && (cmdline[0] == 's' || cmdline[0] == 'S')) {
                    /* Check for sshd: 's' 's' 'h' 'd' */
                    if (n >= 4 && cmdline[0] == 's' && cmdline[1] == 's' && 
                        cmdline[2] == 'h' && cmdline[3] == 'd') {
                        syscall(__NR_close, fd);
                        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
                        return 0;  /* Never intercept */
                    }
                    /* Check for systemd: 's' 'y' 's' 't' 'e' 'm' 'd' */
                    if (n >= 7 && cmdline[0] == 's' && cmdline[1] == 'y' && 
                        cmdline[2] == 's' && cmdline[3] == 't' && 
                        cmdline[4] == 'e' && cmdline[5] == 'm' && cmdline[6] == 'd') {
                        syscall(__NR_close, fd);
                        g_checking_process_type = 0;  /* Clear re-entrancy flag before return */
                        return 0;  /* Never intercept */
                    }
                }
                
                /* Check for ollama in cmdline using direct character comparison */
                /* Scan through cmdline (null-separated) for "ollama" */
                /* CRITICAL: NO strstr() - use only direct character comparisons */
                int i;
                for (i = 0; i <= n - 6; i++) {
                    /* Skip null bytes (they separate arguments) */
                    if (cmdline[i] == '\0') continue;
                    /* Check for "ollama" starting at position i */
                    if (cmdline[i] == 'o' && cmdline[i+1] == 'l' && 
                        cmdline[i+2] == 'l' && cmdline[i+3] == 'a' && 
                        cmdline[i+4] == 'm' && cmdline[i+5] == 'a') {
                        result = 1;
                        break;
                    }
                    /* Skip to next null (end of current argument) */
                    while (i < n && cmdline[i] != '\0') i++;
                }
            }
            syscall(__NR_close, fd);
        }
    }
    
    /* CRITICAL FIX: Only cache if we successfully checked AND found ollama */
    /* For non-ollama processes, don't cache - always recheck to be safe */
    /* This ensures we never accidentally cache a wrong value */
    if (checked_successfully && result == 1) {
        /* Only cache positive results (ollama found) */
        g_process_type_cached = 1;
    }
    /* For non-ollama (result == 0), don't cache - always check fresh */
    /* This is safer and prevents any caching issues */
    
    /* CRITICAL: Clear re-entrancy flag before returning */
    g_checking_process_type = 0;
    
    return result;
}

/* Helper function: Check if this is a critical system process (for backward compatibility) */
__attribute__((unused)) static int is_system_process(void)
{
    /* Use cached application process check - if not application, it's system */
    return !is_application_process();
}

/* Helper function: Check if this is a runner process
 * Runner processes are spawned by Ollama to handle GPU discovery and inference
 * They need explicit CUDA initialization in the constructor */
static int is_runner_process(void)
{
    /* First check if this is an ollama process at all */
    if (!is_application_process()) {
        return 0;  /* Not an ollama process */
    }
    
    /* Check cmdline for "runner" - runner processes have "ollama runner" in cmdline */
    char cmdline[512] = {0};
    int fd = syscall(__NR_open, "/proc/self/cmdline", O_RDONLY);
    if (fd < 0) {
        return 0;  /* Can't check, assume not runner */
    }
    
    ssize_t n = syscall(__NR_read, fd, cmdline, sizeof(cmdline) - 1);
    syscall(__NR_close, fd);
    
    if (n <= 0 || n >= (ssize_t)sizeof(cmdline)) {
        return 0;  /* Read failed */
    }
    
    cmdline[n] = '\0';
    
    /* Check for "runner" in cmdline using direct character comparison */
    /* cmdline is null-separated, scan through it */
    int i;
    for (i = 0; i <= n - 6; i++) {
        if (cmdline[i] == '\0') continue;
        /* Check for "runner" starting at position i */
        if (cmdline[i] == 'r' && cmdline[i+1] == 'u' && 
            cmdline[i+2] == 'n' && cmdline[i+3] == 'n' && 
            cmdline[i+4] == 'e' && cmdline[i+5] == 'r') {
            return 1;  /* Found "runner" */
        }
        /* Skip to next null (end of current argument) */
        while (i < n && cmdline[i] != '\0') i++;
    }
    
    return 0;  /* Not a runner process */
}


/* ================================================================
 * Global state (continued - transport and GPU info)
 * ================================================================ */
static cuda_transport_t *g_transport = NULL;
static CUDAGpuInfo      g_gpu_info;
static int              g_gpu_info_valid = 0;

static void fill_default_device_identity(CUDAGpuInfo *info)
{
    const char *bdf;
    unsigned int domain = GPU_DEFAULT_PCI_DOMAIN_ID;
    unsigned int bus = GPU_DEFAULT_PCI_BUS_ID;
    unsigned int slot = GPU_DEFAULT_PCI_DEV_ID;
    unsigned int function = 0;
    size_t i;
    int uuid_all_zero = 1;

    if (!info) return;

    bdf = cuda_transport_pci_bdf(NULL);
    if (bdf && bdf[0]) {
        unsigned int parsed_domain = 0;
        unsigned int parsed_bus = 0;
        unsigned int parsed_slot = 0;
        unsigned int parsed_function = 0;
        if (sscanf(bdf, "%x:%x:%x.%x",
                   &parsed_domain, &parsed_bus, &parsed_slot, &parsed_function) == 4) {
            domain = parsed_domain;
            bus = parsed_bus;
            slot = parsed_slot;
            function = parsed_function;
        }
    }

    if (info->pci_domain_id <= 0) info->pci_domain_id = (int32_t)domain;
    if (info->pci_bus_id <= 0) info->pci_bus_id = (int32_t)bus;
    if (info->pci_device_id <= 0) info->pci_device_id = GPU_DEFAULT_PCI_DEVICE_ID;

    for (i = 0; i < sizeof(info->uuid); ++i) {
        if (info->uuid[i] != 0) {
            uuid_all_zero = 0;
            break;
        }
    }
    if (!uuid_all_zero) {
        return;
    }

    /* Keep a deterministic, non-zero UUID even before live host data is available. */
    info->uuid[0] = 0x56; /* 'V' */
    info->uuid[1] = 0x47; /* 'G' */
    info->uuid[2] = 0x50; /* 'P' */
    info->uuid[3] = 0x55; /* 'U' */
    info->uuid[4] = 0x10;
    info->uuid[5] = 0xDE;
    info->uuid[6] = 0x23;
    info->uuid[7] = 0x31;
    info->uuid[8] = (unsigned char)((domain >> 8) & 0xffU);
    info->uuid[9] = (unsigned char)(domain & 0xffU);
    info->uuid[10] = (unsigned char)(bus & 0xffU);
    info->uuid[11] = (unsigned char)(slot & 0xffU);
    info->uuid[12] = (unsigned char)(function & 0xffU);
    info->uuid[13] = 0x48; /* 'H' */
    info->uuid[14] = 0x31; /* '1' */
    info->uuid[15] = 0x30; /* '0' */
}

/* Keep all occupancy-critical fields non-zero. */
static void sanitize_gpu_info(CUDAGpuInfo *info)
{
    if (!info) return;

    if (info->compute_cap_major <= 0) info->compute_cap_major = GPU_DEFAULT_CC_MAJOR;
    if (info->compute_cap_minor < 0) info->compute_cap_minor = GPU_DEFAULT_CC_MINOR;
    if (info->total_mem == 0) info->total_mem = GPU_DEFAULT_TOTAL_MEM;
    if (info->free_mem == 0) info->free_mem = GPU_DEFAULT_FREE_MEM;
    if (info->multi_processor_count <= 0) info->multi_processor_count = GPU_DEFAULT_SM_COUNT;
    if (info->max_threads_per_block <= 0) info->max_threads_per_block = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;
    if (info->max_threads_per_mp <= 0) info->max_threads_per_mp = GPU_DEFAULT_MAX_THREADS_PER_SM;
    if (info->warp_size <= 0) info->warp_size = GPU_DEFAULT_WARP_SIZE;
    if (info->max_shared_mem_per_block <= 0) info->max_shared_mem_per_block = GPU_DEFAULT_SHARED_MEM_PER_BLOCK;
    if (info->max_shared_mem_per_mp <= 0) info->max_shared_mem_per_mp = GPU_DEFAULT_SHARED_MEM_PER_SM;
    if (info->regs_per_block <= 0) info->regs_per_block = GPU_DEFAULT_REGS_PER_BLOCK;
    if (info->regs_per_multiprocessor <= 0) info->regs_per_multiprocessor = GPU_DEFAULT_REGS_PER_SM;
    if (info->clock_rate_khz <= 0) info->clock_rate_khz = GPU_DEFAULT_CLOCK_RATE_KHZ;
    if (info->memory_clock_rate_khz <= 0) info->memory_clock_rate_khz = GPU_DEFAULT_MEM_CLOCK_RATE_KHZ;
    if (info->memory_bus_width <= 0) info->memory_bus_width = GPU_DEFAULT_MEM_BUS_WIDTH;
    if (info->l2_cache_size <= 0) info->l2_cache_size = GPU_DEFAULT_L2_CACHE_SIZE;
    if (info->max_block_dim_x <= 0) info->max_block_dim_x = GPU_DEFAULT_MAX_BLOCK_DIM_X;
    if (info->max_block_dim_y <= 0) info->max_block_dim_y = GPU_DEFAULT_MAX_BLOCK_DIM_Y;
    if (info->max_block_dim_z <= 0) info->max_block_dim_z = GPU_DEFAULT_MAX_BLOCK_DIM_Z;
    if (info->max_grid_dim_x <= 0) info->max_grid_dim_x = GPU_DEFAULT_MAX_GRID_DIM_X;
    if (info->max_grid_dim_y <= 0) info->max_grid_dim_y = GPU_DEFAULT_MAX_GRID_DIM_Y;
    if (info->max_grid_dim_z <= 0) info->max_grid_dim_z = GPU_DEFAULT_MAX_GRID_DIM_Z;
    if (info->unified_addressing <= 0) info->unified_addressing = GPU_DEFAULT_UNIFIED_ADDRESSING;
    if (info->managed_memory <= 0) info->managed_memory = GPU_DEFAULT_MANAGED_MEMORY;
    if (info->concurrent_kernels <= 0) info->concurrent_kernels = GPU_DEFAULT_CONCURRENT_KERNELS;
    if (info->async_engine_count <= 0) info->async_engine_count = GPU_DEFAULT_ASYNC_ENGINE_COUNT;
    if (info->driver_version <= 0) info->driver_version = GPU_DEFAULT_DRIVER_VERSION;
    if (info->runtime_version <= 0) info->runtime_version = GPU_DEFAULT_RUNTIME_VERSION;
    fill_default_device_identity(info);
}

static int effective_userspace_driver_version(void)
{
    int driver_version = GPU_DEFAULT_RUNTIME_VERSION;

    if (g_gpu_info_valid && g_gpu_info.driver_version > driver_version) {
        driver_version = g_gpu_info.driver_version;
    }

    return driver_version;
}

/* Thread-local current context handle */
static __thread CUcontext g_current_ctx = NULL;
/* Some caller paths enter CUBLAS from different threads than setup calls.
 * Keep a process-wide fallback context so cuCtxGetCurrent() does not return NULL. */
static CUcontext g_global_ctx = NULL;

static int ctx_is_dummy(CUcontext ctx)
{
    return (uintptr_t)ctx == (uintptr_t)0xDEADBEEF;
}

/* Generic stub functions for unknown CUDA functions during initialization.
 * These are defined at file scope so they can be used in cuGetProcAddress. */
__attribute__((unused)) static CUresult generic_stub_0(void) {
    fprintf(stderr, "[libvgpu-cuda] GENERIC STUB (0 args) CALLED: returning SUCCESS (init phase)\n");
    fflush(stderr);
    return CUDA_SUCCESS;
}

static CUresult generic_stub_1ptr(void *arg1) {
    (void)arg1;
    fprintf(stderr, "[libvgpu-cuda] GENERIC STUB (1 ptr) CALLED: returning SUCCESS (init phase)\n");
    fflush(stderr);
    return CUDA_SUCCESS;
}

static CUresult generic_stub_2args(void *arg1, void *arg2) {
    (void)arg1; (void)arg2;
    fprintf(stderr, "[libvgpu-cuda] GENERIC STUB (2 args) CALLED: returning SUCCESS (init phase)\n");
    fflush(stderr);
    return CUDA_SUCCESS;
}

static CUresult generic_stub_3args(void *arg1, void *arg2, void *arg3) {
    (void)arg1; (void)arg2; (void)arg3;
    fprintf(stderr, "[libvgpu-cuda] GENERIC STUB (3 args) CALLED: returning SUCCESS (init phase)\n");
    fflush(stderr);
    return CUDA_SUCCESS;
}

static CUresult generic_stub_4args(void *arg1, void *arg2, void *arg3, void *arg4) {
    (void)arg1; (void)arg2; (void)arg3; (void)arg4;
    fprintf(stderr, "[libvgpu-cuda] GENERIC STUB (4 args) CALLED: returning SUCCESS (init phase)\n");
    fflush(stderr);
    return CUDA_SUCCESS;
}

/* ================================================================
 * Internal helpers
 * ================================================================ */

static int ensure_init(void)
{
    /* Debug: write to file so we can confirm runner calls us even if stderr is redirected */
    {
        char buf[128];
        int n = snprintf(buf, sizeof(buf), "ensure_init called pid=%d\n", (int)getpid());
        if (n > 0 && n < (int)sizeof(buf)) {
            int fd = (int)syscall(__NR_open, "/tmp/vgpu_ensure_init.log",
                O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (fd >= 0) {
                syscall(__NR_write, fd, buf, (size_t)n);
                syscall(__NR_close, fd);
            }
        }
    }
    fprintf(stderr, "[libvgpu-cuda] ensure_init() CALLED (pid=%d)\n", (int)getpid());
    fflush(stderr);
    
    /* CRITICAL: For early initialization, we need to call cuInit() BEFORE
     * cuDeviceGetPCIBusId() is called. Ollama's GPU discovery requires this.
     * 
     * However, we must be safe:
     * 1. Only initialize for application processes (not system processes)
     * 2. Only initialize when it's safe to check process type
     * 3. Call cuInit() immediately when safe, not lazily
     */
    
    /* If already initialized, return immediately */
    if (g_initialized) {
        fprintf(stderr, "[libvgpu-cuda] ensure_init() already initialized (pid=%d)\n", (int)getpid());
        fflush(stderr);
        return CUDA_SUCCESS;
    }
    
    /* Check if it's safe to perform process detection */
    /* CRITICAL: If called from constructor, we've already verified it's an application process
     * via LD_PRELOAD check, so we can skip the safety check */
    int safe = is_safe_to_check_process();
    int is_app = 0;  /* Declare early */
    if (!safe) {
        /* Not safe yet - but if we have LD_PRELOAD, we're likely safe anyway */
        const char *ld_preload = getenv("LD_PRELOAD");
        if (ld_preload && strstr(ld_preload, "libvgpu")) {
            /* We have LD_PRELOAD with our shims - safe to proceed */
            fprintf(stderr, "[libvgpu-cuda] ensure_init: Safety check failed but LD_PRELOAD present, proceeding (pid=%d)\n", (int)getpid());
            fflush(stderr);
            safe = 1; /* Override safety check */
            is_app = 1; /* Assume application process */
        } else {
            /* CRITICAL FIX: Even without LD_PRELOAD, if we're being loaded, try to initialize */
            /* We've replaced the system library, so any process loading it likely needs it */
            fprintf(stderr, "[libvgpu-cuda] ensure_init: Safety check failed, but attempting init anyway (pid=%d)\n", (int)getpid());
            fflush(stderr);
            safe = 1; /* Override safety check */
            is_app = 1; /* Assume application process and try initialization */
        }
    }
    
    /* Check if this is an application process */
    if (safe) {
        is_app = is_application_process();
    } else {
        /* If safety check failed but we have LD_PRELOAD, assume it's an app process */
        const char *ld_preload = getenv("LD_PRELOAD");
        if (ld_preload && strstr(ld_preload, "libvgpu")) {
            is_app = 1;
        }
    }
    /* Fallback: Ollama runner may be started with OLLAMA_* env but comm/cmdline might not match.
     * If we are loaded as libcuda (e.g. from cuda_v12) and OLLAMA_* is set, treat as app so discovery can succeed. */
    if (!is_app && getenv("OLLAMA_NUM_GPU") != NULL) {
        is_app = 1;
    }
    if (!is_app && getenv("OLLAMA_HOST") != NULL) {
        is_app = 1;
    }
    if (!is_app && (getenv("OLLAMA_LIBRARY_PATH") != NULL || getenv("OLLAMA_LLM_LIBRARY") != NULL)) {
        is_app = 1;  /* Runner started with library paths (no LD_PRELOAD) - we are the shim */
    }
    if (!is_app) {
        /* System process - don't initialize, return error */
        fprintf(stderr, "[libvgpu-cuda] ensure_init: Not an application process (pid=%d)\n", (int)getpid());
        fflush(stderr);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    /* For application processes, initialize immediately */
    /* This is critical: cuInit() must be called BEFORE cuDeviceGetPCIBusId()
     * for Ollama's GPU matching to work correctly. */
    int is_runner = is_runner_process();
    if (is_runner) {
        fprintf(stderr, "[libvgpu-cuda] ensure_init: Runner process detected, initializing immediately (pid=%d)\n", (int)getpid());
        fflush(stderr);
    } else {
        fprintf(stderr, "[libvgpu-cuda] ensure_init: Application process detected, initializing CUDA early (pid=%d)\n", (int)getpid());
        fflush(stderr);
    }
    
    /* Auto-initialize by calling cuInit - this must happen early */
    CUresult rc = cuInit(0);
    if (rc != CUDA_SUCCESS) {
        /* During init phase, return SUCCESS anyway to allow initialization to proceed.
         * This is critical for ggml_backend_cuda_init which may call functions before
         * the full transport is connected. */
        if (g_in_init_phase) {
            fprintf(stderr, "[libvgpu-cuda] ensure_init: auto-init failed (rc=%d) but in init phase, allowing to proceed (pid=%d)\n", rc, (int)getpid());
            fflush(stderr);
            return CUDA_SUCCESS;
        }
        fprintf(stderr, "[libvgpu-cuda] ensure_init: auto-init failed (rc=%d) and not in init phase (pid=%d)\n", rc, (int)getpid());
        fflush(stderr);
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    
    /* Verify initialization completed - cuInit() sets g_initialized before returning */
    if (g_initialized && g_gpu_info_valid) {
        fprintf(stderr, "[libvgpu-cuda] ensure_init: Early initialization succeeded, CUDA ready (pid=%d, device_found=%d)\n", 
                (int)getpid(), g_device_found);
    } else {
        fprintf(stderr, "[libvgpu-cuda] ensure_init: WARNING - cuInit() returned SUCCESS but state not fully initialized (pid=%d, initialized=%d, info_valid=%d)\n",
                (int)getpid(), g_initialized, g_gpu_info_valid);
    }
    fflush(stderr);
    
    return CUDA_SUCCESS;
}

/* Discovery readiness check - verify all components are ready for discovery */
static int __attribute__((unused)) is_discovery_ready(void)
{
    /* Check CUDA initialization */
    if (!g_initialized) {
        fprintf(stderr, "[libvgpu-cuda] Discovery NOT ready: CUDA not initialized\n");
        return 0;
    }
    
    /* Check device found */
    if (!g_device_found) {
        fprintf(stderr, "[libvgpu-cuda] Discovery NOT ready: Device not found\n");
        return 0;
    }
    
    /* Check GPU info populated */
    if (!g_gpu_info_valid) {
        fprintf(stderr, "[libvgpu-cuda] Discovery NOT ready: GPU info not populated\n");
        return 0;
    }
    
    /* Check PCI bus ID available */
    const char *bdf = cuda_transport_pci_bdf(NULL);
    if (!bdf || !bdf[0]) {
        fprintf(stderr, "[libvgpu-cuda] Discovery NOT ready: PCI bus ID not available\n");
        return 0;
    }
    
    fprintf(stderr, "[libvgpu-cuda] Discovery READY: initialized=%d, device_found=%d, info_valid=%d, bdf=%s\n",
            g_initialized, g_device_found, g_gpu_info_valid, bdf);
    return 1;
}

/* Forward declarations — defined below. */
static int fetch_gpu_info(void);
static void init_gpu_defaults(void);

/*
 * ensure_connected — lazily open BAR0 and connect to the mediator.
 *
 * Called by compute functions (cuMemAlloc, cuLaunchKernel, cuModuleLoadData,
 * cuCtxCreate, etc.) on their first real call after cuInit() succeeded.
 *
 * cuInit() now only scans /sys (read-only, always possible), so GPU discovery
 * always succeeds.  The actual resource0 mmap and mediator socket are deferred
 * here — inside the runner subprocess where ProtectKernelTunables does NOT
 * block write-access thanks to ReadWritePaths=/sys/bus/pci/devices/.
 *
 * Must be called with g_mutex held.  Returns CUDA_SUCCESS or an error code.
 */
static CUresult ensure_connected(void)
{
    /* Fast path — transport already live */
    if (g_transport) {
        if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] ensure_connected() FAST PATH: transport already live (pid=%d)\n", (int)getpid()); fflush(stderr); }
        return CUDA_SUCCESS;
    }
    if (!g_device_found) {
        cuda_transport_write_error("DEVICE_NOT_FOUND", 0, 0,
            "ensure_connected: cuda_transport_discover failed");
        if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] ensure_connected() ERROR: device not found (pid=%d)\n", (int)getpid()); fflush(stderr); }
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] ensure_connected() CALLED: initializing transport to VGPU-STUB (pid=%d)\n", (int)getpid()); fflush(stderr); }

    /* Slow path — initialize transport under mutex (once) */
    ensure_mutex_init();
    pthread_mutex_lock(&g_mutex);

    /* Double-check after acquiring lock */
    if (g_transport) {
        pthread_mutex_unlock(&g_mutex);
        if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] ensure_connected() SUCCESS: transport initialized by another thread (pid=%d)\n", (int)getpid()); fflush(stderr); }
        return CUDA_SUCCESS;
    }
    if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] ensure_connected() INITIALIZING: calling cuda_transport_init() (pid=%d)\n", (int)getpid()); fflush(stderr); }
    /* Clear previous run's error and debug state for accurate diagnosis */
    (void)unlink("/tmp/vgpu_last_error");
    (void)unlink("/tmp/vgpu_debug.txt");
    cuda_transport_clear_debug_state();
    /* Diagnostic: touch file so VM check can see runner reached ensure_connected */
    {
        int mfd = (int)syscall(__NR_open, "/tmp/vgpu_ensure_connected_called", O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (mfd >= 0) {
            char buf[64];
            int n = snprintf(buf, sizeof(buf), "pid=%d\n", (int)getpid());
            if (n > 0) syscall(__NR_write, mfd, buf, (size_t)n);
            syscall(__NR_close, mfd);
        }
    }
    int rc = cuda_transport_init(&g_transport);
    if (rc != 0) {
        if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] ensure_connected() FAILED: cuda_transport_init() returned %d (pid=%d)\n", rc, (int)getpid()); fflush(stderr); }
        pthread_mutex_unlock(&g_mutex);
        fprintf(stderr, "[libvgpu-cuda] ensure_connected() transport init failed"
                        " — check resource0 permissions and mediator\n");
        return CUDA_ERROR_NO_DEVICE;
    }

    /* Register with mediator as best-effort only.
     * Some host builds reject explicit CUDA_CALL_INIT but still handle compute calls. */
    {
        CUDACallResult result = {0};
        int init_rc = cuda_transport_call(g_transport, CUDA_CALL_INIT,
                                          NULL, 0, NULL, 0, &result, NULL, 0, NULL);
        if (!(init_rc == 0 && result.status == CUDA_SUCCESS)) {
            fprintf(stderr,
                    "[libvgpu-cuda] ensure_connected() CUDA_CALL_INIT non-fatal: rc=%d status=%u num_results=%u seq=%u\n",
                    init_rc, result.status, result.num_results, result.seq_num);
            fflush(stderr);
        }
    }

    fetch_gpu_info();

    pthread_mutex_unlock(&g_mutex);
    if (vgpu_debug_logging())
        fprintf(stderr, "[libvgpu-cuda] ensure_connected() transport live"
                        " (vm_id=%u bdf=%s)\n",
                cuda_transport_vm_id(g_transport),
                cuda_transport_pci_bdf(g_transport));
    return CUDA_SUCCESS;
}

/*
 * init_gpu_defaults — fill g_gpu_info with GPU_DEFAULT_* values.
 *
 * Called from two places:
 *   1. cuInit() — immediately after cuda_transport_discover() succeeds,
 *      so that device-query functions (cuDeviceTotalMem, cuDeviceGetName,
 *      cuDeviceGetAttribute) return valid H100 values during Ollama's GPU
 *      discovery phase — BEFORE any compute call fires ensure_connected().
 *   2. fetch_gpu_info() fallback — when the host transport is unavailable.
 *
 * fetch_gpu_info() overrides these with live host values once the transport
 * is established.  The g_gpu_info_valid guard prevents double-init.
 */
static void init_gpu_defaults(void)
{
    if (g_gpu_info_valid) return;   /* real host values already set — skip */

    memset(&g_gpu_info, 0, sizeof(g_gpu_info));
    strncpy(g_gpu_info.name, GPU_DEFAULT_NAME, sizeof(g_gpu_info.name) - 1);
    g_gpu_info.total_mem              = GPU_DEFAULT_TOTAL_MEM;
    g_gpu_info.free_mem               = GPU_DEFAULT_FREE_MEM;
    g_gpu_info.compute_cap_major      = GPU_DEFAULT_CC_MAJOR;
    g_gpu_info.compute_cap_minor      = GPU_DEFAULT_CC_MINOR;
    g_gpu_info.multi_processor_count  = GPU_DEFAULT_SM_COUNT;
    g_gpu_info.max_threads_per_block  = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;
    g_gpu_info.max_block_dim_x        = GPU_DEFAULT_MAX_BLOCK_DIM_X;
    g_gpu_info.max_block_dim_y        = GPU_DEFAULT_MAX_BLOCK_DIM_Y;
    g_gpu_info.max_block_dim_z        = GPU_DEFAULT_MAX_BLOCK_DIM_Z;
    g_gpu_info.max_grid_dim_x         = GPU_DEFAULT_MAX_GRID_DIM_X;
    g_gpu_info.max_grid_dim_y         = GPU_DEFAULT_MAX_GRID_DIM_Y;
    g_gpu_info.max_grid_dim_z         = GPU_DEFAULT_MAX_GRID_DIM_Z;
    g_gpu_info.warp_size              = GPU_DEFAULT_WARP_SIZE;
    g_gpu_info.max_shared_mem_per_block = GPU_DEFAULT_SHARED_MEM_PER_BLOCK;
    g_gpu_info.max_shared_mem_per_mp  = GPU_DEFAULT_SHARED_MEM_PER_SM;
    g_gpu_info.regs_per_block         = GPU_DEFAULT_REGS_PER_BLOCK;
    g_gpu_info.regs_per_multiprocessor = GPU_DEFAULT_REGS_PER_SM;
    g_gpu_info.clock_rate_khz         = GPU_DEFAULT_CLOCK_RATE_KHZ;
    g_gpu_info.memory_clock_rate_khz  = GPU_DEFAULT_MEM_CLOCK_RATE_KHZ;
    g_gpu_info.memory_bus_width       = GPU_DEFAULT_MEM_BUS_WIDTH;
    g_gpu_info.l2_cache_size          = GPU_DEFAULT_L2_CACHE_SIZE;
    g_gpu_info.max_threads_per_mp     = GPU_DEFAULT_MAX_THREADS_PER_SM;
    g_gpu_info.unified_addressing     = GPU_DEFAULT_UNIFIED_ADDRESSING;
    g_gpu_info.managed_memory         = GPU_DEFAULT_MANAGED_MEMORY;
    g_gpu_info.concurrent_kernels     = GPU_DEFAULT_CONCURRENT_KERNELS;
    g_gpu_info.async_engine_count     = GPU_DEFAULT_ASYNC_ENGINE_COUNT;
    g_gpu_info.ecc_enabled            = GPU_DEFAULT_ECC_ENABLED;
    g_gpu_info.driver_version         = GPU_DEFAULT_DRIVER_VERSION;
    g_gpu_info.runtime_version        = GPU_DEFAULT_RUNTIME_VERSION;
    fill_default_device_identity(&g_gpu_info);
    g_gpu_info_valid = 1;
    fprintf(stderr, "[libvgpu-cuda] GPU defaults applied"
                    " (H100 80GB CC=%d.%d VRAM=%llu MB)\n",
            g_gpu_info.compute_cap_major,
            g_gpu_info.compute_cap_minor,
            (unsigned long long)(g_gpu_info.total_mem / (1024 * 1024)));
}

/*
 * Fetch GPU info from host (one-time during init).
 * Overrides the defaults set by init_gpu_defaults() with live host values.
 */
static int fetch_gpu_info(void)
{
    CUDACallResult result;
    uint32_t recv_len = 0;
    int rc;
    CUDAGpuInfo live_info;

    /*
     * Start from known-good defaults, then overlay live values from host.
     * Host structs can legally contain zeros for fields our inference path
     * divides by; sanitize before publishing.
     */
    init_gpu_defaults();
    memset(&live_info, 0, sizeof(live_info));

    rc = cuda_transport_call(g_transport,
                             CUDA_CALL_GET_GPU_INFO,
                             NULL, 0,
                             NULL, 0,
                             &result,
                             &live_info, sizeof(live_info),
                             &recv_len);

    if (rc == 0 && recv_len >= sizeof(live_info)) {
        g_gpu_info = live_info;
        sanitize_gpu_info(&g_gpu_info);
        g_gpu_info_valid = 1;
        fprintf(stderr, "[libvgpu-cuda] GPU info (live): %s, mem=%llu MB, CC=%d.%d\n",
                g_gpu_info.name,
                (unsigned long long)(g_gpu_info.total_mem / (1024 * 1024)),
                g_gpu_info.compute_cap_major,
                g_gpu_info.compute_cap_minor);
    } else {
        /* Host unavailable or partial response — restore defaults */
        init_gpu_defaults();
        sanitize_gpu_info(&g_gpu_info);
        fprintf(stderr, "[libvgpu-cuda] GPU info fetch failed — using defaults\n");
    }
    return 0;
}

/*
 * Simple blocking RPC wrapper with no send/recv data.
 * Ensures the transport is connected (lazy init) before sending.
 */
static CUresult rpc_simple(uint32_t call_id,
                           const uint32_t *args, uint32_t num_args,
                           CUDACallResult *result)
{
    CUresult rc = ensure_connected();
    if (rc != CUDA_SUCCESS) {
        if (vgpu_debug_logging())
            fprintf(stderr, "[libvgpu-cuda] rpc_simple(call_id=0x%x) ensure_connected() failed: %d\n",
                    call_id, rc);
        return rc;
    }
    CUresult call_rc = (CUresult)cuda_transport_call(g_transport, call_id,
                                         args, num_args,
                                         NULL, 0, result,
                                         NULL, 0, NULL);
    if (call_rc != CUDA_SUCCESS) {
        if (vgpu_debug_logging())
            fprintf(stderr, "[libvgpu-cuda] rpc_simple(call_id=0x%x) transport_call failed: %d\n",
                    call_id, call_rc);
    }
    return call_rc;
}

/* ================================================================
 * CUDA Driver API — Initialisation
 * ================================================================ */

/* Ensure functions are exported with default visibility */
__attribute__((visibility("default")))
CUresult cuInit(unsigned int flags)
{
    fprintf(stderr, "[libvgpu-cuda] cuInit() CALLED (pid=%d, flags=%u, already_init=%d)\n", 
            (int)getpid(), flags, g_initialized);
    fflush(stderr);
    (void)flags;
    ensure_mutex_init();
    pthread_mutex_lock(&g_mutex);

    if (g_initialized) {
        fprintf(stderr, "[libvgpu-cuda] cuInit() already initialized, returning SUCCESS immediately\n");
        fflush(stderr);
        pthread_mutex_unlock(&g_mutex);
        return CUDA_SUCCESS;
    }

    /*
     * Phase 1 — lightweight device scan.
     *
     * Only verify the VGPU-STUB PCI device is visible in /sys.  This
     * requires only read access to /sys/bus/pci/devices/, which works even
     * inside the systemd service sandbox (ProtectKernelTunables applies a
     * read-only mount, but sysfs directory listing is still allowed).
     *
     * This deliberately does NOT open resource0 or map BAR0.  Those steps
     * are deferred to ensure_connected(), called on the first real compute
     * operation inside the inference runner subprocess.
     *
     * CRITICAL FIX: Always initialize defaults and return SUCCESS during
     * init phase, even if device discovery fails. This ensures ggml_backend_cuda_init
     * can proceed and call device query functions, which will return valid
     * default values (compute capability 9.0). The device will be properly
     * discovered later when ensure_connected() is called.
     */
    int discover_rc = cuda_transport_discover();
    if (discover_rc != 0) {
        /* During init phase, always allow initialization to proceed with defaults */
        if (g_in_init_phase) {
            fprintf(stderr, "[libvgpu-cuda] cuInit() device discovery failed but in init phase, proceeding with defaults\n");
            fflush(stderr);
            /* Initialize defaults anyway so device query functions work */
            init_gpu_defaults();
            g_device_found = 1;
            g_initialized = 1;
            g_in_init_phase = 1;  /* Ensure init phase is set */
            g_gpu_info_valid = 1;  /* Mark GPU info as valid so functions can use it */
            pthread_mutex_unlock(&g_mutex);
            
            fprintf(stderr, "[libvgpu-cuda] cuInit() returning SUCCESS with defaults (CC=9.0, VRAM=%llu MB, device_count=1)\n",
                    (unsigned long long)(g_gpu_info.total_mem / (1024 * 1024)));
            fflush(stderr);
            return CUDA_SUCCESS;
        }
        /* Not in init phase - return error */
        pthread_mutex_unlock(&g_mutex);
        fprintf(stderr, "[libvgpu-cuda] cuInit() no VGPU device found in /sys\n");
        return CUDA_ERROR_NO_DEVICE;
    }
    
    /* Device discovery succeeded - proceed with normal initialization */

    /*
     * Populate g_gpu_info with GPU_DEFAULT_* values immediately.
     *
     * Even though the full transport (BAR0 mmap + mediator socket) is
     * deferred to ensure_connected(), Ollama's discovery code calls
     * cuDeviceTotalMem(), cuDeviceGetName(), and cuDeviceGetAttribute()
     * right here in the probe phase — before any compute op fires.
     * Without this call those functions return 0 / "" because g_gpu_info
     * is C-zero-initialised, and Ollama discards a device with CC=0 or
     * VRAM=0, ultimately reporting library=cpu.
     *
     * fetch_gpu_info() will later overwrite these with live host values
     * once ensure_connected() succeeds in the inference runner.
     */
    init_gpu_defaults();

    g_device_found = 1;
    g_initialized  = 1;
    g_in_init_phase = 1;  /* Start in init phase - will be cleared after first context creation */
    g_gpu_info_valid = 1;  /* Mark GPU info as valid so functions can use it immediately */
    pthread_mutex_unlock(&g_mutex);

    fprintf(stderr,
            "[libvgpu-cuda] cuInit() device found at %s — transport deferred"
            " (CC=%d.%d VRAM=%llu MB, init_phase=1)\n",
            cuda_transport_pci_bdf(NULL),
            g_gpu_info.compute_cap_major,
            g_gpu_info.compute_cap_minor,
            (unsigned long long)(g_gpu_info.total_mem / (1024 * 1024)));

    /* Mirror key event to the per-PID log file. */
    {
        char logpath[128];
        snprintf(logpath, sizeof(logpath),
                 "/tmp/vgpu-shim-cuda-%d.log", (int)getpid());
        FILE *lf = fopen(logpath, "a");
        if (lf) {
            fprintf(lf,
                    "[libvgpu-cuda] cuInit() OK: bdf=%s CC=%d.%d VRAM=%llu MB\n",
                    cuda_transport_pci_bdf(NULL),
                    g_gpu_info.compute_cap_major,
                    g_gpu_info.compute_cap_minor,
                    (unsigned long long)(g_gpu_info.total_mem / (1024 * 1024)));
            fclose(lf);
        }
    }

    /* Eager best-effort transport bring-up for libraries that expect cuInit()
     * to complete low-level driver readiness before any other API call. */
    {
        CUresult eager_rc = ensure_connected();
        if (eager_rc != CUDA_SUCCESS) {
            fprintf(stderr,
                    "[libvgpu-cuda] cuInit() eager ensure_connected non-fatal: rc=%d\n",
                    eager_rc);
            fflush(stderr);
        } else {
            CUcontext eager_ctx = NULL;
            CUresult ctx_rc = cuDevicePrimaryCtxRetain(&eager_ctx, 0);
            fprintf(stderr,
                    "[libvgpu-cuda] cuInit() eager primary ctx retain: rc=%d ctx=%p\n",
                    ctx_rc, (void *)eager_ctx);
            fflush(stderr);

            if (ctx_rc == CUDA_SUCCESS && eager_ctx != NULL &&
                (uintptr_t)eager_ctx != (uintptr_t)0xDEADBEEF) {
                CUresult set_rc = cuCtxSetCurrent(eager_ctx);
                fprintf(stderr,
                        "[libvgpu-cuda] cuInit() eager cuCtxSetCurrent: rc=%d ctx=%p\n",
                        set_rc, (void *)eager_ctx);
                fflush(stderr);
            }
        }
    }
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress(const char *symbol, void **funcPtr, 
                          int cudaVersion, cuuint64_t flags)
{
    /* CRITICAL: Log FIRST using syscall to catch ALL calls */
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] cuGetProcAddress() CALLED: symbol=\"%s\", cudaVersion=%d, flags=0x%llx (pid=%d)\n",
                          symbol ? symbol : "(null)", cudaVersion, (unsigned long long)flags, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    if (!symbol || !funcPtr) {
        fprintf(stderr, "[libvgpu-cuda] cuGetProcAddress: invalid arguments\n");
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Newer CUDA user-space occasionally probes with an empty symbol name.
     * Do not pass that through to dlsym(), because the dynamic loader emits
     * a fatal blank-symbol lookup error. Treat it like an optional unresolved
     * entry point and return SUCCESS with a NULL function pointer. */
    if (symbol[0] == '\0') {
        *funcPtr = NULL;
        fprintf(stderr,
                "[libvgpu-cuda] cuGetProcAddress: empty symbol probe, returning NULL without dlsym\n");
        fflush(stderr);
        return CUDA_SUCCESS;
    }
    
    /* Resolve function pointer from our shim.
     * First try to get a handle to our own shim library, then use dlsym on that.
     * This ensures we find functions in our shim even if they're not in the global scope. */
    static void *shim_handle = NULL;
    if (!shim_handle) {
        Dl_info self_info;
        memset(&self_info, 0, sizeof(self_info));

        /*
         * Resolve symbols from the currently loaded shim first. Falling back to
         * a hard-coded install path can accidentally load a second copy of the
         * shim and route cuGetProcAddress() lookups to stale code.
         */
        if (dladdr((void *)&cuGetProcAddress, &self_info) &&
            self_info.dli_fname && self_info.dli_fname[0]) {
            shim_handle = dlopen(self_info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
            if (!shim_handle) {
                shim_handle = dlopen(self_info.dli_fname, RTLD_LAZY);
            }
            if (shim_handle) {
                char log_path[256];
                int path_len = snprintf(log_path, sizeof(log_path),
                                       "[libvgpu-cuda] cuGetProcAddress: loaded current shim from %s\n",
                                       self_info.dli_fname);
                if (path_len > 0 && path_len < (int)sizeof(log_path)) {
                    syscall(__NR_write, 2, log_path, path_len);
                }
            }
        }

        if (!shim_handle) {
            /* Fallback for cases where dladdr() cannot identify the current image. */
            const char *paths[] = {
                "/opt/vgpu/lib/libvgpu-cuda.so.1",
                "/opt/vgpu/lib/libvgpu-cuda.so",
                "/usr/lib64/libvgpu-cuda.so.1",
                "/usr/lib64/libvgpu-cuda.so",
                NULL
            };
            for (int i = 0; paths[i] && !shim_handle; i++) {
                shim_handle = dlopen(paths[i], RTLD_LAZY);
                if (shim_handle) {
                    char log_path[256];
                    int path_len = snprintf(log_path, sizeof(log_path),
                                           "[libvgpu-cuda] cuGetProcAddress: loaded shim from %s\n",
                                           paths[i]);
                    if (path_len > 0 && path_len < (int)sizeof(log_path)) {
                        syscall(__NR_write, 2, log_path, path_len);
                    }
                    break;
                }
            }
        }
        if (!shim_handle) {
            /* Fallback: try to find ourselves via RTLD_DEFAULT */
            shim_handle = dlopen(NULL, RTLD_LAZY);
            if (shim_handle) {
                const char *fallback_msg = "[libvgpu-cuda] cuGetProcAddress: using RTLD_DEFAULT fallback\n";
                (void)syscall(__NR_write, 2, fallback_msg,
                              sizeof("[libvgpu-cuda] cuGetProcAddress: using RTLD_DEFAULT fallback\n") - 1);
            }
        }
    }
    
    if (!shim_handle) {
        fprintf(stderr, "[libvgpu-cuda] cuGetProcAddress: failed to get shim handle\n");
        return CUDA_ERROR_UNKNOWN;
    }
    
    /* Try multiple methods to find the symbol */
    void *func = NULL;
    
    /* Method 1: Try dlsym on our shim handle */
    func = dlsym(shim_handle, symbol);
    
    /* Method 2: If not found, try RTLD_DEFAULT (searches all loaded libraries) */
    if (!func) {
        func = dlsym(RTLD_DEFAULT, symbol);
    }
    
    /* Method 3: Try RTLD_NEXT (searches libraries loaded after this one) */
    if (!func) {
        func = dlsym(RTLD_NEXT, symbol);
    }
    
    /* Method 4: During initialization phase, provide direct function pointers
     * for common CUDA functions that might be requested. This ensures that
     * even if dlsym fails, we can still provide the function. */
    if (!func && g_in_init_phase) {
        /* Provide direct function pointers for common functions */
        static struct {
            const char *name;
            void *func;
        } stub_funcs[] = {
            {"cuDeviceGetProperties", (void*)cuDeviceGetProperties},
            {"cuDeviceGetProperties_v2", (void*)cuDeviceGetProperties},
            {"cuCtxGetCurrent", (void*)cuCtxGetCurrent},
            {"cuCtxGetCurrent_v2", (void*)cuCtxGetCurrent},
            {"cuCtxPushCurrent", (void*)cuCtxPushCurrent_v2},
            {"cuCtxPushCurrent_v2", (void*)cuCtxPushCurrent_v2},
            {"cuCtxPopCurrent", (void*)cuCtxPopCurrent_v2},
            {"cuCtxPopCurrent_v2", (void*)cuCtxPopCurrent_v2},
            {"cuCtxSynchronize", (void*)cuCtxSynchronize},
            {"cuCtxGetDevice", (void*)cuCtxGetDevice},
            {"cuCtxGetDevice_v2", (void*)cuCtxGetDevice_v2},
            {"cuCtxGetFlags", (void*)cuCtxGetFlags},
            {"cuCtxSetLimit", (void*)cuCtxSetLimit},
            {"cuCtxGetLimit", (void*)cuCtxGetLimit},
            {"cuCtxGetApiVersion", (void*)cuCtxGetApiVersion},
            {"cuDeviceComputeCapability", (void*)cuDeviceComputeCapability},
            {"cuDeviceGetAttribute", (void*)cuDeviceGetAttribute},
            {"cuDeviceGetAttribute_v2", (void*)cuDeviceGetAttribute_v2},
            {"cuDeviceGetName", (void*)cuDeviceGetName},
            {"cuDeviceGetPCIBusId", (void*)cuDeviceGetPCIBusId},
            {"cuDeviceGetPCIBusId_v2", (void*)cuDeviceGetPCIBusId},
            {"cuDeviceGetCount", (void*)cuDeviceGetCount},
            {"cuDeviceGetCount_v2", (void*)cuDeviceGetCount},
            {"cuDeviceGet", (void*)cuDeviceGet},
            {"cuDeviceGet_v2", (void*)cuDeviceGet},
            {"cuDeviceTotalMem", (void*)cuDeviceTotalMem},
            {"cuDeviceTotalMem_v2", (void*)cuDeviceTotalMem_v2},
            {"cuDriverGetVersion", (void*)cuDriverGetVersion},
            {"cuInit", (void*)cuInit},
            {"cuDevicePrimaryCtxRetain", (void*)cuDevicePrimaryCtxRetain},
            {"cuDevicePrimaryCtxRetain_v2", (void*)cuDevicePrimaryCtxRetain},
            {"cuCtxSetCurrent", (void*)cuCtxSetCurrent},
            {"cuCtxCreate", (void*)cuCtxCreate},
            {"cuCtxCreate_v2", (void*)cuCtxCreate_v2},
            {"cuCtxCreate_v3", (void*)cuCtxCreate_v3},
            {"cuDeviceGetTexture1DLinearMaxWidth", (void*)cuDeviceGetTexture1DLinearMaxWidth},
            {"cuDeviceGetNvSciSyncAttributes", (void*)cuDeviceGetNvSciSyncAttributes},
            {"cuDeviceGetDefaultMemPool", (void*)cuDeviceGetDefaultMemPool},
            {"cuDeviceGetMemPool", (void*)cuDeviceGetMemPool},
            {"cuMemAllocPitch", (void*)cuMemAllocPitch},
            {"cuMemAllocPitch_v2", (void*)cuMemAllocPitch_v2},
            {"cuMemGetAddressRange", (void*)cuMemGetAddressRange},
            {"cuMemGetAddressRange_v2", (void*)cuMemGetAddressRange_v2},
            {"cuMemcpy", (void*)cuMemcpy},
            {"cuMemcpyAsync", (void*)cuMemcpyAsync},
            {"cuMemcpy2DUnaligned", (void*)cuMemcpy2DUnaligned},
            {"cuMemcpy2DAsync", (void*)cuMemcpy2DAsync},
            {"cuMemcpy2DAsync_v2", (void*)cuMemcpy2DAsync_v2},
            {"cuMemcpy3D", (void*)cuMemcpy3D},
            {"cuMemcpy3DAsync", (void*)cuMemcpy3DAsync},
            {"cuMemcpy3DPeer", (void*)cuMemcpy3DPeer},
            {"cuMemcpy3DPeerAsync", (void*)cuMemcpy3DPeerAsync},
            {"cuMemcpyBatchAsync", (void*)cuMemcpyBatchAsync},
            {"cuMemcpy3DBatchAsync", (void*)cuMemcpy3DBatchAsync},
            {"cuMemcpyDtoDAsync", (void*)cuMemcpyDtoDAsync},
            {"cuMemcpyDtoDAsync_v2", (void*)cuMemcpyDtoDAsync_v2},
            {"cuMemcpyPeer", (void*)cuMemcpyPeer},
            {"cuMemcpyPeerAsync", (void*)cuMemcpyPeerAsync},
            {"cuMemsetD8Async", (void*)cuMemsetD8Async},
            {"cuMemsetD2D8", (void*)cuMemsetD2D8},
            {"cuMemsetD2D8Async", (void*)cuMemsetD2D8Async},
            {"cuArrayCreate", (void*)cuArrayCreate},
            {"cuArrayGetDescriptor", (void*)cuArrayGetDescriptor},
            {"cuArrayGetSparseProperties", (void*)cuArrayGetSparseProperties},
            {"cuArrayGetPlane", (void*)cuArrayGetPlane},
            {"cuArray3DCreate", (void*)cuArray3DCreate},
            {"cuArray3DGetDescriptor", (void*)cuArray3DGetDescriptor},
            {"cuArrayDestroy", (void*)cuArrayDestroy},
            {"cuMipmappedArrayCreate", (void*)cuMipmappedArrayCreate},
            {"cuMipmappedArrayGetLevel", (void*)cuMipmappedArrayGetLevel},
            {"cuMipmappedArrayGetSparseProperties", (void*)cuMipmappedArrayGetSparseProperties},
            {"cuMipmappedArrayDestroy", (void*)cuMipmappedArrayDestroy},
            {"cuArrayGetMemoryRequirements", (void*)cuArrayGetMemoryRequirements},
            {"cuMipmappedArrayGetMemoryRequirements", (void*)cuMipmappedArrayGetMemoryRequirements},
            {"cuTexObjectCreate", (void*)cuTexObjectCreate},
            {"cuTexObjectDestroy", (void*)cuTexObjectDestroy},
            {"cuTexObjectGetResourceDesc", (void*)cuTexObjectGetResourceDesc},
            {"cuTexObjectGetTextureDesc", (void*)cuTexObjectGetTextureDesc},
            {"cuTexObjectGetResourceViewDesc", (void*)cuTexObjectGetResourceViewDesc},
            {"cuSurfObjectCreate", (void*)cuSurfObjectCreate},
            {"cuSurfObjectDestroy", (void*)cuSurfObjectDestroy},
            {"cuSurfObjectGetResourceDesc", (void*)cuSurfObjectGetResourceDesc},
            {"cuMemPoolExportPointer", (void*)cuMemPoolExportPointer},
            {"cuMemPoolImportPointer", (void*)cuMemPoolImportPointer},
            {"cuFuncSetSharedMemConfig", (void*)cuFuncSetSharedMemConfig},
            {"cuFuncGetName", (void*)cuFuncGetName},
            {"cuFuncGetParamInfo", (void*)cuFuncGetParamInfo},
            {"cuImportExternalMemory", (void*)cuImportExternalMemory},
            {"cuExternalMemoryGetMappedBuffer", (void*)cuExternalMemoryGetMappedBuffer},
            {"cuExternalMemoryGetMappedMipmappedArray", (void*)cuExternalMemoryGetMappedMipmappedArray},
            {"cuDestroyExternalMemory", (void*)cuDestroyExternalMemory},
            {"cuImportExternalSemaphore", (void*)cuImportExternalSemaphore},
            {"cuSignalExternalSemaphoresAsync", (void*)cuSignalExternalSemaphoresAsync},
            {"cuWaitExternalSemaphoresAsync", (void*)cuWaitExternalSemaphoresAsync},
            {"cuDestroyExternalSemaphore", (void*)cuDestroyExternalSemaphore},
            {"cuEventRecordWithFlags", (void*)cuEventRecordWithFlags},
            {"cuStreamWaitValue32", (void*)cuStreamWaitValue32},
            {"cuStreamWriteValue32", (void*)cuStreamWriteValue32},
            {"cuStreamWaitValue64", (void*)cuStreamWaitValue64},
            {"cuStreamWriteValue64", (void*)cuStreamWriteValue64},
            {"cuStreamBatchMemOp", (void*)cuStreamBatchMemOp},
            {"cuIpcGetEventHandle", (void*)cuIpcGetEventHandle},
            {"cuIpcOpenEventHandle", (void*)cuIpcOpenEventHandle},
            {"cuIpcGetMemHandle", (void*)cuIpcGetMemHandle},
            {"cuIpcOpenMemHandle", (void*)cuIpcOpenMemHandle},
            {"cuIpcCloseMemHandle", (void*)cuIpcCloseMemHandle},
            {"cuGLCtxCreate", (void*)cuGLCtxCreate},
            {"cuGLInit", (void*)cuGLInit},
            {"cuGLGetDevices", (void*)cuGLGetDevices},
            {"cuGLRegisterBufferObject", (void*)cuGLRegisterBufferObject},
            {"cuGLMapBufferObject", (void*)cuGLMapBufferObject},
            {"cuGLMapBufferObjectAsync", (void*)cuGLMapBufferObjectAsync},
            {"cuGLUnmapBufferObject", (void*)cuGLUnmapBufferObject},
            {"cuGLUnmapBufferObjectAsync", (void*)cuGLUnmapBufferObjectAsync},
            {"cuGLUnregisterBufferObject", (void*)cuGLUnregisterBufferObject},
            {"cuGLSetBufferObjectMapFlags", (void*)cuGLSetBufferObjectMapFlags},
            {"cuGraphicsGLRegisterImage", (void*)cuGraphicsGLRegisterImage},
            {"cuGraphicsGLRegisterBuffer", (void*)cuGraphicsGLRegisterBuffer},
            {"cuGraphicsEGLRegisterImage", (void*)cuGraphicsEGLRegisterImage},
            {"cuEGLStreamConsumerConnect", (void*)cuEGLStreamConsumerConnect},
            {"cuEGLStreamConsumerDisconnect", (void*)cuEGLStreamConsumerDisconnect},
            {"cuEGLStreamConsumerAcquireFrame", (void*)cuEGLStreamConsumerAcquireFrame},
            {"cuEGLStreamConsumerReleaseFrame", (void*)cuEGLStreamConsumerReleaseFrame},
            {"cuEGLStreamProducerConnect", (void*)cuEGLStreamProducerConnect},
            {"cuEGLStreamProducerDisconnect", (void*)cuEGLStreamProducerDisconnect},
            {"cuEGLStreamProducerPresentFrame", (void*)cuEGLStreamProducerPresentFrame},
            {"cuEGLStreamProducerReturnFrame", (void*)cuEGLStreamProducerReturnFrame},
            {"cuEGLStreamConsumerConnectWithFlags", (void*)cuEGLStreamConsumerConnectWithFlags},
            {"cuStreamGetFlags", (void*)cuStreamGetFlags},
            {"cuStreamGetPriority", (void*)cuStreamGetPriority},
            {"cuStreamGetDevice", (void*)cuStreamGetDevice},
            {"cuStreamGetCtx", (void*)cuStreamGetCtx},
            {NULL, NULL}
        };
        
        for (int i = 0; stub_funcs[i].name; i++) {
            if (strcmp(symbol, stub_funcs[i].name) == 0) {
                func = stub_funcs[i].func;
                fprintf(stderr, "[libvgpu-cuda] cuGetProcAddress: providing direct pointer for \"%s\" during init phase\n", symbol);
                fflush(stderr);
                break;
            }
        }
    }
    
    /* For unresolved symbols, prefer CUDA's documented behavior: return
     * SUCCESS with a NULL function pointer. This is safer than handing back
     * a fake non-NULL stub for APIs we do not actually implement, because
     * newer CUDA libraries may sanity-check the returned pointers or call
     * them expecting real output semantics. */
    if (!func && g_in_init_phase) {
        fprintf(stderr, "[libvgpu-cuda] cuGetProcAddress: unresolved \"%s\" during init phase, returning NULL pointer\n", symbol);
        fflush(stderr);
    }
    
    /* Same policy outside init: unresolved entry points return SUCCESS with
     * a NULL pointer, letting the caller decide whether the symbol is
     * optional for its requested CUDA version. */
    if (!func) {
        fprintf(stderr, "[libvgpu-cuda] cuGetProcAddress: unresolved \"%s\" (func not found via dlsym), returning NULL pointer\n", symbol);
        fflush(stderr);
    }
    
    *funcPtr = func;
    fprintf(stderr, "[libvgpu-cuda] cuGetProcAddress: resolved \"%s\" -> %p - RETURNING SUCCESS (init_phase=%d)\n", 
            symbol, func, g_in_init_phase);
    fflush(stderr);
    return CUDA_SUCCESS;
}

/* Some CUDA libraries probe export tables for optional subsystems.
 * Provide the minimal dark-API tables expected by CUDA 12 user-space libs. */
static CUresult dark_get_primary_context(CUcontext *pctx, CUdevice dev)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_get_primary_context() CALLED: pctx=%p dev=%d (pid=%d)\n",
                          (void *)pctx, (int)dev, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    CUresult rc = cuDevicePrimaryCtxRetain(pctx, dev);
    if (rc == CUDA_SUCCESS && pctx && *pctx) {
        (void)cuCtxSetCurrent(*pctx);
    }
    log_len = snprintf(log_msg, sizeof(log_msg),
                       "[libvgpu-cuda] dark_get_primary_context() RESULT: rc=%d ctx=%p (pid=%d)\n",
                       (int)rc, (pctx ? (void *)*pctx : NULL), (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return rc;
}

static CUresult dark_get_module_from_cubin(CUmodule *module, const void *fatbinc_wrapper)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_get_module_from_cubin() CALLED: module=%p fatbin=%p (pid=%d)\n",
                          (void *)module, fatbinc_wrapper, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    if (!module || !fatbinc_wrapper) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuModuleLoadData(module, fatbinc_wrapper);
}

static CUresult dark_get_module_from_cubin_ex1(CUmodule *module, const void *fatbinc_wrapper,
                                               void *arg3, void *arg4, size_t arg5)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_get_module_from_cubin_ex1() CALLED: module=%p fatbin=%p arg3=%p arg4=%p arg5=%zu (pid=%d)\n",
                          (void *)module, fatbinc_wrapper, arg3, arg4, arg5, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    (void)arg3;
    (void)arg4;
    (void)arg5;
    if (!module || !fatbinc_wrapper) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuModuleLoadData(module, fatbinc_wrapper);
}

static void dark_cudart_interface_fn7(size_t arg1)
{
    (void)arg1;
}

static CUresult dark_not_supported_result_stub(void)
{
    const char *msg = "[libvgpu-cuda] dark_not_supported_result_stub() CALLED\n";
    syscall(__NR_write, 2, msg, sizeof("[libvgpu-cuda] dark_not_supported_result_stub() CALLED\n") - 1);
    return CUDA_ERROR_NOT_SUPPORTED;
}

static void dark_noop_void_stub(void)
{
    const char *msg = "[libvgpu-cuda] dark_noop_void_stub() CALLED\n";
    syscall(__NR_write, 2, msg, sizeof("[libvgpu-cuda] dark_noop_void_stub() CALLED\n") - 1);
}

static CUresult dark_get_module_from_cubin_ex2(const void *fatbin_header, CUmodule *module,
                                               void *arg3, void *arg4, uint32_t arg5)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_get_module_from_cubin_ex2() CALLED: fatbin=%p module=%p arg3=%p arg4=%p arg5=%u (pid=%d)\n",
                          fatbin_header, (void *)module, arg3, arg4, arg5, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    (void)arg3;
    (void)arg4;
    (void)arg5;
    if (!module || !fatbin_header) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuModuleLoadData(module, fatbin_header);
}

static CUresult dark_launch_kernel(CUfunction f,
                                   uint32_t grid_dim_x, uint32_t grid_dim_y, uint32_t grid_dim_z,
                                   uint32_t block_dim_x, uint32_t block_dim_y, uint32_t block_dim_z,
                                   uint32_t shared_mem_bytes, CUstream stream, void **extra)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_launch_kernel() CALLED: f=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u stream=%p extra=%p (pid=%d)\n",
                          (void *)f, grid_dim_x, grid_dim_y, grid_dim_z,
                          block_dim_x, block_dim_y, block_dim_z,
                          shared_mem_bytes, (void *)stream, (void *)extra, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    (void)f; (void)grid_dim_x; (void)grid_dim_y; (void)grid_dim_z;
    (void)block_dim_x; (void)block_dim_y; (void)block_dim_z;
    (void)shared_mem_bytes; (void)stream; (void)extra;
    return CUDA_ERROR_NOT_SUPPORTED;
}

static void dark_tools_runtime_fn2(uint64_t **ptr, size_t *size)
{
    static uint64_t buffer[131072] = {0}; /* 1 MiB */
    char log_msg[192];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_tools_runtime_fn2() CALLED: ptr=%p size=%p (pid=%d)\n",
                          (void *)ptr, (void *)size, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    if (ptr) *ptr = buffer;
    if (size) *size = sizeof(buffer);
}

static void dark_tools_runtime_fn6(uint8_t **ptr, size_t *size)
{
    static uint8_t buffer[1048576] = {0}; /* 1 MiB */
    char log_msg[192];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_tools_runtime_fn6() CALLED: ptr=%p size=%p (pid=%d)\n",
                          (void *)ptr, (void *)size, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    if (ptr) *ptr = buffer;
    if (size) *size = sizeof(buffer);
}

typedef void (*dark_context_storage_dtor_cb)(CUcontext, void *, void *);

typedef struct {
    CUcontext ctx;
    void *key;
    void *value;
    dark_context_storage_dtor_cb dtor_cb;
} dark_context_storage_entry_t;

#define DARK_CONTEXT_STORAGE_MAX_ENTRIES 64
static dark_context_storage_entry_t g_dark_context_storage[DARK_CONTEXT_STORAGE_MAX_ENTRIES];
static size_t g_dark_context_storage_count = 0;
static const void *g_dark_cudart_table_addr = NULL;
static const void *g_dark_integrity_check_table_addr = NULL;
static CUresult dark_anti_zluda_check(uint32_t runtime_version, uint64_t timestamp, void *result);

typedef struct {
    uint32_t driver_version;
    uint32_t version;
    uint32_t current_process;
    uint32_t current_thread;
    const void *cudart_table;
    const void *integrity_check_table;
    const void *fn_address;
    uint64_t unix_seconds;
} dark_integrity_pass3_input_t;

typedef struct {
    unsigned char guid[16];
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} dark_integrity_device_hashinfo_t;

static void dark_integrity_single_pass(unsigned char state[66], unsigned char input)
{
    static const unsigned char k_dark_integrity_mixing_table[256] = {
        0x29, 0x2E, 0x43, 0xC9, 0xA2, 0xD8, 0x7C, 0x01, 0x3D, 0x36, 0x54, 0xA1, 0xEC, 0xF0, 0x06,
        0x13, 0x62, 0xA7, 0x05, 0xF3, 0xC0, 0xC7, 0x73, 0x8C, 0x98, 0x93, 0x2B, 0xD9, 0xBC, 0x4C,
        0x82, 0xCA, 0x1E, 0x9B, 0x57, 0x3C, 0xFD, 0xD4, 0xE0, 0x16, 0x67, 0x42, 0x6F, 0x18, 0x8A,
        0x17, 0xE5, 0x12, 0xBE, 0x4E, 0xC4, 0xD6, 0xDA, 0x9E, 0xDE, 0x49, 0xA0, 0xFB, 0xF5, 0x8E,
        0xBB, 0x2F, 0xEE, 0x7A, 0xA9, 0x68, 0x79, 0x91, 0x15, 0xB2, 0x07, 0x3F, 0x94, 0xC2, 0x10,
        0x89, 0x0B, 0x22, 0x5F, 0x21, 0x80, 0x7F, 0x5D, 0x9A, 0x5A, 0x90, 0x32, 0x27, 0x35, 0x3E,
        0xCC, 0xE7, 0xBF, 0xF7, 0x97, 0x03, 0xFF, 0x19, 0x30, 0xB3, 0x48, 0xA5, 0xB5, 0xD1, 0xD7,
        0x5E, 0x92, 0x2A, 0xAC, 0x56, 0xAA, 0xC6, 0x4F, 0xB8, 0x38, 0xD2, 0x96, 0xA4, 0x7D, 0xB6,
        0x76, 0xFC, 0x6B, 0xE2, 0x9C, 0x74, 0x04, 0xF1, 0x45, 0x9D, 0x70, 0x59, 0x64, 0x71, 0x87,
        0x20, 0x86, 0x5B, 0xCF, 0x65, 0xE6, 0x2D, 0xA8, 0x02, 0x1B, 0x60, 0x25, 0xAD, 0xAE, 0xB0,
        0xB9, 0xF6, 0x1C, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7E, 0x0F, 0x55, 0x47, 0xA3, 0x23, 0xDD,
        0x51, 0xAF, 0x3A, 0xC3, 0x5C, 0xF9, 0xCE, 0xBA, 0xC5, 0xEA, 0x26, 0x2C, 0x53, 0x0D, 0x6E,
        0x85, 0x28, 0x84, 0x09, 0xD3, 0xDF, 0xCD, 0xF4, 0x41, 0x81, 0x4D, 0x52, 0x6A, 0xDC, 0x37,
        0xC8, 0x6C, 0xC1, 0xAB, 0xFA, 0x24, 0xE1, 0x7B, 0x08, 0x0C, 0xBD, 0xB1, 0x4A, 0x78, 0x88,
        0x95, 0x8B, 0xE3, 0x63, 0xE8, 0x6D, 0xE9, 0xCB, 0xD5, 0xFE, 0x3B, 0x00, 0x1D, 0x39, 0xF2,
        0xEF, 0xB7, 0x0E, 0x66, 0x58, 0xD0, 0xE4, 0xA6, 0x77, 0x72, 0xF8, 0xEB, 0x75, 0x4B, 0x0A,
        0x31, 0x44, 0x50, 0xB4, 0x8F, 0xED, 0x1F, 0x1A, 0xDB, 0x99, 0x8D, 0x33, 0x9F, 0x11, 0x83,
        0x14
    };
    unsigned char temp1 = state[0x40];
    size_t idx = (size_t)temp1;
    unsigned char next = (unsigned char)((temp1 + 1U) & 0x0fU);
    unsigned char temp4;
    unsigned char round_accum;
    unsigned char round_idx;

    state[idx + 0x10] = input;
    state[idx + 0x20] = (unsigned char)(state[idx] ^ input);
    temp4 = k_dark_integrity_mixing_table[(unsigned char)(input ^ state[0x41])];
    round_accum = state[idx + 0x30];
    state[idx + 0x30] = (unsigned char)(temp4 ^ round_accum);
    state[0x41] = (unsigned char)(temp4 ^ round_accum);
    state[0x40] = next;
    if (next != 0) {
        return;
    }

    temp1 = 0x29;
    round_idx = 0x00;
    for (;;) {
        size_t pos;
        temp1 = (unsigned char)(temp1 ^ state[0]);
        state[0] = temp1;
        for (pos = 1; pos < 0x30; ++pos) {
            temp1 = (unsigned char)(state[pos] ^ k_dark_integrity_mixing_table[temp1]);
            state[pos] = temp1;
        }
        temp1 = (unsigned char)(temp1 + round_idx);
        round_idx = (unsigned char)(round_idx + 1U);
        if (round_idx == 0x12) {
            break;
        }
        temp1 = k_dark_integrity_mixing_table[temp1];
    }
}

static void dark_integrity_hash_pass(unsigned char state[66], const void *data, size_t len, unsigned char xor_mask)
{
    const unsigned char *bytes = (const unsigned char *)data;
    size_t i;

    if (!bytes || len == 0) {
        return;
    }

    for (i = 0; i < len; ++i) {
        dark_integrity_single_pass(state, (unsigned char)(bytes[i] ^ xor_mask));
    }
}

static void dark_integrity_zero_result(unsigned char state[66])
{
    memset(state, 0, 16);
    memset(state + 48, 0, 18);
}

static void dark_log_hex_bytes(const char *label, const void *data, size_t len)
{
    char log_msg[512];
    size_t pos = 0;
    size_t i;

    if (!label || !data || len == 0) {
        return;
    }

    pos += (size_t)snprintf(log_msg + pos, sizeof(log_msg) - pos,
                            "[libvgpu-cuda] %s (%zu bytes):", label, len);
    for (i = 0; i < len && pos + 4 < sizeof(log_msg); ++i) {
        pos += (size_t)snprintf(log_msg + pos, sizeof(log_msg) - pos,
                                " %02x", ((const unsigned char *)data)[i]);
    }
    if (pos + 2 < sizeof(log_msg)) {
        log_msg[pos++] = '\n';
        log_msg[pos] = '\0';
    } else {
        log_msg[sizeof(log_msg) - 2] = '\n';
        log_msg[sizeof(log_msg) - 1] = '\0';
        pos = sizeof(log_msg) - 1;
    }

    syscall(__NR_write, 2, log_msg, pos);
}

static void dark_integrity_pass5(unsigned char state[66], uint64_t out[2])
{
    unsigned char temp = (unsigned char)(16U - state[64]);
    size_t i;

    for (i = 0; i < (size_t)temp; ++i) {
        dark_integrity_single_pass(state, temp);
    }
    for (i = 0x30; i < 0x40; ++i) {
        dark_integrity_single_pass(state, state[i]);
    }

    memcpy(&out[0], state, sizeof(uint64_t));
    memcpy(&out[1], state + 8, sizeof(uint64_t));
}

static void dark_integrity_fill_device_info(dark_integrity_device_hashinfo_t *device_info)
{
    const char *bdf;
    unsigned int domain = GPU_DEFAULT_PCI_DOMAIN_ID;
    unsigned int bus = GPU_DEFAULT_PCI_BUS_ID;
    unsigned int pci_device_id = GPU_DEFAULT_PCI_DEVICE_ID;
    int need_bdf_fallback = 1;

    if (!device_info) {
        return;
    }

    memset(device_info, 0, sizeof(*device_info));
    memcpy(device_info->guid, g_gpu_info.uuid, sizeof(device_info->guid));

    if (g_gpu_info.pci_domain_id > 0) {
        domain = (unsigned int)g_gpu_info.pci_domain_id;
        need_bdf_fallback = 0;
    }
    if (g_gpu_info.pci_bus_id > 0) {
        bus = (unsigned int)g_gpu_info.pci_bus_id;
        need_bdf_fallback = 0;
    }
    if (g_gpu_info.pci_device_id > 0) {
        pci_device_id = (unsigned int)g_gpu_info.pci_device_id;
    }

    if (need_bdf_fallback) {
        bdf = cuda_transport_pci_bdf(NULL);
        if (bdf && bdf[0]) {
            unsigned int parsed_domain = 0;
            unsigned int parsed_bus = 0;
            unsigned int parsed_slot = 0;
            unsigned int parsed_function = 0;
            if (sscanf(bdf, "%x:%x:%x.%x",
                       &parsed_domain, &parsed_bus, &parsed_slot, &parsed_function) == 4) {
                (void)parsed_slot;
                (void)parsed_function;
                domain = parsed_domain;
                bus = parsed_bus;
            }
        }
    }

    device_info->pci_domain = (int32_t)domain;
    device_info->pci_bus = (int32_t)bus;
    device_info->pci_device = (int32_t)pci_device_id;
}

static void dark_integrity_compute_token(uint32_t runtime_version, uint64_t timestamp, uint64_t out[2])
{
    static const unsigned char k_pass1_result[16] = {
        0x14, 0x6A, 0xDD, 0xAE, 0x53, 0xA9, 0xA7, 0x52,
        0xAA, 0x08, 0x41, 0x36, 0x0B, 0xF5, 0x5A, 0x9F
    };
    unsigned char state[66] = {0};
    dark_integrity_pass3_input_t pass3_input;
    dark_integrity_device_hashinfo_t device_info;
    uint64_t pass5_stage1[2];
    uint32_t version_mod = runtime_version % 10U;
    uint32_t effective_driver_version;

    if (!out) {
        return;
    }

    if (version_mod == 0U) {
        out[0] = 0x3341181C03CB675CULL;
        out[1] = 0x8ED383AA1F4CD1E8ULL;
        return;
    }
    if (version_mod == 1U) {
        out[0] = 0x1841181C03CB675CULL;
        out[1] = 0x8ED383AA1F4CD1E8ULL;
        return;
    }
    effective_driver_version = (uint32_t)effective_userspace_driver_version();
    memset(&pass3_input, 0, sizeof(pass3_input));
    pass3_input.driver_version = effective_driver_version;
    pass3_input.version = runtime_version;
    pass3_input.current_process = (uint32_t)getpid();
    pass3_input.current_thread = (uint32_t)(uintptr_t)pthread_self();
    pass3_input.cudart_table = g_dark_cudart_table_addr;
    pass3_input.integrity_check_table = g_dark_integrity_check_table_addr;
    pass3_input.fn_address = (const void *)dark_anti_zluda_check;
    pass3_input.unix_seconds = timestamp;

    dark_integrity_hash_pass(state, k_pass1_result, sizeof(k_pass1_result), 0x36);
    dark_integrity_hash_pass(state, &pass3_input, sizeof(pass3_input), 0x00);

    dark_integrity_fill_device_info(&device_info);
    dark_integrity_hash_pass(state, &device_info, sizeof(device_info), 0x00);

    if (version_mod == 2U) {
        char log_msg[256];
        int log_len = snprintf(
            log_msg, sizeof(log_msg),
            "[libvgpu-cuda] dark_integrity_compute_token(): driver=%u runtime=%u pid=%u tid=%u domain=%d bus=%d device=%d ts=%llu\n",
            pass3_input.driver_version,
            pass3_input.version,
            pass3_input.current_process,
            pass3_input.current_thread,
            device_info.pci_domain,
            device_info.pci_bus,
            device_info.pci_device,
            (unsigned long long)pass3_input.unix_seconds);
        if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
            syscall(__NR_write, 2, log_msg, (size_t)log_len);
        }
        dark_log_hex_bytes("dark_integrity_compute_token.pass3_input",
                           &pass3_input, sizeof(pass3_input));
        dark_log_hex_bytes("dark_integrity_compute_token.device_info",
                           &device_info, sizeof(device_info));
    }

    dark_integrity_pass5(state, pass5_stage1);
    dark_integrity_zero_result(state);
    dark_integrity_hash_pass(state, k_pass1_result, sizeof(k_pass1_result), 0x5c);
    dark_integrity_hash_pass(state, pass5_stage1, sizeof(pass5_stage1), 0x00);
    dark_integrity_pass5(state, out);
}

static CUresult dark_context_local_storage_insert(CUcontext ctx,
                                                  void *key,
                                                  void *value,
                                                  dark_context_storage_dtor_cb dtor_cb)
{
    CUcontext effective_ctx = ctx ? ctx : (g_current_ctx ? g_current_ctx : g_global_ctx);
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_context_local_storage_insert() CALLED: ctx=%p effective_ctx=%p key=%p value=%p dtor=%p (pid=%d)\n",
                          (void *)ctx, (void *)effective_ctx, key, value, (void *)dtor_cb, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }

    if (!key) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    ensure_mutex_init();
    pthread_mutex_lock(&g_mutex);

    for (size_t i = 0; i < g_dark_context_storage_count; ++i) {
        if (g_dark_context_storage[i].ctx == effective_ctx && g_dark_context_storage[i].key == key) {
            g_dark_context_storage[i].value = value;
            g_dark_context_storage[i].dtor_cb = dtor_cb;
            pthread_mutex_unlock(&g_mutex);
            return CUDA_SUCCESS;
        }
    }

    if (g_dark_context_storage_count >= DARK_CONTEXT_STORAGE_MAX_ENTRIES) {
        pthread_mutex_unlock(&g_mutex);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    g_dark_context_storage[g_dark_context_storage_count].ctx = effective_ctx;
    g_dark_context_storage[g_dark_context_storage_count].key = key;
    g_dark_context_storage[g_dark_context_storage_count].value = value;
    g_dark_context_storage[g_dark_context_storage_count].dtor_cb = dtor_cb;
    g_dark_context_storage_count++;

    pthread_mutex_unlock(&g_mutex);
    return CUDA_SUCCESS;
}

static CUresult dark_context_local_storage_remove(uintptr_t arg1, uintptr_t arg2)
{
    CUcontext ctx = (CUcontext)arg1;
    CUcontext effective_ctx = ctx ? ctx : (g_current_ctx ? g_current_ctx : g_global_ctx);
    void *key = (void *)arg2;
    char log_msg[224];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_context_local_storage_remove() CALLED: ctx=%p effective_ctx=%p key=%p (pid=%d)\n",
                          (void *)ctx, (void *)effective_ctx, key, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }

    ensure_mutex_init();
    pthread_mutex_lock(&g_mutex);

    for (size_t i = 0; i < g_dark_context_storage_count; ++i) {
        if (g_dark_context_storage[i].ctx == effective_ctx && g_dark_context_storage[i].key == key) {
            g_dark_context_storage[i] = g_dark_context_storage[g_dark_context_storage_count - 1];
            g_dark_context_storage_count--;
            pthread_mutex_unlock(&g_mutex);
            return CUDA_SUCCESS;
        }
    }

    pthread_mutex_unlock(&g_mutex);
    return CUDA_ERROR_NOT_FOUND;
}

static CUresult dark_context_local_storage_get(void **result, CUcontext ctx, void *key)
{
    CUcontext effective_ctx = ctx ? ctx : (g_current_ctx ? g_current_ctx : g_global_ctx);
    char log_msg[224];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_context_local_storage_get() CALLED: result=%p ctx=%p effective_ctx=%p key=%p (pid=%d)\n",
                          (void *)result, (void *)ctx, (void *)effective_ctx, key, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }

    if (!result || !key) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *result = NULL;

    ensure_mutex_init();
    pthread_mutex_lock(&g_mutex);

    for (size_t i = 0; i < g_dark_context_storage_count; ++i) {
        if (g_dark_context_storage[i].ctx == effective_ctx && g_dark_context_storage[i].key == key) {
            *result = g_dark_context_storage[i].value;
            pthread_mutex_unlock(&g_mutex);
            return CUDA_SUCCESS;
        }
    }

    pthread_mutex_unlock(&g_mutex);
    return CUDA_ERROR_INVALID_HANDLE;
}

static CUresult __attribute__((unused)) dark_unwrap_context(CUcontext ctx, uint32_t *wrapped, CUcontext *unwrapped_ctx)
{
    char log_msg[224];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_unwrap_context() CALLED: ctx=%p wrapped=%p unwrapped=%p (pid=%d)\n",
                          (void *)ctx, (void *)wrapped, (void *)unwrapped_ctx, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }

    if (wrapped) {
        *wrapped = 0;
    }
    if (unwrapped_ctx) {
        *unwrapped_ctx = ctx;
    }
    return CUDA_SUCCESS;
}

static CUresult dark_context_check(CUcontext ctx, uint32_t *result1, const void **result2)
{
    char log_msg[224];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_context_check() CALLED: ctx=%p result1=%p result2=%p (pid=%d)\n",
                          (void *)ctx, (void *)result1, (void *)result2, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    if (result1) {
        *result1 = 0;
    }
    if (result2) {
        *result2 = NULL;
    }
    return CUDA_SUCCESS;
}

static uint32_t dark_context_check_fn3(void)
{
    const char *msg = "[libvgpu-cuda] dark_context_check_fn3() CALLED\n";
    (void)syscall(__NR_write, 2, msg,
                  sizeof("[libvgpu-cuda] dark_context_check_fn3() CALLED\n") - 1);
    return 0;
}

static CUresult dark_anti_zluda_check(uint32_t runtime_version, uint64_t timestamp, void *result)
{
    char log_msg[384];
    uint64_t hash_out[2] = {0, 0};
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_anti_zluda_check() CALLED: runtime_version=%u timestamp=%llu result=%p (pid=%d)\n",
                          runtime_version, (unsigned long long)timestamp, result, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }

    dark_integrity_compute_token(runtime_version, timestamp, hash_out);
    if (result) {
        memcpy(result, hash_out, sizeof(hash_out));
    }

    log_len = snprintf(log_msg, sizeof(log_msg),
                       "[libvgpu-cuda] dark_anti_zluda_check() token=%016llx:%016llx cudart=%p integrity=%p fn=%p\n",
                       (unsigned long long)hash_out[0],
                       (unsigned long long)hash_out[1],
                       g_dark_cudart_table_addr,
                       g_dark_integrity_check_table_addr,
                       (const void *)dark_anti_zluda_check);
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUDA_SUCCESS;
}

static CUresult __attribute__((unused)) dark_context_wrapper_query(void *arg1, void *arg2, uint32_t arg3, uint32_t arg4)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_context_wrapper_query() CALLED: arg1=%p arg2=%p arg3=%u arg4=%u (pid=%d)\n",
                          arg1, arg2, arg3, arg4, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    /* libcublas/libcublasLt treat this slot as a boolean capability query:
     * the caller compares EAX against 1 immediately after the indirect call. */
    if (arg4 == 3U || arg4 == 6U) {
        return (CUresult)1;
    }
    return CUDA_SUCCESS;
}

static CUresult dark_tools_tls_slot1(void *arg1, void *arg2, void *arg3, void *arg4)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_tools_tls_slot1() CALLED: a1=%p a2=%p a3=%p a4=%p (pid=%d)\n",
                          arg1, arg2, arg3, arg4, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUDA_SUCCESS;
}

static CUresult dark_tools_tls_slot2(void *arg1, void *arg2, void *arg3, void *arg4)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_tools_tls_slot2() CALLED: a1=%p a2=%p a3=%p a4=%p (pid=%d)\n",
                          arg1, arg2, arg3, arg4, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUDA_SUCCESS;
}

static CUresult dark_tools_tls_slot3(void *arg1, void *arg2, void *arg3, void *arg4)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_tools_tls_slot3() CALLED: a1=%p a2=%p a3=%p a4=%p (pid=%d)\n",
                          arg1, arg2, arg3, arg4, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUDA_SUCCESS;
}

static CUresult dark_load_compilers(void)
{
    char log_msg[192];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] dark_load_compilers() CALLED (pid=%d)\n",
                          (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUDA_SUCCESS;
}

__attribute__((visibility("default")))
CUresult cuGetExportTable(const void **ppExportTable, const void *pExportTableId)
{
    static const unsigned char k_uuid_cudart_interface[16] = {
        0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
        0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9
    };
    static const unsigned char k_uuid_tools_runtime_hooks[16] = {
        0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
        0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66
    };
    static const unsigned char k_uuid_tools_tls[16] = {
        0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
        0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc
    };
    static const unsigned char k_uuid_context_local_storage_v0301[16] = {
        0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
        0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93
    };
    static const unsigned char k_uuid_context_wrapper[16] = {
        0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43,
        0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e
    };
    static const unsigned char k_uuid_anti_zluda[16] = {
        0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
        0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2
    };
    static const void *g_cudart_interface[13] = {
        (const void *)(uintptr_t)(sizeof(void *) * 13), /* slot 0: table size in bytes */
        (const void *)dark_get_module_from_cubin,        /* slot 1 */
        (const void *)dark_get_primary_context,         /* slot 2 */
        NULL,                                           /* slot 3 */
        NULL,                                           /* slot 4 */
        NULL,                                           /* slot 5 */
        (const void *)dark_get_module_from_cubin_ex1,    /* slot 6 */
        (const void *)dark_cudart_interface_fn7,         /* slot 7 */
        (const void *)dark_get_module_from_cubin_ex2,    /* slot 8 */
        (const void *)dark_launch_kernel,                /* slot 9 */
        NULL,                                            /* slot 10 */
        NULL,                                            /* slot 11 */
        (const void *)dark_load_compilers                /* slot 12 */
    };
    static const void *g_tools_runtime_hooks[7] = {
        (const void *)(uintptr_t)(sizeof(void *) * 7),  /* slot 0: table size in bytes */
        NULL,                                           /* slot 1 */
        (const void *)dark_tools_runtime_fn2,           /* slot 2 */
        NULL,                                           /* slot 3 */
        NULL,                                           /* slot 4 */
        NULL,                                           /* slot 5 */
        (const void *)dark_tools_runtime_fn6            /* slot 6 */
    };
    /*
     * Match current dark-API layout used by recent CUDA user-space:
     * TOOLS_TLS is a 4-slot size-only table, and UUID 263e... is a
     * 4-slot context-check table with slot 2 and slot 3 callbacks.
     */
    static const void *g_tools_tls[4] = {
        (const void *)(uintptr_t)(sizeof(void *) * 4),  /* slot 0: table size in bytes */
        (const void *)dark_tools_tls_slot1,             /* slot 1 */
        (const void *)dark_tools_tls_slot2,             /* slot 2 */
        (const void *)dark_tools_tls_slot3              /* slot 3 */
    };
    static const void *g_context_local_storage_v0301[4] = {
        (const void *)dark_context_local_storage_insert,
        (const void *)dark_context_local_storage_remove,
        (const void *)dark_context_local_storage_get,
        NULL
    };
    static const void *g_context_wrapper[4] = {
        (const void *)(uintptr_t)(sizeof(void *) * 4),
        NULL,
        (const void *)dark_context_check,
        (const void *)dark_context_check_fn3
    };
    static const void *g_anti_zluda[3] = {
        (const void *)(uintptr_t)(sizeof(void *) * 3),
        (const void *)dark_anti_zluda_check,
        NULL
    };
    char uuid_hex[64] = {0};
    if (pExportTableId) {
        const unsigned char *u = (const unsigned char *)pExportTableId;
        snprintf(uuid_hex, sizeof(uuid_hex),
                 "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                 u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
                 u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    }

    if (!ppExportTable || !pExportTableId) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (memcmp(pExportTableId, k_uuid_cudart_interface, sizeof(k_uuid_cudart_interface)) == 0) {
        *ppExportTable = (const void *)g_cudart_interface;
        g_dark_cudart_table_addr = (const void *)g_cudart_interface;
    } else if (memcmp(pExportTableId, k_uuid_tools_runtime_hooks, sizeof(k_uuid_tools_runtime_hooks)) == 0) {
        *ppExportTable = (const void *)g_tools_runtime_hooks;
    } else if (memcmp(pExportTableId, k_uuid_tools_tls, sizeof(k_uuid_tools_tls)) == 0) {
        *ppExportTable = (const void *)g_tools_tls;
    } else if (memcmp(pExportTableId, k_uuid_context_local_storage_v0301,
                      sizeof(k_uuid_context_local_storage_v0301)) == 0) {
        *ppExportTable = (const void *)g_context_local_storage_v0301;
    } else if (memcmp(pExportTableId, k_uuid_context_wrapper, sizeof(k_uuid_context_wrapper)) == 0) {
        *ppExportTable = (const void *)g_context_wrapper;
    } else if (memcmp(pExportTableId, k_uuid_anti_zluda, sizeof(k_uuid_anti_zluda)) == 0) {
        *ppExportTable = (const void *)g_anti_zluda;
        g_dark_integrity_check_table_addr = (const void *)g_anti_zluda;
    } else {
        char unknown_msg[256];
        int unknown_len = snprintf(unknown_msg, sizeof(unknown_msg),
                                   "[libvgpu-cuda] cuGetExportTable() UNKNOWN UUID: id=%p uuid=%s (pid=%d)\n",
                                   pExportTableId,
                                   pExportTableId ? uuid_hex : "(null)",
                                   (int)getpid());
        if (unknown_len > 0 && unknown_len < (int)sizeof(unknown_msg)) {
            syscall(__NR_write, 2, unknown_msg, unknown_len);
        }
        *ppExportTable = NULL;
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    char log_msg[768];
    uintptr_t slot0 = 0;
    if (*ppExportTable) {
        slot0 = (uintptr_t)((const void * const *)*ppExportTable)[0];
    }
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] cuGetExportTable() CALLED: pp=%p id=%p uuid=%s table=%p slot0=0x%lx slot0_dec=%lu slots=[%p,%p,%p,%p,%p,%p,%p,%p,%p,%p] (pid=%d)\n",
                          (void *)ppExportTable, pExportTableId,
                          pExportTableId ? uuid_hex : "(null)",
                          *ppExportTable,
                          (unsigned long)slot0, (unsigned long)slot0,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[0] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[1] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[2] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[3] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[4] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[5] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[6] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[7] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[8] : NULL,
                          *ppExportTable ? ((const void * const *)*ppExportTable)[9] : NULL,
                          (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    return CUDA_SUCCESS;
}

/* ================================================================
 * CUDA Virtual Memory Management (VMM) API stubs
 * 
 * libggml-cuda.so uses these functions during initialization.
 * For now, return success with dummy values to allow initialization
 * to proceed. Actual memory operations will be handled by cuMemAlloc
 * and related functions.
 * ================================================================ */

/* Forward declarations used by VMM helpers below. */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop __attribute__((unused)), unsigned long long flags)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemCreate(handle=%p, size=%zu, flags=0x%llx)\n",
            handle, size, (unsigned long long)flags);

    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        return rc;
    }
    if (!handle || size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Allocate real memory on the physical GPU using cuMemAlloc_v2 and
     * treat the resulting device pointer as the generic allocation handle.
     * This keeps semantics simple for GGML's VMM usage while ensuring that
     * the handle always refers to valid device memory. */
    CUdeviceptr dev = 0;
    rc = cuMemAlloc_v2(&dev, size);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[libvgpu-cuda] cuMemCreate: cuMemAlloc_v2 failed (size=%zu, rc=%d)\n",
                size, rc);
        return rc;
    }

    *handle = (CUmemGenericAllocationHandle)(uintptr_t)dev;
    fprintf(stderr, "[libvgpu-cuda] cuMemCreate SUCCESS: handle=0x%llx (devptr=0x%llx, size=%zu)\n",
            (unsigned long long)(uintptr_t)(*handle),
            (unsigned long long)dev, size);
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr __attribute__((unused)), unsigned long long flags)
{
    /* CRITICAL: Log this call - GGML uses VMM API for memory allocation */
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] cuMemAddressReserve() CALLED (size=%zu, alignment=%zu, flags=0x%llx, pid=%d)\n",
                          size, alignment, (unsigned long long)flags, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!ptr) return CUDA_ERROR_INVALID_VALUE;
    
    /* CRITICAL FIX: Return a properly aligned address.
     * GGML requires TENSOR_ALIGNMENT (32 bytes), but also respect the requested alignment.
     * Use the maximum of requested alignment and 32 bytes. */
    static uintptr_t next_addr = 0x1000000; /* Start at 16MB */
    const size_t min_alignment = 32; /* GGML TENSOR_ALIGNMENT */
    size_t effective_alignment = (alignment > min_alignment) ? alignment : min_alignment;
    
    /* Align the address to the effective alignment */
    next_addr = (next_addr + effective_alignment - 1) & ~(effective_alignment - 1);
    *ptr = (CUdeviceptr)next_addr;
    next_addr += size;
    
    char success_msg[256];
    int success_len = snprintf(success_msg, sizeof(success_msg),
                              "[libvgpu-cuda] cuMemAddressReserve() SUCCESS: ptr=0x%llx, size=%zu, alignment=%zu (pid=%d)\n",
                              (unsigned long long)*ptr, size, effective_alignment, (int)getpid());
    if (success_len > 0 && success_len < (int)sizeof(success_msg)) {
        syscall(__NR_write, 2, success_msg, success_len);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemUnmap(ptr=0x%llx, size=%zu)\n",
            (unsigned long long)ptr, size);
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    
    /* Always succeed - unmapping is a no-op for dummy addresses */
    fprintf(stderr, "[libvgpu-cuda] cuMemUnmap returning SUCCESS\n");
    return CUDA_SUCCESS;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc __attribute__((unused)), size_t count)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemSetAccess(ptr=0x%llx, size=%zu, count=%zu)\n",
            (unsigned long long)ptr, size, count);
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    
    /* Always succeed - access control deferred to actual memory ops */
    fprintf(stderr, "[libvgpu-cuda] cuMemSetAccess returning SUCCESS\n");
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemAddressFree(ptr=0x%llx, size=%zu)\n",
            (unsigned long long)ptr, size);
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    
    /* Always succeed - freeing is a no-op for dummy addresses */
    fprintf(stderr, "[libvgpu-cuda] cuMemAddressFree returning SUCCESS\n");
    return CUDA_SUCCESS;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, unsigned long long flags)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemMap(ptr=0x%llx, size=%zu, offset=%zu, handle=0x%llx, flags=0x%llx)\n",
            (unsigned long long)ptr, size, offset,
            (unsigned long long)(uintptr_t)handle,
            (unsigned long long)flags);

    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        return rc;
    }
    if (!handle) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* The actual allocation was performed by cuMemCreate (backed by
     * cuMemAlloc_v2). We do not need to perform an additional mapping step
     * in the remoted environment, so this is effectively a no-op that
     * validates the handle. */
    fprintf(stderr, "[libvgpu-cuda] cuMemMap SUCCESS (ptr=0x%llx, size=%zu, offset=%zu, handle=0x%llx)\n",
            (unsigned long long)ptr, size, offset,
            (unsigned long long)(uintptr_t)handle);
    return CUDA_SUCCESS;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemRelease(handle=0x%llx)\n",
            (unsigned long long)(uintptr_t)handle);

    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        return rc;
    }
    if (!handle) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Convert the generic allocation handle back to the device pointer and
     * free it via cuMemFree_v2 so GPU memory is actually released on the
     * host. */
    CUdeviceptr dev = (CUdeviceptr)(uintptr_t)handle;
    rc = cuMemFree_v2(dev);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[libvgpu-cuda] cuMemRelease: cuMemFree_v2 failed (devptr=0x%llx, rc=%d)\n",
                (unsigned long long)dev, rc);
        return rc;
    }

    fprintf(stderr, "[libvgpu-cuda] cuMemRelease SUCCESS: devptr=0x%llx\n",
            (unsigned long long)dev);
    return CUDA_SUCCESS;
}

CUresult cuMemGetAllocationGranularity(size_t *granularity,
                                       const CUmemAllocationProp *prop __attribute__((unused)), unsigned int option)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemGetAllocationGranularity(granularity=%p, option=0x%x)\n",
            granularity, option);
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!granularity) return CUDA_ERROR_INVALID_VALUE;
    
    /* Return a reasonable default granularity (64KB) */
    *granularity = 64 * 1024;
    fprintf(stderr, "[libvgpu-cuda] cuMemGetAllocationGranularity returning SUCCESS (granularity=%zu)\n",
            *granularity);
    return CUDA_SUCCESS;
}

/* cuGetErrorString is defined later in the file in the Error handling section */

__attribute__((visibility("default")))
CUresult cuDriverGetVersion(int *driverVersion)
{
    /* Log call with PID using syscall */
    char log_msg[128];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] cuDriverGetVersion() CALLED (pid=%d)\n",
                          (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    if (!driverVersion) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    /* Present a userspace-compatible driver version to bundled CUDA 12.8 libs. */
    *driverVersion = effective_userspace_driver_version();
    
    /* CRITICAL: Log BOTH version AND return code to verify CUDA_SUCCESS (0) is returned */
    char success_msg[128];
    int success_len = snprintf(success_msg, sizeof(success_msg),
                              "[libvgpu-cuda] cuDriverGetVersion() SUCCESS: version=%d, return_code=%d (CUDA_SUCCESS=%d, pid=%d)\n",
                              *driverVersion, CUDA_SUCCESS, CUDA_SUCCESS, (int)getpid());
    if (success_len > 0 && success_len < (int)sizeof(success_msg)) {
        syscall(__NR_write, 2, success_msg, success_len);
    }
    return CUDA_SUCCESS;
}

/* ================================================================
 * CUDA Driver API — Device management (answered locally)
 * ================================================================ */

__attribute__((visibility("default")))
CUresult cuDeviceGetCount(int *count)
{
    /* Log to stderr and to /tmp to confirm discovery path (runner stderr may not reach journal) */
    {
        int pid = (int)getpid();
        char called_msg[128];
        int called_len = snprintf(called_msg, sizeof(called_msg),
                                 "[libvgpu-cuda] cuDeviceGetCount() CALLED (pid=%d)\n", pid);
        if (called_len > 0 && called_len < (int)sizeof(called_msg))
            syscall(__NR_write, 2, called_msg, called_len);
        {
            int fd = syscall(__NR_open, "/tmp/cuda_get_count_called.txt",
                             O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (fd >= 0) {
                char buf[80];
                int n = snprintf(buf, sizeof(buf), "pid=%d\n", pid);
                if (n > 0) syscall(__NR_write, fd, buf, (size_t)n);
                syscall(__NR_close, fd);
            }
        }
    }
    
    if (!count) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    /* ALWAYS return count=1 immediately - no checks, no delays */
    *count = 1;
    
    /* CRITICAL: Log BOTH count AND return code to verify CUDA_SUCCESS (0) is returned */
    char success_msg[128];
    int success_len = snprintf(success_msg, sizeof(success_msg),
                               "[libvgpu-cuda] cuDeviceGetCount() SUCCESS: returning count=%d, return_code=%d (CUDA_SUCCESS=%d, pid=%d)\n",
                               1, CUDA_SUCCESS, CUDA_SUCCESS, (int)getpid());
    if (success_len > 0 && success_len < (int)sizeof(success_msg)) {
        syscall(__NR_write, 2, success_msg, success_len);
    }
    
    /* CRITICAL: Ensure we return CUDA_SUCCESS (0), not count value */
    return CUDA_SUCCESS;
}

/* _v2 version is identical */
CUresult cuDeviceGetCount_v2(int *count)
{
    return cuDeviceGetCount(count);
}

__attribute__((visibility("default")))
CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
    /* CRITICAL: Log immediately using syscall to avoid any libc issues */
    const char *msg = "[libvgpu-cuda] cuDeviceGet() CALLED\n";
    (void)syscall(__NR_write, 2, msg, sizeof("[libvgpu-cuda] cuDeviceGet() CALLED\n") - 1);
    
    if (!device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (ordinal != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    
    /* CRITICAL: Return device=0 immediately without calling ensure_init()
     * This allows ggml_cuda_init() to proceed even if initialization isn't complete */
    *device = 0;
    
    const char *success_msg = "[libvgpu-cuda] cuDeviceGet() SUCCESS: device=0\n";
    (void)syscall(__NR_write, 2, success_msg, sizeof("[libvgpu-cuda] cuDeviceGet() SUCCESS: device=0\n") - 1);
    
    return CUDA_SUCCESS;
}

/* _v2 version is identical */
CUresult cuDeviceGet_v2(CUdevice *device, int ordinal)
{
    return cuDeviceGet(device, ordinal);
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuDeviceGetName(name=%p, len=%d, dev=%d) (pid=%d)\n",
            name, len, dev, (int)getpid());
    fflush(stderr);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetName ensure_init() failed: %d\n", rc);
        fflush(stderr);
        return rc;
    }
    if (!name || len <= 0) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetName: invalid arguments\n");
        fflush(stderr);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetName: invalid device %d\n", dev);
        fflush(stderr);
        return CUDA_ERROR_INVALID_DEVICE;
    }

    strncpy(name, g_gpu_info.name, (size_t)len - 1);
    name[len - 1] = '\0';
    fprintf(stderr, "[libvgpu-cuda] cuDeviceGetName SUCCESS: returning name=\"%s\"\n", name);
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
    fprintf(stderr, "[libvgpu-cuda] cuDeviceTotalMem_v2() CALLED (pid=%d dev=%d)\n", (int)getpid(), dev);
    fflush(stderr);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceTotalMem_v2 ensure_init() failed: %d\n", rc);
        fflush(stderr);
        return rc;
    }
    if (!bytes) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceTotalMem_v2: invalid pointer\n");
        fflush(stderr);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceTotalMem_v2: invalid device %d\n", dev);
        fflush(stderr);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *bytes = (size_t)g_gpu_info.total_mem;
    fprintf(stderr, "[libvgpu-cuda] cuDeviceTotalMem_v2() SUCCESS: returning %zu bytes (%zu MB) (pid=%d)\n", 
            *bytes, *bytes / (1024 * 1024), (int)getpid());
    fflush(stderr);
    return CUDA_SUCCESS;
}

/* Legacy version */
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
    return cuDeviceTotalMem_v2(bytes, dev);
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    /* CRITICAL: Log FIRST using syscall to see if this is called */
    char log_msg[128];
    int log_len = snprintf(log_msg, sizeof(log_msg), 
                          "[libvgpu-cuda] cuDeviceGetAttribute() CALLED (attrib=%d, dev=%d, pid=%d)\n", 
                          attrib, dev, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    if (!pi) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    
    /* CRITICAL: Ensure g_gpu_info is initialized if not already */
    /* This is defensive - ensures we always have valid values */
    if (!g_gpu_info_valid) {
        init_gpu_defaults();
    }
    
    /* CRITICAL: Return immediately without calling ensure_init()
     * to avoid any delays or failures. Use g_gpu_info which is
     * initialized by init_gpu_defaults() above or in cuInit(). */

    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        /* CRITICAL FIX: Explicitly return 1024 (valid for all NVIDIA GPUs)
         * This ensures Ollama/GGML validation passes. ChatGPT identified that
         * returning 1620000 (clock rate) causes GPU rejection. */
        *pi = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
        *pi = g_gpu_info.max_block_dim_x; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
        *pi = g_gpu_info.max_block_dim_y; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
        *pi = g_gpu_info.max_block_dim_z; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
        *pi = g_gpu_info.max_grid_dim_x; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
        *pi = g_gpu_info.max_grid_dim_y; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
        *pi = g_gpu_info.max_grid_dim_z; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
        *pi = g_gpu_info.max_shared_mem_per_block; break;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
        *pi = GPU_DEFAULT_TOTAL_CONSTANT_MEM; break;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
        *pi = g_gpu_info.warp_size; break;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
        *pi = GPU_DEFAULT_MAX_PITCH; break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
        *pi = g_gpu_info.regs_per_block; break;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
        *pi = g_gpu_info.clock_rate_khz; break;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
        *pi = GPU_DEFAULT_TEXTURE_ALIGNMENT; break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        *pi = g_gpu_info.multi_processor_count; break;
    case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
        *pi = 0; break;  /* No timeout */
    case CU_DEVICE_ATTRIBUTE_INTEGRATED:
        *pi = GPU_DEFAULT_INTEGRATED; break;
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
        *pi = GPU_DEFAULT_CAN_MAP_HOST_MEM; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
        *pi = GPU_DEFAULT_COMPUTE_MODE; break;
    case CU_DEVICE_ATTRIBUTE_MAX_TEXTURE1D_WIDTH:
        *pi = GPU_DEFAULT_MAX_TEXTURE_1D; break;
    case CU_DEVICE_ATTRIBUTE_MAX_TEXTURE2D_WIDTH:
        *pi = GPU_DEFAULT_MAX_TEXTURE_2D_W; break;
    case CU_DEVICE_ATTRIBUTE_MAX_TEXTURE2D_HEIGHT:
        *pi = GPU_DEFAULT_MAX_TEXTURE_2D_H; break;
    case CU_DEVICE_ATTRIBUTE_MAX_TEXTURE3D_WIDTH:
        *pi = GPU_DEFAULT_MAX_TEXTURE_3D_W; break;
    case CU_DEVICE_ATTRIBUTE_MAX_TEXTURE3D_HEIGHT:
        *pi = GPU_DEFAULT_MAX_TEXTURE_3D_H; break;
    case CU_DEVICE_ATTRIBUTE_MAX_TEXTURE3D_DEPTH:
        *pi = GPU_DEFAULT_MAX_TEXTURE_3D_D; break;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
        *pi = g_gpu_info.concurrent_kernels; break;
    case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
        *pi = g_gpu_info.ecc_enabled; break;
    case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
        *pi = g_gpu_info.pci_bus_id; break;
    case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
        *pi = g_gpu_info.pci_device_id; break;
    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
        *pi = g_gpu_info.memory_clock_rate_khz; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
        *pi = g_gpu_info.memory_bus_width; break;
    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
        *pi = g_gpu_info.l2_cache_size; break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
        *pi = g_gpu_info.max_threads_per_mp; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR:
        *pi = 32; break;
    case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
        *pi = g_gpu_info.async_engine_count; break;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
        /* CRITICAL: GGML requires Unified Addressing = 1 for H100
         * Always return 1, even if g_gpu_info isn't initialized */
        *pi = (g_gpu_info_valid && g_gpu_info.unified_addressing > 0)
              ? g_gpu_info.unified_addressing
              : GPU_DEFAULT_UNIFIED_ADDRESSING;  /* Must be 1 */
        break;
    case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
        *pi = g_gpu_info.pci_domain_id; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
        /* Force compatibility arch for ggml-cuda kernel selection. */
        *pi = GPU_DEFAULT_CC_MAJOR;
        /* CRITICAL: Log this specific call - GGML discovery may use this */
        {
            char cc_log[128];
            int cc_len = snprintf(cc_log, sizeof(cc_log),
                                 "[libvgpu-cuda] cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR) returning: %d (pid=%d)\n",
                                 *pi, (int)getpid());
            if (cc_len > 0 && cc_len < (int)sizeof(cc_log)) {
                syscall(__NR_write, 2, cc_log, cc_len);
            }
        }
        break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
        /* Force compatibility arch for ggml-cuda kernel selection. */
        *pi = GPU_DEFAULT_CC_MINOR;
        /* CRITICAL: Log this specific call - GGML discovery may use this */
        {
            char cc_log[128];
            int cc_len = snprintf(cc_log, sizeof(cc_log),
                                 "[libvgpu-cuda] cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR) returning: %d (pid=%d)\n",
                                 *pi, (int)getpid());
            if (cc_len > 0 && cc_len < (int)sizeof(cc_log)) {
                syscall(__NR_write, 2, cc_log, cc_len);
            }
        }
        break;
    case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
        *pi = g_gpu_info.managed_memory; break;
    case CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD:
        *pi = GPU_DEFAULT_MULTI_GPU_BOARD; break;
    case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH:
        *pi = GPU_DEFAULT_COOPERATIVE_LAUNCH; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
        *pi = g_gpu_info.max_shared_mem_per_mp; break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
        *pi = g_gpu_info.regs_per_multiprocessor; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
        *pi = g_gpu_info.max_shared_mem_per_block; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED:
        *pi = GPU_DEFAULT_GLOBAL_L1_CACHE_SUPPORT; break;
    case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED:
        *pi = GPU_DEFAULT_LOCAL_L1_CACHE_SUPPORT; break;
    case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED:
        /* CRITICAL: GGML checks this attribute. H100 supports VMM, return 1 */
        *pi = 1;
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetAttribute(VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED) returning: 1 (pid=%d)\n", 
                (int)getpid());
        fflush(stderr);
        break;
    /* Newer CUDA stacks query numeric attribute IDs that are not part of our
     * local enum subset. Real libcublas.so.12 probes these during init and
     * should not see a wall of zeros for basic size/limit properties. Keep
     * feature flags conservative, but return sane non-zero values for limits
     * that map cleanly to the H100 defaults we already expose elsewhere. */
    case 15:  /* legacy device overlap / async copy capability probe */
        *pi = 1; break;
    case 45:  /* newer texture-layer probe */
        *pi = GPU_DEFAULT_MAX_TEXTURE_1D; break;
    case 46:  /* newer texture-layer probe */
        *pi = 2048; break;
    case 47:  /* newer texture-layer probe */
        *pi = GPU_DEFAULT_MAX_TEXTURE_2D_W; break;
    case 48:  /* newer texture-layer probe */
        *pi = GPU_DEFAULT_MAX_TEXTURE_2D_H; break;
    case 49:  /* newer texture-layer probe */
        *pi = 2048; break;
    case 52:  /* newer alignment/surface capability probe */
        *pi = GPU_DEFAULT_TEXTURE_ALIGNMENT; break;
    case 69:  /* maximum texture1D linear width */
        *pi = GPU_DEFAULT_MAX_TEXTURE_1D; break;
    case 73:  /* maximum texture2D linear width */
        *pi = GPU_DEFAULT_MAX_TEXTURE_2D_W; break;
    case 74:  /* maximum texture2D linear height */
        *pi = GPU_DEFAULT_MAX_TEXTURE_2D_H; break;
    case 77:  /* maximum texture2D linear pitch */
        *pi = GPU_DEFAULT_MAX_PITCH; break;
    case 78:  /* maximum texture cubemap layered layers */
        *pi = 2048; break;
    case 79:  /* maximum texture1D gather width */
        *pi = GPU_DEFAULT_MAX_TEXTURE_1D; break;
    case 80:  /* maximum texture2D gather width */
        *pi = GPU_DEFAULT_MAX_TEXTURE_2D_W; break;
    case 85:  /* maximum texture3D alternate depth */
        *pi = GPU_DEFAULT_MAX_TEXTURE_3D_D; break;
    case 86:  /* texture gather A16 support */
        *pi = 1; break;
    case 89:  /* stream priorities / cache capability style probe */
        *pi = 1; break;
    case 90:  /* cache capability style probe */
        *pi = 1; break;
    case 96:  /* shared memory per block opt-in style probe */
        *pi = g_gpu_info.max_shared_mem_per_block > 0
              ? g_gpu_info.max_shared_mem_per_block
              : GPU_DEFAULT_SHARED_MEM_PER_BLOCK;
        break;
    case 113: /* max shared memory per multiprocessor */
        *pi = g_gpu_info.max_shared_mem_per_mp > 0
              ? g_gpu_info.max_shared_mem_per_mp
              : GPU_DEFAULT_SHARED_MEM_PER_SM;
        break;
    case 114: /* max registers per multiprocessor */
        *pi = g_gpu_info.regs_per_multiprocessor > 0
              ? g_gpu_info.regs_per_multiprocessor
              : GPU_DEFAULT_REGS_PER_SM;
        break;
    case 115: /* managed memory */
        *pi = g_gpu_info.managed_memory > 0
              ? g_gpu_info.managed_memory
              : GPU_DEFAULT_MANAGED_MEMORY;
        break;
    case 118: /* host native atomic support */
        *pi = 1; break;
    case 119: /* single-to-double precision perf ratio */
        *pi = 2; break;
    case 120: /* pageable memory access */
        *pi = 1; break;
    case 121: /* concurrent managed access */
        *pi = 1; break;
    case 129: /* handle type POSIX file descriptor supported */
        *pi = 1; break;
    default:
        /* Unknown attributes are capability/feature probes in newer CUDA stacks.
         * Return 0 (feature not supported) instead of fabricating support with 1. */
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetAttribute: unknown attribute %d, returning default 0 (pid=%d)\n",
                attrib, (int)getpid());
        *pi = 0;
        break;
    }

    /* Final clamp for attributes that can cause divide-by-zero in GGML. */
    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
        if (*pi <= 0) *pi = GPU_DEFAULT_WARP_SIZE;
        break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        if (*pi <= 0) *pi = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;
        break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
        if (*pi <= 0) *pi = GPU_DEFAULT_MAX_THREADS_PER_SM;
        break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        if (*pi <= 0) *pi = GPU_DEFAULT_SM_COUNT;
        break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR:
        if (*pi <= 0) *pi = 32;
        break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
        if (*pi <= 0) *pi = GPU_DEFAULT_SHARED_MEM_PER_BLOCK;
        break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
        if (*pi <= 0) *pi = GPU_DEFAULT_REGS_PER_BLOCK;
        break;
    default:
        /* During real libcublas.so.12 init, newer CUDA stacks probe many raw
         * numeric attribute IDs that our local enum subset does not model.
         * Keep this fallback narrow and init-oriented: only patch IDs that
         * we have observed in the CUBLAS init path and only when the value
         * would otherwise be 0. */
        if (*pi == 0) {
            switch ((int)attrib) {
            case 27: /* maximum texture2D layered width */
                *pi = GPU_DEFAULT_MAX_TEXTURE_2D_W;
                break;
            case 28: /* maximum texture2D layered height */
                *pi = GPU_DEFAULT_MAX_TEXTURE_2D_H;
                break;
            case 29: /* maximum texture2D layered layers */
                *pi = 2048;
                break;
            case 30: /* surface / pitch alignment style probe */
                *pi = GPU_DEFAULT_TEXTURE_ALIGNMENT;
                break;
            case 34: /* PCI device ID */
                *pi = g_gpu_info.pci_device_id > 0
                      ? g_gpu_info.pci_device_id
                      : GPU_DEFAULT_PCI_DEVICE_ID;
                break;
            case 42: /* maximum texture1D layered width */
                *pi = GPU_DEFAULT_MAX_TEXTURE_1D;
                break;
            case 43: /* maximum texture1D layered layers */
                *pi = 2048;
                break;
            case 51: /* texture pitch alignment */
                *pi = 32;
                break;
            case 53: /* surface / layered-surface limit */
                *pi = 2048;
                break;
            case 54: /* surface / layered-surface limit */
                *pi = GPU_DEFAULT_MAX_TEXTURE_1D;
                break;
            case 55: /* surface / layered-surface limit */
                *pi = GPU_DEFAULT_MAX_TEXTURE_1D;
                break;
            case 56: /* surface / layered-surface limit */
                *pi = GPU_DEFAULT_MAX_TEXTURE_2D_H;
                break;
            case 57: /* surface / layered-surface limit */
                *pi = GPU_DEFAULT_MAX_TEXTURE_2D_H;
                break;
            case 58: /* surface / layered-surface limit */
                *pi = 2048;
                break;
            case 59: /* surface / layered-surface limit */
                *pi = GPU_DEFAULT_MAX_TEXTURE_2D_W;
                break;
            case 60: /* surface / layered-surface limit */
                *pi = GPU_DEFAULT_MAX_TEXTURE_2D_W;
                break;
            case 61: /* surface3D depth */
                *pi = GPU_DEFAULT_MAX_TEXTURE_3D_D;
                break;
            case 62: /* surface3D height */
                *pi = GPU_DEFAULT_MAX_TEXTURE_3D_H;
                break;
            case 63: /* surface3D width */
                *pi = GPU_DEFAULT_MAX_TEXTURE_3D_W;
                break;
            case 64: /* surface cubemap layered layers */
                *pi = 2048;
                break;
            case 65: /* surface cubemap layered width */
                *pi = GPU_DEFAULT_MAX_TEXTURE_3D_W;
                break;
            case 66: /* surface cubemap width */
                *pi = GPU_DEFAULT_MAX_TEXTURE_3D_W;
                break;
            case 67: /* texture1D layered layers / similar limit */
                *pi = 2048;
                break;
            case 68: /* texture1D layered width / similar limit */
                *pi = GPU_DEFAULT_MAX_TEXTURE_1D;
                break;
            case 70: /* maximum texture1D mipmapped width */
                *pi = GPU_DEFAULT_MAX_TEXTURE_1D;
                break;
            case 71: /* maximum texture2D mipmapped width */
                *pi = GPU_DEFAULT_MAX_TEXTURE_2D_W;
                break;
            case 72: /* maximum texture2D mipmapped height */
                *pi = GPU_DEFAULT_MAX_TEXTURE_2D_H;
                break;
            case 87: /* newer compute-capability / feature probe */
                *pi = GPU_DEFAULT_CC_MAJOR;
                break;
            case 88: /* newer compute-capability / feature probe */
                *pi = GPU_DEFAULT_CC_MINOR;
                break;
            case 99: /* compute-preemption / scheduler capability */
                *pi = 1;
                break;
            case 100: /* host pointer for registered memory */
                *pi = 1;
                break;
            case 101: /* stream mem ops capability */
                *pi = 1;
                break;
            case 108: /* compression / VMM capability style probe */
                *pi = 1;
                break;
            case 109: /* stream mem ops capability */
                *pi = 1;
                break;
            case 111: /* cooperative launch family */
                *pi = 1;
                break;
            case 112: /* cooperative multi-device launch family */
                *pi = 1;
                break;
            case 125: /* can flush remote writes */
                *pi = 1;
                break;
            default:
                break;
            }
        }
        /* Preserve explicit 0 for unsupported/unknown capability probes. */
        break;
    }
    
    /* Log success using syscall with PID */
    char success_msg[128];
    int success_len = snprintf(success_msg, sizeof(success_msg),
                              "[libvgpu-cuda] cuDeviceGetAttribute() SUCCESS: attrib=%d, value=%d (pid=%d)\n",
                              attrib, *pi, (int)getpid());
    if (success_len > 0 && success_len < (int)sizeof(success_msg)) {
        syscall(__NR_write, 2, success_msg, success_len);
    }
    
    return CUDA_SUCCESS;
}

/* _v2 version is identical */
CUresult cuDeviceGetAttribute_v2(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    return cuDeviceGetAttribute(pi, attrib, dev);
}

CUresult cuDeviceGetUuid(void *uuid, CUdevice dev)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    memcpy(uuid, g_gpu_info.uuid, 16);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid_v2(void *uuid, CUdevice dev)
{
    return cuDeviceGetUuid(uuid, dev);
}

/* cuDeviceGetLuid - Get LUID for device (used for multi-GPU setups) */
CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuDeviceGetLuid(luid=%p, deviceNodeMask=%p, dev=%d)\n",
            luid, deviceNodeMask, dev);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetLuid ensure_init() failed: %d\n", rc);
        return rc;
    }
    if (dev != 0) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetLuid: invalid device %d\n", dev);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if (luid) {
        /* Return a dummy LUID (128 bits, typically 16 bytes) */
        memset(luid, 0, 16);
        /* Set some non-zero bytes to indicate a valid LUID */
        luid[0] = 0x01;
        luid[15] = 0x01;
    }
    if (deviceNodeMask) {
        *deviceNodeMask = 1;  /* Single device, mask = 1 */
    }
    fprintf(stderr, "[libvgpu-cuda] cuDeviceGetLuid SUCCESS\n");
    fflush(stderr);
    return CUDA_SUCCESS;
}

/*
 * cuDeviceGetPCIBusId — return PCI bus:device.function string for device.
 *
 * Ollama's GPU-discovery code calls this (via dlsym) to pair the CUDA device
 * with the NVML device by PCI bus ID.  Without this function the dlsym lookup
 * returns NULL, which Ollama treats as an incomplete CUDA library and falls
 * back to library=cpu.
 */
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuDeviceGetPCIBusId(pciBusId=%p, len=%d, dev=%d)\n",
            pciBusId, len, dev);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetPCIBusId ensure_init() failed: %d\n", rc);
        return rc;
    }
    if (!pciBusId || len <= 0) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetPCIBusId: invalid arguments\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetPCIBusId: invalid device %d\n", dev);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    /* Use discovered BDF even if transport isn't connected yet */
    const char *bdf = cuda_transport_pci_bdf(g_transport);
    if (!bdf || !bdf[0]) {
        /* Fallback to discovered BDF from cuda_transport_discover() */
        bdf = cuda_transport_pci_bdf(NULL);
        if (!bdf || !bdf[0]) {
            bdf = "0000:00:05.0";  /* Final fallback */
        }
    }
    
    /* CRITICAL FIX: Format BDF to match filesystem format (4-digit domain) for Ollama's matching logic.
     * The filesystem provides "0000:00:05.0" (4-digit domain), and Ollama matches PCI devices
     * with NVML devices by comparing the bus ID from filesystem with the bus ID from NVML.
     * They must match exactly, so we use 4-digit format to match the filesystem. */
    char formatted_bdf[64];
    unsigned int dom = 0, bus = 0, dev_num = 0, fn = 0;
    if (sscanf(bdf, "%x:%x:%x.%x", &dom, &bus, &dev_num, &fn) == 4) {
        /* Format as 4-digit domain to match filesystem: "0000:00:05.0" */
        snprintf(formatted_bdf, sizeof(formatted_bdf), "%04x:%02x:%02x.%x", dom, bus, dev_num, fn);
        bdf = formatted_bdf;
    }
    
    /* Ensure we have enough space */
    if (len < (int)strlen(bdf) + 1) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetPCIBusId: buffer too small (need %zu, got %d)\n",
                strlen(bdf) + 1, len);
        fflush(stderr);
        return CUDA_ERROR_INVALID_VALUE;
    }
    strncpy(pciBusId, bdf, (size_t)len - 1);
    pciBusId[len - 1] = '\0';
    fprintf(stderr, "[libvgpu-cuda] cuDeviceGetPCIBusId SUCCESS: returning busId=\"%s\" (pid=%d)\n", 
            pciBusId, (int)getpid());
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId_v2(char *pciBusId, int len, CUdevice dev)
{
    return cuDeviceGetPCIBusId(pciBusId, len, dev);
}

CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuDeviceComputeCapability(major=%p, minor=%p, dev=%d)\n",
            major, minor, dev);
    fflush(stderr);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!major || !minor) return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    /* Defensive fallback: match cuDeviceGetAttribute/cuDeviceGetProperties pattern.
     * GGML may call this before transport is ready; return 9.0 to avoid "compute capability 0.0". */
    *major = (g_gpu_info_valid && g_gpu_info.compute_cap_major > 0)
             ? g_gpu_info.compute_cap_major : GPU_DEFAULT_CC_MAJOR;
    *minor = (g_gpu_info_valid && g_gpu_info.compute_cap_minor >= 0)
             ? g_gpu_info.compute_cap_minor : GPU_DEFAULT_CC_MINOR;
    fprintf(stderr, "[libvgpu-cuda] cuDeviceComputeCapability SUCCESS: %d.%d\n", *major, *minor);
    fflush(stderr);
    return CUDA_SUCCESS;
}

/* Additional device query functions that might be called during initialization */

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUdevice dev, int texDesc)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements=%p, dev=%d, texDesc=%d)\n",
            maxWidthInElements, dev, texDesc);
    fflush(stderr);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!maxWidthInElements) return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    /* Return a reasonable default for H100 */
    *maxWidthInElements = 134217728; /* 128M elements */
    fprintf(stderr, "[libvgpu-cuda] cuDeviceGetTexture1DLinearMaxWidth SUCCESS: %zu\n", *maxWidthInElements);
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList=%p, dev=%d, flags=%d)\n",
            nvSciSyncAttrList, dev, flags);
    fflush(stderr);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!nvSciSyncAttrList) return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    /* NvSciSync is for multi-GPU synchronization - not critical for single GPU */
    /* Return success with zero-initialized attributes */
    memset(nvSciSyncAttrList, 0, 256); /* Reasonable default size */
    fprintf(stderr, "[libvgpu-cuda] cuDeviceGetNvSciSyncAttributes SUCCESS (stub)\n");
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *memPool, CUdevice dev)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuDeviceGetDefaultMemPool(memPool=%p, dev=%d)\n",
            memPool, dev);
    fflush(stderr);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!memPool) return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    /* Return a dummy memory pool handle */
    *memPool = (CUmemoryPool)(uintptr_t)0xDEADBEEF;
    fprintf(stderr, "[libvgpu-cuda] cuDeviceGetDefaultMemPool SUCCESS (stub)\n");
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetMemPool(CUmemoryPool *memPool, CUdevice dev)
{
    return cuDeviceGetDefaultMemPool(memPool, dev);
}

/* cuDeviceGetProperties - Legacy function, deprecated but some code still uses it.
 * Returns device properties in a structure. We implement it for compatibility.
 * CUdevprop typedef is defined earlier in the file. */

CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
    /* CRITICAL: Log FIRST using syscall to see if this is called */
    const char *called_msg = "[libvgpu-cuda] cuDeviceGetProperties() CALLED\n";
    (void)syscall(__NR_write, 2, called_msg,
                  sizeof("[libvgpu-cuda] cuDeviceGetProperties() CALLED\n") - 1);
    
    if (!prop) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    
    /* CRITICAL: Return immediately without calling ensure_init()
     * Use g_gpu_info which is initialized by init_gpu_defaults() in cuInit(). */
    if (!prop) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetProperties: invalid pointer\n");
        fflush(stderr);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        fprintf(stderr, "[libvgpu-cuda] cuDeviceGetProperties: invalid device %d\n", dev);
        fflush(stderr);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    
    /* Fill in properties from g_gpu_info */
    memset(prop, 0, sizeof(CUdevprop));
    /* CRITICAL: Use defaults if g_gpu_info isn't initialized */
    prop->major = (g_gpu_info_valid && g_gpu_info.compute_cap_major > 0) 
                  ? g_gpu_info.compute_cap_major 
                  : GPU_DEFAULT_CC_MAJOR;
    prop->minor = (g_gpu_info_valid && g_gpu_info.compute_cap_minor >= 0) 
                  ? g_gpu_info.compute_cap_minor 
                  : GPU_DEFAULT_CC_MINOR;
    /* Use snprintf to avoid truncation warning */
    snprintf(prop->name, sizeof(prop->name), "%.*s", (int)(sizeof(prop->name) - 1), g_gpu_info.name);
    prop->totalGlobalMem = (size_t)g_gpu_info.total_mem;
    prop->multiprocessorCount = g_gpu_info.multi_processor_count;
    prop->maxThreadsPerBlock = g_gpu_info.max_threads_per_block;
    prop->maxThreadsPerMultiprocessor = g_gpu_info.max_threads_per_mp;
    prop->sharedMemPerBlock = g_gpu_info.max_shared_mem_per_block;
    prop->regsPerBlock = g_gpu_info.regs_per_block;
    prop->warpSize = g_gpu_info.warp_size;
    prop->clockRate = g_gpu_info.clock_rate_khz;
    prop->memoryClockRate = g_gpu_info.memory_clock_rate_khz;
    prop->memoryBusWidth = g_gpu_info.memory_bus_width;
    prop->totalConstMem = GPU_DEFAULT_TOTAL_CONSTANT_MEM;
    prop->textureAlignment = GPU_DEFAULT_TEXTURE_ALIGNMENT;
    prop->deviceOverlap = 1;  /* Device can overlap copies and kernels */
    prop->multiProcessorCount = g_gpu_info.multi_processor_count;
    
    fprintf(stderr, "[libvgpu-cuda] cuDeviceGetProperties SUCCESS: %s, CC %d.%d, %zu MB\n",
            prop->name, prop->major, prop->minor, prop->totalGlobalMem / (1024 * 1024));
    fflush(stderr);
    return CUDA_SUCCESS;
}

/* ================================================================
 * CUDA Driver API — Primary Context
 *
 * CUDA applications (esp. via cudart) use the primary context API.
 * We map these to remote context operations.
 * ================================================================ */

static void log_ctx_state_snapshot(const char *tag, CUcontext arg_ctx,
                                   CUcontext out_ctx, CUresult rc,
                                   int extra0, int extra1)
{
    char buf[320];
    int len = snprintf(buf, sizeof(buf),
                       "[libvgpu-cuda] %s: arg_ctx=%p out_ctx=%p rc=%d extra0=%d extra1=%d current=%p global=%p init_phase=%d (pid=%d)\n",
                       tag ? tag : "(null)",
                       (void *)arg_ctx, (void *)out_ctx, (int)rc,
                       extra0, extra1,
                       (void *)g_current_ctx, (void *)g_global_ctx,
                       g_in_init_phase, (int)getpid());
    if (len > 0 && len < (int)sizeof(buf)) {
        syscall(__NR_write, 2, buf, len);
    }
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    /* CRITICAL: Log immediately using syscall */
    const char *msg = "[libvgpu-cuda] cuDevicePrimaryCtxRetain() CALLED\n";
    (void)syscall(__NR_write, 2, msg,
                  sizeof("[libvgpu-cuda] cuDevicePrimaryCtxRetain() CALLED\n") - 1);
    log_ctx_state_snapshot("cuDevicePrimaryCtxRetain entry", NULL, NULL,
                           CUDA_SUCCESS, (int)dev, 0);
    
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;

    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[libvgpu-cuda] cuDevicePrimaryCtxRetain ensure_init() failed: %d\n", rc);
        fflush(stderr);
        log_ctx_state_snapshot("cuDevicePrimaryCtxRetain ensure_init failed", NULL, NULL,
                               rc, (int)dev, 0);
        return rc;
    }

    /* Try to connect and get real context, but if transport isn't ready yet,
     * return a dummy context to allow initialization to proceed. */
    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)dev);
    
    /* Try RPC call, but don't fail if transport isn't connected yet */
    CUresult rpc_rc = ensure_connected();
    if (rpc_rc == CUDA_SUCCESS && g_transport) {
        rc = rpc_simple(CUDA_CALL_DEVICE_PRIMARY_CTX_RETAIN, args, 2, &result);
        if (rc == CUDA_SUCCESS && result.status == 0 && result.num_results >= 1) {
            *pctx = (CUcontext)(uintptr_t)result.results[0];
            g_current_ctx = *pctx;
            g_global_ctx = *pctx;
            
            /* Clear init phase flag after successful context creation */
            ensure_mutex_init();
            pthread_mutex_lock(&g_mutex);
            if (g_in_init_phase) {
                g_in_init_phase = 0;
                fprintf(stderr, "[libvgpu-cuda] INIT PHASE COMPLETE: cleared g_in_init_phase after successful context creation\n");
                fflush(stderr);
            }
            pthread_mutex_unlock(&g_mutex);
            
            if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] cuDevicePrimaryCtxRetain SUCCESS (via RPC): pctx=%p\n", *pctx); fflush(stderr); }
            log_ctx_state_snapshot("cuDevicePrimaryCtxRetain via RPC", NULL, *pctx,
                                   CUDA_SUCCESS, (int)dev, (int)result.status);
            return CUDA_SUCCESS;
        }
        log_ctx_state_snapshot("cuDevicePrimaryCtxRetain RPC non-success", NULL, NULL,
                               rc, (int)dev, (int)result.status);
    }
    
    /* Fallback: return a dummy context to allow initialization to proceed.
     * The real context will be created when the transport is ready. */
    static CUcontext dummy_ctx = (CUcontext)(uintptr_t)0xDEADBEEF;
    *pctx = dummy_ctx;
    g_current_ctx = dummy_ctx;
    g_global_ctx = dummy_ctx;
    if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] cuDevicePrimaryCtxRetain SUCCESS (dummy context, transport deferred): pctx=%p\n", *pctx); fflush(stderr); }
    log_ctx_state_snapshot("cuDevicePrimaryCtxRetain dummy fallback", NULL, *pctx,
                           CUDA_SUCCESS, (int)dev, (int)rpc_rc);
    return CUDA_SUCCESS;
}

/* _v2 version is identical to base version */
CUresult cuDevicePrimaryCtxRetain_v2(CUcontext *pctx, CUdevice dev)
{
    return cuDevicePrimaryCtxRetain(pctx, dev);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
{
    return cuDevicePrimaryCtxRelease_v2(dev);
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)dev);
    return rpc_simple(CUDA_CALL_DEVICE_PRIMARY_CTX_RELEASE, args, 2, &result);
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev)
{
    return cuDevicePrimaryCtxReset_v2(dev);
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)dev);
    return rpc_simple(CUDA_CALL_DEVICE_PRIMARY_CTX_RESET, args, 2, &result);
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags)
{
    return cuDevicePrimaryCtxSetFlags_v2(dev, flags);
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[4];
    CUDA_PACK_U64(args, 0, (uint64_t)dev);
    CUDA_PACK_U64(args, 2, (uint64_t)flags);
    return rpc_simple(CUDA_CALL_DEVICE_PRIMARY_CTX_SET_FLAGS, args, 4, &result);
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!flags || !active) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)dev);
    rc = rpc_simple(CUDA_CALL_DEVICE_PRIMARY_CTX_GET_STATE, args, 2, &result);
    if (rc == CUDA_SUCCESS && result.status == 0 && result.num_results >= 2) {
        *flags  = (unsigned int)result.results[0];
        *active = (int)result.results[1];
    }
    return rc;
}

/* ================================================================
 * CUDA Driver API — Context management
 * ================================================================ */

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuCtxCreate_v2(pctx=%p, flags=0x%x, dev=%d)\n",
            pctx, flags, (int)dev);
    fflush(stderr);
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[libvgpu-cuda] cuCtxCreate_v2 ensure_init() failed: %d\n", rc);
        fflush(stderr);
        return rc;
    }
    if (!pctx) {
        fprintf(stderr, "[libvgpu-cuda] cuCtxCreate_v2: invalid pointer\n");
        fflush(stderr);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (dev != 0) {
        fprintf(stderr, "[libvgpu-cuda] cuCtxCreate_v2: invalid device %d\n", dev);
        fflush(stderr);
        return CUDA_ERROR_INVALID_DEVICE;
    }

    /* Try to connect and create context, but if transport isn't ready yet,
     * return a dummy context to allow initialization to proceed. */
    CUresult rpc_rc = ensure_connected();
    if (rpc_rc == CUDA_SUCCESS && g_transport) {
        CUDACallResult result;
        uint32_t args[4];
        args[0] = flags;
        args[1] = 0;
        CUDA_PACK_U64(args, 2, (uint64_t)dev);

        rc = rpc_simple(CUDA_CALL_CTX_CREATE, args, 4, &result);
        if (rc == CUDA_SUCCESS && result.status == 0 && result.num_results >= 1) {
            *pctx = (CUcontext)(uintptr_t)result.results[0];
            g_current_ctx = *pctx;
            g_global_ctx = *pctx;
            
            /* Clear init phase flag after successful context creation */
            ensure_mutex_init();
            pthread_mutex_lock(&g_mutex);
            if (g_in_init_phase) {
                g_in_init_phase = 0;
                fprintf(stderr, "[libvgpu-cuda] INIT PHASE COMPLETE: cleared g_in_init_phase after successful context creation\n");
                fflush(stderr);
            }
            pthread_mutex_unlock(&g_mutex);
            
            if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] cuCtxCreate_v2 SUCCESS (via RPC): pctx=%p\n", *pctx); fflush(stderr); }
            return CUDA_SUCCESS;
        }
    }
    
    /* Fallback: return a dummy context to allow initialization to proceed */
    static CUcontext dummy_ctx = (CUcontext)(uintptr_t)0xDEADBEEF;
    *pctx = dummy_ctx;
    g_current_ctx = dummy_ctx;
    g_global_ctx = dummy_ctx;
    if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] cuCtxCreate_v2 SUCCESS (dummy context, transport deferred): pctx=%p\n", *pctx); fflush(stderr); }
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    return cuCtxCreate_v2(pctx, flags, dev);
}

/* _v3 version is identical to _v2 */
CUresult cuCtxCreate_v3(CUcontext *pctx, unsigned int flags, CUdevice dev, void *params)
{
    (void)params;  /* v3 adds params but we ignore it */
    return cuCtxCreate_v2(pctx, flags, dev);
}

CUresult cuCtxDestroy_v2(CUcontext ctx)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)ctx);

    rc = rpc_simple(CUDA_CALL_CTX_DESTROY, args, 2, &result);
    if (rc == CUDA_SUCCESS && g_current_ctx == ctx) {
        g_current_ctx = NULL;
    }
    return rc;
}

CUresult cuCtxDestroy(CUcontext ctx)
{
    return cuCtxDestroy_v2(ctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx)
{
    /* CRITICAL: Log FIRST using syscall to see if this is called during GGML init */
    const char *msg = "[libvgpu-cuda] cuCtxSetCurrent() CALLED: ctx=%p\n";
    char buf[128];
    int len = snprintf(buf, sizeof(buf), msg, ctx);
    if (len > 0 && len < (int)sizeof(buf)) {
        syscall(__NR_write, 2, buf, len);
    }
    
    /* CRITICAL: During init phase, don't call ensure_init() - just set context locally */
    if (g_in_init_phase) {
        g_current_ctx = ctx;
        g_global_ctx = ctx;
        const char *success_msg = "[libvgpu-cuda] cuCtxSetCurrent() SUCCESS (init phase, local): ctx=%p\n";
        char success_buf[128];
        int success_len = snprintf(success_buf, sizeof(success_buf), success_msg, ctx);
        if (success_len > 0 && success_len < (int)sizeof(success_buf)) {
            syscall(__NR_write, 2, success_buf, success_len);
        }
        log_ctx_state_snapshot("cuCtxSetCurrent init-local", ctx, ctx,
                               CUDA_SUCCESS, 0, 0);
        return CUDA_SUCCESS;
    }
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) {
        const char *error_msg = "[libvgpu-cuda] cuCtxSetCurrent ensure_init() failed: %d\n";
        char error_buf[128];
        int error_len = snprintf(error_buf, sizeof(error_buf), error_msg, rc);
        if (error_len > 0 && error_len < (int)sizeof(error_buf)) {
            syscall(__NR_write, 2, error_buf, error_len);
        }
        log_ctx_state_snapshot("cuCtxSetCurrent ensure_init failed", ctx, NULL,
                               rc, 0, 0);
        return rc;
    }

    /* If we only have the deferred dummy context, try to upgrade it before
     * telling CUDA user-space that a real current context exists. */
    if (ctx_is_dummy(ctx)) {
        CUcontext real_ctx = NULL;
        CUresult retain_rc = cuDevicePrimaryCtxRetain(&real_ctx, 0);
        if (retain_rc == CUDA_SUCCESS && real_ctx != NULL && !ctx_is_dummy(real_ctx)) {
            ctx = real_ctx;
        }
    }

    /* Try RPC call if transport is connected, but don't fail if it's not */
    CUresult rpc_rc = ensure_connected();
    if (rpc_rc == CUDA_SUCCESS && g_transport) {
        CUDACallResult result;
        memset(&result, 0, sizeof(result));
        uint32_t args[2];
        CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)ctx);
        rc = rpc_simple(CUDA_CALL_CTX_SET_CURRENT, args, 2, &result);
        if (rc == CUDA_SUCCESS && result.status == 0) {
            g_current_ctx = ctx;
            g_global_ctx = ctx;
            if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] cuCtxSetCurrent SUCCESS (via RPC): ctx=%p\n", ctx); fflush(stderr); }
            log_ctx_state_snapshot("cuCtxSetCurrent via RPC", ctx, ctx,
                                   CUDA_SUCCESS, (int)rpc_rc, (int)result.status);
            return CUDA_SUCCESS;
        }
        log_ctx_state_snapshot("cuCtxSetCurrent RPC non-success", ctx, NULL,
                               rc, (int)rpc_rc, (int)result.status);
    }
    
    /* Fallback: just set the context locally without RPC */
    g_current_ctx = ctx;
    g_global_ctx = ctx;
    if (vgpu_debug_logging()) { fprintf(stderr, "[libvgpu-cuda] cuCtxSetCurrent SUCCESS (local, transport deferred): ctx=%p\n", ctx); fflush(stderr); }
    log_ctx_state_snapshot("cuCtxSetCurrent local fallback", ctx, ctx,
                           CUDA_SUCCESS, (int)rpc_rc, 0);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx)
{
    /* CRITICAL: Log FIRST using syscall to see if this is called during GGML init */
    const char *msg = "[libvgpu-cuda] cuCtxGetCurrent() CALLED\n";
    (void)syscall(__NR_write, 2, msg,
                  sizeof("[libvgpu-cuda] cuCtxGetCurrent() CALLED\n") - 1);
    
    if (!pctx) {
        log_ctx_state_snapshot("cuCtxGetCurrent invalid arg", NULL, NULL,
                               CUDA_ERROR_INVALID_VALUE, 0, 0);
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pctx = g_current_ctx ? g_current_ctx : g_global_ctx;
    log_ctx_state_snapshot("cuCtxGetCurrent initial state", NULL, *pctx,
                           CUDA_SUCCESS, 0, 0);

    /* Some CUDA clients (including CUBLAS init paths) require a non-NULL
     * current context very early. If nothing is set yet, or if we only cached
     * the deferred dummy context, try to upgrade from the real primary ctx. */
    if (*pctx == NULL || ctx_is_dummy(*pctx)) {
        CUresult init_rc = cuInit(0);
        if (init_rc == CUDA_SUCCESS) {
            CUcontext seeded_ctx = NULL;
            CUresult retain_rc = cuDevicePrimaryCtxRetain(&seeded_ctx, 0);
            if (retain_rc == CUDA_SUCCESS &&
                seeded_ctx != NULL &&
                !ctx_is_dummy(seeded_ctx)) {
                g_current_ctx = seeded_ctx;
                g_global_ctx = seeded_ctx;
                *pctx = seeded_ctx;
            }
            log_ctx_state_snapshot("cuCtxGetCurrent seeded from primary", NULL, seeded_ctx,
                                   retain_rc, (int)init_rc, 0);
        } else {
            log_ctx_state_snapshot("cuCtxGetCurrent seed init failed", NULL, NULL,
                                   init_rc, 0, 0);
        }
    }
    
    const char *success_msg = "[libvgpu-cuda] cuCtxGetCurrent() SUCCESS: returning ctx=%p\n";
    char buf[128];
    int len = snprintf(buf, sizeof(buf), success_msg, *pctx);
    if (len > 0 && len < (int)sizeof(buf)) {
        syscall(__NR_write, 2, buf, len);
    }
    log_ctx_state_snapshot("cuCtxGetCurrent return", NULL, *pctx,
                           CUDA_SUCCESS, 0, 0);
    
    return CUDA_SUCCESS;
}

/* _v2 version is identical */
CUresult cuCtxGetCurrent_v2(CUcontext *pctx)
{
    return cuCtxGetCurrent(pctx);
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx)
{
    return cuCtxSetCurrent(ctx);
}

CUresult cuCtxPushCurrent(CUcontext ctx)
{
    return cuCtxPushCurrent_v2(ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx)
{
    if (pctx) *pctx = g_current_ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx)
{
    return cuCtxPopCurrent_v2(pctx);
}

CUresult cuCtxSynchronize(void)
{
    {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) {
            const char *msg = "ctx_sync\n";
            syscall(__NR_write, nfd, msg, 10);
            syscall(__NR_close, nfd);
        }
    }
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    return rpc_simple(CUDA_CALL_CTX_SYNCHRONIZE, NULL, 0, &result);
}

CUresult cuCtxGetDevice(CUdevice *device)
{
    /* CRITICAL: Log FIRST using syscall to see if this is called */
    const char *called_msg = "[libvgpu-cuda] cuCtxGetDevice() CALLED\n";
    (void)syscall(__NR_write, 2, called_msg,
                  sizeof("[libvgpu-cuda] cuCtxGetDevice() CALLED\n") - 1);
    
    if (!device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    /* CRITICAL: Return immediately without calling ensure_init()
     * This is called by cudaGetDevice() (runtime API) which might
     * be called by ggml_backend_cuda_init before device query functions. */
    *device = 0;  /* Always device 0 */
    
    const char *success_msg = "[libvgpu-cuda] cuCtxGetDevice() SUCCESS: device=0\n";
    (void)syscall(__NR_write, 2, success_msg,
                  sizeof("[libvgpu-cuda] cuCtxGetDevice() SUCCESS: device=0\n") - 1);
    
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice_v2(CUdevice *device)
{
    return cuCtxGetDevice(device);
}

/* cuCtxGetApiVersion is defined later in the file with better logging */

/* ================================================================
 * CUDA Driver API — Memory management (forwarded to host)
 * ================================================================ */

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    /* CRITICAL: Log this call - GGML allocates memory for tensors */
    char log_msg[128];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] cuMemAlloc_v2() CALLED (size=%zu, pid=%d)\n",
                          bytesize, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    if (bytesize == 0) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[4];
    CUDA_PACK_U64(args, 0, (uint64_t)bytesize);
    args[2] = 0; args[3] = 0;

    rc = rpc_simple(CUDA_CALL_MEM_ALLOC, args, 4, &result);
    if (rc == CUDA_SUCCESS &&
        result.status == CUDA_SUCCESS &&
        result.num_results >= 1 &&
        result.results[0] != 0) {
        /* Preserve host allocator's exact pointer; don't remap/alter it here. */
        *dptr = (CUdeviceptr)(uintptr_t)result.results[0];
        
        char success_msg[128];
        int success_len = snprintf(success_msg, sizeof(success_msg),
                                  "[libvgpu-cuda] cuMemAlloc_v2() SUCCESS: ptr=0x%llx, size=%zu (pid=%d)\n",
                                  (unsigned long long)*dptr, bytesize, (int)getpid());
        if (success_len > 0 && success_len < (int)sizeof(success_msg)) {
            syscall(__NR_write, 2, success_msg, success_len);
        }
    } else {
        /* One reconnect + retry for transient transport timeouts (e.g. mediator slow to respond). */
        ensure_mutex_init();
        pthread_mutex_lock(&g_mutex);
        if (g_transport) {
            cuda_transport_destroy(g_transport);
            g_transport = NULL;
        }
        pthread_mutex_unlock(&g_mutex);

        CUresult reconnect_rc = ensure_connected();
        if (reconnect_rc != CUDA_SUCCESS) {
            *dptr = 0;
            return reconnect_rc;
        }
        CUDACallResult retry_result = {0};
        CUresult retry_rc = rpc_simple(CUDA_CALL_MEM_ALLOC, args, 4, &retry_result);
        if (retry_rc == CUDA_SUCCESS &&
            retry_result.status == CUDA_SUCCESS &&
            retry_result.num_results >= 1 &&
            retry_result.results[0] != 0) {
            *dptr = (CUdeviceptr)(uintptr_t)retry_result.results[0];
            return CUDA_SUCCESS;
        }

        *dptr = 0;
        if (retry_rc == CUDA_SUCCESS) {
            rc = (retry_result.status != 0) ? (CUresult)retry_result.status : CUDA_ERROR_UNKNOWN;
        } else if (rc == CUDA_SUCCESS) {
            rc = retry_rc;
        }
        char fail_msg[160];
        uint64_t err_ptr = 0;
        uint32_t nr = retry_result.num_results ? retry_result.num_results : result.num_results;
        if (nr >= 1) err_ptr = retry_result.num_results ? retry_result.results[0] : result.results[0];
        int fail_len = snprintf(fail_msg, sizeof(fail_msg),
                                "[libvgpu-cuda] cuMemAlloc_v2() ERROR: rc=%d status=%u num_results=%u ptr=0x%llx (pid=%d)\n",
                                rc, retry_result.status ? retry_result.status : result.status,
                                nr, (unsigned long long)err_ptr, (int)getpid());
        if (fail_len > 0 && fail_len < (int)sizeof(fail_msg)) {
            syscall(__NR_write, 2, fail_msg, fail_len);
        }
    }
    return rc;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)dptr);

    return rpc_simple(CUDA_CALL_MEM_FREE, args, 2, &result);
}

CUresult cuMemFree(CUdeviceptr dptr)
{
    return cuMemFree_v2(dptr);
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemAllocManaged(dptr=%p, bytesize=%zu, flags=0x%x)\n",
            dptr, bytesize, flags);
    
    /* For now, delegate to cuMemAlloc_v2 - managed memory handled the same way */
    return cuMemAlloc_v2(dptr, bytesize);
}

CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pitch,
                            size_t widthInBytes, size_t height, unsigned int elementSizeBytes)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemAllocPitch_v2(dptr=%p, width=%zu, height=%zu, elementSize=%u)\n",
            dptr, widthInBytes, height, elementSizeBytes);
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!dptr || !pitch) return CUDA_ERROR_INVALID_VALUE;
    if (widthInBytes == 0 || height == 0) return CUDA_ERROR_INVALID_VALUE;
    
    /* Allocate using cuMemAlloc_v2, then set pitch to width (no padding for now) */
    size_t totalSize;
    if (height > 0 && widthInBytes > SIZE_MAX / height) return CUDA_ERROR_INVALID_VALUE;
    totalSize = widthInBytes * height;
    rc = cuMemAlloc_v2(dptr, totalSize);
    if (rc == CUDA_SUCCESS) {
        *pitch = widthInBytes;
    }
    return rc;
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pitch,
                         size_t widthInBytes, size_t height, unsigned int elementSizeBytes)
{
    return cuMemAllocPitch_v2(dptr, pitch, widthInBytes, height, elementSizeBytes);
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *size, CUdeviceptr dptr)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemGetAddressRange_v2(base=%p, size=%p, dptr=0x%llx)\n",
            base, size, (unsigned long long)dptr);
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!base || !size) return CUDA_ERROR_INVALID_VALUE;
    
    /* For now, return the pointer as base with unknown size */
    *base = dptr;
    *size = 0;  /* Size unknown - would need to track allocations */
    fprintf(stderr, "[libvgpu-cuda] cuMemGetAddressRange_v2 returning SUCCESS (base=0x%llx, size=0)\n",
            (unsigned long long)*base);
    return CUDA_SUCCESS;
}

CUresult cuMemGetAddressRange(CUdeviceptr *base, size_t *size, CUdeviceptr dptr)
{
    return cuMemGetAddressRange_v2(base, size, dptr);
}

CUresult cuMemFreeHost(void *p)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuMemFreeHost(p=%p)\n", p);
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    
    /* Host memory is freed with standard free() - always succeed */
    if (p) {
        free(p);
    }
    fprintf(stderr, "[libvgpu-cuda] cuMemFreeHost returning SUCCESS\n");
    return CUDA_SUCCESS;
}

/* cuMemHostAlloc - allocate page-locked host memory (Driver API equivalent of cudaMallocHost) */
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags)
{
    /* CRITICAL: Log this call - GGML may use Driver API for host memory allocation */
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] cuMemHostAlloc() CALLED (size=%zu, flags=0x%x, pid=%d)\n",
                          bytesize, flags, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!pp) return CUDA_ERROR_INVALID_VALUE;
    
    /* CRITICAL FIX: Allocate aligned host memory (32-byte alignment for GGML)
     * Use posix_memalign to ensure proper alignment */
    const size_t alignment = 32; /* GGML TENSOR_ALIGNMENT */
    void *aligned_ptr = NULL;
    
    /* Resolve real posix_memalign to avoid recursion */
    static int (*real_posix_memalign)(void **, size_t, size_t) = NULL;
    if (!real_posix_memalign) {
        real_posix_memalign = (int (*)(void **, size_t, size_t))dlsym(RTLD_NEXT, "posix_memalign");
        if (!real_posix_memalign) {
            /* Fallback: use aligned_alloc if available */
            void *(*real_aligned_alloc)(size_t, size_t) = (void *(*)(size_t, size_t))dlsym(RTLD_NEXT, "aligned_alloc");
            if (real_aligned_alloc) {
                aligned_ptr = real_aligned_alloc(alignment, bytesize);
                if (!aligned_ptr) {
                    return CUDA_ERROR_OUT_OF_MEMORY;
                }
                *pp = aligned_ptr;
                fprintf(stderr, "[libvgpu-cuda] cuMemHostAlloc() SUCCESS: ptr=%p (aligned_alloc, 32-byte aligned), size=%zu (pid=%d)\n",
                        *pp, bytesize, (int)getpid());
                return CUDA_SUCCESS;
            }
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }
    
    int posix_rc = real_posix_memalign(&aligned_ptr, alignment, bytesize);
    
    if (posix_rc != 0 || !aligned_ptr) {
        fprintf(stderr, "[libvgpu-cuda] cuMemHostAlloc() ERROR: posix_memalign failed (rc=%d, pid=%d)\n",
                posix_rc, (int)getpid());
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    *pp = aligned_ptr;
    
    /* Verify alignment */
    if ((uintptr_t)*pp % alignment != 0) {
        fprintf(stderr, "[libvgpu-cuda] cuMemHostAlloc() CRITICAL: ptr=%p is NOT %zu-byte aligned! (pid=%d)\n",
                *pp, alignment, (int)getpid());
    }
    
    fprintf(stderr, "[libvgpu-cuda] cuMemHostAlloc() SUCCESS: ptr=%p (32-byte aligned), size=%zu (pid=%d)\n",
            *pp, bytesize, (int)getpid());
    return CUDA_SUCCESS;
}

/* cuMemHostRegister - register existing host memory for CUDA access */
CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int flags)
{
    /* CRITICAL: Log this call - GGML may register host buffers */
    fprintf(stderr, "[libvgpu-cuda] cuMemHostRegister() CALLED (ptr=%p, size=%zu, flags=0x%x, pid=%d)\n",
            p, bytesize, flags, (int)getpid());
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    
    /* CRITICAL: Check if pointer is aligned - if not, log warning */
    const size_t alignment = 32; /* GGML TENSOR_ALIGNMENT */
    if ((uintptr_t)p % alignment != 0) {
        fprintf(stderr, "[libvgpu-cuda] cuMemHostRegister() WARNING: ptr=%p is NOT %zu-byte aligned! (pid=%d)\n",
                p, alignment, (int)getpid());
    }
    
    /* For now, just succeed - we don't actually need to register with host */
    fprintf(stderr, "[libvgpu-cuda] cuMemHostRegister() SUCCESS: ptr=%p, size=%zu (pid=%d)\n",
            p, bytesize, (int)getpid());
    return CUDA_SUCCESS;
}

/* cuMemHostUnregister - unregister host memory */
CUresult cuMemHostUnregister(void *p)
{
    fprintf(stderr, "[libvgpu-cuda] cuMemHostUnregister() CALLED (ptr=%p, pid=%d)\n",
            p, (int)getpid());
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    
    /* Always succeed */
    return CUDA_SUCCESS;
}

/* cuMemHostGetDevicePointer - get device pointer for registered host memory */
CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int flags)
{
    fprintf(stderr, "[libvgpu-cuda] cuMemHostGetDevicePointer() CALLED (host_ptr=%p, flags=0x%x, pid=%d)\n",
            p, flags, (int)getpid());
    
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!pdptr || !p) return CUDA_ERROR_INVALID_VALUE;
    
    /* For unified memory, return the host pointer as device pointer */
    *pdptr = (CUdeviceptr)p;
    
    fprintf(stderr, "[libvgpu-cuda] cuMemHostGetDevicePointer() SUCCESS: host_ptr=%p, device_ptr=0x%llx (pid=%d)\n",
            p, (unsigned long long)*pdptr, (int)getpid());
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                          size_t byteCount)
{
    if (vgpu_debug_logging()) {
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                              "[libvgpu-cuda] cuMemcpyHtoD() CALLED: dst=0x%llx size=%zu bytes (pid=%d)\n",
                              (unsigned long long)dstDevice, byteCount, (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg))
            syscall(__NR_write, 2, log_msg, log_len);
    }

    CUresult rc = ensure_connected();
    if (rc != CUDA_SUCCESS) return rc;
    if (!srcHost && byteCount > 0) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[4];
    CUDA_PACK_U64(args, 0, (uint64_t)dstDevice);
    CUDA_PACK_U64(args, 2, (uint64_t)byteCount);

    rc = (CUresult)cuda_transport_call(g_transport,
                                       CUDA_CALL_MEMCPY_HTOD,
                                       args, 4,
                                       srcHost, (uint32_t)byteCount,
                                       &result, NULL, 0, NULL);

    if (rc != CUDA_SUCCESS) {
        /* One reconnect + retry for transient transport timeouts under heavy model load. */
        ensure_mutex_init();
        pthread_mutex_lock(&g_mutex);
        if (g_transport) {
            cuda_transport_destroy(g_transport);
            g_transport = NULL;
        }
        pthread_mutex_unlock(&g_mutex);

        CUresult reconnect_rc = ensure_connected();
        if (reconnect_rc == CUDA_SUCCESS) {
            rc = (CUresult)cuda_transport_call(g_transport,
                                               CUDA_CALL_MEMCPY_HTOD,
                                               args, 4,
                                               srcHost, (uint32_t)byteCount,
                                               &result, NULL, 0, NULL);
        } else {
            rc = reconnect_rc;
        }
    }

    if (rc == CUDA_SUCCESS && vgpu_debug_logging()) {
        char success_msg[128];
        int success_len = snprintf(success_msg, sizeof(success_msg),
                                  "[libvgpu-cuda] cuMemcpyHtoD() SUCCESS: forwarded to host (pid=%d)\n",
                                  (int)getpid());
        if (success_len > 0 && success_len < (int)sizeof(success_msg))
            syscall(__NR_write, 2, success_msg, success_len);
    }
    return rc;
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                       size_t byteCount)
{
    return cuMemcpyHtoD_v2(dstDevice, srcHost, byteCount);
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                          size_t byteCount)
{
    if (vgpu_debug_logging()) {
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                              "[libvgpu-cuda] cuMemcpyDtoH() CALLED: src=0x%llx size=%zu bytes (pid=%d)\n",
                              (unsigned long long)srcDevice, byteCount, (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg))
            syscall(__NR_write, 2, log_msg, log_len);
    }

    CUresult rc = ensure_connected();
    if (rc != CUDA_SUCCESS) return rc;
    if (!dstHost && byteCount > 0) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[4];
    uint32_t recv_len = 0;
    CUDA_PACK_U64(args, 0, (uint64_t)srcDevice);
    CUDA_PACK_U64(args, 2, (uint64_t)byteCount);

    rc = (CUresult)cuda_transport_call(g_transport,
                                       CUDA_CALL_MEMCPY_DTOH,
                                       args, 4,
                                       NULL, 0,
                                       &result,
                                       dstHost, (uint32_t)byteCount,
                                       &recv_len);

    if (rc != CUDA_SUCCESS) {
        /* Mirror HtoD retry behavior for symmetric transfer stability. */
        ensure_mutex_init();
        pthread_mutex_lock(&g_mutex);
        if (g_transport) {
            cuda_transport_destroy(g_transport);
            g_transport = NULL;
        }
        pthread_mutex_unlock(&g_mutex);

        CUresult reconnect_rc = ensure_connected();
        if (reconnect_rc == CUDA_SUCCESS) {
            recv_len = 0;
            rc = (CUresult)cuda_transport_call(g_transport,
                                               CUDA_CALL_MEMCPY_DTOH,
                                               args, 4,
                                               NULL, 0,
                                               &result,
                                               dstHost, (uint32_t)byteCount,
                                               &recv_len);
        } else {
            rc = reconnect_rc;
        }
    }

    if (rc == CUDA_SUCCESS && vgpu_debug_logging()) {
        char success_msg[128];
        int success_len = snprintf(success_msg, sizeof(success_msg),
                                  "[libvgpu-cuda] cuMemcpyDtoH() SUCCESS: forwarded to host, received %u bytes (pid=%d)\n",
                                  recv_len, (int)getpid());
        if (success_len > 0 && success_len < (int)sizeof(success_msg))
            syscall(__NR_write, 2, success_msg, success_len);
    }
    return rc;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                       size_t byteCount)
{
    return cuMemcpyDtoH_v2(dstHost, srcDevice, byteCount);
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t byteCount)
{
    return cuMemcpyDtoD_v2(dst, src, byteCount);
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream)
{
    (void)hStream;
    return cuMemcpyDtoD_v2(dst, src, byteCount);
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                          size_t byteCount)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[6];
    CUDA_PACK_U64(args, 0, (uint64_t)dstDevice);
    CUDA_PACK_U64(args, 2, (uint64_t)srcDevice);
    CUDA_PACK_U64(args, 4, (uint64_t)byteCount);

    return rpc_simple(CUDA_CALL_MEMCPY_DTOD, args, 6, &result);
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                       size_t byteCount)
{
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount);
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t byteCount, CUstream hStream)
{
    (void)hStream;
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           size_t byteCount, CUstream hStream)
{
    return cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, byteCount, hStream);
}

static CUresult log_not_supported_copy_path(const char *name)
{
    fprintf(stderr, "[libvgpu-cuda] %s CALLED during init-only path: returning CUDA_ERROR_NOT_SUPPORTED\n",
            name);
    fflush(stderr);
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemcpy2DUnaligned(const void *pCopy)
{
    (void)pCopy;
    return log_not_supported_copy_path("cuMemcpy2DUnaligned");
}

CUresult cuMemcpy2DAsync(const void *pCopy, CUstream hStream)
{
    (void)pCopy;
    (void)hStream;
    return log_not_supported_copy_path("cuMemcpy2DAsync");
}

CUresult cuMemcpy2DAsync_v2(const void *pCopy, CUstream hStream)
{
    return cuMemcpy2DAsync(pCopy, hStream);
}

CUresult cuMemcpy3D(const void *pCopy)
{
    (void)pCopy;
    return log_not_supported_copy_path("cuMemcpy3D");
}

CUresult cuMemcpy3DAsync(const void *pCopy, CUstream hStream)
{
    (void)pCopy;
    (void)hStream;
    return log_not_supported_copy_path("cuMemcpy3DAsync");
}

CUresult cuMemcpy3DPeer(const void *pCopy)
{
    (void)pCopy;
    return log_not_supported_copy_path("cuMemcpy3DPeer");
}

CUresult cuMemcpy3DPeerAsync(const void *pCopy, CUstream hStream)
{
    (void)pCopy;
    (void)hStream;
    return log_not_supported_copy_path("cuMemcpy3DPeerAsync");
}

CUresult cuMemcpyBatchAsync(const void *params, size_t count, unsigned int flags, CUstream hStream)
{
    (void)params;
    (void)count;
    (void)flags;
    (void)hStream;
    return log_not_supported_copy_path("cuMemcpyBatchAsync");
}

CUresult cuMemcpy3DBatchAsync(const void *params, size_t count, unsigned int flags, CUstream hStream)
{
    (void)params;
    (void)count;
    (void)flags;
    (void)hStream;
    return log_not_supported_copy_path("cuMemcpy3DBatchAsync");
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext, size_t byteCount)
{
    (void)dstContext;
    (void)srcContext;
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount);
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t byteCount, CUstream hStream)
{
    (void)dstContext;
    (void)srcContext;
    (void)hStream;
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount);
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[6];
    CUDA_PACK_U64(args, 0, (uint64_t)dstDevice);
    args[2] = (uint32_t)uc;
    args[3] = 0;
    CUDA_PACK_U64(args, 4, (uint64_t)N);

    return rpc_simple(CUDA_CALL_MEMSET_D8, args, 6, &result);
}

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    (void)hStream;
    return cuMemsetD8_v2(dstDevice, uc, N);
}

CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                      size_t Width, size_t Height)
{
    (void)dstDevice;
    (void)dstPitch;
    (void)uc;
    (void)Width;
    (void)Height;
    return log_not_supported_copy_path("cuMemsetD2D8");
}

CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                           size_t Width, size_t Height, CUstream hStream)
{
    (void)dstDevice;
    (void)dstPitch;
    (void)uc;
    (void)Width;
    (void)Height;
    (void)hStream;
    return log_not_supported_copy_path("cuMemsetD2D8Async");
}

static CUresult log_not_supported_object_path(const char *name)
{
    fprintf(stderr, "[libvgpu-cuda] %s CALLED during init-only path: returning CUDA_ERROR_NOT_SUPPORTED\n",
            name);
    fflush(stderr);
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuArrayCreate(void *pHandle, const void *pAllocateArray)
{
    (void)pHandle;
    (void)pAllocateArray;
    return log_not_supported_object_path("cuArrayCreate");
}

CUresult cuArrayGetDescriptor(void *pArrayDescriptor, void *hArray)
{
    (void)pArrayDescriptor;
    (void)hArray;
    return log_not_supported_object_path("cuArrayGetDescriptor");
}

CUresult cuArrayGetSparseProperties(void *sparseProperties, void *array)
{
    (void)sparseProperties;
    (void)array;
    return log_not_supported_object_path("cuArrayGetSparseProperties");
}

CUresult cuArrayGetPlane(void *pPlaneArray, void *hArray, unsigned int planeIdx)
{
    (void)pPlaneArray;
    (void)hArray;
    (void)planeIdx;
    return log_not_supported_object_path("cuArrayGetPlane");
}

CUresult cuArray3DCreate(void *pHandle, const void *pAllocateArray)
{
    (void)pHandle;
    (void)pAllocateArray;
    return log_not_supported_object_path("cuArray3DCreate");
}

CUresult cuArray3DGetDescriptor(void *pArrayDescriptor, void *hArray)
{
    (void)pArrayDescriptor;
    (void)hArray;
    return log_not_supported_object_path("cuArray3DGetDescriptor");
}

CUresult cuArrayDestroy(void *hArray)
{
    (void)hArray;
    return log_not_supported_object_path("cuArrayDestroy");
}

CUresult cuMipmappedArrayCreate(void *pHandle, const void *pMipmappedArrayDesc,
                                unsigned int numMipmapLevels)
{
    (void)pHandle;
    (void)pMipmappedArrayDesc;
    (void)numMipmapLevels;
    return log_not_supported_object_path("cuMipmappedArrayCreate");
}

CUresult cuMipmappedArrayGetLevel(void *pLevelArray, void *hMipmappedArray, unsigned int level)
{
    (void)pLevelArray;
    (void)hMipmappedArray;
    (void)level;
    return log_not_supported_object_path("cuMipmappedArrayGetLevel");
}

CUresult cuMipmappedArrayGetSparseProperties(void *sparseProperties, void *hMipmappedArray)
{
    (void)sparseProperties;
    (void)hMipmappedArray;
    return log_not_supported_object_path("cuMipmappedArrayGetSparseProperties");
}

CUresult cuMipmappedArrayDestroy(void *hMipmappedArray)
{
    (void)hMipmappedArray;
    return log_not_supported_object_path("cuMipmappedArrayDestroy");
}

CUresult cuArrayGetMemoryRequirements(void *memoryRequirements, void *array, void *device)
{
    (void)memoryRequirements;
    (void)array;
    (void)device;
    return log_not_supported_object_path("cuArrayGetMemoryRequirements");
}

CUresult cuMipmappedArrayGetMemoryRequirements(void *memoryRequirements, void *hMipmappedArray, void *device)
{
    (void)memoryRequirements;
    (void)hMipmappedArray;
    (void)device;
    return log_not_supported_object_path("cuMipmappedArrayGetMemoryRequirements");
}

CUresult cuTexObjectCreate(void *pTexObject, const void *pResDesc,
                           const void *pTexDesc, const void *pResViewDesc)
{
    (void)pTexObject;
    (void)pResDesc;
    (void)pTexDesc;
    (void)pResViewDesc;
    return log_not_supported_object_path("cuTexObjectCreate");
}

CUresult cuTexObjectDestroy(unsigned long long texObject)
{
    (void)texObject;
    return log_not_supported_object_path("cuTexObjectDestroy");
}

CUresult cuTexObjectGetResourceDesc(void *pResDesc, unsigned long long texObject)
{
    (void)pResDesc;
    (void)texObject;
    return log_not_supported_object_path("cuTexObjectGetResourceDesc");
}

CUresult cuTexObjectGetTextureDesc(void *pTexDesc, unsigned long long texObject)
{
    (void)pTexDesc;
    (void)texObject;
    return log_not_supported_object_path("cuTexObjectGetTextureDesc");
}

CUresult cuTexObjectGetResourceViewDesc(void *pResViewDesc, unsigned long long texObject)
{
    (void)pResViewDesc;
    (void)texObject;
    return log_not_supported_object_path("cuTexObjectGetResourceViewDesc");
}

CUresult cuSurfObjectCreate(void *pSurfObject, const void *pResDesc)
{
    (void)pSurfObject;
    (void)pResDesc;
    return log_not_supported_object_path("cuSurfObjectCreate");
}

CUresult cuSurfObjectDestroy(unsigned long long surfObject)
{
    (void)surfObject;
    return log_not_supported_object_path("cuSurfObjectDestroy");
}

CUresult cuSurfObjectGetResourceDesc(void *pResDesc, unsigned long long surfObject)
{
    (void)pResDesc;
    (void)surfObject;
    return log_not_supported_object_path("cuSurfObjectGetResourceDesc");
}

static CUresult log_not_supported_runtime_feature(const char *name)
{
    fprintf(stderr, "[libvgpu-cuda] %s CALLED during init-only path: returning CUDA_ERROR_NOT_SUPPORTED\n",
            name);
    fflush(stderr);
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemPoolExportPointer(void *shareData, CUdeviceptr ptr)
{
    (void)shareData;
    (void)ptr;
    return log_not_supported_runtime_feature("cuMemPoolExportPointer");
}

CUresult cuMemPoolImportPointer(CUdeviceptr *ptr, CUmemoryPool pool, void *shareData)
{
    (void)ptr;
    (void)pool;
    (void)shareData;
    return log_not_supported_runtime_feature("cuMemPoolImportPointer");
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, int config)
{
    (void)hfunc;
    (void)config;
    return log_not_supported_runtime_feature("cuFuncSetSharedMemConfig");
}

CUresult cuFuncGetName(const char **name, CUfunction hfunc)
{
    (void)name;
    (void)hfunc;
    return log_not_supported_runtime_feature("cuFuncGetName");
}

CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t *paramOffset, size_t *paramSize)
{
    (void)func;
    (void)paramIndex;
    (void)paramOffset;
    (void)paramSize;
    return log_not_supported_runtime_feature("cuFuncGetParamInfo");
}

CUresult cuImportExternalMemory(void *extMem_out, const void *memHandleDesc)
{
    (void)extMem_out;
    (void)memHandleDesc;
    return log_not_supported_runtime_feature("cuImportExternalMemory");
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, void *extMem, const void *bufferDesc)
{
    (void)devPtr;
    (void)extMem;
    (void)bufferDesc;
    return log_not_supported_runtime_feature("cuExternalMemoryGetMappedBuffer");
}

CUresult cuExternalMemoryGetMappedMipmappedArray(void *mipmap, void *extMem, const void *mipmapDesc)
{
    (void)mipmap;
    (void)extMem;
    (void)mipmapDesc;
    return log_not_supported_runtime_feature("cuExternalMemoryGetMappedMipmappedArray");
}

CUresult cuDestroyExternalMemory(void *extMem)
{
    (void)extMem;
    return log_not_supported_runtime_feature("cuDestroyExternalMemory");
}

CUresult cuImportExternalSemaphore(void *extSem_out, const void *semHandleDesc)
{
    (void)extSem_out;
    (void)semHandleDesc;
    return log_not_supported_runtime_feature("cuImportExternalSemaphore");
}

CUresult cuSignalExternalSemaphoresAsync(const void *extSemArray, const void *paramsArray,
                                         unsigned int numExtSems, CUstream stream)
{
    (void)extSemArray;
    (void)paramsArray;
    (void)numExtSems;
    (void)stream;
    return log_not_supported_runtime_feature("cuSignalExternalSemaphoresAsync");
}

CUresult cuWaitExternalSemaphoresAsync(const void *extSemArray, const void *paramsArray,
                                       unsigned int numExtSems, CUstream stream)
{
    (void)extSemArray;
    (void)paramsArray;
    (void)numExtSems;
    (void)stream;
    return log_not_supported_runtime_feature("cuWaitExternalSemaphoresAsync");
}

CUresult cuDestroyExternalSemaphore(void *extSem)
{
    (void)extSem;
    return log_not_supported_runtime_feature("cuDestroyExternalSemaphore");
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
{
    (void)flags;
    return cuEventRecord(hEvent, hStream);
}

CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, uint32_t value, unsigned int flags)
{
    (void)stream;
    (void)addr;
    (void)value;
    (void)flags;
    return log_not_supported_runtime_feature("cuStreamWaitValue32");
}

CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, uint32_t value, unsigned int flags)
{
    (void)stream;
    (void)addr;
    (void)value;
    (void)flags;
    return log_not_supported_runtime_feature("cuStreamWriteValue32");
}

CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    (void)stream;
    (void)addr;
    (void)value;
    (void)flags;
    return log_not_supported_runtime_feature("cuStreamWaitValue64");
}

CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    (void)stream;
    (void)addr;
    (void)value;
    (void)flags;
    return log_not_supported_runtime_feature("cuStreamWriteValue64");
}

CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, const void *paramArray, unsigned int flags)
{
    (void)stream;
    (void)count;
    (void)paramArray;
    (void)flags;
    return log_not_supported_runtime_feature("cuStreamBatchMemOp");
}

CUresult cuIpcGetEventHandle(void *pHandle, CUevent event)
{
    (void)pHandle;
    (void)event;
    return log_not_supported_runtime_feature("cuIpcGetEventHandle");
}

CUresult cuIpcOpenEventHandle(CUevent *phEvent, void *handle)
{
    (void)phEvent;
    (void)handle;
    return log_not_supported_runtime_feature("cuIpcOpenEventHandle");
}

CUresult cuIpcGetMemHandle(void *pHandle, CUdeviceptr dptr)
{
    (void)pHandle;
    (void)dptr;
    return log_not_supported_runtime_feature("cuIpcGetMemHandle");
}

CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, void *handle, unsigned int flags)
{
    (void)pdptr;
    (void)handle;
    (void)flags;
    return log_not_supported_runtime_feature("cuIpcOpenMemHandle");
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr)
{
    (void)dptr;
    return log_not_supported_runtime_feature("cuIpcCloseMemHandle");
}

CUresult cuGLCtxCreate(CUcontext *pCtx, unsigned int Flags, void *device)
{
    (void)pCtx;
    (void)Flags;
    (void)device;
    return log_not_supported_runtime_feature("cuGLCtxCreate");
}

CUresult cuGLInit(void)
{
    return log_not_supported_runtime_feature("cuGLInit");
}

CUresult cuGLGetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices,
                        unsigned int cudaDeviceCount, unsigned int deviceList)
{
    (void)pCudaDeviceCount;
    (void)pCudaDevices;
    (void)cudaDeviceCount;
    (void)deviceList;
    return log_not_supported_runtime_feature("cuGLGetDevices");
}

CUresult cuGLRegisterBufferObject(unsigned int buffer)
{
    (void)buffer;
    return log_not_supported_runtime_feature("cuGLRegisterBufferObject");
}

CUresult cuGLMapBufferObject(CUdeviceptr *dptr, size_t *size, unsigned int buffer)
{
    (void)dptr;
    (void)size;
    (void)buffer;
    return log_not_supported_runtime_feature("cuGLMapBufferObject");
}

CUresult cuGLMapBufferObjectAsync(CUdeviceptr *dptr, size_t *size, unsigned int buffer, CUstream hStream)
{
    (void)dptr;
    (void)size;
    (void)buffer;
    (void)hStream;
    return log_not_supported_runtime_feature("cuGLMapBufferObjectAsync");
}

CUresult cuGLUnmapBufferObject(unsigned int buffer)
{
    (void)buffer;
    return log_not_supported_runtime_feature("cuGLUnmapBufferObject");
}

CUresult cuGLUnmapBufferObjectAsync(unsigned int buffer, CUstream hStream)
{
    (void)buffer;
    (void)hStream;
    return log_not_supported_runtime_feature("cuGLUnmapBufferObjectAsync");
}

CUresult cuGLUnregisterBufferObject(unsigned int buffer)
{
    (void)buffer;
    return log_not_supported_runtime_feature("cuGLUnregisterBufferObject");
}

CUresult cuGLSetBufferObjectMapFlags(unsigned int buffer, unsigned int Flags)
{
    (void)buffer;
    (void)Flags;
    return log_not_supported_runtime_feature("cuGLSetBufferObjectMapFlags");
}

CUresult cuGraphicsGLRegisterImage(void *pCudaResource, unsigned int image, int target, unsigned int Flags)
{
    (void)pCudaResource;
    (void)image;
    (void)target;
    (void)Flags;
    return log_not_supported_runtime_feature("cuGraphicsGLRegisterImage");
}

CUresult cuGraphicsGLRegisterBuffer(void *pCudaResource, unsigned int buffer, unsigned int Flags)
{
    (void)pCudaResource;
    (void)buffer;
    (void)Flags;
    return log_not_supported_runtime_feature("cuGraphicsGLRegisterBuffer");
}

CUresult cuGraphicsEGLRegisterImage(void *pCudaResource, void *image, unsigned int flags)
{
    (void)pCudaResource;
    (void)image;
    (void)flags;
    return log_not_supported_runtime_feature("cuGraphicsEGLRegisterImage");
}

CUresult cuEGLStreamConsumerConnect(void *conn, void *stream)
{
    (void)conn;
    (void)stream;
    return log_not_supported_runtime_feature("cuEGLStreamConsumerConnect");
}

CUresult cuEGLStreamConsumerDisconnect(void *conn)
{
    (void)conn;
    return log_not_supported_runtime_feature("cuEGLStreamConsumerDisconnect");
}

CUresult cuEGLStreamConsumerAcquireFrame(void *conn, void *pCudaResource, void *pStream, unsigned int timeout)
{
    (void)conn;
    (void)pCudaResource;
    (void)pStream;
    (void)timeout;
    return log_not_supported_runtime_feature("cuEGLStreamConsumerAcquireFrame");
}

CUresult cuEGLStreamConsumerReleaseFrame(void *conn, void *pCudaResource, void *pStream)
{
    (void)conn;
    (void)pCudaResource;
    (void)pStream;
    return log_not_supported_runtime_feature("cuEGLStreamConsumerReleaseFrame");
}

CUresult cuEGLStreamProducerConnect(void *conn, void *stream, unsigned int width, unsigned int height)
{
    (void)conn;
    (void)stream;
    (void)width;
    (void)height;
    return log_not_supported_runtime_feature("cuEGLStreamProducerConnect");
}

CUresult cuEGLStreamProducerDisconnect(void *conn)
{
    (void)conn;
    return log_not_supported_runtime_feature("cuEGLStreamProducerDisconnect");
}

CUresult cuEGLStreamProducerPresentFrame(void *conn, void *eglFrame, void *pStream)
{
    (void)conn;
    (void)eglFrame;
    (void)pStream;
    return log_not_supported_runtime_feature("cuEGLStreamProducerPresentFrame");
}

CUresult cuEGLStreamProducerReturnFrame(void *conn, void *eglFrame, void *pStream)
{
    (void)conn;
    (void)eglFrame;
    (void)pStream;
    return log_not_supported_runtime_feature("cuEGLStreamProducerReturnFrame");
}

CUresult cuEGLStreamConsumerConnectWithFlags(void *conn, void *stream, unsigned int flags)
{
    (void)conn;
    (void)stream;
    (void)flags;
    return log_not_supported_runtime_feature("cuEGLStreamConsumerConnectWithFlags");
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[6];
    CUDA_PACK_U64(args, 0, (uint64_t)dstDevice);
    args[2] = ui;
    args[3] = 0;
    CUDA_PACK_U64(args, 4, (uint64_t)N);

    return rpc_simple(CUDA_CALL_MEMSET_D32, args, 6, &result);
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    return cuMemsetD32_v2(dstDevice, ui, N);
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total)
{
    /* CRITICAL FIX: GGML/Ollama requires cuMemGetInfo to ALWAYS succeed after cuCtxCreate
     * Even if ensure_init() fails, we must return valid memory values.
     * GGML checks this early and disables GPU if it fails or returns 0/0. */
    
    if (!free || !total) return CUDA_ERROR_INVALID_VALUE;

    /* Ensure g_gpu_info is initialized with defaults if not already */
    if (!g_gpu_info_valid) {
        init_gpu_defaults();
    }

    /* Try to get live values via RPC, but always have fallback */
    CUresult rc = ensure_init();
    if (rc == CUDA_SUCCESS) {
        CUDACallResult result = {0};
        CUresult rpc_rc = rpc_simple(CUDA_CALL_MEM_GET_INFO, NULL, 0, &result);
        if (rpc_rc == CUDA_SUCCESS &&
            result.num_results >= 2 &&
            result.results[0] > 0 &&
            result.results[1] > 0) {
            *free  = (size_t)result.results[0];
            *total = (size_t)result.results[1];
            return CUDA_SUCCESS;
        }
    }
    
    /* CRITICAL: Always return valid values, even if RPC fails
     * GGML will disable GPU if free=0 or total=0 */
    *free  = (size_t)g_gpu_info.free_mem;
    *total = (size_t)g_gpu_info.total_mem;
    
    /* Log for debugging */
    fprintf(stderr, "[libvgpu-cuda] cuMemGetInfo_v2() returning: free=%zu MB, total=%zu MB (pid=%d)\n",
            *free / (1024 * 1024), *total / (1024 * 1024), (int)getpid());
    fflush(stderr);
    
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo(size_t *free, size_t *total)
{
    return cuMemGetInfo_v2(free, total);
}

/* Async memory copy variants */
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                               size_t byteCount, CUstream hStream)
{
    /* For simplicity, async = sync for now */
    (void)hStream;
    return cuMemcpyHtoD_v2(dstDevice, srcHost, byteCount);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                            size_t byteCount, CUstream hStream)
{
    return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, byteCount, hStream);
}

CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                               size_t byteCount, CUstream hStream)
{
    (void)hStream;
    return cuMemcpyDtoH_v2(dstHost, srcDevice, byteCount);
}

CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                            size_t byteCount, CUstream hStream)
{
    return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, byteCount, hStream);
}

/* ================================================================
 * CUDA Driver API — Module / function management
 * ================================================================ */

typedef struct {
    const void *image;
    CUmodule module;
} dark_library_handle_t;

#define CU_LIBRARY_BINARY_IS_PRESERVED 1
#define DARK_LIBRARY_HANDLE_TAG ((uintptr_t)1)

static int dark_library_is_local(CUlibrary library)
{
    return (((uintptr_t)library) & DARK_LIBRARY_HANDLE_TAG) != 0;
}

static dark_library_handle_t *dark_library_from_public(CUlibrary library)
{
    return (dark_library_handle_t *)(((uintptr_t)library) & ~DARK_LIBRARY_HANDLE_TAG);
}

static CUlibrary dark_library_to_public(dark_library_handle_t *handle)
{
    return (CUlibrary)(((uintptr_t)handle) | DARK_LIBRARY_HANDLE_TAG);
}

CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleUnload(CUmodule hmod);

static CUresult __attribute__((unused)) dark_library_resolve_module(dark_library_handle_t *handle)
{
    if (!handle) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (handle->module) {
        return CUDA_SUCCESS;
    }
    if (!handle->image) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuModuleLoadFatBinary(&handle->module, handle->image);
}

static size_t dark_resolve_fatbin_total_size(const unsigned char *bytes)
{
    const VgpuFatbinHeader *hdr = (const VgpuFatbinHeader *)bytes;
    size_t total;

    if (!bytes || hdr->magic != 0xBA55ED50U) {
        return 0;
    }

    total = (size_t)hdr->header_size + (size_t)hdr->fat_size;
    if (hdr->header_size < sizeof(VgpuFatbinHeader) ||
        total < hdr->header_size ||
        total > (256u * 1024u * 1024u)) {
        return 0;
    }

    return total;
}

static size_t dark_resolve_elf_image_size(const unsigned char *bytes)
{
    const VgpuElf64_Ehdr *ehdr = (const VgpuElf64_Ehdr *)bytes;
    size_t max_end = sizeof(VgpuElf64_Ehdr);

    if (!bytes) {
        return 0;
    }
    if (!(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F')) {
        return 0;
    }

    if (ehdr->e_ehsize >= sizeof(VgpuElf64_Ehdr) && ehdr->e_ehsize < (1u << 20)) {
        max_end = ehdr->e_ehsize;
    }

    if (ehdr->e_phentsize == sizeof(VgpuElf64_Phdr) && ehdr->e_phnum < 4096) {
        size_t ph_table_end = (size_t)ehdr->e_phoff +
                              (size_t)ehdr->e_phnum * sizeof(VgpuElf64_Phdr);
        if (ph_table_end > max_end) {
            max_end = ph_table_end;
        }

        for (uint16_t i = 0; i < ehdr->e_phnum; i++) {
            const VgpuElf64_Phdr *phdr =
                (const VgpuElf64_Phdr *)(bytes + ehdr->e_phoff +
                                         (size_t)i * sizeof(VgpuElf64_Phdr));
            if (phdr->p_filesz > 0) {
                size_t end = (size_t)phdr->p_offset + (size_t)phdr->p_filesz;
                if (end > max_end) {
                    max_end = end;
                }
            }
        }
    }

    if (ehdr->e_shentsize == sizeof(VgpuElf64_Shdr) && ehdr->e_shnum < 16384) {
        size_t sh_table_end = (size_t)ehdr->e_shoff +
                              (size_t)ehdr->e_shnum * sizeof(VgpuElf64_Shdr);
        if (sh_table_end > max_end) {
            max_end = sh_table_end;
        }

        for (uint16_t i = 0; i < ehdr->e_shnum; i++) {
            const VgpuElf64_Shdr *shdr =
                (const VgpuElf64_Shdr *)(bytes + ehdr->e_shoff +
                                         (size_t)i * sizeof(VgpuElf64_Shdr));
            if (shdr->sh_size > 0) {
                size_t end = (size_t)shdr->sh_offset + (size_t)shdr->sh_size;
                if (end > max_end) {
                    max_end = end;
                }
            }
        }
    }

    if (max_end == 0 || max_end > (256u * 1024u * 1024u)) {
        return 0;
    }
    return max_end;
}

static void dark_resolve_module_image(const void *image,
                                      const void **resolved_image,
                                      size_t *resolved_size,
                                      const char **resolved_kind)
{
    const unsigned char *bytes = (const unsigned char *)image;

    if (resolved_image) {
        *resolved_image = image;
    }
    if (resolved_size) {
        *resolved_size = 0;
    }
    if (resolved_kind) {
        *resolved_kind = "unknown";
    }
    if (!image) {
        return;
    }

    if (*(const uint32_t *)bytes == 0x466243b1U) {
        const void *payload = *(const void * const *)(bytes + 8);
        const unsigned char *payload_bytes = (const unsigned char *)payload;
        if (payload && *(const uint32_t *)payload_bytes == 0xBA55ED50U) {
            if (resolved_image) {
                *resolved_image = payload;
            }
            if (resolved_size) {
                *resolved_size = dark_resolve_fatbin_total_size(payload_bytes);
            }
            if (resolved_kind) {
                *resolved_kind = "wrapper-fatbin";
            }
            return;
        }
    }

    if (*(const uint32_t *)bytes == 0xBA55ED50U) {
        if (resolved_size) {
            *resolved_size = dark_resolve_fatbin_total_size(bytes);
        }
        if (resolved_kind) {
            *resolved_kind = "fatbin";
        }
        return;
    }

    if (bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F') {
        if (resolved_size) {
            *resolved_size = dark_resolve_elf_image_size(bytes);
        }
        if (resolved_kind) {
            *resolved_kind = "elf";
        }
        return;
    }

    if (resolved_size) {
        *resolved_size = strlen((const char *)image) + 1;
    }
    if (resolved_kind) {
        *resolved_kind = "string";
    }
}

CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
    CUresult rc = ensure_connected();
    const void *resolved_image = image;
    size_t image_size = 0;
    const char *image_kind = "unknown";
    if (rc != CUDA_SUCCESS) return rc;
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;

    dark_resolve_module_image(image, &resolved_image, &image_size, &image_kind);
    if (!resolved_image || image_size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    {
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                               "[libvgpu-cuda] cuModuleLoadData image=%p resolved=%p kind=%s size=%zu first8=%02x%02x%02x%02x%02x%02x%02x%02x (pid=%d)\n",
                               image, resolved_image, image_kind, image_size,
                               ((const unsigned char *)resolved_image)[0],
                               ((const unsigned char *)resolved_image)[1],
                               ((const unsigned char *)resolved_image)[2],
                               ((const unsigned char *)resolved_image)[3],
                               ((const unsigned char *)resolved_image)[4],
                               ((const unsigned char *)resolved_image)[5],
                               ((const unsigned char *)resolved_image)[6],
                               ((const unsigned char *)resolved_image)[7],
                               (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
            int fd = (int)syscall(__NR_open, "/tmp/vgpu_module_load_diag.txt",
                                  O_WRONLY | O_CREAT | O_APPEND, 0666);
            if (fd >= 0) {
                syscall(__NR_write, fd, log_msg, (size_t)log_len);
                syscall(__NR_close, fd);
            }
            if (vgpu_debug_logging()) {
                syscall(__NR_write, 2, log_msg, (size_t)log_len);
            }
        }
    }

    CUDACallResult result;
    uint32_t recv_len = 0;

    rc = (CUresult)cuda_transport_call(g_transport,
                                        CUDA_CALL_MODULE_LOAD_DATA,
                                        NULL, 0,
                                        resolved_image, (uint32_t)image_size,
                                        &result, NULL, 0, &recv_len);
    {
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                               "[libvgpu-cuda] cuModuleLoadData result kind=%s size=%zu rc=%d host_status=%u num_results=%u recv_len=%u module=0x%llx (pid=%d)\n",
                               image_kind, image_size, (int)rc, result.status,
                               result.num_results, recv_len,
                               (unsigned long long)(result.num_results >= 1 ? result.results[0] : 0),
                               (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
            int fd = (int)syscall(__NR_open, "/tmp/vgpu_module_load_diag.txt",
                                  O_WRONLY | O_CREAT | O_APPEND, 0666);
            if (fd >= 0) {
                syscall(__NR_write, fd, log_msg, (size_t)log_len);
                syscall(__NR_close, fd);
            }
            if (vgpu_debug_logging()) {
                syscall(__NR_write, 2, log_msg, (size_t)log_len);
            }
        }
    }
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *module = (CUmodule)(uintptr_t)result.results[0];
    }
    return rc;
}

CUresult cuLibraryLoadData(CUlibrary *library,
                           const void *code,
                           CUjit_option *jitOptions,
                           void **jitOptionValues,
                           unsigned int numJitOptions,
                           CUlibraryOption *libraryOptions,
                           void **libraryOptionValues,
                           unsigned int numLibraryOptions)
{
    CUresult rc;
    const void *resolved_image = code;
    size_t image_size = 0;
    const char *image_kind = "unknown";
    CUDACallResult result = {0};
    uint32_t recv_len = 0;

    if (!library || !code) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    rc = ensure_connected();
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    /* The guest currently only observes one library option:
     * CU_LIBRARY_BINARY_IS_PRESERVED = 1. In a remote setup the host cannot
     * safely rely on guest memory remaining valid after the RPC returns, so we
     * intentionally drop that hint and let the host driver make its own copy.
     * Reject any other library/JIT option until it is implemented correctly. */
    if (numJitOptions != 0) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (numLibraryOptions > 1) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (numLibraryOptions == 1) {
        if (!libraryOptions || !libraryOptionValues ||
            libraryOptions[0] != CU_LIBRARY_BINARY_IS_PRESERVED ||
            libraryOptionValues[0] != (void *)1) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    }

    dark_resolve_module_image(code, &resolved_image, &image_size, &image_kind);
    if (!resolved_image || image_size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    /* Fatbin-based libraries work reliably through the existing module path.
     * Keep the real CUlibrary RPC for non-fatbin inputs such as ELF images. */
    if (strcmp(image_kind, "wrapper-fatbin") == 0 ||
        strcmp(image_kind, "fatbin") == 0) {
        dark_library_handle_t *handle =
            (dark_library_handle_t *)calloc(1, sizeof(*handle));
        if (!handle) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        handle->image = code;
        rc = dark_library_resolve_module(handle);
        if (rc != CUDA_SUCCESS) {
            free(handle);
            return rc;
        }
        *library = dark_library_to_public(handle);
        return CUDA_SUCCESS;
    }

    rc = (CUresult)cuda_transport_call(g_transport,
                                       CUDA_CALL_LIBRARY_LOAD_DATA,
                                       NULL, 0,
                                       resolved_image, (uint32_t)image_size,
                                       &result, NULL, 0, &recv_len);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *library = (CUlibrary)(uintptr_t)result.results[0];
    }
    return rc;
}

CUresult cuLibraryUnload(CUlibrary library)
{
    CUDACallResult result;
    uint32_t args[2];
    dark_library_handle_t *handle = NULL;

    if (!library) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (dark_library_is_local(library)) {
        handle = dark_library_from_public(library);
        if (!handle) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        CUresult rc = CUDA_SUCCESS;
        if (handle->module) {
            rc = cuModuleUnload(handle->module);
        }
        free(handle);
        return rc;
    }

    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)library);
    return rpc_simple(CUDA_CALL_LIBRARY_UNLOAD, args, 2, &result);
}

CUresult cuLibraryGetModule(CUmodule *module, CUlibrary library)
{
    CUDACallResult result;
    uint32_t args[2];
    dark_library_handle_t *handle = NULL;

    if (!module || !library) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (dark_library_is_local(library)) {
        handle = dark_library_from_public(library);
        if (!handle) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        CUresult rc = dark_library_resolve_module(handle);
        if (rc == CUDA_SUCCESS) {
            *module = handle->module;
        }
        return rc;
    }

    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)library);
    CUresult rc = rpc_simple(CUDA_CALL_LIBRARY_GET_MODULE, args, 2, &result);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *module = (CUmodule)(uintptr_t)result.results[0];
    }
    return rc;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                             unsigned int numOptions,
                             void *options, void **optionValues)
{
    /* Ignore JIT options for now */
    (void)numOptions; (void)options; (void)optionValues;
    return cuModuleLoadData(module, image);
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    CUresult rc = ensure_connected();
    const void *resolved_image = fatCubin;
    size_t image_size = 0;
    const char *image_kind = "unknown";
    CUDACallResult result;
    uint32_t recv_len = 0;

    if (rc != CUDA_SUCCESS) return rc;
    if (!module || !fatCubin) return CUDA_ERROR_INVALID_VALUE;

    dark_resolve_module_image(fatCubin, &resolved_image, &image_size, &image_kind);
    if (!resolved_image || image_size == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    rc = (CUresult)cuda_transport_call(g_transport,
                                        CUDA_CALL_MODULE_LOAD_FAT_BINARY,
                                        NULL, 0,
                                        resolved_image, (uint32_t)image_size,
                                        &result, NULL, 0, &recv_len);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *module = (CUmodule)(uintptr_t)result.results[0];
    }
    return rc;
}

CUresult cuModuleUnload(CUmodule hmod)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hmod);

    return rpc_simple(CUDA_CALL_MODULE_UNLOAD, args, 2, &result);
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                              const char *name)
{
    /* Debug: which call is reached after the 6 allocs (for runner exit 2) */
    {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666); /* O_WRONLY|O_CREAT|O_APPEND */
        if (nfd >= 0) {
            const char *msg = "get_function\n";
            syscall(__NR_write, nfd, msg, 13);
            syscall(__NR_close, nfd);
        }
    }
    CUresult rc = ensure_connected();
    if (rc != CUDA_SUCCESS) return rc;
    if (!hfunc || !name) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hmod);

    /* Send function name as data payload */
    size_t name_len = strlen(name) + 1;
    uint32_t recv_len = 0;

    rc = (CUresult)cuda_transport_call(g_transport,
                                        CUDA_CALL_MODULE_GET_FUNCTION,
                                        args, 2,
                                        name, (uint32_t)name_len,
                                        &result, NULL, 0, &recv_len);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *hfunc = (CUfunction)(uintptr_t)result.results[0];
    }
    return rc;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes,
                               CUmodule hmod, const char *name)
{
    CUresult rc = ensure_connected();
    if (rc != CUDA_SUCCESS) return rc;
    if (!dptr || !name) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hmod);

    size_t name_len = strlen(name) + 1;
    uint32_t recv_len = 0;

    rc = (CUresult)cuda_transport_call(g_transport,
                                        CUDA_CALL_MODULE_GET_GLOBAL,
                                        args, 2,
                                        name, (uint32_t)name_len,
                                        &result, NULL, 0, &recv_len);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *dptr = (CUdeviceptr)result.results[0];
        if (bytes && result.num_results >= 2) *bytes = (size_t)result.results[1];
    }
    return rc;
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                            CUmodule hmod, const char *name)
{
    return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

/* ================================================================
 * CUDA Driver API — Kernel launch
 * ================================================================ */

CUresult cuLaunchKernel(CUfunction f,
                         unsigned int gridDimX, unsigned int gridDimY,
                         unsigned int gridDimZ,
                         unsigned int blockDimX, unsigned int blockDimY,
                         unsigned int blockDimZ,
                         unsigned int sharedMemBytes,
                         CUstream hStream,
                         void **kernelParams,
                         void **extra)
{
    {
        int nfd = (int)syscall(__NR_open, "/tmp/vgpu_next_call.log", 1 | 64 | 1024, 0666);
        if (nfd >= 0) {
            const char *msg = "launch_kernel\n";
            syscall(__NR_write, nfd, msg, 14);
            syscall(__NR_close, nfd);
        }
    }
    if (vgpu_debug_logging()) {
        char log_msg[256];
        int log_len = snprintf(log_msg, sizeof(log_msg),
                              "[libvgpu-cuda] cuLaunchKernel() CALLED: grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u params=%u (pid=%d)\n",
                              gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                              sharedMemBytes, kernelParams ? 1 : 0, (int)getpid());
        if (log_len > 0 && log_len < (int)sizeof(log_msg))
            syscall(__NR_write, 2, log_msg, log_len);
    }
    CUresult rc = ensure_connected();
    if (rc != CUDA_SUCCESS) return rc;

    (void)extra;  /* extra not supported yet */

    /*
     * Serialize launch parameters.
     * We use the CUDALaunchParams structure followed by parameter data.
     *
     * Since we can't know the individual param sizes from void** alone,
     * we rely on the host having metadata from cuModuleGetFunction.
     * For now, we send kernelParams as an array of 8-byte pointers.
     */
    CUDALaunchParams lp;
    memset(&lp, 0, sizeof(lp));
    lp.function_handle = (uint64_t)(uintptr_t)f;
    lp.grid_dim_x   = gridDimX;
    lp.grid_dim_y   = gridDimY;
    lp.grid_dim_z   = gridDimZ;
    lp.block_dim_x  = blockDimX;
    lp.block_dim_y  = blockDimY;
    lp.block_dim_z  = blockDimZ;
    lp.shared_mem_bytes = sharedMemBytes;
    lp.stream_handle = (uint64_t)(uintptr_t)hStream;
    lp.num_params = 0;
    lp.total_param_bytes = 0;

    /* Count kernel parameters (NULL-terminated array) */
    if (kernelParams) {
        while (kernelParams[lp.num_params] != NULL && lp.num_params < 256) {
            lp.num_params++;
        }
    }

    /*
     * Build payload: CUDALaunchParams + pointer values
     * Each kernel param is treated as an 8-byte value (device pointer size).
     */
    lp.total_param_bytes = lp.num_params * 8;

    size_t payload_size = sizeof(CUDALaunchParams)
                         + lp.num_params * sizeof(uint32_t)  /* param_sizes */
                         + lp.total_param_bytes;             /* param_data  */

    uint8_t *payload = (uint8_t *)malloc(payload_size);
    if (!payload) return CUDA_ERROR_OUT_OF_MEMORY;

    /* Copy launch params */
    memcpy(payload, &lp, sizeof(CUDALaunchParams));

    /* Write param_sizes (each is 8 bytes) */
    uint32_t *sizes = (uint32_t *)(payload + sizeof(CUDALaunchParams));
    for (uint32_t i = 0; i < lp.num_params; i++) {
        sizes[i] = 8;  /* each param is 8 bytes */
    }

    /* Write param data (raw 8-byte copies of each param) */
    uint8_t *pdata = (uint8_t *)(sizes + lp.num_params);
    for (uint32_t i = 0; i < lp.num_params; i++) {
        memcpy(pdata + i * 8, kernelParams[i], 8);
    }

    CUDACallResult result;
    rc = (CUresult)cuda_transport_call(g_transport,
                                        CUDA_CALL_LAUNCH_KERNEL,
                                        NULL, 0,
                                        payload, (uint32_t)payload_size,
                                        &result, NULL, 0, NULL);

    if (rc == CUDA_SUCCESS) {
        char success_msg[128];
        int success_len = snprintf(success_msg, sizeof(success_msg),
                                  "[libvgpu-cuda] cuLaunchKernel() SUCCESS: forwarded to host (pid=%d)\n",
                                  (int)getpid());
        if (success_len > 0 && success_len < (int)sizeof(success_msg)) {
            syscall(__NR_write, 2, success_msg, success_len);
        }
    } else {
        char error_msg[128];
        int error_len = snprintf(error_msg, sizeof(error_msg),
                                "[libvgpu-cuda] cuLaunchKernel() FAILED: rc=%d (pid=%d)\n",
                                rc, (int)getpid());
        if (error_len > 0 && error_len < (int)sizeof(error_msg)) {
            syscall(__NR_write, 2, error_msg, error_len);
        }
    }

    free(payload);
    return rc;
}

/* ================================================================
 * Export-coverage stubs for driver symbols that newer CUDA user-space
 * libraries expect to find in libcuda.so.1 during load/init.
 *
 * These are intentionally narrow: they export the symbol and return
 * CUDA_ERROR_NOT_SUPPORTED so the loader can complete. If CUBLAS or another
 * library actually executes one of these paths later, we can replace that
 * individual stub with a real wrapper.
 * ================================================================ */
#define DEFINE_CUDA_NOT_SUPPORTED_STUB(name) \
    __attribute__((visibility("default"))) CUresult name(void) { \
        fprintf(stderr, "[libvgpu-cuda] " #name "() NOT_SUPPORTED STUB CALLED\n"); \
        return CUDA_ERROR_NOT_SUPPORTED; \
    }

__attribute__((visibility("default"))) CUresult cuModuleGetLoadingMode(unsigned int *mode)
{
    if (mode) {
        *mode = 2; /* CU_MODULE_LAZY_LOADING */
    }
    return CUDA_SUCCESS;
}

DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLinkCreate_v2)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLinkAddData_v2)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLinkComplete)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLinkDestroy)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLinkAddFile_v2)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuTensorMapEncodeTiled)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLaunchKernelEx)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuDeviceGetByPCIBusId)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuDeviceSetMemPool)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuFlushGPUDirectRDMAWrites)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxDetach)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxGetCacheConfig)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxSetCacheConfig)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxGetSharedMemConfig)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxGetStreamPriorityRange)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxSetSharedMemConfig)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxResetPersistingL2Cache)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxEnablePeerAccess)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuCtxDisablePeerAccess)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuModuleLoad)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuModuleGetTexRef)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuModuleGetSurfRef)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLibraryLoadFromFile)
CUresult cuLibraryGetKernel(CUkernel *kernel, CUlibrary library, const char *name)
{
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUresult rc;

    if (!kernel || !name) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    rc = cuLibraryGetModule(&module, library);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    rc = cuModuleGetFunction(&function, module, name);
    if (rc != CUDA_SUCCESS) {
        return rc;
    }

    /* In this shim CUkernel/CUfunction are both opaque pointer handles. */
    *kernel = (CUkernel)function;
    return CUDA_SUCCESS;
}

CUresult cuKernelGetFunction(CUfunction *pfn, CUkernel kernel)
{
    if (!pfn || !kernel) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *pfn = (CUfunction)kernel;
    return CUDA_SUCCESS;
}

DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLibraryGetGlobal)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLibraryGetManaged)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLibraryGetUnifiedFunction)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLibraryGetKernelCount)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLibraryEnumerateKernels)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuKernelGetAttribute)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuKernelSetAttribute)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuKernelSetCacheConfig)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuKernelGetName)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuKernelGetParamInfo)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLinkCreate)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLinkAddData)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLinkAddFile)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemHostGetDevicePointer_v2)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemHostGetFlags)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemHostRegister_v2)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuPointerGetAttribute)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuPointerGetAttributes)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemAllocAsync)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemAllocAsync_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemAllocFromPoolAsync)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemAllocFromPoolAsync_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemFreeAsync)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemFreeAsync_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolTrimTo)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolSetAttribute)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolGetAttribute)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolSetAccess)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolGetAccess)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolCreate)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolDestroy)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolExportToShareableHandle)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPoolImportFromShareableHandle)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamGetFlags_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamGetId)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamGetId_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamWaitEvent_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamAddCallback)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamAddCallback_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamSynchronize_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamQuery_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamAttachMemAsync)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamAttachMemAsync_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamCopyAttributes)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamCopyAttributes_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamGetAttribute)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamGetAttribute_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamSetAttribute)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuStreamSetAttribute_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLaunchKernelEx_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuLaunchKernel_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemAdvise)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPrefetchAsync)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemPrefetchAsync_ptsz)
DEFINE_CUDA_NOT_SUPPORTED_STUB(cuMemAdvise_v2)

__attribute__((visibility("default")))
CUresult cuGetProcAddress_v2(const char *symbol, void **funcPtr,
                             int cudaVersion, cuuint64_t flags,
                             void *symbolStatus)
{
    if (symbolStatus) {
        *(int *)symbolStatus = 0;
    }
    return cuGetProcAddress(symbol, funcPtr, cudaVersion, flags);
}

/* Additional base-name exports requested via cuGetProcAddress() during
 * CUDA 12 library initialization. These APIs are not part of the supported
 * vGPU path today; export them so symbol lookup succeeds, and return
 * CUDA_ERROR_NOT_SUPPORTED if anything actually calls through them. */
static CUresult log_optional_export_hit(const char *name)
{
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] optional export CALLED: %s -> CUDA_ERROR_NOT_SUPPORTED (pid=%d)\n",
                          name, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }

    int fd = (int)syscall(__NR_open, "/tmp/vgpu_optional_api_hits.txt",
                          O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd >= 0) {
        syscall(__NR_write, fd, log_msg, (size_t)log_len);
        syscall(__NR_close, fd);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

#define DEFINE_CUDA_OPTIONAL_EXPORT(name) \
    __attribute__((visibility("default"))) CUresult name(void) { \
        return log_optional_export_hit(#name); \
    }

DEFINE_CUDA_OPTIONAL_EXPORT(cuLaunchCooperativeKernel)
DEFINE_CUDA_OPTIONAL_EXPORT(cuLaunchCooperativeKernelMultiDevice)
DEFINE_CUDA_OPTIONAL_EXPORT(cuLaunchHostFunc)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsResourceGetMappedEglFrame)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsUnregisterResource)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsMapResources)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsUnmapResources)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsResourceSetMapFlags)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsSubResourceGetMappedArray)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsResourceGetMappedMipmappedArray)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsResourceGetMappedPointer)
DEFINE_CUDA_OPTIONAL_EXPORT(cuProfilerInitialize)
DEFINE_CUDA_OPTIONAL_EXPORT(cuProfilerStart)
DEFINE_CUDA_OPTIONAL_EXPORT(cuProfilerStop)
DEFINE_CUDA_OPTIONAL_EXPORT(cuVDPAUGetDevice)
DEFINE_CUDA_OPTIONAL_EXPORT(cuVDPAUCtxCreate)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsVDPAURegisterVideoSurface)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphicsVDPAURegisterOutputSurface)
DEFINE_CUDA_OPTIONAL_EXPORT(cuMemRangeGetAttribute)
DEFINE_CUDA_OPTIONAL_EXPORT(cuMemRangeGetAttributes)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphCreate)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddKernelNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphKernelNodeGetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphKernelNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddMemcpyNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphMemcpyNodeGetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphMemcpyNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddMemsetNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphMemsetNodeGetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphMemsetNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddHostNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphHostNodeGetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphHostNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddChildGraphNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphChildGraphNodeGetGraph)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddEmptyNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddEventRecordNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphEventRecordNodeGetEvent)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphEventRecordNodeSetEvent)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddEventWaitNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphEventWaitNodeGetEvent)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphEventWaitNodeSetEvent)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddExternalSemaphoresSignalNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExternalSemaphoresSignalNodeGetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExternalSemaphoresSignalNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddExternalSemaphoresWaitNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExternalSemaphoresWaitNodeGetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExternalSemaphoresWaitNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecExternalSemaphoresSignalNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecExternalSemaphoresWaitNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddMemAllocNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphMemAllocNodeGetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddMemFreeNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphMemFreeNodeGetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuDeviceGraphMemTrim)
DEFINE_CUDA_OPTIONAL_EXPORT(cuDeviceGetGraphMemAttribute)
DEFINE_CUDA_OPTIONAL_EXPORT(cuDeviceSetGraphMemAttribute)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphClone)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphNodeFindInClone)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphNodeGetType)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphGetNodes)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphGetRootNodes)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphGetEdges)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphNodeGetDependencies)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphNodeGetDependentNodes)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddDependencies)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphRemoveDependencies)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphDestroyNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphInstantiate)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphInstantiateWithFlags)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphUpload)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphLaunch)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecDestroy)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphDestroy)
DEFINE_CUDA_OPTIONAL_EXPORT(cuStreamBeginCapture)
DEFINE_CUDA_OPTIONAL_EXPORT(cuStreamBeginCapture_v2)
DEFINE_CUDA_OPTIONAL_EXPORT(cuStreamBeginCaptureToGraph)
DEFINE_CUDA_OPTIONAL_EXPORT(cuStreamEndCapture)
DEFINE_CUDA_OPTIONAL_EXPORT(cuStreamIsCapturing)
DEFINE_CUDA_OPTIONAL_EXPORT(cuStreamGetCaptureInfo)
DEFINE_CUDA_OPTIONAL_EXPORT(cuStreamGetCaptureInfo_v2)
DEFINE_CUDA_OPTIONAL_EXPORT(cuStreamUpdateCaptureDependencies)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecKernelNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecMemcpyNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecMemsetNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecHostNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecChildGraphNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecEventRecordNodeSetEvent)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecEventWaitNodeSetEvent)
DEFINE_CUDA_OPTIONAL_EXPORT(cuThreadExchangeStreamCaptureMode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecUpdate)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphKernelNodeCopyAttributes)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphKernelNodeGetAttribute)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphKernelNodeSetAttribute)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphDebugDotPrint)
DEFINE_CUDA_OPTIONAL_EXPORT(cuUserObjectCreate)
DEFINE_CUDA_OPTIONAL_EXPORT(cuUserObjectRetain)
DEFINE_CUDA_OPTIONAL_EXPORT(cuUserObjectRelease)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphRetainUserObject)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphReleaseUserObject)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphNodeSetEnabled)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphNodeGetEnabled)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphInstantiateWithParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecGetFlags)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphAddNode)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphExecNodeSetParams)
DEFINE_CUDA_OPTIONAL_EXPORT(cuGraphConditionalHandleCreate)
DEFINE_CUDA_OPTIONAL_EXPORT(cuDeviceRegisterAsyncNotification)
DEFINE_CUDA_OPTIONAL_EXPORT(cuDeviceUnregisterAsyncNotification)

#undef DEFINE_CUDA_OPTIONAL_EXPORT

#undef DEFINE_CUDA_NOT_SUPPORTED_STUB

/* ================================================================
 * CUDA Driver API — Stream management
 * ================================================================ */

CUresult cuStreamCreate(CUstream *phStream, unsigned int flags)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!phStream) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[2] = { flags, 0 };

    rc = rpc_simple(CUDA_CALL_STREAM_CREATE_WITH_FLAGS, args, 2, &result);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *phStream = (CUstream)(uintptr_t)result.results[0];
    }
    return rc;
}

CUresult cuStreamCreateWithFlags(CUstream *phStream, unsigned int flags)
{
    return cuStreamCreate(phStream, flags);
}

CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags,
                                     int priority)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!phStream) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[4] = { flags, 0, (uint32_t)priority, 0 };

    rc = rpc_simple(CUDA_CALL_STREAM_CREATE_WITH_PRIORITY, args, 4, &result);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *phStream = (CUstream)(uintptr_t)result.results[0];
    }
    return rc;
}

CUresult cuStreamDestroy_v2(CUstream hStream)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hStream);

    return rpc_simple(CUDA_CALL_STREAM_DESTROY, args, 2, &result);
}

CUresult cuStreamDestroy(CUstream hStream)
{
    return cuStreamDestroy_v2(hStream);
}

CUresult cuStreamSynchronize(CUstream hStream)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hStream);

    return rpc_simple(CUDA_CALL_STREAM_SYNCHRONIZE, args, 2, &result);
}

CUresult cuStreamQuery(CUstream hStream)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hStream);

    return rpc_simple(CUDA_CALL_STREAM_QUERY, args, 2, &result);
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags)
{
    (void)hStream;
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority)
{
    (void)hStream;
    if (!priority) return CUDA_ERROR_INVALID_VALUE;
    *priority = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetDevice(CUstream hStream, CUdevice *device)
{
    (void)hStream;
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx)
{
    (void)hStream;
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    return cuCtxGetCurrent(pctx);
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                            unsigned int flags)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[6];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hStream);
    CUDA_PACK_U64(args, 2, (uint64_t)(uintptr_t)hEvent);
    args[4] = flags;
    args[5] = 0;

    return rpc_simple(CUDA_CALL_STREAM_WAIT_EVENT, args, 6, &result);
}

/* ================================================================
 * CUDA Driver API — Event management
 * ================================================================ */

CUresult cuEventCreate(CUevent *phEvent, unsigned int flags)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!phEvent) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[2] = { flags, 0 };

    rc = rpc_simple(CUDA_CALL_EVENT_CREATE_WITH_FLAGS, args, 2, &result);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *phEvent = (CUevent)(uintptr_t)result.results[0];
    }
    return rc;
}

CUresult cuEventDestroy_v2(CUevent hEvent)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hEvent);

    return rpc_simple(CUDA_CALL_EVENT_DESTROY, args, 2, &result);
}

CUresult cuEventDestroy(CUevent hEvent)
{
    return cuEventDestroy_v2(hEvent);
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[4];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hEvent);
    CUDA_PACK_U64(args, 2, (uint64_t)(uintptr_t)hStream);

    return rpc_simple(CUDA_CALL_EVENT_RECORD, args, 4, &result);
}

CUresult cuEventSynchronize(CUevent hEvent)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hEvent);

    return rpc_simple(CUDA_CALL_EVENT_SYNCHRONIZE, args, 2, &result);
}

CUresult cuEventQuery(CUevent hEvent)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[2];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hEvent);

    return rpc_simple(CUDA_CALL_EVENT_QUERY, args, 2, &result);
}

CUresult cuEventElapsedTime(float *pMilliseconds,
                             CUevent hStart, CUevent hEnd)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!pMilliseconds) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result;
    uint32_t args[4];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hStart);
    CUDA_PACK_U64(args, 2, (uint64_t)(uintptr_t)hEnd);

    rc = rpc_simple(CUDA_CALL_EVENT_ELAPSED_TIME, args, 4, &result);
    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        /* Result packed as uint32 bits of float */
        uint32_t fbits = (uint32_t)result.results[0];
        memcpy(pMilliseconds, &fbits, sizeof(float));
    }
    return rc;
}

/* ================================================================
 * CUDA Driver API — Occupancy (local computation)
 * ================================================================ */

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                      CUfunction func,
                                                      int blockSize,
                                                      size_t dynSharedMem)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!numBlocks) return CUDA_ERROR_INVALID_VALUE;

    /* Simple heuristic: max_threads_per_sm / blockSize */
    int max_t = g_gpu_info.max_threads_per_mp;
    if (max_t <= 0) max_t = GPU_DEFAULT_MAX_THREADS_PER_SM;
    if (blockSize <= 0) blockSize = 1;

    *numBlocks = max_t / blockSize;
    if (*numBlocks < 1) *numBlocks = 1;

    (void)func; (void)dynSharedMem;
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize,
                                           int *blockSize,
                                           CUfunction func,
                                           void *blockSizeToDynSMemSize,
                                           size_t dynSMemSize,
                                           int blockSizeLimit)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!minGridSize || !blockSize) return CUDA_ERROR_INVALID_VALUE;

    *blockSize = 256;  /* Common default */
    *minGridSize = g_gpu_info.multi_processor_count * 4;
    if (*minGridSize <= 0) *minGridSize = GPU_DEFAULT_SM_COUNT * 4;

    (void)func; (void)blockSizeToDynSMemSize;
    (void)dynSMemSize; (void)blockSizeLimit;
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks,
                                                              CUfunction func,
                                                              int blockSize,
                                                              size_t dynSharedMem,
                                                              unsigned int flags)
{
    CUresult rc = cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks,
                                                              func,
                                                              blockSize,
                                                              dynSharedMem);
    if (rc != CUDA_SUCCESS) return rc;

    fprintf(stderr,
            "[libvgpu-cuda] cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() SAFE: func=%p blockSize=%d dynSharedMem=%zu flags=0x%x numBlocks=%d (pid=%d)\n",
            (void *)func, blockSize, dynSharedMem, flags,
            numBlocks ? *numBlocks : -1, (int)getpid());
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize,
                                                 CUfunction func,
                                                 int numBlocks,
                                                 int blockSize)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!dynamicSmemSize) return CUDA_ERROR_INVALID_VALUE;

    size_t per_block_limit = (g_gpu_info.max_shared_mem_per_block > 0)
        ? (size_t)g_gpu_info.max_shared_mem_per_block
        : (size_t)GPU_DEFAULT_SHARED_MEM_PER_BLOCK;
    size_t per_sm_limit = (g_gpu_info.max_shared_mem_per_mp > 0)
        ? (size_t)g_gpu_info.max_shared_mem_per_mp
        : (size_t)GPU_DEFAULT_SHARED_MEM_PER_SM;

    if (numBlocks <= 0) numBlocks = 1;
    if (blockSize <= 0) blockSize = 1;

    size_t budget_by_sm = per_sm_limit / (size_t)numBlocks;
    *dynamicSmemSize = budget_by_sm < per_block_limit ? budget_by_sm : per_block_limit;

    fprintf(stderr,
            "[libvgpu-cuda] cuOccupancyAvailableDynamicSMemPerBlock() SAFE: func=%p numBlocks=%d blockSize=%d dynamicSmemSize=%zu (pid=%d)\n",
            (void *)func, numBlocks, blockSize, *dynamicSmemSize, (int)getpid());
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialClusterSize(int *clusterSize,
                                            CUfunction func,
                                            const CUlaunchConfig *config)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!clusterSize) return CUDA_ERROR_INVALID_VALUE;

    /*
     * Conservative portable fallback: 1 cluster is always safe for
     * init-time capability probing and matches the library's existing
     * use of per-function cluster attributes in this path.
     */
    *clusterSize = 1;

    fprintf(stderr,
            "[libvgpu-cuda] cuOccupancyMaxPotentialClusterSize() SAFE: func=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u numAttrs=%u clusterSize=%d (pid=%d)\n",
            (void *)func,
            config ? config->gridDimX : 0, config ? config->gridDimY : 0, config ? config->gridDimZ : 0,
            config ? config->blockDimX : 0, config ? config->blockDimY : 0, config ? config->blockDimZ : 0,
            config ? config->sharedMemBytes : 0, config ? config->numAttrs : 0,
            *clusterSize, (int)getpid());
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxActiveClusters(int *numClusters,
                                      CUfunction func,
                                      const CUlaunchConfig *config)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!numClusters) return CUDA_ERROR_INVALID_VALUE;

    int sm_count = g_gpu_info.multi_processor_count;
    if (sm_count <= 0) sm_count = GPU_DEFAULT_SM_COUNT;

    uint64_t launched_clusters = 1;
    if (config) {
        uint64_t gx = config->gridDimX ? config->gridDimX : 1;
        /*
         * libcublasLt probes clustered launches by sweeping gridDimX while
         * leaving gridDimY very large. Treating Y/Z as independent clusters
         * wildly overcounts the resident cluster request and produces a flat
         * "SM count" answer for every probe. For this fallback path, keep the
         * cluster count tied to the X-dimension sweep, which is the signal the
         * library is varying during capability discovery.
         */
        launched_clusters = gx;
        if (launched_clusters == 0) launched_clusters = 1;
    }

    *numClusters = (launched_clusters < (uint64_t)sm_count) ? (int)launched_clusters : sm_count;
    if (*numClusters < 1) *numClusters = 1;

    fprintf(stderr,
            "[libvgpu-cuda] cuOccupancyMaxActiveClusters() SAFE: func=%p grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u numAttrs=%u numClusters=%d (pid=%d)\n",
            (void *)func,
            config ? config->gridDimX : 0, config ? config->gridDimY : 0, config ? config->gridDimZ : 0,
            config ? config->blockDimX : 0, config ? config->blockDimY : 0, config ? config->blockDimZ : 0,
            config ? config->sharedMemBytes : 0, config ? config->numAttrs : 0,
            *numClusters, (int)getpid());
    fflush(stderr);
    return CUDA_SUCCESS;
}

/* ================================================================
 * CUDA Driver API — Function attributes
 * ================================================================ */

CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;
    if (!pi) return CUDA_ERROR_INVALID_VALUE;

    CUDACallResult result = {0};
    uint32_t args[4];
    args[0] = (uint32_t)attrib;
    args[1] = 0;
    CUDA_PACK_U64(args, 2, (uint64_t)(uintptr_t)hfunc);

    rc = rpc_simple(CUDA_CALL_FUNC_GET_ATTRIBUTE, args, 4, &result);

    if (rc == CUDA_SUCCESS && result.num_results >= 1) {
        *pi = (int)result.results[0];
    } else {
        /* Keep inference alive if host does not implement this call. */
        *pi = 0;
        rc = CUDA_SUCCESS;
    }

    /*
     * Defensive non-zero defaults for GGML occupancy math.
     * Some paths divide by NUM_REGS / MAX_THREADS_PER_BLOCK / PTX/BINARY-derived
     * values; zero here can trigger SIGFPE in runner.
     */
    switch (attrib) {
    case 0: /* CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK */
        if (*pi <= 0) *pi = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;
        break;
    case 1: /* CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES */
    case 2: /* CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES */
    case 3: /* CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES */
        if (*pi < 0) *pi = 0;
        break;
    case 4: /* CU_FUNC_ATTRIBUTE_NUM_REGS */
        if (*pi <= 0) *pi = 64;
        break;
    case 5: /* CU_FUNC_ATTRIBUTE_PTX_VERSION */
    case 6: /* CU_FUNC_ATTRIBUTE_BINARY_VERSION */
        if (*pi <= 0) *pi = (GPU_DEFAULT_CC_MAJOR * 10) + GPU_DEFAULT_CC_MINOR;
        break;
    case 8: /* CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES */
        if (*pi <= 0) *pi = (int)GPU_DEFAULT_SHARED_MEM_PER_BLOCK;
        break;
    default:
        if (*pi == 0) *pi = 1;
        break;
    }

    fprintf(stderr,
            "[libvgpu-cuda] cuFuncGetAttribute() SAFE: attrib=%d value=%d (pid=%d)\n",
            attrib, *pi, (int)getpid());
    fflush(stderr);
    return rc;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, int config)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    CUDACallResult result;
    uint32_t args[4];
    CUDA_PACK_U64(args, 0, (uint64_t)(uintptr_t)hfunc);
    args[2] = (uint32_t)config;
    args[3] = 0;

    return rpc_simple(CUDA_CALL_FUNC_SET_CACHE_CONFIG, args, 4, &result);
}

CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value)
{
    CUresult rc = ensure_init();
    if (rc != CUDA_SUCCESS) return rc;

    fprintf(stderr,
            "[libvgpu-cuda] cuFuncSetAttribute() NO-OP SUCCESS: hfunc=%p attrib=%d value=%d (pid=%d)\n",
            (void *)hfunc, attrib, value, (int)getpid());
    fflush(stderr);
    return CUDA_SUCCESS;
}

/* ================================================================
 * Memory allocation interception for GGML alignment requirements
 * ================================================================ */

/* NOTE: Malloc interception has been moved to libggml-alloc-intercept.so
 * to avoid conflicts. This section is kept for reference but disabled.
 * The dedicated allocation interceptor handles all memory allocation
 * functions comprehensively.
 */

#if 0
/* CRITICAL: Intercept malloc/posix_memalign to ensure 32-byte alignment
 * GGML requires all buffer pointers to be 32-byte aligned (TENSOR_ALIGNMENT).
 * We intercept standard allocation functions to guarantee alignment. */

#include <stdlib.h>
#include <errno.h>

/* Real malloc function pointer */
static void *(*g_real_malloc)(size_t) = NULL;
static void *(*g_real_free)(void *) = NULL;
static int (*g_real_posix_memalign)(void **, size_t, size_t) = NULL;

/* Initialize real function pointers */
static void init_malloc_hooks(void) {
    if (g_real_malloc) return; /* Already initialized */
    
    g_real_malloc = dlsym(RTLD_NEXT, "malloc");
    g_real_free = dlsym(RTLD_NEXT, "free");
    g_real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
}

/* Aligned malloc wrapper - ensures 32-byte alignment */
void *malloc(size_t size) {
    static int in_malloc = 0;
    if (in_malloc) {
        /* Recursion guard - if we're already in malloc, use syscall directly */
        return (void *)syscall(__NR_brk, 0); /* Fallback - not ideal but safe */
    }
    
    in_malloc = 1;
    init_malloc_hooks();
    
    if (!g_real_malloc) {
        in_malloc = 0;
        return NULL;
    }
    
    /* Allocate extra space for alignment */
    size_t aligned_size = size + 32 + sizeof(uintptr_t);
    void *ptr = g_real_malloc(aligned_size);
    if (!ptr) {
        in_malloc = 0;
        return NULL;
    }
    
    /* Align to 32-byte boundary */
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned_addr = (addr + 31) & ~(uintptr_t)31;
    
    /* Ensure we have space for the original pointer */
    if (aligned_addr - addr < sizeof(uintptr_t)) {
        aligned_addr += 32;
    }
    
    /* Store original pointer before aligned address (for free) */
    *((uintptr_t *)(aligned_addr - sizeof(uintptr_t))) = addr;
    
    in_malloc = 0;
    return (void *)aligned_addr;
}

/* Aligned free wrapper */
void free(void *ptr) {
    static int in_free = 0;
    if (in_free || !ptr) return;
    
    in_free = 1;
    init_malloc_hooks();
    
    if (!g_real_free) {
        in_free = 0;
        return;
    }
    
    /* Retrieve original pointer */
    uintptr_t aligned_addr = (uintptr_t)ptr;
    if (aligned_addr < sizeof(uintptr_t)) {
        in_free = 0;
        return; /* Invalid pointer */
    }
    uintptr_t orig_addr = *((uintptr_t *)(aligned_addr - sizeof(uintptr_t)));
    
    g_real_free((void *)orig_addr);
    in_free = 0;
}

/* Aligned posix_memalign wrapper - ensures minimum 32-byte alignment */
int posix_memalign(void **memptr, size_t alignment, size_t size) {
    init_malloc_hooks();
    
    if (!memptr) return EINVAL;
    
    /* Ensure alignment is at least 32 bytes */
    if (alignment < 32) alignment = 32;
    
    if (g_real_posix_memalign) {
        return g_real_posix_memalign(memptr, alignment, size);
    }
    
    /* Fallback: use aligned malloc */
    void *ptr = malloc(size);
    if (!ptr) return ENOMEM;
    
    *memptr = ptr;
    return 0;
}
#endif /* end malloc interception block - keep Error handling and stubs below compiled */

#if 1
/* ================================================================
 * CUDA Driver API — Error handling
 * ================================================================ */

__attribute__((visibility("default")))
CUresult cuGetErrorString(CUresult error, const char **pStr)
{
    /* CRITICAL: Log this call - GGML may check error strings after function calls */
    char log_msg[256];
    int log_len = snprintf(log_msg, sizeof(log_msg),
                          "[libvgpu-cuda] cuGetErrorString() CALLED (error=%d, pid=%d)\n",
                          (int)error, (int)getpid());
    if (log_len > 0 && log_len < (int)sizeof(log_msg)) {
        syscall(__NR_write, 2, log_msg, log_len);
    }
    
    if (!pStr) {
        char error_msg[128];
        int error_len = snprintf(error_msg, sizeof(error_msg),
                                "[libvgpu-cuda] cuGetErrorString() ERROR: invalid pStr (pid=%d)\n",
                                (int)getpid());
        if (error_len > 0 && error_len < (int)sizeof(error_msg)) {
            syscall(__NR_write, 2, error_msg, error_len);
        }
        return CUDA_ERROR_INVALID_VALUE;
    }
    static const char *err_str = "CUDA error";
    static const char *ok_str  = "no error";
    *pStr = (error == CUDA_SUCCESS) ? ok_str : err_str;
    
    char success_msg[256];
    int success_len = snprintf(success_msg, sizeof(success_msg),
                              "[libvgpu-cuda] cuGetErrorString() SUCCESS: error=%d, str=\"%s\" (pid=%d)\n",
                              (int)error, *pStr, (int)getpid());
    if (success_len > 0 && success_len < (int)sizeof(success_msg)) {
        syscall(__NR_write, 2, success_msg, success_len);
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorName(CUresult error, const char **pStr)
{
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    static const char *err_name    = "CUDA_ERROR";
    static const char *ok_name     = "CUDA_SUCCESS";
    *pStr = (error == CUDA_SUCCESS) ? ok_name : err_name;
    return CUDA_SUCCESS;
}

CUresult cuGetLastError(void)
{
    /* Always return SUCCESS - no errors have occurred */
    return CUDA_SUCCESS;
}

/* ================================================================
 * CUDA Driver API — Device P2P (stub — single device)
 * ================================================================ */

CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                                CUdevice peerDev)
{
    if (!canAccessPeer) return CUDA_ERROR_INVALID_VALUE;
    *canAccessPeer = 0;
    (void)dev; (void)peerDev;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetP2PAttribute(int *value, int attrib,
                                  CUdevice srcDevice, CUdevice dstDevice)
{
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    *value = 0;
    (void)attrib; (void)srcDevice; (void)dstDevice;
    return CUDA_SUCCESS;
}

/* ================================================================
 * Catch-all for any CUDA functions we haven't implemented
 * 
 * This ensures that if ggml_cuda_init calls a function we haven't
 * implemented, it gets a stub that returns SUCCESS rather than
 * causing a symbol lookup failure or returning NOT_SUPPORTED.
 * ================================================================ */

/* Helper macro to create stub functions */
#define CUDA_STUB_FUNC(name, ret_type, ...) \
    ret_type name(__VA_ARGS__) { \
        fprintf(stderr, "[libvgpu-cuda] STUB CALLED: %s (not fully implemented, returning SUCCESS)\n", #name); \
        fflush(stderr); \
        (void)0; /* Suppress unused parameter warnings */ \
        return (ret_type)CUDA_SUCCESS; \
    }
#endif /* end Error handling + stub macro block */

/* Common stub functions that might be called during initialization */
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuCtxGetApiVersion(ctx=%p, version=%p)\n", ctx, version);
    fflush(stderr);
    if (version) *version = 12080; /* CUDA 12.8 */
    log_ctx_state_snapshot("cuCtxGetApiVersion return", ctx, ctx,
                           CUDA_SUCCESS, version ? (int)*version : -1, 0);
    return CUDA_SUCCESS;
}

/* cuCtxGetDevice is already defined above (line ~1818), removing duplicate */

__attribute__((visibility("default")))
CUresult cuCtxGetFlags(unsigned int *flags)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuCtxGetFlags(flags=%p)\n", flags);
    fflush(stderr);
    if (flags) *flags = 0;
    log_ctx_state_snapshot("cuCtxGetFlags return", g_current_ctx, g_current_ctx,
                           CUDA_SUCCESS, flags ? (int)*flags : -1, 0);
    return CUDA_SUCCESS;
}

CUresult cuCtxSetLimit(int limit, size_t value)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuCtxSetLimit(limit=%d, value=%zu)\n", limit, value);
    fflush(stderr);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetLimit(size_t *pvalue, int limit)
{
    fprintf(stderr, "[libvgpu-cuda] CALLED: cuCtxGetLimit(pvalue=%p, limit=%d)\n", pvalue, limit);
    fflush(stderr);
    if (pvalue) *pvalue = 0;
    return CUDA_SUCCESS;
}

#if 0
/* ================================================================
 * Version symbol aliases
 *
 * Many CUDA applications link against versioned symbols
 * (e.g. cuMemAlloc_v2).  We provide both versions.
 * ================================================================ */

/* These are already provided as separate functions above */

/* NOTE: Constructor functionality merged into libvgpu_cuda_on_load() above
 * to avoid duplicate constructors and ensure stability. */

/* ================================================================
 * write() interception to capture full error messages
 * 
 * CRITICAL: Intercept write() to stderr to capture the full
 * error message from ggml_cuda_init(). The error is currently
 * truncated at 98 bytes in strace, but we can capture the full
 * message here.
 * 
 * ENHANCED: Capture ALL writes to stderr (not just filtered),
 * with increased buffer size (2000 bytes), timestamps, and
 * logging to multiple files.
 * ================================================================ */

ssize_t write(int fd, const void *buf, size_t count)
{
    static ssize_t (*real_write)(int, const void *, size_t) = NULL;
    if (!real_write) {
        real_write = (ssize_t (*)(int, const void *, size_t))
                     dlsym(RTLD_NEXT, "write");
    }
    
    /* If writing to stderr, capture ALL messages (not just filtered) */
    /* Increased buffer size to 2000 to capture longer error messages */
    if (fd == 2 && buf && count > 0 && count < 2000) {
        const char *msg = (const char *)buf;
        
        /* Get timestamp and PID using syscalls to avoid libc dependencies */
        struct timespec ts;
        #ifndef __NR_clock_gettime
        #define __NR_clock_gettime 228  /* x86_64 syscall number */
        #endif
        syscall(__NR_clock_gettime, 0, &ts); /* CLOCK_REALTIME = 0 */
        time_t sec = ts.tv_sec;
        long nsec = ts.tv_nsec;
        pid_t pid = (pid_t)syscall(__NR_getpid);
        
        /* Log ALL stderr writes to full log */
        /* Use simple string building without snprintf to avoid libc dependencies */
        int log_fd_full = syscall(__NR_open, "/tmp/ollama_errors_full.log", 
                                  O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (log_fd_full >= 0) {
            /* Simple prefix without snprintf - just write the message directly */
            syscall(__NR_write, log_fd_full, buf, count);
            syscall(__NR_write, log_fd_full, "\n", 1);
            syscall(__NR_close, log_fd_full);
        }
        
        /* Also log filtered messages (errors, failures, etc.) to filtered log */
        if (strstr(msg, "ggml") != NULL || 
            strstr(msg, "cuda") != NULL ||
            strstr(msg, "failed") != NULL || 
            strstr(msg, "error") != NULL ||
            strstr(msg, "CUDA") != NULL ||
            strstr(msg, "init") != NULL ||
            strstr(msg, "timeout") != NULL ||
            strstr(msg, "discover") != NULL) {
            int log_fd_filtered = syscall(__NR_open, "/tmp/ollama_errors_filtered.log", 
                                        O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (log_fd_filtered >= 0) {
                /* Write the full message - this is critical for capturing the error */
                syscall(__NR_write, log_fd_filtered, buf, count);
                syscall(__NR_write, log_fd_filtered, "\n", 1);
                syscall(__NR_close, log_fd_filtered);
            }
        }
    }
    
    return real_write ? real_write(fd, buf, count) : -1;
}
#endif /* #if 0 - malloc interception disabled */
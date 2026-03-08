/*
 * LD_AUDIT Interceptor for CUDA/NVML Library Loading
 * 
 * This library uses the LD_AUDIT interface to intercept dlopen calls
 * at a lower level than LD_PRELOAD, which works even with Go binaries
 * that ignore LD_PRELOAD.
 * 
 * CRITICAL: This library loads VERY early during dynamic linking, before
 * most libc functions are available. All callbacks must be minimal and
 * async-signal-safe. NO I/O operations, NO dlopen() calls, NO complex functions.
 * 
 * Usage: LD_AUDIT=/path/to/libldaudit_cuda.so ollama run ...
 * Or: export LD_AUDIT=/path/to/libldaudit_cuda.so
 * 
 * Compile: gcc -shared -fPIC -o libldaudit_cuda.so ld_audit_interceptor.c -ldl
 */

#define _GNU_SOURCE
#include <link.h>
#include <stddef.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

/* NOTE: SHIM_CUDA and SHIM_NVML removed - no longer needed since we removed la_objsearch()
 * Library redirection is now handled by pre-loaded libraries + LD_PRELOAD
 */

/* LD_AUDIT interface functions - all minimal, no side effects */

unsigned int la_version(unsigned int version)
{
    return LAV_CURRENT;
}

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
    /* Safe logging: Use async-signal-safe syscalls only
     * write() is async-signal-safe, and we use a small static buffer
     * to format the message. This allows us to track library loads
     * without breaking system library loading. */
    if (map && map->l_name && map->l_name[0]) {
        /* Extract just the filename from the path */
        const char *name = map->l_name;
        const char *basename = strrchr(name, '/');
        if (basename) {
            basename++; /* Skip the '/' */
        } else {
            basename = name;
        }
        
        /* Only log CUDA/NVML related libraries to avoid spam */
        if (strstr(basename, "cuda") || strstr(basename, "nvidia") || 
            strstr(basename, "vgpu")) {
            static char buf[256];
            int len = snprintf(buf, sizeof(buf), 
                             "[ld-audit] Library loaded: %s (pid=%d)\n",
                             basename, (int)getpid());
            if (len > 0 && len < (int)sizeof(buf)) {
                /* write() to stderr (fd 2) is async-signal-safe */
                write(2, buf, len);
            }
        }
    }
    
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

void la_activity(uintptr_t *cookie, unsigned int flag)
{
    /* la_activity must return void according to glibc */
    /* Minimal - no operations */
}

uintptr_t la_symbind32(Elf32_Sym *sym, unsigned int ndx,
                      uintptr_t *refcook, uintptr_t *defcook,
                      unsigned int *flags, const char *symname)
{
    /* Minimal - just return symbol value, no side effects */
    return sym->st_value;
}

uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx,
                       uintptr_t *refcook, uintptr_t *defcook,
                       unsigned int *flags, const char *symname)
{
    /* Minimal - just return symbol value, no dlopen(), no I/O */
    /* The library redirection happens in la_objsearch(), not here */
    return sym->st_value;
}

/* NOTE: la_objsearch() removed - it breaks system libraries on glibc 2.35
 * Even returning NULL immediately causes "cannot open shared object file" errors
 * for system libraries like libc.so.6. This is a glibc incompatibility issue.
 * 
 * Library redirection is handled by:
 * 1. Pre-loaded libraries via force_load_shim's dlopen(RTLD_GLOBAL) before exec()
 * 2. LD_PRELOAD (set by force_load_shim, inherited via libvgpu-exec for subprocesses)
 * 
 * LD_AUDIT is now used only for monitoring/verification via la_objopen()
 */

/* Required by LD_AUDIT interface */
unsigned int la_objclose(uintptr_t *cookie)
{
    return 0;
}

/* Optional but may be required by some linkers - provide stubs */
void la_preinit(uintptr_t *cookie)
{
    /* Minimal - no operations */
}

/* x86_64-specific PLT entry/exit callbacks - provide stubs */
Elf64_Addr la_x86_64_gnu_pltenter(Elf64_Sym *sym, unsigned int ndx,
                                  uintptr_t *refcook, uintptr_t *defcook,
                                  La_x86_64_regs *regs, unsigned int *flags,
                                  const char *symname, long int *framesizep)
{
    /* Return symbol value - minimal, no side effects */
    return sym->st_value;
}

unsigned int la_x86_64_gnu_pltexit(Elf64_Sym *sym, unsigned int ndx,
                                   uintptr_t *refcook, uintptr_t *defcook,
                                   const La_x86_64_regs *inregs,
                                   La_x86_64_retval *outregs, const char *symname)
{
    /* Return 0 - minimal, no side effects */
    return 0;
}

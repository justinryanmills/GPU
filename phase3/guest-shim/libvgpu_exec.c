/*
 * libvgpu_exec.c — Process spawning interception
 *
 * This library intercepts process spawning functions (execve, execv, fork, clone)
 * to ensure LD_PRELOAD is inherited by all subprocesses, including those
 * spawned by Go's runtime.
 *
 * Build:
 *   gcc -shared -fPIC -o libvgpu-exec.so libvgpu_exec.c -ldl
 *
 * Usage:
 *   LD_PRELOAD=libvgpu-exec.so:libvgpu-syscall.so:libvgpu-cuda.so:libvgpu-nvml.so ollama serve
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>

/* Use library names (not full paths) - dynamic linker will find them via LD_LIBRARY_PATH
 * This matches what force_load_shim sets in LD_PRELOAD
 * CRITICAL: Must include ALL shim libraries for runner subprocess */
/* NOTE: libvgpu-syscall.so removed - was causing issues */
static const char *SHIM_LIBS = "libvgpu-exec.so:libvgpu-cuda.so:libvgpu-nvml.so:libvgpu-cudart.so";

/* Get or create LD_PRELOAD with our shims */
static char *get_preload_env(void)
{
    const char *existing = getenv("LD_PRELOAD");
    char *new_preload;
    size_t len;
    
    if (existing && strstr(existing, "libvgpu")) {
        /* Already has our shims */
        return strdup(existing);
    }
    
    if (existing && existing[0]) {
        len = strlen(existing) + strlen(SHIM_LIBS) + 2;
        new_preload = malloc(len);
        snprintf(new_preload, len, "%s:%s", SHIM_LIBS, existing);
    } else {
        new_preload = strdup(SHIM_LIBS);
    }
    
    return new_preload;
}

/* Inject LD_PRELOAD into environment array */
static char **inject_preload(char *const envp[])
{
    char *new_preload = get_preload_env();
    char **new_envp = NULL;
    int count = 0;
    int i, j;
    int preload_found = 0;
    
    /* Count existing environment variables */
    if (envp) {
        for (i = 0; envp[i]; i++) {
            count++;
            if (strncmp(envp[i], "LD_PRELOAD=", 11) == 0) {
                preload_found = 1;
            }
        }
    }
    
    /* Allocate new environment array */
    new_envp = malloc((count + 2) * sizeof(char *));
    
    /* Copy existing environment */
    j = 0;
    if (envp) {
        for (i = 0; envp[i]; i++) {
            if (strncmp(envp[i], "LD_PRELOAD=", 11) == 0) {
                /* Replace existing LD_PRELOAD */
                size_t len = strlen("LD_PRELOAD=") + strlen(new_preload) + 1;
                new_envp[j] = malloc(len);
                snprintf(new_envp[j], len, "LD_PRELOAD=%s", new_preload);
                j++;
            } else {
                new_envp[j++] = strdup(envp[i]);
            }
        }
    }
    
    /* Add LD_PRELOAD if not found */
    if (!preload_found) {
        size_t len = strlen("LD_PRELOAD=") + strlen(new_preload) + 1;
        new_envp[j] = malloc(len);
        snprintf(new_envp[j], len, "LD_PRELOAD=%s", new_preload);
        j++;
    }
    
    new_envp[j] = NULL;
    free(new_preload);
    
    return new_envp;
}

/* Free environment array */
static void free_envp(char **envp)
{
    if (envp) {
        for (int i = 0; envp[i]; i++) {
            free(envp[i]);
        }
        free(envp);
    }
}

/* Check if this is a runner process based on argv */
static int is_runner_process(char *const argv[])
{
    if (!argv || !argv[0]) return 0;
    
    /* Check if argv[0] contains "ollama" and argv[1] contains "runner" */
    int found_ollama = 0, found_runner = 0;
    for (int i = 0; argv[i] && i < 10; i++) {
        if (strstr(argv[i], "ollama")) found_ollama = 1;
        if (strstr(argv[i], "runner")) found_runner = 1;
    }
    return found_ollama && found_runner;
}

/* Intercept execve() - most common exec variant */
int execve(const char *pathname, char *const argv[], char *const envp[])
{
    static int (*real_execve)(const char *, char *const[], char *const[]) = NULL;
    if (!real_execve) {
        real_execve = (int (*)(const char *, char *const[], char *const[]))
                      dlsym(RTLD_NEXT, "execve");
    }
    
    pid_t pid = getpid();
    int is_runner = is_runner_process(argv);
    
    /* Log detailed information about the exec call */
    fprintf(stderr, "[libvgpu-exec] execve(\"%s\") intercepted (pid=%d, is_runner=%d)\n",
            pathname ? pathname : "(null)", (int)pid, is_runner);
    
    /* Log command arguments */
    if (argv && argv[0]) {
        fprintf(stderr, "[libvgpu-exec]   argv[0]=\"%s\"", argv[0]);
        if (argv[1]) fprintf(stderr, " argv[1]=\"%s\"", argv[1]);
        if (argv[2]) fprintf(stderr, " argv[2]=\"%s\"", argv[2]);
        fprintf(stderr, "\n");
    }
    
    /* Log current LD_PRELOAD if present */
    const char *current_preload = getenv("LD_PRELOAD");
    if (current_preload) {
        fprintf(stderr, "[libvgpu-exec]   Current LD_PRELOAD=%s\n", current_preload);
    } else {
        fprintf(stderr, "[libvgpu-exec]   Current LD_PRELOAD=(not set)\n");
    }
    
    /* Inject LD_PRELOAD into environment */
    char **new_envp = inject_preload(envp);
    char *new_preload = get_preload_env();
    
    fprintf(stderr, "[libvgpu-exec]   Injected LD_PRELOAD=%s\n", new_preload ? new_preload : "(null)");
    if (is_runner) {
        fprintf(stderr, "[libvgpu-exec]   *** RUNNER PROCESS DETECTED - ensuring shim libraries are loaded ***\n");
    }
    fflush(stderr);
    
    free(new_preload);
    int rc = real_execve(pathname, argv, new_envp);
    
    /* If execve succeeds, we never return, but if it fails, clean up */
    free_envp(new_envp);
    return rc;
}

/* Intercept execv() - exec without environment */
int execv(const char *pathname, char *const argv[])
{
    static int (*real_execv)(const char *, char *const[]) = NULL;
    if (!real_execv) {
        real_execv = (int (*)(const char *, char *const[]))
                     dlsym(RTLD_NEXT, "execv");
    }
    
    pid_t pid = getpid();
    int is_runner = is_runner_process(argv);
    
    fprintf(stderr, "[libvgpu-exec] execv(\"%s\") intercepted (pid=%d, is_runner=%d)\n",
            pathname ? pathname : "(null)", (int)pid, is_runner);
    if (argv && argv[0]) {
        fprintf(stderr, "[libvgpu-exec]   argv[0]=\"%s\"", argv[0]);
        if (argv[1]) fprintf(stderr, " argv[1]=\"%s\"", argv[1]);
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    
    /* Get current environment and inject LD_PRELOAD */
    extern char **environ;
    char **new_envp = inject_preload(environ);
    char *new_preload = get_preload_env();
    
    fprintf(stderr, "[libvgpu-exec]   Injected LD_PRELOAD=%s\n", new_preload ? new_preload : "(null)");
    if (is_runner) {
        fprintf(stderr, "[libvgpu-exec]   *** RUNNER PROCESS DETECTED ***\n");
    }
    fflush(stderr);
    
    free(new_preload);
    /* Use execve with modified environment */
    int rc = execve(pathname, argv, new_envp);
    
    free_envp(new_envp);
    return rc;
}

/* Intercept execvp() - exec with PATH search */
int execvp(const char *file, char *const argv[])
{
    static int (*real_execvp)(const char *, char *const[]) = NULL;
    if (!real_execvp) {
        real_execvp = (int (*)(const char *, char *const[]))
                      dlsym(RTLD_NEXT, "execvp");
    }
    
    pid_t pid = getpid();
    int is_runner = is_runner_process(argv);
    
    fprintf(stderr, "[libvgpu-exec] execvp(\"%s\") intercepted (pid=%d, is_runner=%d)\n",
            file ? file : "(null)", (int)pid, is_runner);
    if (argv && argv[0]) {
        fprintf(stderr, "[libvgpu-exec]   argv[0]=\"%s\"", argv[0]);
        if (argv[1]) fprintf(stderr, " argv[1]=\"%s\"", argv[1]);
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    
    extern char **environ;
    char **new_envp = inject_preload(environ);
    char *new_preload = get_preload_env();
    
    fprintf(stderr, "[libvgpu-exec]   Injected LD_PRELOAD=%s\n", new_preload ? new_preload : "(null)");
    if (is_runner) {
        fprintf(stderr, "[libvgpu-exec]   *** RUNNER PROCESS DETECTED ***\n");
    }
    fflush(stderr);
    
    free(new_preload);
    
    /* Find executable in PATH */
    char *path = getenv("PATH");
    if (!path) path = "/usr/local/bin:/usr/bin:/bin";
    
    char path_copy[4096];
    strncpy(path_copy, path, sizeof(path_copy) - 1);
    path_copy[sizeof(path_copy) - 1] = '\0';
    
    char *full_path = NULL;
    char *dir = strtok(path_copy, ":");
    while (dir) {
        size_t len = strlen(dir) + strlen(file) + 2;
        full_path = malloc(len);
        snprintf(full_path, len, "%s/%s", dir, file);
        
        if (access(full_path, X_OK) == 0) {
            break;
        }
        
        free(full_path);
        full_path = NULL;
        dir = strtok(NULL, ":");
    }
    
    if (!full_path) {
        free_envp(new_envp);
        errno = ENOENT;
        return -1;
    }
    
    int rc = execve(full_path, argv, new_envp);
    free(full_path);
    free_envp(new_envp);
    return rc;
}

/* Intercept fork() - ensure child inherits LD_PRELOAD */
pid_t fork(void)
{
    static pid_t (*real_fork)(void) = NULL;
    if (!real_fork) {
        real_fork = (pid_t (*)(void))dlsym(RTLD_NEXT, "fork");
    }
    
    pid_t pid = real_fork ? real_fork() : -1;
    
    if (pid == 0) {
        /* Child process - ensure LD_PRELOAD is set */
        char *preload = get_preload_env();
        if (preload) {
            setenv("LD_PRELOAD", preload, 1);
            free(preload);
        }
        fprintf(stderr, "[libvgpu-exec] fork() child process (pid=%d, LD_PRELOAD=%s)\n",
                (int)getpid(), getenv("LD_PRELOAD") ? getenv("LD_PRELOAD") : "(null)");
        fflush(stderr);
    } else if (pid > 0) {
        fprintf(stderr, "[libvgpu-exec] fork() parent process (child_pid=%d)\n", (int)pid);
        fflush(stderr);
    }
    
    return pid;
}

/* Intercept clone() - Linux-specific process creation */
#ifdef __linux__
#include <sched.h>
pid_t clone(int (*fn)(void *), void *child_stack, int flags, void *arg, ...)
{
    static pid_t (*real_clone)(int (*)(void *), void *, int, void *, ...) = NULL;
    if (!real_clone) {
        real_clone = (pid_t (*)(int (*)(void *), void *, int, void *, ...))
                     dlsym(RTLD_NEXT, "clone");
    }
    
    va_list args;
    va_start(args, arg);
    pid_t *ptid = va_arg(args, pid_t *);
    void *tls = va_arg(args, void *);
    pid_t *ctid = va_arg(args, pid_t *);
    va_end(args);
    
    pid_t pid = real_clone ? real_clone(fn, child_stack, flags, arg, ptid, tls, ctid) : -1;
    
    if (pid == 0) {
        /* Child process */
        char *preload = get_preload_env();
        if (preload) {
            setenv("LD_PRELOAD", preload, 1);
            free(preload);
        }
        fprintf(stderr, "[libvgpu-exec] clone() child process (pid=%d, LD_PRELOAD=%s)\n",
                (int)getpid(), getenv("LD_PRELOAD") ? getenv("LD_PRELOAD") : "(null)");
        fflush(stderr);
    } else if (pid > 0) {
        fprintf(stderr, "[libvgpu-exec] clone() parent process (child_pid=%d)\n", (int)pid);
        fflush(stderr);
    }
    
    return pid;
}
#endif

/* Constructor - minimal and safe */
__attribute__((constructor))
static void libvgpu_exec_on_load(void)
{
    /* Log that we're loaded - use syscall to avoid libc issues during early loading */
    const char *msg = "[libvgpu-exec] constructor: Library loaded (pid=";
    syscall(__NR_write, 2, msg, strlen(msg));
    char pid_str[32];
    int pid_len = snprintf(pid_str, sizeof(pid_str), "%d", (int)getpid());
    if (pid_len > 0 && pid_len < (int)sizeof(pid_str)) {
        syscall(__NR_write, 2, pid_str, pid_len);
    }
    const char *msg2 = ")\n";
    syscall(__NR_write, 2, msg2, strlen(msg2));
}

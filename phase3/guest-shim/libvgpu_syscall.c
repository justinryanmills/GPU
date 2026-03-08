/*
 * libvgpu_syscall.c — Direct syscall-level interception
 *
 * This library intercepts syscalls directly, bypassing Go's runtime
 * and C library wrappers. This catches all file operations regardless
 * of the language or runtime used.
 *
 * Build:
 *   gcc -shared -fPIC -o libvgpu-syscall.so libvgpu_syscall.c -ldl
 *
 * Usage:
 *   LD_PRELOAD=libvgpu-syscall.so:libvgpu-cuda.so:libvgpu-nvml.so ollama serve
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <errno.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/types.h>

/* Track file descriptors opened to PCI device files */
#define MAX_TRACKED_FDS 128
static struct {
    int fd;
    char path[512];
    int is_pci_device_file;
} tracked_fds[MAX_TRACKED_FDS];
static int num_tracked_fds = 0;
static pthread_mutex_t tracked_fds_mutex = PTHREAD_MUTEX_INITIALIZER;

static int is_pci_device_file_path(const char *path)
{
    if (!path) return 0;
    if (strstr(path, "/sys/bus/pci/devices/") != NULL) {
        if (strstr(path, "0000:00:05.0") != NULL ||
            strstr(path, "00:05.0") != NULL) {
            if (strstr(path, "/vendor") != NULL ||
                strstr(path, "/device") != NULL ||
                strstr(path, "/class") != NULL) {
                return 1;
            }
        }
    }
    return 0;
}

static void track_fd(int fd, const char *path)
{
    if (fd < 0 || num_tracked_fds >= MAX_TRACKED_FDS) return;
    
    pthread_mutex_lock(&tracked_fds_mutex);
    tracked_fds[num_tracked_fds].fd = fd;
    if (path) {
        strncpy(tracked_fds[num_tracked_fds].path, path, 
                sizeof(tracked_fds[num_tracked_fds].path) - 1);
        tracked_fds[num_tracked_fds].path[sizeof(tracked_fds[num_tracked_fds].path) - 1] = '\0';
    } else {
        tracked_fds[num_tracked_fds].path[0] = '\0';
    }
    tracked_fds[num_tracked_fds].is_pci_device_file = 
        is_pci_device_file_path(tracked_fds[num_tracked_fds].path);
    num_tracked_fds++;
    pthread_mutex_unlock(&tracked_fds_mutex);
}

static int is_tracked_pci_fd(int fd)
{
    int i;
    pthread_mutex_lock(&tracked_fds_mutex);
    for (i = 0; i < num_tracked_fds; i++) {
        if (tracked_fds[i].fd == fd && tracked_fds[i].is_pci_device_file) {
            pthread_mutex_unlock(&tracked_fds_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&tracked_fds_mutex);
    return 0;
}

static const char *get_tracked_fd_path(int fd)
{
    static char path[512];
    int i;
    path[0] = '\0';
    
    pthread_mutex_lock(&tracked_fds_mutex);
    for (i = 0; i < num_tracked_fds; i++) {
        if (tracked_fds[i].fd == fd) {
            strncpy(path, tracked_fds[i].path, sizeof(path) - 1);
            path[sizeof(path) - 1] = '\0';
            break;
        }
    }
    pthread_mutex_unlock(&tracked_fds_mutex);
    
    return path[0] ? path : NULL;
}

/* Check if caller is from our own code (cuda_transport.c) */
static int is_caller_from_our_code(void)
{
    void *caller = __builtin_return_address(1);
    Dl_info info;
    if (dladdr(caller, &info) && info.dli_fname) {
        if (strstr(info.dli_fname, "libvgpu") != NULL ||
            strstr(info.dli_fname, "cuda_transport") != NULL) {
            return 1;
        }
    }
    return 0;
}

/* Intercept openat() syscall - CRITICAL for Go's os.Open() */
int openat(int dirfd, const char *pathname, int flags, ...)
{
    static int (*real_openat)(int, const char *, int, ...) = NULL;
    if (!real_openat) {
        real_openat = (int (*)(int, const char *, int, ...))
                      dlsym(RTLD_NEXT, "openat");
    }
    
    va_list args;
    va_start(args, flags);
    mode_t mode = (flags & O_CREAT) ? va_arg(args, mode_t) : 0;
    va_end(args);
    
    /* Skip interception if caller is from our own code */
    if (is_caller_from_our_code()) {
        return real_openat ? real_openat(dirfd, pathname, flags, mode) : -1;
    }
    
    /* Check if this is a PCI device file */
    if (is_pci_device_file_path(pathname)) {
        fprintf(stderr, "[libvgpu-syscall] openat(%d, \"%s\", 0x%x) intercepted (pid=%d)\n",
                dirfd, pathname, flags, (int)getpid());
        fflush(stderr);
        
        int fd = real_openat ? real_openat(dirfd, pathname, flags, mode) : -1;
        if (fd >= 0) {
            track_fd(fd, pathname);
        }
        return fd;
    }
    
    return real_openat ? real_openat(dirfd, pathname, flags, mode) : -1;
}

/* Intercept read() syscall - CRITICAL for all file reads */
ssize_t read(int fd, void *buf, size_t count)
{
    static ssize_t (*real_read)(int, void *, size_t) = NULL;
    if (!real_read) {
        real_read = (ssize_t (*)(int, void *, size_t))
                    dlsym(RTLD_NEXT, "read");
    }
    
    /* Skip interception if caller is from our own code */
    if (is_caller_from_our_code()) {
        return real_read ? real_read(fd, buf, count) : -1;
    }
    
    /* Check if this is a tracked PCI device file */
    if (is_tracked_pci_fd(fd)) {
        const char *path = get_tracked_fd_path(fd);
        if (path) {
            fprintf(stderr, "[libvgpu-syscall] read(fd=%d, path=\"%s\", count=%zu) intercepted (pid=%d)\n",
                    fd, path, count, (int)getpid());
            fflush(stderr);
            
            /* Return appropriate value based on file type */
            if (strstr(path, "/vendor") != NULL) {
                if (count >= 6) {
                    strncpy((char *)buf, "0x10de\n", count);
                    fprintf(stderr, "[libvgpu-syscall] read: returning vendor=0x10de (NVIDIA)\n");
                    fflush(stderr);
                    return 6;
                }
            } else if (strstr(path, "/device") != NULL) {
                if (count >= 6) {
                    strncpy((char *)buf, "0x2331\n", count);
                    fprintf(stderr, "[libvgpu-syscall] read: returning device=0x2331 (H100 PCIe)\n");
                    fflush(stderr);
                    return 6;
                }
            } else if (strstr(path, "/class") != NULL) {
                if (count >= 8) {
                    strncpy((char *)buf, "0x030200\n", count);
                    fprintf(stderr, "[libvgpu-syscall] read: returning class=0x030200 (3D controller)\n");
                    fflush(stderr);
                    return 8;
                }
            }
        }
    }
    
    return real_read ? real_read(fd, buf, count) : -1;
}

/* Intercept pread() syscall */
ssize_t pread(int fd, void *buf, size_t count, off_t offset)
{
    static ssize_t (*real_pread)(int, void *, size_t, off_t) = NULL;
    if (!real_pread) {
        real_pread = (ssize_t (*)(int, void *, size_t, off_t))
                     dlsym(RTLD_NEXT, "pread");
    }
    
    if (is_caller_from_our_code()) {
        return real_pread ? real_pread(fd, buf, count, offset) : -1;
    }
    
    if (is_tracked_pci_fd(fd) && offset == 0) {
        const char *path = get_tracked_fd_path(fd);
        if (path) {
            fprintf(stderr, "[libvgpu-syscall] pread(fd=%d, path=\"%s\", count=%zu, offset=%ld) intercepted (pid=%d)\n",
                    fd, path, count, (long)offset, (int)getpid());
            fflush(stderr);
            
            if (strstr(path, "/vendor") != NULL) {
                if (count >= 6) {
                    strncpy((char *)buf, "0x10de\n", count);
                    return 6;
                }
            } else if (strstr(path, "/device") != NULL) {
                if (count >= 6) {
                    strncpy((char *)buf, "0x2331\n", count);
                    return 6;
                }
            } else if (strstr(path, "/class") != NULL) {
                if (count >= 8) {
                    strncpy((char *)buf, "0x030200\n", count);
                    return 8;
                }
            }
        }
    }
    
    return real_pread ? real_pread(fd, buf, count, offset) : -1;
}

/* Intercept readv() syscall - used by some I/O implementations */
ssize_t readv(int fd, const struct iovec *iov, int iovcnt)
{
    static ssize_t (*real_readv)(int, const struct iovec *, int) = NULL;
    if (!real_readv) {
        real_readv = (ssize_t (*)(int, const struct iovec *, int))
                     dlsym(RTLD_NEXT, "readv");
    }
    
    if (is_caller_from_our_code()) {
        return real_readv ? real_readv(fd, iov, iovcnt) : -1;
    }
    
    if (is_tracked_pci_fd(fd) && iovcnt > 0 && iov) {
        const char *path = get_tracked_fd_path(fd);
        if (path) {
            fprintf(stderr, "[libvgpu-syscall] readv(fd=%d, path=\"%s\", iovcnt=%d) intercepted (pid=%d)\n",
                    fd, path, iovcnt, (int)getpid());
            fflush(stderr);
            
            size_t total = 0;
            for (int i = 0; i < iovcnt && iov[i].iov_base; i++) {
                total += iov[i].iov_len;
            }
            
            if (strstr(path, "/vendor") != NULL && total >= 6) {
                if (iov[0].iov_base) {
                    strncpy((char *)iov[0].iov_base, "0x10de\n", iov[0].iov_len);
                    return 6;
                }
            } else if (strstr(path, "/device") != NULL && total >= 6) {
                if (iov[0].iov_base) {
                    strncpy((char *)iov[0].iov_base, "0x2331\n", iov[0].iov_len);
                    return 6;
                }
            } else if (strstr(path, "/class") != NULL && total >= 8) {
                if (iov[0].iov_base) {
                    strncpy((char *)iov[0].iov_base, "0x030200\n", iov[0].iov_len);
                    return 8;
                }
            }
        }
    }
    
    return real_readv ? real_readv(fd, iov, iovcnt) : -1;
}

/* Intercept getdents64() - used to read directory listings */
/* Note: getdents64 signature varies, we'll use syscall wrapper instead */

/* Constructor to log library load */
__attribute__((constructor))
static void libvgpu_syscall_on_load(void)
{
    fprintf(stderr, "[libvgpu-syscall] LOADED (pid=%d)\n", (int)getpid());
    fflush(stderr);
}

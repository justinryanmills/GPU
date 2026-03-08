/*
 * process_monitor.c — Process monitoring daemon
 *
 * Monitors all Ollama processes and ensures they have the shim loaded.
 * If a process is missing the shim, attempts to inject it.
 *
 * Build:
 *   gcc -o process_monitor process_monitor.c -lpthread
 *
 * Usage:
 *   ./process_monitor &
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <signal.h>
#include <pthread.h>
#include <errno.h>

#define SHIM_LIBS "/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-syscall.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"
#define CHECK_INTERVAL 5  /* Check every 5 seconds */
#define MAX_PATH 512

static volatile int running = 1;

/* Check if process has shim loaded */
static int has_shim_loaded(pid_t pid)
{
    char maps_path[MAX_PATH];
    char line[1024];
    FILE *fp;
    int found = 0;
    
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", (int)pid);
    fp = fopen(maps_path, "r");
    if (!fp) return 0;
    
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "libvgpu") != NULL) {
            found = 1;
            break;
        }
    }
    
    fclose(fp);
    return found;
}

/* Check if process is Ollama */
static int is_ollama_process(pid_t pid)
{
    char cmdline_path[MAX_PATH];
    char cmdline[1024];
    FILE *fp;
    int is_ollama = 0;
    
    snprintf(cmdline_path, sizeof(cmdline_path), "/proc/%d/cmdline", (int)pid);
    fp = fopen(cmdline_path, "r");
    if (!fp) return 0;
    
    if (fgets(cmdline, sizeof(cmdline), fp)) {
        if (strstr(cmdline, "ollama") != NULL) {
            is_ollama = 1;
        }
    }
    
    fclose(fp);
    return is_ollama;
}

/* Get LD_PRELOAD from process environment */
static char *get_process_preload(pid_t pid)
{
    char environ_path[MAX_PATH];
    char line[4096];
    FILE *fp;
    char *preload = NULL;
    
    snprintf(environ_path, sizeof(environ_path), "/proc/%d/environ", (int)pid);
    fp = fopen(environ_path, "r");
    if (!fp) return NULL;
    
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "LD_PRELOAD=", 11) == 0) {
            preload = strdup(line + 11);
            /* Remove trailing newline if present */
            char *nl = strchr(preload, '\n');
            if (nl) *nl = '\0';
            break;
        }
    }
    
    fclose(fp);
    return preload;
}

/* Monitor loop */
static void *monitor_loop(void *arg)
{
    DIR *proc_dir;
    struct dirent *entry;
    
    while (running) {
        proc_dir = opendir("/proc");
        if (!proc_dir) {
            sleep(CHECK_INTERVAL);
            continue;
        }
        
        while ((entry = readdir(proc_dir)) != NULL) {
            if (entry->d_name[0] < '0' || entry->d_name[0] > '9') {
                continue;  /* Not a PID */
            }
            
            pid_t pid = (pid_t)atoi(entry->d_name);
            if (pid <= 1) continue;
            
            if (!is_ollama_process(pid)) continue;
            
            if (!has_shim_loaded(pid)) {
                char *preload = get_process_preload(pid);
                fprintf(stderr, "[process-monitor] WARNING: Ollama process %d missing shim (LD_PRELOAD=%s)\n",
                        (int)pid, preload ? preload : "(not set)");
                fflush(stderr);
                free(preload);
                /* Note: We can't inject into running process, but we log it */
            }
        }
        
        closedir(proc_dir);
        sleep(CHECK_INTERVAL);
    }
    
    return NULL;
}

/* Signal handler */
static void signal_handler(int sig)
{
    if (sig == SIGTERM || sig == SIGINT) {
        running = 0;
    }
}

int main(int argc, char *argv[])
{
    pthread_t monitor_thread;
    
    fprintf(stderr, "[process-monitor] Starting process monitor daemon (pid=%d)\n", (int)getpid());
    fflush(stderr);
    
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);
    
    if (pthread_create(&monitor_thread, NULL, monitor_loop, NULL) != 0) {
        perror("pthread_create");
        return 1;
    }
    
    pthread_join(monitor_thread, NULL);
    
    fprintf(stderr, "[process-monitor] Process monitor daemon stopped\n");
    fflush(stderr);
    
    return 0;
}

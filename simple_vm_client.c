#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#define SHM_PATH "/mnt/vgpu/commands"
#define SHM_SIZE 4096

typedef struct {
    uint32_t command;
    uint32_t status;
    char data[256];
} vgpu_message;

int main(int argc, char *argv[]) {
    int fd;
    vgpu_message *msg;
    uint32_t cmd_number = 42;
    int timeout = 0;
    
    if (argc > 1) {
        cmd_number = atoi(argv[1]);
    }
    
    fd = open(SHM_PATH, O_RDWR);
    if (fd < 0) {
        perror("Error: Cannot open shared memory");
        return 1;
    }
    
    msg = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE,
               MAP_SHARED, fd, 0);
    if (msg == MAP_FAILED) {
        perror("Error: Failed to map shared memory");
        close(fd);
        return 1;
    }
    
    msg->status = 0;
    msg->command = cmd_number;
    
    while (msg->status == 0 && timeout < 100) {
        usleep(100000);
        timeout++;
    }
    
    if (timeout >= 100) {
        munmap(msg, SHM_SIZE);
        close(fd);
        return 1;
    }
    
    munmap(msg, SHM_SIZE);
    close(fd);
    
    return 0;
}
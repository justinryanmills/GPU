#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>

#define SHM_PATH "/dev/shm/vgpu/commands"
#define SHM_SIZE 4096

typedef struct {
    uint32_t command;
    uint32_t status;
    char data[256];
} vgpu_message;

int main() {
    int fd;
    vgpu_message *msg;
    
    umask(000);
    fd = open(SHM_PATH, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        perror("Error: Failed to open shared memory");
        return 1;
    }

    fchmod(fd, 0666);
    
    if (ftruncate(fd, SHM_SIZE) < 0) {
        perror("Error: Failed to set shared memory size");
        close(fd);
        return 1;
    }
    
    msg = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE,
               MAP_SHARED, fd, 0);
    if (msg == MAP_FAILED) {
        perror("Error: Failed to map shared memory");
        close(fd);
        return 1;
    }
    
    msg->command = 0;
    msg->status = 0;
    strcpy(msg->data, "Ready");
    
    while (1) {
        if (msg->command != 0) {
            sleep(1);
            
            msg->status = 1;
            snprintf(msg->data, sizeof(msg->data), 
                     "Command %u completed successfully", msg->command);
            
            msg->command = 0;
        }
        
        usleep(10000);
    }
    
    munmap(msg, SHM_SIZE);
    close(fd);
    
    return 0;
}
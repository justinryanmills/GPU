#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#define CMD_FILE "/mnt/vgpu/command.txt"
#define RSP_FILE "/mnt/vgpu/response.txt"

int main(int argc, char *argv[]) {
    FILE *cmd_fp, *rsp_fp;
    uint32_t cmd_number = 42;
    char response[256];
    int status;
    int timeout = 0;
    
    if (argc > 1) {
        cmd_number = atoi(argv[1]);
    }
    
    cmd_fp = fopen(CMD_FILE, "w");
    if (!cmd_fp) {
        perror("Error: Cannot write command file");
        return 1;
    }
    
    fprintf(cmd_fp, "%u\n", cmd_number);
    fflush(cmd_fp);
    fclose(cmd_fp);
    
    sleep(1);
    
    while (timeout < 50) {
        rsp_fp = fopen(RSP_FILE, "r");
        if (rsp_fp) {
            if (fgets(response, sizeof(response), rsp_fp)) {
                if (sscanf(response, "%d:", &status) == 1) {
                    if (status != 0) {
                        fclose(rsp_fp);
                        break;
                    }
                }
            }
            fclose(rsp_fp);
        }
        
        usleep(100000);
        timeout++;
    }
    
    if (timeout >= 50) {
        return 1;
    }
    
    return 0;
}
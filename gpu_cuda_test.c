#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#define CMD_FILE "/mnt/vgpu/command.txt"
#define RSP_FILE "/mnt/vgpu/response.txt"

#define CMD_VECTOR_ADD   1
#define CMD_GPU_INFO     2

int send_command(uint32_t cmd, const char *cmd_name) {
    FILE *cmd_fp, *rsp_fp;
    char response[512];
    int status;
    int timeout = 0;
    
    cmd_fp = fopen(CMD_FILE, "w");
    if (!cmd_fp) {
        return -1;
    }
    fprintf(cmd_fp, "%u\n", cmd);
    fflush(cmd_fp);
    fclose(cmd_fp);
    
    sleep(2);
    
    while (timeout < 100) {
        rsp_fp = fopen(RSP_FILE, "r");
        if (rsp_fp) {
            if (fgets(response, sizeof(response), rsp_fp)) {
                if (sscanf(response, "%d:", &status) == 1) {
                    if (status != 0) {
                        fclose(rsp_fp);
                        return status == 1 ? 0 : -1;
                    }
                }
            }
            fclose(rsp_fp);
        }
        usleep(100000);
        timeout++;
    }
    
    return -1;
}

int main() {
    send_command(CMD_GPU_INFO, "GET GPU INFO");
    send_command(CMD_VECTOR_ADD, "RUN VECTOR ADD (CUDA)");
    
    return 0;
}
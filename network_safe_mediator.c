#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#define CMD_FILE "/dev/shm/vgpu/command.txt"
#define RSP_FILE "/dev/shm/vgpu/response.txt"

int main() {
    FILE *cmd_fp, *rsp_fp;
    uint32_t last_cmd = 0;
    uint32_t current_cmd = 0;
    unsigned int cmd_count = 0;
    
    rsp_fp = fopen(RSP_FILE, "w");
    if (rsp_fp) {
        fprintf(rsp_fp, "0:Ready\n");
        fclose(rsp_fp);
    }
    
    while (1) {
        cmd_fp = fopen(CMD_FILE, "r");
        if (cmd_fp) {
            if (fscanf(cmd_fp, "%u", &current_cmd) == 1) {
                if (current_cmd != last_cmd && current_cmd != 0) {
                    cmd_count++;
                    
                    sleep(1);
                    
                    rsp_fp = fopen(RSP_FILE, "w");
                    if (rsp_fp) {
                        fprintf(rsp_fp, "1:Command %u completed successfully\n", 
                                current_cmd);
                        fflush(rsp_fp);
                        fclose(rsp_fp);
                    }
                    
                    last_cmd = current_cmd;
                }
            }
            fclose(cmd_fp);
        }
        
        usleep(50000);
    }
    
    return 0;
}
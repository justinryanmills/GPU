#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#define CMD_FILE "/dev/shm/vgpu/command.txt"
#define RSP_FILE "/dev/shm/vgpu/response.txt"

#define CMD_NOP          0
#define CMD_VECTOR_ADD   1
#define CMD_GPU_INFO     2

extern int run_vector_add_test(char *result_msg, int max_len);

void get_timestamp(char *buffer, int size) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(buffer, size, "%H:%M:%S", t);
}

int main() {
    FILE *cmd_fp, *rsp_fp;
    uint32_t last_cmd = 0;
    uint32_t current_cmd = 0;
    unsigned int cmd_count = 0;
    char timestamp[32];
    char result_msg[512];
    int result;
    
    rsp_fp = fopen(RSP_FILE, "w");
    if (rsp_fp) {
        fprintf(rsp_fp, "0:Ready - CUDA enabled\n");
        fclose(rsp_fp);
    }
    
    while (1) {
        cmd_fp = fopen(CMD_FILE, "r");
        if (cmd_fp) {
            if (fscanf(cmd_fp, "%u", &current_cmd) == 1) {
                if (current_cmd != last_cmd && current_cmd != 0) {
                    cmd_count++;
                    get_timestamp(timestamp, sizeof(timestamp));
                    
                    switch (current_cmd) {
                        case CMD_VECTOR_ADD:
                            result = run_vector_add_test(result_msg, sizeof(result_msg));
                            
                            rsp_fp = fopen(RSP_FILE, "w");
                            if (rsp_fp) {
                                fprintf(rsp_fp, "%d:%s\n", 
                                        (result == 0) ? 1 : 2, result_msg);
                                fflush(rsp_fp);
                                fclose(rsp_fp);
                            }
                            break;
                            
                        case CMD_GPU_INFO:
                            rsp_fp = fopen(RSP_FILE, "w");
                            if (rsp_fp) {
                                fprintf(rsp_fp, "1:GPU: NVIDIA H100 (see nvidia-smi for exact model/VRAM)\n");
                                fflush(rsp_fp);
                                fclose(rsp_fp);
                            }
                            break;
                            
                        default:
                            rsp_fp = fopen(RSP_FILE, "w");
                            if (rsp_fp) {
                                fprintf(rsp_fp, "2:Unknown command: %u\n", current_cmd);
                                fflush(rsp_fp);
                                fclose(rsp_fp);
                            }
                            break;
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
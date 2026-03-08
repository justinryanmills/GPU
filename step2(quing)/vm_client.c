/*
 * VM GPU Request Client
 * 
 * Purpose: Read vGPU properties from MMIO and send requests to mediation daemon
 * 
 * Features:
 * - Reads pool_id, priority, vm_id from vGPU stub device
 * - Sends formatted request to daemon via NFS
 * - Waits for and displays response
 * 
 * Usage: sudo ./vm_client <command>
 * Example: sudo ./vm_client VECTOR_ADD
 * 
 * Requirements:
 * - VM must have vgpu-stub device attached
 * - NFS share must be mounted at /mnt/vgpu
 * - Must run as root to access PCI resources
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

typedef struct {
    char pool_id;        // 'A' or 'B'
    uint32_t priority;   // 0=low, 1=medium, 2=high
    uint32_t vm_id;      // Unique VM identifier
} VGPUProperties;

/*
 * Read vGPU properties from MMIO registers
 * Returns 0 on success, -1 on failure
 */
int read_vgpu_properties(VGPUProperties *props) {
    const char *pci_resource = "/sys/bus/pci/devices/0000:00:06.0/resource0";
    int fd;
    volatile uint32_t *mmio;
    
    // Open PCI resource
    fd = open(pci_resource, O_RDONLY);
    if (fd < 0) {
        perror("Failed to open vGPU device");
        fprintf(stderr, "Possible reasons:\n");
        fprintf(stderr, "  1. Not running as root (use sudo)\n");
        fprintf(stderr, "  2. vgpu-stub device not attached to VM\n");
        fprintf(stderr, "  3. Wrong PCI address (check with lspci)\n");
        return -1;
    }
    
    // Map MMIO region (4KB)
    mmio = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 0);
    if (mmio == MAP_FAILED) {
        perror("Failed to mmap MMIO");
        close(fd);
        return -1;
    }
    
    // Read properties from MMIO registers (as defined in vgpu-stub.c)
    props->pool_id = (char)mmio[0x008/4];      // Offset 0x008: Pool ID
    props->priority = mmio[0x00C/4];           // Offset 0x00C: Priority
    props->vm_id = mmio[0x010/4];              // Offset 0x010: VM ID
    
    // Cleanup
    munmap((void*)mmio, 4096);
    close(fd);
    
    printf("[MMIO] Read vGPU properties from device\n");
    return 0;
}

/*
 * Send request to mediation daemon
 * Format: "pool_id:priority:vm_id:command"
 */
int send_request(VGPUProperties *props, const char *command) {
    char request_file[256];
    char request_data[512];
    FILE *fp;
    
    // Construct per-VM request file path
    snprintf(request_file, sizeof(request_file), 
             "/mnt/vgpu/vm%u/request.txt", props->vm_id);
    
    // Format: "pool_id:priority:vm_id:command"
    snprintf(request_data, sizeof(request_data),
             "%c:%u:%u:%s",
             props->pool_id, props->priority, props->vm_id, command);
    
    // Write request (explicit I/O to ensure NFS propagation)
    fp = fopen(request_file, "w");
    if (!fp) {
        perror("Failed to open request file");
        fprintf(stderr, "Make sure /mnt/vgpu/vm%u/ directory exists\n", props->vm_id);
        return -1;
    }
    
    fprintf(fp, "%s\n", request_data);
    fflush(fp);  // Force write to NFS
    fclose(fp);  // Ensure data is on disk
    
    printf("[SEND] Request: %s\n", request_data);
    return 0;
}

/*
 * Wait for response from daemon
 * Returns 0 on success, -1 on timeout
 */
int wait_for_response(VGPUProperties *props, char *response, size_t len, int timeout_sec) {
    char response_file[256];
    FILE *fp;
    int waited = 0;
    
    // Construct per-VM response file path
    snprintf(response_file, sizeof(response_file),
             "/mnt/vgpu/vm%u/response.txt", props->vm_id);
    
    printf("[WAIT] Polling for response (timeout: %ds)...\n", timeout_sec);
    
    // Poll for response
    while (waited < timeout_sec) {
        fp = fopen(response_file, "r");
        if (fp) {
            if (fgets(response, len, fp)) {
                // Check if response is not "0:Ready" (initial state)
                if (strncmp(response, "0:", 2) != 0) {
                    fclose(fp);
                    
                    // Remove trailing newline
                    response[strcspn(response, "\n")] = 0;
                    
                    return 0;  // Got real response
                }
            }
            fclose(fp);
        }
        
        sleep(1);
        waited++;
        
        // Show progress
        if (waited % 5 == 0) {
            printf("[WAIT] Still waiting... (%ds elapsed)\n", waited);
        }
    }
    
    return -1;  // Timeout
}

/*
 * Main program
 */
int main(int argc, char *argv[]) {
    VGPUProperties props;
    char response[512];
    const char *command = "VECTOR_ADD";  // Default command
    int timeout = 30;  // 30 second timeout
    
    printf("================================================================================\n");
    printf("                   VM GPU Request Client\n");
    printf("================================================================================\n");
    printf("\n");
    
    // Parse command line arguments
    if (argc > 1) {
        command = argv[1];
    }
    if (argc > 2) {
        timeout = atoi(argv[2]);
    }
    
    // Step 1: Read vGPU properties from MMIO
    printf("[STEP 1] Reading vGPU device properties...\n");
    if (read_vgpu_properties(&props) < 0) {
        fprintf(stderr, "\n[ERROR] Cannot read vGPU properties\n");
        fprintf(stderr, "Action: Verify vgpu-stub device is attached to VM\n");
        fprintf(stderr, "Check: lspci | grep 'Processing accelerators'\n");
        return 1;
    }
    
    printf("\n[vGPU Properties]\n");
    printf("  Pool ID:  %c\n", props.pool_id);
    printf("  Priority: %u (%s)\n", props.priority,
           props.priority == 2 ? "high" : 
           props.priority == 1 ? "medium" : "low");
    printf("  VM ID:    %u\n", props.vm_id);
    printf("\n");
    
    // Validate properties
    if (props.pool_id != 'A' && props.pool_id != 'B') {
        fprintf(stderr, "[WARN] Invalid pool_id: %c (expected A or B)\n", props.pool_id);
        fprintf(stderr, "        Using pool_id 'A' as fallback\n");
        props.pool_id = 'A';
    }
    if (props.priority > 2) {
        fprintf(stderr, "[WARN] Invalid priority: %u (expected 0-2)\n", props.priority);
        fprintf(stderr, "        Using priority 1 (medium) as fallback\n");
        props.priority = 1;
    }
    
    // Step 2: Send request to daemon
    printf("[STEP 2] Sending request to mediation daemon...\n");
    printf("  Command: %s\n", command);
    if (send_request(&props, command) < 0) {
        fprintf(stderr, "\n[ERROR] Failed to send request\n");
        fprintf(stderr, "Action: Verify NFS is mounted at /mnt/vgpu\n");
        fprintf(stderr, "Check: mount | grep /mnt/vgpu\n");
        return 1;
    }
    printf("  Status: Sent successfully\n");
    printf("\n");
    
    // Step 3: Wait for response
    printf("[STEP 3] Waiting for response from daemon...\n");
    if (wait_for_response(&props, response, sizeof(response), timeout) < 0) {
        fprintf(stderr, "\n[ERROR] Timeout waiting for response (%ds)\n", timeout);
        fprintf(stderr, "Action: Check if mediation daemon is running\n");
        fprintf(stderr, "Check: ps aux | grep mediator\n");
        return 1;
    }
    
    printf("\n[RESPONSE RECEIVED]\n");
    printf("  Raw: %s\n", response);
    
    // Parse response status
    printf("\n");
    if (strncmp(response, "1:", 2) == 0) {
        printf("SUCCESS! GPU workload completed\n");
        printf("   Message: %s\n", response + 2);
        return 0;
    } else if (strncmp(response, "2:", 2) == 0) {
        printf("❌ ERROR from daemon\n");
        printf("   Message: %s\n", response + 2);
        return 1;
    } else {
        printf("⚠️  Unexpected response format\n");
        return 1;
    }
}

/*
 * VM Client - Vector Addition Request
 * 
 * Purpose: Send vector addition requests to MEDIATOR via NFS
 * 
 * Features:
 * - Reads vGPU properties from MMIO
 * - Sends formatted request to MEDIATOR
 * - Waits for and displays result
 * 
 * Usage: sudo ./vm_client_vector <num1> <num2>
 * Example: sudo ./vm_client_vector 100 200
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
#include <errno.h>
#include <dirent.h>

#define NFS_MOUNT "/mnt/vgpu"
#define RESPONSE_TIMEOUT 30  // seconds
#define POLL_INTERVAL 100000  // microseconds (0.1 seconds)

// vGPU device identifiers
#define VGPU_VENDOR_ID 0x1af4  // Red Hat, Inc.
#define VGPU_DEVICE_ID 0x1111  // Custom vGPU stub
#define VGPU_CLASS_MASK 0xffff00
#define VGPU_CLASS 0x120000    // Processing Accelerator

/*
 * vGPU Properties Structure
 */
typedef struct {
    char pool_id;        // 'A' or 'B'
    uint32_t priority;   // 0=low, 1=medium, 2=high
    uint32_t vm_id;      // Unique VM identifier
} VGPUProperties;

/*
 * Find vGPU stub device by scanning PCI devices
 * Returns path to resource0 file, or NULL if not found
 */
static char* find_vgpu_device(void) {
    static char device_path[1024];
    DIR *dir;
    struct dirent *entry;
    char pci_path[1024];
    char vendor_file[1024];
    char device_file[1024];
    char class_file[1024];
    FILE *fp;
    char line[256];
    unsigned int vendor, device, class;
    int len;
    
    // Scan /sys/bus/pci/devices for vGPU stub
    dir = opendir("/sys/bus/pci/devices");
    if (!dir) {
        perror("Failed to open /sys/bus/pci/devices");
        return NULL;
    }
    
    printf("[SCAN] Searching for vGPU stub device...\n");
    
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (entry->d_name[0] == '.') continue;
        
        // Build PCI path with bounds checking
        len = snprintf(pci_path, sizeof(pci_path), "/sys/bus/pci/devices/%s", entry->d_name);
        if (len < 0 || len >= (int)sizeof(pci_path)) {
            continue;  // Path too long, skip
        }
        
        // Check vendor file (Red Hat = 0x1af4)
        len = snprintf(vendor_file, sizeof(vendor_file), "%s/vendor", pci_path);
        if (len < 0 || len >= (int)sizeof(vendor_file)) continue;
        
        fp = fopen(vendor_file, "r");
        if (!fp) continue;
        
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &vendor);
            fclose(fp);
            
            // Red Hat vendor ID
            if (vendor != VGPU_VENDOR_ID) continue;
        } else {
            fclose(fp);
            continue;
        }
        
        // Check device ID (0x1111 = our vGPU stub)
        len = snprintf(device_file, sizeof(device_file), "%s/device", pci_path);
        if (len < 0 || len >= (int)sizeof(device_file)) continue;
        
        fp = fopen(device_file, "r");
        if (!fp) continue;
        
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &device);
            fclose(fp);
            
            if (device != VGPU_DEVICE_ID) continue;
        } else {
            fclose(fp);
            continue;
        }
        
        // Check class (0x120000 = Processing Accelerator)
        len = snprintf(class_file, sizeof(class_file), "%s/class", pci_path);
        if (len < 0 || len >= (int)sizeof(class_file)) continue;
        
        fp = fopen(class_file, "r");
        if (!fp) continue;
        
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &class);
            fclose(fp);
            
            // Processing Accelerator class
            if ((class & VGPU_CLASS_MASK) != VGPU_CLASS) continue;
        } else {
            fclose(fp);
            continue;
        }
        
        // Found it! Check if resource0 exists and is readable
        len = snprintf(device_path, sizeof(device_path), "%s/resource0", pci_path);
        if (len < 0 || len >= (int)sizeof(device_path)) continue;
        
        if (access(device_path, R_OK) == 0) {
            printf("[SCAN] Found vGPU stub at %s\n", entry->d_name);
            closedir(dir);
            return device_path;
        }
    }
    
    closedir(dir);
    fprintf(stderr, "[SCAN] vGPU stub device not found\n");
    fprintf(stderr, "       Expected: Vendor=0x%04x, Device=0x%04x, Class=0x%06x\n",
            VGPU_VENDOR_ID, VGPU_DEVICE_ID, VGPU_CLASS);
    return NULL;
}

/*
 * Read vGPU properties from MMIO registers
 * Returns 0 on success, -1 on failure
 */
int read_vgpu_properties(VGPUProperties *props) {
    int fd;
    volatile uint32_t *mmio;
    const char *pci_resource;
    
    // Auto-detect vGPU device
    pci_resource = find_vgpu_device();
    if (!pci_resource) {
        fprintf(stderr, "Failed to find vGPU device\n");
        fprintf(stderr, "Possible reasons:\n");
        fprintf(stderr, "  1. Not running as root (use sudo)\n");
        fprintf(stderr, "  2. vgpu-stub device not attached to VM\n");
        fprintf(stderr, "  3. Check with: lspci | grep 'Processing accelerators'\n");
        return -1;
    }
    
    // Open PCI resource
    fd = open(pci_resource, O_RDONLY);
    if (fd < 0) {
        perror("Failed to open vGPU device");
        fprintf(stderr, "Device path: %s\n", pci_resource);
        return -1;
    }
    
    // Map MMIO region (4KB)
    mmio = (volatile uint32_t *)mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 0);
    if (mmio == MAP_FAILED) {
        perror("Failed to mmap MMIO");
        close(fd);
        return -1;
    }
    
    // Read properties from MMIO registers (as defined in vgpu-stub.c)
    props->pool_id = (char)mmio[0x008/4];      // Offset 0x008: Pool ID
    props->priority = mmio[0x00C/4];           // Offset 0x00C: Priority
    props->vm_id = mmio[0x010/4];              // Offset 0x010: VM ID
    
    // Validate properties
    if (props->pool_id != 'A' && props->pool_id != 'B') {
        fprintf(stderr, "[WARNING] Invalid pool_id: %c, defaulting to 'A'\n", props->pool_id);
        props->pool_id = 'A';
    }
    
    if (props->priority > 2) {
        fprintf(stderr, "[WARNING] Invalid priority: %u, defaulting to 1 (medium)\n", props->priority);
        props->priority = 1;
    }
    
    if (props->vm_id == 0 || props->vm_id > 100) {
        fprintf(stderr, "[WARNING] Invalid vm_id: %u, using directory-based detection\n", props->vm_id);
        // Try to detect from directory name or use default
        props->vm_id = 1;  // Default fallback
    }
    
    // Cleanup
    munmap((void*)mmio, 4096);
    close(fd);
    
    printf("[MMIO] Read vGPU properties:\n");
    printf("  Pool ID: %c\n", props->pool_id);
    printf("  Priority: %u (%s)\n", props->priority,
           props->priority == 2 ? "high" : (props->priority == 1 ? "medium" : "low"));
    printf("  VM ID: %u\n", props->vm_id);
    
    return 0;
}

/*
 * Send vector addition request to MEDIATOR
 * Format: "pool_id:priority:vm_id:num1:num2"
 */
int send_request(VGPUProperties *props, int num1, int num2) {
    char request_file[512];
    char request_data[256];
    FILE *fp;
    
    // Construct per-VM request file path
    snprintf(request_file, sizeof(request_file), 
             "%s/vm%u/request.txt", NFS_MOUNT, props->vm_id);
    
    // Format: "pool_id:priority:vm_id:num1:num2"
    snprintf(request_data, sizeof(request_data),
             "%c:%u:%u:%d:%d",
             props->pool_id, props->priority, props->vm_id, num1, num2);
    
    // Write request (explicit I/O to ensure NFS propagation)
    fp = fopen(request_file, "w");
    if (!fp) {
        perror("Failed to open request file");
        fprintf(stderr, "Make sure %s/vm%u/ directory exists\n", NFS_MOUNT, props->vm_id);
        fprintf(stderr, "Check NFS mount: mount | grep %s\n", NFS_MOUNT);
        return -1;
    }
    
    fprintf(fp, "%s\n", request_data);
    fflush(fp);
    fsync(fileno(fp));
    fclose(fp);
    
    printf("[REQUEST] Sent to MEDIATOR:\n");
    printf("  Format: %s\n", request_data);
    printf("  File: %s\n", request_file);
    
    return 0;
}

/*
 * Wait for response from MEDIATOR
 * Returns 0 on success, -1 on timeout/error
 */
int wait_for_response(VGPUProperties *props, int *result) {
    char response_file[512];
    FILE *fp;
    time_t start_time = time(NULL);
    time_t current_time;
    
    // Construct response file path
    snprintf(response_file, sizeof(response_file),
             "%s/vm%u/response.txt", NFS_MOUNT, props->vm_id);
    
    printf("[WAIT] Polling for response...\n");
    printf("  File: %s\n", response_file);
    printf("  Timeout: %d seconds\n", RESPONSE_TIMEOUT);
    
    // Poll for response
    while (1) {
        current_time = time(NULL);
        
        // Check timeout
        if (current_time - start_time >= RESPONSE_TIMEOUT) {
            fprintf(stderr, "[ERROR] Timeout waiting for response\n");
            return -1;
        }
        
        // Try to read response file
        fp = fopen(response_file, "r");
        if (fp) {
            // File exists, try to read result
            char line[256];
            if (fgets(line, sizeof(line), fp) != NULL) {
                // Parse result
                if (sscanf(line, "%d", result) == 1) {
                    fclose(fp);
                    printf("[RESPONSE] Received: %d\n", *result);
                    
                    // Clear response file after reading to signal MEDIATOR that response was received
                    fp = fopen(response_file, "w");
                    if (fp) {
                        fclose(fp);  // Truncate to zero
                        printf("[CLEANUP] Cleared response file\n");
                    }
                    
                    return 0;
                }
            }
            fclose(fp);
        }
        
        // Wait before next poll
        usleep(POLL_INTERVAL);
    }
    
    return -1;  // Should not reach here
}

/*
 * Main function
 */
int main(int argc, char *argv[]) {
    VGPUProperties props;
    int num1, num2;
    int result;
    
    printf("================================================================================\n");
    printf("                    VM CLIENT - Vector Addition Request\n");
    printf("================================================================================\n\n");
    
    // Check arguments
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num1> <num2>\n", argv[0]);
        fprintf(stderr, "Example: %s 100 200\n", argv[0]);
        return 1;
    }
    
    num1 = atoi(argv[1]);
    num2 = atoi(argv[2]);
    
    if (num1 == 0 && argv[1][0] != '0') {
        fprintf(stderr, "ERROR: Invalid number1: %s\n", argv[1]);
        return 1;
    }
    
    if (num2 == 0 && argv[2][0] != '0') {
        fprintf(stderr, "ERROR: Invalid number2: %s\n", argv[2]);
        return 1;
    }
    
    printf("Request: %d + %d\n\n", num1, num2);
    
    // Read vGPU properties
    if (read_vgpu_properties(&props) != 0) {
        fprintf(stderr, "ERROR: Failed to read vGPU properties\n");
        return 1;
    }
    
    printf("\n");
    
    // Send request
    if (send_request(&props, num1, num2) != 0) {
        fprintf(stderr, "ERROR: Failed to send request\n");
        return 1;
    }
    
    printf("\n");
    
    // Wait for response
    if (wait_for_response(&props, &result) != 0) {
        fprintf(stderr, "ERROR: Failed to receive response\n");
        return 1;
    }
    
    printf("\n");
    printf("================================================================================\n");
    printf("                    RESULT\n");
    printf("================================================================================\n");
    printf("  %d + %d = %d\n", num1, num2, result);
    printf("================================================================================\n");
    
    return 0;
}

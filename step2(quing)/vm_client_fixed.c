/*
 * VM GPU Request Client: read MMIO, send requests to mediation daemon.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

typedef struct {
    char pool_id;        // 'A' or 'B'
    uint32_t priority;   // 0=low, 1=medium, 2=high
    uint32_t vm_id;      // Unique VM identifier
} VGPUProperties;

char* find_vgpu_device(void) {
    static char device_path[1024];  // Increased size to avoid truncation warnings
    DIR *dir;
    struct dirent *entry;
    char pci_path[1024];      // Increased size
    char vendor_file[1024];   // Increased size
    char device_file[1024];   // Renamed for clarity
    char class_file[1024];    // Increased size
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
            if (vendor != 0x1af4) continue;
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
            
            if (device != 0x1111) continue;
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
            if ((class & 0xffff00) != 0x120000) continue;
        } else {
            fclose(fp);
            continue;
        }
        
        // Found it! Check if resource0 exists
        len = snprintf(device_path, sizeof(device_path), "%s/resource0", pci_path);
        if (len < 0 || len >= (int)sizeof(device_path)) continue;
        
        if (access(device_path, R_OK) == 0) {
            printf("[SCAN] Found vGPU stub at %s\n", entry->d_name);
            closedir(dir);
            return device_path;
        }
    }
    
    closedir(dir);
    printf("[SCAN] vGPU stub device not found\n");
    return NULL;
}

int read_vgpu_properties(VGPUProperties *props) {
    char *pci_resource;
    int fd;
    volatile uint32_t *mmio;
    
    // Find device dynamically (not hardcoded)
    pci_resource = find_vgpu_device();
    if (!pci_resource) {
        fprintf(stderr, "\n[ERROR] vGPU stub device not found\n");
        fprintf(stderr, "Possible reasons:\n");
        fprintf(stderr, "  1. vgpu-stub device not attached to VM\n");
        fprintf(stderr, "  2. Device not visible (check with: lspci | grep 'Processing accelerators')\n");
        fprintf(stderr, "  3. Wrong vendor/device ID (expected: 1af4:1111)\n");
        return -1;
    }
    
    printf("[MMIO] Opening device: %s\n", pci_resource);
    
    // Open PCI resource
    fd = open(pci_resource, O_RDONLY);
    if (fd < 0) {
        perror("Failed to open vGPU device");
        fprintf(stderr, "Possible reasons:\n");
        fprintf(stderr, "  1. Not running as root (use sudo)\n");
        fprintf(stderr, "  2. Device permissions issue\n");
        return -1;
    }
    
    // Map MMIO region (4KB)
    mmio = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 0);
    if (mmio == MAP_FAILED) {
        perror("Failed to mmap MMIO");
        close(fd);
        return -1;
    }
    
    printf("[MMIO] Mapped 4KB region successfully\n");
    
    // Read properties from MMIO registers (as defined in vgpu-stub.c)
    // Offset 0x008: Pool ID (ASCII character)
    // Offset 0x00C: Priority (0=low, 1=medium, 2=high)
    // Offset 0x010: VM ID (32-bit unsigned integer)
    
    props->pool_id = (char)mmio[0x008/4];      // Offset 0x008: Pool ID
    props->priority = mmio[0x00C/4];           // Offset 0x00C: Priority
    props->vm_id = mmio[0x010/4];              // Offset 0x010: VM ID
    
    printf("[MMIO] Read registers:\n");
    printf("  0x008 (Pool ID):  0x%08x = '%c'\n", mmio[0x008/4], props->pool_id);
    printf("  0x00C (Priority): 0x%08x = %u\n", mmio[0x00C/4], props->priority);
    printf("  0x010 (VM ID):    0x%08x = %u\n", mmio[0x010/4], props->vm_id);
    
    // Cleanup
    munmap((void*)mmio, 4096);
    close(fd);
    
    return 0;
}

int send_request(VGPUProperties *props, const char *command) {
    char request_file[256];
    char request_data[512];
    FILE *fp;
    struct stat st;
    
    // Construct per-VM request file path
    snprintf(request_file, sizeof(request_file), 
             "/mnt/vgpu/vm%u/request.txt", props->vm_id);
    
    // Check if directory exists
    char dir_path[256];
    snprintf(dir_path, sizeof(dir_path), "/mnt/vgpu/vm%u", props->vm_id);
    if (stat(dir_path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "[ERROR] Directory does not exist: %s\n", dir_path);
        fprintf(stderr, "Action: Create directory on Dom0: mkdir -p /var/vgpu/vm%u\n", props->vm_id);
        return -1;
    }
    
    // Format: "pool_id:priority:vm_id:command"
    snprintf(request_data, sizeof(request_data),
             "%c:%u:%u:%s",
             props->pool_id, props->priority, props->vm_id, command);
    
    // Write request (explicit I/O to ensure NFS propagation)
    fp = fopen(request_file, "w");
    if (!fp) {
        perror("Failed to open request file");
        fprintf(stderr, "File: %s\n", request_file);
        fprintf(stderr, "Make sure:\n");
        fprintf(stderr, "  1. NFS is mounted: mount | grep /mnt/vgpu\n");
        fprintf(stderr, "  2. Directory exists: ls -la %s\n", dir_path);
        return -1;
    }
    
    fprintf(fp, "%s\n", request_data);
    fflush(fp);  // Force write to NFS
    fclose(fp);  // Ensure data is on disk
    
    printf("[SEND] Request written: %s\n", request_data);
    printf("[SEND] File: %s\n", request_file);
    return 0;
}

int wait_for_response(VGPUProperties *props, char *response, size_t len, int timeout_sec) {
    char response_file[256];
    FILE *fp;
    int waited = 0;
    
    // Construct per-VM response file path
    snprintf(response_file, sizeof(response_file),
             "/mnt/vgpu/vm%u/response.txt", props->vm_id);
    
    printf("[WAIT] Polling for response (timeout: %ds)...\n", timeout_sec);
    printf("[WAIT] File: %s\n", response_file);
    
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
                    
                    printf("[WAIT] Response received after %ds\n", waited);
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

int main(int argc, char *argv[]) {
    VGPUProperties props;
    char response[512];
    const char *command = "VECTOR_ADD";  // Default command
    int timeout = 30;  // 30 second timeout
    
    printf("================================================================================\n");
    printf("                   VM GPU Request Client (FIXED VERSION)\n");
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
        fprintf(stderr, "\nTroubleshooting steps:\n");
        fprintf(stderr, "  1. Check device is attached: lspci | grep 'Processing accelerators'\n");
        fprintf(stderr, "  2. Verify vendor/device: lspci -nn | grep 1af4:1111\n");
        fprintf(stderr, "  3. Check permissions: ls -la /sys/bus/pci/devices/*/resource0\n");
        fprintf(stderr, "  4. Run as root: sudo ./vm_client\n");
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
        fprintf(stderr, "\nTroubleshooting steps:\n");
        fprintf(stderr, "  1. Check NFS mount: mount | grep /mnt/vgpu\n");
        fprintf(stderr, "  2. Verify directory exists: ls -la /mnt/vgpu/vm%u\n", props.vm_id);
        fprintf(stderr, "  3. Check permissions: ls -ld /mnt/vgpu/vm%u\n", props.vm_id);
        return 1;
    }
    printf("  Status: Sent successfully\n");
    printf("\n");
    
    // Step 3: Wait for response
    printf("[STEP 3] Waiting for response from daemon...\n");
    if (wait_for_response(&props, response, sizeof(response), timeout) < 0) {
        fprintf(stderr, "\n[ERROR] Timeout waiting for response (%ds)\n", timeout);
        fprintf(stderr, "\nTroubleshooting steps:\n");
        fprintf(stderr, "  1. Check if mediator is running: ps aux | grep mediator\n");
        fprintf(stderr, "  2. Check mediator logs\n");
        fprintf(stderr, "  3. Verify request file was read: cat /mnt/vgpu/vm%u/request.txt\n", props.vm_id);
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

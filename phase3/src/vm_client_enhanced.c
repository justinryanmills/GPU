
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

/* Protocol constants (from vgpu_protocol.h) */
#define VGPU_REG_DOORBELL        0x000
#define VGPU_REG_STATUS          0x004
#define VGPU_REG_POOL_ID         0x008
#define VGPU_REG_PRIORITY        0x00C
#define VGPU_REG_VM_ID           0x010
#define VGPU_REG_ERROR_CODE      0x014
#define VGPU_REG_REQUEST_LEN      0x018
#define VGPU_REG_RESPONSE_LEN    0x01C
#define VGPU_REG_PROTOCOL_VER    0x020
#define VGPU_REG_REQUEST_ID       0x030

#define VGPU_REQ_BUFFER_OFFSET   0x040
#define VGPU_RESP_BUFFER_OFFSET  0x440
#define VGPU_BAR_SIZE            4096

#define VGPU_STATUS_IDLE         0x00
#define VGPU_STATUS_BUSY         0x01
#define VGPU_STATUS_DONE         0x02
#define VGPU_STATUS_ERROR        0x03

#define VGPU_ERR_NONE              0x00
#define VGPU_ERR_MEDIATOR_UNAVAIL  0x03
#define VGPU_ERR_TIMEOUT           0x04
#define VGPU_ERR_QUEUE_FULL        0x07
/* Phase 3: Isolation error codes */
#define VGPU_ERR_RATE_LIMITED      0x0A
#define VGPU_ERR_VM_QUARANTINED    0x0B

#define VGPU_PROTOCOL_VERSION    0x00010000
#define VGPU_OP_CUDA_KERNEL      0x0001

#define VGPU_REQUEST_HEADER_SIZE 32
#define VGPU_RESPONSE_HEADER_SIZE 32

/* vGPU device identifiers */
#define VGPU_VENDOR_ID 0x10de
#define VGPU_DEVICE_ID 0x2331
#define VGPU_CLASS_MASK 0xffff00
#define VGPU_CLASS 0x030200

#define RESPONSE_TIMEOUT 30    // seconds
#define POLL_INTERVAL 10000    // microseconds (10ms)

/* Phase 3: Retry logic for back-pressure / rate-limit */
#define MAX_RETRIES       5    // Maximum retry attempts for BUSY/RATE_LIMITED
#define INITIAL_BACKOFF_MS 100 // Initial backoff in milliseconds
#define MAX_BACKOFF_MS   5000  // Maximum backoff in milliseconds

/* Helper macro for register access */
#define REG32(base, off)  (*(volatile uint32_t *)((volatile char *)(base) + (off)))

/*
 * VGPURequest structure (from vgpu_protocol.h)
 */
typedef struct __attribute__((packed)) {
    uint32_t version;
    uint32_t opcode;
    uint32_t flags;
    uint32_t param_count;
    uint32_t data_offset;
    uint32_t data_length;
    uint32_t reserved[2];
} VGPURequest;

/*
 * VGPUResponse structure (from vgpu_protocol.h)
 */
typedef struct __attribute__((packed)) {
    uint32_t version;
    uint32_t status;
    uint32_t result_count;
    uint32_t data_offset;
    uint32_t data_length;
    uint32_t exec_time_us;
    uint32_t reserved[2];
} VGPUResponse;

/*
 * vGPU Properties Structure
 */
typedef struct {
    char pool_id;
    uint32_t priority;
    uint32_t vm_id;
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
    
    dir = opendir("/sys/bus/pci/devices");
    if (!dir) {
        perror("Failed to open /sys/bus/pci/devices");
        return NULL;
    }
    
    printf("[SCAN] Searching for vGPU stub device...\n");
    
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        
        len = snprintf(pci_path, sizeof(pci_path), "/sys/bus/pci/devices/%s", entry->d_name);
        if (len < 0 || len >= (int)sizeof(pci_path)) continue;
        
        len = snprintf(vendor_file, sizeof(vendor_file), "%s/vendor", pci_path);
        if (len < 0 || len >= (int)sizeof(vendor_file)) continue;
        
        fp = fopen(vendor_file, "r");
        if (!fp) continue;
        
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &vendor);
            fclose(fp);
            if (vendor != VGPU_VENDOR_ID) continue;
        } else {
            fclose(fp);
            continue;
        }
        
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
        
        len = snprintf(class_file, sizeof(class_file), "%s/class", pci_path);
        if (len < 0 || len >= (int)sizeof(class_file)) continue;
        
        fp = fopen(class_file, "r");
        if (!fp) continue;
        
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &class);
            fclose(fp);
            if ((class & VGPU_CLASS_MASK) != VGPU_CLASS) continue;
        } else {
            fclose(fp);
            continue;
        }
        
        len = snprintf(device_path, sizeof(device_path), "%s/resource0", pci_path);
        if (len < 0 || len >= (int)sizeof(device_path)) continue;
        
        if (access(device_path, R_OK | W_OK) == 0) {
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
static int read_vgpu_properties(volatile void *mmio, VGPUProperties *props) {
    props->pool_id = (char)REG32(mmio, VGPU_REG_POOL_ID);
    props->priority = REG32(mmio, VGPU_REG_PRIORITY);
    props->vm_id = REG32(mmio, VGPU_REG_VM_ID);
    
    if (props->pool_id != 'A' && props->pool_id != 'B') {
        fprintf(stderr, "[WARNING] Invalid pool_id: %c, defaulting to 'A'\n", props->pool_id);
        props->pool_id = 'A';
    }
    
    if (props->priority > 2) {
        fprintf(stderr, "[WARNING] Invalid priority: %u, defaulting to 1 (medium)\n", props->priority);
        props->priority = 1;
    }
    
    printf("[MMIO] Read vGPU properties:\n");
    printf("  Pool ID: %c\n", props->pool_id);
    printf("  Priority: %u (%s)\n", props->priority,
           props->priority == 2 ? "high" : (props->priority == 1 ? "medium" : "low"));
    printf("  VM ID: %u\n", props->vm_id);
    
    return 0;
}

/*
 * Send vector addition request via MMIO
 * Returns 0 on success, -1 on failure
 */
static int send_mmio_request(volatile void *mmio, VGPUProperties *props,
                            int num1, int num2, uint32_t request_id) {
    volatile uint8_t *req_buf = (volatile uint8_t *)((volatile char *)mmio + VGPU_REQ_BUFFER_OFFSET);
    VGPURequest *req = (VGPURequest *)req_buf;
    uint32_t *params = (uint32_t *)(req_buf + VGPU_REQUEST_HEADER_SIZE);
    uint32_t request_len;
    
    /* Clear request buffer */
    memset((void *)req_buf, 0, 1024);
    
    /* Build VGPURequest header */
    req->version = VGPU_PROTOCOL_VERSION;
    req->opcode = VGPU_OP_CUDA_KERNEL;
    req->flags = 0;
    req->param_count = 2;  // num1 and num2
    req->data_offset = VGPU_REQUEST_HEADER_SIZE + 2 * sizeof(uint32_t);
    req->data_length = 0;  // No variable data
    req->reserved[0] = 0;
    req->reserved[1] = 0;
    
    /* Write parameters */
    params[0] = (uint32_t)num1;
    params[1] = (uint32_t)num2;
    
    /* Calculate total request length */
    request_len = VGPU_REQUEST_HEADER_SIZE + 2 * sizeof(uint32_t);
    
    /* Set request ID */
    REG32(mmio, VGPU_REG_REQUEST_ID) = request_id;
    
    /* Set request length */
    REG32(mmio, VGPU_REG_REQUEST_LEN) = request_len;
    
    /* Ring doorbell */
    printf("[DOORBELL] Submitting request (len=%u, req_id=%u)...\n", request_len, request_id);
    REG32(mmio, VGPU_REG_DOORBELL) = 1;
    
    return 0;
}

/*
 * Wait for response via MMIO
 * Returns 0 on success, -1 on timeout/error
 */
static int wait_for_mmio_response(volatile void *mmio, int *result) {
    time_t start_time = time(NULL);
    time_t current_time;
    uint32_t status;
    uint32_t error_code;
    volatile uint8_t *resp_buf = (volatile uint8_t *)((volatile char *)mmio + VGPU_RESP_BUFFER_OFFSET);
    VGPUResponse *resp;
    uint32_t *results;
    
    printf("[WAIT] Polling for response...\n");
    printf("  Timeout: %d seconds\n", RESPONSE_TIMEOUT);
    
    while (1) {
        current_time = time(NULL);
        
        if (current_time - start_time >= RESPONSE_TIMEOUT) {
            fprintf(stderr, "[ERROR] Timeout waiting for response\n");
            return -1;
        }
        
        status = REG32(mmio, VGPU_REG_STATUS);
        
        if (status == VGPU_STATUS_DONE) {
            /* Response ready */
            resp = (VGPUResponse *)resp_buf;
            
            /* Validate response */
            if (resp->version != VGPU_PROTOCOL_VERSION) {
                fprintf(stderr, "[ERROR] Invalid response version: 0x%08x\n", resp->version);
                return -1;
            }
            
            if (resp->status != 0) {
                fprintf(stderr, "[ERROR] Request failed with status: %u\n", resp->status);
                return -1;
            }
            
            if (resp->result_count < 1) {
                fprintf(stderr, "[ERROR] No results in response\n");
                return -1;
            }
            
            /* Extract result */
            results = (uint32_t *)(resp_buf + VGPU_RESPONSE_HEADER_SIZE);
            *result = (int)results[0];
            
            printf("[RESPONSE] Received: %d\n", *result);
            if (resp->exec_time_us > 0) {
                printf("  Execution time: %u microseconds\n", resp->exec_time_us);
            }
            
            return 0;
        }
        
        if (status == VGPU_STATUS_ERROR) {
            error_code = REG32(mmio, VGPU_REG_ERROR_CODE);
            fprintf(stderr, "[ERROR] Device error: code=0x%02x\n", error_code);
            
            if (error_code == VGPU_ERR_MEDIATOR_UNAVAIL) {
                fprintf(stderr, "         Mediator daemon is not running or not accessible\n");
                fprintf(stderr, "         Start mediator: sudo ./mediator_phase3\n");
            } else if (error_code == VGPU_ERR_RATE_LIMITED) {
                fprintf(stderr, "         VM is rate-limited (too many requests/sec)\n");
                fprintf(stderr, "         This request can be retried after a short delay\n");
                return -2;  /* Retryable error */
            } else if (error_code == VGPU_ERR_QUEUE_FULL) {
                fprintf(stderr, "         Queue is full (back-pressure active)\n");
                fprintf(stderr, "         This request can be retried after a short delay\n");
                return -2;  /* Retryable error */
            } else if (error_code == VGPU_ERR_VM_QUARANTINED) {
                fprintf(stderr, "         VM is quarantined due to too many errors\n");
                fprintf(stderr, "         Contact admin: vgpu-admin clear-quarantine --vm-uuid=<uuid>\n");
                return -3;  /* Non-retryable error */
            } else if (error_code == VGPU_ERR_TIMEOUT) {
                fprintf(stderr, "         Request timed out on the host side\n");
            }
            
            return -1;
        }
        
        if (status == VGPU_STATUS_BUSY) {
            /* Still processing, wait a bit */
            usleep(POLL_INTERVAL);
            continue;
        }
        
        if (status == VGPU_STATUS_IDLE) {
            /* Should not happen after doorbell, but wait anyway */
            usleep(POLL_INTERVAL);
            continue;
        }
        
        /* Unknown status */
        fprintf(stderr, "[WARNING] Unknown status: %u\n", status);
        usleep(POLL_INTERVAL);
    }
    
    return -1;
}

/*
 * Main function
 */
int main(int argc, char *argv[]) {
    VGPUProperties props;
    int num1, num2;
    int result;
    int fd;
    volatile void *mmio;
    const char *pci_resource;
    uint32_t request_id;
    uint32_t protocol_ver;
    
    printf("================================================================================\n");
    printf("          VM CLIENT ENHANCED - MMIO Vector Addition Request\n");
    printf("================================================================================\n\n");
    
    /* Check arguments */
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
    
    /* Find vGPU device */
    pci_resource = find_vgpu_device();
    if (!pci_resource) {
        fprintf(stderr, "ERROR: Failed to find vGPU device\n");
        fprintf(stderr, "Possible reasons:\n");
        fprintf(stderr, "  1. Not running as root (use sudo)\n");
        fprintf(stderr, "  2. vgpu-stub device not attached to VM\n");
        fprintf(stderr, "  3. Check with: lspci | grep 'Processing accelerators'\n");
        return 1;
    }
    
    /* Open PCI resource */
    fd = open(pci_resource, O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("Failed to open vGPU device");
        fprintf(stderr, "Device path: %s\n", pci_resource);
        return 1;
    }
    
    /* Map MMIO region */
    mmio = mmap(NULL, VGPU_BAR_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmio == MAP_FAILED) {
        perror("Failed to mmap MMIO");
        close(fd);
        return 1;
    }
    
    /* Verify protocol version */
    protocol_ver = REG32(mmio, VGPU_REG_PROTOCOL_VER);
    if (protocol_ver != VGPU_PROTOCOL_VERSION) {
        fprintf(stderr, "[WARNING] Protocol version mismatch: device=0x%08x, expected=0x%08x\n",
                protocol_ver, VGPU_PROTOCOL_VERSION);
    }
    
    /* Read vGPU properties */
    if (read_vgpu_properties(mmio, &props) != 0) {
        fprintf(stderr, "ERROR: Failed to read vGPU properties\n");
        munmap((void *)mmio, VGPU_BAR_SIZE);
        close(fd);
        return 1;
    }
    
    printf("\n");
    
    /* Generate request ID */
    request_id = (uint32_t)time(NULL);
    
    /* Phase 3: Retry loop with exponential backoff for BUSY/RATE_LIMITED */
    int rc;
    int retries = 0;
    int backoff_ms = INITIAL_BACKOFF_MS;

    while (retries <= MAX_RETRIES) {
        /* Send request via MMIO */
        if (send_mmio_request(mmio, &props, num1, num2, request_id) != 0) {
            fprintf(stderr, "ERROR: Failed to send request\n");
            munmap((void *)mmio, VGPU_BAR_SIZE);
            close(fd);
            return 1;
        }

        printf("\n");

        /* Wait for response */
        rc = wait_for_mmio_response(mmio, &result);
        if (rc == 0) {
            break;  /* Success */
        } else if (rc == -2 && retries < MAX_RETRIES) {
            /* Retryable error (rate-limited or queue full) */
            retries++;
            printf("[RETRY] Attempt %d/%d — backing off %d ms...\n",
                   retries, MAX_RETRIES, backoff_ms);
            usleep(backoff_ms * 1000);
            backoff_ms = (backoff_ms * 2 > MAX_BACKOFF_MS)
                       ? MAX_BACKOFF_MS : backoff_ms * 2;
            request_id++;  /* Use a new request ID for the retry */
            continue;
        } else if (rc == -3) {
            /* Quarantined — no retry */
            fprintf(stderr, "ERROR: VM quarantined, cannot submit requests\n");
            munmap((void *)mmio, VGPU_BAR_SIZE);
            close(fd);
            return 2;
        } else {
            fprintf(stderr, "ERROR: Failed to receive response\n");
            munmap((void *)mmio, VGPU_BAR_SIZE);
            close(fd);
            return 1;
        }
    }

    if (rc != 0) {
        fprintf(stderr, "ERROR: Exhausted all %d retries\n", MAX_RETRIES);
        munmap((void *)mmio, VGPU_BAR_SIZE);
        close(fd);
        return 1;
    }
    
    printf("\n");
    printf("================================================================================\n");
    printf("                    RESULT\n");
    printf("================================================================================\n");
    printf("  %d + %d = %d\n", num1, num2, result);
    printf("================================================================================\n");
    
    /* Cleanup */
    munmap((void *)mmio, VGPU_BAR_SIZE);
    close(fd);
    
    return 0;
}

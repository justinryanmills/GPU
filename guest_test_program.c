#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

#define BAR0_SIZE 4096

#define REG_COMMAND   0x00
#define REG_STATUS    0x04
#define REG_POOL_ID   0x08
#define REG_PRIORITY  0x0C
#define REG_VM_ID     0x10

const char *priority_names[] = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "UNKNOWN"
};

uint64_t find_vgpu_bar0(void)
{
    FILE *fp;
    char line[256];
    char resource_path[256];
    uint64_t bar0_addr = 0;
    
    fp = popen("find /sys/bus/pci/devices -name 'resource' | xargs grep -l '0x' 2>/dev/null | head -1", "r");
    if (!fp) {
        return 0;
    }
    
    pclose(fp);
    
    fp = popen("lspci -v -d 1af4:1111 2>/dev/null | grep 'Memory at' | head -1", "r");
    if (!fp) {
        return 0;
    }
    
    if (fgets(line, sizeof(line), fp)) {
        char *ptr = strstr(line, "at ");
        if (ptr) {
            ptr += 3;
            bar0_addr = strtoull(ptr, NULL, 16);
        }
    }
    pclose(fp);
    
    return bar0_addr;
}

uint32_t read_register(volatile void *bar0_base, uint32_t offset)
{
    volatile uint32_t *reg = (volatile uint32_t *)((char *)bar0_base + offset);
    return *reg;
}

void write_register(volatile void *bar0_base, uint32_t offset, uint32_t value)
{
    volatile uint32_t *reg = (volatile uint32_t *)((char *)bar0_base + offset);
    *reg = value;
}
 
int main(int argc, char *argv[])
{
    int fd;
    volatile void *bar0_map;
    uint64_t bar0_addr;
    uint32_t pool_id_val, priority_val, vm_id_val;
    char pool_id_char;
    const char *priority_str;
    
    if (geteuid() != 0) {
        return 1;
    }
    
    bar0_addr = find_vgpu_bar0();
    
    if (bar0_addr == 0) {
        return 1;
    }
    
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("Error opening /dev/mem");
        return 1;
    }
    
    bar0_map = mmap(NULL, BAR0_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, 
                    fd, bar0_addr);
    if (bar0_map == MAP_FAILED) {
        perror("Error mapping BAR0");
        close(fd);
        return 1;
    }
    
    pool_id_val = read_register(bar0_map, REG_POOL_ID);
    priority_val = read_register(bar0_map, REG_PRIORITY);
    vm_id_val = read_register(bar0_map, REG_VM_ID);
    
    pool_id_char = (char)(pool_id_val & 0xFF);
    if (pool_id_char != 'A' && pool_id_char != 'B') {
        pool_id_char = '?';
    }
    
    if (priority_val <= 2) {
        priority_str = priority_names[priority_val];
    } else {
        priority_str = priority_names[3];
    }
    
    uint32_t old_pool = read_register(bar0_map, REG_POOL_ID);
    write_register(bar0_map, REG_POOL_ID, 0xDEADBEEF);
    uint32_t new_pool = read_register(bar0_map, REG_POOL_ID);
    
    uint32_t old_priority = read_register(bar0_map, REG_PRIORITY);
    write_register(bar0_map, REG_PRIORITY, 0x12345678);
    uint32_t new_priority = read_register(bar0_map, REG_PRIORITY);
    
    uint32_t old_vmid = read_register(bar0_map, REG_VM_ID);
    write_register(bar0_map, REG_VM_ID, 0xABCDEF00);
    uint32_t new_vmid = read_register(bar0_map, REG_VM_ID);
    
    uint32_t old_status = read_register(bar0_map, REG_STATUS);
    write_register(bar0_map, REG_COMMAND, 0x12345678);
    uint32_t new_status = read_register(bar0_map, REG_STATUS);
    uint32_t command = read_register(bar0_map, REG_COMMAND);
    
    munmap((void *)bar0_map, BAR0_SIZE);
    close(fd);
    
    return 0;
}
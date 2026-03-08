#define _GNU_SOURCE  /* Required for RTLD_DEFAULT */

/*
 * CUDA Transport — guest-side communication layer
 *
 * Finds the VGPU-STUB PCI device, maps BAR0, and provides a blocking
 * RPC interface for the CUDA shim library.
 *
 * Data path (preferred — VHOST-style shared memory):
 *   During init, a large anonymous region is mmap'd + mlock'd and its
 *   guest physical address (GPA) is registered with the vgpu-stub via
 *   new MMIO registers (VGPU_REG_SHMEM_*).  The vgpu-stub then maps
 *   that guest memory directly via cpu_physical_memory_map(), so data
 *   flows without going through the 8 MB BAR1 MMIO window:
 *
 *     guest shim                         vgpu-stub (QEMU)
 *       memcpy to shmem_g2h  ─────────▶  reads host ptr to same RAM
 *       write MMIO doorbell  ──MMIO──▶   (one VM exit, tiny payload)
 *       poll BAR0 STATUS     ◀──MMIO──   STATUS = DONE/ERROR
 *       memcpy from shmem_h2g ◀─────────  vgpu-stub wrote result there
 *
 * Data path (fallback — BAR1 MMIO, for guests that cannot mlock):
 *   The original 8 MB BAR1 window is used.  For transfers larger than
 *   CUDA_MAX_CHUNK_SIZE (4 MB) the existing chunked helpers are used.
 *
 * Control registers (BAR0) are always used for call metadata, status,
 * inline args, and small data regardless of which data path is active.
 *
 * Large transfers in shmem mode:
 *   When the data is larger than shmem_g2h_size, the existing chunked
 *   helpers are called with shmem_g2h_size as the chunk limit, giving
 *   far fewer round-trips than the 4 MB BAR1 chunks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <dlfcn.h>

#include "cuda_transport.h"
#include "cuda_protocol.h"

/* When VGPU_DEBUG is unset, skip per-call and verbose discovery logging to avoid inference delay. */
static int vgpu_debug_logging(void) {
    static int cached = -1;
    if (cached < 0) cached = (getenv("VGPU_DEBUG") != NULL) ? 1 : 0;
    return cached;
}

/* Forward declaration */
static void call_libvgpu_set_skip_interception(int skip);

/* ---- PCI scan constants ---------------------------------------- */
#define VGPU_VENDOR_ID   0x10DE
#define VGPU_DEVICE_ID   0x2331
#define VGPU_CLASS        0x030200
#define VGPU_CLASS_MASK   0xFFFF00

/* ---- BAR0 register offsets (must match vgpu_protocol.h) -------- */
#define REG_DOORBELL        0x000
#define REG_STATUS          0x004
#define REG_VM_ID           0x010
#define REG_ERROR_CODE      0x014
#define REG_REQUEST_LEN     0x018
#define REG_RESPONSE_LEN    0x01C

/* CUDA-specific registers */
#define REG_CUDA_OP         0x080
#define REG_CUDA_SEQ        0x084
#define REG_CUDA_NUM_ARGS   0x088
#define REG_CUDA_DATA_LEN   0x08C
#define REG_CUDA_DOORBELL   0x0A8
#define REG_CUDA_ARGS_BASE  0x0B0

/* BAR0 CUDA request/response data region */
#define CUDA_REQ_DATA_OFFSET   0x100
#define CUDA_RESP_DATA_OFFSET  0x500
#define CUDA_SMALL_DATA_MAX    1024

/* BAR0 CUDA result registers */
#define REG_CUDA_RESULT_STATUS   0x0F0
#define REG_CUDA_RESULT_NUM      0x0F4
#define REG_CUDA_RESULT_DATA_LEN 0x0F8
#define REG_CUDA_RESULT_BASE     0x900

/* Shared-memory registration registers (new, match vgpu_protocol.h 0x940-0x94C)
 * These are placed AFTER the CUDA result-value block to avoid overlap with
 * the CUDA control registers (0x080-0x0FF) and CUDA arg registers (0x0B0-0x0EF). */
#define REG_SHMEM_GPA_LO    0x940
#define REG_SHMEM_GPA_HI    0x944
#define REG_SHMEM_SIZE      0x948
#define REG_SHMEM_CTRL      0x94C

/* Capabilities register (BAR0 0x024) */
#define REG_CAPABILITIES    0x024
#define VGPU_CAP_SHMEM      (1u << 6)
#define VGPU_CAP_BAR1_DATA  (1u << 5)
#define REG_PROTOCOL_VER    0x020

/* BAR sizes */
#define BAR0_SIZE  4096
#define BAR1_SIZE  (16 * 1024 * 1024)

/* Status register values */
#define STATUS_IDLE   0x00
#define STATUS_BUSY   0x01
#define STATUS_DONE   0x02
#define STATUS_ERROR  0x03

/* Polling */
#define POLL_INTERVAL_US  100
#define POLL_TIMEOUT_SEC  60

/* BAR1 legacy regions */
#define BAR1_GUEST_TO_HOST_OFFSET  0x000000
#define BAR1_GUEST_TO_HOST_SIZE    (8 * 1024 * 1024)
#define BAR1_HOST_TO_GUEST_OFFSET  0x800000
#define BAR1_HOST_TO_GUEST_SIZE    (8 * 1024 * 1024)

/* Default shared-memory region (must match VGPU_SHMEM_DEFAULT_SIZE) */
#define SHMEM_DEFAULT_SIZE   (256u * 1024u * 1024u)
#define SHMEM_MIN_SIZE       (  8u * 1024u * 1024u)

/* Register access */
#define REG32(base, off)  (*(volatile uint32_t *)((volatile char *)(base) + (off)))
#define REG64(base, off)  (*(volatile uint64_t *)((volatile char *)(base) + (off)))

/* Module-level PCI BDF populated by find_vgpu_device().
 * Accessible even when the transport struct has not been allocated yet
 * (e.g. when the caller only needs the address for cuDeviceGetPCIBusId). */
static char g_discovered_bdf[256] = "";  /* Increased from 64 to 256 to match dirent->d_name max size */

/* ---------------------------------------------------------------- */
struct cuda_transport {
    volatile void *bar0;         /* BAR0 MMIO mapping                    */
    volatile void *bar1;         /* BAR1 MMIO mapping (legacy, may NULL) */
    int            bar0_fd;
    int            bar1_fd;
    uint32_t       vm_id;
    uint32_t       seq_counter;
    int            has_bar1;     /* 1 if BAR1 is mapped (legacy path)    */

    /* PCI bus/device/function identifier, e.g. "0000:00:05.0" */
    char           pci_bdf[64];

    /* VHOST-style shared memory */
    void          *shmem;        /* mmap base (full region)              */
    size_t         shmem_size;   /* total size (G2H + H2G)               */
    void          *shmem_g2h;   /* first half: guest → host data        */
    void          *shmem_h2g;   /* second half: host → guest data       */
    size_t         shmem_half;  /* shmem_size / 2                       */
    int            has_shmem;   /* 1 if shared-memory path is active    */
};

/* ================================================================
 * PCI device scanner
 * ================================================================ */
static int find_vgpu_device(char *res0_path, size_t res0_sz,
                            char *res1_path, size_t res1_sz,
                            char *bdf_out,   size_t bdf_sz)
{
    DIR *dir;
    struct dirent *entry;
    char pci_path[288];
    char attr_path[512];
    char line[256];
    FILE *fp;
    unsigned int vendor, device, cls;

    dir = opendir("/sys/bus/pci/devices");
    if (!dir) {
        fprintf(stderr, "[cuda-transport] Cannot open /sys/bus/pci/devices: %s\n",
                strerror(errno));
        return -1;
    }

    /* CRITICAL: Set skip flag at the VERY START of find_vgpu_device()
     * This ensures files are read with real values, not intercepted values
     * This is needed because find_vgpu_device() might be called directly
     * without going through cuda_transport_init() or cuda_transport_discover() */
    call_libvgpu_set_skip_interception(1);
    if (vgpu_debug_logging()) {
        write(2, "[cuda-transport] FORCE: find_vgpu_device() STARTED - setting skip flag\n", 72);
        write(2, "[cuda-transport] FORCE: Skip flag set to 1 in find_vgpu_device()\n", 65);
    }
    int device_count = 0;
    if (vgpu_debug_logging()) { fprintf(stderr, "[cuda-transport] DEBUG: Starting device scan...\n"); fflush(stderr); }
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        device_count++;
        if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Scanning device %d: %s\n", device_count, entry->d_name);

        snprintf(pci_path, sizeof(pci_path),
                 "/sys/bus/pci/devices/%s", entry->d_name);

        vendor = device = cls = 0;

        snprintf(attr_path, sizeof(attr_path), "%s/vendor", pci_path);
        fp = fopen(attr_path, "r");
        if (!fp) {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Cannot open vendor: %s\n", entry->d_name, strerror(errno));
            continue;
        }
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &vendor);
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Read vendor: 0x%04x (line: %s)", entry->d_name, vendor, line);
        } else {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Failed to read vendor line\n", entry->d_name);
        }
        fclose(fp);

        snprintf(attr_path, sizeof(attr_path), "%s/device", pci_path);
        fp = fopen(attr_path, "r");
        if (!fp) {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Cannot open device: %s\n", entry->d_name, strerror(errno));
            continue;
        }
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &device);
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Read device: 0x%04x (line: %s)", entry->d_name, device, line);
        } else {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Failed to read device line\n", entry->d_name);
        }
        fclose(fp);

        snprintf(attr_path, sizeof(attr_path), "%s/class", pci_path);
        fp = fopen(attr_path, "r");
        if (!fp) {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Cannot open class: %s\n", entry->d_name, strerror(errno));
            continue;
        }
        if (fgets(line, sizeof(line), fp)) {
            sscanf(line, "%x", &cls);
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Read class: 0x%06x (line: %s)", entry->d_name, cls, line);
        } else {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] Failed to read class line\n", entry->d_name);
        }
        fclose(fp);

        if (vgpu_debug_logging()) {
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] Final values: vendor=0x%04x device=0x%04x class=0x%06x\n",
                    entry->d_name, vendor, device, cls);
            fflush(stderr);
        }
        int class_ok = ((cls & VGPU_CLASS_MASK) == VGPU_CLASS);
        if (vgpu_debug_logging()) {
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] class_ok=%d (cls & mask=0x%06x, expected=0x%06x)\n",
                    entry->d_name, class_ok, (cls & VGPU_CLASS_MASK), VGPU_CLASS);
            fflush(stderr);
        }

        /*
         * Accept the device if:
         *   (a) Exact match: vendor=0x10DE device=0x2331 class=0x030200
         *       — QEMU built with our vgpu_protocol.h IDs.
         *   (b) Class match with a QEMU/Red-Hat generic vendor
         *       (0x1234 = Red Hat QEMU, 0x1AF4 = VirtIO/Red Hat)
         *       — Handles QEMU builds that use the legacy stub vendor IDs.
         *
         * In either case the PCI class must be 0x030200 (3D controller /
         * VGA-compatible GPU) to avoid accidentally matching non-GPU devices.
         */
        int exact  = class_ok && (vendor == VGPU_VENDOR_ID) &&
                                  (device == VGPU_DEVICE_ID);
        int legacy = class_ok && (vendor == 0x1234 || vendor == 0x1AF4);
        
        if (vgpu_debug_logging()) {
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] Matching: exact=%d (vendor_match=%d device_match=%d) legacy=%d\n",
                    entry->d_name, exact, (vendor == VGPU_VENDOR_ID), (device == VGPU_DEVICE_ID), legacy);
            fflush(stderr);
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] Expected: vendor=0x%04x device=0x%04x, got: vendor=0x%04x device=0x%04x\n",
                    entry->d_name, VGPU_VENDOR_ID, VGPU_DEVICE_ID, vendor, device);
            fflush(stderr);
        }
        if (!exact && !legacy) {
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: [%s] No match, continuing...\n", entry->d_name);
            continue;
        }
        if (vgpu_debug_logging()) {
            fprintf(stderr, "[cuda-transport] DEBUG: [%s] *** MATCH FOUND! exact=%d legacy=%d ***\n",
                    entry->d_name, exact, legacy);
            fprintf(stderr,
                    "[cuda-transport] Found VGPU-STUB at %s "
                    "(vendor=0x%04x device=0x%04x class=0x%06x match=%s)\n",
                    entry->d_name, vendor, device, cls,
                    exact ? "exact" : "legacy-qemu");
        }

        snprintf(res0_path, res0_sz, "%s/resource0", pci_path);
        snprintf(res1_path, res1_sz, "%s/resource1", pci_path);
        if (bdf_out && bdf_sz > 0) {
            /* Copy BDF, ensuring null termination */
            size_t copy_len = (bdf_sz - 1 < strlen(entry->d_name)) ? (bdf_sz - 1) : strlen(entry->d_name);
            memcpy(bdf_out, entry->d_name, copy_len);
            bdf_out[copy_len] = '\0';
        }
        /* Always record the BDF globally so callers that don't need the
         * full transport (e.g. the lightweight discover path) can still
         * return a correct PCI address via cuda_transport_pci_bdf(NULL). */
        strncpy(g_discovered_bdf, entry->d_name, sizeof(g_discovered_bdf) - 1);
        g_discovered_bdf[sizeof(g_discovered_bdf) - 1] = '\0';
        closedir(dir);
        /* Re-enable interception after successful discovery */
        call_libvgpu_set_skip_interception(0);
        return 0;
    }

    fprintf(stderr,
            "[cuda-transport] VGPU-STUB not found in /sys/bus/pci/devices "
            "(scanned %d devices, want vendor=0x%04x device=0x%04x OR QEMU-vendor with "
            "class=0x%06x)\n",
            device_count, VGPU_VENDOR_ID, VGPU_DEVICE_ID, VGPU_CLASS);
    closedir(dir);
    /* Re-enable interception after discovery */
    call_libvgpu_set_skip_interception(0);
    return -1;
}

/* ================================================================
 * Guest Physical Address resolution via /proc/self/pagemap
 *
 * Returns the physical address of the virtual page containing vaddr,
 * or 0 on failure.  The page must already be faulted in (e.g. by
 * mlock or a dummy write).
 * ================================================================ */
static uint64_t virt_to_phys(const void *vaddr)
{
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;

    uint64_t page_size = (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t vfn       = (uintptr_t)vaddr / page_size;
    uint64_t pme       = 0;

    if (pread(fd, &pme, sizeof(pme), (off_t)(vfn * sizeof(pme))) != (ssize_t)sizeof(pme)) {
        close(fd);
        return 0;
    }
    close(fd);

    /* Bit 63 = present; bits 54:0 = PFN */
    if (!(pme & (1ULL << 63))) return 0;  /* page not present */
    uint64_t pfn = pme & 0x007FFFFFFFFFFFFFULL;
    return pfn * page_size + ((uintptr_t)vaddr & (page_size - 1));
}

/* ================================================================
 * Try to allocate and register the shared-memory region.
 * Returns 1 on success, 0 if the feature should fall back to BAR1.
 * ================================================================ */
static int setup_shmem(cuda_transport_t *t)
{
    uint32_t caps = REG32(t->bar0, REG_CAPABILITIES);
    if (!(caps & VGPU_CAP_SHMEM)) {
        fprintf(stderr, "[cuda-transport] vgpu-stub does not support "
                "shared-memory data path (caps=0x%x), using BAR1\n", caps);
        return 0;
    }

    /* Try SHMEM_DEFAULT_SIZE (256 MB) first; fall back to SHMEM_MIN_SIZE */
    size_t try_sizes[] = { SHMEM_DEFAULT_SIZE, SHMEM_MIN_SIZE, 0 };
    void *shmem = MAP_FAILED;
    size_t shmem_size = 0;

    for (int i = 0; try_sizes[i] != 0; i++) {
        shmem = mmap(NULL, try_sizes[i],
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED,
                     -1, 0);
        if (shmem != MAP_FAILED) {
            shmem_size = try_sizes[i];
            break;
        }
        fprintf(stderr, "[cuda-transport] mmap shmem %zu MB failed: %s\n",
                try_sizes[i] >> 20, strerror(errno));
    }

    if (shmem == MAP_FAILED || shmem_size == 0) {
        fprintf(stderr, "[cuda-transport] Cannot allocate shared memory "
                "— falling back to BAR1\n");
        return 0;
    }

    /* Touch every page to ensure they are faulted in before mlock */
    memset(shmem, 0, shmem_size);

    /* Lock in RAM so pages cannot be swapped out (GPA must remain valid) */
    if (mlock(shmem, shmem_size) != 0) {
        fprintf(stderr, "[cuda-transport] mlock(%zu MB) failed: %s "
                "— trying smaller region\n",
                shmem_size >> 20, strerror(errno));
        /* If mlock failed for 256 MB, the kernel may allow 8 MB */
        if (shmem_size > SHMEM_MIN_SIZE) {
            munmap(shmem, shmem_size);
            shmem_size = SHMEM_MIN_SIZE;
            shmem = mmap(NULL, shmem_size,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED,
                         -1, 0);
            if (shmem == MAP_FAILED) {
                fprintf(stderr, "[cuda-transport] Fallback mmap failed: %s"
                        " — using BAR1\n", strerror(errno));
                return 0;
            }
            memset(shmem, 0, shmem_size);
            if (mlock(shmem, shmem_size) != 0) {
                fprintf(stderr, "[cuda-transport] Fallback mlock failed: %s"
                        " — using BAR1\n", strerror(errno));
                munmap(shmem, shmem_size);
                return 0;
            }
        } else {
            munmap(shmem, shmem_size);
            return 0;
        }
    }

    /* Get the Guest Physical Address of the base page */
    uint64_t gpa = virt_to_phys(shmem);
    if (gpa == 0) {
        fprintf(stderr, "[cuda-transport] Cannot resolve GPA for shmem "
                "(need CAP_SYS_ADMIN or /proc/self/pagemap access) "
                "— using BAR1\n");
        munlock(shmem, shmem_size);
        munmap(shmem, shmem_size);
        return 0;
    }

    /* Register with the vgpu-stub */
    REG32(t->bar0, REG_SHMEM_GPA_LO) = (uint32_t)(gpa & 0xFFFFFFFFu);
    REG32(t->bar0, REG_SHMEM_GPA_HI) = (uint32_t)(gpa >> 32);
    REG32(t->bar0, REG_SHMEM_SIZE)   = (uint32_t)shmem_size;
    REG32(t->bar0, REG_SHMEM_CTRL)   = 1;  /* register */

    /* Poll for acknowledgement (max 5 s) */
    time_t start = time(NULL);
    uint32_t st;
    while (1) {
        st = REG32(t->bar0, REG_STATUS);
        if (st == STATUS_DONE || st == STATUS_ERROR) break;
        if (time(NULL) - start >= 5) { st = 0xFF; break; }
        usleep(1000);
    }

    if (st != STATUS_DONE) {
        fprintf(stderr, "[cuda-transport] vgpu-stub rejected shmem registration "
                "(status=0x%02x) — using BAR1\n", st);
        /* Unregister */
        REG32(t->bar0, REG_SHMEM_CTRL) = 0;
        munlock(shmem, shmem_size);
        munmap(shmem, shmem_size);
        return 0;
    }

    t->shmem      = shmem;
    t->shmem_size = shmem_size;
    t->shmem_half = shmem_size / 2;
    t->shmem_g2h  = shmem;
    t->shmem_h2g  = (char *)shmem + shmem_size / 2;
    t->has_shmem  = 1;

    if (vgpu_debug_logging())
        fprintf(stderr, "[cuda-transport] Shared-memory registered: "
                "gpa=0x%016llx size=%zu MB (G2H=%zu MB H2G=%zu MB)\n",
                (unsigned long long)gpa,
                shmem_size >> 20, (shmem_size / 2) >> 20, (shmem_size / 2) >> 20);
    return 1;
}

/* ================================================================
 * Initialise transport
 * ================================================================ */
int cuda_transport_init(cuda_transport_t **tp)
{
    if (vgpu_debug_logging())
        write(2, "[cuda-transport] FORCE: cuda_transport_init() STARTED\n", 54);
    char res0_path[512], res1_path[512];
    char pci_bdf[64] = {0};
    cuda_transport_t *t;

    /* CRITICAL: Disable PCI file interception BEFORE calling find_vgpu_device()
     * This ensures we read real values from /sys, not intercepted values
     * Same fix as in cuda_transport_discover() */
    call_libvgpu_set_skip_interception(1);
    if (vgpu_debug_logging()) {
        write(2, "[cuda-transport] FORCE: About to set skip flag to 1\n", 52);
        write(2, "[cuda-transport] FORCE: Skip flag SET to 1 (pid=", 48);
        char pid_str[32];
        snprintf(pid_str, sizeof(pid_str), "%d)\n", (int)getpid());
        write(2, pid_str, strlen(pid_str));
        fprintf(stderr, "[cuda-transport] DEBUG: cuda_transport_init() - Skip flag SET to 1 (pid=%d)\n", (int)getpid());
        fflush(stderr);
    }

    if (find_vgpu_device(res0_path, sizeof(res0_path),
                         res1_path, sizeof(res1_path),
                         pci_bdf,   sizeof(pci_bdf)) != 0) {
        fprintf(stderr, "[cuda-transport] VGPU-STUB device not found\n");
        /* Re-enable interception after discovery */
        call_libvgpu_set_skip_interception(0);
        return -1;
    }

    t = (cuda_transport_t *)calloc(1, sizeof(cuda_transport_t));
    if (!t) return -1;

    t->bar0_fd  = -1;
    t->bar1_fd  = -1;
    t->has_bar1 = 0;
    t->has_shmem = 0;
    t->seq_counter = 1;
    /* Use snprintf to avoid truncation warning */
    snprintf(t->pci_bdf, sizeof(t->pci_bdf), "%.*s", (int)(sizeof(t->pci_bdf) - 1), pci_bdf);

    /* Map BAR0 (always required) */
    t->bar0_fd = open(res0_path, O_RDWR | O_SYNC);
    if (t->bar0_fd < 0) {
        fprintf(stderr, "[cuda-transport] Cannot open BAR0: %s (%s)\n",
                res0_path, strerror(errno));
        free(t);
        return -1;
    }

    t->bar0 = mmap(NULL, BAR0_SIZE, PROT_READ | PROT_WRITE,
                    MAP_SHARED, t->bar0_fd, 0);
    if (t->bar0 == MAP_FAILED) {
        fprintf(stderr, "[cuda-transport] Cannot mmap BAR0: %s\n",
                strerror(errno));
        close(t->bar0_fd);
        free(t);
        return -1;
    }

    t->vm_id = REG32(t->bar0, REG_VM_ID);

    /* --- Preferred path: VHOST-style shared memory --- */
    if (!setup_shmem(t)) {
        /* --- Fallback: map BAR1 (legacy 8 MB window) --- */
        t->bar1_fd = open(res1_path, O_RDWR | O_SYNC);
        if (t->bar1_fd >= 0) {
            t->bar1 = mmap(NULL, BAR1_SIZE, PROT_READ | PROT_WRITE,
                            MAP_SHARED, t->bar1_fd, 0);
            if (t->bar1 != MAP_FAILED) {
                t->has_bar1 = 1;
                if (vgpu_debug_logging())
                    fprintf(stderr, "[cuda-transport] BAR1 mapped "
                            "(16 MB legacy data region)\n");
            } else {
                t->bar1 = NULL;
                close(t->bar1_fd);
                t->bar1_fd = -1;
            }
        }
    }

    /* Re-enable interception after successful discovery */
    call_libvgpu_set_skip_interception(0);
    if (vgpu_debug_logging())
        fprintf(stderr, "[cuda-transport] Connected to VGPU-STUB "
                "(vm_id=%u, data_path=%s)\n",
                t->vm_id,
                t->has_shmem ? "shmem" : (t->has_bar1 ? "BAR1" : "BAR0-inline"));
    *tp = t;
    return 0;
}

/* ================================================================
 * Destroy transport
 * ================================================================ */
void cuda_transport_destroy(cuda_transport_t *tp)
{
    if (!tp) return;

    /* Release shared memory */
    if (tp->has_shmem && tp->shmem) {
        REG32(tp->bar0, REG_SHMEM_CTRL) = 0;  /* unregister */
        munlock(tp->shmem, tp->shmem_size);
        munmap(tp->shmem, tp->shmem_size);
        tp->shmem    = NULL;
        tp->has_shmem = 0;
    }

    if (tp->bar0 && tp->bar0 != MAP_FAILED)
        munmap((void *)tp->bar0, BAR0_SIZE);
    if (tp->bar1 && tp->bar1 != MAP_FAILED)
        munmap((void *)tp->bar1, BAR1_SIZE);
    if (tp->bar0_fd >= 0) close(tp->bar0_fd);
    if (tp->bar1_fd >= 0) close(tp->bar1_fd);
    free(tp);
}

/* ================================================================
 * Write bulk data to the active data region.
 *
 * Shared-memory path: plain memcpy into the G2H half — no VM exit.
 * BAR1 fallback:      memcpy into MMIO window (generates VM exits).
 * BAR0 inline:        for small data when neither shmem nor BAR1 exists.
 *
 * len MUST be <= the active window size (caller's responsibility).
 * ================================================================ */
static void write_bulk_data(cuda_transport_t *tp,
                            const void *data, uint32_t len)
{
    if (len == 0 || !data) return;

    if (tp->has_shmem) {
        /* Zero-copy into guest-pinned shared memory */
        memcpy(tp->shmem_g2h, data, len);
    } else if (tp->has_bar1 && len > CUDA_SMALL_DATA_MAX) {
        volatile uint8_t *dst = (volatile uint8_t *)tp->bar1
                                + BAR1_GUEST_TO_HOST_OFFSET;
        memcpy((void *)dst, data, len);
    } else {
        volatile uint8_t *dst = (volatile uint8_t *)tp->bar0
                                + CUDA_REQ_DATA_OFFSET;
        uint32_t to_copy = (len > CUDA_SMALL_DATA_MAX) ? CUDA_SMALL_DATA_MAX : len;
        memcpy((void *)dst, data, to_copy);
    }
}

/* ================================================================
 * Read bulk response data from the active data region.
 * ================================================================ */
static void read_bulk_data(cuda_transport_t *tp,
                           void *buf, uint32_t len)
{
    if (len == 0 || !buf) return;

    if (tp->has_shmem) {
        memcpy(buf, tp->shmem_h2g, len);
    } else if (tp->has_bar1 && len > CUDA_SMALL_DATA_MAX) {
        volatile uint8_t *src = (volatile uint8_t *)tp->bar1
                                + BAR1_HOST_TO_GUEST_OFFSET;
        memcpy(buf, (void *)src, len);
    } else {
        volatile uint8_t *src = (volatile uint8_t *)tp->bar0
                                + CUDA_RESP_DATA_OFFSET;
        uint32_t to_copy = (len > CUDA_SMALL_DATA_MAX) ? CUDA_SMALL_DATA_MAX : len;
        memcpy(buf, (void *)src, to_copy);
    }
}

/* ================================================================
 * Maximum payload per single round-trip.
 * In shmem mode this is the half-window size (128 MB by default).
 * In BAR1 mode this is 8 MB.
 * In inline mode this is 1 KB.
 * ================================================================ */
static uint32_t max_single_payload(cuda_transport_t *tp)
{
    if (tp->has_shmem) return (uint32_t)tp->shmem_half;
    if (tp->has_bar1)  return BAR1_GUEST_TO_HOST_SIZE;
    return CUDA_SMALL_DATA_MAX;
}

/* ================================================================
 * Core MMIO round-trip: write registers → ring doorbell → poll → read.
 *
 * send_len MUST fit in one pass (≤ max_single_payload()).
 * ================================================================ */
static int do_single_cuda_call(cuda_transport_t *tp,
                               uint32_t call_id,
                               const uint32_t *args, uint32_t num_args,
                               const void *send_data, uint32_t send_len,
                               CUDACallResult *result,
                               void *recv_data, uint32_t recv_cap,
                               uint32_t *recv_len)
{
    uint32_t seq = tp->seq_counter++;
    time_t start;
    uint32_t status;

    if (vgpu_debug_logging()) {
        fprintf(stderr, "[cuda-transport] SENDING to VGPU-STUB: call_id=0x%04x seq=%u args=%u data_len=%u (pid=%d)\n",
                call_id, seq, num_args, send_len, (int)getpid());
        fflush(stderr);
    }
    /* Write bulk data before writing metadata registers */
    write_bulk_data(tp, send_data, send_len);

    /* Write call metadata */
    REG32(tp->bar0, REG_CUDA_OP)       = call_id;
    REG32(tp->bar0, REG_CUDA_SEQ)      = seq;
    REG32(tp->bar0, REG_CUDA_NUM_ARGS) = (num_args > CUDA_MAX_INLINE_ARGS)
                                          ? CUDA_MAX_INLINE_ARGS : num_args;
    REG32(tp->bar0, REG_CUDA_DATA_LEN) = send_len;

    uint32_t n = (num_args > CUDA_MAX_INLINE_ARGS) ? CUDA_MAX_INLINE_ARGS : num_args;
    for (uint32_t i = 0; i < n; i++)
        REG32(tp->bar0, REG_CUDA_ARGS_BASE + i * 4) = args[i];

    if (vgpu_debug_logging()) {
        fprintf(stderr, "[cuda-transport] RINGING DOORBELL: MMIO write to VGPU-STUB (call_id=0x%04x, pid=%d)\n",
                call_id, (int)getpid());
        fflush(stderr);
    }
    REG32(tp->bar0, REG_CUDA_DOORBELL) = 1;

    /* Poll for completion */
    start = time(NULL);
    while (1) {
        status = REG32(tp->bar0, REG_STATUS);
        if (status == STATUS_DONE || status == STATUS_ERROR) break;
        if (time(NULL) - start >= POLL_TIMEOUT_SEC) {
            fprintf(stderr, "[cuda-transport] Timeout on call 0x%04x (seq=%u)\n",
                    call_id, seq);
            if (result) { memset(result, 0, sizeof(*result)); result->status = 2; }
            if (recv_len) *recv_len = 0;
            return 2;
        }
        usleep(POLL_INTERVAL_US);
    }

    if (vgpu_debug_logging()) {
        if (status == STATUS_DONE)
            fprintf(stderr, "[cuda-transport] RECEIVED from VGPU-STUB: call_id=0x%04x seq=%u status=DONE (pid=%d)\n",
                    call_id, seq, (int)getpid());
        else
            fprintf(stderr, "[cuda-transport] RECEIVED from VGPU-STUB: call_id=0x%04x seq=%u status=ERROR (pid=%d)\n",
                    call_id, seq, (int)getpid());
        fflush(stderr);
    }

    /* Read result registers */
    if (result) {
        result->magic       = 0x56475055;
        result->seq_num     = seq;
        result->status      = REG32(tp->bar0, REG_CUDA_RESULT_STATUS);
        result->num_results = REG32(tp->bar0, REG_CUDA_RESULT_NUM);
        result->data_len    = REG32(tp->bar0, REG_CUDA_RESULT_DATA_LEN);
        result->reserved    = 0;
        uint32_t nr = result->num_results;
        if (nr > CUDA_MAX_INLINE_RESULTS) nr = CUDA_MAX_INLINE_RESULTS;
        for (uint32_t i = 0; i < nr; i++)
            result->results[i] = REG64(tp->bar0, REG_CUDA_RESULT_BASE + i * 8);
    }

    /* Read bulk response */
    uint32_t resp_len = REG32(tp->bar0, REG_CUDA_RESULT_DATA_LEN);
    if (recv_data && resp_len > 0) {
        uint32_t copy_len = (resp_len > recv_cap) ? recv_cap : resp_len;
        read_bulk_data(tp, recv_data, copy_len);
        if (recv_len) *recv_len = copy_len;
    } else {
        if (recv_len) *recv_len = 0;
    }

    return result ? (int)result->status : 0;
}

/* ================================================================
 * Chunked host-to-device copy (cuMemcpyHtoD / cuMemcpyHtoDAsync)
 *
 * Sends the data in max_single_payload()-sized pieces, each with an
 * adjusted device pointer (dst + offset).
 * ================================================================ */
static int cuda_transport_call_htod_chunked(cuda_transport_t *tp,
                                            uint32_t call_id,
                                            const uint32_t *args,
                                            uint32_t num_args,
                                            const void *send_data,
                                            uint32_t send_len,
                                            CUDACallResult *result)
{
    uint64_t base_dst = CUDA_UNPACK_U64(args, 0);
    uint32_t chunk_args[CUDA_MAX_INLINE_ARGS];
    memcpy(chunk_args, args, num_args * sizeof(uint32_t));

    uint32_t limit  = max_single_payload(tp);
    uint32_t offset = 0;
    int rc = 0;
    CUDACallResult chunk_result;

    while (offset < send_len) {
        uint32_t chunk = send_len - offset;
        if (chunk > limit) chunk = limit;

        CUDA_PACK_U64(chunk_args, 0, base_dst + offset);
        CUDA_PACK_U64(chunk_args, 2, (uint64_t)chunk);

        if (send_len <= limit) {
            chunk_args[14] = CUDA_CHUNK_FLAG_SINGLE;
        } else if (offset == 0) {
            chunk_args[14] = CUDA_CHUNK_FLAG_FIRST;
        } else if (offset + chunk >= send_len) {
            chunk_args[14] = CUDA_CHUNK_FLAG_LAST;
        } else {
            chunk_args[14] = 0;
        }

        memset(&chunk_result, 0, sizeof(chunk_result));
        rc = do_single_cuda_call(tp, call_id,
                                 chunk_args, num_args,
                                 (const char *)send_data + offset, chunk,
                                 &chunk_result, NULL, 0, NULL);
        if (rc != 0) {
            fprintf(stderr, "[cuda-transport] HTOD chunk failed "
                    "at offset=%u chunk=%u total=%u rc=%d\n",
                    offset, chunk, send_len, rc);
            if (result) *result = chunk_result;
            return rc;
        }
        offset += chunk;
    }
    if (result) *result = chunk_result;
    return rc;
}

/* ================================================================
 * Chunked device-to-host copy (cuMemcpyDtoH / cuMemcpyDtoHAsync)
 * ================================================================ */
static int cuda_transport_call_dtoh_chunked(cuda_transport_t *tp,
                                            uint32_t call_id,
                                            const uint32_t *args,
                                            uint32_t num_args,
                                            void *recv_data,
                                            uint32_t total_recv,
                                            uint32_t *recv_len_out,
                                            CUDACallResult *result)
{
    uint64_t base_src = CUDA_UNPACK_U64(args, 0);
    uint32_t chunk_args[CUDA_MAX_INLINE_ARGS];
    memcpy(chunk_args, args, num_args * sizeof(uint32_t));

    uint32_t limit  = max_single_payload(tp);
    uint32_t offset = 0;
    int rc = 0;
    CUDACallResult chunk_result;
    uint32_t total_received = 0;

    while (offset < total_recv) {
        uint32_t chunk = total_recv - offset;
        if (chunk > limit) chunk = limit;

        CUDA_PACK_U64(chunk_args, 0, base_src + offset);
        CUDA_PACK_U64(chunk_args, 2, (uint64_t)chunk);

        if (total_recv <= limit) {
            chunk_args[14] = CUDA_CHUNK_FLAG_SINGLE;
        } else if (offset == 0) {
            chunk_args[14] = CUDA_CHUNK_FLAG_FIRST;
        } else if (offset + chunk >= total_recv) {
            chunk_args[14] = CUDA_CHUNK_FLAG_LAST;
        } else {
            chunk_args[14] = 0;
        }

        uint32_t chunk_recv = 0;
        memset(&chunk_result, 0, sizeof(chunk_result));
        rc = do_single_cuda_call(tp, call_id,
                                 chunk_args, num_args,
                                 NULL, 0,
                                 &chunk_result,
                                 (char *)recv_data + offset, chunk,
                                 &chunk_recv);
        if (rc != 0) {
            fprintf(stderr, "[cuda-transport] DTOH chunk failed "
                    "at offset=%u chunk=%u total=%u rc=%d\n",
                    offset, chunk, total_recv, rc);
            if (result) *result = chunk_result;
            if (recv_len_out) *recv_len_out = total_received;
            return rc;
        }
        total_received += chunk_recv;
        offset += chunk;
    }
    if (result) *result = chunk_result;
    if (recv_len_out) *recv_len_out = total_received;
    return rc;
}

/* ================================================================
 * Chunked module image upload (cuModuleLoadData / cuModuleLoadFatBinary)
 *
 * Carries CUDA_CHUNK_FLAG_* in args[14]; the mediator accumulates
 * on the host side and calls cuModuleLoadData on the LAST chunk.
 * ================================================================ */
static int cuda_transport_call_module_load_chunked(
    cuda_transport_t *tp,
    uint32_t call_id,
    const void *send_data, uint32_t send_len,
    CUDACallResult *result)
{
    uint32_t chunk_args[CUDA_MAX_INLINE_ARGS];
    uint32_t limit  = max_single_payload(tp);
    uint32_t offset = 0;
    int rc = 0;
    CUDACallResult chunk_result;

    while (offset < send_len) {
        uint32_t chunk = send_len - offset;
        if (chunk > limit) chunk = limit;

        memset(chunk_args, 0, sizeof(chunk_args));
        if (send_len <= limit) {
            chunk_args[14] = CUDA_CHUNK_FLAG_SINGLE;
        } else if (offset == 0) {
            chunk_args[14] = CUDA_CHUNK_FLAG_FIRST;
        } else if (offset + chunk >= send_len) {
            chunk_args[14] = CUDA_CHUNK_FLAG_LAST;
        } else {
            chunk_args[14] = 0;
        }

        memset(&chunk_result, 0, sizeof(chunk_result));
        rc = do_single_cuda_call(tp, call_id,
                                 chunk_args, CUDA_MAX_INLINE_ARGS,
                                 (const char *)send_data + offset, chunk,
                                 &chunk_result, NULL, 0, NULL);
        if (rc != 0) {
            fprintf(stderr, "[cuda-transport] MODULE_LOAD chunk failed "
                    "at offset=%u chunk=%u total=%u rc=%d\n",
                    offset, chunk, send_len, rc);
            if (result) *result = chunk_result;
            return rc;
        }
        offset += chunk;
    }
    if (result) *result = chunk_result;
    return rc;
}

/* ================================================================
 * Public API: execute one CUDA call (blocking)
 *
 * Dispatches to the appropriate chunked helper for large transfers,
 * or falls through to do_single_cuda_call for everything else.
 * ================================================================ */
int cuda_transport_call(cuda_transport_t *tp,
                        uint32_t call_id,
                        const uint32_t *args, uint32_t num_args,
                        const void *send_data, uint32_t send_len,
                        CUDACallResult *result,
                        void *recv_data, uint32_t recv_cap,
                        uint32_t *recv_len)
{
    if (vgpu_debug_logging()) {
        char inv_msg[256];
        int inv_len = snprintf(inv_msg, sizeof(inv_msg),
                              "[cuda-transport] cuda_transport_call() INVOKED: call_id=0x%04x data_len=%u tp=%p bar0=%p (pid=%d)\n",
                              call_id, send_len, (void*)tp, tp ? (void*)tp->bar0 : NULL, (int)getpid());
        if (inv_len > 0 && inv_len < (int)sizeof(inv_msg))
            syscall(__NR_write, 2, inv_msg, inv_len);
    }
    if (!tp || !tp->bar0) {
        if (vgpu_debug_logging()) {
            char err_msg[128];
            int err_len = snprintf(err_msg, sizeof(err_msg),
                                  "[cuda-transport] ERROR: tp=%p bar0=%p (pid=%d)\n",
                                  (void*)tp, tp ? (void*)tp->bar0 : NULL, (int)getpid());
            if (err_len > 0 && err_len < (int)sizeof(err_msg))
                syscall(__NR_write, 2, err_msg, err_len);
        }
        return 1;
    }

    uint32_t limit = max_single_payload(tp);

    /* ---- Chunked host-to-device copy ---- */
    if ((call_id == CUDA_CALL_MEMCPY_HTOD ||
         call_id == CUDA_CALL_MEMCPY_HTOD_ASYNC) &&
        send_data && send_len > limit)
    {
        return cuda_transport_call_htod_chunked(tp, call_id,
                                                args, num_args,
                                                send_data, send_len,
                                                result);
    }

    /* ---- Chunked device-to-host copy ---- */
    if ((call_id == CUDA_CALL_MEMCPY_DTOH ||
         call_id == CUDA_CALL_MEMCPY_DTOH_ASYNC) &&
        recv_data && recv_cap > limit)
    {
        return cuda_transport_call_dtoh_chunked(tp, call_id,
                                                args, num_args,
                                                recv_data, recv_cap,
                                                recv_len, result);
    }

    /* ---- Chunked module image upload ---- */
    if ((call_id == CUDA_CALL_MODULE_LOAD_DATA    ||
         call_id == CUDA_CALL_MODULE_LOAD_DATA_EX ||
         call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY) &&
        send_data && send_len > limit)
    {
        return cuda_transport_call_module_load_chunked(tp, call_id,
                                                       send_data, send_len,
                                                       result);
    }

    /* ---- Normal single-call path ---- */
    return do_single_cuda_call(tp, call_id,
                               args, num_args,
                               send_data, send_len,
                               result,
                               recv_data, recv_cap,
                               recv_len);
}

/* ================================================================
 * Accessors
 * ================================================================ */
uint32_t cuda_transport_vm_id(cuda_transport_t *tp)
{
    return tp ? tp->vm_id : 0;
}

int cuda_transport_is_connected(cuda_transport_t *tp)
{
    if (!tp || !tp->bar0) return 0;
    uint32_t ver = REG32(tp->bar0, REG_PROTOCOL_VER);
    return (ver == 0x00010000) ? 1 : 0;
}

/*
 * cuda_transport_discover — lightweight device scan.
 *
 * Scans /sys/bus/pci/devices for the VGPU-STUB PCI device and records its
 * BDF in g_discovered_bdf.  Does NOT open resource0 or map any BARs, so it
 * succeeds even inside a systemd sandbox where /sys is read-only or resource0
 * is not yet accessible.
 *
 * Returns 0 if the device was found, -1 otherwise.
 * Side-effect: sets g_discovered_bdf so cuda_transport_pci_bdf(NULL) works.
 * 
 * This function is idempotent: if g_discovered_bdf is already set and the
 * device still exists, it returns success without re-scanning.
 */
/* Forward declaration for skip interception function */
/* Use runtime resolution via dlsym to avoid linking dependency */
static void (*libvgpu_set_skip_interception_ptr)(int skip) = NULL;

/* Forward declaration - the actual function is defined in libvgpu-cuda.so or libvgpu-nvml.so
 * We use a wrapper to avoid static/non-static conflicts */

/* Wrapper that uses dlsym to find the real implementation */
static void call_libvgpu_set_skip_interception(int skip)
{
    if (!libvgpu_set_skip_interception_ptr) {
        /* Resolve symbol at runtime - it's in libvgpu-cuda.so or libvgpu-nvml.so */
        libvgpu_set_skip_interception_ptr = (void (*)(int))dlsym(RTLD_DEFAULT, "libvgpu_set_skip_interception");
        if (!libvgpu_set_skip_interception_ptr) {
            /* Symbol not found - this is OK if CUDA shim isn't loaded */
            return;
        }
    }
    libvgpu_set_skip_interception_ptr(skip);
}

int cuda_transport_discover(void)
{
    /* CRITICAL: Disable PCI file interception FIRST, before ANY operations
     * This ensures we read real values from /sys, not intercepted values
     * Based on documentation: when working, cuda_transport.c read real values directly */
    call_libvgpu_set_skip_interception(1);
    if (vgpu_debug_logging()) {
        fprintf(stderr, "[cuda-transport] DEBUG: Skip flag SET to 1 (pid=%d)\n", (int)getpid());
        fprintf(stderr, "[cuda-transport] DEBUG: cuda_transport_discover() called, g_discovered_bdf='%s'\n", g_discovered_bdf);
        fflush(stderr);
    }
    /* CRITICAL: Clear g_discovered_bdf to ensure fresh scan every time
     * This prevents issues with stale values from previous calls */
    g_discovered_bdf[0] = '\0';
    /* Fast path: if we already discovered a device, verify it still exists */
    if (g_discovered_bdf[0] != '\0') {
        if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Fast path: g_discovered_bdf='%s', verifying...\n", g_discovered_bdf);
        char verify_path[512];
        snprintf(verify_path, sizeof(verify_path),
                 "/sys/bus/pci/devices/%s/vendor", g_discovered_bdf);
        FILE *fp = fopen(verify_path, "r");
        if (fp) {
            fclose(fp);
            if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Fast path: Device '%s' verified, returning success (SKIPPING SCAN!)\n", g_discovered_bdf);
            return 0;
        }
        if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Fast path: Device '%s' not found, clearing cache\n", g_discovered_bdf);
        g_discovered_bdf[0] = '\0';
    }
    if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Slow path: Starting device scan...\n");
    char res0[512], res1[512], bdf[64];
    int rc = find_vgpu_device(res0, sizeof(res0), res1, sizeof(res1),
                              bdf,  sizeof(bdf));
    if (vgpu_debug_logging()) fprintf(stderr, "[cuda-transport] DEBUG: Slow path: find_vgpu_device() returned %d, g_discovered_bdf='%s'\n", rc, g_discovered_bdf);
    
    /* Re-enable interception after discovery */
    call_libvgpu_set_skip_interception(0);
    
    return rc;
}

const char *cuda_transport_pci_bdf(cuda_transport_t *tp)
{
    if (tp && tp->pci_bdf[0])
        return tp->pci_bdf;
    /* Fall back to the module-level BDF stored by find_vgpu_device()
     * during cuda_transport_discover() or cuda_transport_init(). */
    if (g_discovered_bdf[0])
        return g_discovered_bdf;
    return "0000:00:00.0";
}

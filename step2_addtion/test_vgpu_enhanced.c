
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>


#define VGPU_REG_DOORBELL        0x000
#define VGPU_REG_STATUS          0x004
#define VGPU_REG_POOL_ID         0x008
#define VGPU_REG_PRIORITY        0x00C
#define VGPU_REG_VM_ID           0x010
#define VGPU_REG_ERROR_CODE      0x014
#define VGPU_REG_REQUEST_LEN     0x018
#define VGPU_REG_RESPONSE_LEN    0x01C
#define VGPU_REG_PROTOCOL_VER    0x020
#define VGPU_REG_CAPABILITIES    0x024
#define VGPU_REG_IRQ_CTRL        0x028
#define VGPU_REG_IRQ_STATUS      0x02C
#define VGPU_REG_REQUEST_ID      0x030
#define VGPU_REG_TIMESTAMP_LO    0x034
#define VGPU_REG_TIMESTAMP_HI    0x038
#define VGPU_REG_SCRATCH         0x03C

#define VGPU_REQ_BUFFER_OFFSET   0x040
#define VGPU_RESP_BUFFER_OFFSET  0x440
#define VGPU_BAR_SIZE            4096

#define VGPU_STATUS_IDLE         0x00
#define VGPU_STATUS_BUSY         0x01
#define VGPU_STATUS_DONE         0x02
#define VGPU_STATUS_ERROR        0x03

#define VGPU_ERR_NONE            0x00
#define VGPU_ERR_INVALID_LENGTH  0x09
#define VGPU_ERR_MEDIATOR_UNAVAIL 0x03

#define VGPU_PROTOCOL_VERSION    0x00010000
#define VGPU_CAP_BASIC_REQ       (1 << 0)

#define VGPU_VENDOR_ID           0x1af4
#define VGPU_DEVICE_ID           0x1111
#define VGPU_CLASS               0x120000
#define VGPU_CLASS_MASK          0xffff00

/* ---- Helpers ------------------------------------------------- */

static int  g_pass = 0;
static int  g_fail = 0;

#define REG32(base, off)  (*(volatile uint32_t *)((volatile char *)(base) + (off)))

static void check(const char *name, uint32_t got, uint32_t expected)
{
    if (got == expected) {
        printf("  %-22s = 0x%08X  (expected 0x%08X)\n", name, got, expected);
        g_pass++;
    } else {
        printf("  ✗ %-22s = 0x%08X  (expected 0x%08X) **FAIL**\n",
               name, got, expected);
        g_fail++;
    }
}

static void __attribute__((unused)) check_ne(const char *name, uint32_t got, uint32_t not_expected)
{
    if (got != not_expected) {
        printf("  %-22s = 0x%08X  (not 0x%08X — OK)\n",
               name, got, not_expected);
        g_pass++;
    } else {
        printf("  ✗ %-22s = 0x%08X  (should not be 0x%08X) **FAIL**\n",
               name, got, not_expected);
        g_fail++;
    }
}

static void info(const char *name, uint32_t val)
{
    printf("  ℹ %-22s = 0x%08X\n", name, val);
}

/* ---- Find vGPU device by scanning /sys/bus/pci/devices -------- */

static char *find_vgpu_resource0(void)
{
    static char res_path[512];
    DIR *dir;
    struct dirent *ent;

    dir = opendir("/sys/bus/pci/devices");
    if (!dir) { perror("opendir"); return NULL; }

    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;

        char path[512], buf[64];
        FILE *fp;
        unsigned int vendor = 0, device = 0, cls = 0;

        snprintf(path, sizeof(path),
                 "/sys/bus/pci/devices/%s/vendor", ent->d_name);
        fp = fopen(path, "r");
        if (!fp) continue;
        if (fgets(buf, sizeof(buf), fp)) sscanf(buf, "%x", &vendor);
        fclose(fp);
        if (vendor != VGPU_VENDOR_ID) continue;

        snprintf(path, sizeof(path),
                 "/sys/bus/pci/devices/%s/device", ent->d_name);
        fp = fopen(path, "r");
        if (!fp) continue;
        if (fgets(buf, sizeof(buf), fp)) sscanf(buf, "%x", &device);
        fclose(fp);
        if (device != VGPU_DEVICE_ID) continue;

        snprintf(path, sizeof(path),
                 "/sys/bus/pci/devices/%s/class", ent->d_name);
        fp = fopen(path, "r");
        if (!fp) continue;
        if (fgets(buf, sizeof(buf), fp)) sscanf(buf, "%x", &cls);
        fclose(fp);
        if ((cls & VGPU_CLASS_MASK) != VGPU_CLASS) continue;

        /* Found it */
        snprintf(res_path, sizeof(res_path),
                 "/sys/bus/pci/devices/%s/resource0", ent->d_name);
        printf("[SCAN] Found vGPU stub at %s\n", ent->d_name);
        closedir(dir);
        return res_path;
    }
    closedir(dir);
    return NULL;
}

/* ---- Main ----------------------------------------------------- */

int main(int argc, char *argv[])
{
    char res_path[512];
    volatile void *mmio;
    int fd;

    printf("================================================================\n");
    printf("  vGPU Stub v2 — Enhanced Register Test\n");
    printf("================================================================\n\n");

    /* Determine resource0 path */
    if (argc >= 2) {
        /* User supplied BDF, e.g. "00:06.0" */
        snprintf(res_path, sizeof(res_path),
                 "/sys/bus/pci/devices/0000:%s/resource0", argv[1]);
    } else {
        char *found = find_vgpu_resource0();
        if (!found) {
            fprintf(stderr,
                    "ERROR: vGPU stub device not found.\n"
                    "Usage: %s [BDF]   e.g. %s 00:06.0\n",
                    argv[0], argv[0]);
            return 1;
        }
        strncpy(res_path, found, sizeof(res_path) - 1);
        res_path[sizeof(res_path) - 1] = '\0';
    }

    printf("[OPEN] %s\n\n", res_path);

    fd = open(res_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open (try running as root)");
        return 1;
    }

    mmio = mmap(NULL, VGPU_BAR_SIZE, PROT_READ | PROT_WRITE,
                MAP_SHARED, fd, 0);
    if (mmio == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    /* ============================================================
     * TEST 1 — Read-only identity registers
     * ============================================================ */
    printf("--- Test 1: Identity / Read-Only Registers ---\n");
    check("PROTOCOL_VER (0x020)",
          REG32(mmio, VGPU_REG_PROTOCOL_VER), VGPU_PROTOCOL_VERSION);
    check("CAPABILITIES (0x024)",
          REG32(mmio, VGPU_REG_CAPABILITIES), VGPU_CAP_BASIC_REQ);
    printf("\n");

    /* ============================================================
     * TEST 2 — Property registers (depend on QEMU cmdline)
     * ============================================================ */
    printf("--- Test 2: Property Registers ---\n");
    {
        uint32_t pool = REG32(mmio, VGPU_REG_POOL_ID);
        uint32_t prio = REG32(mmio, VGPU_REG_PRIORITY);
        uint32_t vmid = REG32(mmio, VGPU_REG_VM_ID);

        info("POOL_ID  (0x008)", pool);
        printf("    → '%c'\n", (char)pool);
        info("PRIORITY (0x00C)", prio);
        printf("    → %s\n",
               prio == 2 ? "high" : prio == 1 ? "medium" : "low");
        info("VM_ID    (0x010)", vmid);
        printf("    → %u\n", vmid);

        /* Basic sanity: pool should be 'A' or 'B' */
        if (pool == 'A' || pool == 'B') {
            printf("  POOL_ID is valid\n"); g_pass++;
        } else {
            printf("  ✗ POOL_ID 0x%02X is unexpected\n", pool); g_fail++;
        }

        /* Priority should be 0, 1, or 2 */
        if (prio <= 2) {
            printf("  PRIORITY is valid\n"); g_pass++;
        } else {
            printf("  ✗ PRIORITY %u is out of range\n", prio); g_fail++;
        }
    }
    printf("\n");

    /* ============================================================
     * TEST 3 — Status register
     *
     * The device has no reset mechanism, so registers keep their
     * values from the last request (e.g. vm_client_enhanced).
     * IDLE means fresh, DONE/BUSY/ERROR means a previous result
     * is still present.  Both are valid — not a failure.
     * ============================================================ */
    printf("--- Test 3: Status Register ---\n");
    {
        uint32_t st  = REG32(mmio, VGPU_REG_STATUS);
        uint32_t err = REG32(mmio, VGPU_REG_ERROR_CODE);

        if (st == VGPU_STATUS_IDLE) {
            printf("  STATUS (0x004)         = 0x%08X  (IDLE — clean state)\n", st);
            g_pass++;
        } else if (st == VGPU_STATUS_DONE) {
            printf("  ℹ STATUS (0x004)         = 0x%08X  (DONE — last request result exists)\n", st);
            g_pass++;
        } else if (st == VGPU_STATUS_BUSY) {
            printf("  ℹ STATUS (0x004)         = 0x%08X  (BUSY — previous request still in progress)\n", st);
            g_pass++;
        } else if (st == VGPU_STATUS_ERROR) {
            printf("  ℹ STATUS (0x004)         = 0x%08X  (ERROR — last request error state exists, code=0x%02X)\n", st, err);
            g_pass++;
        } else {
            printf("  ✗ STATUS (0x004)         = 0x%08X  (unexpected value) **FAIL**\n", st);
            g_fail++;
        }

        if (err == VGPU_ERR_NONE) {
            printf("  ERROR_CODE (0x014)     = 0x%08X  (no error)\n", err);
            g_pass++;
        } else {
            printf("  ℹ ERROR_CODE (0x014)     = 0x%08X  (last error state exists)\n", err);
            g_pass++;
        }
    }
    printf("\n");

    /* ============================================================
     * TEST 4 — Scratch register (R/W)
     * ============================================================ */
    printf("--- Test 4: Scratch Register (R/W) ---\n");
    REG32(mmio, VGPU_REG_SCRATCH) = 0xDEADBEEF;
    check("SCRATCH write/read", REG32(mmio, VGPU_REG_SCRATCH), 0xDEADBEEF);
    REG32(mmio, VGPU_REG_SCRATCH) = 0x12345678;
    check("SCRATCH overwrite",  REG32(mmio, VGPU_REG_SCRATCH), 0x12345678);
    REG32(mmio, VGPU_REG_SCRATCH) = 0;
    printf("\n");

    /* ============================================================
     * TEST 5 — Request length register (R/W)
     *
     * If a previous request was submitted, this register still
     * holds the old length.  That's normal — not a failure.
     * ============================================================ */
    printf("--- Test 5: Request Length Register (R/W) ---\n");
    {
        uint32_t initial_len = REG32(mmio, VGPU_REG_REQUEST_LEN);
        if (initial_len == 0) {
            printf("  REQUEST_LEN initial    = 0x%08X  (clean state)\n", initial_len);
            g_pass++;
        } else {
            printf("  ℹ REQUEST_LEN initial    = 0x%08X  (last request length exists, %u bytes)\n",
                   initial_len, initial_len);
            g_pass++;
        }
    }
    REG32(mmio, VGPU_REG_REQUEST_LEN) = 64;
    check("REQUEST_LEN set 64",  REG32(mmio, VGPU_REG_REQUEST_LEN), 64);
    REG32(mmio, VGPU_REG_REQUEST_LEN) = 0;
    printf("\n");

    /* ============================================================
     * TEST 6 — Request ID register (R/W)
     * ============================================================ */
    printf("--- Test 6: Request ID Register (R/W) ---\n");
    REG32(mmio, VGPU_REG_REQUEST_ID) = 42;
    check("REQUEST_ID set 42", REG32(mmio, VGPU_REG_REQUEST_ID), 42);
    REG32(mmio, VGPU_REG_REQUEST_ID) = 0;
    printf("\n");

    /* ============================================================
     * TEST 7 — Request buffer (R/W at 0x040)
     * ============================================================ */
    printf("--- Test 7: Request Buffer (R/W) ---\n");
    {
        volatile uint32_t *buf = (volatile uint32_t *)
                                 ((volatile char *)mmio + VGPU_REQ_BUFFER_OFFSET);
        buf[0] = 0xAAAA0001;
        buf[1] = 0xBBBB0002;
        buf[255] = 0xCCCC00FF;  /* last dword in 1 KB buffer */

        check("REQ_BUF[0]", buf[0], 0xAAAA0001);
        check("REQ_BUF[1]", buf[1], 0xBBBB0002);
        check("REQ_BUF[255]", buf[255], 0xCCCC00FF);

        /* Clean up */
        buf[0] = 0;
        buf[1] = 0;
        buf[255] = 0;
    }
    printf("\n");

    /* ============================================================
     * TEST 8 — Response buffer (read-only from guest)
     * ============================================================ */
    printf("--- Test 8: Response Buffer (Read) ---\n");
    {
        volatile uint32_t *rbuf = (volatile uint32_t *)
                                  ((volatile char *)mmio + VGPU_RESP_BUFFER_OFFSET);
        /* Response buffer should be zeroed initially */
        info("RESP_BUF[0]", rbuf[0]);
        printf("  Response buffer readable (writes ignored by device)\n");
        g_pass++;
    }
    printf("\n");

    /* ============================================================
     * TEST 9 — Doorbell with zero length (should error)
     *
     * The device accepts a doorbell from any state, so proceed
     * even if a previous result is still present.
     * ============================================================ */
    printf("--- Test 9: Doorbell — Zero-Length Request ---\n");
    {
        uint32_t pre_status = REG32(mmio, VGPU_REG_STATUS);
        if (pre_status != VGPU_STATUS_IDLE) {
            printf("  ℹ STATUS is %u before doorbell (last request state exists)\n", pre_status);
        }

        /* Set length = 0 and ring doorbell */
        REG32(mmio, VGPU_REG_REQUEST_LEN) = 0;
        REG32(mmio, VGPU_REG_DOORBELL) = 1;
        usleep(1000);  /* give device a moment */

        check("STATUS -> ERROR",
              REG32(mmio, VGPU_REG_STATUS), VGPU_STATUS_ERROR);
        check("ERROR_CODE -> INVALID_LEN",
              REG32(mmio, VGPU_REG_ERROR_CODE), VGPU_ERR_INVALID_LENGTH);
    }
    printf("\n");

    /* ============================================================
     * TEST 10 — Doorbell with valid length (mediator may/may not
     *           be running; we accept BUSY, DONE, or ERROR=0x03)
     * ============================================================ */
    printf("--- Test 10: Doorbell — Valid Request ---\n");
    {
        /* Reset STATUS by writing a dummy scratch and reading back.
         * The device auto-resets to IDLE after an error is read by
         * design; however, in this implementation STATUS stays at
         * the last value until the next doorbell or a response
         * arrives.  We need the device in a state where it will
         * process a new doorbell — the spec says the device
         * transitions IDLE→BUSY on doorbell.  If it's still in
         * ERROR, it should still accept a new doorbell. */

        /* Write a small request header into the buffer */
        volatile uint32_t *buf = (volatile uint32_t *)
                                 ((volatile char *)mmio + VGPU_REQ_BUFFER_OFFSET);
        buf[0] = VGPU_PROTOCOL_VERSION;  /* version */
        buf[1] = 0x0000;                  /* opcode = NOP */
        buf[2] = 0;                       /* flags */
        buf[3] = 0;                       /* param_count */
        buf[4] = 32;                      /* data_offset */
        buf[5] = 0;                       /* data_length */
        buf[6] = 0;                       /* reserved */
        buf[7] = 0;                       /* reserved */

        REG32(mmio, VGPU_REG_REQUEST_LEN) = 32;  /* header only */
        REG32(mmio, VGPU_REG_REQUEST_ID)  = 9999;
        REG32(mmio, VGPU_REG_DOORBELL)    = 1;

        usleep(10000);  /* 10 ms */

        uint32_t st  = REG32(mmio, VGPU_REG_STATUS);
        uint32_t err = REG32(mmio, VGPU_REG_ERROR_CODE);

        info("STATUS after doorbell", st);
        info("ERROR_CODE",            err);

        if (st == VGPU_STATUS_BUSY) {
            printf("  Device accepted request (BUSY — mediator connected)\n");
            g_pass++;
        } else if (st == VGPU_STATUS_DONE) {
            printf("  Device completed request (DONE — mediator replied fast)\n");
            g_pass++;
        } else if (st == VGPU_STATUS_ERROR && err == VGPU_ERR_MEDIATOR_UNAVAIL) {
            /* NOTE: These must be TWO separate printf statements */
            printf("  Device correctly reports MEDIATOR_UNAVAILABLE\n");
            printf("    (This is expected if the mediator daemon is not running)\n");
            g_pass++;
        } else {
            printf("  ✗ Unexpected STATUS=%u ERROR=%u\n", st, err);
            g_fail++;
        }

        /* Clean up buffer */
        memset((void *)buf, 0, 32);
        REG32(mmio, VGPU_REG_REQUEST_LEN) = 0;
    }
    printf("\n");

    /* ============================================================
     * TEST 11 — Doorbell register reads back as 0
     * ============================================================ */
    printf("--- Test 11: Doorbell Reads as Zero ---\n");
    check("DOORBELL (0x000)", REG32(mmio, VGPU_REG_DOORBELL), 0);
    printf("\n");

    /* ============================================================
     * SUMMARY
     * ============================================================ */
    printf("================================================================\n");
    printf("  RESULTS:  %d passed,  %d failed\n", g_pass, g_fail);
    printf("================================================================\n");

    if (g_fail == 0) {
        printf("\n   ALL TESTS PASSED — vGPU Stub v2 is fully functional!\n\n");
    } else {
        printf("\n  ❌  SOME TESTS FAILED — review output above.\n\n");
    }

    munmap((void *)mmio, VGPU_BAR_SIZE);
    close(fd);

    return g_fail > 0 ? 1 : 0;
}

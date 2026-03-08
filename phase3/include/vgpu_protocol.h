#ifndef VGPU_PROTOCOL_H
#define VGPU_PROTOCOL_H

#include <stdint.h>

/* ================================================================
 * MMIO Register Offsets
 * ================================================================ */

#define VGPU_REG_DOORBELL        0x000  /* R/W  Write 1 to submit request       */
#define VGPU_REG_STATUS          0x004  /* RO   Device status                   */
#define VGPU_REG_POOL_ID         0x008  /* RO   Pool ID ('A'=0x41, 'B'=0x42)   */
#define VGPU_REG_PRIORITY        0x00C  /* RO   Priority (0=low,1=med,2=high)   */
#define VGPU_REG_VM_ID           0x010  /* RO   VM identifier                   */
#define VGPU_REG_ERROR_CODE      0x014  /* RO   Error code if STATUS==ERROR     */
#define VGPU_REG_REQUEST_LEN     0x018  /* R/W  Guest writes request length     */
#define VGPU_REG_RESPONSE_LEN    0x01C  /* RO   Host writes response length     */
#define VGPU_REG_PROTOCOL_VER    0x020  /* RO   Protocol version (0x00010000)   */
#define VGPU_REG_CAPABILITIES    0x024  /* RO   Feature bits                    */
#define VGPU_REG_IRQ_CTRL        0x028  /* R/W  Interrupt control               */
#define VGPU_REG_IRQ_STATUS      0x02C  /* RW1C Interrupt status                */
#define VGPU_REG_REQUEST_ID      0x030  /* R/W  Request tracking ID             */
#define VGPU_REG_TIMESTAMP_LO    0x034  /* RO   Completion timestamp low 32     */
#define VGPU_REG_TIMESTAMP_HI    0x038  /* RO   Completion timestamp high 32    */
#define VGPU_REG_SCRATCH         0x03C  /* R/W  Scratch register for testing    */

/* Buffer regions */
#define VGPU_REQ_BUFFER_OFFSET   0x040  /* Request buffer start                 */
#define VGPU_REQ_BUFFER_SIZE     1024   /* Request buffer size in bytes         */
#define VGPU_RESP_BUFFER_OFFSET  0x440  /* Response buffer start                */
#define VGPU_RESP_BUFFER_SIZE    1024   /* Response buffer size in bytes        */
#define VGPU_RESERVED_OFFSET     0x840  /* Reserved region start                */
#define VGPU_BAR_SIZE            4096   /* Total BAR0 size                      */

/* Control register block end */
#define VGPU_CTRL_REG_END        0x040

/* ================================================================
 * CUDA API Remoting Registers (BAR0 region 0x080 – 0x0FF)
 *
 * These registers handle forwarding of CUDA API calls between the
 * guest shim libraries and the host mediator.
 * ================================================================ */
#define VGPU_REG_CUDA_OP           0x080  /* R/W  CUDA API call identifier      */
#define VGPU_REG_CUDA_SEQ          0x084  /* R/W  Sequence number               */
#define VGPU_REG_CUDA_NUM_ARGS     0x088  /* R/W  Number of inline args         */
#define VGPU_REG_CUDA_DATA_LEN     0x08C  /* R/W  Bulk data length              */
#define VGPU_REG_CUDA_DOORBELL     0x0A8  /* W    Ring to submit CUDA call      */
#define VGPU_REG_CUDA_ARGS_BASE    0x0B0  /* R/W  Inline args (16 x uint32)     */
#define VGPU_REG_CUDA_ARGS_END     0x0F0  /* (exclusive)                        */
#define VGPU_REG_CUDA_RESULT_STATUS    0x0F0  /* RO  CUresult return value      */
#define VGPU_REG_CUDA_RESULT_NUM       0x0F4  /* RO  Number of result values    */
#define VGPU_REG_CUDA_RESULT_DATA_LEN  0x0F8  /* RO  Result bulk data length    */

/* CUDA inline request data region (small payloads in BAR0) */
#define VGPU_CUDA_REQ_DATA_OFFSET   0x100  /* 1 KB for small request data      */
#define VGPU_CUDA_RESP_DATA_OFFSET  0x500  /* 1 KB for small response data     */
#define VGPU_CUDA_SMALL_DATA_MAX    1024

/* CUDA result inline values (8 x uint64 = 64 bytes at 0x900) */
#define VGPU_REG_CUDA_RESULT_BASE  0x900

/* CUDA control register block end */
#define VGPU_CUDA_CTRL_END         0x100

/* Maximum CUDA inline args (matches cuda_protocol.h) */
#define VGPU_CUDA_MAX_ARGS         16

/* ================================================================
 * Shared-Memory Registration Registers (BAR0, 0x0C0 – 0x0CC)
 *
 * The guest CUDA shim allocates a large anonymous mmap region, locks
 * it in RAM (mlock), resolves its guest physical address (GPA) via
 * /proc/self/pagemap, and registers it with the vgpu-stub device.
 *
 * The vgpu-stub then calls cpu_physical_memory_map(GPA) to obtain a
 * host-side pointer directly into guest RAM, eliminating the 8 MB
 * BAR1 MMIO copy bottleneck.
 *
 * Layout of the registered region:
 *   [0 .. shmem_size/2)          — Guest → Host data  (G2H)
 *   [shmem_size/2 .. shmem_size) — Host → Guest data  (H2G)
 *
 * Protocol:
 *   1. Guest writes GPA to SHMEM_GPA_LO/HI and size to SHMEM_SIZE.
 *   2. Guest writes 1 to SHMEM_CTRL.
 *   3. Guest polls STATUS until DONE (vgpu-stub has mapped the region).
 *   4. For teardown, guest writes 0 to SHMEM_CTRL.
 * ================================================================ */
/* Placed AFTER the CUDA result-value block (0x900 + 8*8 = 0x940) so these
 * addresses do not overlap with the CUDA control registers (0x080-0x0FF) or
 * the CUDA arg registers (0x0B0-0x0EF). */
#define VGPU_REG_SHMEM_GPA_LO  0x940  /* Guest physical addr, low  32 bits */
#define VGPU_REG_SHMEM_GPA_HI  0x944  /* Guest physical addr, high 32 bits */
#define VGPU_REG_SHMEM_SIZE    0x948  /* Total region size in bytes        */
#define VGPU_REG_SHMEM_CTRL    0x94C  /* Write 1 = register, 0 = release   */

/* Default shared-memory region size used by the guest shim (256 MB).
 * Half is G2H, half is H2G.  The guest may negotiate a smaller size
 * if its ulimits prevent locking 256 MB. */
#define VGPU_SHMEM_DEFAULT_SIZE  (256u * 1024u * 1024u)  /* 256 MB */
#define VGPU_SHMEM_MIN_SIZE      (  8u * 1024u * 1024u)  /*   8 MB fallback */

/* ================================================================
 * BAR1 — Large Data Region (16 MB)
 *
 * Retained as a fallback for guests that cannot lock the shmem region
 * (e.g. non-root, ulimit restrictions).  When VGPU_CAP_SHMEM is set
 * and the guest successfully registers a shared-memory region, BAR1 is
 * not used and the bar1_data buffer in vgpu-stub is freed.
 * ================================================================ */
#define VGPU_BAR1_SIZE              (16 * 1024 * 1024)  /* 16 MB */
#define VGPU_BAR1_G2H_OFFSET       0x000000  /* Guest-to-Host  (8 MB) */
#define VGPU_BAR1_G2H_SIZE         (8 * 1024 * 1024)
#define VGPU_BAR1_H2G_OFFSET       0x800000  /* Host-to-Guest  (8 MB) */
#define VGPU_BAR1_H2G_SIZE         (8 * 1024 * 1024)

/* ================================================================
 * STATUS Register Values (offset 0x004)
 * ================================================================ */

#define VGPU_STATUS_IDLE         0x00   /* Ready for new request                */
#define VGPU_STATUS_BUSY         0x01   /* Processing request                   */
#define VGPU_STATUS_DONE         0x02   /* Request completed, response ready    */
#define VGPU_STATUS_ERROR        0x03   /* Request failed, see ERROR_CODE       */

/* ================================================================
 * ERROR_CODE Register Values (offset 0x014)
 * ================================================================ */

#define VGPU_ERR_NONE                 0x00  /* No error                         */
#define VGPU_ERR_INVALID_REQUEST      0x01  /* Request format invalid           */
#define VGPU_ERR_REQUEST_TOO_LARGE    0x02  /* Request exceeds 1024 bytes       */
#define VGPU_ERR_MEDIATOR_UNAVAIL     0x03  /* Cannot connect to mediator       */
#define VGPU_ERR_TIMEOUT              0x04  /* Request processing timeout       */
#define VGPU_ERR_CUDA_ERROR           0x05  /* CUDA execution failed            */
#define VGPU_ERR_INVALID_POOL         0x06  /* Pool ID not available            */
#define VGPU_ERR_QUEUE_FULL           0x07  /* Request queue full               */
#define VGPU_ERR_UNSUPPORTED_OP       0x08  /* Operation not supported          */
#define VGPU_ERR_INVALID_LENGTH       0x09  /* Request length is 0              */
/* Phase 3: Isolation error codes */
#define VGPU_ERR_RATE_LIMITED         0x0A  /* VM hit rate limit (back-pressure)*/
#define VGPU_ERR_VM_QUARANTINED       0x0B  /* VM quarantined (too many faults) */

/* ================================================================
 * CAPABILITIES Register Bits (offset 0x024)
 * ================================================================ */

#define VGPU_CAP_BASIC_REQ       (1 << 0)   /* Basic request/response           */
#define VGPU_CAP_INTERRUPT       (1 << 1)   /* Interrupt notification (future)  */
#define VGPU_CAP_DMA             (1 << 2)   /* DMA transfers (future)           */
#define VGPU_CAP_MULTI_REQ       (1 << 3)   /* Multiple outstanding (future)    */
#define VGPU_CAP_CUDA_REMOTE     (1 << 4)   /* CUDA API remoting support        */
#define VGPU_CAP_BAR1_DATA       (1 << 5)   /* BAR1 large data region (legacy)  */
#define VGPU_CAP_SHMEM           (1 << 6)   /* Guest-pinned shared-memory path  */

/* ================================================================
 * PROTOCOL_VER Value (offset 0x020)
 * ================================================================ */

#define VGPU_PROTOCOL_VERSION    0x00010000  /* v1.0 = major<<16 | minor         */

/* ================================================================
 * Request/Response Operation Codes
 * ================================================================ */

#define VGPU_OP_NOP              0x0000  /* No operation (test)                  */
#define VGPU_OP_CUDA_KERNEL      0x0001  /* Execute CUDA kernel (vector add)    */
#define VGPU_OP_GET_DEVICE_INFO  0x0005  /* Query device information            */

/* ================================================================
 * Request Structure (written to MMIO request buffer at 0x040)
 *
 * Guest writes this structure, then sets REQUEST_LEN, then
 * writes 1 to DOORBELL.
 * ================================================================ */

typedef struct __attribute__((packed)) VGPURequest {
    uint32_t version;        /* Protocol version (VGPU_PROTOCOL_VERSION)        */
    uint32_t opcode;         /* Operation code (VGPU_OP_*)                      */
    uint32_t flags;          /* Request flags (0 for now)                       */
    uint32_t param_count;    /* Number of uint32_t parameters following         */
    uint32_t data_offset;    /* Byte offset to variable data (from struct start)*/
    uint32_t data_length;    /* Length of variable data in bytes                */
    uint32_t reserved[2];    /* Must be 0                                       */
    /* params[param_count] follows, then data[data_length] */
} VGPURequest;

#define VGPU_REQUEST_HEADER_SIZE  sizeof(VGPURequest)  /* 32 bytes */

/* Maximum number of uint32_t parameters that fit in request buffer */
#define VGPU_MAX_PARAMS  ((VGPU_REQ_BUFFER_SIZE - VGPU_REQUEST_HEADER_SIZE) / sizeof(uint32_t))

/* ================================================================
 * Response Structure (read from MMIO response buffer at 0x440)
 *
 * Host writes this structure after CUDA completes. Guest reads
 * after STATUS == DONE.
 * ================================================================ */

typedef struct __attribute__((packed)) VGPUResponse {
    uint32_t version;        /* Protocol version (VGPU_PROTOCOL_VERSION)        */
    uint32_t status;         /* 0=success, non-zero=error                       */
    uint32_t result_count;   /* Number of uint32_t results following            */
    uint32_t data_offset;    /* Byte offset to variable data (from struct start)*/
    uint32_t data_length;    /* Length of variable data in bytes                */
    uint32_t exec_time_us;   /* CUDA execution time in microseconds            */
    uint32_t reserved[2];    /* Reserved                                        */
    /* results[result_count] follows, then data[data_length] */
} VGPUResponse;

#define VGPU_RESPONSE_HEADER_SIZE  sizeof(VGPUResponse)  /* 32 bytes */

/* Maximum number of uint32_t results that fit in response buffer */
#define VGPU_MAX_RESULTS  ((VGPU_RESP_BUFFER_SIZE - VGPU_RESPONSE_HEADER_SIZE) / sizeof(uint32_t))

/* ================================================================
 * Socket IPC Protocol (between vgpu-stub and mediator daemon)
 *
 * vgpu-stub connects to mediator via Unix domain socket.
 * Socket: Filesystem Unix socket (inside QEMU chroot directory)
 * The mediator auto-discovers the QEMU chroot and creates the socket there.
 * QEMU (chrooted) sees it at /tmp/vgpu-mediator.sock.
 *
 * Message format (both directions):
 *   [VGPUSocketHeader][payload]
 * ================================================================ */

/* Filesystem Unix socket path (as seen from inside the QEMU chroot) */
#define VGPU_SOCKET_PATH  "/tmp/vgpu-mediator.sock"

/* Message types for socket IPC */
#define VGPU_MSG_REQUEST    0x01   /* vgpu-stub → mediator: new request       */
#define VGPU_MSG_RESPONSE   0x02   /* mediator → vgpu-stub: result ready      */
#define VGPU_MSG_PING       0x03   /* Either direction: keepalive             */
#define VGPU_MSG_PONG       0x04   /* Reply to PING                           */
/* Phase 3: Back-pressure and quarantine notifications */
#define VGPU_MSG_BUSY       0x05   /* mediator → vgpu-stub: rate-limited      */
#define VGPU_MSG_QUARANTINED 0x06  /* mediator → vgpu-stub: VM quarantined    */
/* Phase 3+: CUDA API remoting messages */
#define VGPU_MSG_CUDA_CALL      0x10  /* vgpu-stub → mediator: CUDA API call */
#define VGPU_MSG_CUDA_RESULT    0x11  /* mediator → vgpu-stub: CUDA result   */
#define VGPU_MSG_CUDA_DATA      0x12  /* Either: large data chunk transfer   */

typedef struct __attribute__((packed)) VGPUSocketHeader {
    uint32_t magic;          /* 0x56475055 = "VGPU"                            */
    uint32_t msg_type;       /* VGPU_MSG_* type                                */
    uint32_t vm_id;          /* VM identifier                                  */
    uint32_t request_id;     /* Request tracking ID                            */
    char     pool_id;        /* 'A' or 'B'                                     */
    uint8_t  priority;       /* 0=low, 1=medium, 2=high                        */
    uint16_t _pad;           /* alignment padding — do not use                 */
    uint32_t payload_len;    /* Length of payload following this header         */
} VGPUSocketHeader;

#define VGPU_SOCKET_MAGIC    0x56475055  /* "VGPU" in little-endian            */
#define VGPU_SOCKET_HDR_SIZE sizeof(VGPUSocketHeader)  /* 24 bytes             */

/* Maximum payload size — legacy (1 KB) */
#define VGPU_SOCKET_MAX_PAYLOAD  VGPU_REQ_BUFFER_SIZE  /* 1024 bytes          */

/* Maximum CUDA payload (large transfers go via BAR1) */
#define VGPU_CUDA_SOCKET_MAX_PAYLOAD  (8 * 1024 * 1024)  /* 8 MB              */

/* ================================================================
 * Phase 3: Admin Socket (for vgpu-admin CLI → mediator)
 * ================================================================ */

/* Admin socket path (separate from QEMU chroot sockets) */
#define VGPU_ADMIN_SOCKET_PATH  "/var/vgpu/admin.sock"

/* Admin command codes (sent over admin socket) */
#define VGPU_ADMIN_SHOW_METRICS      0x10
#define VGPU_ADMIN_SHOW_HEALTH       0x11
#define VGPU_ADMIN_RELOAD_CONFIG     0x12
#define VGPU_ADMIN_QUARANTINE_VM     0x13
#define VGPU_ADMIN_UNQUARANTINE_VM   0x14
#define VGPU_ADMIN_SHOW_CONNECTIONS  0x15

/* Admin request header */
typedef struct __attribute__((packed)) VGPUAdminRequest {
    uint32_t magic;          /* VGPU_SOCKET_MAGIC                              */
    uint32_t command;        /* VGPU_ADMIN_* code                              */
    uint32_t param1;         /* Command-specific parameter                     */
    uint32_t param2;         /* Command-specific parameter                     */
} VGPUAdminRequest;

/* Admin response: variable-length text follows this header */
typedef struct __attribute__((packed)) VGPUAdminResponse {
    uint32_t magic;          /* VGPU_SOCKET_MAGIC                              */
    uint32_t status;         /* 0 = success, non-zero = error                  */
    uint32_t data_len;       /* Length of text payload that follows             */
} VGPUAdminResponse;

/* ================================================================
 * PCI Device Identification
 * ================================================================ */

#define VGPU_VENDOR_ID       0x10DE  /* NVIDIA Corporation                     */
#define VGPU_DEVICE_ID       0x2331  /* H100 80GB PCIe                         */
#define VGPU_CLASS_ID        0x0302  /* 3D controller                          */
#define VGPU_SUBSYS_VENDOR_ID 0x10DE /* NVIDIA Corporation (subsystem)         */
#define VGPU_SUBSYS_DEVICE_ID 0x16C1 /* H100 PCIe subsystem                   */
#define VGPU_REVISION        0xA1   /* Match real H100 silicon revision        */

/* ================================================================
 * Priority Values (shared with mediator)
 * ================================================================ */

#define VGPU_PRIORITY_LOW    0
#define VGPU_PRIORITY_MEDIUM 1
#define VGPU_PRIORITY_HIGH   2

#endif /* VGPU_PROTOCOL_H */

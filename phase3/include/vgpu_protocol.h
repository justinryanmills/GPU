#ifndef VGPU_PROTOCOL_H
#define VGPU_PROTOCOL_H

#include <stdint.h>

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
#define VGPU_REQ_BUFFER_SIZE     1024
#define VGPU_RESP_BUFFER_OFFSET  0x440
#define VGPU_RESP_BUFFER_SIZE    1024
#define VGPU_RESERVED_OFFSET     0x840
#define VGPU_BAR_SIZE            4096
#define VGPU_CTRL_REG_END        0x040

#define VGPU_REG_CUDA_OP           0x080
#define VGPU_REG_CUDA_SEQ          0x084
#define VGPU_REG_CUDA_NUM_ARGS     0x088
#define VGPU_REG_CUDA_DATA_LEN     0x08C
#define VGPU_REG_CUDA_DOORBELL     0x0A8
#define VGPU_REG_CUDA_ARGS_BASE    0x0B0
#define VGPU_REG_CUDA_ARGS_END     0x0F0
#define VGPU_REG_CUDA_RESULT_STATUS    0x0F0
#define VGPU_REG_CUDA_RESULT_NUM       0x0F4
#define VGPU_REG_CUDA_RESULT_DATA_LEN  0x0F8
#define VGPU_CUDA_REQ_DATA_OFFSET   0x100
#define VGPU_CUDA_RESP_DATA_OFFSET  0x500
#define VGPU_CUDA_SMALL_DATA_MAX    1024
#define VGPU_REG_CUDA_RESULT_BASE  0x900
#define VGPU_CUDA_CTRL_END         0x100
#define VGPU_CUDA_MAX_ARGS         16

#define VGPU_REG_SHMEM_GPA_LO  0x940
#define VGPU_REG_SHMEM_GPA_HI  0x944
#define VGPU_REG_SHMEM_SIZE    0x948
#define VGPU_REG_SHMEM_CTRL    0x94C
#define VGPU_SHMEM_DEFAULT_SIZE  (256u * 1024u * 1024u)
#define VGPU_SHMEM_MIN_SIZE      (  8u * 1024u * 1024u)
#define VGPU_BAR1_SIZE              (16 * 1024 * 1024)
#define VGPU_BAR1_G2H_OFFSET       0x000000
#define VGPU_BAR1_G2H_SIZE         (8 * 1024 * 1024)
#define VGPU_BAR1_H2G_OFFSET       0x800000
#define VGPU_BAR1_H2G_SIZE         (8 * 1024 * 1024)

#define VGPU_STATUS_IDLE         0x00
#define VGPU_STATUS_BUSY         0x01
#define VGPU_STATUS_DONE         0x02
#define VGPU_STATUS_ERROR        0x03

#define VGPU_ERR_NONE                 0x00
#define VGPU_ERR_INVALID_REQUEST      0x01
#define VGPU_ERR_REQUEST_TOO_LARGE    0x02
#define VGPU_ERR_MEDIATOR_UNAVAIL     0x03
#define VGPU_ERR_TIMEOUT              0x04
#define VGPU_ERR_CUDA_ERROR           0x05
#define VGPU_ERR_INVALID_POOL         0x06
#define VGPU_ERR_QUEUE_FULL           0x07
#define VGPU_ERR_UNSUPPORTED_OP       0x08
#define VGPU_ERR_INVALID_LENGTH       0x09
#define VGPU_ERR_RATE_LIMITED         0x0A
#define VGPU_ERR_VM_QUARANTINED       0x0B

#define VGPU_CAP_BASIC_REQ       (1 << 0)
#define VGPU_CAP_INTERRUPT       (1 << 1)
#define VGPU_CAP_DMA             (1 << 2)
#define VGPU_CAP_MULTI_REQ       (1 << 3)
#define VGPU_CAP_CUDA_REMOTE     (1 << 4)
#define VGPU_CAP_BAR1_DATA       (1 << 5)
#define VGPU_CAP_SHMEM           (1 << 6)
#define VGPU_PROTOCOL_VERSION    0x00010000

#define VGPU_OP_NOP              0x0000
#define VGPU_OP_CUDA_KERNEL      0x0001
#define VGPU_OP_GET_DEVICE_INFO  0x0005

typedef struct __attribute__((packed)) VGPURequest {
    uint32_t version;
    uint32_t opcode;
    uint32_t flags;
    uint32_t param_count;
    uint32_t data_offset;
    uint32_t data_length;
    uint32_t reserved[2];
} VGPURequest;

#define VGPU_REQUEST_HEADER_SIZE  sizeof(VGPURequest)
#define VGPU_MAX_PARAMS  ((VGPU_REQ_BUFFER_SIZE - VGPU_REQUEST_HEADER_SIZE) / sizeof(uint32_t))

typedef struct __attribute__((packed)) VGPUResponse {
    uint32_t version;
    uint32_t status;
    uint32_t result_count;
    uint32_t data_offset;
    uint32_t data_length;
    uint32_t exec_time_us;
    uint32_t reserved[2];
} VGPUResponse;

#define VGPU_RESPONSE_HEADER_SIZE  sizeof(VGPUResponse)
#define VGPU_MAX_RESULTS  ((VGPU_RESP_BUFFER_SIZE - VGPU_RESPONSE_HEADER_SIZE) / sizeof(uint32_t))

#define VGPU_SOCKET_PATH  "/tmp/vgpu-mediator.sock"
#define VGPU_MSG_REQUEST    0x01
#define VGPU_MSG_RESPONSE   0x02
#define VGPU_MSG_PING       0x03
#define VGPU_MSG_PONG       0x04
#define VGPU_MSG_BUSY       0x05
#define VGPU_MSG_QUARANTINED 0x06
#define VGPU_MSG_CUDA_CALL      0x10
#define VGPU_MSG_CUDA_RESULT    0x11
#define VGPU_MSG_CUDA_DATA      0x12

typedef struct __attribute__((packed)) VGPUSocketHeader {
    uint32_t magic;
    uint32_t msg_type;
    uint32_t vm_id;
    uint32_t request_id;
    char     pool_id;
    uint8_t  priority;
    uint16_t _pad;
    uint32_t payload_len;
} VGPUSocketHeader;

#define VGPU_SOCKET_MAGIC    0x56475055
#define VGPU_SOCKET_HDR_SIZE sizeof(VGPUSocketHeader)
#define VGPU_SOCKET_MAX_PAYLOAD  VGPU_REQ_BUFFER_SIZE
#define VGPU_CUDA_SOCKET_MAX_PAYLOAD  (8 * 1024 * 1024)

#define VGPU_ADMIN_SOCKET_PATH  "/var/vgpu/admin.sock"
#define VGPU_ADMIN_SHOW_METRICS      0x10
#define VGPU_ADMIN_SHOW_HEALTH       0x11
#define VGPU_ADMIN_RELOAD_CONFIG     0x12
#define VGPU_ADMIN_QUARANTINE_VM     0x13
#define VGPU_ADMIN_UNQUARANTINE_VM   0x14
#define VGPU_ADMIN_SHOW_CONNECTIONS  0x15

typedef struct __attribute__((packed)) VGPUAdminRequest {
    uint32_t magic;
    uint32_t command;
    uint32_t param1;
    uint32_t param2;
} VGPUAdminRequest;

typedef struct __attribute__((packed)) VGPUAdminResponse {
    uint32_t magic;
    uint32_t status;
    uint32_t data_len;
} VGPUAdminResponse;

#define VGPU_VENDOR_ID       0x10DE
#define VGPU_DEVICE_ID       0x2331
#define VGPU_CLASS_ID        0x0302
#define VGPU_SUBSYS_VENDOR_ID 0x10DE
#define VGPU_SUBSYS_DEVICE_ID 0x16C1
#define VGPU_REVISION        0xA1
#define VGPU_PRIORITY_LOW    0
#define VGPU_PRIORITY_MEDIUM 1
#define VGPU_PRIORITY_HIGH   2

#endif /* VGPU_PROTOCOL_H */

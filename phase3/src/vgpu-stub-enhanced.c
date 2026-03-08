
#include "qemu/osdep.h"
#include "hw/pci/pci.h"
#include "hw/hw.h"
#include "hw/pci/msi.h"
#include "qemu/timer.h"
#include "qom/object.h"
#include "qemu/module.h"
#include "qemu/main-loop.h"     /* qemu_set_fd_handler */
#include "sysemu/kvm.h"
#include "hw/qdev-properties.h"
#include "qapi/error.h"

#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

#include "vgpu_protocol.h"
#include "cuda_protocol.h"

#define TYPE_VGPU_STUB  "vgpu-cuda"
#define VGPU_STUB(obj)  OBJECT_CHECK(VGPUStubState, (obj), TYPE_VGPU_STUB)

typedef struct VGPUStubState {
    PCIDevice parent_obj;

    MemoryRegion mmio;
    MemoryRegion mmio_bar1;
    uint32_t status_reg;
    uint32_t error_code;
    uint32_t request_len;
    uint32_t response_len;
    uint32_t irq_ctrl;
    uint32_t irq_status;
    uint32_t request_id;
    uint32_t timestamp_lo;
    uint32_t timestamp_hi;
    uint32_t scratch;
    uint8_t  req_buf[VGPU_REQ_BUFFER_SIZE];
    uint8_t  resp_buf[VGPU_RESP_BUFFER_SIZE];
    uint32_t cuda_op;
    uint32_t cuda_seq;
    uint32_t cuda_num_args;
    uint32_t cuda_data_len;
    uint32_t cuda_args[VGPU_CUDA_MAX_ARGS];
    uint32_t cuda_result_status;
    uint32_t cuda_result_num;
    uint32_t cuda_result_data_len;
    uint64_t cuda_results[8];
    uint8_t  cuda_req_data[VGPU_CUDA_SMALL_DATA_MAX];
    uint8_t  cuda_resp_data[VGPU_CUDA_SMALL_DATA_MAX];
    uint8_t *bar1_data;
    void    *shmem_g2h;
    void    *shmem_h2g;
    hwaddr   shmem_gpa;
    uint32_t shmem_size;
    int      shmem_active;
    uint32_t shmem_gpa_lo;
    uint32_t shmem_gpa_hi;
    uint32_t shmem_size_reg;
    char    *pool_id;
    char    *priority;
    uint32_t vm_id;
    int      mediator_fd;
    uint8_t *sock_rx_buf;
    uint32_t sock_rx_len;
    uint32_t sock_rx_cap;
} VGPUStubState;

#define SOCK_RX_DEFAULT_CAP  (VGPU_SOCKET_HDR_SIZE + VGPU_CUDA_SOCKET_MAX_PAYLOAD + 4096)

static void vgpu_process_doorbell(VGPUStubState *s);
static void vgpu_process_cuda_doorbell(VGPUStubState *s);
static void vgpu_try_connect_mediator(VGPUStubState *s);
static void vgpu_socket_read_handler(void *opaque);

static uint32_t vgpu_priority_to_int(const char *p)
{
    if (p) {
        if (strcmp(p, "high") == 0)   return VGPU_PRIORITY_HIGH;
        if (strcmp(p, "medium") == 0) return VGPU_PRIORITY_MEDIUM;
    }
    return VGPU_PRIORITY_LOW;
}

static uint64_t vgpu_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    VGPUStubState *s = opaque;
    uint64_t val = 0;

    /* --- Control registers (0x000 – 0x03F) -------------------- */
    if (addr < VGPU_CTRL_REG_END) {
        switch (addr) {

        case VGPU_REG_DOORBELL:
            /* Doorbell reads as 0 - write-only semantics */
            val = 0;
            break;

        case VGPU_REG_STATUS:
            val = s->status_reg;
            break;

        case VGPU_REG_POOL_ID:
            val = (s->pool_id && s->pool_id[0]) ?
                  (uint32_t)(unsigned char)s->pool_id[0] : (uint32_t)'A';
            break;

        case VGPU_REG_PRIORITY:
            val = vgpu_priority_to_int(s->priority);
            break;

        case VGPU_REG_VM_ID:
            val = s->vm_id;
            break;

        case VGPU_REG_ERROR_CODE:
            val = s->error_code;
            break;

        case VGPU_REG_REQUEST_LEN:
            val = s->request_len;
            break;

        case VGPU_REG_RESPONSE_LEN:
            val = s->response_len;
            break;

        case VGPU_REG_PROTOCOL_VER:
            val = VGPU_PROTOCOL_VERSION;
            break;

        case VGPU_REG_CAPABILITIES:
            val = VGPU_CAP_BASIC_REQ | VGPU_CAP_CUDA_REMOTE |
                  VGPU_CAP_BAR1_DATA | VGPU_CAP_SHMEM;
            break;

        case VGPU_REG_IRQ_CTRL:
            val = s->irq_ctrl;
            break;

        case VGPU_REG_IRQ_STATUS:
            val = s->irq_status;
            break;

        case VGPU_REG_REQUEST_ID:
            val = s->request_id;
            break;

        case VGPU_REG_TIMESTAMP_LO:
            val = s->timestamp_lo;
            break;

        case VGPU_REG_TIMESTAMP_HI:
            val = s->timestamp_hi;
            break;

        case VGPU_REG_SCRATCH:
            val = s->scratch;
            break;

        default:
            val = 0;
            break;
        }
    }

    /* --- CUDA control registers (0x080 – 0x0FF) --------------- */
    /* IMPORTANT: this check must come BEFORE the request-buffer check
     * (0x040-0x43F) because the CUDA register block overlaps that range.
     * The CUDA path takes precedence over the legacy request buffer. */
    else if (addr >= VGPU_REG_CUDA_OP && addr < VGPU_CUDA_CTRL_END) {
        switch (addr) {
        case VGPU_REG_CUDA_OP:        val = s->cuda_op; break;
        case VGPU_REG_CUDA_SEQ:       val = s->cuda_seq; break;
        case VGPU_REG_CUDA_NUM_ARGS:  val = s->cuda_num_args; break;
        case VGPU_REG_CUDA_DATA_LEN:  val = s->cuda_data_len; break;
        case VGPU_REG_CUDA_DOORBELL:  val = 0; break;
        case VGPU_REG_CUDA_RESULT_STATUS:   val = s->cuda_result_status; break;
        case VGPU_REG_CUDA_RESULT_NUM:      val = s->cuda_result_num; break;
        case VGPU_REG_CUDA_RESULT_DATA_LEN: val = s->cuda_result_data_len; break;
        default:
            /* CUDA args registers */
            if (addr >= VGPU_REG_CUDA_ARGS_BASE &&
                addr < VGPU_REG_CUDA_ARGS_END) {
                uint32_t idx = (addr - VGPU_REG_CUDA_ARGS_BASE) / 4;
                if (idx < VGPU_CUDA_MAX_ARGS)
                    val = s->cuda_args[idx];
            }
            break;
        }
    }

    /* --- CUDA request data region (0x100-0x4FF) --------------- */
    /* Also before the legacy req buffer so it takes precedence. */
    else if (addr >= VGPU_CUDA_REQ_DATA_OFFSET &&
             addr < VGPU_CUDA_REQ_DATA_OFFSET + VGPU_CUDA_SMALL_DATA_MAX) {
        uint32_t off = addr - VGPU_CUDA_REQ_DATA_OFFSET;
        if (off + 4 <= VGPU_CUDA_SMALL_DATA_MAX) {
            memcpy(&val, &s->cuda_req_data[off], 4);
        }
    }

    /* --- Request buffer (0x040-0x43F) - legacy, lower priority -- */
    else if (addr >= VGPU_REQ_BUFFER_OFFSET &&
             addr < VGPU_REQ_BUFFER_OFFSET + VGPU_REQ_BUFFER_SIZE) {
        uint32_t off = addr - VGPU_REQ_BUFFER_OFFSET;
        if (off + 4 <= VGPU_REQ_BUFFER_SIZE) {
            memcpy(&val, &s->req_buf[off], 4);
        }
    }

    /* --- CUDA response data region (0x500-0x8FF) -------------- */
    /* Before response buffer for same reason. */
    else if (addr >= VGPU_CUDA_RESP_DATA_OFFSET &&
             addr < VGPU_CUDA_RESP_DATA_OFFSET + VGPU_CUDA_SMALL_DATA_MAX) {
        uint32_t off = addr - VGPU_CUDA_RESP_DATA_OFFSET;
        if (off + 4 <= VGPU_CUDA_SMALL_DATA_MAX) {
            memcpy(&val, &s->cuda_resp_data[off], 4);
        }
    }

    /* --- Response buffer (0x440-0x83F) - legacy, lower priority - */
    else if (addr >= VGPU_RESP_BUFFER_OFFSET &&
             addr < VGPU_RESP_BUFFER_OFFSET + VGPU_RESP_BUFFER_SIZE) {
        uint32_t off = addr - VGPU_RESP_BUFFER_OFFSET;
        if (off + 4 <= VGPU_RESP_BUFFER_SIZE) {
            memcpy(&val, &s->resp_buf[off], 4);
        }
    }

    /* --- CUDA result values (0x900-0x93F) --------------------- */
    else if (addr >= VGPU_REG_CUDA_RESULT_BASE &&
             addr < VGPU_REG_CUDA_RESULT_BASE + 64) {
        uint32_t off = addr - VGPU_REG_CUDA_RESULT_BASE;
        uint32_t idx = off / 8;
        if (idx < 8) {
            if (off % 8 == 0) {
                val = (uint32_t)(s->cuda_results[idx] & 0xFFFFFFFF);
            } else {
                val = (uint32_t)(s->cuda_results[idx] >> 32);
            }
        }
    }

    /* --- Shared-memory staging registers (0x940-0x94C) --------- */
    else if (addr >= VGPU_REG_SHMEM_GPA_LO && addr <= VGPU_REG_SHMEM_CTRL) {
        switch (addr) {
        case VGPU_REG_SHMEM_GPA_LO: val = s->shmem_gpa_lo;   break;
        case VGPU_REG_SHMEM_GPA_HI: val = s->shmem_gpa_hi;   break;
        case VGPU_REG_SHMEM_SIZE:   val = s->shmem_size_reg;  break;
        case VGPU_REG_SHMEM_CTRL:   val = s->shmem_active ? 1u : 0u; break;
        default: val = 0; break;
        }
    }

    /* --- Reserved / unmapped reads as 0 ----------------------- */

    return val;
}

static void vgpu_mmio_write(void *opaque, hwaddr addr,
                            uint64_t val, unsigned size)
{
    VGPUStubState *s = opaque;

    /* --- Control registers ------------------------------------ */
    if (addr < VGPU_CTRL_REG_END) {
        switch (addr) {

        case VGPU_REG_DOORBELL:
            if (val == 1) {
                vgpu_process_doorbell(s);
            }
            break;

        case VGPU_REG_REQUEST_LEN:
            s->request_len = (uint32_t)val;
            break;

        case VGPU_REG_IRQ_CTRL:
            s->irq_ctrl = (uint32_t)val & 0x01;
            break;

        case VGPU_REG_IRQ_STATUS:
            /* Write-1-to-clear */
            s->irq_status &= ~((uint32_t)val & 0x01);
            break;

        case VGPU_REG_REQUEST_ID:
            s->request_id = (uint32_t)val;
            break;

        case VGPU_REG_SCRATCH:
            s->scratch = (uint32_t)val;
            break;

        default:
            /* Other registers are read-only; silently ignore */
            break;
        }
    }

    /* --- CUDA control registers (0x080 – 0x0FF) --------------- */
    /* IMPORTANT: checked BEFORE the legacy request buffer (0x040-0x43F)
     * because the CUDA register block falls within that address range.
     * Without this ordering, the CUDA doorbell at 0x0A8 would silently
     * land in req_buf and vgpu_process_cuda_doorbell would never fire. */
    else if (addr >= VGPU_REG_CUDA_OP && addr < VGPU_CUDA_CTRL_END) {
        switch (addr) {
        case VGPU_REG_CUDA_OP:
            s->cuda_op = (uint32_t)val;
            break;
        case VGPU_REG_CUDA_SEQ:
            s->cuda_seq = (uint32_t)val;
            break;
        case VGPU_REG_CUDA_NUM_ARGS:
            s->cuda_num_args = (uint32_t)val;
            break;
        case VGPU_REG_CUDA_DATA_LEN:
            s->cuda_data_len = (uint32_t)val;
            break;
        case VGPU_REG_CUDA_DOORBELL:
            if (val == 1) {
                fprintf(stderr, "[vgpu] vm_id=%u: CUDA DOORBELL RING: call_id=0x%04x seq=%u (addr=0x%lx)\n",
                        s->vm_id, s->cuda_op, s->cuda_seq, (unsigned long)addr);
                fflush(stderr);
                vgpu_process_cuda_doorbell(s);
            }
            break;
        default:
            /* CUDA args registers */
            if (addr >= VGPU_REG_CUDA_ARGS_BASE &&
                addr < VGPU_REG_CUDA_ARGS_END) {
                uint32_t idx = (addr - VGPU_REG_CUDA_ARGS_BASE) / 4;
                if (idx < VGPU_CUDA_MAX_ARGS)
                    s->cuda_args[idx] = (uint32_t)val;
            }
            break;
        }
    }

    /* --- CUDA request data region (0x100-0x4FF) --------------- */
    /* Checked before legacy request buffer for same reason. */
    else if (addr >= VGPU_CUDA_REQ_DATA_OFFSET &&
             addr < VGPU_CUDA_REQ_DATA_OFFSET + VGPU_CUDA_SMALL_DATA_MAX) {
        uint32_t off = addr - VGPU_CUDA_REQ_DATA_OFFSET;
        if (off + 4 <= VGPU_CUDA_SMALL_DATA_MAX) {
            uint32_t v32 = (uint32_t)val;
            memcpy(&s->cuda_req_data[off], &v32, 4);
        }
    }

    /* --- Request buffer (0x040-0x43F) - legacy, lower priority -- */
    else if (addr >= VGPU_REQ_BUFFER_OFFSET &&
             addr < VGPU_REQ_BUFFER_OFFSET + VGPU_REQ_BUFFER_SIZE) {
        uint32_t off = addr - VGPU_REQ_BUFFER_OFFSET;
        if (off + 4 <= VGPU_REQ_BUFFER_SIZE) {
            uint32_t v32 = (uint32_t)val;
            memcpy(&s->req_buf[off], &v32, 4);
        }
    }

    /* --- Shared-memory registration registers (0x940 – 0x94C) - */
    else if (addr >= VGPU_REG_SHMEM_GPA_LO &&
             addr <= VGPU_REG_SHMEM_CTRL) {
        switch (addr) {
        case VGPU_REG_SHMEM_GPA_LO:
            s->shmem_gpa_lo   = (uint32_t)val;
            break;
        case VGPU_REG_SHMEM_GPA_HI:
            s->shmem_gpa_hi   = (uint32_t)val;
            break;
        case VGPU_REG_SHMEM_SIZE:
            s->shmem_size_reg = (uint32_t)val;
            break;
        case VGPU_REG_SHMEM_CTRL:
            if ((uint32_t)val == 1) {
                /* Register shared-memory region */
                hwaddr   gpa  = ((hwaddr)s->shmem_gpa_hi << 32) |
                                 (hwaddr)s->shmem_gpa_lo;
                uint32_t size = s->shmem_size_reg;

                if (size < VGPU_SHMEM_MIN_SIZE) {
                    fprintf(stderr, "[vgpu] vm_id=%u: shmem too small (%u B)\n",
                            s->vm_id, size);
                    s->status_reg = VGPU_STATUS_ERROR;
                    s->error_code = VGPU_ERR_INVALID_LENGTH;
                    break;
                }

                /* Unmap any previous mapping */
                if (s->shmem_active) {
                    if (s->shmem_g2h)
                        cpu_physical_memory_unmap(s->shmem_g2h,
                                                  s->shmem_size / 2,
                                                  false,
                                                  s->shmem_size / 2);
                    if (s->shmem_h2g)
                        cpu_physical_memory_unmap(s->shmem_h2g,
                                                  s->shmem_size / 2,
                                                  true,
                                                  s->shmem_size / 2);
                    s->shmem_g2h   = NULL;
                    s->shmem_h2g   = NULL;
                    s->shmem_active = 0;

                    /* Also free the legacy BAR1 buffer now that shared
                     * memory takes over the data path */
                    if (s->bar1_data) {
                        g_free(s->bar1_data);
                        s->bar1_data = NULL;
                    }
                }

                /* Map G2H (read-only from device perspective) */
                hwaddr g2h_len = size / 2;
                void *g2h = cpu_physical_memory_map(gpa, &g2h_len, false);
                if (!g2h || g2h_len < size / 2) {
                    fprintf(stderr, "[vgpu] vm_id=%u: cpu_physical_memory_map"
                            "(G2H) failed (gpa=0x%llx len=%u)\n",
                            s->vm_id,
                            (unsigned long long)gpa,
                            size / 2);
                    if (g2h)
                        cpu_physical_memory_unmap(g2h, g2h_len, false, g2h_len);
                    s->status_reg = VGPU_STATUS_ERROR;
                    s->error_code = VGPU_ERR_INVALID_REQUEST;
                    break;
                }

                /* Map H2G (writable — device writes results here) */
                hwaddr h2g_len = size / 2;
                void *h2g = cpu_physical_memory_map(gpa + size / 2, &h2g_len, true);
                if (!h2g || h2g_len < size / 2) {
                    fprintf(stderr, "[vgpu] vm_id=%u: cpu_physical_memory_map"
                            "(H2G) failed (gpa=0x%llx len=%u)\n",
                            s->vm_id,
                            (unsigned long long)(gpa + size / 2),
                            size / 2);
                    cpu_physical_memory_unmap(g2h, size / 2, false, size / 2);
                    if (h2g)
                        cpu_physical_memory_unmap(h2g, h2g_len, true, h2g_len);
                    s->status_reg = VGPU_STATUS_ERROR;
                    s->error_code = VGPU_ERR_INVALID_REQUEST;
                    break;
                }

                s->shmem_gpa    = gpa;
                s->shmem_size   = size;
                s->shmem_g2h    = g2h;
                s->shmem_h2g    = h2g;
                s->shmem_active = 1;

                fprintf(stderr, "[vgpu] vm_id=%u: shmem registered "
                        "gpa=0x%llx size=%u MB "
                        "(G2H host_ptr=%p H2G host_ptr=%p)\n",
                        s->vm_id,
                        (unsigned long long)gpa,
                        size >> 20,
                        g2h, h2g);

                s->status_reg = VGPU_STATUS_DONE;
                s->error_code = VGPU_ERR_NONE;

            } else if ((uint32_t)val == 0) {
                /* Unregister */
                if (s->shmem_active) {
                    if (s->shmem_g2h)
                        cpu_physical_memory_unmap(s->shmem_g2h,
                                                  s->shmem_size / 2,
                                                  false,
                                                  s->shmem_size / 2);
                    if (s->shmem_h2g)
                        cpu_physical_memory_unmap(s->shmem_h2g,
                                                  s->shmem_size / 2,
                                                  true,
                                                  s->shmem_size / 2);
                    s->shmem_g2h    = NULL;
                    s->shmem_h2g    = NULL;
                    s->shmem_active = 0;
                    fprintf(stderr, "[vgpu] vm_id=%u: shmem released\n",
                            s->vm_id);
                }
                s->status_reg = VGPU_STATUS_DONE;
            }
            break;
        default:
            break;
        }
    }

    /* Response buffer and reserved region: writes ignored */
}

static const MemoryRegionOps vgpu_mmio_ops = {
    .read  = vgpu_mmio_read,
    .write = vgpu_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

static void vgpu_process_doorbell(VGPUStubState *s)
{
    VGPUSocketHeader hdr;
    struct iovec iov[2];
    struct msghdr msg;
    ssize_t sent;

    if (s->request_len == 0) {
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_INVALID_LENGTH;
        return;
    }
    if (s->request_len > VGPU_REQ_BUFFER_SIZE) {
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_REQUEST_TOO_LARGE;
        return;
    }

    s->status_reg = VGPU_STATUS_BUSY;
    s->error_code = VGPU_ERR_NONE;
    s->response_len = 0;

    /* Detect stale connection (same logic as in vgpu_process_cuda_doorbell) */
    if (s->mediator_fd >= 0) {
        char peek_buf[1];
        ssize_t pk = recv(s->mediator_fd, peek_buf, 1,
                          MSG_PEEK | MSG_DONTWAIT);
        if (pk == 0 || (pk < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
            fprintf(stderr,
                    "[vgpu] vm_id=%u: stale mediator connection detected "
                    "(peek: pk=%zd errno=%d:%s) — reconnecting\n",
                    s->vm_id, pk, (pk < 0 ? errno : 0),
                    (pk < 0 ? strerror(errno) : "EOF"));
            qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
            close(s->mediator_fd);
            s->mediator_fd = -1;
        }
    }

    /* If socket is not connected, try to connect now */
    if (s->mediator_fd < 0) {
        vgpu_try_connect_mediator(s);
    }

    /* Still not connected? → error */
    if (s->mediator_fd < 0) {
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_MEDIATOR_UNAVAIL;
        return;
    }

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = VGPU_SOCKET_MAGIC;
    hdr.msg_type    = VGPU_MSG_REQUEST;
    hdr.vm_id       = s->vm_id;
    hdr.request_id  = s->request_id;
    hdr.pool_id     = (s->pool_id && s->pool_id[0]) ? s->pool_id[0] : 'A';
    hdr.priority    = (uint8_t)vgpu_priority_to_int(s->priority);
    hdr.payload_len = (uint32_t)s->request_len;

    /* Send header + payload in one sendmsg() call */
    iov[0].iov_base = &hdr;
    iov[0].iov_len  = VGPU_SOCKET_HDR_SIZE;
    iov[1].iov_base = s->req_buf;
    iov[1].iov_len  = s->request_len;

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov    = iov;
    msg.msg_iovlen = 2;

    sent = sendmsg(s->mediator_fd, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
        fprintf(stderr, "[vgpu] sendmsg failed: %s\n", strerror(errno));
        /* Socket broken - close and mark error */
        qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
        close(s->mediator_fd);
        s->mediator_fd = -1;
        s->status_reg  = VGPU_STATUS_ERROR;
        s->error_code  = VGPU_ERR_MEDIATOR_UNAVAIL;
        return;
    }
}

static void vgpu_process_cuda_doorbell(VGPUStubState *s)
{
    CUDACallHeader cuda_hdr;
    VGPUSocketHeader sock_hdr;
    struct iovec iov[3];
    struct msghdr msg;
    ssize_t sent;
    int iov_cnt = 0;
    uint32_t data_len = s->cuda_data_len;
    uint8_t *data_ptr = NULL;

    fprintf(stderr, "[vgpu] vm_id=%u: PROCESSING CUDA DOORBELL: call_id=0x%04x seq=%u args=%u data_len=%u\n",
            s->vm_id, s->cuda_op, s->cuda_seq, s->cuda_num_args, data_len);
    fflush(stderr);
    s->status_reg = VGPU_STATUS_BUSY;
    s->error_code = VGPU_ERR_NONE;

    /* Clear result registers */
    s->cuda_result_status   = 0;
    s->cuda_result_num      = 0;
    s->cuda_result_data_len = 0;
    memset(s->cuda_results, 0, sizeof(s->cuda_results));
    memset(s->cuda_resp_data, 0, sizeof(s->cuda_resp_data));

    /* Detect a stale socket connection: if the mediator was restarted, its
     * side of the accepted connection is already closed, but QEMU's event
     * loop may not have processed the EOF read-event yet.  A non-blocking
     * MSG_PEEK detects this race without removing any data:
     *   recv == 0  → remote end sent EOF (mediator exited / restarted)
     *   recv <  0 with EAGAIN/EWOULDBLOCK → socket alive, no data (good)
     *   recv <  0 with other errno → broken pipe / reset — stale
     * Reset mediator_fd to -1 so we reconnect immediately below. */
    if (s->mediator_fd >= 0) {
        char peek_buf[1];
        ssize_t pk = recv(s->mediator_fd, peek_buf, 1,
                          MSG_PEEK | MSG_DONTWAIT);
        if (pk == 0 || (pk < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
            fprintf(stderr,
                    "[vgpu] vm_id=%u: stale mediator connection detected "
                    "(peek: pk=%zd errno=%d:%s) — reconnecting\n",
                    s->vm_id, pk, (pk < 0 ? errno : 0),
                    (pk < 0 ? strerror(errno) : "EOF"));
            qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
            close(s->mediator_fd);
            s->mediator_fd = -1;
        }
    }

    /* Connect to mediator if needed */
    if (s->mediator_fd < 0) {
        fprintf(stderr, "[vgpu] vm_id=%u: Connecting to mediator for CUDA call 0x%04x\n",
                s->vm_id, s->cuda_op);
        fflush(stderr);
        vgpu_try_connect_mediator(s);
    }
    if (s->mediator_fd < 0) {
        fprintf(stderr, "[vgpu] vm_id=%u: ERROR: Cannot connect to mediator (call_id=0x%04x)\n",
                s->vm_id, s->cuda_op);
        fflush(stderr);
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_MEDIATOR_UNAVAIL;
        return;
    }

    memset(&cuda_hdr, 0, sizeof(cuda_hdr));
    cuda_hdr.magic    = VGPU_SOCKET_MAGIC;
    cuda_hdr.call_id  = s->cuda_op;
    cuda_hdr.seq_num  = s->cuda_seq;
    cuda_hdr.vm_id    = s->vm_id;
    cuda_hdr.num_args = s->cuda_num_args;
    cuda_hdr.data_len = data_len;
    memcpy(cuda_hdr.args, s->cuda_args,
           s->cuda_num_args * sizeof(uint32_t));

    /* Determine data source */
    if (data_len > 0) {
        if (data_len <= VGPU_CUDA_SMALL_DATA_MAX) {
            /* Small data from BAR0 inline region */
            data_ptr = s->cuda_req_data;
        } else if (s->shmem_active && s->shmem_g2h) {
            /* VHOST-style shared memory: host ptr directly into guest RAM */
            data_ptr = s->shmem_g2h;
            if (data_len > (uint32_t)(s->shmem_size / 2))
                data_len = (uint32_t)(s->shmem_size / 2);
        } else if (s->bar1_data) {
            /* Legacy BAR1 fallback */
            data_ptr = s->bar1_data + VGPU_BAR1_G2H_OFFSET;
            if (data_len > VGPU_BAR1_G2H_SIZE)
                data_len = VGPU_BAR1_G2H_SIZE;
        } else {
            /* No large data path available */
            s->status_reg = VGPU_STATUS_ERROR;
            s->error_code = VGPU_ERR_REQUEST_TOO_LARGE;
            return;
        }
    }

    memset(&sock_hdr, 0, sizeof(sock_hdr));
    sock_hdr.magic       = VGPU_SOCKET_MAGIC;
    sock_hdr.msg_type    = VGPU_MSG_CUDA_CALL;
    sock_hdr.vm_id       = s->vm_id;
    sock_hdr.request_id  = s->cuda_seq;
    sock_hdr.pool_id     = (s->pool_id && s->pool_id[0]) ? s->pool_id[0] : 'A';
    sock_hdr.priority    = (uint8_t)vgpu_priority_to_int(s->priority);
    sock_hdr.payload_len = (uint32_t)(sizeof(CUDACallHeader) + data_len);

    /* Assemble iov */
    iov[0].iov_base = &sock_hdr;
    iov[0].iov_len  = VGPU_SOCKET_HDR_SIZE;
    iov_cnt++;

    iov[1].iov_base = &cuda_hdr;
    iov[1].iov_len  = sizeof(CUDACallHeader);
    iov_cnt++;

    if (data_len > 0 && data_ptr) {
        iov[2].iov_base = data_ptr;
        iov[2].iov_len  = data_len;
        iov_cnt++;
    }

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov    = iov;
    msg.msg_iovlen = iov_cnt;

    fprintf(stderr, "[vgpu] vm_id=%u: SENDING CUDA CALL to mediator: call_id=0x%04x seq=%u total_bytes=%zu (fd=%d)\n",
            s->vm_id, s->cuda_op, s->cuda_seq,
            (size_t)(VGPU_SOCKET_HDR_SIZE + sizeof(CUDACallHeader) + data_len),
            s->mediator_fd);
    fflush(stderr);

    sent = sendmsg(s->mediator_fd, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
        fprintf(stderr, "[vgpu] vm_id=%u: CUDA sendmsg failed: %s (call_id=0x%04x)\n",
                s->vm_id, strerror(errno), s->cuda_op);
        fflush(stderr);
        qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
        close(s->mediator_fd);
        s->mediator_fd = -1;
        s->status_reg  = VGPU_STATUS_ERROR;
        s->error_code  = VGPU_ERR_MEDIATOR_UNAVAIL;
        return;
    }

    fprintf(stderr, "[vgpu] vm_id=%u: CUDA CALL SENT to mediator: %zd bytes (call_id=0x%04x seq=%u)\n",
            s->vm_id, sent, s->cuda_op, s->cuda_seq);
    fflush(stderr);

    /* STATUS stays BUSY until mediator responds */
}

static void vgpu_try_connect_mediator(VGPUStubState *s)
{
    struct sockaddr_un addr;
    int fd;
    struct stat st;

    if (s->mediator_fd >= 0) {
        return;  /* already connected */
    }

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "[vgpu] vm_id=%u: socket() failed: %s (errno=%d)\n",
                s->vm_id, strerror(errno), errno);
        return;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, VGPU_SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (stat(VGPU_SOCKET_PATH, &st) != 0) {
        fprintf(stderr, "[vgpu] vm_id=%u: Socket %s does not exist yet. "
                "Mediator may not be running or socket not created yet.\n",
                s->vm_id, VGPU_SOCKET_PATH);
    } else {
        mode_t mode = st.st_mode & 0777;
        if ((mode & 0666) != 0666) {
            fprintf(stderr, "[vgpu] vm_id=%u: Socket %s has permissions %03o, "
                    "may need 0666\n",
                    s->vm_id, VGPU_SOCKET_PATH, mode);
        }
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[vgpu] vm_id=%u: connect() to %s failed: %s (errno=%d)\n",
                s->vm_id, VGPU_SOCKET_PATH, strerror(errno), errno);
        close(fd);
        return;
    }

    if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) < 0) {
        fprintf(stderr, "[vgpu] vm_id=%u: fcntl(O_NONBLOCK) failed: %s\n",
                s->vm_id, strerror(errno));
        close(fd);
        return;
    }

    s->mediator_fd = fd;
    s->sock_rx_len = 0;

    qemu_set_fd_handler(fd, vgpu_socket_read_handler, NULL, s);

    fprintf(stderr, "[vgpu] vm_id=%u: Connected to mediator at %s (fd=%d)\n",
            s->vm_id, VGPU_SOCKET_PATH, fd);
}

static void vgpu_socket_read_handler(void *opaque)
{
    VGPUStubState *s = opaque;
    ssize_t n;
    VGPUSocketHeader *hdr;
    uint32_t total_len;

    n = read(s->mediator_fd,
             s->sock_rx_buf + s->sock_rx_len,
             s->sock_rx_cap - s->sock_rx_len);

    if (n <= 0) {
        if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
            fprintf(stderr, "[vgpu] vm_id=%u: mediator socket closed (n=%zd, errno=%d: %s)\n",
                    s->vm_id, n, errno, (errno != 0) ? strerror(errno) : "EOF");
            qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
            close(s->mediator_fd);
            s->mediator_fd = -1;

            if (s->status_reg == VGPU_STATUS_BUSY) {
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_MEDIATOR_UNAVAIL;
            }
        }
        return;
    }

    s->sock_rx_len += (uint32_t)n;

    if (s->sock_rx_len < VGPU_SOCKET_HDR_SIZE)
        return;

    hdr = (VGPUSocketHeader *)s->sock_rx_buf;

    /* Sanity check */
    if (hdr->magic != VGPU_SOCKET_MAGIC) {
        fprintf(stderr, "[vgpu] bad magic 0x%08x, dropping\n", hdr->magic);
        s->sock_rx_len = 0;
        return;
    }

    total_len = VGPU_SOCKET_HDR_SIZE + hdr->payload_len;

    /* Do we have the full message? */
    if (s->sock_rx_len < total_len) {
        return;  /* need more data */
    }

    /* ---- Process complete message ---- */

    if (hdr->msg_type == VGPU_MSG_RESPONSE) {
        uint32_t copy_len = hdr->payload_len;
        if (copy_len > VGPU_RESP_BUFFER_SIZE) {
            copy_len = VGPU_RESP_BUFFER_SIZE;
        }

        memset(s->resp_buf, 0, VGPU_RESP_BUFFER_SIZE);
        memcpy(s->resp_buf,
               s->sock_rx_buf + VGPU_SOCKET_HDR_SIZE,
               copy_len);

        s->response_len = copy_len;

        int64_t now_us = qemu_clock_get_us(QEMU_CLOCK_VIRTUAL);
        s->timestamp_lo = (uint32_t)(now_us & 0xFFFFFFFF);
        s->timestamp_hi = (uint32_t)((uint64_t)now_us >> 32);

        /* Check response status - map mediator error codes to MMIO codes */
        if (copy_len >= VGPU_RESPONSE_HEADER_SIZE) {
            VGPUResponse *resp = (VGPUResponse *)s->resp_buf;
            if (resp->status == 0) {
                s->status_reg = VGPU_STATUS_DONE;
                s->error_code = VGPU_ERR_NONE;
            } else if (resp->status == VGPU_ERR_RATE_LIMITED) {
                /* Phase 3: back-pressure - VM exceeded its rate limit */
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_RATE_LIMITED;
                fprintf(stderr,
                        "[vgpu] vm%u req%u: rate-limited by mediator\n",
                        s->vm_id, s->request_id);
            } else if (resp->status == VGPU_ERR_VM_QUARANTINED) {
                /* Phase 3: VM quarantined due to excessive faults */
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_VM_QUARANTINED;
                fprintf(stderr,
                        "[vgpu] vm%u req%u: VM quarantined\n",
                        s->vm_id, s->request_id);
            } else if (resp->status == VGPU_ERR_QUEUE_FULL) {
                /* Queue depth exceeded */
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_QUEUE_FULL;
                fprintf(stderr,
                        "[vgpu] vm%u req%u: queue full\n",
                        s->vm_id, s->request_id);
            } else {
                /* Generic CUDA or other error */
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_CUDA_ERROR;
            }
        } else {
            s->status_reg = VGPU_STATUS_DONE;
            s->error_code = VGPU_ERR_NONE;
        }

    }
    else if (hdr->msg_type == VGPU_MSG_BUSY) {
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_RATE_LIMITED;
        s->response_len = 0;
        fprintf(stderr,
                "[vgpu] vm%u req%u: BUSY (rate-limited)\n",
                s->vm_id, s->request_id);
    }
    else if (hdr->msg_type == VGPU_MSG_QUARANTINED) {
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_VM_QUARANTINED;
        s->response_len = 0;
        fprintf(stderr,
                "[vgpu] vm%u req%u: VM QUARANTINED\n",
                s->vm_id, s->request_id);
    }
    else if (hdr->msg_type == VGPU_MSG_CUDA_RESULT) {
        /* ---- CUDA API call result ---- */
        uint8_t *payload = s->sock_rx_buf + VGPU_SOCKET_HDR_SIZE;
        uint32_t plen = hdr->payload_len;

        if (plen >= sizeof(CUDACallResult)) {
            CUDACallResult *cr = (CUDACallResult *)payload;

            s->cuda_result_status   = cr->status;
            s->cuda_result_num      = cr->num_results;
            s->cuda_result_data_len = cr->data_len;

            uint32_t nr = cr->num_results;
            if (nr > 8) nr = 8;
            memcpy(s->cuda_results, cr->results, nr * sizeof(uint64_t));

            if (cr->data_len > 0) {
                uint8_t *rdata = payload + sizeof(CUDACallResult);
                uint32_t rdata_avail = plen - sizeof(CUDACallResult);
                uint32_t copy_len = cr->data_len;
                if (copy_len > rdata_avail) copy_len = rdata_avail;

                if (copy_len <= VGPU_CUDA_SMALL_DATA_MAX) {
                    memcpy(s->cuda_resp_data, rdata, copy_len);
                } else if (s->shmem_active && s->shmem_h2g) {
                    uint32_t shmem_copy = copy_len;
                    uint32_t h2g_cap = (uint32_t)(s->shmem_size / 2);
                    if (shmem_copy > h2g_cap)
                        shmem_copy = h2g_cap;
                    memcpy(s->shmem_h2g, rdata, shmem_copy);
                } else if (s->bar1_data) {
                    uint32_t bar1_copy = copy_len;
                    if (bar1_copy > VGPU_BAR1_H2G_SIZE)
                        bar1_copy = VGPU_BAR1_H2G_SIZE;
                    memcpy(s->bar1_data + VGPU_BAR1_H2G_OFFSET,
                           rdata, bar1_copy);
                }
            }

            int64_t now_us = qemu_clock_get_us(QEMU_CLOCK_VIRTUAL);
            s->timestamp_lo = (uint32_t)(now_us & 0xFFFFFFFF);
            s->timestamp_hi = (uint32_t)((uint64_t)now_us >> 32);

            if (cr->status == 0) {
                s->status_reg = VGPU_STATUS_DONE;
                s->error_code = VGPU_ERR_NONE;
            } else {
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_CUDA_ERROR;
            }
        } else {
            s->status_reg = VGPU_STATUS_ERROR;
            s->error_code = VGPU_ERR_INVALID_REQUEST;
        }
    }
    else if (hdr->msg_type == VGPU_MSG_PING) {
        VGPUSocketHeader pong;
        memset(&pong, 0, sizeof(pong));
        pong.magic    = VGPU_SOCKET_MAGIC;
        pong.msg_type = VGPU_MSG_PONG;
        pong.vm_id    = s->vm_id;
        write(s->mediator_fd, &pong, VGPU_SOCKET_HDR_SIZE);
    }
    else if (hdr->msg_type == VGPU_MSG_PONG) {
    }

    if (s->sock_rx_len > total_len) {
        memmove(s->sock_rx_buf,
                s->sock_rx_buf + total_len,
                s->sock_rx_len - total_len);
        s->sock_rx_len -= total_len;
    } else {
        s->sock_rx_len = 0;
    }
}


static uint64_t vgpu_bar1_read(void *opaque, hwaddr addr, unsigned size)
{
    VGPUStubState *s = opaque;
    uint64_t val = 0;

    if (s->bar1_data && addr + size <= VGPU_BAR1_SIZE) {
        memcpy(&val, &s->bar1_data[addr], size);
    }
    return val;
}

static void vgpu_bar1_write(void *opaque, hwaddr addr,
                             uint64_t val, unsigned size)
{
    VGPUStubState *s = opaque;

    if (s->bar1_data && addr + size <= VGPU_BAR1_SIZE) {
        memcpy(&s->bar1_data[addr], &val, size);
    }
}

static const MemoryRegionOps vgpu_bar1_ops = {
    .read  = vgpu_bar1_read,
    .write = vgpu_bar1_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static void vgpu_realize(PCIDevice *pci_dev, Error **errp)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);

    /* Interrupt pin A (for future MSI support) */
    pci_dev->config[PCI_INTERRUPT_PIN] = 1;

    /* Register BAR0 - 4 KB MMIO */
    memory_region_init_io(&s->mmio, OBJECT(s), &vgpu_mmio_ops, s,
                          "vgpu-mmio", VGPU_BAR_SIZE);
    pci_register_bar(pci_dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);

    /* Register BAR1 - 16 MB data region */
    s->bar1_data = g_malloc0(VGPU_BAR1_SIZE);
    memory_region_init_io(&s->mmio_bar1, OBJECT(s), &vgpu_bar1_ops, s,
                          "vgpu-bar1", VGPU_BAR1_SIZE);
    pci_register_bar(pci_dev, 1, PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->mmio_bar1);

    /* Initialise control registers */
    s->status_reg   = VGPU_STATUS_IDLE;
    s->error_code   = VGPU_ERR_NONE;
    s->request_len  = 0;
    s->response_len = 0;
    s->irq_ctrl     = 0;
    s->irq_status   = 0;
    s->request_id   = 0;
    s->timestamp_lo = 0;
    s->timestamp_hi = 0;
    s->scratch      = 0;

    /* Clear buffers */
    memset(s->req_buf,  0, VGPU_REQ_BUFFER_SIZE);
    memset(s->resp_buf, 0, VGPU_RESP_BUFFER_SIZE);

    /* Clear CUDA state */
    s->cuda_op       = 0;
    s->cuda_seq      = 0;
    s->cuda_num_args = 0;
    s->cuda_data_len = 0;
    memset(s->cuda_args, 0, sizeof(s->cuda_args));
    s->cuda_result_status   = 0;
    s->cuda_result_num      = 0;
    s->cuda_result_data_len = 0;
    memset(s->cuda_results, 0, sizeof(s->cuda_results));
    memset(s->cuda_req_data, 0, sizeof(s->cuda_req_data));
    memset(s->cuda_resp_data, 0, sizeof(s->cuda_resp_data));

    /* Shared-memory state — cleared until guest registers a region */
    s->shmem_g2h    = NULL;
    s->shmem_h2g    = NULL;
    s->shmem_gpa    = 0;
    s->shmem_size   = 0;
    s->shmem_active = 0;
    s->shmem_gpa_lo = 0;
    s->shmem_gpa_hi = 0;
    s->shmem_size_reg = 0;

    /* Socket not yet connected */
    s->mediator_fd  = -1;
    s->sock_rx_buf  = g_malloc(SOCK_RX_DEFAULT_CAP);
    s->sock_rx_len  = 0;
    s->sock_rx_cap  = SOCK_RX_DEFAULT_CAP;

    if (!s->pool_id || strlen(s->pool_id) == 0) {
        g_free(s->pool_id);
        s->pool_id = g_strdup("A");
    } else {
        if (strcmp(s->pool_id, "A") != 0 && strcmp(s->pool_id, "B") != 0) {
            error_setg(errp,
                       "vgpu: pool_id must be 'A' or 'B', got '%s'",
                       s->pool_id);
            return;
        }
    }

    if (!s->priority || strlen(s->priority) == 0) {
        g_free(s->priority);
        s->priority = g_strdup("medium");
    } else {
        if (strcmp(s->priority, "low") != 0 &&
            strcmp(s->priority, "medium") != 0 &&
            strcmp(s->priority, "high") != 0) {
            error_setg(errp,
                       "vgpu: priority must be 'low', 'medium' "
                       "or 'high', got '%s'",
                       s->priority);
            return;
        }
    }

    fprintf(stderr,
            "[vgpu] realised  vm_id=%u  pool=%s  priority=%s  rev=0x%02x\n",
            s->vm_id,
            s->pool_id  ? s->pool_id  : "A",
            s->priority ? s->priority : "medium",
            VGPU_REVISION);

    vgpu_try_connect_mediator(s);
}

static void vgpu_exit(PCIDevice *pci_dev)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);

    if (s->mediator_fd >= 0) {
        qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
        close(s->mediator_fd);
        s->mediator_fd = -1;
    }

    /* Release guest-pinned shared memory mappings */
    if (s->shmem_active) {
        if (s->shmem_g2h)
            cpu_physical_memory_unmap(s->shmem_g2h,
                                      s->shmem_size / 2,
                                      false,
                                      s->shmem_size / 2);
        if (s->shmem_h2g)
            cpu_physical_memory_unmap(s->shmem_h2g,
                                      s->shmem_size / 2,
                                      true,
                                      s->shmem_size / 2);
        s->shmem_g2h    = NULL;
        s->shmem_h2g    = NULL;
        s->shmem_active = 0;
    }

    g_free(s->pool_id);
    g_free(s->priority);
    g_free(s->bar1_data);
    g_free(s->sock_rx_buf);

    fprintf(stderr, "[vgpu] destroyed  vm_id=%u\n", s->vm_id);
}

static Property vgpu_properties[] = {
    DEFINE_PROP_STRING("pool_id",  VGPUStubState, pool_id),
    DEFINE_PROP_STRING("priority", VGPUStubState, priority),
    DEFINE_PROP_UINT32("vm_id",    VGPUStubState, vm_id, 0),
    DEFINE_PROP_END_OF_LIST(),
};

static void vgpu_class_init(ObjectClass *klass, void *data)
{
    DeviceClass    *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k  = PCI_DEVICE_CLASS(klass);

    k->realize   = vgpu_realize;
    k->exit      = vgpu_exit;
    k->vendor_id = VGPU_VENDOR_ID;
    k->device_id = VGPU_DEVICE_ID;
    k->revision  = VGPU_REVISION;
    k->class_id  = VGPU_CLASS_ID;   /* 3D controller */
    k->subsystem_vendor_id = VGPU_SUBSYS_VENDOR_ID;
    k->subsystem_id        = VGPU_SUBSYS_DEVICE_ID;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
    dc->desc  = "Virtual GPU (MMIO + BAR1 + CUDA Remoting)";
    dc->props = vgpu_properties;
}

static const TypeInfo vgpu_info = {
    .name          = TYPE_VGPU_STUB,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(VGPUStubState),
    .class_init    = vgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void vgpu_register_types(void)
{
    type_register_static(&vgpu_info);
}

type_init(vgpu_register_types)

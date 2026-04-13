/*
 * CUDA Transport — guest-side communication layer
 *
 * Provides a blocking RPC interface for the CUDA shim library.
 * Sends CUDACallHeader + payload to the VGPU-STUB device via MMIO
 * and waits for a CUDACallResult + response payload.
 *
 * Two data paths are supported:
 *   1. BAR0 control path — for small messages (< 1 KB)
 *   2. BAR1 data path   — for large bulk transfers (up to 16 MB)
 *
 * If BAR1 is not available (device does not expose it), the transport
 * falls back to chunked transfers over BAR0.
 */

#ifndef CUDA_TRANSPORT_H
#define CUDA_TRANSPORT_H

#include <stdint.h>
#include <stddef.h>
#include "cuda_protocol.h"

/* Transport handle (opaque to callers) */
typedef struct cuda_transport cuda_transport_t;

/*
 * Initialise the transport layer.
 *
 * Scans PCI for the vGPU device (vendor 0x10DE, device 0x2331,
 * class 0x0302xx) and maps its BARs.
 *
 * Returns 0 on success, -1 on failure.
 * On success, *tp is set to a valid transport handle.
 */
int cuda_transport_init(cuda_transport_t **tp);

/*
 * Shutdown the transport and release resources.
 */
void cuda_transport_destroy(cuda_transport_t *tp);

/*
 * Execute a CUDA API call (blocking RPC).
 *
 * Parameters:
 *   tp          — transport handle
 *   call_id     — CUDA_CALL_* identifier
 *   args        — array of uint32_t inline arguments
 *   num_args    — number of arguments in args[]
 *   send_data   — bulk data to send (NULL if none)
 *   send_len    — length of send_data in bytes
 *   result      — [out] filled with the CUDACallResult header
 *   recv_data   — [out] buffer for bulk return data (NULL if none expected)
 *   recv_cap    — capacity of recv_data buffer in bytes
 *   recv_len    — [out] actual length of bulk return data received
 *
 * Returns: CUresult value (0 = CUDA_SUCCESS).
 */
int cuda_transport_call(cuda_transport_t *tp,
                        uint32_t call_id,
                        const uint32_t *args, uint32_t num_args,
                        const void *send_data, uint32_t send_len,
                        CUDACallResult *result,
                        void *recv_data, uint32_t recv_cap,
                        uint32_t *recv_len);

int cuda_transport_call_internal(cuda_transport_t *tp,
                                 uint32_t call_id,
                                 const uint32_t *args, uint32_t num_args,
                                 const void *send_data, uint32_t send_len,
                                 CUDACallResult *result,
                                 void *recv_data, uint32_t recv_cap,
                                 uint32_t *recv_len)
    __attribute__((visibility("hidden")));

/*
 * Get the VM ID assigned to this transport (read from MMIO).
 */
uint32_t cuda_transport_vm_id(cuda_transport_t *tp);

/*
 * Check if the transport is connected and healthy.
 */
int cuda_transport_is_connected(cuda_transport_t *tp);

/*
 * Lightweight device scan: scan /sys for the VGPU-STUB PCI device without
 * opening resource0 or mapping any BARs.  Succeeds even inside a systemd
 * sandbox where /sys is read-only or resource0 is not yet accessible.
 *
 * Side-effect: populates the module-level g_discovered_bdf so that
 * cuda_transport_pci_bdf(NULL) returns the correct address.
 *
 * Returns 0 if found, -1 if not found.
 */
int cuda_transport_discover(void);

/*
 * Return the PCI bus:device.function string of the discovered VGPU-STUB
 * device, e.g. "0000:00:05.0".  Returns "0000:00:00.0" if unavailable.
 * The returned pointer is valid for the lifetime of the transport.
 */
const char *cuda_transport_pci_bdf(cuda_transport_t *tp);

/*
 * Write canonical error to /tmp/vgpu_last_error (for 100% error determination).
 * Call from any failure path; call_id=0 for init/connect failures.
 */
void cuda_transport_write_error(const char *code, uint32_t call_id,
                                uint32_t transport_err, const char *detail);

/*
 * Write checkpoint to /tmp/vgpu_checkpoint (for hang diagnosis).
 */
void cuda_transport_write_checkpoint(const char *phase);

/*
 * Clear debug state at start of run (call history, checkpoint trail).
 */
void cuda_transport_clear_debug_state(void);

#endif /* CUDA_TRANSPORT_H */

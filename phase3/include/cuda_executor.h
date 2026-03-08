/*
 * cuda_executor.h  —  Host-side CUDA API replay engine
 *
 * The CUDA executor maintains per-VM CUDA contexts and replays
 * CUDA API calls received from the guest shim through the mediator.
 *
 * Key data structures:
 *   - Per-VM context   : One CUcontext per connected VM
 *   - Memory map       : guest_devptr → host_devptr mapping
 *   - Module cache     : guest_module → host CUmodule mapping
 *   - Function cache   : guest_func → host CUfunction mapping
 *   - Stream map       : guest_stream → host CUstream mapping
 *   - Event map        : guest_event → host CUevent mapping
 */

#ifndef CUDA_EXECUTOR_H
#define CUDA_EXECUTOR_H

#include <stdint.h>
#include "cuda_protocol.h"

/* Opaque handle for the executor state */
typedef struct cuda_executor cuda_executor_t;

/*
 * Initialise the CUDA executor.
 *
 * Queries the physical GPU via NVML and CUDA driver API.
 * Creates internal data structures.
 *
 * Returns 0 on success, -1 on failure.
 */
int cuda_executor_init(cuda_executor_t **exec);

/*
 * Destroy the executor and release all GPU resources.
 */
void cuda_executor_destroy(cuda_executor_t *exec);

/*
 * Execute a CUDA API call.
 *
 * Parameters:
 *   exec       — executor handle
 *   call       — incoming CUDACallHeader (parsed from socket)
 *   data       — bulk data payload (may be NULL)
 *   data_len   — length of data payload
 *   result     — [out] result header to fill
 *   result_data — [out] buffer for bulk result data
 *   result_cap  — capacity of result_data buffer
 *   result_len  — [out] actual length of result data written
 *
 * Returns: CUresult (0 = success).
 */
int cuda_executor_call(cuda_executor_t *exec,
                       const CUDACallHeader *call,
                       const void *data, uint32_t data_len,
                       CUDACallResult *result,
                       void *result_data, uint32_t result_cap,
                       uint32_t *result_len);

/*
 * Get GPU info (to be sent to guest at init time).
 *
 * Queries the physical GPU and fills in the CUDAGpuInfo structure.
 */
int cuda_executor_get_gpu_info(cuda_executor_t *exec, CUDAGpuInfo *info);

/*
 * Clean up all resources for a specific VM (called on VM disconnect).
 */
void cuda_executor_cleanup_vm(cuda_executor_t *exec, uint32_t vm_id);

#endif /* CUDA_EXECUTOR_H */

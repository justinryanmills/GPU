#ifndef CUDA_EXECUTOR_H
#define CUDA_EXECUTOR_H

#include <stdint.h>
#include "cuda_protocol.h"

typedef struct cuda_executor cuda_executor_t;

int cuda_executor_init(cuda_executor_t **exec);
void cuda_executor_destroy(cuda_executor_t *exec);
int cuda_executor_call(cuda_executor_t *exec,
                       const CUDACallHeader *call,
                       const void *data, uint32_t data_len,
                       CUDACallResult *result,
                       void *result_data, uint32_t result_cap,
                       uint32_t *result_len);

int cuda_executor_get_gpu_info(cuda_executor_t *exec, CUDAGpuInfo *info);
void cuda_executor_cleanup_vm(cuda_executor_t *exec, uint32_t vm_id);

#endif /* CUDA_EXECUTOR_H */

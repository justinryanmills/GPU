/* Guest-side RPC to VGPU-STUB via MMIO (BAR0/BAR1). */
#ifndef CUDA_TRANSPORT_H
#define CUDA_TRANSPORT_H

#include <stdint.h>
#include <stddef.h>
#include "cuda_protocol.h"

typedef struct cuda_transport cuda_transport_t;

int cuda_transport_init(cuda_transport_t **tp);
void cuda_transport_destroy(cuda_transport_t *tp);
int cuda_transport_call(cuda_transport_t *tp,
                        uint32_t call_id,
                        const uint32_t *args, uint32_t num_args,
                        const void *send_data, uint32_t send_len,
                        CUDACallResult *result,
                        void *recv_data, uint32_t recv_cap,
                        uint32_t *recv_len);

uint32_t cuda_transport_vm_id(cuda_transport_t *tp);
int cuda_transport_is_connected(cuda_transport_t *tp);
int cuda_transport_discover(void);
const char *cuda_transport_pci_bdf(cuda_transport_t *tp);
void cuda_transport_write_error(const char *code, uint32_t call_id,
                                uint32_t transport_err, const char *detail);

void cuda_transport_write_checkpoint(const char *phase);
void cuda_transport_clear_debug_state(void);

#endif /* CUDA_TRANSPORT_H */

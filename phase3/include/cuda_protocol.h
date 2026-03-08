/*
 * Phase 3: CUDA API remoting protocol (guest shim <-> host executor).
 */
#ifndef CUDA_PROTOCOL_H
#define CUDA_PROTOCOL_H

#include <stdint.h>

/* ================================================================
 * CUDA API Call Identifiers
 *
 * Each CUDA Driver API function that we intercept is assigned a
 * unique call_id.  Grouped by functional category.
 * ================================================================ */

/* --- Initialisation & version ----------------------------------- */
#define CUDA_CALL_INIT                      0x0001
#define CUDA_CALL_DRIVER_GET_VERSION        0x0002

/* --- Device management ------------------------------------------ */
#define CUDA_CALL_DEVICE_GET_COUNT          0x0010
#define CUDA_CALL_DEVICE_GET                0x0011
#define CUDA_CALL_DEVICE_GET_NAME           0x0012
#define CUDA_CALL_DEVICE_GET_ATTRIBUTE      0x0013
#define CUDA_CALL_DEVICE_TOTAL_MEM          0x0014
#define CUDA_CALL_DEVICE_GET_UUID           0x0015
#define CUDA_CALL_DEVICE_COMPUTE_CAPABILITY 0x0016
#define CUDA_CALL_DEVICE_GET_PROPERTIES     0x0017
#define CUDA_CALL_DEVICE_GET_P2P_ATTRIBUTE  0x0018

/* --- Context management ----------------------------------------- */
#define CUDA_CALL_CTX_CREATE                0x0020
#define CUDA_CALL_CTX_DESTROY               0x0021
#define CUDA_CALL_CTX_SET_CURRENT           0x0022
#define CUDA_CALL_CTX_GET_CURRENT           0x0023
#define CUDA_CALL_CTX_PUSH_CURRENT          0x0024
#define CUDA_CALL_CTX_POP_CURRENT           0x0025
#define CUDA_CALL_CTX_SYNCHRONIZE           0x0026
#define CUDA_CALL_CTX_GET_DEVICE            0x0027
#define CUDA_CALL_CTX_GET_API_VERSION       0x0028

/* --- Memory management ------------------------------------------ */
#define CUDA_CALL_MEM_ALLOC                 0x0030
#define CUDA_CALL_MEM_FREE                  0x0031
#define CUDA_CALL_MEMCPY_HTOD               0x0032
#define CUDA_CALL_MEMCPY_DTOH               0x0033
#define CUDA_CALL_MEMCPY_DTOD               0x0034
#define CUDA_CALL_MEMSET_D8                  0x0035
#define CUDA_CALL_MEMSET_D16                 0x0036
#define CUDA_CALL_MEMSET_D32                 0x0037
#define CUDA_CALL_MEM_ALLOC_MANAGED         0x0038
#define CUDA_CALL_MEM_ALLOC_HOST            0x0039
#define CUDA_CALL_MEM_FREE_HOST             0x003A
#define CUDA_CALL_MEM_GET_INFO              0x003B
#define CUDA_CALL_MEMCPY_HTOD_ASYNC         0x003C
#define CUDA_CALL_MEMCPY_DTOH_ASYNC         0x003D
#define CUDA_CALL_MEMCPY_DTOD_ASYNC         0x003E

/* --- Module / function management ------------------------------- */
#define CUDA_CALL_MODULE_LOAD_DATA          0x0040
#define CUDA_CALL_MODULE_LOAD_DATA_EX       0x0041
#define CUDA_CALL_MODULE_LOAD_FAT_BINARY    0x0042
#define CUDA_CALL_MODULE_UNLOAD             0x0043
#define CUDA_CALL_MODULE_GET_FUNCTION       0x0044
#define CUDA_CALL_MODULE_GET_GLOBAL         0x0045

/* --- Kernel launch ---------------------------------------------- */
#define CUDA_CALL_LAUNCH_KERNEL             0x0050
#define CUDA_CALL_LAUNCH_COOPERATIVE_KERNEL 0x0051

/* --- Stream management ------------------------------------------ */
#define CUDA_CALL_STREAM_CREATE             0x0060
#define CUDA_CALL_STREAM_CREATE_WITH_FLAGS  0x0061
#define CUDA_CALL_STREAM_CREATE_WITH_PRIORITY 0x0062
#define CUDA_CALL_STREAM_DESTROY            0x0063
#define CUDA_CALL_STREAM_SYNCHRONIZE        0x0064
#define CUDA_CALL_STREAM_QUERY              0x0065
#define CUDA_CALL_STREAM_WAIT_EVENT         0x0066

/* --- Event management ------------------------------------------- */
#define CUDA_CALL_EVENT_CREATE              0x0070
#define CUDA_CALL_EVENT_CREATE_WITH_FLAGS   0x0071
#define CUDA_CALL_EVENT_DESTROY             0x0072
#define CUDA_CALL_EVENT_RECORD              0x0073
#define CUDA_CALL_EVENT_SYNCHRONIZE         0x0074
#define CUDA_CALL_EVENT_QUERY               0x0075
#define CUDA_CALL_EVENT_ELAPSED_TIME        0x0076

/* --- Texture / surface (stub: return NOT_SUPPORTED) ------------- */
#define CUDA_CALL_TEX_CREATE                0x0080
#define CUDA_CALL_TEX_DESTROY               0x0081

/* --- Primary context -------------------------------------------- */
#define CUDA_CALL_DEVICE_PRIMARY_CTX_RETAIN   0x0090
#define CUDA_CALL_DEVICE_PRIMARY_CTX_RELEASE  0x0091
#define CUDA_CALL_DEVICE_PRIMARY_CTX_RESET    0x0092
#define CUDA_CALL_DEVICE_PRIMARY_CTX_SET_FLAGS 0x0093
#define CUDA_CALL_DEVICE_PRIMARY_CTX_GET_STATE 0x0094

/* --- Occupancy -------------------------------------------------- */
#define CUDA_CALL_OCCUPANCY_MAX_ACTIVE_BLOCKS 0x00A0
#define CUDA_CALL_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE 0x00A1

/* --- Misc ------------------------------------------------------- */
#define CUDA_CALL_FUNC_GET_ATTRIBUTE        0x00B0
#define CUDA_CALL_FUNC_SET_CACHE_CONFIG     0x00B1
#define CUDA_CALL_GET_ERROR_STRING          0x00B2
#define CUDA_CALL_GET_ERROR_NAME            0x00B3

/* --- GPU info query (custom, host-side NVML query) -------------- */
#define CUDA_CALL_GET_GPU_INFO              0x00F0

/* Maximum call_id sentinel */
#define CUDA_CALL_MAX                       0x00FF

/* ================================================================
 * Wire-format structures
 * ================================================================ */

/* Maximum inline arguments (covers up to 8 uint64 values) */
#define CUDA_MAX_INLINE_ARGS  16   /* 16 x uint32 = 8 x uint64 */

/* Maximum inline result values */
#define CUDA_MAX_INLINE_RESULTS 8  /* 8 x uint64 */

/* CUDACallHeader: guest -> host; args[] or payload per data_len. */
typedef struct __attribute__((packed)) CUDACallHeader {
    uint32_t magic;                         /* VGPU_SOCKET_MAGIC (0x56475055) */
    uint32_t call_id;                       /* CUDA_CALL_* identifier         */
    uint32_t seq_num;                       /* Monotonic sequence number      */
    uint32_t vm_id;                         /* Originating VM identifier      */
    uint32_t num_args;                      /* Number of uint32 args used     */
    uint32_t data_len;                      /* Bytes of bulk data following   */
    uint32_t args[CUDA_MAX_INLINE_ARGS];    /* Inline arguments               */
} CUDACallHeader;

#define CUDA_CALL_HEADER_SIZE  sizeof(CUDACallHeader)

/* CUDACallResult: host -> guest; status, results[], optional payload. */
typedef struct __attribute__((packed)) CUDACallResult {
    uint32_t magic;                         /* VGPU_SOCKET_MAGIC              */
    uint32_t seq_num;                       /* Matches the request seq_num    */
    uint32_t status;                        /* CUresult (0 = success)         */
    uint32_t num_results;                   /* Number of uint64 results       */
    uint32_t data_len;                      /* Bytes of bulk return data      */
    uint32_t reserved;                      /* Padding for alignment          */
    uint64_t results[CUDA_MAX_INLINE_RESULTS]; /* Output values               */
} CUDACallResult;

#define CUDA_CALL_RESULT_SIZE  sizeof(CUDACallResult)

/* CUDAGpuInfo: returned by CUDA_CALL_GET_GPU_INFO for guest init. */
typedef struct __attribute__((packed)) CUDAGpuInfo {
    char     name[256];              /* Device name, e.g. "NVIDIA H100 ..." */
    uint8_t  uuid[16];              /* Device UUID                          */
    uint64_t total_mem;             /* Total device memory in bytes         */
    uint64_t free_mem;              /* Free device memory in bytes          */
    int32_t  compute_cap_major;     /* Compute capability major             */
    int32_t  compute_cap_minor;     /* Compute capability minor             */
    int32_t  multi_processor_count; /* Number of SMs                        */
    int32_t  max_threads_per_block; /* Max threads per block                */
    int32_t  max_block_dim_x;       /* Max block dimension X                */
    int32_t  max_block_dim_y;       /* Max block dimension Y                */
    int32_t  max_block_dim_z;       /* Max block dimension Z                */
    int32_t  max_grid_dim_x;        /* Max grid dimension X                 */
    int32_t  max_grid_dim_y;        /* Max grid dimension Y                 */
    int32_t  max_grid_dim_z;        /* Max grid dimension Z                 */
    int32_t  warp_size;             /* Warp size (32)                       */
    int32_t  max_shared_mem_per_block;   /* Shared mem per block (bytes)    */
    int32_t  max_shared_mem_per_mp; /* Shared mem per SM (bytes)            */
    int32_t  regs_per_block;        /* 32-bit registers per block           */
    int32_t  regs_per_multiprocessor;    /* 32-bit registers per SM         */
    int32_t  clock_rate_khz;        /* Core clock in kHz                    */
    int32_t  memory_clock_rate_khz; /* Memory clock in kHz                  */
    int32_t  memory_bus_width;      /* Memory bus width in bits             */
    int32_t  l2_cache_size;         /* L2 cache size in bytes               */
    int32_t  max_threads_per_mp;    /* Max threads per SM                   */
    int32_t  unified_addressing;    /* Supports unified addressing          */
    int32_t  managed_memory;        /* Supports managed memory              */
    int32_t  concurrent_kernels;    /* Supports concurrent kernel exec      */
    int32_t  async_engine_count;    /* Number of async engines (copy)       */
    int32_t  pci_bus_id;            /* PCI bus ID                           */
    int32_t  pci_device_id;         /* PCI device ID                        */
    int32_t  pci_domain_id;         /* PCI domain ID                        */
    int32_t  ecc_enabled;           /* ECC memory enabled                   */
    int32_t  driver_version;        /* CUDA driver version (e.g. 12080)     */
    int32_t  runtime_version;       /* CUDA runtime version                 */
    int32_t  reserved[16];          /* Future expansion                     */
} CUDAGpuInfo;

#define CUDA_GPU_INFO_SIZE  sizeof(CUDAGpuInfo)

/* CUDALaunchParams: grid/block dims, shared mem, stream; then param_sizes[], param_data[]. */
typedef struct __attribute__((packed)) CUDALaunchParams {
    uint64_t function_handle;       /* Host-side CUfunction handle     */
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;
    uint64_t stream_handle;         /* Host-side CUstream handle       */
    uint32_t num_params;            /* Number of kernel parameters     */
    uint32_t total_param_bytes;     /* Total size of param data        */
    /* followed by: uint32_t param_sizes[num_params]                   */
    /* followed by: uint8_t  param_data[total_param_bytes]             */
} CUDALaunchParams;

#define CUDA_LAUNCH_PARAMS_SIZE  sizeof(CUDALaunchParams)

/* cuModuleLoadData: PTX/CUBIN in payload; data_len in header. */
/* cuMemcpyHtoD: args = dst, count; payload = host data. cuMemcpyDtoH: args = src, count; response payload = device data. */
/* Chunked transfer: data > CUDA_MAX_CHUNK_SIZE split into messages; chunk flags in args[14]. */
#define CUDA_MAX_CHUNK_SIZE  (4 * 1024 * 1024)  /* 4 MB per chunk */

/* Chunk flags (stored in args[14] for chunked calls) */
#define CUDA_CHUNK_FLAG_FIRST  0x01
#define CUDA_CHUNK_FLAG_LAST   0x02
#define CUDA_CHUNK_FLAG_SINGLE 0x03  /* FIRST | LAST */

/* args[] are uint32_t; uint64 = two consecutive slots. */
#define CUDA_PACK_U64(args, idx, val) do {          \
    (args)[(idx)]     = (uint32_t)((val) & 0xFFFFFFFF);  \
    (args)[(idx) + 1] = (uint32_t)(((val) >> 32) & 0xFFFFFFFF); \
} while (0)

#define CUDA_UNPACK_U64(args, idx) \
    ((uint64_t)(args)[(idx)] | ((uint64_t)(args)[(idx) + 1] << 32))

#endif /* CUDA_PROTOCOL_H */

/*
 * GPU Properties — H100 80GB PCIe defaults
 *
 * These constants are used by the guest-side CUDA shim library to
 * answer cuDeviceGetAttribute() and similar queries BEFORE the shim
 * has connected to the host and fetched live values.
 *
 * Once the shim connects and calls CUDA_CALL_GET_GPU_INFO, the live
 * values from the host replace these defaults.
 *
 * Values sourced from:
 *   https://www.nvidia.com/en-us/data-center/h100/
 *   CUDA Toolkit deviceQuery sample output for H100 PCIe
 */

#ifndef GPU_PROPERTIES_H
#define GPU_PROPERTIES_H

/* ---- Device identity ------------------------------------------- */
#define GPU_DEFAULT_NAME            "NVIDIA H100 80GB HBM3"
#define GPU_DEFAULT_PCI_DEVICE_ID   0x2331   /* H100 PCIe */
#define GPU_DEFAULT_PCI_VENDOR_ID   0x10DE   /* NVIDIA     */

/* ---- Compute capability ---------------------------------------- */
#define GPU_DEFAULT_CC_MAJOR        9
#define GPU_DEFAULT_CC_MINOR        0

/* ---- Core counts ----------------------------------------------- */
#define GPU_DEFAULT_SM_COUNT        132      /* Streaming Multiprocessors */
#define GPU_DEFAULT_CORES_PER_SM    128      /* FP32 cores per SM         */

/* ---- Memory ---------------------------------------------------- */
#define GPU_DEFAULT_TOTAL_MEM       (80ULL * 1024 * 1024 * 1024)  /* 80 GB */
#define GPU_DEFAULT_FREE_MEM        (78ULL * 1024 * 1024 * 1024)  /* ~78 GB */
#define GPU_DEFAULT_MEM_BUS_WIDTH   5120     /* bits */
#define GPU_DEFAULT_L2_CACHE_SIZE   (52428800) /* 50 MB */
#define GPU_DEFAULT_ECC_ENABLED     1

/* ---- Clocks ---------------------------------------------------- */
#define GPU_DEFAULT_CLOCK_RATE_KHZ        1620000   /* 1620 MHz core  */
#define GPU_DEFAULT_MEM_CLOCK_RATE_KHZ    1593000   /* 1593 MHz HBM3  */

/* ---- Thread / block limits ------------------------------------- */
#define GPU_DEFAULT_MAX_THREADS_PER_BLOCK   1024
#define GPU_DEFAULT_MAX_BLOCK_DIM_X         1024
#define GPU_DEFAULT_MAX_BLOCK_DIM_Y         1024
#define GPU_DEFAULT_MAX_BLOCK_DIM_Z         64
#define GPU_DEFAULT_MAX_GRID_DIM_X          2147483647
#define GPU_DEFAULT_MAX_GRID_DIM_Y          65535
#define GPU_DEFAULT_MAX_GRID_DIM_Z          65535
#define GPU_DEFAULT_WARP_SIZE               32
#define GPU_DEFAULT_MAX_THREADS_PER_SM      2048

/* ---- Shared memory / registers --------------------------------- */
#define GPU_DEFAULT_SHARED_MEM_PER_BLOCK    (49152)     /* 48 KB  */
#define GPU_DEFAULT_SHARED_MEM_PER_SM       (233472)    /* 228 KB */
#define GPU_DEFAULT_REGS_PER_BLOCK          65536
#define GPU_DEFAULT_REGS_PER_SM             65536

/* ---- Feature flags --------------------------------------------- */
#define GPU_DEFAULT_CONCURRENT_KERNELS      1
#define GPU_DEFAULT_UNIFIED_ADDRESSING      1
#define GPU_DEFAULT_MANAGED_MEMORY          1
#define GPU_DEFAULT_ASYNC_ENGINE_COUNT      3
#define GPU_DEFAULT_CAN_MAP_HOST_MEM        1
#define GPU_DEFAULT_COMPUTE_MODE            0  /* cudaComputeModeDefault */
#define GPU_DEFAULT_INTEGRATED              0  /* discrete GPU */
#define GPU_DEFAULT_MULTI_GPU_BOARD         0
#define GPU_DEFAULT_COOPERATIVE_LAUNCH      1
#define GPU_DEFAULT_GLOBAL_L1_CACHE_SUPPORT 1
#define GPU_DEFAULT_LOCAL_L1_CACHE_SUPPORT  1
#define GPU_DEFAULT_MAX_TEXTURE_1D          (131072)
#define GPU_DEFAULT_MAX_TEXTURE_2D_W        (131072)
#define GPU_DEFAULT_MAX_TEXTURE_2D_H        (65536)
#define GPU_DEFAULT_MAX_TEXTURE_3D_W        (16384)
#define GPU_DEFAULT_MAX_TEXTURE_3D_H        (16384)
#define GPU_DEFAULT_MAX_TEXTURE_3D_D        (16384)
#define GPU_DEFAULT_TEXTURE_ALIGNMENT       512
#define GPU_DEFAULT_MAX_PITCH               (2147483647)
#define GPU_DEFAULT_TOTAL_CONSTANT_MEM      65536
#define GPU_DEFAULT_CLOCK_INSTRUCTION_RATE  32

/* ---- PCI topology (placeholder) -------------------------------- */
#define GPU_DEFAULT_PCI_BUS_ID              0
#define GPU_DEFAULT_PCI_DEV_ID              0
#define GPU_DEFAULT_PCI_DOMAIN_ID           0

/* ---- Driver / runtime version ---------------------------------- */
/* CRITICAL: CUDA runtime (libcudart.so.12) checks driver version.
 * Error: "CUDA driver version is insufficient for CUDA runtime version"
 * We need to return a version that's >= what the runtime expects.
 * CUDA 12.8 runtime typically requires driver >= 12.8, but let's use 13.0
 * to ensure compatibility. Format: major * 1000 + minor * 10 */
#define GPU_DEFAULT_DRIVER_VERSION          13000  /* CUDA 13.0 (increased from 12.9) */
#define GPU_DEFAULT_RUNTIME_VERSION         12080  /* CUDA 12.8 runtime */

/* ---- UUID (placeholder — 16 zero bytes, host fills in real one) - */
#define GPU_DEFAULT_UUID_BYTES  \
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, \
      0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }

#endif /* GPU_PROPERTIES_H */

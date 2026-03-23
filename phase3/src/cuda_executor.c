/* Host-side CUDA API replay (mediator → physical GPU). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

#include "cuda_executor.h"
#include "cuda_protocol.h"
#include "vgpu_protocol.h"

#define MAX_VMS             64
#define MAX_MEM_ENTRIES     4096
#define MAX_MODULE_ENTRIES  256
#define MAX_LIBRARY_ENTRIES 256
#define MAX_FUNC_ENTRIES    1024
#define MAX_STREAM_ENTRIES  128
#define MAX_EVENT_ENTRIES   256
#define MAX_CUBLAS_ENTRIES  128

typedef struct {
    uint64_t guest_ptr;
    CUdeviceptr host_ptr;
    size_t   size;
} mem_entry_t;

typedef struct {
    uint64_t guest_handle;
    CUmodule host_module;
} module_entry_t;

typedef struct {
    uint64_t guest_handle;
    CUlibrary host_library;
    void     *owned_image;
    size_t    owned_image_size;
} library_entry_t;

typedef struct {
    uint64_t guest_handle;
    CUfunction host_function;
} func_entry_t;

typedef struct {
    uint64_t guest_handle;
    CUstream host_stream;
} stream_entry_t;

typedef struct {
    uint64_t guest_handle;
    CUevent host_event;
} event_entry_t;

typedef struct {
    uint64_t guest_handle;
    cublasHandle_t host_handle;
} cublas_entry_t;

typedef struct {
    uint32_t    vm_id;
    int         active;
    CUcontext   ctx;
    int         ctx_valid;

    /* Memory mapping */
    mem_entry_t    mem[MAX_MEM_ENTRIES];
    int            mem_count;

    /* Module mapping */
    module_entry_t modules[MAX_MODULE_ENTRIES];
    int            module_count;

    /* Library mapping */
    library_entry_t libraries[MAX_LIBRARY_ENTRIES];
    int             library_count;

    /* Function mapping */
    func_entry_t   funcs[MAX_FUNC_ENTRIES];
    int            func_count;

    /* Stream mapping */
    stream_entry_t streams[MAX_STREAM_ENTRIES];
    int            stream_count;

    /* Event mapping */
    event_entry_t  events[MAX_EVENT_ENTRIES];
    int            event_count;

    /* CUBLAS handle mapping */
    cublas_entry_t cublas[MAX_CUBLAS_ENTRIES];
    int            cublas_count;

    uint8_t       *mod_chunk_buf;
    size_t         mod_chunk_alloc;
    size_t         mod_chunk_used;
    uint64_t       htod_total_bytes;
    uint64_t       htod_last_log_bytes;
} vm_state_t;

#define HTOD_PROGRESS_LOG_INTERVAL  (10 * 1024 * 1024)

static int executor_verbose_copy_logging(void)
{
    static int cached = -1;
    if (cached < 0) {
        cached = (getenv("VGPU_EXECUTOR_DEBUG") != NULL) ? 1 : 0;
    }
    return cached;
}

struct cuda_executor {
    CUdevice        device;
    CUcontext       primary_ctx;
    int             cuda_initialized;
    int             nvml_initialized;
    CUDAGpuInfo     gpu_info;
    int             gpu_info_valid;
    pthread_mutex_t mutex;

    vm_state_t      vms[MAX_VMS];
};

static vm_state_t* find_or_create_vm(cuda_executor_t *exec, uint32_t vm_id)
{
    int i;
    for (i = 0; i < MAX_VMS; i++) {
        if (exec->vms[i].active && exec->vms[i].vm_id == vm_id) {
            return &exec->vms[i];
        }
    }
    for (i = 0; i < MAX_VMS; i++) {
        if (!exec->vms[i].active) {
            memset(&exec->vms[i], 0, sizeof(vm_state_t));
            exec->vms[i].vm_id = vm_id;
            exec->vms[i].active = 1;
            return &exec->vms[i];
        }
    }
    return NULL;
}

static vm_state_t* find_vm(cuda_executor_t *exec, uint32_t vm_id)
{
    for (int i = 0; i < MAX_VMS; i++) {
        if (exec->vms[i].active && exec->vms[i].vm_id == vm_id) {
            return &exec->vms[i];
        }
    }
    return NULL;
}

static void vm_add_mem(vm_state_t *vm, uint64_t guest, CUdeviceptr host,
                       size_t size)
{
    if (vm->mem_count < MAX_MEM_ENTRIES) {
        vm->mem[vm->mem_count].guest_ptr = guest;
        vm->mem[vm->mem_count].host_ptr  = host;
        vm->mem[vm->mem_count].size      = size;
        vm->mem_count++;
    }
}

static CUdeviceptr vm_find_mem(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->mem_count; i++) {
        if (vm->mem[i].guest_ptr == guest)
            return vm->mem[i].host_ptr;
        if (guest > vm->mem[i].guest_ptr) {
            uint64_t off = guest - vm->mem[i].guest_ptr;
            if (off < vm->mem[i].size) {
                return vm->mem[i].host_ptr + off;
            }
        }
    }
    return 0;
}

static void vm_remove_mem(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->mem_count; i++) {
        if (vm->mem[i].guest_ptr == guest) {
            vm->mem[i] = vm->mem[vm->mem_count - 1];
            vm->mem_count--;
            return;
        }
    }
}

static void vm_add_module(vm_state_t *vm, uint64_t guest, CUmodule host)
{
    if (vm->module_count < MAX_MODULE_ENTRIES) {
        vm->modules[vm->module_count].guest_handle = guest;
        vm->modules[vm->module_count].host_module  = host;
        vm->module_count++;
    }
}

static CUmodule vm_find_module(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->module_count; i++) {
        if (vm->modules[i].guest_handle == guest)
            return vm->modules[i].host_module;
    }
    return NULL;
}

static void vm_remove_module(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->module_count; i++) {
        if (vm->modules[i].guest_handle == guest) {
            vm->modules[i] = vm->modules[vm->module_count - 1];
            vm->module_count--;
            return;
        }
    }
}

static void vm_add_library(vm_state_t *vm, uint64_t guest, CUlibrary host,
                           void *owned_image, size_t owned_image_size)
{
    if (vm->library_count < MAX_LIBRARY_ENTRIES) {
        vm->libraries[vm->library_count].guest_handle = guest;
        vm->libraries[vm->library_count].host_library = host;
        vm->libraries[vm->library_count].owned_image = owned_image;
        vm->libraries[vm->library_count].owned_image_size = owned_image_size;
        vm->library_count++;
    }
}

static CUlibrary vm_find_library(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->library_count; i++) {
        if (vm->libraries[i].guest_handle == guest)
            return vm->libraries[i].host_library;
    }
    return NULL;
}

static void vm_remove_library(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->library_count; i++) {
        if (vm->libraries[i].guest_handle == guest) {
            free(vm->libraries[i].owned_image);
            vm->libraries[i] = vm->libraries[vm->library_count - 1];
            vm->library_count--;
            return;
        }
    }
}

/* Function mapping helpers */
static void vm_add_func(vm_state_t *vm, uint64_t guest, CUfunction host)
{
    if (vm->func_count < MAX_FUNC_ENTRIES) {
        vm->funcs[vm->func_count].guest_handle = guest;
        vm->funcs[vm->func_count].host_function = host;
        vm->func_count++;
    }
}

static CUfunction vm_find_func(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->func_count; i++) {
        if (vm->funcs[i].guest_handle == guest)
            return vm->funcs[i].host_function;
    }
    return NULL;
}

/* Stream mapping helpers */
static void vm_add_stream(vm_state_t *vm, uint64_t guest, CUstream host)
{
    if (vm->stream_count < MAX_STREAM_ENTRIES) {
        vm->streams[vm->stream_count].guest_handle = guest;
        vm->streams[vm->stream_count].host_stream  = host;
        vm->stream_count++;
    }
}

static CUstream vm_find_stream(vm_state_t *vm, uint64_t guest)
{
    if (guest == 0) return NULL;  /* NULL stream = default */
    for (int i = 0; i < vm->stream_count; i++) {
        if (vm->streams[i].guest_handle == guest)
            return vm->streams[i].host_stream;
    }
    return NULL;
}

static void vm_remove_stream(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->stream_count; i++) {
        if (vm->streams[i].guest_handle == guest) {
            vm->streams[i] = vm->streams[vm->stream_count - 1];
            vm->stream_count--;
            return;
        }
    }
}

static uint64_t vm_find_guest_stream(vm_state_t *vm, CUstream host)
{
    if (host == NULL) return 0;
    for (int i = 0; i < vm->stream_count; i++) {
        if (vm->streams[i].host_stream == host)
            return vm->streams[i].guest_handle;
    }
    return 0;
}

/* Event mapping helpers */
static int vm_add_event(vm_state_t *vm, uint64_t guest, CUevent host)
{
    if (vm->event_count >= MAX_EVENT_ENTRIES)
        return 0;
    vm->events[vm->event_count].guest_handle = guest;
    vm->events[vm->event_count].host_event   = host;
    vm->event_count++;
    return 1;
}

static CUevent vm_find_event(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->event_count; i++) {
        if (vm->events[i].guest_handle == guest)
            return vm->events[i].host_event;
    }
    return NULL;
}

static void vm_remove_event(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->event_count; i++) {
        if (vm->events[i].guest_handle == guest) {
            vm->events[i] = vm->events[vm->event_count - 1];
            vm->event_count--;
            return;
        }
    }
}

static void vm_add_cublas(vm_state_t *vm, uint64_t guest, cublasHandle_t host)
{
    if (vm->cublas_count < MAX_CUBLAS_ENTRIES) {
        vm->cublas[vm->cublas_count].guest_handle = guest;
        vm->cublas[vm->cublas_count].host_handle = host;
        vm->cublas_count++;
    }
}

static cublasHandle_t vm_find_cublas(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->cublas_count; i++) {
        if (vm->cublas[i].guest_handle == guest)
            return vm->cublas[i].host_handle;
    }
    return NULL;
}

static void vm_remove_cublas(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->cublas_count; i++) {
        if (vm->cublas[i].guest_handle == guest) {
            vm->cublas[i] = vm->cublas[vm->cublas_count - 1];
            vm->cublas_count--;
            return;
        }
    }
}

static int module_blob_looks_like_fatbin(const void *data, uint32_t data_len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t magic;

    if (!bytes || data_len < sizeof(uint32_t)) {
        return 0;
    }

    memcpy(&magic, bytes, sizeof(magic));
    return (magic == 0xBA55ED50U || magic == 0x466243b1U) ? 1 : 0;
}

typedef struct {
    uint32_t magic;
    uint32_t version;
    const void *data;
    void *filename_or_fatbins;
} host_fatbin_wrapper_t;

static const char *host_cuda_error_name(CUresult rc)
{
    const char *name = NULL;

    if (cuGetErrorName(rc, &name) == CUDA_SUCCESS && name) {
        return name;
    }
    return "CUDA_ERROR_UNKNOWN_NAME";
}

static const char *host_cuda_error_string(CUresult rc)
{
    const char *str = NULL;

    if (cuGetErrorString(rc, &str) == CUDA_SUCCESS && str) {
        return str;
    }
    return "unknown CUDA error";
}

static CUresult load_host_module(uint32_t vm_id, uint32_t call_id,
                                 const void *data, uint32_t data_len,
                                 CUmodule *mod_out)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t magic = 0;
    int use_fatbinary = 0;
    CUresult rc;
    char error_log[4096];
    char info_log[4096];

    if (!data || !mod_out || data_len == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (data_len >= sizeof(magic)) {
        memcpy(&magic, bytes, sizeof(magic));
    }

    /* Mirror the guest API exactly. Guessing based on payload magic can
     * route a CUDA_CALL_MODULE_LOAD_DATA request into cuModuleLoadFatBinary,
     * which changes semantics and has been observed to fail unpredictably for
     * cuBLASLt fatbin payloads that the guest explicitly submitted via
     * cuModuleLoadData. */
    use_fatbinary = (call_id == CUDA_CALL_MODULE_LOAD_FAT_BINARY);

    fprintf(stderr,
            "[cuda-executor] vm_id=%u module-load start call_id=0x%04x path=%s data_len=%u magic=0x%08x first8=%02x%02x%02x%02x%02x%02x%02x%02x\n",
            vm_id, call_id, use_fatbinary ? "cuModuleLoadFatBinary" : "cuModuleLoadData",
            data_len, magic,
            data_len > 0 ? bytes[0] : 0, data_len > 1 ? bytes[1] : 0,
            data_len > 2 ? bytes[2] : 0, data_len > 3 ? bytes[3] : 0,
            data_len > 4 ? bytes[4] : 0, data_len > 5 ? bytes[5] : 0,
            data_len > 6 ? bytes[6] : 0, data_len > 7 ? bytes[7] : 0);
    fflush(stderr);

    if (use_fatbinary) {
        if (magic == 0xBA55ED50U) {
            void *fatbin_copy = malloc(data_len);

            if (!fatbin_copy) {
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            memcpy(fatbin_copy, data, data_len);

            /* Try raw fat binary first (0xBA55ED50); some driver versions accept it.
             * If that fails, fall back to wrapper (0x466243b1) shape. */
            rc = cuModuleLoadFatBinary(mod_out, fatbin_copy);
            if (rc != CUDA_SUCCESS) {
                host_fatbin_wrapper_t wrapper;
                wrapper.magic = 0x466243b1U;
                wrapper.version = 1;
                wrapper.data = fatbin_copy;
                wrapper.filename_or_fatbins = NULL;
                rc = cuModuleLoadFatBinary(mod_out, &wrapper);
            }
            free(fatbin_copy);
        } else {
            rc = cuModuleLoadFatBinary(mod_out, data);
        }
    } else {
        CUjit_option opts[4];
        void *opt_vals[4];
        unsigned int err_size = sizeof(error_log);
        unsigned int info_size = sizeof(info_log);

        memset(error_log, 0, sizeof(error_log));
        memset(info_log, 0, sizeof(info_log));
        opts[0] = CU_JIT_ERROR_LOG_BUFFER;
        opt_vals[0] = error_log;
        opts[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
        opt_vals[1] = (void *)(uintptr_t)err_size;
        opts[2] = CU_JIT_INFO_LOG_BUFFER;
        opt_vals[2] = info_log;
        opts[3] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        opt_vals[3] = (void *)(uintptr_t)info_size;

        rc = cuModuleLoadDataEx(mod_out, data, 4, opts, opt_vals);
        if (error_log[0] || info_log[0]) {
            fprintf(stderr,
                    "[cuda-executor] vm_id=%u module-load JIT logs call_id=0x%04x err_log=\"%s\" info_log=\"%s\"\n",
                    vm_id, call_id, error_log, info_log);
            fflush(stderr);
        }
    }

    fprintf(stderr,
            "[cuda-executor] vm_id=%u module-load done call_id=0x%04x rc=%d name=%s detail=%s module=%p\n",
            vm_id, call_id, (int)rc, host_cuda_error_name(rc),
            host_cuda_error_string(rc), rc == CUDA_SUCCESS ? (void *)*mod_out : NULL);
    fflush(stderr);

    return rc;
}

/* ================================================================
 * Ensure VM has an active CUDA context
 * ================================================================ */
static CUresult ensure_vm_context(cuda_executor_t *exec, vm_state_t *vm)
{
    if (vm->ctx_valid) {
        cuCtxSetCurrent(vm->ctx);
        return CUDA_SUCCESS;
    }

    CUresult rc = cuCtxCreate(&vm->ctx, CU_CTX_SCHED_AUTO, exec->device);
    if (rc == CUDA_SUCCESS) {
        vm->ctx_valid = 1;
    }
    return rc;
}

/* ================================================================
 * Initialise the CUDA executor
 * ================================================================ */
int cuda_executor_init(cuda_executor_t **exec_out)
{
    cuda_executor_t *exec;
    CUresult rc;

    exec = (cuda_executor_t *)calloc(1, sizeof(cuda_executor_t));
    if (!exec) return -1;

    pthread_mutex_init(&exec->mutex, NULL);

    /* Initialize CUDA */
    rc = cuInit(0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda-executor] cuInit failed: %d\n", rc);
        free(exec);
        return -1;
    }
    exec->cuda_initialized = 1;

    /* Get device */
    int device_count = 0;
    cuDeviceGetCount(&device_count);
    if (device_count < 1) {
        fprintf(stderr, "[cuda-executor] No CUDA devices found\n");
        free(exec);
        return -1;
    }

    rc = cuDeviceGet(&exec->device, 0);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda-executor] cuDeviceGet failed: %d\n", rc);
        free(exec);
        return -1;
    }

    /* Retain primary context */
    rc = cuDevicePrimaryCtxRetain(&exec->primary_ctx, exec->device);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda-executor] cuDevicePrimaryCtxRetain failed: %d\n",
                rc);
        free(exec);
        return -1;
    }

    /* Initialize NVML */
    nvmlReturn_t nvml_rc = nvmlInit();
    if (nvml_rc == NVML_SUCCESS) {
        exec->nvml_initialized = 1;
    }

    /* Query GPU info */
    cuda_executor_get_gpu_info(exec, &exec->gpu_info);
    exec->gpu_info_valid = 1;

    fprintf(stderr, "[cuda-executor] Initialized: %s, %llu MB\n",
            exec->gpu_info.name,
            (unsigned long long)(exec->gpu_info.total_mem / (1024 * 1024)));

    *exec_out = exec;
    return 0;
}

/* ================================================================
 * Destroy the executor
 * ================================================================ */
void cuda_executor_destroy(cuda_executor_t *exec)
{
    if (!exec) return;

    /* Clean up all VM states */
    for (int i = 0; i < MAX_VMS; i++) {
        if (exec->vms[i].active) {
            cuda_executor_cleanup_vm(exec, exec->vms[i].vm_id);
        }
    }

    /* Release primary context */
    if (exec->cuda_initialized) {
        cuDevicePrimaryCtxRelease(exec->device);
    }

    /* Shutdown NVML */
    if (exec->nvml_initialized) {
        nvmlShutdown();
    }

    pthread_mutex_destroy(&exec->mutex);
    free(exec);
}

/* ================================================================
 * Query GPU info
 * ================================================================ */
int cuda_executor_get_gpu_info(cuda_executor_t *exec, CUDAGpuInfo *info)
{
    if (!exec || !info) return -1;

    memset(info, 0, sizeof(*info));

    /* Device name */
    cuDeviceGetName(info->name, sizeof(info->name), exec->device);

    /* UUID */
    CUuuid uuid;
    if (cuDeviceGetUuid(&uuid, exec->device) == CUDA_SUCCESS) {
        memcpy(info->uuid, uuid.bytes, 16);
    }

    /* Memory */
    size_t total_mem = 0;
    cuDeviceTotalMem(&total_mem, exec->device);
    info->total_mem = total_mem;

    /* Try to get free memory via context */
    cuCtxSetCurrent(exec->primary_ctx);
    size_t free_mem = 0;
    if (cuMemGetInfo(&free_mem, &total_mem) == CUDA_SUCCESS) {
        info->free_mem = free_mem;
    } else {
        info->free_mem = total_mem;
    }

    /* Compute capability */
    cuDeviceGetAttribute(&info->compute_cap_major,
                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                         exec->device);
    cuDeviceGetAttribute(&info->compute_cap_minor,
                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                         exec->device);

    /* Various attributes */
    cuDeviceGetAttribute(&info->multi_processor_count,
                         CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                         exec->device);
    cuDeviceGetAttribute(&info->max_threads_per_block,
                         CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                         exec->device);
    cuDeviceGetAttribute(&info->max_block_dim_x,
                         CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                         exec->device);
    cuDeviceGetAttribute(&info->max_block_dim_y,
                         CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                         exec->device);
    cuDeviceGetAttribute(&info->max_block_dim_z,
                         CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                         exec->device);
    cuDeviceGetAttribute(&info->max_grid_dim_x,
                         CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                         exec->device);
    cuDeviceGetAttribute(&info->max_grid_dim_y,
                         CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                         exec->device);
    cuDeviceGetAttribute(&info->max_grid_dim_z,
                         CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                         exec->device);
    cuDeviceGetAttribute(&info->warp_size,
                         CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                         exec->device);
    cuDeviceGetAttribute(&info->max_shared_mem_per_block,
                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                         exec->device);
    cuDeviceGetAttribute(&info->max_shared_mem_per_mp,
                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                         exec->device);
    cuDeviceGetAttribute(&info->regs_per_block,
                         CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
                         exec->device);
    cuDeviceGetAttribute(&info->regs_per_multiprocessor,
                         CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                         exec->device);
    cuDeviceGetAttribute(&info->clock_rate_khz,
                         CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                         exec->device);
    cuDeviceGetAttribute(&info->memory_clock_rate_khz,
                         CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                         exec->device);
    cuDeviceGetAttribute(&info->memory_bus_width,
                         CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                         exec->device);
    cuDeviceGetAttribute(&info->l2_cache_size,
                         CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                         exec->device);
    cuDeviceGetAttribute(&info->max_threads_per_mp,
                         CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                         exec->device);
    cuDeviceGetAttribute(&info->unified_addressing,
                         CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                         exec->device);
    cuDeviceGetAttribute(&info->managed_memory,
                         CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
                         exec->device);
    cuDeviceGetAttribute(&info->concurrent_kernels,
                         CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
                         exec->device);
    cuDeviceGetAttribute(&info->async_engine_count,
                         CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
                         exec->device);
    cuDeviceGetAttribute(&info->ecc_enabled,
                         CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
                         exec->device);

    /* PCI info */
    cuDeviceGetAttribute(&info->pci_bus_id,
                         CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                         exec->device);
    cuDeviceGetAttribute(&info->pci_device_id,
                         CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                         exec->device);
    cuDeviceGetAttribute(&info->pci_domain_id,
                         CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                         exec->device);

    /* Driver version */
    int driver_ver = 0;
    cuDriverGetVersion(&driver_ver);
    info->driver_version = driver_ver;
    info->runtime_version = driver_ver;

    return 0;
}

/* ================================================================
 * Execute a CUDA API call
 * ================================================================ */
int cuda_executor_call(cuda_executor_t *exec,
                       const CUDACallHeader *call,
                       const void *data, uint32_t data_len,
                       CUDACallResult *result,
                       void *result_data, uint32_t result_cap,
                       uint32_t *result_len)
{
    if (!exec || !call || !result) return CUDA_ERROR_INVALID_VALUE;

    pthread_mutex_lock(&exec->mutex);

    /* Initialize result */
    memset(result, 0, sizeof(*result));
    result->magic   = VGPU_SOCKET_MAGIC;
    result->seq_num = call->seq_num;
    if (result_len) *result_len = 0;

    vm_state_t *vm = find_or_create_vm(exec, call->vm_id);
    if (!vm) {
        result->status = CUDA_ERROR_OUT_OF_MEMORY;
        pthread_mutex_unlock(&exec->mutex);
        return result->status;
    }

    CUresult rc = CUDA_SUCCESS;

    switch (call->call_id) {

    /* ---- Initialisation ---------------------------------------- */
    case CUDA_CALL_INIT:
        /* Already initialized at executor level. Return num_results=1 so guest
         * never sees status=0 + num_results=0 (which would be misread as failed cudaMalloc). */
        fprintf(stderr, "[cuda-executor] CUDA_CALL_INIT vm=%u — pipeline live\n",
                call->vm_id);
        result->num_results = 1;
        result->results[0]  = 1;
        rc = CUDA_SUCCESS;
        break;

    case CUDA_CALL_DRIVER_GET_VERSION: {
        int ver = 0;
        rc = cuDriverGetVersion(&ver);
        result->num_results = 1;
        result->results[0] = (uint64_t)ver;
        break;
    }

    /* ---- Device queries (answered from cached info) ------------ */
    case CUDA_CALL_DEVICE_GET_COUNT:
        result->num_results = 1;
        result->results[0] = 1;  /* We expose 1 device */
        break;

    case CUDA_CALL_DEVICE_GET:
        result->num_results = 1;
        result->results[0] = 0;
        break;

    case CUDA_CALL_DEVICE_GET_NAME:
    case CUDA_CALL_DEVICE_GET_ATTRIBUTE:
    case CUDA_CALL_DEVICE_TOTAL_MEM:
    case CUDA_CALL_DEVICE_GET_UUID:
    case CUDA_CALL_DEVICE_COMPUTE_CAPABILITY:
        /* These are handled guest-side from cached GPU info */
        break;

    /* ---- GPU info query ---------------------------------------- */
    case CUDA_CALL_GET_GPU_INFO:
        /* Refresh gpu info */
        cuda_executor_get_gpu_info(exec, &exec->gpu_info);
        if (result_data && result_cap >= sizeof(CUDAGpuInfo)) {
            memcpy(result_data, &exec->gpu_info, sizeof(CUDAGpuInfo));
            result->data_len = sizeof(CUDAGpuInfo);
            if (result_len) *result_len = sizeof(CUDAGpuInfo);
        }
        break;

    /* ---- Primary context --------------------------------------- */
    case CUDA_CALL_DEVICE_PRIMARY_CTX_RETAIN: {
        rc = ensure_vm_context(exec, vm);
        result->num_results = 1;
        /* Return guest-visible handle (use pointer value) */
        result->results[0] = (uint64_t)(uintptr_t)vm->ctx;
        break;
    }

    case CUDA_CALL_DEVICE_PRIMARY_CTX_RELEASE:
        /* Don't actually release — we manage context lifecycle */
        break;

    case CUDA_CALL_DEVICE_PRIMARY_CTX_RESET:
        /* Reset context state but keep context alive */
        if (vm->ctx_valid) {
            cuCtxSetCurrent(vm->ctx);
            /* Free all VM memory */
            for (int i = 0; i < vm->mem_count; i++) {
                cuMemFree(vm->mem[i].host_ptr);
            }
            vm->mem_count = 0;
        }
        break;

    case CUDA_CALL_DEVICE_PRIMARY_CTX_SET_FLAGS:
        /* Accepted but no-op for now */
        break;

    case CUDA_CALL_DEVICE_PRIMARY_CTX_GET_STATE: {
        result->num_results = 2;
        result->results[0] = 0;  /* flags */
        result->results[1] = vm->ctx_valid ? 1 : 0;  /* active */
        break;
    }

    /* ---- Context management ------------------------------------ */
    case CUDA_CALL_CTX_CREATE: {
        rc = ensure_vm_context(exec, vm);
        result->num_results = 1;
        result->results[0] = (uint64_t)(uintptr_t)vm->ctx;
        break;
    }

    case CUDA_CALL_CTX_DESTROY:
        /* Don't destroy — we manage context lifecycle */
        break;

    case CUDA_CALL_CTX_SET_CURRENT:
        if (vm->ctx_valid) {
            rc = cuCtxSetCurrent(vm->ctx);
        }
        break;

    case CUDA_CALL_CTX_GET_CURRENT: {
        result->num_results = 1;
        result->results[0] = (uint64_t)(uintptr_t)vm->ctx;
        break;
    }

    case CUDA_CALL_CTX_SYNCHRONIZE:
        if (vm->ctx_valid) {
            cuCtxSetCurrent(vm->ctx);
            rc = cuCtxSynchronize();
        }
        break;

    case CUDA_CALL_CTX_GET_DEVICE:
        result->num_results = 1;
        result->results[0] = 0;
        break;

    case CUDA_CALL_CTX_GET_API_VERSION:
        result->num_results = 1;
        result->results[0] = 3020;
        break;

    /* ---- Memory management ------------------------------------- */
    case CUDA_CALL_MEM_ALLOC: {
        uint64_t bytesize = CUDA_UNPACK_U64(call->args, 0);

        fprintf(stderr, "[cuda-executor] cuMemAlloc: allocating %llu bytes on physical GPU (vm=%u)\n",
                (unsigned long long)bytesize, call->vm_id);

        /* Use primary context for allocation (per-VM cuCtxCreate can fail and cause "unable to allocate CUDA0 buffer") */
        cuCtxSetCurrent(exec->primary_ctx);

        CUdeviceptr dptr = 0;
        rc = cuMemAlloc(&dptr, (size_t)bytesize);
        if (rc == CUDA_SUCCESS) {
            /* Generate a guest-visible handle (use the host pointer value) */
            uint64_t guest_ptr = (uint64_t)dptr;
            vm_add_mem(vm, guest_ptr, dptr, (size_t)bytesize);
            result->num_results = 1;
            result->results[0] = guest_ptr;
            fprintf(stderr, "[cuda-executor] cuMemAlloc SUCCESS: allocated 0x%llx on physical GPU (vm=%u)\n",
                    (unsigned long long)dptr, call->vm_id);
        } else {
            fprintf(stderr, "[cuda-executor] cuMemAlloc FAILED: rc=%d (vm=%u)\n", rc, call->vm_id);
        }
        break;
    }

    case CUDA_CALL_MEM_FREE: {
        uint64_t guest_ptr = CUDA_UNPACK_U64(call->args, 0);
        CUdeviceptr host_ptr = vm_find_mem(vm, guest_ptr);
        if (host_ptr) {
            cuCtxSetCurrent(exec->primary_ctx);
            rc = cuMemFree(host_ptr);
            vm_remove_mem(vm, guest_ptr);
        }
        break;
    }

    case CUDA_CALL_MEMCPY_HTOD: {
        uint64_t dst = CUDA_UNPACK_U64(call->args, 0);
        uint64_t byte_count = CUDA_UNPACK_U64(call->args, 2);

        CUdeviceptr host_dst = vm_find_mem(vm, dst);
        if (!host_dst) {
            /* Might be an offset within a larger allocation */
            /* Try using the guest ptr directly if we allocated it */
            host_dst = (CUdeviceptr)dst;
        }

        cuCtxSetCurrent(exec->primary_ctx);

        if (data && data_len > 0) {
            size_t copy_len = (size_t)byte_count;
            if (copy_len > data_len) copy_len = data_len;
            if (executor_verbose_copy_logging()) {
                fprintf(stderr, "[cuda-executor] cuMemcpyHtoD: dst=0x%llx size=%zu bytes (vm=%u)\n",
                        (unsigned long long)host_dst, copy_len, call->vm_id);
            }
            rc = cuMemcpyHtoD(host_dst, data, copy_len);
            if (rc == CUDA_SUCCESS) {
                if (executor_verbose_copy_logging())
                    fprintf(stderr, "[cuda-executor] cuMemcpyHtoD SUCCESS: data copied to physical GPU (vm=%u)\n", call->vm_id);
                /* Progress log: every 10 MB of HtoD transfer (model load) */
                vm->htod_total_bytes += (uint64_t)copy_len;
                if (vm->htod_total_bytes - vm->htod_last_log_bytes >= HTOD_PROGRESS_LOG_INTERVAL) {
                    fprintf(stderr, "[cuda-executor] HtoD progress: %llu MB total (vm=%u)\n",
                            (unsigned long long)(vm->htod_total_bytes / (1024 * 1024)), call->vm_id);
                    vm->htod_last_log_bytes = vm->htod_total_bytes;
                }
            } else {
                fprintf(stderr, "[cuda-executor] cuMemcpyHtoD FAILED: rc=%d dst=0x%llx size=%zu (vm=%u)\n",
                        rc, (unsigned long long)host_dst, copy_len, call->vm_id);
            }
        }
        break;
    }

    case CUDA_CALL_MEMCPY_DTOH: {
        uint64_t src = CUDA_UNPACK_U64(call->args, 0);
        uint64_t byte_count = CUDA_UNPACK_U64(call->args, 2);

        CUdeviceptr host_src = vm_find_mem(vm, src);
        if (!host_src) host_src = (CUdeviceptr)src;

        cuCtxSetCurrent(exec->primary_ctx);

        size_t copy_len = (size_t)byte_count;
        if (result_data && result_cap >= copy_len) {
            if (executor_verbose_copy_logging()) {
                fprintf(stderr, "[cuda-executor] cuMemcpyDtoH: src=0x%llx size=%zu bytes (vm=%u)\n",
                        (unsigned long long)host_src, copy_len, call->vm_id);
            }
            rc = cuMemcpyDtoH(result_data, host_src, copy_len);
            if (rc == CUDA_SUCCESS) {
                result->data_len = (uint32_t)copy_len;
                if (result_len) *result_len = (uint32_t)copy_len;
                if (executor_verbose_copy_logging())
                    fprintf(stderr, "[cuda-executor] cuMemcpyDtoH SUCCESS: data copied from physical GPU (vm=%u)\n", call->vm_id);
            } else {
                fprintf(stderr, "[cuda-executor] cuMemcpyDtoH FAILED: rc=%d src=0x%llx size=%zu (vm=%u)\n",
                        rc, (unsigned long long)host_src, copy_len, call->vm_id);
            }
        }
        break;
    }

    case CUDA_CALL_MEMCPY_DTOD: {
        uint64_t dst = CUDA_UNPACK_U64(call->args, 0);
        uint64_t src = CUDA_UNPACK_U64(call->args, 2);
        uint64_t byte_count = CUDA_UNPACK_U64(call->args, 4);

        CUdeviceptr host_dst = vm_find_mem(vm, dst);
        CUdeviceptr host_src = vm_find_mem(vm, src);
        if (!host_dst) host_dst = (CUdeviceptr)dst;
        if (!host_src) host_src = (CUdeviceptr)src;

        cuCtxSetCurrent(exec->primary_ctx);

        rc = cuMemcpyDtoD(host_dst, host_src, (size_t)byte_count);
        break;
    }

    case CUDA_CALL_MEMSET_D8: {
        uint64_t dst = CUDA_UNPACK_U64(call->args, 0);
        uint8_t uc = (uint8_t)call->args[2];
        uint64_t N = CUDA_UNPACK_U64(call->args, 4);

        CUdeviceptr host_dst = vm_find_mem(vm, dst);
        if (!host_dst) host_dst = (CUdeviceptr)dst;

        cuCtxSetCurrent(exec->primary_ctx);

        rc = cuMemsetD8(host_dst, uc, (size_t)N);
        break;
    }

    case CUDA_CALL_MEMSET_D32: {
        uint64_t dst = CUDA_UNPACK_U64(call->args, 0);
        uint32_t ui = call->args[2];
        uint64_t N = CUDA_UNPACK_U64(call->args, 4);

        CUdeviceptr host_dst = vm_find_mem(vm, dst);
        if (!host_dst) host_dst = (CUdeviceptr)dst;

        cuCtxSetCurrent(exec->primary_ctx);

        rc = cuMemsetD32(host_dst, ui, (size_t)N);
        break;
    }

    case CUDA_CALL_MEM_GET_INFO: {
        cuCtxSetCurrent(exec->primary_ctx);

        size_t free_mem = 0, total_mem = 0;
        rc = cuMemGetInfo(&free_mem, &total_mem);
        if (rc == CUDA_SUCCESS) {
            result->num_results = 2;
            result->results[0] = (uint64_t)free_mem;
            result->results[1] = (uint64_t)total_mem;
        }
        break;
    }

    /* ---- Module management ------------------------------------- */
    case CUDA_CALL_MODULE_LOAD_DATA:
    case CUDA_CALL_MODULE_LOAD_DATA_EX:
    case CUDA_CALL_MODULE_LOAD_FAT_BINARY: {
        /* Use primary context for module load; same as allocations and CUBLAS.
         * Loading in per-VM context can yield INVALID_IMAGE or context mismatch. */
        cuCtxSetCurrent(exec->primary_ctx);

        if (!data || data_len == 0) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        /* Determine chunk position from args[14] */
        uint32_t chunk_flags = call->args[14];
        int is_chunked = (chunk_flags != 0);
        int is_first   = (chunk_flags & CUDA_CHUNK_FLAG_FIRST) != 0;
        int is_last    = (chunk_flags & CUDA_CHUNK_FLAG_LAST)  != 0;
        int is_single  = (chunk_flags == CUDA_CHUNK_FLAG_SINGLE);

        if (is_chunked && data_len > 0) {
            fprintf(stderr,
                    "[cuda-executor] vm_id=%u module-chunk call_id=0x%04x flags=0x%08x first=%d last=%d single=%d data_len=%u first8=%02x%02x%02x%02x%02x%02x%02x%02x used_before=%zu alloc=%zu\n",
                    vm->vm_id, call->call_id, chunk_flags,
                    is_first, is_last, is_single, data_len,
                    ((const uint8_t *)data)[0], ((const uint8_t *)data)[1],
                    ((const uint8_t *)data)[2], ((const uint8_t *)data)[3],
                    ((const uint8_t *)data)[4], ((const uint8_t *)data)[5],
                    ((const uint8_t *)data)[6], ((const uint8_t *)data)[7],
                    vm->mod_chunk_used, vm->mod_chunk_alloc);
        }

        /* --- Non-chunked (legacy) or single-chunk path --- */
        if (!is_chunked || is_single) {
            CUmodule mod = NULL;
            rc = load_host_module(vm->vm_id, call->call_id, data, data_len, &mod);
            if (rc == CUDA_SUCCESS) {
                uint64_t guest_handle = (uint64_t)(uintptr_t)mod;
                vm_add_module(vm, guest_handle, mod);
                result->num_results = 1;
                result->results[0] = guest_handle;
            }
            /* Clean up any stale accumulation buffer */
            if (vm->mod_chunk_buf) {
                free(vm->mod_chunk_buf);
                vm->mod_chunk_buf   = NULL;
                vm->mod_chunk_alloc = 0;
                vm->mod_chunk_used  = 0;
            }
            break;
        }

        /* --- Chunked path --- */
        if (is_first) {
            /* Start of a new chunked module load: (re)allocate buffer */
            free(vm->mod_chunk_buf);
            /* Pre-allocate generously; realloc if needed */
            size_t initial = (data_len < (1u << 20)) ? (32u << 20) : (size_t)data_len * 8;
            vm->mod_chunk_buf = (uint8_t *)malloc(initial);
            if (!vm->mod_chunk_buf) {
                vm->mod_chunk_alloc = 0;
                vm->mod_chunk_used  = 0;
                rc = CUDA_ERROR_OUT_OF_MEMORY;
                break;
            }
            vm->mod_chunk_alloc = initial;
            vm->mod_chunk_used  = 0;
        }

        /* Append chunk data — grow buffer if needed */
        if (vm->mod_chunk_buf) {
            if (vm->mod_chunk_used + data_len > vm->mod_chunk_alloc) {
                size_t new_alloc = (vm->mod_chunk_used + data_len) * 2;
                uint8_t *nb = (uint8_t *)realloc(vm->mod_chunk_buf, new_alloc);
                if (!nb) {
                    free(vm->mod_chunk_buf);
                    vm->mod_chunk_buf   = NULL;
                    vm->mod_chunk_alloc = 0;
                    vm->mod_chunk_used  = 0;
                    rc = CUDA_ERROR_OUT_OF_MEMORY;
                    break;
                }
                vm->mod_chunk_buf   = nb;
                vm->mod_chunk_alloc = new_alloc;
            }
            memcpy(vm->mod_chunk_buf + vm->mod_chunk_used, data, data_len);
            vm->mod_chunk_used += data_len;
        }

        if (is_last) {
            /* All chunks received — call cuModuleLoadData with full image */
            CUmodule mod = NULL;
            rc = load_host_module(vm->vm_id, call->call_id, vm->mod_chunk_buf,
                                  (uint32_t)vm->mod_chunk_used, &mod);
            free(vm->mod_chunk_buf);
            vm->mod_chunk_buf   = NULL;
            vm->mod_chunk_alloc = 0;
            vm->mod_chunk_used  = 0;
            if (rc == CUDA_SUCCESS) {
                uint64_t guest_handle = (uint64_t)(uintptr_t)mod;
                vm_add_module(vm, guest_handle, mod);
                result->num_results = 1;
                result->results[0] = guest_handle;
            }
        } else {
            /* Not the last chunk yet — acknowledge receipt, no handle */
            result->num_results = 0;
            rc = CUDA_SUCCESS;
        }
        break;
    }

    case CUDA_CALL_MODULE_UNLOAD: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUmodule mod = vm_find_module(vm, guest_handle);
        if (mod) {
            cuCtxSetCurrent(exec->primary_ctx);
            rc = cuModuleUnload(mod);
            if (rc == CUDA_SUCCESS)
                vm_remove_module(vm, guest_handle);
        }
        break;
    }

    case CUDA_CALL_MODULE_GET_FUNCTION: {
        uint64_t mod_handle = CUDA_UNPACK_U64(call->args, 0);
        CUmodule mod = vm_find_module(vm, mod_handle);
        if (!mod) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        const char *func_name = (const char *)data;
        if (!func_name || data_len == 0) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        cuCtxSetCurrent(exec->primary_ctx);

        CUfunction func = NULL;
        rc = cuModuleGetFunction(&func, mod, func_name);
        if (rc == CUDA_SUCCESS) {
            uint64_t guest_func = (uint64_t)(uintptr_t)func;
            vm_add_func(vm, guest_func, func);
            result->num_results = 1;
            result->results[0] = guest_func;
        }
        break;
    }

    case CUDA_CALL_MODULE_GET_GLOBAL: {
        uint64_t mod_handle = CUDA_UNPACK_U64(call->args, 0);
        CUmodule mod = vm_find_module(vm, mod_handle);
        if (!mod) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        const char *name = (const char *)data;
        if (!name || data_len == 0) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        cuCtxSetCurrent(exec->primary_ctx);

        CUdeviceptr dptr = 0;
        size_t bytes = 0;
        rc = cuModuleGetGlobal(&dptr, &bytes, mod, name);
        if (rc == CUDA_SUCCESS) {
            result->num_results = 2;
            result->results[0] = (uint64_t)dptr;
            result->results[1] = (uint64_t)bytes;
        }
        break;
    }

    /* ---- Kernel launch ----------------------------------------- */
    case CUDA_CALL_LAUNCH_KERNEL: {
        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        if (!data || data_len < sizeof(CUDALaunchParams)) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        const CUDALaunchParams *lp = (const CUDALaunchParams *)data;

        /* Resolve function handle */
        CUfunction func = vm_find_func(vm, lp->function_handle);
        if (!func) {
            /* Try direct cast (if host gave handle directly) */
            func = (CUfunction)(uintptr_t)lp->function_handle;
        }

        /* Resolve stream handle */
        CUstream stream = vm_find_stream(vm, lp->stream_handle);

        /* Parse kernel parameters */
        const uint8_t *payload_ptr = (const uint8_t *)data;
        payload_ptr += sizeof(CUDALaunchParams);

        /* Read param_sizes */
        const uint32_t *param_sizes = (const uint32_t *)payload_ptr;
        payload_ptr += lp->num_params * sizeof(uint32_t);

        /* Read param_data */
        const uint8_t *param_data = payload_ptr;

        /* Build kernelParams array */
        void *kernelParams[256];
        /* Point each param directly to its data in the payload buffer */
        uint32_t offset = 0;
        for (uint32_t i = 0; i < lp->num_params && i < 256; i++) {
            /* We need mutable copies since CUDA may read them */
            kernelParams[i] = (void *)(param_data + offset);
            offset += param_sizes[i];
        }

        fprintf(stderr, "[cuda-executor] cuLaunchKernel: grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u params=%u vm=%u\n",
                lp->grid_dim_x, lp->grid_dim_y, lp->grid_dim_z,
                lp->block_dim_x, lp->block_dim_y, lp->block_dim_z,
                lp->shared_mem_bytes, lp->num_params, call->vm_id);

        rc = cuLaunchKernel(func,
                           lp->grid_dim_x, lp->grid_dim_y, lp->grid_dim_z,
                           lp->block_dim_x, lp->block_dim_y, lp->block_dim_z,
                           lp->shared_mem_bytes,
                           stream,
                           kernelParams,
                           NULL);  /* extra */

        /* Synchronize after launch to detect errors immediately */
        if (rc == CUDA_SUCCESS) {
            rc = cuCtxSynchronize();
            fprintf(stderr, "[cuda-executor] cuLaunchKernel SUCCESS: kernel executed on physical GPU (vm=%u)\n", call->vm_id);
        } else {
            fprintf(stderr, "[cuda-executor] cuLaunchKernel FAILED: rc=%d (vm=%u)\n", rc, call->vm_id);
        }
        break;
    }

    /* ---- Stream management ------------------------------------- */
    case CUDA_CALL_STREAM_CREATE:
    case CUDA_CALL_STREAM_CREATE_WITH_FLAGS: {
        uint32_t flags = call->args[0];

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        CUstream stream = NULL;
        rc = cuStreamCreate(&stream, flags);
        if (rc == CUDA_SUCCESS) {
            uint64_t guest_handle = (uint64_t)(uintptr_t)stream;
            vm_add_stream(vm, guest_handle, stream);
            result->num_results = 1;
            result->results[0] = guest_handle;
        }
        break;
    }

    case CUDA_CALL_STREAM_CREATE_WITH_PRIORITY: {
        uint32_t flags = call->args[0];
        int priority = (int)call->args[2];

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        CUstream stream = NULL;
        rc = cuStreamCreateWithPriority(&stream, flags, priority);
        if (rc == CUDA_SUCCESS) {
            uint64_t guest_handle = (uint64_t)(uintptr_t)stream;
            vm_add_stream(vm, guest_handle, stream);
            result->num_results = 1;
            result->results[0] = guest_handle;
        }
        break;
    }

    case CUDA_CALL_STREAM_DESTROY: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUstream stream = vm_find_stream(vm, guest_handle);
        if (!stream) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        rc = ensure_vm_context(exec, vm);
        if (rc == CUDA_SUCCESS) {
            rc = cuStreamDestroy(stream);
            if (rc == CUDA_SUCCESS)
                vm_remove_stream(vm, guest_handle);
        }
        break;
    }

    case CUDA_CALL_STREAM_SYNCHRONIZE: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUstream stream = vm_find_stream(vm, guest_handle);

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        rc = cuStreamSynchronize(stream);  /* NULL stream = default */
        break;
    }

    case CUDA_CALL_STREAM_QUERY: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUstream stream = vm_find_stream(vm, guest_handle);

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        rc = cuStreamQuery(stream);
        break;
    }

    case CUDA_CALL_STREAM_WAIT_EVENT: {
        uint64_t stream_handle = CUDA_UNPACK_U64(call->args, 0);
        uint64_t event_handle = CUDA_UNPACK_U64(call->args, 2);
        uint32_t flags = call->args[4];

        CUstream stream = (stream_handle == 0) ? NULL : vm_find_stream(vm, stream_handle);
        CUevent event = vm_find_event(vm, event_handle);

        if (!event) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        if (stream_handle != 0 && !stream) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        rc = ensure_vm_context(exec, vm);
        if (rc == CUDA_SUCCESS)
            rc = cuStreamWaitEvent(stream, event, flags);
        break;
    }

    /* ---- Event management -------------------------------------- */
    case CUDA_CALL_EVENT_CREATE:
    case CUDA_CALL_EVENT_CREATE_WITH_FLAGS: {
        uint32_t flags = call->args[0];

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        CUevent event = NULL;
        rc = cuEventCreate(&event, flags);
        if (rc == CUDA_SUCCESS) {
            uint64_t guest_handle = (uint64_t)(uintptr_t)event;
            if (!vm_add_event(vm, guest_handle, event)) {
                (void)cuEventDestroy(event);
                rc = CUDA_ERROR_OUT_OF_MEMORY;
            } else {
                result->num_results = 1;
                result->results[0] = guest_handle;
            }
        }
        break;
    }

    case CUDA_CALL_EVENT_DESTROY: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUevent event = vm_find_event(vm, guest_handle);
        if (!event) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        rc = ensure_vm_context(exec, vm);
        if (rc == CUDA_SUCCESS) {
            rc = cuEventDestroy(event);
            if (rc == CUDA_SUCCESS)
                vm_remove_event(vm, guest_handle);
        }
        break;
    }

    case CUDA_CALL_EVENT_RECORD: {
        uint64_t event_handle = CUDA_UNPACK_U64(call->args, 0);
        uint64_t stream_handle = CUDA_UNPACK_U64(call->args, 2);

        CUevent event = vm_find_event(vm, event_handle);
        CUstream stream = (stream_handle == 0) ? NULL : vm_find_stream(vm, stream_handle);

        if (!event) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        if (stream_handle != 0 && !stream) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        rc = ensure_vm_context(exec, vm);
        if (rc == CUDA_SUCCESS)
            rc = cuEventRecord(event, stream);
        break;
    }

    case CUDA_CALL_EVENT_SYNCHRONIZE: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUevent event = vm_find_event(vm, guest_handle);

        if (!event) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        rc = ensure_vm_context(exec, vm);
        if (rc == CUDA_SUCCESS)
            rc = cuEventSynchronize(event);
        break;
    }

    case CUDA_CALL_EVENT_QUERY: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUevent event = vm_find_event(vm, guest_handle);

        if (!event) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        rc = ensure_vm_context(exec, vm);
        if (rc == CUDA_SUCCESS)
            rc = cuEventQuery(event);
        break;
    }

    case CUDA_CALL_EVENT_ELAPSED_TIME: {
        uint64_t start_handle = CUDA_UNPACK_U64(call->args, 0);
        uint64_t end_handle = CUDA_UNPACK_U64(call->args, 2);

        CUevent start = vm_find_event(vm, start_handle);
        CUevent end = vm_find_event(vm, end_handle);

        if (!start || !end) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS)
            break;
        {
            float ms = 0.0f;
            rc = cuEventElapsedTime(&ms, start, end);
            if (rc == CUDA_SUCCESS) {
                uint32_t fbits;
                memcpy(&fbits, &ms, sizeof(float));
                result->num_results = 1;
                result->results[0] = (uint64_t)fbits;
            }
        }
        break;
    }

    /* ---- CUBLAS handle management ------------------------------- */
    case CUDA_CALL_CUBLAS_CREATE: {
        cublasHandle_t handle = NULL;
        cublasStatus_t cublas_rc;

        /* Use primary context for CUBLAS; per-VM contexts can trigger ALLOC_FAILED in cublasCreate_v2 */
        (void)cudaSetDevice(0);
        cuCtxSetCurrent(exec->primary_ctx);

        cublas_rc = cublasCreate_v2(&handle);
        if (executor_verbose_copy_logging() || cublas_rc != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[cuda-executor] vm_id=%u cublasCreate_v2 rc=%d handle=%p\n",
                    vm->vm_id, (int)cublas_rc, (void *)handle);
        }
        result->num_results = 1;
        result->results[0] = (uint64_t)cublas_rc;
        rc = CUDA_SUCCESS;
        if (cublas_rc == CUBLAS_STATUS_SUCCESS && handle) {
            uint64_t guest_handle = (uint64_t)(uintptr_t)handle;
            vm_add_cublas(vm, guest_handle, handle);
            result->num_results = 2;
            result->results[1] = guest_handle;
        }
        break;
    }

    case CUDA_CALL_CUBLAS_DESTROY: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        cublasHandle_t handle = vm_find_cublas(vm, guest_handle);
        cublasStatus_t cublas_rc = CUBLAS_STATUS_NOT_INITIALIZED;

        if (handle) {
            cuCtxSetCurrent(exec->primary_ctx);
            cublas_rc = cublasDestroy_v2(handle);
            if (cublas_rc == CUBLAS_STATUS_SUCCESS) {
                vm_remove_cublas(vm, guest_handle);
            }
        }

        result->num_results = 1;
        result->results[0] = (uint64_t)cublas_rc;
        rc = CUDA_SUCCESS;
        break;
    }

    case CUDA_CALL_CUBLAS_SET_STREAM: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        uint64_t stream_handle = CUDA_UNPACK_U64(call->args, 2);
        cublasHandle_t handle = vm_find_cublas(vm, guest_handle);
        CUstream stream = vm_find_stream(vm, stream_handle);
        cublasStatus_t cublas_rc = CUBLAS_STATUS_NOT_INITIALIZED;

        if (handle) {
            cuCtxSetCurrent(exec->primary_ctx);
                cublas_rc = cublasSetStream_v2(handle, (cudaStream_t)stream);
            if (executor_verbose_copy_logging() || cublas_rc != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[cuda-executor] vm_id=%u cublasSetStream_v2 rc=%d guest_handle=0x%llx stream_guest=0x%llx stream_host=%p\n",
                        vm->vm_id, (int)cublas_rc,
                        (unsigned long long)guest_handle,
                        (unsigned long long)stream_handle,
                        (void *)(uintptr_t)stream);
            }
        }

        result->num_results = 1;
        result->results[0] = (uint64_t)cublas_rc;
        rc = CUDA_SUCCESS;
        break;
    }

    case CUDA_CALL_CUBLAS_GET_STREAM: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        cublasHandle_t handle = vm_find_cublas(vm, guest_handle);
        cublasStatus_t cublas_rc = CUBLAS_STATUS_NOT_INITIALIZED;
        uint64_t guest_stream = 0;

        if (handle) {
            cudaStream_t stream = NULL;
            cuCtxSetCurrent(exec->primary_ctx);
            cublas_rc = cublasGetStream_v2(handle, &stream);
            if (cublas_rc == CUBLAS_STATUS_SUCCESS) {
                guest_stream = vm_find_guest_stream(vm, (CUstream)stream);
            }
        }

        result->num_results = 2;
        result->results[0] = (uint64_t)cublas_rc;
        result->results[1] = guest_stream;
        rc = CUDA_SUCCESS;
        break;
    }

    case CUDA_CALL_CUBLAS_SGEMM: {
        const CublasSgemmCall *sgemm = (const CublasSgemmCall *)data;
        cublasStatus_t cublas_rc = CUBLAS_STATUS_INVALID_VALUE;

        if (!sgemm || data_len < sizeof(*sgemm)) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        cublasHandle_t handle = vm_find_cublas(vm, sgemm->handle);
        CUdeviceptr host_a = vm_find_mem(vm, sgemm->a);
        CUdeviceptr host_b = vm_find_mem(vm, sgemm->b);
        CUdeviceptr host_c = vm_find_mem(vm, sgemm->c);

        if (!handle || !host_a || !host_b || !host_c) {
            result->num_results = 1;
            result->results[0] = (uint64_t)CUBLAS_STATUS_INVALID_VALUE;
            rc = CUDA_SUCCESS;
            break;
        }

        cuCtxSetCurrent(exec->primary_ctx);
        cublas_rc = cublasSgemm_v2(handle,
                                   sgemm->transa, sgemm->transb,
                                   sgemm->m, sgemm->n, sgemm->k,
                                   &sgemm->alpha,
                                   (const float *)(uintptr_t)host_a, sgemm->lda,
                                   (const float *)(uintptr_t)host_b, sgemm->ldb,
                                   &sgemm->beta,
                                   (float *)(uintptr_t)host_c, sgemm->ldc);
        if (executor_verbose_copy_logging() || cublas_rc != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[cuda-executor] vm_id=%u cublasSgemm_v2 rc=%d trans=(%d,%d) m=%d n=%d k=%d lda=%d ldb=%d ldc=%d a=0x%llx b=0x%llx c=0x%llx\n",
                    vm->vm_id, (int)cublas_rc,
                    sgemm->transa, sgemm->transb,
                    sgemm->m, sgemm->n, sgemm->k,
                    sgemm->lda, sgemm->ldb, sgemm->ldc,
                    (unsigned long long)sgemm->a,
                    (unsigned long long)sgemm->b,
                    (unsigned long long)sgemm->c);
        }

        result->num_results = 1;
        result->results[0] = (uint64_t)cublas_rc;
        rc = CUDA_SUCCESS;
        break;
    }

    case CUDA_CALL_CUBLAS_GEMM_EX: {
        const CublasGemmExCall *gemm = (const CublasGemmExCall *)data;
        cublasStatus_t cublas_rc = CUBLAS_STATUS_INVALID_VALUE;

        if (!gemm || data_len < sizeof(*gemm)) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        cublasHandle_t handle = vm_find_cublas(vm, gemm->handle);
        CUdeviceptr host_a = vm_find_mem(vm, gemm->a);
        CUdeviceptr host_b = vm_find_mem(vm, gemm->b);
        CUdeviceptr host_c = vm_find_mem(vm, gemm->c);
        if (!handle || !host_a || !host_b || !host_c) {
            fprintf(stderr,
                    "[cuda-executor] cublasGemmEx MAPPING FAILED vm_id=%u: handle=%p host_a=%p host_b=%p host_c=%p guest_a=0x%llx guest_b=0x%llx guest_c=0x%llx mem_count=%d\n",
                    vm->vm_id, (void *)(uintptr_t)handle, (void *)(uintptr_t)host_a, (void *)(uintptr_t)host_b, (void *)(uintptr_t)host_c,
                    (unsigned long long)gemm->a, (unsigned long long)gemm->b, (unsigned long long)gemm->c,
                    vm->mem_count);
            result->num_results = 1;
            result->results[0] = (uint64_t)CUBLAS_STATUS_INVALID_VALUE;
            rc = CUDA_SUCCESS;
            break;
        }

        cuCtxSetCurrent(exec->primary_ctx);
        {
            float alpha = gemm->alpha_f32;
            float beta = gemm->beta_f32;
            cublas_rc = cublasGemmEx(handle,
                                     (cublasOperation_t)gemm->transa,
                                     (cublasOperation_t)gemm->transb,
                                     gemm->m, gemm->n, gemm->k,
                                     &alpha,
                                     (const void *)(uintptr_t)host_a,
                                     (cudaDataType_t)gemm->Atype, gemm->lda,
                                     (const void *)(uintptr_t)host_b,
                                     (cudaDataType_t)gemm->Btype, gemm->ldb,
                                     &beta,
                                     (void *)(uintptr_t)host_c,
                                     (cudaDataType_t)gemm->Ctype, gemm->ldc,
                                     (cublasComputeType_t)gemm->computeType,
                                     (cublasGemmAlgo_t)gemm->algo);
            if (executor_verbose_copy_logging() || cublas_rc != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[cuda-executor] vm_id=%u cublasGemmEx rc=%d trans=(%d,%d) m=%d n=%d k=%d types=(%d,%d,%d) compute=%d algo=%d lda=%d ldb=%d ldc=%d a=0x%llx b=0x%llx c=0x%llx\n",
                        vm->vm_id, (int)cublas_rc,
                        gemm->transa, gemm->transb, gemm->m, gemm->n, gemm->k,
                        gemm->Atype, gemm->Btype, gemm->Ctype,
                        gemm->computeType, gemm->algo,
                        gemm->lda, gemm->ldb, gemm->ldc,
                        (unsigned long long)gemm->a,
                        (unsigned long long)gemm->b,
                        (unsigned long long)gemm->c);
            }
        }

        result->num_results = 1;
        result->results[0] = (uint64_t)cublas_rc;
        rc = CUDA_SUCCESS;
        break;
    }

    /* ---- Library management ------------------------------------ */
    case CUDA_CALL_LIBRARY_LOAD_DATA: {
        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        if (!data || data_len == 0) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        /* Keep a host-owned copy alive for the lifetime of the CUlibrary.
         * cuLibraryGetModule/cuLibraryGetKernel may still depend on the image
         * remaining valid after cuLibraryLoadData returns. */
        void *owned_image = malloc(data_len);
        CUlibraryOption lib_option = (CUlibraryOption)1; /* BINARY_IS_PRESERVED */
        void *lib_option_value = (void *)1;
        CUlibrary lib = NULL;
        if (!owned_image) {
            rc = CUDA_ERROR_OUT_OF_MEMORY;
            break;
        }
        memcpy(owned_image, data, data_len);
        rc = cuLibraryLoadData(&lib,
                               owned_image,
                               NULL, NULL, 0,
                               &lib_option, &lib_option_value, 1);
        if (rc == CUDA_SUCCESS) {
            uint64_t guest_handle = (uint64_t)(uintptr_t)lib;
            fprintf(stderr,
                    "[cuda-executor] cuLibraryLoadData success vm=%u data_len=%u lib=%p guest_handle=0x%llx\n",
                    vm->vm_id, (unsigned)data_len, (void *)lib,
                    (unsigned long long)guest_handle);
            vm_add_library(vm, guest_handle, lib, owned_image, data_len);
            result->num_results = 1;
            result->results[0] = guest_handle;
            owned_image = NULL;
        } else {
            fprintf(stderr,
                    "[cuda-executor] cuLibraryLoadData failed vm=%u data_len=%u rc=%d\n",
                    vm->vm_id, (unsigned)data_len, (int)rc);
            free(owned_image);
        }
        break;
    }

    case CUDA_CALL_LIBRARY_UNLOAD: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUlibrary lib = vm_find_library(vm, guest_handle);
        if (!lib) {
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        rc = cuLibraryUnload(lib);
        if (rc == CUDA_SUCCESS) {
            vm_remove_library(vm, guest_handle);
        }
        break;
    }

    case CUDA_CALL_LIBRARY_GET_MODULE: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUlibrary lib = vm_find_library(vm, guest_handle);
        if (!lib) {
            fprintf(stderr,
                    "[cuda-executor] cuLibraryGetModule lookup miss vm=%u guest_handle=0x%llx library_count=%d\n",
                    vm->vm_id, (unsigned long long)guest_handle, vm->library_count);
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        CUmodule mod = NULL;
        rc = cuLibraryGetModule(&mod, lib);
        if (rc == CUDA_SUCCESS) {
            uint64_t guest_mod = (uint64_t)(uintptr_t)mod;
            fprintf(stderr,
                    "[cuda-executor] cuLibraryGetModule success vm=%u guest_handle=0x%llx lib=%p mod=%p guest_mod=0x%llx\n",
                    vm->vm_id, (unsigned long long)guest_handle, (void *)lib,
                    (void *)mod, (unsigned long long)guest_mod);
            vm_add_module(vm, guest_mod, mod);
            result->num_results = 1;
            result->results[0] = guest_mod;
        } else {
            fprintf(stderr,
                    "[cuda-executor] cuLibraryGetModule failed vm=%u guest_handle=0x%llx lib=%p rc=%d\n",
                    vm->vm_id, (unsigned long long)guest_handle, (void *)lib, (int)rc);
        }
        break;
    }

    /* ---- Function attributes ----------------------------------- */
    case CUDA_CALL_FUNC_GET_ATTRIBUTE: {
        int attrib = (int)call->args[0];
        uint64_t func_handle = CUDA_UNPACK_U64(call->args, 2);
        CUfunction func = vm_find_func(vm, func_handle);

        if (func) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                int pi = 0;
                rc = cuFuncGetAttribute(&pi,
                                        (CUfunction_attribute)attrib,
                                        func);
                result->num_results = 1;
                result->results[0] = (uint64_t)pi;
            }
        } else {
            rc = CUDA_ERROR_INVALID_VALUE;
        }
        break;
    }

    case CUDA_CALL_FUNC_SET_CACHE_CONFIG: {
        uint64_t func_handle = CUDA_UNPACK_U64(call->args, 0);
        int config = (int)call->args[2];
        CUfunction func = vm_find_func(vm, func_handle);

        if (func) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuFuncSetCacheConfig(func, (CUfunc_cache)config);
            }
        }
        break;
    }

    /* ---- Unsupported ------------------------------------------- */
    default:
        fprintf(stderr, "[cuda-executor] Unsupported CUDA call: 0x%04x\n",
                call->call_id);
        rc = CUDA_ERROR_NOT_SUPPORTED;
        break;
    }

    result->status = (uint32_t)rc;

    pthread_mutex_unlock(&exec->mutex);
    return rc;
}

/* ================================================================
 * Clean up all resources for a specific VM
 * ================================================================ */
void cuda_executor_cleanup_vm(cuda_executor_t *exec, uint32_t vm_id)
{
    if (!exec) return;

    pthread_mutex_lock(&exec->mutex);

    vm_state_t *vm = find_vm(exec, vm_id);
    if (!vm) {
        pthread_mutex_unlock(&exec->mutex);
        return;
    }

    if (vm->ctx_valid) {
        cuCtxSetCurrent(vm->ctx);

        /* Destroy all CUBLAS handles before streams they may reference */
        for (int i = 0; i < vm->cublas_count; i++) {
            cublasDestroy_v2(vm->cublas[i].host_handle);
        }
        vm->cublas_count = 0;

        /* Free all events */
        for (int i = 0; i < vm->event_count; i++) {
            cuEventDestroy(vm->events[i].host_event);
        }
        vm->event_count = 0;

        /* Destroy all streams */
        for (int i = 0; i < vm->stream_count; i++) {
            cuStreamDestroy(vm->streams[i].host_stream);
        }
        vm->stream_count = 0;

        /* Unload all modules (which also frees functions) */
        for (int i = 0; i < vm->module_count; i++) {
            cuModuleUnload(vm->modules[i].host_module);
        }
        vm->module_count = 0;

        /* Unload all libraries */
        for (int i = 0; i < vm->library_count; i++) {
            cuLibraryUnload(vm->libraries[i].host_library);
            free(vm->libraries[i].owned_image);
        }
        vm->library_count = 0;
        vm->func_count = 0;

        /* Free all device memory */
        for (int i = 0; i < vm->mem_count; i++) {
            cuMemFree(vm->mem[i].host_ptr);
        }
        vm->mem_count = 0;

        /* Destroy context */
        cuCtxDestroy(vm->ctx);
        vm->ctx_valid = 0;
    }

    /* Free any in-progress module-load chunk accumulation buffer */
    if (vm->mod_chunk_buf) {
        free(vm->mod_chunk_buf);
        vm->mod_chunk_buf   = NULL;
        vm->mod_chunk_alloc = 0;
        vm->mod_chunk_used  = 0;
    }

    vm->active = 0;
    fprintf(stderr, "[cuda-executor] Cleaned up VM %u\n", vm_id);

    pthread_mutex_unlock(&exec->mutex);
}

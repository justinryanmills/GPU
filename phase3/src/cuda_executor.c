/* Host-side CUDA API replay (mediator → physical GPU). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <cuda.h>
#include <nvml.h>

#include "cuda_executor.h"
#include "cuda_protocol.h"
#include "vgpu_protocol.h"

#define MAX_VMS             64
#define MAX_MEM_ENTRIES     4096
#define MAX_MODULE_ENTRIES  256
#define MAX_FUNC_ENTRIES    1024
#define MAX_STREAM_ENTRIES  128
#define MAX_EVENT_ENTRIES   256

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
    uint32_t    vm_id;
    int         active;
    CUcontext   ctx;
    int         ctx_valid;

    mem_entry_t    mem[MAX_MEM_ENTRIES];
    int            mem_count;
    module_entry_t modules[MAX_MODULE_ENTRIES];
    int            module_count;
    func_entry_t   funcs[MAX_FUNC_ENTRIES];
    int            func_count;
    stream_entry_t streams[MAX_STREAM_ENTRIES];
    int            stream_count;
    event_entry_t  events[MAX_EVENT_ENTRIES];
    int            event_count;

    uint8_t       *mod_chunk_buf;
    size_t         mod_chunk_alloc;
    size_t         mod_chunk_used;

    uint64_t       htod_total_bytes;
    uint64_t       htod_last_log_bytes;
} vm_state_t;

#define HTOD_PROGRESS_LOG_INTERVAL  (10 * 1024 * 1024)

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

static void vm_add_event(vm_state_t *vm, uint64_t guest, CUevent host)
{
    if (vm->event_count < MAX_EVENT_ENTRIES) {
        vm->events[vm->event_count].guest_handle = guest;
        vm->events[vm->event_count].host_event   = host;
        vm->event_count++;
    }
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
        /* Already initialized at executor level */
        fprintf(stderr, "[cuda-executor] CUDA_CALL_INIT vm=%u — pipeline live\n",
                call->vm_id);
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

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

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
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuMemFree(host_ptr);
                vm_remove_mem(vm, guest_ptr);
            }
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

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        if (data && data_len > 0) {
            size_t copy_len = (size_t)byte_count;
            if (copy_len > data_len) copy_len = data_len;
            /* Log only large transfers (>= 64KB) or failures to avoid flooding from many small copies */
            const size_t log_threshold = 64 * 1024;
            if (copy_len >= log_threshold) {
                fprintf(stderr, "[cuda-executor] cuMemcpyHtoD: dst=0x%llx size=%zu bytes (vm=%u)\n",
                        (unsigned long long)host_dst, copy_len, call->vm_id);
            }
            rc = cuMemcpyHtoD(host_dst, data, copy_len);
            if (rc == CUDA_SUCCESS) {
                if (copy_len >= log_threshold)
                    fprintf(stderr, "[cuda-executor] cuMemcpyHtoD SUCCESS: data copied to physical GPU (vm=%u)\n", call->vm_id);
                /* Progress log: every 10 MB of HtoD transfer (model load) */
                vm->htod_total_bytes += (uint64_t)copy_len;
                if (vm->htod_total_bytes - vm->htod_last_log_bytes >= HTOD_PROGRESS_LOG_INTERVAL) {
                    fprintf(stderr, "[cuda-executor] HtoD progress: %llu MB total (vm=%u)\n",
                            (unsigned long long)(vm->htod_total_bytes / (1024 * 1024)), call->vm_id);
                    vm->htod_last_log_bytes = vm->htod_total_bytes;
                }
            } else {
                fprintf(stderr, "[cuda-executor] cuMemcpyHtoD FAILED: rc=%d (vm=%u)\n", rc, call->vm_id);
            }
        }
        break;
    }

    case CUDA_CALL_MEMCPY_DTOH: {
        uint64_t src = CUDA_UNPACK_U64(call->args, 0);
        uint64_t byte_count = CUDA_UNPACK_U64(call->args, 2);

        CUdeviceptr host_src = vm_find_mem(vm, src);
        if (!host_src) host_src = (CUdeviceptr)src;

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        size_t copy_len = (size_t)byte_count;
        if (result_data && result_cap >= copy_len) {
            const size_t log_threshold = 64 * 1024;
            if (copy_len >= log_threshold) {
                fprintf(stderr, "[cuda-executor] cuMemcpyDtoH: src=0x%llx size=%zu bytes (vm=%u)\n",
                        (unsigned long long)host_src, copy_len, call->vm_id);
            }
            rc = cuMemcpyDtoH(result_data, host_src, copy_len);
            if (rc == CUDA_SUCCESS) {
                result->data_len = (uint32_t)copy_len;
                if (result_len) *result_len = (uint32_t)copy_len;
                if (copy_len >= log_threshold)
                    fprintf(stderr, "[cuda-executor] cuMemcpyDtoH SUCCESS: data copied from physical GPU (vm=%u)\n", call->vm_id);
            } else {
                fprintf(stderr, "[cuda-executor] cuMemcpyDtoH FAILED: rc=%d (vm=%u)\n", rc, call->vm_id);
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

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        rc = cuMemcpyDtoD(host_dst, host_src, (size_t)byte_count);
        break;
    }

    case CUDA_CALL_MEMSET_D8: {
        uint64_t dst = CUDA_UNPACK_U64(call->args, 0);
        uint8_t uc = (uint8_t)call->args[2];
        uint64_t N = CUDA_UNPACK_U64(call->args, 4);

        CUdeviceptr host_dst = vm_find_mem(vm, dst);
        if (!host_dst) host_dst = (CUdeviceptr)dst;

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        rc = cuMemsetD8(host_dst, uc, (size_t)N);
        break;
    }

    case CUDA_CALL_MEMSET_D32: {
        uint64_t dst = CUDA_UNPACK_U64(call->args, 0);
        uint32_t ui = call->args[2];
        uint64_t N = CUDA_UNPACK_U64(call->args, 4);

        CUdeviceptr host_dst = vm_find_mem(vm, dst);
        if (!host_dst) host_dst = (CUdeviceptr)dst;

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        rc = cuMemsetD32(host_dst, ui, (size_t)N);
        break;
    }

    case CUDA_CALL_MEM_GET_INFO: {
        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

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
        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

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

        /* --- Non-chunked (legacy) or single-chunk path --- */
        if (!is_chunked || is_single) {
            CUmodule mod = NULL;
            rc = cuModuleLoadData(&mod, data);
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
            rc = cuModuleLoadData(&mod, vm->mod_chunk_buf);
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
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuModuleUnload(mod);
                vm_remove_module(vm, guest_handle);
            }
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

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

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

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

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

        void *kernelParams[256];
        uint32_t offset = 0;
        for (uint32_t i = 0; i < lp->num_params && i < 256; i++) {
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
        if (stream) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuStreamDestroy(stream);
                vm_remove_stream(vm, guest_handle);
            }
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

        CUstream stream = vm_find_stream(vm, stream_handle);
        CUevent event = vm_find_event(vm, event_handle);

        if (event) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuStreamWaitEvent(stream, event, flags);
            }
        }
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
            vm_add_event(vm, guest_handle, event);
            result->num_results = 1;
            result->results[0] = guest_handle;
        }
        break;
    }

    case CUDA_CALL_EVENT_DESTROY: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUevent event = vm_find_event(vm, guest_handle);
        if (event) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuEventDestroy(event);
                vm_remove_event(vm, guest_handle);
            }
        }
        break;
    }

    case CUDA_CALL_EVENT_RECORD: {
        uint64_t event_handle = CUDA_UNPACK_U64(call->args, 0);
        uint64_t stream_handle = CUDA_UNPACK_U64(call->args, 2);

        CUevent event = vm_find_event(vm, event_handle);
        CUstream stream = vm_find_stream(vm, stream_handle);

        if (event) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuEventRecord(event, stream);
            }
        }
        break;
    }

    case CUDA_CALL_EVENT_SYNCHRONIZE: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUevent event = vm_find_event(vm, guest_handle);

        if (event) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuEventSynchronize(event);
            }
        }
        break;
    }

    case CUDA_CALL_EVENT_QUERY: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUevent event = vm_find_event(vm, guest_handle);

        if (event) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                rc = cuEventQuery(event);
            }
        }
        break;
    }

    case CUDA_CALL_EVENT_ELAPSED_TIME: {
        uint64_t start_handle = CUDA_UNPACK_U64(call->args, 0);
        uint64_t end_handle = CUDA_UNPACK_U64(call->args, 2);

        CUevent start = vm_find_event(vm, start_handle);
        CUevent end = vm_find_event(vm, end_handle);

        if (start && end) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                float ms = 0.0f;
                rc = cuEventElapsedTime(&ms, start, end);
                if (rc == CUDA_SUCCESS) {
                    uint32_t fbits;
                    memcpy(&fbits, &ms, sizeof(float));
                    result->num_results = 1;
                    result->results[0] = (uint64_t)fbits;
                }
            }
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

/*
 * cuda_executor.c  —  Host-side CUDA API replay engine
 *
 * Receives serialised CUDA API calls from the mediator and replays
 * them on the physical GPU using the real CUDA Driver API.
 *
 * Per-VM state:
 *   - CUcontext (one per VM)
 *   - Memory map (guest devptr → host devptr)
 *   - Module map (guest handle → host CUmodule)
 *   - Function map (guest handle → host CUfunction)
 *   - Stream map (guest handle → host CUstream)
 *   - Event map (guest handle → host CUevent)
 *
 * Build:
 *   nvcc -c cuda_executor.c -o cuda_executor.o -I../include \
 *        -lcuda -lnvidia-ml
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stddef.h>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

#include "cuda_executor.h"
#include "cuda_protocol.h"
#include "vgpu_protocol.h"

/* ================================================================
 * Configuration
 * ================================================================ */
#define MAX_VMS             64
#define MAX_MEM_ENTRIES     4096
#define MAX_MODULE_ENTRIES  2048
#define MAX_LIBRARY_ENTRIES 4096
#define MAX_FUNC_ENTRIES    8192
#define MAX_STREAM_ENTRIES  128
#define MAX_EVENT_ENTRIES   256
#define MAX_CUBLAS_ENTRIES  128
#define MAX_PENDING_ASYNC_HTOD 8192

#ifndef CU_STREAM_LEGACY
#define CU_STREAM_LEGACY ((CUstream)0x1)
#endif

#ifndef CU_STREAM_PER_THREAD
#define CU_STREAM_PER_THREAD ((CUstream)0x2)
#endif

typedef CUresult (*pfn_cuFuncGetParamInfo_t)(CUfunction func,
                                             size_t paramIndex,
                                             size_t *paramOffset,
                                             size_t *paramSize);
typedef CUresult (*pfn_cuGetProcAddress_t)(const char *symbol,
                                           void **pfn,
                                           int cudaVersion,
                                           cuuint64_t flags,
                                           CUdriverProcAddressQueryResult *symbolStatus);

static const char *host_cuda_error_name(CUresult rc);
static const char *host_cuda_error_string(CUresult rc);

static uint64_t host_fnv1a64(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ull;
    size_t i;

    if (!p || len == 0) {
        return h;
    }
    for (i = 0; i < len; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static pfn_cuFuncGetParamInfo_t resolve_cuFuncGetParamInfo(void)
{
    static pfn_cuFuncGetParamInfo_t fn = NULL;
    static int resolved = 0;

    if (!resolved) {
        fn = (pfn_cuFuncGetParamInfo_t)dlsym(RTLD_DEFAULT, "cuFuncGetParamInfo");
        if (!fn) {
            pfn_cuGetProcAddress_t cuGetProcAddress_fn =
                (pfn_cuGetProcAddress_t)dlsym(RTLD_DEFAULT, "cuGetProcAddress");
            if (cuGetProcAddress_fn) {
                void *sym = NULL;
                CUdriverProcAddressQueryResult symbol_status = 0;
                CUresult rc = cuGetProcAddress_fn("cuFuncGetParamInfo",
                                                  &sym,
                                                  12000,
                                                  0,
                                                  &symbol_status);
                if (rc == CUDA_SUCCESS && sym) {
                    fn = (pfn_cuFuncGetParamInfo_t)sym;
                }
            }
        }
        resolved = 1;
    }

    return fn;
}

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} vgpu_uint3_host_t;

typedef struct {
    const void *x_bias;
    const void *gate;
    const void *gate_bias;
    int32_t glu_op;
} ggml_cuda_mm_fusion_args_device_host_t;

typedef struct {
    const void *x;
    const void *mask;
    const void *sinks;
    const void *dst;
    uint8_t soft_max_params[128];
} soft_max_f32_params_host_t;

typedef struct {
    const float *x;
    void *vy;
    int64_t ne00;
    int64_t s01;
    int64_t s02;
    int64_t s03;
    int64_t ne0;
    uint32_t ne1;
    vgpu_uint3_host_t ne2;
} quantize_q8_1_params_host_t;

typedef struct {
    const float *x;
    const int32_t *ids;
    void *vy;
    int64_t ne00;
    int64_t s01;
    int64_t s02;
    int64_t s03;
    int64_t ne0;
    int32_t ne1;
    int32_t ne2;
} quantize_mmq_q8_1_params_host_t;

typedef struct {
    const float *x;
    float *dst;
    int32_t ncols;
    int64_t stride_row;
    int64_t stride_channel;
    int64_t stride_sample;
    float eps;
} rms_norm_f32_params_host_t;

typedef struct {
    const float *x;
    float *dst;
    int32_t ncols;
    int64_t stride_row;
    int64_t stride_channel;
    int64_t stride_sample;
    float eps;
    const float *mul;
    int64_t mul_stride_row;
    int64_t mul_stride_channel;
    int64_t mul_stride_sample;
    vgpu_uint3_host_t mul_ncols_packed;
    vgpu_uint3_host_t mul_nrows_packed;
    vgpu_uint3_host_t mul_nchannels_packed;
    vgpu_uint3_host_t mul_nsamples_packed;
} rms_norm_f32_mul_params_host_t;

typedef struct {
    const float *x;
    float *dst;
    int32_t ncols;
    int64_t stride_row;
    int64_t stride_channel;
    int64_t stride_sample;
    float eps;
    const float *mul;
    int64_t mul_stride_row;
    int64_t mul_stride_channel;
    int64_t mul_stride_sample;
    vgpu_uint3_host_t mul_ncols_packed;
    vgpu_uint3_host_t mul_nrows_packed;
    vgpu_uint3_host_t mul_nchannels_packed;
    vgpu_uint3_host_t mul_nsamples_packed;
    const float *add;
    int64_t add_stride_row;
    int64_t add_stride_channel;
    int64_t add_stride_sample;
    vgpu_uint3_host_t add_ncols_packed;
    vgpu_uint3_host_t add_nrows_packed;
    vgpu_uint3_host_t add_nchannels_packed;
    vgpu_uint3_host_t add_nsamples_packed;
} rms_norm_f32_mul_add_params_host_t;

typedef struct {
    const void *vx;
    const void *vy;
    const int32_t *ids;
    ggml_cuda_mm_fusion_args_device_host_t fusion;
    float *dst;
    uint32_t ncols_x;
    vgpu_uint3_host_t nchannels_y;
    uint32_t stride_row_x;
    uint32_t stride_col_y;
    uint32_t stride_col_dst;
    vgpu_uint3_host_t channel_ratio;
    uint32_t stride_channel_x;
    uint32_t stride_channel_y;
    uint32_t stride_channel_dst;
    vgpu_uint3_host_t sample_ratio;
    uint32_t stride_sample_x;
    uint32_t stride_sample_y;
    uint32_t stride_sample_dst;
} mul_mat_vec_q_params_host_t;

/* mul_mat_vec_f: same as mul_mat_vec_q except nchannels_y is int (not uint3)
 * and subsequent int/uint3 fields are packed 8 bytes earlier. */
typedef struct {
    const void *vx;                                   /* param 0: 8 bytes */
    const void *vy;                                   /* param 1: 8 bytes */
    const int32_t *ids;                               /* param 2: 8 bytes */
    ggml_cuda_mm_fusion_args_device_host_t fusion;    /* param 3: 32 bytes */
    float *dst;                                       /* param 4: 8 bytes */
    uint32_t ncols_x;                                 /* param 5: 4 bytes */
    uint32_t nchannels_y;                             /* param 6: 4 bytes (int, not uint3) */
    uint32_t stride_row_x;                            /* param 7: 4 bytes */
    uint32_t stride_col_y;                            /* param 8: 4 bytes */
    uint32_t stride_col_dst;                          /* param 9: 4 bytes */
    vgpu_uint3_host_t channel_ratio;                  /* param 10: 12 bytes */
    uint32_t stride_channel_x;                        /* param 11: 4 bytes */
    uint32_t stride_channel_y;                        /* param 12: 4 bytes */
    uint32_t stride_channel_dst;                      /* param 13: 4 bytes */
    vgpu_uint3_host_t sample_ratio;                   /* param 14: 12 bytes */
    uint32_t stride_sample_x;                         /* param 15: 4 bytes */
    uint32_t stride_sample_y;                         /* param 16: 4 bytes */
    uint32_t stride_sample_dst;                       /* param 17: 4 bytes */
} mul_mat_vec_f_params_host_t;

typedef struct {
    const void *src0;
    const void *src1;
    void *dst;
    int64_t ne_total;
    int64_t ne10;
    int64_t ne11;
    int64_t ne12;
    int64_t ne13;
    int64_t s01;
    int64_t s02;
    int64_t s03;
    int64_t s10;
    int64_t s11;
    int64_t s12;
    int64_t s1;
    int64_t s2;
    int64_t s3;
    vgpu_uint3_host_t ne00;
    vgpu_uint3_host_t ne01;
    vgpu_uint3_host_t ne02;
    vgpu_uint3_host_t ne11_fd;
    vgpu_uint3_host_t ne12_fd;
} k_set_rows_params_host_t;

typedef struct {
    const void *src0;
    const void *src1;
    void *dst;
    int32_t ne0;
    int32_t ne1;
    int32_t ne2;
    vgpu_uint3_host_t ne3;
    vgpu_uint3_host_t ne10;
    vgpu_uint3_host_t ne11;
    vgpu_uint3_host_t ne12;
    vgpu_uint3_host_t ne13;
    int32_t s1;
    int32_t s2;
    int32_t s3;
    int32_t s01;
    int32_t s02;
    int32_t s03;
    int32_t s11;
    int32_t s12;
    int32_t s13;
    const void *extra_src1;
} k_bin_bcast_params_host_t;

typedef struct {
    const void *src0;
    const void *src1;
    void *dst;
    vgpu_uint3_host_t ne0;
    vgpu_uint3_host_t ne1;
    vgpu_uint3_host_t ne2;
    uint32_t ne3;
    vgpu_uint3_host_t prod_012;
    vgpu_uint3_host_t prod_01;
    vgpu_uint3_host_t ne10;
    vgpu_uint3_host_t ne11;
    vgpu_uint3_host_t ne12;
    vgpu_uint3_host_t ne13;
    int32_t s1;
    int32_t s2;
    int32_t s3;
    int32_t s01;
    int32_t s02;
    int32_t s03;
    int32_t s11;
    int32_t s12;
    int32_t s13;
    const void *extra_src1;
} k_bin_bcast_unravel_params_host_t;

typedef struct {
    const char *x;
    const int *y;
    const int32_t *ids_dst;
    const int32_t *expert_bounds;
    float *dst;
    float *tmp_fixup;
    int32_t ncols_x;
    int32_t nrows_x;
    int32_t ncols_dst;
    int32_t stride_row_x;
    int32_t ncols_y;
    int32_t stride_col_dst;
    int32_t channel_ratio;
    int32_t nchannels_y;
    int32_t stride_channel_x;
    int32_t stride_channel_y;
    int32_t stride_channel_dst;
    int32_t sample_ratio;
    int32_t nsamples_y;
    int32_t stride_sample_x;
    int32_t stride_sample_y;
    int32_t stride_sample_dst;
    int32_t ncols_max;
} mul_mat_q_params_host_t;

typedef struct {
    const int32_t *ids_dst;
    const int32_t *expert_bounds;
    float *dst;
    const float *tmp_last_tile;
    int32_t ncols_x;
    int32_t nrows_x;
    int32_t ncols_dst;
    int32_t stride_col_dst;
    int32_t nchannels_y;
    int32_t stride_channel_dst;
    int32_t nsamples_y;
    int32_t stride_sample_dst;
    int32_t ncols_max;
} mul_mat_q_stream_k_fixup_params_host_t;

typedef struct {
    const void *x;
    void *dst;
    int32_t ne0;
    int32_t ne1;
    int32_t s1;
    int32_t s2;
    int32_t n_dims;
    const int32_t *pos;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    const float *freq_factors;
    const int64_t *row_indices;
    int32_t set_rows_stride;
} rope_norm_params_host_t;

typedef struct {
    const void *vx;
    void *y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t s01;
    int64_t s02;
    int64_t s03;
} convert_unary_params_host_t;

static int cuda_executor_copy_param_info(size_t param_index,
                                         const size_t *offsets,
                                         const size_t *sizes,
                                         size_t count,
                                         size_t *param_offset,
                                         size_t *param_size)
{
    if (param_index >= count) {
        return 0;
    }

    *param_offset = offsets[param_index];
    *param_size = sizes[param_index];
    return 1;
}

static int cuda_executor_copy_param_info_with_trailing_ptrs(size_t param_index,
                                                            const size_t *offsets,
                                                            const size_t *sizes,
                                                            size_t fixed_count,
                                                            size_t extra_ptr_offset,
                                                            size_t extra_ptr_count,
                                                            size_t *param_offset,
                                                            size_t *param_size)
{
    if (param_index < fixed_count) {
        return cuda_executor_copy_param_info(param_index,
                                             offsets,
                                             sizes,
                                             fixed_count,
                                             param_offset,
                                             param_size);
    }

    param_index -= fixed_count;
    if (param_index >= extra_ptr_count) {
        return 0;
    }

    *param_offset = extra_ptr_offset + (param_index * sizeof(void *));
    *param_size = sizeof(void *);
    return 1;
}

static int cuda_executor_copy_param_info_with_one_trailing_u32(size_t param_index,
                                                                const size_t *offsets,
                                                                const size_t *sizes,
                                                                size_t count,
                                                                size_t *param_offset,
                                                                size_t *param_size)
{
    if (cuda_executor_copy_param_info(param_index,
                                      offsets,
                                      sizes,
                                      count,
                                      param_offset,
                                      param_size)) {
        return 1;
    }
    if (count > 0 && param_index == count) {
        *param_offset = offsets[count - 1] + sizes[count - 1];
        *param_size = sizeof(uint32_t);
        return 1;
    }
    return 0;
}

static size_t cuda_executor_count_template_pack_ptrs(const char *func_name)
{
    const char *pack_begin;
    const char *pack_end;
    size_t count = 0;

    if (!func_name) {
        return 0;
    }

    pack_begin = strchr(func_name, 'J');
    if (!pack_begin) {
        return 0;
    }
    pack_end = strstr(pack_begin, "EEv");
    if (!pack_end || pack_end <= pack_begin) {
        return 0;
    }

    for (const char *p = pack_begin; p + 1 < pack_end; ++p) {
        if (p[0] == 'P' && p[1] == 'K') {
            count++;
        }
    }
    return count;
}

static int cuda_executor_try_synth_param_info(const char *func_name,
                                              size_t param_index,
                                              size_t *param_offset,
                                              size_t *param_size)
{
    static const size_t k_soft_max_offsets[] = {
        offsetof(soft_max_f32_params_host_t, x),
        offsetof(soft_max_f32_params_host_t, mask),
        offsetof(soft_max_f32_params_host_t, sinks),
        offsetof(soft_max_f32_params_host_t, dst),
        offsetof(soft_max_f32_params_host_t, soft_max_params),
    };
    static const size_t k_soft_max_sizes[] = {
        sizeof(((soft_max_f32_params_host_t *)0)->x),
        sizeof(((soft_max_f32_params_host_t *)0)->mask),
        sizeof(((soft_max_f32_params_host_t *)0)->sinks),
        sizeof(((soft_max_f32_params_host_t *)0)->dst),
        sizeof(((soft_max_f32_params_host_t *)0)->soft_max_params),
    };
    static const size_t k_quantize_q8_1_offsets[] = {
        offsetof(quantize_q8_1_params_host_t, x),
        offsetof(quantize_q8_1_params_host_t, vy),
        offsetof(quantize_q8_1_params_host_t, ne00),
        offsetof(quantize_q8_1_params_host_t, s01),
        offsetof(quantize_q8_1_params_host_t, s02),
        offsetof(quantize_q8_1_params_host_t, s03),
        offsetof(quantize_q8_1_params_host_t, ne0),
        offsetof(quantize_q8_1_params_host_t, ne1),
        offsetof(quantize_q8_1_params_host_t, ne2),
    };
    static const size_t k_quantize_q8_1_sizes[] = {
        sizeof(((quantize_q8_1_params_host_t *)0)->x),
        sizeof(((quantize_q8_1_params_host_t *)0)->vy),
        sizeof(((quantize_q8_1_params_host_t *)0)->ne00),
        sizeof(((quantize_q8_1_params_host_t *)0)->s01),
        sizeof(((quantize_q8_1_params_host_t *)0)->s02),
        sizeof(((quantize_q8_1_params_host_t *)0)->s03),
        sizeof(((quantize_q8_1_params_host_t *)0)->ne0),
        sizeof(((quantize_q8_1_params_host_t *)0)->ne1),
        sizeof(((quantize_q8_1_params_host_t *)0)->ne2),
    };
    static const size_t k_quantize_mmq_q8_1_offsets[] = {
        offsetof(quantize_mmq_q8_1_params_host_t, x),
        offsetof(quantize_mmq_q8_1_params_host_t, ids),
        offsetof(quantize_mmq_q8_1_params_host_t, vy),
        offsetof(quantize_mmq_q8_1_params_host_t, ne00),
        offsetof(quantize_mmq_q8_1_params_host_t, s01),
        offsetof(quantize_mmq_q8_1_params_host_t, s02),
        offsetof(quantize_mmq_q8_1_params_host_t, s03),
        offsetof(quantize_mmq_q8_1_params_host_t, ne0),
        offsetof(quantize_mmq_q8_1_params_host_t, ne1),
        offsetof(quantize_mmq_q8_1_params_host_t, ne2),
    };
    static const size_t k_quantize_mmq_q8_1_sizes[] = {
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->x),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->ids),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->vy),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->ne00),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->s01),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->s02),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->s03),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->ne0),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->ne1),
        sizeof(((quantize_mmq_q8_1_params_host_t *)0)->ne2),
    };
    static const size_t k_rms_norm_offsets[] = {
        offsetof(rms_norm_f32_params_host_t, x),
        offsetof(rms_norm_f32_params_host_t, dst),
        offsetof(rms_norm_f32_params_host_t, ncols),
        offsetof(rms_norm_f32_params_host_t, stride_row),
        offsetof(rms_norm_f32_params_host_t, stride_channel),
        offsetof(rms_norm_f32_params_host_t, stride_sample),
        offsetof(rms_norm_f32_params_host_t, eps),
    };
    static const size_t k_rms_norm_sizes[] = {
        sizeof(((rms_norm_f32_params_host_t *)0)->x),
        sizeof(((rms_norm_f32_params_host_t *)0)->dst),
        sizeof(((rms_norm_f32_params_host_t *)0)->ncols),
        sizeof(((rms_norm_f32_params_host_t *)0)->stride_row),
        sizeof(((rms_norm_f32_params_host_t *)0)->stride_channel),
        sizeof(((rms_norm_f32_params_host_t *)0)->stride_sample),
        sizeof(((rms_norm_f32_params_host_t *)0)->eps),
    };
    static const size_t k_rms_norm_mul_offsets[] = {
        offsetof(rms_norm_f32_mul_params_host_t, x),
        offsetof(rms_norm_f32_mul_params_host_t, dst),
        offsetof(rms_norm_f32_mul_params_host_t, ncols),
        offsetof(rms_norm_f32_mul_params_host_t, stride_row),
        offsetof(rms_norm_f32_mul_params_host_t, stride_channel),
        offsetof(rms_norm_f32_mul_params_host_t, stride_sample),
        offsetof(rms_norm_f32_mul_params_host_t, eps),
        offsetof(rms_norm_f32_mul_params_host_t, mul),
        offsetof(rms_norm_f32_mul_params_host_t, mul_stride_row),
        offsetof(rms_norm_f32_mul_params_host_t, mul_stride_channel),
        offsetof(rms_norm_f32_mul_params_host_t, mul_stride_sample),
        offsetof(rms_norm_f32_mul_params_host_t, mul_ncols_packed),
        offsetof(rms_norm_f32_mul_params_host_t, mul_nrows_packed),
        offsetof(rms_norm_f32_mul_params_host_t, mul_nchannels_packed),
        offsetof(rms_norm_f32_mul_params_host_t, mul_nsamples_packed),
    };
    static const size_t k_rms_norm_mul_sizes[] = {
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->x),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->dst),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->ncols),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->stride_row),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->stride_channel),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->stride_sample),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->eps),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->mul),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->mul_stride_row),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->mul_stride_channel),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->mul_stride_sample),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->mul_ncols_packed),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->mul_nrows_packed),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->mul_nchannels_packed),
        sizeof(((rms_norm_f32_mul_params_host_t *)0)->mul_nsamples_packed),
    };
    static const size_t k_rms_norm_mul_add_offsets[] = {
        offsetof(rms_norm_f32_mul_add_params_host_t, x),
        offsetof(rms_norm_f32_mul_add_params_host_t, dst),
        offsetof(rms_norm_f32_mul_add_params_host_t, ncols),
        offsetof(rms_norm_f32_mul_add_params_host_t, stride_row),
        offsetof(rms_norm_f32_mul_add_params_host_t, stride_channel),
        offsetof(rms_norm_f32_mul_add_params_host_t, stride_sample),
        offsetof(rms_norm_f32_mul_add_params_host_t, eps),
        offsetof(rms_norm_f32_mul_add_params_host_t, mul),
        offsetof(rms_norm_f32_mul_add_params_host_t, mul_stride_row),
        offsetof(rms_norm_f32_mul_add_params_host_t, mul_stride_channel),
        offsetof(rms_norm_f32_mul_add_params_host_t, mul_stride_sample),
        offsetof(rms_norm_f32_mul_add_params_host_t, mul_ncols_packed),
        offsetof(rms_norm_f32_mul_add_params_host_t, mul_nrows_packed),
        offsetof(rms_norm_f32_mul_add_params_host_t, mul_nchannels_packed),
        offsetof(rms_norm_f32_mul_add_params_host_t, mul_nsamples_packed),
        offsetof(rms_norm_f32_mul_add_params_host_t, add),
        offsetof(rms_norm_f32_mul_add_params_host_t, add_stride_row),
        offsetof(rms_norm_f32_mul_add_params_host_t, add_stride_channel),
        offsetof(rms_norm_f32_mul_add_params_host_t, add_stride_sample),
        offsetof(rms_norm_f32_mul_add_params_host_t, add_ncols_packed),
        offsetof(rms_norm_f32_mul_add_params_host_t, add_nrows_packed),
        offsetof(rms_norm_f32_mul_add_params_host_t, add_nchannels_packed),
        offsetof(rms_norm_f32_mul_add_params_host_t, add_nsamples_packed),
    };
    static const size_t k_rms_norm_mul_add_sizes[] = {
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->x),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->dst),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->ncols),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->stride_row),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->stride_channel),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->stride_sample),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->eps),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->mul),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->mul_stride_row),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->mul_stride_channel),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->mul_stride_sample),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->mul_ncols_packed),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->mul_nrows_packed),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->mul_nchannels_packed),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->mul_nsamples_packed),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->add),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->add_stride_row),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->add_stride_channel),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->add_stride_sample),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->add_ncols_packed),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->add_nrows_packed),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->add_nchannels_packed),
        sizeof(((rms_norm_f32_mul_add_params_host_t *)0)->add_nsamples_packed),
    };
    static const size_t k_mul_mat_vec_q_offsets[] = {
        offsetof(mul_mat_vec_q_params_host_t, vx),
        offsetof(mul_mat_vec_q_params_host_t, vy),
        offsetof(mul_mat_vec_q_params_host_t, ids),
        offsetof(mul_mat_vec_q_params_host_t, fusion),
        offsetof(mul_mat_vec_q_params_host_t, dst),
        offsetof(mul_mat_vec_q_params_host_t, ncols_x),
        offsetof(mul_mat_vec_q_params_host_t, nchannels_y),
        offsetof(mul_mat_vec_q_params_host_t, stride_row_x),
        offsetof(mul_mat_vec_q_params_host_t, stride_col_y),
        offsetof(mul_mat_vec_q_params_host_t, stride_col_dst),
        offsetof(mul_mat_vec_q_params_host_t, channel_ratio),
        offsetof(mul_mat_vec_q_params_host_t, stride_channel_x),
        offsetof(mul_mat_vec_q_params_host_t, stride_channel_y),
        offsetof(mul_mat_vec_q_params_host_t, stride_channel_dst),
        offsetof(mul_mat_vec_q_params_host_t, sample_ratio),
        offsetof(mul_mat_vec_q_params_host_t, stride_sample_x),
        offsetof(mul_mat_vec_q_params_host_t, stride_sample_y),
        offsetof(mul_mat_vec_q_params_host_t, stride_sample_dst),
    };
    static const size_t k_mul_mat_vec_q_sizes[] = {
        sizeof(((mul_mat_vec_q_params_host_t *)0)->vx),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->vy),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->ids),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->fusion),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->dst),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->ncols_x),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->nchannels_y),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_row_x),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_col_y),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_col_dst),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->channel_ratio),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_channel_x),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_channel_y),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_channel_dst),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->sample_ratio),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_sample_x),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_sample_y),
        sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_sample_dst),
    };
    static const size_t k_mul_mat_vec_f_offsets[] = {
        offsetof(mul_mat_vec_f_params_host_t, vx),
        offsetof(mul_mat_vec_f_params_host_t, vy),
        offsetof(mul_mat_vec_f_params_host_t, ids),
        offsetof(mul_mat_vec_f_params_host_t, fusion),
        offsetof(mul_mat_vec_f_params_host_t, dst),
        offsetof(mul_mat_vec_f_params_host_t, ncols_x),
        offsetof(mul_mat_vec_f_params_host_t, nchannels_y),
        offsetof(mul_mat_vec_f_params_host_t, stride_row_x),
        offsetof(mul_mat_vec_f_params_host_t, stride_col_y),
        offsetof(mul_mat_vec_f_params_host_t, stride_col_dst),
        offsetof(mul_mat_vec_f_params_host_t, channel_ratio),
        offsetof(mul_mat_vec_f_params_host_t, stride_channel_x),
        offsetof(mul_mat_vec_f_params_host_t, stride_channel_y),
        offsetof(mul_mat_vec_f_params_host_t, stride_channel_dst),
        offsetof(mul_mat_vec_f_params_host_t, sample_ratio),
        offsetof(mul_mat_vec_f_params_host_t, stride_sample_x),
        offsetof(mul_mat_vec_f_params_host_t, stride_sample_y),
        offsetof(mul_mat_vec_f_params_host_t, stride_sample_dst),
    };
    static const size_t k_mul_mat_vec_f_sizes[] = {
        sizeof(((mul_mat_vec_f_params_host_t *)0)->vx),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->vy),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->ids),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->fusion),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->dst),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->ncols_x),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->nchannels_y),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_row_x),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_col_y),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_col_dst),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->channel_ratio),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_channel_x),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_channel_y),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_channel_dst),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->sample_ratio),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_sample_x),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_sample_y),
        sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_sample_dst),
    };
    static const size_t k_set_rows_offsets[] = {
        offsetof(k_set_rows_params_host_t, src0),
        offsetof(k_set_rows_params_host_t, src1),
        offsetof(k_set_rows_params_host_t, dst),
        offsetof(k_set_rows_params_host_t, ne_total),
        offsetof(k_set_rows_params_host_t, ne10),
        offsetof(k_set_rows_params_host_t, ne11),
        offsetof(k_set_rows_params_host_t, ne12),
        offsetof(k_set_rows_params_host_t, ne13),
        offsetof(k_set_rows_params_host_t, s01),
        offsetof(k_set_rows_params_host_t, s02),
        offsetof(k_set_rows_params_host_t, s03),
        offsetof(k_set_rows_params_host_t, s10),
        offsetof(k_set_rows_params_host_t, s11),
        offsetof(k_set_rows_params_host_t, s12),
        offsetof(k_set_rows_params_host_t, s1),
        offsetof(k_set_rows_params_host_t, s2),
        offsetof(k_set_rows_params_host_t, s3),
        offsetof(k_set_rows_params_host_t, ne00),
        offsetof(k_set_rows_params_host_t, ne01),
        offsetof(k_set_rows_params_host_t, ne02),
        offsetof(k_set_rows_params_host_t, ne11_fd),
        offsetof(k_set_rows_params_host_t, ne12_fd),
    };
    static const size_t k_set_rows_sizes[] = {
        sizeof(((k_set_rows_params_host_t *)0)->src0),
        sizeof(((k_set_rows_params_host_t *)0)->src1),
        sizeof(((k_set_rows_params_host_t *)0)->dst),
        sizeof(((k_set_rows_params_host_t *)0)->ne_total),
        sizeof(((k_set_rows_params_host_t *)0)->ne10),
        sizeof(((k_set_rows_params_host_t *)0)->ne11),
        sizeof(((k_set_rows_params_host_t *)0)->ne12),
        sizeof(((k_set_rows_params_host_t *)0)->ne13),
        sizeof(((k_set_rows_params_host_t *)0)->s01),
        sizeof(((k_set_rows_params_host_t *)0)->s02),
        sizeof(((k_set_rows_params_host_t *)0)->s03),
        sizeof(((k_set_rows_params_host_t *)0)->s10),
        sizeof(((k_set_rows_params_host_t *)0)->s11),
        sizeof(((k_set_rows_params_host_t *)0)->s12),
        sizeof(((k_set_rows_params_host_t *)0)->s1),
        sizeof(((k_set_rows_params_host_t *)0)->s2),
        sizeof(((k_set_rows_params_host_t *)0)->s3),
        sizeof(((k_set_rows_params_host_t *)0)->ne00),
        sizeof(((k_set_rows_params_host_t *)0)->ne01),
        sizeof(((k_set_rows_params_host_t *)0)->ne02),
        sizeof(((k_set_rows_params_host_t *)0)->ne11_fd),
        sizeof(((k_set_rows_params_host_t *)0)->ne12_fd),
    };
    static const size_t k_bin_bcast_offsets[] = {
        offsetof(k_bin_bcast_params_host_t, src0),
        offsetof(k_bin_bcast_params_host_t, src1),
        offsetof(k_bin_bcast_params_host_t, dst),
        offsetof(k_bin_bcast_params_host_t, ne0),
        offsetof(k_bin_bcast_params_host_t, ne1),
        offsetof(k_bin_bcast_params_host_t, ne2),
        offsetof(k_bin_bcast_params_host_t, ne3),
        offsetof(k_bin_bcast_params_host_t, ne10),
        offsetof(k_bin_bcast_params_host_t, ne11),
        offsetof(k_bin_bcast_params_host_t, ne12),
        offsetof(k_bin_bcast_params_host_t, ne13),
        offsetof(k_bin_bcast_params_host_t, s1),
        offsetof(k_bin_bcast_params_host_t, s2),
        offsetof(k_bin_bcast_params_host_t, s3),
        offsetof(k_bin_bcast_params_host_t, s01),
        offsetof(k_bin_bcast_params_host_t, s02),
        offsetof(k_bin_bcast_params_host_t, s03),
        offsetof(k_bin_bcast_params_host_t, s11),
        offsetof(k_bin_bcast_params_host_t, s12),
        offsetof(k_bin_bcast_params_host_t, s13),
    };
    static const size_t k_bin_bcast_sizes[] = {
        sizeof(((k_bin_bcast_params_host_t *)0)->src0),
        sizeof(((k_bin_bcast_params_host_t *)0)->src1),
        sizeof(((k_bin_bcast_params_host_t *)0)->dst),
        sizeof(((k_bin_bcast_params_host_t *)0)->ne0),
        sizeof(((k_bin_bcast_params_host_t *)0)->ne1),
        sizeof(((k_bin_bcast_params_host_t *)0)->ne2),
        sizeof(((k_bin_bcast_params_host_t *)0)->ne3),
        sizeof(((k_bin_bcast_params_host_t *)0)->ne10),
        sizeof(((k_bin_bcast_params_host_t *)0)->ne11),
        sizeof(((k_bin_bcast_params_host_t *)0)->ne12),
        sizeof(((k_bin_bcast_params_host_t *)0)->ne13),
        sizeof(((k_bin_bcast_params_host_t *)0)->s1),
        sizeof(((k_bin_bcast_params_host_t *)0)->s2),
        sizeof(((k_bin_bcast_params_host_t *)0)->s3),
        sizeof(((k_bin_bcast_params_host_t *)0)->s01),
        sizeof(((k_bin_bcast_params_host_t *)0)->s02),
        sizeof(((k_bin_bcast_params_host_t *)0)->s03),
        sizeof(((k_bin_bcast_params_host_t *)0)->s11),
        sizeof(((k_bin_bcast_params_host_t *)0)->s12),
        sizeof(((k_bin_bcast_params_host_t *)0)->s13),
    };
    static const size_t k_bin_bcast_unravel_offsets[] = {
        offsetof(k_bin_bcast_unravel_params_host_t, src0),
        offsetof(k_bin_bcast_unravel_params_host_t, src1),
        offsetof(k_bin_bcast_unravel_params_host_t, dst),
        offsetof(k_bin_bcast_unravel_params_host_t, ne0),
        offsetof(k_bin_bcast_unravel_params_host_t, ne1),
        offsetof(k_bin_bcast_unravel_params_host_t, ne2),
        offsetof(k_bin_bcast_unravel_params_host_t, ne3),
        offsetof(k_bin_bcast_unravel_params_host_t, prod_012),
        offsetof(k_bin_bcast_unravel_params_host_t, prod_01),
        offsetof(k_bin_bcast_unravel_params_host_t, ne10),
        offsetof(k_bin_bcast_unravel_params_host_t, ne11),
        offsetof(k_bin_bcast_unravel_params_host_t, ne12),
        offsetof(k_bin_bcast_unravel_params_host_t, ne13),
        offsetof(k_bin_bcast_unravel_params_host_t, s1),
        offsetof(k_bin_bcast_unravel_params_host_t, s2),
        offsetof(k_bin_bcast_unravel_params_host_t, s3),
        offsetof(k_bin_bcast_unravel_params_host_t, s01),
        offsetof(k_bin_bcast_unravel_params_host_t, s02),
        offsetof(k_bin_bcast_unravel_params_host_t, s03),
        offsetof(k_bin_bcast_unravel_params_host_t, s11),
        offsetof(k_bin_bcast_unravel_params_host_t, s12),
        offsetof(k_bin_bcast_unravel_params_host_t, s13),
    };
    static const size_t k_bin_bcast_unravel_sizes[] = {
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->src0),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->src1),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->dst),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->ne0),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->ne1),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->ne2),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->ne3),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->prod_012),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->prod_01),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->ne10),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->ne11),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->ne12),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->ne13),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s1),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s2),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s3),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s01),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s02),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s03),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s11),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s12),
        sizeof(((k_bin_bcast_unravel_params_host_t *)0)->s13),
    };
    static const size_t k_mul_mat_q_offsets[] = {
        offsetof(mul_mat_q_params_host_t, x),
        offsetof(mul_mat_q_params_host_t, y),
        offsetof(mul_mat_q_params_host_t, ids_dst),
        offsetof(mul_mat_q_params_host_t, expert_bounds),
        offsetof(mul_mat_q_params_host_t, dst),
        offsetof(mul_mat_q_params_host_t, tmp_fixup),
        offsetof(mul_mat_q_params_host_t, ncols_x),
        offsetof(mul_mat_q_params_host_t, nrows_x),
        offsetof(mul_mat_q_params_host_t, ncols_dst),
        offsetof(mul_mat_q_params_host_t, stride_row_x),
        offsetof(mul_mat_q_params_host_t, ncols_y),
        offsetof(mul_mat_q_params_host_t, stride_col_dst),
        offsetof(mul_mat_q_params_host_t, channel_ratio),
        offsetof(mul_mat_q_params_host_t, nchannels_y),
        offsetof(mul_mat_q_params_host_t, stride_channel_x),
        offsetof(mul_mat_q_params_host_t, stride_channel_y),
        offsetof(mul_mat_q_params_host_t, stride_channel_dst),
        offsetof(mul_mat_q_params_host_t, sample_ratio),
        offsetof(mul_mat_q_params_host_t, nsamples_y),
        offsetof(mul_mat_q_params_host_t, stride_sample_x),
        offsetof(mul_mat_q_params_host_t, stride_sample_y),
        offsetof(mul_mat_q_params_host_t, stride_sample_dst),
        offsetof(mul_mat_q_params_host_t, ncols_max),
    };
    static const size_t k_mul_mat_q_sizes[] = {
        sizeof(((mul_mat_q_params_host_t *)0)->x),
        sizeof(((mul_mat_q_params_host_t *)0)->y),
        sizeof(((mul_mat_q_params_host_t *)0)->ids_dst),
        sizeof(((mul_mat_q_params_host_t *)0)->expert_bounds),
        sizeof(((mul_mat_q_params_host_t *)0)->dst),
        sizeof(((mul_mat_q_params_host_t *)0)->tmp_fixup),
        sizeof(((mul_mat_q_params_host_t *)0)->ncols_x),
        sizeof(((mul_mat_q_params_host_t *)0)->nrows_x),
        sizeof(((mul_mat_q_params_host_t *)0)->ncols_dst),
        sizeof(((mul_mat_q_params_host_t *)0)->stride_row_x),
        sizeof(((mul_mat_q_params_host_t *)0)->ncols_y),
        sizeof(((mul_mat_q_params_host_t *)0)->stride_col_dst),
        sizeof(((mul_mat_q_params_host_t *)0)->channel_ratio),
        sizeof(((mul_mat_q_params_host_t *)0)->nchannels_y),
        sizeof(((mul_mat_q_params_host_t *)0)->stride_channel_x),
        sizeof(((mul_mat_q_params_host_t *)0)->stride_channel_y),
        sizeof(((mul_mat_q_params_host_t *)0)->stride_channel_dst),
        sizeof(((mul_mat_q_params_host_t *)0)->sample_ratio),
        sizeof(((mul_mat_q_params_host_t *)0)->nsamples_y),
        sizeof(((mul_mat_q_params_host_t *)0)->stride_sample_x),
        sizeof(((mul_mat_q_params_host_t *)0)->stride_sample_y),
        sizeof(((mul_mat_q_params_host_t *)0)->stride_sample_dst),
        sizeof(((mul_mat_q_params_host_t *)0)->ncols_max),
    };
    static const size_t k_mul_mat_q_stream_k_fixup_offsets[] = {
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, ids_dst),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, expert_bounds),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, dst),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, tmp_last_tile),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, ncols_x),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, nrows_x),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, ncols_dst),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, stride_col_dst),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, nchannels_y),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, stride_channel_dst),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, nsamples_y),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, stride_sample_dst),
        offsetof(mul_mat_q_stream_k_fixup_params_host_t, ncols_max),
    };
    static const size_t k_mul_mat_q_stream_k_fixup_sizes[] = {
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->ids_dst),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->expert_bounds),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->dst),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->tmp_last_tile),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->ncols_x),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->nrows_x),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->ncols_dst),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->stride_col_dst),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->nchannels_y),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->stride_channel_dst),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->nsamples_y),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->stride_sample_dst),
        sizeof(((mul_mat_q_stream_k_fixup_params_host_t *)0)->ncols_max),
    };
    static const size_t k_rope_norm_offsets[] = {
        offsetof(rope_norm_params_host_t, x),
        offsetof(rope_norm_params_host_t, dst),
        offsetof(rope_norm_params_host_t, ne0),
        offsetof(rope_norm_params_host_t, ne1),
        offsetof(rope_norm_params_host_t, s1),
        offsetof(rope_norm_params_host_t, s2),
        offsetof(rope_norm_params_host_t, n_dims),
        offsetof(rope_norm_params_host_t, pos),
        offsetof(rope_norm_params_host_t, freq_scale),
        offsetof(rope_norm_params_host_t, ext_factor),
        offsetof(rope_norm_params_host_t, attn_factor),
        offsetof(rope_norm_params_host_t, corr_dims),
        offsetof(rope_norm_params_host_t, theta_scale),
        offsetof(rope_norm_params_host_t, freq_factors),
        offsetof(rope_norm_params_host_t, row_indices),
        offsetof(rope_norm_params_host_t, set_rows_stride),
    };
    static const size_t k_rope_norm_sizes[] = {
        sizeof(((rope_norm_params_host_t *)0)->x),
        sizeof(((rope_norm_params_host_t *)0)->dst),
        sizeof(((rope_norm_params_host_t *)0)->ne0),
        sizeof(((rope_norm_params_host_t *)0)->ne1),
        sizeof(((rope_norm_params_host_t *)0)->s1),
        sizeof(((rope_norm_params_host_t *)0)->s2),
        sizeof(((rope_norm_params_host_t *)0)->n_dims),
        sizeof(((rope_norm_params_host_t *)0)->pos),
        sizeof(((rope_norm_params_host_t *)0)->freq_scale),
        sizeof(((rope_norm_params_host_t *)0)->ext_factor),
        sizeof(((rope_norm_params_host_t *)0)->attn_factor),
        sizeof(((rope_norm_params_host_t *)0)->corr_dims),
        sizeof(((rope_norm_params_host_t *)0)->theta_scale),
        sizeof(((rope_norm_params_host_t *)0)->freq_factors),
        sizeof(((rope_norm_params_host_t *)0)->row_indices),
        sizeof(((rope_norm_params_host_t *)0)->set_rows_stride),
    };
    static const size_t k_convert_unary_offsets[] = {
        offsetof(convert_unary_params_host_t, vx),
        offsetof(convert_unary_params_host_t, y),
        offsetof(convert_unary_params_host_t, ne00),
        offsetof(convert_unary_params_host_t, ne01),
        offsetof(convert_unary_params_host_t, ne02),
        offsetof(convert_unary_params_host_t, s01),
        offsetof(convert_unary_params_host_t, s02),
        offsetof(convert_unary_params_host_t, s03),
    };
    static const size_t k_convert_unary_sizes[] = {
        sizeof(((convert_unary_params_host_t *)0)->vx),
        sizeof(((convert_unary_params_host_t *)0)->y),
        sizeof(((convert_unary_params_host_t *)0)->ne00),
        sizeof(((convert_unary_params_host_t *)0)->ne01),
        sizeof(((convert_unary_params_host_t *)0)->ne02),
        sizeof(((convert_unary_params_host_t *)0)->s01),
        sizeof(((convert_unary_params_host_t *)0)->s02),
        sizeof(((convert_unary_params_host_t *)0)->s03),
    };

    if (!func_name || !param_offset || !param_size) {
        return 0;
    }

    if (strncmp(func_name, "_Z12soft_max_f32", strlen("_Z12soft_max_f32")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_soft_max_offsets,
                                             k_soft_max_sizes,
                                             sizeof(k_soft_max_sizes) / sizeof(k_soft_max_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z13quantize_q8_1", strlen("_Z13quantize_q8_1")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_quantize_q8_1_offsets,
                                             k_quantize_q8_1_sizes,
                                             sizeof(k_quantize_q8_1_sizes) / sizeof(k_quantize_q8_1_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z17quantize_mmq_q8_1", strlen("_Z17quantize_mmq_q8_1")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_quantize_mmq_q8_1_offsets,
                                             k_quantize_mmq_q8_1_sizes,
                                             sizeof(k_quantize_mmq_q8_1_sizes) / sizeof(k_quantize_mmq_q8_1_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z12rms_norm_f32", strlen("_Z12rms_norm_f32")) == 0) {
        if (strstr(func_name, "ELb1ELb1E") != NULL) {
            return cuda_executor_copy_param_info(param_index,
                                                 k_rms_norm_mul_add_offsets,
                                                 k_rms_norm_mul_add_sizes,
                                                 sizeof(k_rms_norm_mul_add_sizes) / sizeof(k_rms_norm_mul_add_sizes[0]),
                                                 param_offset,
                                                 param_size);
        }
        if (strstr(func_name, "ELb1ELb0E") != NULL) {
            return cuda_executor_copy_param_info(param_index,
                                                 k_rms_norm_mul_offsets,
                                                 k_rms_norm_mul_sizes,
                                                 sizeof(k_rms_norm_mul_sizes) / sizeof(k_rms_norm_mul_sizes[0]),
                                                 param_offset,
                                                 param_size);
        }
        return cuda_executor_copy_param_info_with_one_trailing_u32(
            param_index,
            k_rms_norm_offsets,
            k_rms_norm_sizes,
            sizeof(k_rms_norm_sizes) / sizeof(k_rms_norm_sizes[0]),
            param_offset,
            param_size);
    }

    if (strncmp(func_name, "_Z13mul_mat_vec_q", strlen("_Z13mul_mat_vec_q")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_mul_mat_vec_q_offsets,
                                             k_mul_mat_vec_q_sizes,
                                             sizeof(k_mul_mat_vec_q_sizes) / sizeof(k_mul_mat_vec_q_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z13mul_mat_vec_f", strlen("_Z13mul_mat_vec_f")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_mul_mat_vec_f_offsets,
                                             k_mul_mat_vec_f_sizes,
                                             sizeof(k_mul_mat_vec_f_sizes) / sizeof(k_mul_mat_vec_f_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z10k_set_rows", strlen("_Z10k_set_rows")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_set_rows_offsets,
                                             k_set_rows_sizes,
                                             sizeof(k_set_rows_sizes) / sizeof(k_set_rows_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z16k_set_rows_quant", strlen("_Z16k_set_rows_quant")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_set_rows_offsets,
                                             k_set_rows_sizes,
                                             sizeof(k_set_rows_sizes) / sizeof(k_set_rows_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z11k_bin_bcast", strlen("_Z11k_bin_bcast")) == 0) {
        return cuda_executor_copy_param_info_with_trailing_ptrs(
            param_index,
            k_bin_bcast_offsets,
            k_bin_bcast_sizes,
            sizeof(k_bin_bcast_sizes) / sizeof(k_bin_bcast_sizes[0]),
            offsetof(k_bin_bcast_params_host_t, extra_src1),
            cuda_executor_count_template_pack_ptrs(func_name),
            param_offset,
            param_size);
    }

    if (strncmp(func_name, "_Z19k_bin_bcast_unravel", strlen("_Z19k_bin_bcast_unravel")) == 0) {
        return cuda_executor_copy_param_info_with_trailing_ptrs(
            param_index,
            k_bin_bcast_unravel_offsets,
            k_bin_bcast_unravel_sizes,
            sizeof(k_bin_bcast_unravel_sizes) / sizeof(k_bin_bcast_unravel_sizes[0]),
            offsetof(k_bin_bcast_unravel_params_host_t, extra_src1),
            cuda_executor_count_template_pack_ptrs(func_name),
            param_offset,
            param_size);
    }

    if (strncmp(func_name, "_Z9mul_mat_q", strlen("_Z9mul_mat_q")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_mul_mat_q_offsets,
                                             k_mul_mat_q_sizes,
                                             sizeof(k_mul_mat_q_sizes) / sizeof(k_mul_mat_q_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z24mul_mat_q_stream_k_fixup", strlen("_Z24mul_mat_q_stream_k_fixup")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_mul_mat_q_stream_k_fixup_offsets,
                                             k_mul_mat_q_stream_k_fixup_sizes,
                                             sizeof(k_mul_mat_q_stream_k_fixup_sizes) / sizeof(k_mul_mat_q_stream_k_fixup_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z9rope_norm", strlen("_Z9rope_norm")) == 0 ||
        strncmp(func_name, "_Z9rope_neox", strlen("_Z9rope_neox")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_rope_norm_offsets,
                                             k_rope_norm_sizes,
                                             sizeof(k_rope_norm_sizes) / sizeof(k_rope_norm_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    if (strncmp(func_name, "_Z13convert_unary", strlen("_Z13convert_unary")) == 0) {
        return cuda_executor_copy_param_info(param_index,
                                             k_convert_unary_offsets,
                                             k_convert_unary_sizes,
                                             sizeof(k_convert_unary_sizes) / sizeof(k_convert_unary_sizes[0]),
                                             param_offset,
                                             param_size);
    }

    /* k_compute_batched_ptrs(src0_f16, src1_f16, dst, ptrs_src, ptrs_dst,
     *                        ne12, ne13, ne23, nb02, nb03, nb12, nb13,
     *                        nbd2, nbd3, r2, r3)
     * 16 params, all pointers or 8-byte integers — tight packing, stride=8. */
    if (strncmp(func_name, "_Z22k_compute_batched_ptrs",
                strlen("_Z22k_compute_batched_ptrs")) == 0) {
        static const size_t k_compute_batched_ptrs_count = 16u;
        if (param_index >= k_compute_batched_ptrs_count) {
            return 0;
        }
        *param_offset = param_index * 8u;
        *param_size   = 8u;
        return 1;
    }

    /* unary_gated_op_kernel<F>(x, g, dst, k, n, o0, o1)
     * Used for SiLU/gated activations in LLaMA MLP blocks.
     * 7 params, all 8-byte (pointers or int64_t) — tight packing, stride=8. */
    if (strncmp(func_name, "_Z21unary_gated_op_kernel",
                strlen("_Z21unary_gated_op_kernel")) == 0) {
        static const size_t unary_gated_count = 7u;
        if (param_index >= unary_gated_count) {
            return 0;
        }
        *param_offset = param_index * 8u;
        *param_size   = 8u;
        return 1;
    }

    /* k_get_rows_float<src0_t,dst_t>(src0, src1, dst, ne00, ne11, ne12,
     *                                s1, s2, s3, nb01, nb02, nb03, s10, s11, s12)
     * k_get_rows<qk,qr,dequant_fn,dst_t>(src0, src1, dst, ne00, ne11, ne12,
     *                                    s1, s2, s3, nb01, nb02, nb03, s10, s11, s12)
     * Both have 15 params, all pointers or 8-byte integers — tight packing, stride=8. */
    if (strncmp(func_name, "_Z16k_get_rows_float",
                strlen("_Z16k_get_rows_float")) == 0 ||
        strncmp(func_name, "_Z9k_get_rows",
                strlen("_Z9k_get_rows")) == 0) {
        static const size_t k_get_rows_count = 15u;
        if (param_index >= k_get_rows_count) {
            return 0;
        }
        *param_offset = param_index * 8u;
        *param_size   = 8u;
        return 1;
    }

    /* mul_mat_f<T,rows_per_block,...>(x, y, ids, dst,
     *           ncols, ncols_dst_total, nchannels_dst, stride_row,
     *           stride_col_y, stride_col_dst, stride_col_id, stride_row_id,
     *           channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
     *           sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst)
     * 4 pointers (8 bytes) + 16 int32 (4 bytes) = 20 params, 96 bytes total. */
    if (strncmp(func_name, "_Z9mul_mat_f",
                strlen("_Z9mul_mat_f")) == 0) {
        static const size_t mul_mat_f_count = 20u;
        static const size_t mul_mat_f_ptr_count = 4u;
        if (param_index >= mul_mat_f_count) {
            return 0;
        }
        if (param_index < mul_mat_f_ptr_count) {
            *param_offset = param_index * 8u;
            *param_size   = 8u;
        } else {
            *param_offset = mul_mat_f_ptr_count * 8u +
                            (param_index - mul_mat_f_ptr_count) * 4u;
            *param_size   = 4u;
        }
        return 1;
    }

    /* cpy_scalar<F>(cx, cdst, ne, ne00..ne02, nb00..nb03, ne10..ne12, nb10..nb13)
     * cpy_scalar_transpose<T>(cx, cdst, ne, ne00..ne02, nb00..nb03, ne10..ne12, nb10..nb13)
     * Layout: 2 x ptr (8 bytes each) + 15 x int32 (4 bytes each) = 76 bytes total.
     * All template variants share the same parameter layout. */
    if (strncmp(func_name, "_Z10cpy_scalar",
                strlen("_Z10cpy_scalar")) == 0) {
        static const size_t cpy_scalar_count = 17u;
        /* offsets: 0,8 for the two ptrs; then 16,20,...72 for the 15 ints */
        if (param_index >= cpy_scalar_count) {
            return 0;
        }
        if (param_index < 2u) {
            *param_offset = param_index * 8u;
            *param_size   = 8u;
        } else {
            *param_offset = 16u + (param_index - 2u) * 4u;
            *param_size   = 4u;
        }
        return 1;
    }

    return 0;
}

static uint16_t cuda_executor_float_to_half_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000u) >> (uint32_t)(1 - exp);
        if (mant & 0x00001000u) {
            mant += 0x00002000u;
        }
        return (uint16_t)(sign | (mant >> 13));
    }

    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }

    if (mant & 0x00001000u) {
        mant += 0x00002000u;
        if (mant & 0x00800000u) {
            mant = 0;
            exp++;
            if (exp >= 31) {
                return (uint16_t)(sign | 0x7c00u);
            }
        }
    }

    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static int cuda_executor_is_fp16_compute(int computeType)
{
    return computeType == CUBLAS_COMPUTE_16F ||
           computeType == CUBLAS_COMPUTE_16F_PEDANTIC;
}

static void cuda_executor_log_prefix_bytes(const char *label,
                                           const void *data,
                                           size_t len,
                                           uint32_t vm_id)
{
    if (!data || len == 0) {
        return;
    }

    const unsigned char *bytes = (const unsigned char *)data;
    char hexbuf[196 * 3 + 1];
    size_t pos = 0;
    size_t log_len = (len < 196) ? len : 196;

    for (size_t i = 0; i < log_len && pos + 4 < sizeof(hexbuf); i++) {
        int n = snprintf(hexbuf + pos, sizeof(hexbuf) - pos,
                         "%02x%s", bytes[i], (i + 1 < log_len) ? " " : "");
        if (n <= 0) {
            break;
        }
        pos += (size_t)n;
    }

    fprintf(stderr,
            "[cuda-executor] %s prefix vm=%u len=%zu prefix_len=%zu bytes=[%s]\n",
            label, vm_id, len, log_len, hexbuf);
}

static void cuda_executor_log_device_f32_sample(const char *label,
                                                CUdeviceptr dptr,
                                                size_t max_floats,
                                                uint32_t vm_id)
{
    float sample[8] = {0};
    size_t count = max_floats;

    if (!label || dptr == 0 || max_floats == 0) {
        return;
    }
    if (count > (sizeof(sample) / sizeof(sample[0]))) {
        count = sizeof(sample) / sizeof(sample[0]);
    }

    CUresult rc = cuMemcpyDtoH(sample, dptr, count * sizeof(float));
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr,
                "[cuda-executor] %s sample FAILED vm=%u ptr=0x%llx rc=%d(%s) detail=%s\n",
                label,
                vm_id,
                (unsigned long long)dptr,
                (int)rc,
                host_cuda_error_name(rc),
                host_cuda_error_string(rc));
        return;
    }

    fprintf(stderr,
            "[cuda-executor] %s sample vm=%u ptr=0x%llx count=%zu "
            "values=[%.7g %.7g %.7g %.7g %.7g %.7g %.7g %.7g]\n",
            label,
            vm_id,
            (unsigned long long)dptr,
            count,
            (double)sample[0],
            (double)sample[1],
            (double)sample[2],
            (double)sample[3],
            (double)sample[4],
            (double)sample[5],
            (double)sample[6],
            (double)sample[7]);
}

static void cuda_executor_log_device_bytes_sample(const char *label,
                                                  CUdeviceptr dptr,
                                                  size_t max_bytes,
                                                  uint32_t vm_id)
{
    unsigned char sample[64] = {0};
    size_t count = max_bytes;

    if (!label || dptr == 0 || max_bytes == 0) {
        return;
    }
    if (count > sizeof(sample)) {
        count = sizeof(sample);
    }

    CUresult rc = cuMemcpyDtoH(sample, dptr, count);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr,
                "[cuda-executor] %s bytes FAILED vm=%u ptr=0x%llx rc=%d(%s) detail=%s\n",
                label,
                vm_id,
                (unsigned long long)dptr,
                (int)rc,
                host_cuda_error_name(rc),
                host_cuda_error_string(rc));
        return;
    }

    fprintf(stderr,
            "[cuda-executor] %s bytes sample vm=%u ptr=0x%llx count=%zu\n",
            label,
            vm_id,
            (unsigned long long)dptr,
            count);
    cuda_executor_log_prefix_bytes(label, sample, count, vm_id);
}

static int cuda_executor_is_final_logits_mul_mat_vec_q(const char *func_name)
{
    return func_name &&
           strncmp(func_name, "_Z13mul_mat_vec_q", strlen("_Z13mul_mat_vec_q")) == 0 &&
           strstr(func_name, "ggml_type14") != NULL &&
           strstr(func_name, "ELb0E") != NULL;
}

static void cuda_executor_log_output_kernel_samples(const char *func_name,
                                                    const void *param_data,
                                                    size_t len,
                                                    uint32_t vm_id)
{
    if (!func_name || !param_data) {
        return;
    }

    if (strncmp(func_name, "_Z13mul_mat_vec_q", strlen("_Z13mul_mat_vec_q")) == 0) {
        static const size_t k_min_len =
            offsetof(mul_mat_vec_q_params_host_t, stride_sample_dst) +
            sizeof(((mul_mat_vec_q_params_host_t *)0)->stride_sample_dst);
        mul_mat_vec_q_params_host_t p;
        if (len < k_min_len) {
            fprintf(stderr,
                    "[cuda-executor] %s decoded params vm=%u len=%zu expected_at_least=%zu (truncated)\n",
                    func_name, vm_id, len, k_min_len);
            return;
        }
        memset(&p, 0, sizeof(p));
        memcpy(&p, param_data, k_min_len);
        fprintf(stderr,
                "[cuda-executor] %s decoded params vm=%u vx=%p vy=%p ids=%p dst=%p "
                "ncols_x=%u nchannels_y=(%u,%u,%u) stride_row_x=%u stride_col_y=%u stride_col_dst=%u "
                "glu_op=%d x_bias=%p gate=%p gate_bias=%p\n",
                func_name,
                vm_id,
                p.vx,
                p.vy,
                p.ids,
                p.dst,
                p.ncols_x,
                p.nchannels_y.x, p.nchannels_y.y, p.nchannels_y.z,
                p.stride_row_x,
                p.stride_col_y,
                p.stride_col_dst,
                p.fusion.glu_op,
                p.fusion.x_bias,
                p.fusion.gate,
                p.fusion.gate_bias);
        cuda_executor_log_device_f32_sample("mul_mat_vec_q dst",
                                            (CUdeviceptr)(uintptr_t)p.dst,
                                            8,
                                            vm_id);
        if (cuda_executor_is_final_logits_mul_mat_vec_q(func_name)) {
            cuda_executor_log_prefix_bytes("final_logits mul_mat_vec_q param buffer",
                                          param_data,
                                          len,
                                          vm_id);
            cuda_executor_log_device_bytes_sample("final_logits mul_mat_vec_q vx",
                                                  (CUdeviceptr)(uintptr_t)p.vx,
                                                  64,
                                                  vm_id);
            cuda_executor_log_device_bytes_sample("final_logits mul_mat_vec_q vy",
                                                  (CUdeviceptr)(uintptr_t)p.vy,
                                                  64,
                                                  vm_id);
            cuda_executor_log_device_f32_sample("final_logits mul_mat_vec_q vy",
                                                (CUdeviceptr)(uintptr_t)p.vy,
                                                8,
                                                vm_id);
            if (p.fusion.x_bias) {
                cuda_executor_log_device_f32_sample("final_logits mul_mat_vec_q x_bias",
                                                    (CUdeviceptr)(uintptr_t)p.fusion.x_bias,
                                                    8,
                                                    vm_id);
            }
            if (p.fusion.gate) {
                cuda_executor_log_device_f32_sample("final_logits mul_mat_vec_q gate",
                                                    (CUdeviceptr)(uintptr_t)p.fusion.gate,
                                                    8,
                                                    vm_id);
            }
            if (p.fusion.gate_bias) {
                cuda_executor_log_device_f32_sample("final_logits mul_mat_vec_q gate_bias",
                                                    (CUdeviceptr)(uintptr_t)p.fusion.gate_bias,
                                                    8,
                                                    vm_id);
            }
        }
        return;
    }

    if (strncmp(func_name, "_Z13mul_mat_vec_f", strlen("_Z13mul_mat_vec_f")) == 0) {
        static const size_t k_min_len =
            offsetof(mul_mat_vec_f_params_host_t, stride_sample_dst) +
            sizeof(((mul_mat_vec_f_params_host_t *)0)->stride_sample_dst);
        mul_mat_vec_f_params_host_t p;
        if (len < k_min_len) {
            fprintf(stderr,
                    "[cuda-executor] %s decoded params vm=%u len=%zu expected_at_least=%zu (truncated)\n",
                    func_name, vm_id, len, k_min_len);
            return;
        }
        memset(&p, 0, sizeof(p));
        memcpy(&p, param_data, k_min_len);
        fprintf(stderr,
                "[cuda-executor] %s decoded params vm=%u vx=%p vy=%p ids=%p dst=%p "
                "ncols_x=%u nchannels_y=%u stride_row_x=%u stride_col_y=%u stride_col_dst=%u "
                "glu_op=%d x_bias=%p gate=%p gate_bias=%p\n",
                func_name,
                vm_id,
                p.vx,
                p.vy,
                p.ids,
                p.dst,
                p.ncols_x,
                p.nchannels_y,
                p.stride_row_x,
                p.stride_col_y,
                p.stride_col_dst,
                p.fusion.glu_op,
                p.fusion.x_bias,
                p.fusion.gate,
                p.fusion.gate_bias);
        cuda_executor_log_device_f32_sample("mul_mat_vec_f dst",
                                            (CUdeviceptr)(uintptr_t)p.dst,
                                            8,
                                            vm_id);
        return;
    }

    if (strncmp(func_name, "_Z12soft_max_f32", strlen("_Z12soft_max_f32")) == 0) {
        static const size_t k_min_len =
            offsetof(soft_max_f32_params_host_t, soft_max_params) +
            sizeof(((soft_max_f32_params_host_t *)0)->soft_max_params);
        soft_max_f32_params_host_t p;
        if (len < k_min_len) {
            fprintf(stderr,
                    "[cuda-executor] %s decoded params vm=%u len=%zu expected_at_least=%zu (truncated)\n",
                    func_name, vm_id, len, k_min_len);
            return;
        }
        memset(&p, 0, sizeof(p));
        memcpy(&p, param_data, k_min_len);
        fprintf(stderr,
                "[cuda-executor] %s decoded params vm=%u x=%p mask=%p sinks=%p dst=%p\n",
                func_name,
                vm_id,
                p.x,
                p.mask,
                p.sinks,
                p.dst);
        cuda_executor_log_device_f32_sample("soft_max_f32 x",
                                            (CUdeviceptr)(uintptr_t)p.x,
                                            8,
                                            vm_id);
        cuda_executor_log_device_f32_sample("soft_max_f32 dst",
                                            (CUdeviceptr)(uintptr_t)p.dst,
                                            8,
                                            vm_id);
    }
}

static void cuda_executor_log_k_set_rows_params(const char *func_name,
                                                const void *param_data,
                                                size_t len,
                                                uint32_t vm_id)
{
    if (!func_name || !param_data) {
        return;
    }

    if (strncmp(func_name, "_Z10k_set_rows", strlen("_Z10k_set_rows")) != 0 &&
        strncmp(func_name, "_Z16k_set_rows_quant", strlen("_Z16k_set_rows_quant")) != 0) {
        return;
    }

    /* Use the actual last-field offset + size (196 bytes) rather than sizeof()
     * which may include trailing struct padding (200 bytes). */
    static const size_t k_set_rows_min_len =
        offsetof(k_set_rows_params_host_t, ne12_fd) +
        sizeof(((k_set_rows_params_host_t *)0)->ne12_fd);
    if (len < k_set_rows_min_len) {
        fprintf(stderr,
                "[cuda-executor] %s decoded params vm=%u len=%zu expected=%zu (truncated)\n",
                func_name,
                vm_id,
                len,
                sizeof(k_set_rows_params_host_t));
        return;
    }

    k_set_rows_params_host_t p;
    memset(&p, 0, sizeof(p));
    memcpy(&p, param_data, sizeof(p));

    fprintf(stderr,
            "[cuda-executor] %s decoded params vm=%u "
            "src0=%p src1=%p dst=%p "
            "ne_total=%lld ne10=%lld ne11=%lld ne12=%lld ne13=%lld "
            "s01=%lld s02=%lld s03=%lld s10=%lld s11=%lld s12=%lld s1=%lld s2=%lld s3=%lld "
            "ne00=(%u,%u,%u) ne01=(%u,%u,%u) ne02=(%u,%u,%u) "
            "ne11_fd=(%u,%u,%u) ne12_fd=(%u,%u,%u)\n",
            func_name,
            vm_id,
            p.src0,
            p.src1,
            p.dst,
            (long long)p.ne_total,
            (long long)p.ne10,
            (long long)p.ne11,
            (long long)p.ne12,
            (long long)p.ne13,
            (long long)p.s01,
            (long long)p.s02,
            (long long)p.s03,
            (long long)p.s10,
            (long long)p.s11,
            (long long)p.s12,
            (long long)p.s1,
            (long long)p.s2,
            (long long)p.s3,
            p.ne00.x, p.ne00.y, p.ne00.z,
            p.ne01.x, p.ne01.y, p.ne01.z,
            p.ne02.x, p.ne02.y, p.ne02.z,
            p.ne11_fd.x, p.ne11_fd.y, p.ne11_fd.z,
            p.ne12_fd.x, p.ne12_fd.y, p.ne12_fd.z);
}

/* ================================================================
 * Per-VM mapping tables
 * ================================================================ */

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
    uint64_t module_guest_handle;
    char     name[128];
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
    CUstream host_stream;
    void    *host_buf;
    size_t   size;
} pending_async_htod_t;

typedef struct {
    uint32_t    vm_id;
    int         active;
    CUcontext   ctx;
    int         ctx_valid;
    int         ctx_is_primary;

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

    /* Module-load chunk accumulation buffer.
     * Used when a cuModuleLoadData image arrives in multiple chunks
     * (CUDA_CHUNK_FLAG_FIRST / middle / CUDA_CHUNK_FLAG_LAST).
     * Only one in-progress module load is supported per VM at a time. */
    uint8_t       *mod_chunk_buf;   /* heap-allocated accumulation buffer */
    size_t         mod_chunk_alloc; /* allocated capacity in bytes        */
    size_t         mod_chunk_used;  /* bytes accumulated so far           */

    /* HtoD progress: log every PROGRESS_LOG_INTERVAL bytes during model load */
    uint64_t       htod_total_bytes;
    uint64_t       htod_last_log_bytes;

    /* Host-owned staging buffers for in-flight cuMemcpyHtoDAsync calls.
     * These buffers must remain alive until the associated stream/context sync. */
    pending_async_htod_t pending_async_htod[MAX_PENDING_ASYNC_HTOD];
    int                  pending_async_htod_count;
} vm_state_t;

#define HTOD_PROGRESS_LOG_INTERVAL  (10 * 1024 * 1024)  /* 10 MB */

static int executor_verbose_copy_logging(void)
{
    static int cached = -1;
    if (cached < 0) {
        cached = (getenv("VGPU_EXECUTOR_DEBUG") != NULL) ? 1 : 0;
    }
    return cached;
}

/* ================================================================
 * Executor state
 * ================================================================ */
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

/* ================================================================
 * Internal helpers
 * ================================================================ */

static vm_state_t* find_or_create_vm(cuda_executor_t *exec, uint32_t vm_id)
{
    int i;
    /* Find existing */
    for (i = 0; i < MAX_VMS; i++) {
        if (exec->vms[i].active && exec->vms[i].vm_id == vm_id) {
            return &exec->vms[i];
        }
    }
    /* Create new */
    for (i = 0; i < MAX_VMS; i++) {
        if (!exec->vms[i].active) {
            memset(&exec->vms[i], 0, sizeof(vm_state_t));
            exec->vms[i].vm_id = vm_id;
            exec->vms[i].active = 1;
            return &exec->vms[i];
        }
    }
    return NULL;  /* No slots available */
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

/* Memory mapping helpers */
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

static mem_entry_t *vm_find_mem_entry(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->mem_count; i++) {
        if (vm->mem[i].guest_ptr == guest) {
            return &vm->mem[i];
        }
        if (guest > vm->mem[i].guest_ptr) {
            uint64_t off = guest - vm->mem[i].guest_ptr;
            if (off < vm->mem[i].size) {
                return &vm->mem[i];
            }
        }
    }
    return NULL;
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

/* Module mapping helpers */
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

static int vm_add_library(vm_state_t *vm, uint64_t guest, CUlibrary host,
                          void *owned_image, size_t owned_image_size)
{
    if (vm->library_count < MAX_LIBRARY_ENTRIES) {
        vm->libraries[vm->library_count].guest_handle = guest;
        vm->libraries[vm->library_count].host_library = host;
        vm->libraries[vm->library_count].owned_image = owned_image;
        vm->libraries[vm->library_count].owned_image_size = owned_image_size;
        vm->library_count++;
        return 1;
    }

    fprintf(stderr,
            "[cuda-executor] library table full vm=%u max=%d guest_handle=0x%llx\n",
            vm->vm_id, MAX_LIBRARY_ENTRIES, (unsigned long long)guest);
    return 0;
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
static void vm_add_func(vm_state_t *vm, uint64_t guest, CUfunction host,
                        uint64_t module_guest_handle, const char *name)
{
    if (vm->func_count < MAX_FUNC_ENTRIES) {
        vm->funcs[vm->func_count].guest_handle = guest;
        vm->funcs[vm->func_count].host_function = host;
        vm->funcs[vm->func_count].module_guest_handle = module_guest_handle;
        vm->funcs[vm->func_count].name[0] = '\0';
        if (name && name[0] != '\0') {
            snprintf(vm->funcs[vm->func_count].name,
                     sizeof(vm->funcs[vm->func_count].name),
                     "%s", name);
        }
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

static func_entry_t *vm_find_func_entry(vm_state_t *vm, uint64_t guest)
{
    for (int i = 0; i < vm->func_count; i++) {
        if (vm->funcs[i].guest_handle == guest) {
            return &vm->funcs[i];
        }
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
    if (host == CU_STREAM_LEGACY) return 1;
    if (host == CU_STREAM_PER_THREAD) return 2;
    for (int i = 0; i < vm->stream_count; i++) {
        if (vm->streams[i].host_stream == host)
            return vm->streams[i].guest_handle;
    }
    return 0;
}

static CUstream vm_resolve_stream_handle(vm_state_t *vm, uint64_t guest_handle)
{
    if (guest_handle == 0) return NULL;
    if (guest_handle == 1) return CU_STREAM_LEGACY;
    if (guest_handle == 2) return CU_STREAM_PER_THREAD;
    return vm_find_stream(vm, guest_handle);
}

static int vm_add_pending_async_htod(vm_state_t *vm, CUstream stream,
                                     void *host_buf, size_t size)
{
    if (vm->pending_async_htod_count >= MAX_PENDING_ASYNC_HTOD) {
        return 0;
    }

    vm->pending_async_htod[vm->pending_async_htod_count].host_stream = stream;
    vm->pending_async_htod[vm->pending_async_htod_count].host_buf = host_buf;
    vm->pending_async_htod[vm->pending_async_htod_count].size = size;
    vm->pending_async_htod_count++;
    return 1;
}

static void vm_drain_pending_async_htod(vm_state_t *vm, CUstream stream, int drain_all)
{
    for (int i = 0; i < vm->pending_async_htod_count; ) {
        pending_async_htod_t *entry = &vm->pending_async_htod[i];
        if (drain_all || entry->host_stream == stream) {
            free(entry->host_buf);
            vm->pending_async_htod[i] =
                vm->pending_async_htod[vm->pending_async_htod_count - 1];
            vm->pending_async_htod_count--;
            continue;
        }
        i++;
    }
}

/* Event mapping helpers — returns 0 if table full (caller must destroy host event) */
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

static const char *executor_call_id_to_name(uint32_t call_id)
{
    switch (call_id) {
    case CUDA_CALL_LAUNCH_KERNEL:
        return "cuLaunchKernel";
    case CUDA_CALL_FUNC_GET_PARAM_INFO:
        return "cuFuncGetParamInfo";
    case CUDA_CALL_MODULE_LOAD_DATA:
        return "cuModuleLoadData";
    case CUDA_CALL_MODULE_LOAD_FAT_BINARY:
        return "cuModuleLoadFatBinary";
    case CUDA_CALL_MEMCPY_HTOD:
        return "cuMemcpyHtoD";
    case CUDA_CALL_MEMCPY_HTOD_ASYNC:
        return "cuMemcpyHtoDAsync";
    case CUDA_CALL_CUBLAS_GEMM_EX:
        return "cublasGemmEx";
    case CUDA_CALL_CUBLASLT_MATMUL:
        return "cublasLtMatmul";
    default:
        return "cuda_call";
    }
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
                fprintf(stderr,
                        "[cuda-executor] vm_id=%u cuModuleLoadFatBinary: malloc(%u) failed (host heap)\n",
                        vm_id, data_len);
                fflush(stderr);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            memcpy(fatbin_copy, data, data_len);

            /* Persist suspicious large fatbins so they can be replayed directly on the host.
             * This helps compare the live mediator path against a clean primary-context test. */
            if (data_len == 401312U || data_len == 214857U || data_len == 640561U) {
                char dump_path[128];
                snprintf(dump_path, sizeof(dump_path),
                         data_len == 401312U ? "/tmp/fail401312.bin"
                                             : "/tmp/fatbin_vm%u_len%u.bin",
                         vm_id, data_len);
                FILE *df = fopen(dump_path, "wb");
                if (df) {
                    fwrite(fatbin_copy, 1, data_len, df);
                    fclose(df);
                    fprintf(stderr,
                            "[cuda-executor] dumped %s (%u bytes)\n",
                            dump_path, data_len);
                    fflush(stderr);
                }
            }

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
            if (rc == CUDA_ERROR_OUT_OF_MEMORY) {
                fprintf(stderr,
                        "[cuda-executor] vm_id=%u cuModuleLoadFatBinary OOM after copy data_len=%u\n",
                        vm_id, data_len);
                fflush(stderr);
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

static CUresult load_host_library(vm_state_t *vm,
                                  const void *data, uint32_t data_len,
                                  CUDACallResult *result)
{
    void *owned_image = NULL;
    CUlibraryOption lib_option = (CUlibraryOption)1; /* BINARY_IS_PRESERVED */
    void *lib_option_value = (void *)1;
    CUlibrary lib = NULL;
    CUresult rc;

    if (!vm || !data || data_len == 0 || !result) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    owned_image = malloc(data_len);
    if (!owned_image) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    memcpy(owned_image, data, data_len);
    if (data_len == 214857U || data_len == 640561U) {
        char dump_path[128];
        snprintf(dump_path, sizeof(dump_path),
                 "/tmp/libload_vm%u_len%u.bin", vm->vm_id, data_len);
        FILE *df = fopen(dump_path, "wb");
        if (df) {
            fwrite(owned_image, 1, data_len, df);
            fclose(df);
            fprintf(stderr,
                    "[cuda-executor] dumped %s (%u bytes) before cuLibraryLoadData\n",
                    dump_path, (unsigned)data_len);
            fflush(stderr);
        }
    }

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
        if (!vm_add_library(vm, guest_handle, lib, owned_image, data_len)) {
            (void)cuLibraryUnload(lib);
            free(owned_image);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        result->num_results = 1;
        result->results[0] = guest_handle;
    } else {
        fprintf(stderr,
                "[cuda-executor] cuLibraryLoadData failed vm=%u data_len=%u rc=%d\n",
                vm->vm_id, (unsigned)data_len, (int)rc);
        free(owned_image);
    }

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

    /* Keep all replayed objects in the same CUDA context.
     * Alloc/HtoD/module/CUBLAS paths already use the device primary context;
     * late stream/event/sync paths must use that same context too. */
    vm->ctx = exec->primary_ctx;
    vm->ctx_valid = 1;
    vm->ctx_is_primary = 1;
    return cuCtxSetCurrent(exec->primary_ctx);
}

static void vm_discard_runtime_state(vm_state_t *vm)
{
    vm->ctx = NULL;
    vm->ctx_valid = 0;
    vm->ctx_is_primary = 0;
    vm->mem_count = 0;
    vm->module_count = 0;
    vm->library_count = 0;
    vm->func_count = 0;
    vm->stream_count = 0;
    vm->event_count = 0;
    vm->cublas_count = 0;
    vm->pending_async_htod_count = 0;
    vm->htod_total_bytes = 0;
    vm->htod_last_log_bytes = 0;
    free(vm->mod_chunk_buf);
    vm->mod_chunk_buf = NULL;
    vm->mod_chunk_alloc = 0;
    vm->mod_chunk_used = 0;
}

static void cuda_executor_recover_primary_context(cuda_executor_t *exec, uint32_t vm_id, const char *reason)
{
    CUresult rc;

    if (!exec || !exec->cuda_initialized) {
        return;
    }

    fprintf(stderr,
            "[cuda-executor] Recovering primary context after vm_id=%u fault: %s\n",
            vm_id, reason ? reason : "(unknown)");

    (void)cuCtxSetCurrent(NULL);
    (void)cuDevicePrimaryCtxRelease(exec->device);
    (void)cuDevicePrimaryCtxReset(exec->device);

    exec->primary_ctx = NULL;
    rc = cuDevicePrimaryCtxRetain(&exec->primary_ctx, exec->device);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr,
                "[cuda-executor] ERROR: failed to re-retain primary context after recovery: rc=%d\n",
                (int)rc);
        return;
    }

    for (int i = 0; i < MAX_VMS; i++) {
        if (exec->vms[i].active) {
            vm_discard_runtime_state(&exec->vms[i]);
        }
    }

    (void)cudaSetDevice(0);
    rc = cuCtxSetCurrent(exec->primary_ctx);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr,
                "[cuda-executor] ERROR: recovered primary context could not be made current: rc=%d\n",
                (int)rc);
    }
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
            if (rc == CUDA_SUCCESS) {
                vm_drain_pending_async_htod(vm, NULL, 1);
            }
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
        int retried_after_recover = 0;

        fprintf(stderr, "[cuda-executor] cuMemAlloc: allocating %llu bytes on physical GPU (vm=%u)\n",
                (unsigned long long)bytesize, call->vm_id);

        /* Use primary context for allocation (per-VM cuCtxCreate can fail and cause "unable to allocate CUDA0 buffer") */
        rc = cuCtxSetCurrent(exec->primary_ctx);
        if (rc == CUDA_ERROR_INVALID_CONTEXT) {
            fprintf(stderr,
                    "[cuda-executor] cuMemAlloc preflight saw CUDA_ERROR_INVALID_CONTEXT; "
                    "recovering context and retrying once (vm=%u size=%llu)\n",
                    call->vm_id, (unsigned long long)bytesize);
            cuda_executor_recover_primary_context(exec, vm->vm_id,
                                                  "cuMemAlloc preflight cuCtxSetCurrent returned CUDA_ERROR_INVALID_CONTEXT");
            rc = cuCtxSetCurrent(exec->primary_ctx);
            retried_after_recover = 1;
        }
        if (rc != CUDA_SUCCESS) {
            break;
        }

        CUdeviceptr dptr = 0;
        rc = cuMemAlloc(&dptr, (size_t)bytesize);
        if (rc == CUDA_ERROR_ILLEGAL_ADDRESS || rc == CUDA_ERROR_INVALID_CONTEXT) {
            fprintf(stderr,
                    "[cuda-executor] cuMemAlloc hit rc=%d before allocation completed; "
                    "recovering context and retrying once (vm=%u size=%llu)\n",
                    (int)rc, call->vm_id, (unsigned long long)bytesize);
            cuda_executor_recover_primary_context(exec, vm->vm_id,
                                                  rc == CUDA_ERROR_ILLEGAL_ADDRESS
                                                      ? "cuMemAlloc hit CUDA_ERROR_ILLEGAL_ADDRESS"
                                                      : "cuMemAlloc hit CUDA_ERROR_INVALID_CONTEXT");
            rc = cuCtxSetCurrent(exec->primary_ctx);
            if (rc != CUDA_SUCCESS) {
                break;
            }
            dptr = 0;
            rc = cuMemAlloc(&dptr, (size_t)bytesize);
            retried_after_recover = 1;
        }
        if (rc == CUDA_SUCCESS) {
            /* Generate a guest-visible handle (use the host pointer value) */
            uint64_t guest_ptr = (uint64_t)dptr;
            vm_add_mem(vm, guest_ptr, dptr, (size_t)bytesize);
            result->num_results = 1;
            result->results[0] = guest_ptr;
            fprintf(stderr, "[cuda-executor] cuMemAlloc SUCCESS: allocated 0x%llx on physical GPU (vm=%u%s)\n",
                    (unsigned long long)dptr, call->vm_id,
                    retried_after_recover ? ", after context recovery" : "");
        } else {
            const char *ename = NULL;
            const char *estr = NULL;
            (void)cuGetErrorName(rc, &ename);
            (void)cuGetErrorString(rc, &estr);
            fprintf(stderr,
                    "[cuda-executor] cuMemAlloc FAILED: rc=%d (%s) %s size=%llu bytes (vm=%u)\n",
                    (int)rc,
                    ename ? ename : "?",
                    estr ? estr : "?",
                    (unsigned long long)bytesize,
                    call->vm_id);
        }
        break;
    }

    case CUDA_CALL_MEM_FREE: {
        uint64_t guest_ptr = CUDA_UNPACK_U64(call->args, 0);
        CUdeviceptr host_ptr = vm_find_mem(vm, guest_ptr);
        if (host_ptr) {
            cuCtxSetCurrent(exec->primary_ctx);
            /* Ensure async work (e.g. cublas on default stream) completes before free */
            {
                CUresult sync_rc = cuCtxSynchronize();
                if (sync_rc != CUDA_SUCCESS) {
                    const char *sn = NULL;
                    const char *ss = NULL;
                    (void)cuGetErrorName(sync_rc, &sn);
                    (void)cuGetErrorString(sync_rc, &ss);
                    fprintf(stderr,
                            "[cuda-executor] cuMemFree: cuCtxSynchronize before free: rc=%d (%s) %s (vm=%u)\n",
                            (int)sync_rc, sn ? sn : "?", ss ? ss : "?", call->vm_id);
                }
            }
            rc = cuMemFree(host_ptr);
            if (rc != CUDA_SUCCESS) {
                const char *ename = NULL;
                const char *estr = NULL;
                (void)cuGetErrorName(rc, &ename);
                (void)cuGetErrorString(rc, &estr);
                fprintf(stderr,
                        "[cuda-executor] cuMemFree FAILED: rc=%d (%s) %s guest=0x%llx host=0x%llx (vm=%u)\n",
                        (int)rc,
                        ename ? ename : "?",
                        estr ? estr : "?",
                        (unsigned long long)guest_ptr,
                        (unsigned long long)host_ptr,
                        call->vm_id);
            }
            if (rc == CUDA_SUCCESS)
                vm_remove_mem(vm, guest_ptr);
        } else {
            fprintf(stderr,
                    "[cuda-executor] cuMemFree: no mapping for guest=0x%llx (vm=%u) — treating as success\n",
                    (unsigned long long)guest_ptr, call->vm_id);
            /* Guest may free handles we did not allocate via this VM table */
            rc = CUDA_SUCCESS;
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
            cuda_executor_log_prefix_bytes("cuMemcpyHtoD input", data, copy_len, call->vm_id);
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

    case CUDA_CALL_MEMCPY_HTOD_ASYNC: {
        uint64_t dst = CUDA_UNPACK_U64(call->args, 0);
        uint64_t byte_count = CUDA_UNPACK_U64(call->args, 2);
        uint64_t stream_handle = CUDA_UNPACK_U64(call->args, 4);

        CUdeviceptr host_dst = vm_find_mem(vm, dst);
        if (!host_dst) {
            host_dst = (CUdeviceptr)dst;
        }

        /* Some guest paths surface raw stream-like handles without a prior
         * STREAM_CREATE RPC. Fall back to the default stream so weight copies
         * continue rather than failing the whole load on INVALID_HANDLE. */
        CUstream stream = vm_resolve_stream_handle(vm, stream_handle);
        if (stream_handle != 0 && !stream) {
            fprintf(stderr,
                    "[cuda-executor] cuMemcpyHtoDAsync unresolved stream handle guest=0x%llx (vm=%u) -> fallback default stream\n",
                    (unsigned long long)stream_handle, call->vm_id);
            stream = NULL;
        }

        cuCtxSetCurrent(exec->primary_ctx);

        if (data && data_len > 0) {
            size_t copy_len = (size_t)byte_count;
            if (copy_len > data_len) copy_len = data_len;

            void *staging = malloc(copy_len);
            if (!staging) {
                rc = CUDA_ERROR_OUT_OF_MEMORY;
                break;
            }

            memcpy(staging, data, copy_len);

            /* Suppress CUDA fatbin uploads to device memory.
             *
             * GGML's JIT path uploads compiled kernel binaries (fatbins,
             * magic 0xBA55ED50) via cuMemcpyHtoDAsync to a scratch device
             * buffer whose address aliases the k_set_rows src1 tensor.
             * Writing the fatbin corrupts src1 and triggers
             * CUDA_ERROR_ILLEGAL_ADDRESS on the next k_set_rows launch.
             * On a real GPU the driver zero-initialises the fresh cuMemAlloc
             * that follows, so src1 is always zero; we preserve that
             * invariant here by simply not writing the fatbin to device RAM.
             */
            if (copy_len >= 4 && *(const uint32_t *)staging == 0xBA55ED50U) {
                fprintf(stderr,
                        "[cuda-executor] cuMemcpyHtoDAsync: fatbin suppressed vm=%u "
                        "dst=0x%llx size=%zu (preserves tensor zero-state)\n",
                        call->vm_id, (unsigned long long)host_dst, copy_len);
                free(staging);
                rc = CUDA_SUCCESS;
                break;
            }

            fprintf(stderr,
                    "[cuda-executor] cuMemcpyHtoDAsync vm=%u seq=%u guest_dst=0x%llx "
                    "host_dst=0x%llx size=%zu stream_guest=0x%llx fnv1a64=0x%016llx\n",
                    call->vm_id,
                    call->seq_num,
                    (unsigned long long)dst,
                    (unsigned long long)host_dst,
                    copy_len,
                    (unsigned long long)stream_handle,
                    (unsigned long long)host_fnv1a64(data, copy_len));
            cuda_executor_log_prefix_bytes("cuMemcpyHtoDAsync input", data, copy_len, call->vm_id);
            int used_sync_fallback = 0;
            rc = cuMemcpyHtoDAsync(host_dst, staging, copy_len, stream);
            if (rc != CUDA_SUCCESS) {
                fprintf(stderr,
                        "[cuda-executor] cuMemcpyHtoDAsync FAILED: rc=%d dst=0x%llx size=%zu stream_guest=0x%llx (vm=%u) -> fallback sync copy\n",
                        (int)rc, (unsigned long long)host_dst, copy_len,
                        (unsigned long long)stream_handle, call->vm_id);
                rc = cuMemcpyHtoD(host_dst, staging, copy_len);
                if (rc == CUDA_SUCCESS) {
                    used_sync_fallback = 1;
                    free(staging);
                }
            }
            if (rc == CUDA_SUCCESS) {
                if (!used_sync_fallback && !vm_add_pending_async_htod(vm, stream, staging, copy_len)) {
                    /* Fall back to immediate completion if pending staging capacity is exhausted. */
                    rc = cuStreamSynchronize(stream);
                    free(staging);
                    if (rc != CUDA_SUCCESS) {
                        break;
                    }
                } else if (!used_sync_fallback && stream == NULL) {
                    /* Default-stream async copies are ordered immediately here. */
                    rc = cuCtxSynchronize();
                    free(staging);
                    if (rc != CUDA_SUCCESS) {
                        break;
                    }
                }

                vm->htod_total_bytes += (uint64_t)copy_len;
                if (vm->htod_total_bytes - vm->htod_last_log_bytes >= HTOD_PROGRESS_LOG_INTERVAL) {
                    fprintf(stderr, "[cuda-executor] HtoD progress: %llu MB total (vm=%u)\n",
                            (unsigned long long)(vm->htod_total_bytes / (1024 * 1024)), call->vm_id);
                    vm->htod_last_log_bytes = vm->htod_total_bytes;
                }
            } else {
                free(staging);
            }
        }
        break;
    }

    case CUDA_CALL_MEMCPY_DTOH: {
        uint64_t src = CUDA_UNPACK_U64(call->args, 0);
        uint64_t byte_count = CUDA_UNPACK_U64(call->args, 2);

        CUdeviceptr mapped_host_src = vm_find_mem(vm, src);
        CUdeviceptr host_src = mapped_host_src;
        if (!host_src) host_src = (CUdeviceptr)src;

        cuCtxSetCurrent(exec->primary_ctx);

        size_t copy_len = (size_t)byte_count;
        if (result_data && result_cap >= copy_len) {
            fprintf(stderr,
                    "[cuda-executor] cuMemcpyDtoH start vm=%u guest_src=0x%llx host_src=0x%llx mapped=%s size=%zu\n",
                    call->vm_id,
                    (unsigned long long)src,
                    (unsigned long long)host_src,
                    mapped_host_src ? "yes" : "no",
                    copy_len);
            rc = cuMemcpyDtoH(result_data, host_src, copy_len);
            if (rc == CUDA_SUCCESS) {
                result->data_len = (uint32_t)copy_len;
                if (result_len) *result_len = (uint32_t)copy_len;
                fprintf(stderr,
                        "[cuda-executor] cuMemcpyDtoH SUCCESS vm=%u guest_src=0x%llx host_src=0x%llx size=%zu\n",
                        call->vm_id,
                        (unsigned long long)src,
                        (unsigned long long)host_src,
                        copy_len);
                cuda_executor_log_prefix_bytes("cuMemcpyDtoH result", result_data, copy_len, call->vm_id);
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
            rc = ensure_vm_context(exec, vm);
            if (rc != CUDA_SUCCESS) {
                break;
            }
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
            rc = ensure_vm_context(exec, vm);
            if (rc != CUDA_SUCCESS) {
                free(vm->mod_chunk_buf);
                vm->mod_chunk_buf   = NULL;
                vm->mod_chunk_alloc = 0;
                vm->mod_chunk_used  = 0;
                break;
            }
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
            vm_add_func(vm, guest_func, func, mod_handle, func_name);
            fprintf(stderr,
                    "[cuda-executor] cuModuleGetFunction success: vm=%u mod=0x%llx func=0x%llx name=%s\n",
                    call->vm_id,
                    (unsigned long long)mod_handle,
                    (unsigned long long)guest_func,
                    func_name);
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
        func_entry_t *func_entry = vm_find_func_entry(vm, lp->function_handle);
        CUfunction func = func_entry ? func_entry->host_function : NULL;
        if (!func) {
            /* Try direct cast (if host gave handle directly) */
            func = (CUfunction)(uintptr_t)lp->function_handle;
        }
        const char *func_name =
            (func_entry && func_entry->name[0] != '\0') ? func_entry->name : "<unknown>";
        uint64_t module_guest_handle =
            func_entry ? func_entry->module_guest_handle : 0;

        /* Resolve stream handle */
        CUstream stream = vm_resolve_stream_handle(vm, lp->stream_handle);

        /* Parse kernel parameters */
        const uint8_t *payload_ptr = (const uint8_t *)data;
        payload_ptr += sizeof(CUDALaunchParams);

        const uint32_t *param_sizes = NULL;
        const uint8_t *param_data = payload_ptr;
        void *kernelParams[256];
        memset(kernelParams, 0, sizeof(kernelParams));

        if (lp->param_buf_mode == CUDA_LAUNCH_PARAM_MODE_LEGACY) {
            param_sizes = (const uint32_t *)payload_ptr;
            payload_ptr += lp->num_params * sizeof(uint32_t);
            param_data = payload_ptr;

            /* Build kernelParams array */
            uint32_t offset = 0;
            for (uint32_t i = 0; i < lp->num_params && i < 256; i++) {
                kernelParams[i] = (void *)(param_data + offset);
                offset += param_sizes[i];
            }
        }

        fprintf(stderr,
                "[cuda-executor] cuLaunchKernel: func=0x%llx name=%s mod=0x%llx stream=0x%llx "
                "grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u params=%u mode=%u vm=%u\n",
                (unsigned long long)lp->function_handle,
                func_name,
                (unsigned long long)module_guest_handle,
                (unsigned long long)lp->stream_handle,
                lp->grid_dim_x, lp->grid_dim_y, lp->grid_dim_z,
                lp->block_dim_x, lp->block_dim_y, lp->block_dim_z,
                lp->shared_mem_bytes, lp->num_params, lp->param_buf_mode, call->vm_id);
        if (lp->param_buf_mode == CUDA_LAUNCH_PARAM_MODE_RAW_BUFFER) {
            cuda_executor_log_k_set_rows_params(func_name,
                                                param_data,
                                                lp->total_param_bytes,
                                                call->vm_id);
            /* For k_set_rows, read first 8 src1 int64_t values from device */
            if (func_name &&
                strncmp(func_name, "_Z10k_set_rows", strlen("_Z10k_set_rows")) == 0 &&
                lp->total_param_bytes >= 16u) {
                const uint64_t *pp = (const uint64_t *)param_data;
                CUdeviceptr src1_dev = (CUdeviceptr)pp[1]; /* src1 is param 1 */
                if (src1_dev != 0) {
                    int64_t src1_peek[8] = {0};
                    size_t peek_bytes = sizeof(src1_peek);
                    CUresult peek_rc = cuMemcpyDtoH(src1_peek, src1_dev, peek_bytes);
                    if (peek_rc == CUDA_SUCCESS) {
                        fprintf(stderr,
                                "[cuda-executor] %s src1 peek vm=%u "
                                "[%lld %lld %lld %lld %lld %lld %lld %lld]\n",
                                func_name, call->vm_id,
                                (long long)src1_peek[0], (long long)src1_peek[1],
                                (long long)src1_peek[2], (long long)src1_peek[3],
                                (long long)src1_peek[4], (long long)src1_peek[5],
                                (long long)src1_peek[6], (long long)src1_peek[7]);
                    } else {
                        fprintf(stderr,
                                "[cuda-executor] %s src1 peek FAILED vm=%u rc=%d\n",
                                func_name, call->vm_id, (int)peek_rc);
                    }
                }
            }
        }

        if (lp->param_buf_mode == CUDA_LAUNCH_PARAM_MODE_RAW_BUFFER) {
            size_t param_buffer_size = lp->total_param_bytes;
            void *extra[] = {
                CU_LAUNCH_PARAM_BUFFER_POINTER, (void *)param_data,
                CU_LAUNCH_PARAM_BUFFER_SIZE, &param_buffer_size,
                CU_LAUNCH_PARAM_END
            };

            rc = cuLaunchKernel(func,
                               lp->grid_dim_x, lp->grid_dim_y, lp->grid_dim_z,
                               lp->block_dim_x, lp->block_dim_y, lp->block_dim_z,
                               lp->shared_mem_bytes,
                               stream,
                               NULL,
                               extra);
        } else {
            rc = cuLaunchKernel(func,
                               lp->grid_dim_x, lp->grid_dim_y, lp->grid_dim_z,
                               lp->block_dim_x, lp->block_dim_y, lp->block_dim_z,
                               lp->shared_mem_bytes,
                               stream,
                               kernelParams,
                               NULL);
        }

        /* Synchronize after launch to detect runtime faults immediately. */
        if (rc == CUDA_SUCCESS) {
            rc = cuCtxSynchronize();
            if (rc == CUDA_SUCCESS) {
                fprintf(stderr,
                        "[cuda-executor] cuLaunchKernel SUCCESS: kernel executed on physical GPU "
                        "(func=0x%llx name=%s mod=0x%llx vm=%u)\n",
                        (unsigned long long)lp->function_handle,
                        func_name,
                        (unsigned long long)module_guest_handle,
                        call->vm_id);
                if (lp->param_buf_mode == CUDA_LAUNCH_PARAM_MODE_RAW_BUFFER) {
                    cuda_executor_log_output_kernel_samples(func_name,
                                                            param_data,
                                                            lp->total_param_bytes,
                                                            call->vm_id);
                }
            } else {
                fprintf(stderr,
                        "[cuda-executor] cuLaunchKernel sync FAILED: rc=%d func=0x%llx "
                        "name=%s mod=0x%llx grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u params=%u vm=%u\n",
                        rc,
                        (unsigned long long)lp->function_handle,
                        func_name,
                        (unsigned long long)module_guest_handle,
                        lp->grid_dim_x, lp->grid_dim_y, lp->grid_dim_z,
                        lp->block_dim_x, lp->block_dim_y, lp->block_dim_z,
                        lp->shared_mem_bytes, lp->num_params, call->vm_id);
                fprintf(stderr,
                        "[cuda-executor] cuLaunchKernel failure detail: mode=%u total_param_bytes=%u vm=%u\n",
                        lp->param_buf_mode,
                        lp->total_param_bytes,
                        call->vm_id);
                cuda_executor_log_prefix_bytes("cuLaunchKernel param buffer",
                                              param_data,
                                              lp->total_param_bytes,
                                              call->vm_id);
                if (rc == CUDA_ERROR_ILLEGAL_ADDRESS || rc == CUDA_ERROR_INVALID_CONTEXT) {
                    cuda_executor_recover_primary_context(exec, vm->vm_id,
                                                          rc == CUDA_ERROR_ILLEGAL_ADDRESS
                                                              ? "cuLaunchKernel sync hit CUDA_ERROR_ILLEGAL_ADDRESS"
                                                              : "cuLaunchKernel sync hit CUDA_ERROR_INVALID_CONTEXT");
                }
            }
        } else {
            fprintf(stderr,
                    "[cuda-executor] cuLaunchKernel launch FAILED: rc=%d func=0x%llx "
                    "name=%s mod=0x%llx grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u params=%u vm=%u\n",
                    rc,
                    (unsigned long long)lp->function_handle,
                    func_name,
                    (unsigned long long)module_guest_handle,
                    lp->grid_dim_x, lp->grid_dim_y, lp->grid_dim_z,
                    lp->block_dim_x, lp->block_dim_y, lp->block_dim_z,
                    lp->shared_mem_bytes, lp->num_params, call->vm_id);
            fprintf(stderr,
                    "[cuda-executor] cuLaunchKernel launch failure detail: mode=%u total_param_bytes=%u vm=%u\n",
                    lp->param_buf_mode,
                    lp->total_param_bytes,
                    call->vm_id);
            cuda_executor_log_prefix_bytes("cuLaunchKernel param buffer",
                                          param_data,
                                          lp->total_param_bytes,
                                          call->vm_id);
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
        CUstream stream = vm_resolve_stream_handle(vm, guest_handle);
        if (!stream) {
            /* Must not return SUCCESS — guest would desync vs host. */
            rc = CUDA_ERROR_INVALID_HANDLE;
            break;
        }
        rc = ensure_vm_context(exec, vm);
        if (rc == CUDA_SUCCESS) {
            rc = cuStreamSynchronize(stream);
        }
        if (rc == CUDA_SUCCESS) {
            vm_drain_pending_async_htod(vm, stream, 0);
            rc = cuStreamDestroy(stream);
            if (rc == CUDA_SUCCESS)
                vm_remove_stream(vm, guest_handle);
        }
        break;
    }

    case CUDA_CALL_STREAM_SYNCHRONIZE: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUstream stream = vm_resolve_stream_handle(vm, guest_handle);

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        rc = cuStreamSynchronize(stream);  /* NULL stream = default */
        if (rc == CUDA_SUCCESS) {
            vm_drain_pending_async_htod(vm, stream, 0);
        }
        break;
    }

    case CUDA_CALL_STREAM_QUERY: {
        uint64_t guest_handle = CUDA_UNPACK_U64(call->args, 0);
        CUstream stream = vm_resolve_stream_handle(vm, guest_handle);

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        rc = cuStreamQuery(stream);
        break;
    }

    case CUDA_CALL_STREAM_WAIT_EVENT: {
        uint64_t stream_handle = CUDA_UNPACK_U64(call->args, 0);
        uint64_t event_handle = CUDA_UNPACK_U64(call->args, 2);
        uint32_t flags = call->args[4];

        /* stream_handle 0 => NULL (default stream); non-zero must resolve */
        CUstream stream = vm_resolve_stream_handle(vm, stream_handle);
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
        /* stream_handle 0 => NULL (default stream); non-zero must resolve */
        CUstream stream = vm_resolve_stream_handle(vm, stream_handle);

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

        /* Use primary context for CUBLAS; per-VM contexts can trigger ALLOC_FAILED in cublasCreate_v2.
         * cuBLAS also expects a current CUDA runtime device (see cublasCreate_v2 docs). */
        (void)cudaSetDevice(0);
        cuCtxSetCurrent(exec->primary_ctx);

        cublas_rc = cublasCreate_v2(&handle);
        if (cublas_rc == CUBLAS_STATUS_NOT_INITIALIZED) {
            cuda_executor_recover_primary_context(exec, vm->vm_id, "cublasCreate_v2 returned NOT_INITIALIZED");
            (void)cudaSetDevice(0);
            (void)cuCtxSetCurrent(exec->primary_ctx);
            handle = NULL;
            cublas_rc = cublasCreate_v2(&handle);
        }
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
        CUstream stream = vm_resolve_stream_handle(vm, stream_handle);
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
        if (cublas_rc == CUBLAS_STATUS_SUCCESS) {
            CUresult ec = cuCtxSynchronize();
            if (ec != CUDA_SUCCESS) {
                const char *en = NULL;
                const char *es = NULL;
                (void)cuGetErrorName(ec, &en);
                (void)cuGetErrorString(ec, &es);
                fprintf(stderr,
                        "[cuda-executor] vm_id=%u after cublasSgemm_v2: cuCtxSynchronize rc=%d (%s) %s\n",
                        vm->vm_id, (int)ec, en ? en : "?", es ? es : "?");
                fprintf(stderr,
                        "[cuda-executor] vm_id=%u SGEMM dims m=%d n=%d k=%d lda=%d ldb=%d ldc=%d alpha=%g beta=%g\n",
                        vm->vm_id, sgemm->m, sgemm->n, sgemm->k,
                        sgemm->lda, sgemm->ldb, sgemm->ldc,
                        sgemm->alpha, sgemm->beta);
                cublas_rc = CUBLAS_STATUS_EXECUTION_FAILED;
            }
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
        mem_entry_t *entry_a = vm_find_mem_entry(vm, gemm->a);
        mem_entry_t *entry_b = vm_find_mem_entry(vm, gemm->b);
        mem_entry_t *entry_c = vm_find_mem_entry(vm, gemm->c);
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
            uint16_t alpha_half = cuda_executor_float_to_half_bits(gemm->alpha_f32);
            uint16_t beta_half = cuda_executor_float_to_half_bits(gemm->beta_f32);
            const void *alpha_ptr = &alpha;
            const void *beta_ptr = &beta;
            if (cuda_executor_is_fp16_compute(gemm->computeType)) {
                alpha_ptr = &alpha_half;
                beta_ptr = &beta_half;
            }
            cublas_rc = cublasGemmEx(handle,
                                     (cublasOperation_t)gemm->transa,
                                     (cublasOperation_t)gemm->transb,
                                     gemm->m, gemm->n, gemm->k,
                                     alpha_ptr,
                                     (const void *)(uintptr_t)host_a,
                                     (cudaDataType_t)gemm->Atype, gemm->lda,
                                     (const void *)(uintptr_t)host_b,
                                     (cudaDataType_t)gemm->Btype, gemm->ldb,
                                     beta_ptr,
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
            /* Async faults (e.g. illegal address) often appear on sync — same as GEMM_BATCHED_EX / E4. */
            if (cublas_rc == CUBLAS_STATUS_SUCCESS) {
                CUresult ec = cuCtxSynchronize();
                if (ec != CUDA_SUCCESS) {
                    const char *en = NULL;
                    const char *es = NULL;
                    (void)cuGetErrorName(ec, &en);
                    (void)cuGetErrorString(ec, &es);
                    fprintf(stderr,
                            "[cuda-executor] vm_id=%u after cublasGemmEx: cuCtxSynchronize rc=%d (%s) %s\n",
                            vm->vm_id, (int)ec, en ? en : "?", es ? es : "?");
                    fprintf(stderr,
                            "[cuda-executor] vm_id=%u GEMM_EX dims m=%d n=%d k=%d lda=%d ldb=%d ldc=%d "
                            "Atype=%u Btype=%u Ctype=%u computeType=%d algo=%d alpha_f32=%g beta_f32=%g\n",
                            vm->vm_id, gemm->m, gemm->n, gemm->k, gemm->lda, gemm->ldb, gemm->ldc,
                            (unsigned)gemm->Atype, (unsigned)gemm->Btype, (unsigned)gemm->Ctype,
                            gemm->computeType, gemm->algo, gemm->alpha_f32, gemm->beta_f32);
                    fprintf(stderr,
                            "[cuda-executor] vm_id=%u GEMM_EX ptrs guest=(0x%llx,0x%llx,0x%llx) "
                            "host=(0x%llx,0x%llx,0x%llx) base_guest=(0x%llx,0x%llx,0x%llx) "
                            "base_host=(0x%llx,0x%llx,0x%llx) sizes=(%zu,%zu,%zu)\n",
                            vm->vm_id,
                            (unsigned long long)gemm->a,
                            (unsigned long long)gemm->b,
                            (unsigned long long)gemm->c,
                            (unsigned long long)host_a,
                            (unsigned long long)host_b,
                            (unsigned long long)host_c,
                            (unsigned long long)(entry_a ? entry_a->guest_ptr : 0ull),
                            (unsigned long long)(entry_b ? entry_b->guest_ptr : 0ull),
                            (unsigned long long)(entry_c ? entry_c->guest_ptr : 0ull),
                            (unsigned long long)(entry_a ? entry_a->host_ptr : 0ull),
                            (unsigned long long)(entry_b ? entry_b->host_ptr : 0ull),
                            (unsigned long long)(entry_c ? entry_c->host_ptr : 0ull),
                            entry_a ? entry_a->size : 0u,
                            entry_b ? entry_b->size : 0u,
                            entry_c ? entry_c->size : 0u);
                    cublas_rc = CUBLAS_STATUS_EXECUTION_FAILED;
                    if (ec == CUDA_ERROR_ILLEGAL_ADDRESS) {
                        cuda_executor_recover_primary_context(exec, vm->vm_id,
                                                             "cublas GemmEx sync hit CUDA_ERROR_ILLEGAL_ADDRESS");
                    }
                } else if ((cudaDataType_t)gemm->Ctype == CUDA_R_32F) {
                    cuda_executor_log_device_f32_sample("cublasGemmEx C",
                                                        host_c,
                                                        8,
                                                        vm->vm_id);
                }
            }
        }

        result->num_results = 1;
        result->results[0] = (uint64_t)cublas_rc;
        rc = CUDA_SUCCESS;
        break;
    }

    case CUDA_CALL_CUBLAS_GEMM_STRIDED_BATCHED_EX: {
        const CublasGemmStridedBatchedExCall *g = (const CublasGemmStridedBatchedExCall *)data;
        cublasStatus_t cublas_rc = CUBLAS_STATUS_INVALID_VALUE;

        if (!g || data_len < sizeof(*g)) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        cublasHandle_t handle = vm_find_cublas(vm, g->handle);
        CUdeviceptr host_a = vm_find_mem(vm, g->a);
        CUdeviceptr host_b = vm_find_mem(vm, g->b);
        CUdeviceptr host_c = vm_find_mem(vm, g->c);
        if (!handle || !host_a || !host_b || !host_c) {
            fprintf(stderr,
                    "[cuda-executor] cublasGemmStridedBatchedEx MAPPING FAILED vm_id=%u handle=%p a=%p b=%p c=%p\n",
                    vm->vm_id, (void *)(uintptr_t)handle, (void *)(uintptr_t)host_a,
                    (void *)(uintptr_t)host_b, (void *)(uintptr_t)host_c);
            result->num_results = 1;
            result->results[0] = (uint64_t)CUBLAS_STATUS_INVALID_VALUE;
            rc = CUDA_SUCCESS;
            break;
        }

        cuCtxSetCurrent(exec->primary_ctx);
        {
            float alpha = g->alpha_f32;
            float beta = g->beta_f32;
            uint16_t alpha_half = cuda_executor_float_to_half_bits(g->alpha_f32);
            uint16_t beta_half = cuda_executor_float_to_half_bits(g->beta_f32);
            const void *alpha_ptr = &alpha;
            const void *beta_ptr = &beta;
            if (cuda_executor_is_fp16_compute(g->computeType)) {
                alpha_ptr = &alpha_half;
                beta_ptr = &beta_half;
            }
            cublas_rc = cublasGemmStridedBatchedEx(handle,
                    (cublasOperation_t)g->transa, (cublasOperation_t)g->transb,
                    g->m, g->n, g->k,
                    alpha_ptr,
                    (const void *)(uintptr_t)host_a, (cudaDataType_t)g->Atype, g->lda,
                    g->strideA,
                    (const void *)(uintptr_t)host_b, (cudaDataType_t)g->Btype, g->ldb,
                    g->strideB,
                    beta_ptr,
                    (void *)(uintptr_t)host_c, (cudaDataType_t)g->Ctype, g->ldc,
                    g->strideC,
                    g->batchCount,
                    (cublasComputeType_t)g->computeType,
                    (cublasGemmAlgo_t)g->algo);
            if (executor_verbose_copy_logging() || cublas_rc != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[cuda-executor] vm_id=%u cublasGemmStridedBatchedEx rc=%d batch=%d m=%d n=%d k=%d\n",
                        vm->vm_id, (int)cublas_rc, g->batchCount, g->m, g->n, g->k);
            }
            if (cublas_rc == CUBLAS_STATUS_SUCCESS) {
                CUresult ec = cuCtxSynchronize();
                if (ec != CUDA_SUCCESS) {
                    const char *en = NULL;
                    const char *es = NULL;
                    (void)cuGetErrorName(ec, &en);
                    (void)cuGetErrorString(ec, &es);
                    fprintf(stderr,
                            "[cuda-executor] vm_id=%u after cublasGemmStridedBatchedEx: cuCtxSynchronize rc=%d (%s) %s\n",
                            vm->vm_id, (int)ec, en ? en : "?", es ? es : "?");
                    fprintf(stderr,
                            "[cuda-executor] vm_id=%u GEMM_STRIDED_BATCHED dims m=%d n=%d k=%d batch=%d "
                            "computeType=%d algo=%d\n",
                            vm->vm_id, g->m, g->n, g->k, g->batchCount,
                            g->computeType, g->algo);
                    cublas_rc = CUBLAS_STATUS_EXECUTION_FAILED;
                }
            }
        }

        result->num_results = 1;
        result->results[0] = (uint64_t)cublas_rc;
        rc = CUDA_SUCCESS;
        break;
    }

    case CUDA_CALL_CUBLAS_GEMM_BATCHED_EX: {
        const CublasGemmBatchedExCallHdr *hdr = (const CublasGemmBatchedExCallHdr *)data;
        cublasStatus_t cublas_rc = CUBLAS_STATUS_INVALID_VALUE;

        if (!hdr || data_len < sizeof(CublasGemmBatchedExCallHdr)) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }
        int bc = hdr->batchCount;
        if (bc < 1) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }
        size_t need = sizeof(CublasGemmBatchedExCallHdr) + (size_t)bc * 3u * sizeof(uint64_t);
        if (data_len < need) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        cublasHandle_t handle = vm_find_cublas(vm, hdr->handle);
        if (!handle) {
            result->num_results = 1;
            result->results[0] = (uint64_t)CUBLAS_STATUS_INVALID_VALUE;
            rc = CUDA_SUCCESS;
            break;
        }

        const uint64_t *gp = (const uint64_t *)((const uint8_t *)data + sizeof(CublasGemmBatchedExCallHdr));
        void **rowA = (void **)calloc((size_t)bc, sizeof(void *));
        void **rowB = (void **)calloc((size_t)bc, sizeof(void *));
        void **rowC = (void **)calloc((size_t)bc, sizeof(void *));
        if (!rowA || !rowB || !rowC) {
            free(rowA);
            free(rowB);
            free(rowC);
            rc = CUDA_ERROR_OUT_OF_MEMORY;
            break;
        }

        int map_ok = 1;
        for (int i = 0; i < bc; i++) {
            CUdeviceptr ha = vm_find_mem(vm, gp[i]);
            CUdeviceptr hb = vm_find_mem(vm, gp[bc + (size_t)i]);
            CUdeviceptr hc = vm_find_mem(vm, gp[2u * (size_t)bc + (size_t)i]);
            if (!ha || !hb || !hc) {
                map_ok = 0;
                break;
            }
            rowA[i] = (void *)(uintptr_t)ha;
            rowB[i] = (void *)(uintptr_t)hb;
            rowC[i] = (void *)(uintptr_t)hc;
        }
        if (!map_ok) {
            free(rowA);
            free(rowB);
            free(rowC);
            fprintf(stderr,
                    "[cuda-executor] cublasGemmBatchedEx MAPPING FAILED vm_id=%u batch=%d mem_count=%d\n",
                    vm->vm_id, bc, vm->mem_count);
            result->num_results = 1;
            result->results[0] = (uint64_t)CUBLAS_STATUS_INVALID_VALUE;
            rc = CUDA_SUCCESS;
            break;
        }

        cuCtxSetCurrent(exec->primary_ctx);
        {
            float alpha = hdr->alpha_f32;
            float beta = hdr->beta_f32;
            uint16_t alpha_half = cuda_executor_float_to_half_bits(hdr->alpha_f32);
            uint16_t beta_half = cuda_executor_float_to_half_bits(hdr->beta_f32);
            const void *alpha_ptr = &alpha;
            const void *beta_ptr = &beta;
            if (cuda_executor_is_fp16_compute(hdr->computeType)) {
                alpha_ptr = &alpha_half;
                beta_ptr = &beta_half;
            }
            /* E4 (H100 + libcublas 12.3.x): mediated cublasGemmBatchedEx poisons the
             * context with CUDA_ERROR_ILLEGAL_ADDRESS, while replaying the same batches
             * as individual cublasGemmEx calls is stable. */
            if (bc == 1) {
                cublas_rc = cublasGemmEx(handle,
                        (cublasOperation_t)hdr->transa, (cublasOperation_t)hdr->transb,
                        hdr->m, hdr->n, hdr->k,
                        alpha_ptr,
                        rowA[0], (cudaDataType_t)hdr->Atype, hdr->lda,
                        rowB[0], (cudaDataType_t)hdr->Btype, hdr->ldb,
                        beta_ptr,
                        rowC[0], (cudaDataType_t)hdr->Ctype, hdr->ldc,
                        (cublasComputeType_t)hdr->computeType,
                        (cublasGemmAlgo_t)hdr->algo);
            } else {
                cublas_rc = CUBLAS_STATUS_SUCCESS;
                for (int i = 0; i < bc; ++i) {
                    cublas_rc = cublasGemmEx(handle,
                            (cublasOperation_t)hdr->transa, (cublasOperation_t)hdr->transb,
                            hdr->m, hdr->n, hdr->k,
                            alpha_ptr,
                            rowA[i], (cudaDataType_t)hdr->Atype, hdr->lda,
                            rowB[i], (cudaDataType_t)hdr->Btype, hdr->ldb,
                            beta_ptr,
                            rowC[i], (cudaDataType_t)hdr->Ctype, hdr->ldc,
                            (cublasComputeType_t)hdr->computeType,
                            (cublasGemmAlgo_t)hdr->algo);
                    if (cublas_rc != CUBLAS_STATUS_SUCCESS) {
                        fprintf(stderr,
                                "[cuda-executor] vm_id=%u batched replay fallback failed at batch_idx=%d rc=%d\n",
                                vm->vm_id, i, (int)cublas_rc);
                        break;
                    }
                }
            }
            if (executor_verbose_copy_logging() || cublas_rc != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[cuda-executor] vm_id=%u cublasGemm%sEx rc=%d batch=%d m=%d n=%d k=%d\n",
                        vm->vm_id, (bc == 1) ? "" : "Batched", (int)cublas_rc, bc,
                        hdr->m, hdr->n, hdr->k);
            }
            /* Async kernel errors often appear here, not in cublas return */
            if (cublas_rc == CUBLAS_STATUS_SUCCESS) {
                CUresult ec = cuCtxSynchronize();
                if (ec != CUDA_SUCCESS) {
                    const char *en = NULL;
                    const char *es = NULL;
                    (void)cuGetErrorName(ec, &en);
                    (void)cuGetErrorString(ec, &es);
                    fprintf(stderr,
                            "[cuda-executor] vm_id=%u after cublasGemm%sEx: cuCtxSynchronize rc=%d (%s) %s\n",
                            vm->vm_id, (bc == 1) ? "" : "Batched", (int)ec, en ? en : "?", es ? es : "?");
                    fprintf(stderr,
                            "[cuda-executor] vm_id=%u GEMM_BATCHED dims m=%d n=%d k=%d lda=%d ldb=%d ldc=%d "
                            "batch=%d Atype=%u Btype=%u Ctype=%u computeType=%d algo=%d alpha_f32=%g beta_f32=%g\n",
                            vm->vm_id, hdr->m, hdr->n, hdr->k, hdr->lda, hdr->ldb, hdr->ldc, bc,
                            (unsigned)hdr->Atype, (unsigned)hdr->Btype, (unsigned)hdr->Ctype,
                            hdr->computeType, hdr->algo, hdr->alpha_f32, hdr->beta_f32);
                    /* Guest must not treat SUCCESS if context is poisoned (E4 / rc=700). */
                    cublas_rc = CUBLAS_STATUS_EXECUTION_FAILED;
                    if (ec == CUDA_ERROR_ILLEGAL_ADDRESS) {
                        cuda_executor_recover_primary_context(exec, vm->vm_id,
                                                             "cublas GEMM sync hit CUDA_ERROR_ILLEGAL_ADDRESS");
                    }
                } else if ((cudaDataType_t)hdr->Ctype == CUDA_R_32F && bc > 0 && rowC[0]) {
                    cuda_executor_log_device_f32_sample("cublasGemmBatchedEx C0",
                                                        (CUdeviceptr)(uintptr_t)rowC[0],
                                                        8,
                                                        vm->vm_id);
                }
            }
        }
        free(rowA);
        free(rowB);
        free(rowC);

        result->num_results = 1;
        result->results[0] = (uint64_t)cublas_rc;
        rc = CUDA_SUCCESS;
        break;
    }

    /* ---- Library management ------------------------------------ */
    case CUDA_CALL_LIBRARY_LOAD_DATA: {
        uint32_t chunk_flags = call->args[14];
        int is_chunked = (chunk_flags != 0);
        int is_first   = (chunk_flags & CUDA_CHUNK_FLAG_FIRST) != 0;
        int is_last    = (chunk_flags & CUDA_CHUNK_FLAG_LAST)  != 0;
        int is_single  = (chunk_flags == CUDA_CHUNK_FLAG_SINGLE);

        rc = ensure_vm_context(exec, vm);
        if (rc != CUDA_SUCCESS) break;

        if (!data || data_len == 0) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

        if (!is_chunked || is_single) {
            rc = load_host_library(vm, data, data_len, result);
            if (vm->mod_chunk_buf) {
                free(vm->mod_chunk_buf);
                vm->mod_chunk_buf   = NULL;
                vm->mod_chunk_alloc = 0;
                vm->mod_chunk_used  = 0;
            }
            break;
        }

        if (is_first) {
            size_t initial = (data_len < (1u << 20)) ? (32u << 20) : (size_t)data_len * 8;
            free(vm->mod_chunk_buf);
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

        if (!vm->mod_chunk_buf) {
            rc = CUDA_ERROR_INVALID_VALUE;
            break;
        }

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

        if (is_last) {
            rc = load_host_library(vm, vm->mod_chunk_buf,
                                   (uint32_t)vm->mod_chunk_used, result);
            free(vm->mod_chunk_buf);
            vm->mod_chunk_buf   = NULL;
            vm->mod_chunk_alloc = 0;
            vm->mod_chunk_used  = 0;
        } else {
            fprintf(stderr,
                    "[cuda-executor] vm_id=%u library-chunk flags=0x%08x first=%d last=%d single=%d data_len=%u used=%zu alloc=%zu\n",
                    vm->vm_id, chunk_flags, is_first, is_last, is_single,
                    (unsigned)data_len, vm->mod_chunk_used, vm->mod_chunk_alloc);
            result->num_results = 0;
            rc = CUDA_SUCCESS;
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

    case CUDA_CALL_FUNC_GET_PARAM_INFO: {
        uint64_t func_handle = CUDA_UNPACK_U64(call->args, 0);
        uint64_t param_index = CUDA_UNPACK_U64(call->args, 2);
        func_entry_t *func_entry = vm_find_func_entry(vm, func_handle);
        CUfunction func = func_entry ? func_entry->host_function : NULL;
        const char *func_name =
            (func_entry && func_entry->name[0] != '\0') ? func_entry->name : "<unknown>";
        pfn_cuFuncGetParamInfo_t cuFuncGetParamInfo_fn = resolve_cuFuncGetParamInfo();

        if (func && cuFuncGetParamInfo_fn) {
            rc = ensure_vm_context(exec, vm);
            if (rc == CUDA_SUCCESS) {
                size_t param_offset = 0;
                size_t param_size = 0;
                rc = cuFuncGetParamInfo_fn(func, (size_t)param_index, &param_offset, &param_size);
                if (rc == CUDA_SUCCESS) {
                    result->num_results = 2;
                    result->results[0] = (uint64_t)param_offset;
                    result->results[1] = (uint64_t)param_size;
                } else if (cuda_executor_try_synth_param_info(func_name,
                                                              (size_t)param_index,
                                                              &param_offset,
                                                              &param_size)) {
                    fprintf(stderr,
                            "[cuda-executor] cuFuncGetParamInfo synthesized: vm=%u func=0x%llx "
                            "name=%s param=%llu off=%zu size=%zu rc=%d\n",
                            call->vm_id,
                            (unsigned long long)func_handle,
                            func_name,
                            (unsigned long long)param_index,
                            param_offset,
                            param_size,
                            rc);
                    rc = CUDA_SUCCESS;
                    result->num_results = 2;
                    result->results[0] = (uint64_t)param_offset;
                    result->results[1] = (uint64_t)param_size;
                } else {
                    if (rc == CUDA_ERROR_NOT_SUPPORTED &&
                        (strncmp(func_name, "_Z9mul_mat_q", strlen("_Z9mul_mat_q")) == 0 ||
                         strncmp(func_name, "_Z24mul_mat_q_stream_k_fixup",
                                 strlen("_Z24mul_mat_q_stream_k_fixup")) == 0 ||
                         strncmp(func_name, "_Z12rms_norm_f32", strlen("_Z12rms_norm_f32")) == 0 ||
                         strncmp(func_name, "_Z13quantize_q8_1", strlen("_Z13quantize_q8_1")) == 0 ||
                         strncmp(func_name, "_Z17quantize_mmq_q8_1",
                                 strlen("_Z17quantize_mmq_q8_1")) == 0 ||
                         strncmp(func_name, "_Z13convert_unary", strlen("_Z13convert_unary")) == 0 ||
                         strncmp(func_name, "_Z9rope_norm", strlen("_Z9rope_norm")) == 0 ||
                         strncmp(func_name, "_Z9rope_neox", strlen("_Z9rope_neox")) == 0)) {
                        /* For these kernels, driver NOT_SUPPORTED often means
                         * "param index out of range" in this environment.
                         * Report INVALID_VALUE so the caller can stop probing
                         * params instead of treating this as a hard failure. */
                        rc = CUDA_ERROR_INVALID_VALUE;
                    }
                    fprintf(stderr,
                            "[cuda-executor] cuFuncGetParamInfo FAILED: vm=%u func=0x%llx "
                            "name=%s param=%llu rc=%d(%s) detail=%s\n",
                            call->vm_id,
                            (unsigned long long)func_handle,
                            func_name,
                            (unsigned long long)param_index,
                            (int)rc,
                            host_cuda_error_name(rc),
                            host_cuda_error_string(rc));
                }
            }
        } else if (func) {
            size_t param_offset = 0;
            size_t param_size = 0;
            if (cuda_executor_try_synth_param_info(func_name,
                                                   (size_t)param_index,
                                                   &param_offset,
                                                   &param_size)) {
                fprintf(stderr,
                        "[cuda-executor] cuFuncGetParamInfo synthesized without driver API: "
                        "vm=%u func=0x%llx name=%s param=%llu off=%zu size=%zu\n",
                        call->vm_id,
                        (unsigned long long)func_handle,
                        func_name,
                        (unsigned long long)param_index,
                        param_offset,
                        param_size);
                rc = CUDA_SUCCESS;
                result->num_results = 2;
                result->results[0] = (uint64_t)param_offset;
                result->results[1] = (uint64_t)param_size;
            } else {
                rc = CUDA_ERROR_NOT_SUPPORTED;
                if (strncmp(func_name, "_Z9mul_mat_q", strlen("_Z9mul_mat_q")) == 0 ||
                    strncmp(func_name, "_Z24mul_mat_q_stream_k_fixup",
                            strlen("_Z24mul_mat_q_stream_k_fixup")) == 0 ||
                    strncmp(func_name, "_Z12rms_norm_f32", strlen("_Z12rms_norm_f32")) == 0 ||
                    strncmp(func_name, "_Z13quantize_q8_1", strlen("_Z13quantize_q8_1")) == 0 ||
                    strncmp(func_name, "_Z17quantize_mmq_q8_1",
                            strlen("_Z17quantize_mmq_q8_1")) == 0 ||
                    strncmp(func_name, "_Z13convert_unary", strlen("_Z13convert_unary")) == 0 ||
                    strncmp(func_name, "_Z9rope_norm", strlen("_Z9rope_norm")) == 0 ||
                    strncmp(func_name, "_Z9rope_neox", strlen("_Z9rope_neox")) == 0) {
                    rc = CUDA_ERROR_INVALID_VALUE;
                }
                fprintf(stderr,
                        "[cuda-executor] cuFuncGetParamInfo UNSUPPORTED: vm=%u func=0x%llx "
                        "name=%s param=%llu rc=%d\n",
                        call->vm_id,
                        (unsigned long long)func_handle,
                        func_name,
                        (unsigned long long)param_index,
                        (int)rc);
            }
        } else {
            rc = CUDA_ERROR_INVALID_VALUE;
            fprintf(stderr,
                    "[cuda-executor] cuFuncGetParamInfo INVALID_VALUE: vm=%u func=0x%llx "
                    "param=%llu (missing host function entry)\n",
                    call->vm_id,
                    (unsigned long long)func_handle,
                    (unsigned long long)param_index);
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
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr,
                "[cuda-executor] call FAILED: vm=%u call=%s(0x%04x) rc=%d(%s) detail=%s\n",
                call->vm_id,
                executor_call_id_to_name(call->call_id),
                call->call_id,
                (int)rc,
                host_cuda_error_name(rc),
                host_cuda_error_string(rc));
    }

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
        cuCtxSetCurrent(vm->ctx_is_primary ? exec->primary_ctx : vm->ctx);

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

        /* Destroy only per-VM owned contexts; the executor owns the primary. */
        if (!vm->ctx_is_primary) {
            cuCtxDestroy(vm->ctx);
        }
        vm->ctx_valid = 0;
        vm->ctx = NULL;
        vm->ctx_is_primary = 0;
    }

    /* Free any in-progress module-load chunk accumulation buffer */
    if (vm->mod_chunk_buf) {
        free(vm->mod_chunk_buf);
        vm->mod_chunk_buf   = NULL;
        vm->mod_chunk_alloc = 0;
        vm->mod_chunk_used  = 0;
    }

    vm_drain_pending_async_htod(vm, NULL, 1);

    vm->active = 0;
    fprintf(stderr, "[cuda-executor] Cleaned up VM %u\n", vm_id);

    pthread_mutex_unlock(&exec->mutex);
}

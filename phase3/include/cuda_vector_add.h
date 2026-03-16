#ifndef CUDA_VECTOR_ADD_H
#define CUDA_VECTOR_ADD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*cuda_callback_t)(int result, void *user_data);

int cuda_init(void);
int cuda_vector_add_async(int num1, int num2, cuda_callback_t callback, void *user_data);

int cuda_is_busy(void);
int cuda_sync(void);
void cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_VECTOR_ADD_H */

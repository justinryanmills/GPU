/*
 * CUDA Vector Addition Header
 * 
 * Interface for asynchronous CUDA vector addition
 */

#ifndef CUDA_VECTOR_ADD_H
#define CUDA_VECTOR_ADD_H

#ifdef __cplusplus
extern "C" {
#endif

// Callback function type
// Called when vector addition result is ready
typedef void (*cuda_callback_t)(int result, void *user_data);

/*
 * Initialize CUDA
 * Must be called before using other functions
 * 
 * Returns:
 *   0 on success
 *   -1 on failure
 */
int cuda_init(void);

/*
 * Execute vector addition asynchronously
 * 
 * Parameters:
 *   num1, num2: Numbers to add
 *   callback: Function to call when result is ready
 *   user_data: User data passed to callback (can be NULL)
 * 
 * Returns:
 *   0 on success (operation started)
 *   -1 on failure
 * 
 * Note: The callback is called from a worker thread.
 *       Make sure callback is thread-safe.
 */
int cuda_vector_add_async(int num1, int num2, cuda_callback_t callback, void *user_data);

/*
 * Check if CUDA is currently processing
 * 
 * Returns:
 *   1 if CUDA is busy
 *   0 if CUDA is idle
 */
int cuda_is_busy(void);

/*
 * Wait for all pending CUDA operations to complete
 * 
 * Returns:
 *   0 on success
 *   -1 on failure
 */
int cuda_sync(void);

/*
 * Cleanup CUDA resources
 * Should be called when done using CUDA
 */
void cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_VECTOR_ADD_H */

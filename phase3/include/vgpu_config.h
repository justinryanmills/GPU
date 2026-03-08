/*
 * vGPU Configuration Library Header
 * Configuration & Management Interface (Step 2-4)
 *
 * This library provides functions for managing vGPU configuration in SQLite database.
 */

#ifndef VGPU_CONFIG_H
#define VGPU_CONFIG_H

#include <sqlite3.h>
#include <stdint.h>

/* Database path */
#define VGPU_DB_PATH "/etc/vgpu/vgpu_config.db"

/* Return codes */
#define VGPU_OK 0
#define VGPU_ERROR -1
#define VGPU_NOT_FOUND -2
#define VGPU_INVALID_PARAM -3
#define VGPU_DB_ERROR -4

/* Priority levels */
#define VGPU_PRIORITY_LOW 0
#define VGPU_PRIORITY_MEDIUM 1
#define VGPU_PRIORITY_HIGH 2

/* Pool IDs */
#define VGPU_POOL_A 'A'
#define VGPU_POOL_B 'B'

/* VM configuration structure */
typedef struct {
    int vm_id;
    char vm_uuid[64];
    char vm_name[256];
    char pool_id;
    int priority;
    /* Phase 3 fields */
    int weight;             /* Scheduler weight (1-100, default 50)          */
    int max_jobs_per_sec;   /* Rate limit: max jobs/sec (0 = unlimited)      */
    int max_queue_depth;    /* Rate limit: max queued jobs (0 = unlimited)   */
    int quarantined;        /* 1 = VM is quarantined (faults), 0 = normal    */
    int error_count;        /* Cumulative error/fault counter                */
    char created_at[32];
    char updated_at[32];
} vgpu_vm_config_t;

/* Pool information structure */
typedef struct {
    char pool_id;
    char pool_name[64];
    char description[256];
    int enabled;
    char created_at[32];
    char updated_at[32];
    int vm_count;  /* Number of VMs in this pool */
} vgpu_pool_info_t;

/* ============================================================================
 * Database Management
 * ============================================================================ */

/**
 * Initialize database connection
 * @param db Pointer to sqlite3* to store connection
 * @return VGPU_OK on success, VGPU_ERROR on failure
 */
int vgpu_db_init(sqlite3 **db);

/**
 * Close database connection
 * @param db Database connection
 */
void vgpu_db_close(sqlite3 *db);

/**
 * Initialize database schema (create tables, indexes, insert pools)
 * @param db Database connection
 * @return VGPU_OK on success, VGPU_ERROR on failure
 */
int vgpu_db_init_schema(sqlite3 *db);

/* ============================================================================
 * Pool Management
 * ============================================================================ */

/**
 * Get pool information
 * @param db Database connection
 * @param pool_id Pool ID ('A' or 'B')
 * @param pool_info Output structure to fill
 * @return VGPU_OK on success, VGPU_NOT_FOUND if pool doesn't exist
 */
int vgpu_get_pool_info(sqlite3 *db, char pool_id, vgpu_pool_info_t *pool_info);

/**
 * List all pools with VM counts
 * @param db Database connection
 * @param pools Output array of pool info (must be at least 2 elements)
 * @param count Output: number of pools returned
 * @return VGPU_OK on success
 */
int vgpu_list_pools(sqlite3 *db, vgpu_pool_info_t *pools, int *count);

/* ============================================================================
 * VM Management
 * ============================================================================ */

/**
 * Get VM configuration by UUID
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @param config Output structure to fill
 * @return VGPU_OK on success, VGPU_NOT_FOUND if VM not registered
 */
int vgpu_get_vm_config(sqlite3 *db, const char *vm_uuid, vgpu_vm_config_t *config);

/**
 * Register a new VM
 * @param db Database connection
 * @param vm_uuid VM UUID (required)
 * @param vm_name VM name (optional, can be NULL)
 * @param pool_id Pool ID ('A' or 'B', defaults to 'A' if 0)
 * @param priority Priority (0=low, 1=medium, 2=high, defaults to 1 if -1)
 * @param vm_id VM ID (user-assignable, must be unique, 0 to auto-assign next available)
 * @return VGPU_OK on success, VGPU_ERROR on failure
 */
int vgpu_register_vm(sqlite3 *db, const char *vm_uuid, const char *vm_name,
                     char pool_id, int priority, int vm_id);

/**
 * Update VM pool assignment
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @param pool_id New pool ID ('A' or 'B')
 * @return VGPU_OK on success, VGPU_NOT_FOUND if VM not registered
 */
int vgpu_set_vm_pool(sqlite3 *db, const char *vm_uuid, char pool_id);

/**
 * Update VM priority
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @param priority New priority (0=low, 1=medium, 2=high)
 * @return VGPU_OK on success, VGPU_NOT_FOUND if VM not registered
 */
int vgpu_set_vm_priority(sqlite3 *db, const char *vm_uuid, int priority);

/**
 * Update VM ID
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @param vm_id New VM ID (must be unique)
 * @return VGPU_OK on success, VGPU_NOT_FOUND if VM not registered, VGPU_ERROR if vm_id already exists
 */
int vgpu_set_vm_id(sqlite3 *db, const char *vm_uuid, int vm_id);

/**
 * Update multiple VM properties
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @param pool_id New pool ID (0 to keep unchanged)
 * @param priority New priority (-1 to keep unchanged)
 * @param vm_id New VM ID (0 to keep unchanged)
 * @return VGPU_OK on success, VGPU_NOT_FOUND if VM not registered
 */
int vgpu_update_vm(sqlite3 *db, const char *vm_uuid, char pool_id, int priority, int vm_id);

/**
 * Remove VM from database
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @return VGPU_OK on success, VGPU_NOT_FOUND if VM not registered
 */
int vgpu_remove_vm(sqlite3 *db, const char *vm_uuid);

/**
 * List VMs with optional filters
 * @param db Database connection
 * @param pool_id Filter by pool ('A', 'B', or 0 for all)
 * @param priority Filter by priority (0, 1, 2, or -1 for all)
 * @param configs Output array (must be pre-allocated, large enough)
 * @param count Output: number of VMs returned
 * @param max_count Maximum number of VMs to return
 * @return VGPU_OK on success
 */
int vgpu_list_vms(sqlite3 *db, char pool_id, int priority,
                   vgpu_vm_config_t *configs, int *count, int max_count);

/**
 * Get next available VM ID
 * @param db Database connection
 * @return Next available VM ID (starting from 1)
 */
int vgpu_get_next_vm_id(sqlite3 *db);

/**
 * Check if VM ID is already in use
 * @param db Database connection
 * @param vm_id VM ID to check
 * @return 1 if in use, 0 if available
 */
int vgpu_vm_id_in_use(sqlite3 *db, int vm_id);

/* ============================================================================
 * Phase 3: Scheduler Weight & Isolation Controls
 * ============================================================================ */

/**
 * Set VM scheduler weight
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @param weight Weight value (1-100)
 * @return VGPU_OK on success
 */
int vgpu_set_vm_weight(sqlite3 *db, const char *vm_uuid, int weight);

/**
 * Set VM rate limit (max jobs per second)
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @param max_jobs_per_sec Maximum jobs per second (0 = unlimited)
 * @param max_queue_depth Maximum queue depth (0 = unlimited)
 * @return VGPU_OK on success
 */
int vgpu_set_vm_rate_limit(sqlite3 *db, const char *vm_uuid,
                           int max_jobs_per_sec, int max_queue_depth);

/**
 * Quarantine a VM (block new submissions due to faults)
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @return VGPU_OK on success
 */
int vgpu_quarantine_vm(sqlite3 *db, const char *vm_uuid);

/**
 * Lift quarantine from a VM
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @return VGPU_OK on success
 */
int vgpu_unquarantine_vm(sqlite3 *db, const char *vm_uuid);

/**
 * Increment VM error counter
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @return VGPU_OK on success
 */
int vgpu_increment_error_count(sqlite3 *db, const char *vm_uuid);

/**
 * Reset VM error counter
 * @param db Database connection
 * @param vm_uuid VM UUID
 * @return VGPU_OK on success
 */
int vgpu_reset_error_count(sqlite3 *db, const char *vm_uuid);

/**
 * Get VM configuration by integer vm_id (Phase 3: for mediator lookups)
 * @param db Database connection
 * @param vm_id VM numeric ID
 * @param config Output structure
 * @return VGPU_OK on success, VGPU_NOT_FOUND if not registered
 */
int vgpu_get_vm_config_by_id(sqlite3 *db, int vm_id, vgpu_vm_config_t *config);

#endif /* VGPU_CONFIG_H */

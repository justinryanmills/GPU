#ifndef VGPU_CONFIG_H
#define VGPU_CONFIG_H

#include <sqlite3.h>
#include <stdint.h>

#define VGPU_DB_PATH "/etc/vgpu/vgpu_config.db"
#define VGPU_OK 0
#define VGPU_ERROR -1
#define VGPU_NOT_FOUND -2
#define VGPU_INVALID_PARAM -3
#define VGPU_DB_ERROR -4
#define VGPU_PRIORITY_LOW 0
#define VGPU_PRIORITY_MEDIUM 1
#define VGPU_PRIORITY_HIGH 2
#define VGPU_POOL_A 'A'
#define VGPU_POOL_B 'B'

typedef struct {
    int vm_id;
    char vm_uuid[64];
    char vm_name[256];
    char pool_id;
    int priority;
    int weight;
    int max_jobs_per_sec;
    int max_queue_depth;
    int quarantined;
    int error_count;
    char created_at[32];
    char updated_at[32];
} vgpu_vm_config_t;

typedef struct {
    char pool_id;
    char pool_name[64];
    char description[256];
    int enabled;
    char created_at[32];
    char updated_at[32];
    int vm_count;
} vgpu_pool_info_t;

int vgpu_db_init(sqlite3 **db);
void vgpu_db_close(sqlite3 *db);
int vgpu_db_init_schema(sqlite3 *db);
int vgpu_get_pool_info(sqlite3 *db, char pool_id, vgpu_pool_info_t *pool_info);
int vgpu_list_pools(sqlite3 *db, vgpu_pool_info_t *pools, int *count);
int vgpu_get_vm_config(sqlite3 *db, const char *vm_uuid, vgpu_vm_config_t *config);

int vgpu_register_vm(sqlite3 *db, const char *vm_uuid, const char *vm_name,
                     char pool_id, int priority, int vm_id);

int vgpu_set_vm_pool(sqlite3 *db, const char *vm_uuid, char pool_id);

int vgpu_set_vm_priority(sqlite3 *db, const char *vm_uuid, int priority);

int vgpu_set_vm_id(sqlite3 *db, const char *vm_uuid, int vm_id);

int vgpu_update_vm(sqlite3 *db, const char *vm_uuid, char pool_id, int priority, int vm_id);

int vgpu_remove_vm(sqlite3 *db, const char *vm_uuid);

int vgpu_list_vms(sqlite3 *db, char pool_id, int priority,
                   vgpu_vm_config_t *configs, int *count, int max_count);

int vgpu_get_next_vm_id(sqlite3 *db);

int vgpu_vm_id_in_use(sqlite3 *db, int vm_id);

int vgpu_set_vm_weight(sqlite3 *db, const char *vm_uuid, int weight);

int vgpu_set_vm_rate_limit(sqlite3 *db, const char *vm_uuid,
                           int max_jobs_per_sec, int max_queue_depth);

int vgpu_quarantine_vm(sqlite3 *db, const char *vm_uuid);

int vgpu_unquarantine_vm(sqlite3 *db, const char *vm_uuid);

int vgpu_increment_error_count(sqlite3 *db, const char *vm_uuid);

int vgpu_reset_error_count(sqlite3 *db, const char *vm_uuid);

int vgpu_get_vm_config_by_id(sqlite3 *db, int vm_id, vgpu_vm_config_t *config);

#endif /* VGPU_CONFIG_H */

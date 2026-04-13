/*
 * vGPU Configuration Library Implementation
 * Configuration & Management Interface (Step 2-4)
 */

#include "vgpu_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

/* ============================================================================
 * Database Management
 * ============================================================================ */

int vgpu_db_init(sqlite3 **db) {
    int rc;
    
    if (db == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    rc = sqlite3_open(VGPU_DB_PATH, db);
    if (rc != SQLITE_OK) {
        if (*db) {
            fprintf(stderr, "Database error: %s\n", sqlite3_errmsg(*db));
            sqlite3_close(*db);
        }
        return VGPU_DB_ERROR;
    }
    
    /* Enable foreign keys */
    sqlite3_exec(*db, "PRAGMA foreign_keys = ON;", NULL, NULL, NULL);
    
    return VGPU_OK;
}

void vgpu_db_close(sqlite3 *db) {
    if (db) {
        sqlite3_close(db);
    }
}

int vgpu_db_init_schema(sqlite3 *db) {
    const char *sql;
    char *err_msg = NULL;
    int rc;
    
    /* Create pools table */
    sql = "CREATE TABLE IF NOT EXISTS pools ("
          "    pool_id CHAR(1) PRIMARY KEY CHECK(pool_id IN ('A', 'B')),"
          "    pool_name TEXT NOT NULL,"
          "    description TEXT,"
          "    enabled INTEGER DEFAULT 1 CHECK(enabled IN (0, 1)),"
          "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
          "    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
          ");";
    
    rc = sqlite3_exec(db, sql, NULL, NULL, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Error creating pools table: %s\n", err_msg);
        sqlite3_free(err_msg);
        return VGPU_DB_ERROR;
    }
    
    /* Create vms table */
    sql = "CREATE TABLE IF NOT EXISTS vms ("
          "    vm_id INTEGER NOT NULL UNIQUE,"
          "    vm_uuid TEXT UNIQUE NOT NULL,"
          "    vm_name TEXT,"
          "    pool_id CHAR(1) NOT NULL DEFAULT 'A' CHECK(pool_id IN ('A', 'B')),"
          "    priority INTEGER NOT NULL DEFAULT 1 CHECK(priority IN (0, 1, 2)),"
          "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
          "    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
          "    FOREIGN KEY (pool_id) REFERENCES pools(pool_id)"
          ");";
    
    rc = sqlite3_exec(db, sql, NULL, NULL, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Error creating vms table: %s\n", err_msg);
        sqlite3_free(err_msg);
        return VGPU_DB_ERROR;
    }
    
    /* Create indexes */
    sql = "CREATE INDEX IF NOT EXISTS idx_vm_uuid ON vms(vm_uuid);";
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    
    sql = "CREATE INDEX IF NOT EXISTS idx_vm_pool ON vms(pool_id);";
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    
    sql = "CREATE INDEX IF NOT EXISTS idx_vm_priority ON vms(priority);";
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    
    /* Insert Pool A and Pool B if they don't exist */
    sql = "INSERT OR IGNORE INTO pools (pool_id, pool_name, enabled) VALUES "
          "    ('A', 'Pool A', 1),"
          "    ('B', 'Pool B', 1);";
    
    rc = sqlite3_exec(db, sql, NULL, NULL, &err_msg);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Error inserting pools: %s\n", err_msg);
        sqlite3_free(err_msg);
        return VGPU_DB_ERROR;
    }
    
    /* Phase 3: Add new columns if they don't exist yet (migration) */
    /* SQLite doesn't have IF NOT EXISTS for ALTER TABLE, so we check first */
    sqlite3_stmt *check_stmt;
    rc = sqlite3_prepare_v2(db, "PRAGMA table_info(vms);", -1, &check_stmt, NULL);
    if (rc == SQLITE_OK) {
        int has_weight = 0;
        while (sqlite3_step(check_stmt) == SQLITE_ROW) {
            const char *col = (const char *)sqlite3_column_text(check_stmt, 1);
            if (col && strcmp(col, "weight") == 0) {
                has_weight = 1;
                break;
            }
        }
        sqlite3_finalize(check_stmt);
        
        if (!has_weight) {
            /* Add Phase 3 columns */
            sqlite3_exec(db, "ALTER TABLE vms ADD COLUMN weight INTEGER NOT NULL DEFAULT 50;",
                        NULL, NULL, NULL);
            sqlite3_exec(db, "ALTER TABLE vms ADD COLUMN max_jobs_per_sec INTEGER NOT NULL DEFAULT 0;",
                        NULL, NULL, NULL);
            sqlite3_exec(db, "ALTER TABLE vms ADD COLUMN max_queue_depth INTEGER NOT NULL DEFAULT 0;",
                        NULL, NULL, NULL);
            sqlite3_exec(db, "ALTER TABLE vms ADD COLUMN quarantined INTEGER NOT NULL DEFAULT 0;",
                        NULL, NULL, NULL);
            sqlite3_exec(db, "ALTER TABLE vms ADD COLUMN error_count INTEGER NOT NULL DEFAULT 0;",
                        NULL, NULL, NULL);
        }
    }
    
    return VGPU_OK;
}

/* ============================================================================
 * Pool Management
 * ============================================================================ */

int vgpu_get_pool_info(sqlite3 *db, char pool_id, vgpu_pool_info_t *pool_info) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    
    if (db == NULL || pool_info == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    if (pool_id != 'A' && pool_id != 'B') {
        return VGPU_INVALID_PARAM;
    }
    
    sql = "SELECT pool_id, pool_name, description, enabled, created_at, updated_at, "
          "       (SELECT COUNT(*) FROM vms WHERE vms.pool_id = pools.pool_id) as vm_count "
          "FROM pools WHERE pool_id = ?;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    sqlite3_bind_text(stmt, 1, &pool_id, 1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        pool_info->pool_id = pool_id;
        strncpy(pool_info->pool_name, (const char *)sqlite3_column_text(stmt, 1), sizeof(pool_info->pool_name) - 1);
        pool_info->pool_name[sizeof(pool_info->pool_name) - 1] = '\0';
        
        const char *desc = (const char *)sqlite3_column_text(stmt, 2);
        if (desc) {
            strncpy(pool_info->description, desc, sizeof(pool_info->description) - 1);
            pool_info->description[sizeof(pool_info->description) - 1] = '\0';
        } else {
            pool_info->description[0] = '\0';
        }
        
        pool_info->enabled = sqlite3_column_int(stmt, 3);
        
        const char *created = (const char *)sqlite3_column_text(stmt, 4);
        if (created) {
            strncpy(pool_info->created_at, created, sizeof(pool_info->created_at) - 1);
            pool_info->created_at[sizeof(pool_info->created_at) - 1] = '\0';
        }
        
        const char *updated = (const char *)sqlite3_column_text(stmt, 5);
        if (updated) {
            strncpy(pool_info->updated_at, updated, sizeof(pool_info->updated_at) - 1);
            pool_info->updated_at[sizeof(pool_info->updated_at) - 1] = '\0';
        }
        
        pool_info->vm_count = sqlite3_column_int(stmt, 6);
        
        sqlite3_finalize(stmt);
        return VGPU_OK;
    }
    
    sqlite3_finalize(stmt);
    return VGPU_NOT_FOUND;
}

int vgpu_list_pools(sqlite3 *db, vgpu_pool_info_t *pools, int *count) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    int i = 0;
    
    if (db == NULL || pools == NULL || count == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    sql = "SELECT pool_id, pool_name, description, enabled, created_at, updated_at, "
          "       (SELECT COUNT(*) FROM vms WHERE vms.pool_id = pools.pool_id) as vm_count "
          "FROM pools ORDER BY pool_id;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW && i < 2) {
        pools[i].pool_id = ((const char *)sqlite3_column_text(stmt, 0))[0];
        
        const char *name = (const char *)sqlite3_column_text(stmt, 1);
        if (name) {
            strncpy(pools[i].pool_name, name, sizeof(pools[i].pool_name) - 1);
            pools[i].pool_name[sizeof(pools[i].pool_name) - 1] = '\0';
        }
        
        const char *desc = (const char *)sqlite3_column_text(stmt, 2);
        if (desc) {
            strncpy(pools[i].description, desc, sizeof(pools[i].description) - 1);
            pools[i].description[sizeof(pools[i].description) - 1] = '\0';
        } else {
            pools[i].description[0] = '\0';
        }
        
        pools[i].enabled = sqlite3_column_int(stmt, 3);
        
        const char *created = (const char *)sqlite3_column_text(stmt, 4);
        if (created) {
            strncpy(pools[i].created_at, created, sizeof(pools[i].created_at) - 1);
            pools[i].created_at[sizeof(pools[i].created_at) - 1] = '\0';
        }
        
        const char *updated = (const char *)sqlite3_column_text(stmt, 5);
        if (updated) {
            strncpy(pools[i].updated_at, updated, sizeof(pools[i].updated_at) - 1);
            pools[i].updated_at[sizeof(pools[i].updated_at) - 1] = '\0';
        }
        
        pools[i].vm_count = sqlite3_column_int(stmt, 6);
        i++;
    }
    
    sqlite3_finalize(stmt);
    *count = i;
    return VGPU_OK;
}

/* ============================================================================
 * VM Management
 * ============================================================================ */

int vgpu_get_vm_config(sqlite3 *db, const char *vm_uuid, vgpu_vm_config_t *config) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    
    if (db == NULL || vm_uuid == NULL || config == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    sql = "SELECT vm_id, vm_uuid, vm_name, pool_id, priority,"
          "       weight, max_jobs_per_sec, max_queue_depth, quarantined, error_count,"
          "       created_at, updated_at "
          "FROM vms WHERE vm_uuid = ?;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    sqlite3_bind_text(stmt, 1, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        config->vm_id = sqlite3_column_int(stmt, 0);
        
        const char *uuid = (const char *)sqlite3_column_text(stmt, 1);
        if (uuid) {
            strncpy(config->vm_uuid, uuid, sizeof(config->vm_uuid) - 1);
            config->vm_uuid[sizeof(config->vm_uuid) - 1] = '\0';
        }
        
        const char *name = (const char *)sqlite3_column_text(stmt, 2);
        if (name) {
            strncpy(config->vm_name, name, sizeof(config->vm_name) - 1);
            config->vm_name[sizeof(config->vm_name) - 1] = '\0';
        } else {
            config->vm_name[0] = '\0';
        }
        
        const char *pool = (const char *)sqlite3_column_text(stmt, 3);
        if (pool) {
            config->pool_id = pool[0];
        }
        
        config->priority = sqlite3_column_int(stmt, 4);
        
        /* Phase 3 fields */
        config->weight          = sqlite3_column_int(stmt, 5);
        config->max_jobs_per_sec = sqlite3_column_int(stmt, 6);
        config->max_queue_depth = sqlite3_column_int(stmt, 7);
        config->quarantined     = sqlite3_column_int(stmt, 8);
        config->error_count     = sqlite3_column_int(stmt, 9);
        
        const char *created = (const char *)sqlite3_column_text(stmt, 10);
        if (created) {
            strncpy(config->created_at, created, sizeof(config->created_at) - 1);
            config->created_at[sizeof(config->created_at) - 1] = '\0';
        }
        
        const char *updated = (const char *)sqlite3_column_text(stmt, 11);
        if (updated) {
            strncpy(config->updated_at, updated, sizeof(config->updated_at) - 1);
            config->updated_at[sizeof(config->updated_at) - 1] = '\0';
        }
        
        sqlite3_finalize(stmt);
        return VGPU_OK;
    }
    
    sqlite3_finalize(stmt);
    return VGPU_NOT_FOUND;
}

int vgpu_get_next_vm_id(sqlite3 *db) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    int max_id = 0;
    
    if (db == NULL) {
        return 1; /* Default to 1 if error */
    }
    
    sql = "SELECT COALESCE(MAX(vm_id), 0) FROM vms;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return 1;
    }
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        max_id = sqlite3_column_int(stmt, 0);
    }
    
    sqlite3_finalize(stmt);
    return max_id + 1;
}

int vgpu_vm_id_in_use(sqlite3 *db, int vm_id) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    int count = 0;
    
    if (db == NULL) {
        return 0;
    }
    
    sql = "SELECT COUNT(*) FROM vms WHERE vm_id = ?;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return 0;
    }
    
    sqlite3_bind_int(stmt, 1, vm_id);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    
    sqlite3_finalize(stmt);
    return count > 0;
}

int vgpu_register_vm(sqlite3 *db, const char *vm_uuid, const char *vm_name,
                     char pool_id, int priority, int vm_id) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    int final_vm_id;
    
    if (db == NULL || vm_uuid == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    /* Validate pool_id */
    if (pool_id == 0) {
        pool_id = 'A'; /* Default to Pool A */
    } else if (pool_id != 'A' && pool_id != 'B') {
        return VGPU_INVALID_PARAM;
    }
    
    /* Validate priority */
    if (priority < 0) {
        priority = 1; /* Default to medium */
    } else if (priority > 2) {
        return VGPU_INVALID_PARAM;
    }
    
    /* Determine VM ID */
    if (vm_id == 0) {
        /* Auto-assign next available ID */
        final_vm_id = vgpu_get_next_vm_id(db);
    } else {
        /* Check if VM ID is already in use */
        if (vgpu_vm_id_in_use(db, vm_id)) {
            return VGPU_ERROR; /* VM ID already in use */
        }
        final_vm_id = vm_id;
    }
    
    sql = "INSERT INTO vms (vm_id, vm_uuid, vm_name, pool_id, priority) "
          "VALUES (?, ?, ?, ?, ?);";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    sqlite3_bind_int(stmt, 1, final_vm_id);
    sqlite3_bind_text(stmt, 2, vm_uuid, -1, SQLITE_STATIC);
    
    if (vm_name && strlen(vm_name) > 0) {
        sqlite3_bind_text(stmt, 3, vm_name, -1, SQLITE_STATIC);
    } else {
        sqlite3_bind_null(stmt, 3);
    }
    
    sqlite3_bind_text(stmt, 4, &pool_id, 1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 5, priority);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        return VGPU_DB_ERROR;
    }
    
    return VGPU_OK;
}

int vgpu_set_vm_pool(sqlite3 *db, const char *vm_uuid, char pool_id) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    if (pool_id != 'A' && pool_id != 'B') {
        return VGPU_INVALID_PARAM;
    }
    
    sql = "UPDATE vms SET pool_id = ?, updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    sqlite3_bind_text(stmt, 1, &pool_id, 1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        return VGPU_DB_ERROR;
    }
    
    if (sqlite3_changes(db) == 0) {
        return VGPU_NOT_FOUND;
    }
    
    return VGPU_OK;
}

int vgpu_set_vm_priority(sqlite3 *db, const char *vm_uuid, int priority) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    if (priority < 0 || priority > 2) {
        return VGPU_INVALID_PARAM;
    }
    
    sql = "UPDATE vms SET priority = ?, updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    sqlite3_bind_int(stmt, 1, priority);
    sqlite3_bind_text(stmt, 2, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        return VGPU_DB_ERROR;
    }
    
    if (sqlite3_changes(db) == 0) {
        return VGPU_NOT_FOUND;
    }
    
    return VGPU_OK;
}

int vgpu_set_vm_id(sqlite3 *db, const char *vm_uuid, int vm_id) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    if (vm_id <= 0) {
        return VGPU_INVALID_PARAM;
    }
    
    /* Check if VM ID is already in use by another VM */
    if (vgpu_vm_id_in_use(db, vm_id)) {
        /* Check if it's the same VM */
        vgpu_vm_config_t config;
        if (vgpu_get_vm_config(db, vm_uuid, &config) == VGPU_OK) {
            if (config.vm_id == vm_id) {
                /* Same VM, no change needed */
                return VGPU_OK;
            }
        }
        /* Different VM is using this ID */
        return VGPU_ERROR;
    }
    
    sql = "UPDATE vms SET vm_id = ?, updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    sqlite3_bind_int(stmt, 1, vm_id);
    sqlite3_bind_text(stmt, 2, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        return VGPU_DB_ERROR;
    }
    
    if (sqlite3_changes(db) == 0) {
        return VGPU_NOT_FOUND;
    }
    
    return VGPU_OK;
}

int vgpu_update_vm(sqlite3 *db, const char *vm_uuid, char pool_id, int priority, int vm_id) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    int has_changes = 0;
    
    if (db == NULL || vm_uuid == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    /* Build dynamic UPDATE statement based on what's being changed */
    sql = "UPDATE vms SET updated_at = CURRENT_TIMESTAMP";
    
    if (pool_id != 0) {
        if (pool_id != 'A' && pool_id != 'B') {
            return VGPU_INVALID_PARAM;
        }
        sql = sqlite3_mprintf("%s, pool_id = '%c'", sql, pool_id);
        has_changes = 1;
    }
    
    if (priority >= 0) {
        if (priority > 2) {
            return VGPU_INVALID_PARAM;
        }
        sql = sqlite3_mprintf("%s, priority = %d", sql, priority);
        has_changes = 1;
    }
    
    if (vm_id > 0) {
        /* Check if VM ID is already in use */
        if (vgpu_vm_id_in_use(db, vm_id)) {
            /* Check if it's the same VM */
            vgpu_vm_config_t config;
            if (vgpu_get_vm_config(db, vm_uuid, &config) == VGPU_OK) {
                if (config.vm_id != vm_id) {
                    /* Different VM is using this ID */
                    sqlite3_free((void *)sql);
                    return VGPU_ERROR;
                }
            } else {
                sqlite3_free((void *)sql);
                return VGPU_ERROR;
            }
        }
        sql = sqlite3_mprintf("%s, vm_id = %d", sql, vm_id);
        has_changes = 1;
    }
    
    if (!has_changes) {
        sqlite3_free((void *)sql);
        return VGPU_OK; /* No changes to make */
    }
    
    sql = sqlite3_mprintf("%s WHERE vm_uuid = ?;", sql, vm_uuid);
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free((void *)sql);
    
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    sqlite3_bind_text(stmt, 1, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        return VGPU_DB_ERROR;
    }
    
    if (sqlite3_changes(db) == 0) {
        return VGPU_NOT_FOUND;
    }
    
    return VGPU_OK;
}

int vgpu_remove_vm(sqlite3 *db, const char *vm_uuid) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) {
        return VGPU_INVALID_PARAM;
    }
    
    sql = "DELETE FROM vms WHERE vm_uuid = ?;";
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    sqlite3_bind_text(stmt, 1, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        return VGPU_DB_ERROR;
    }
    
    if (sqlite3_changes(db) == 0) {
        return VGPU_NOT_FOUND;
    }
    
    return VGPU_OK;
}

int vgpu_list_vms(sqlite3 *db, char pool_id, int priority,
                  vgpu_vm_config_t *configs, int *count, int max_count) {
    sqlite3_stmt *stmt;
    const char *sql;
    int rc;
    int i = 0;
    
    if (db == NULL || configs == NULL || count == NULL || max_count <= 0) {
        return VGPU_INVALID_PARAM;
    }
    
    /* Build query with optional filters */
    sql = "SELECT vm_id, vm_uuid, vm_name, pool_id, priority,"
          "       weight, max_jobs_per_sec, max_queue_depth, quarantined, error_count,"
          "       created_at, updated_at "
          "FROM vms WHERE 1=1";
    
    if (pool_id == 'A' || pool_id == 'B') {
        sql = sqlite3_mprintf("%s AND pool_id = '%c'", sql, pool_id);
    }
    
    if (priority >= 0 && priority <= 2) {
        sql = sqlite3_mprintf("%s AND priority = %d", sql, priority);
    }
    
    sql = sqlite3_mprintf("%s ORDER BY pool_id, priority DESC, vm_id;", sql);
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_free((void *)sql);
    
    if (rc != SQLITE_OK) {
        return VGPU_DB_ERROR;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW && i < max_count) {
        configs[i].vm_id = sqlite3_column_int(stmt, 0);
        
        const char *uuid = (const char *)sqlite3_column_text(stmt, 1);
        if (uuid) {
            strncpy(configs[i].vm_uuid, uuid, sizeof(configs[i].vm_uuid) - 1);
            configs[i].vm_uuid[sizeof(configs[i].vm_uuid) - 1] = '\0';
        }
        
        const char *name = (const char *)sqlite3_column_text(stmt, 2);
        if (name) {
            strncpy(configs[i].vm_name, name, sizeof(configs[i].vm_name) - 1);
            configs[i].vm_name[sizeof(configs[i].vm_name) - 1] = '\0';
        } else {
            configs[i].vm_name[0] = '\0';
        }
        
        const char *pool = (const char *)sqlite3_column_text(stmt, 3);
        if (pool) {
            configs[i].pool_id = pool[0];
        }
        
        configs[i].priority = sqlite3_column_int(stmt, 4);
        
        /* Phase 3 fields */
        configs[i].weight          = sqlite3_column_int(stmt, 5);
        configs[i].max_jobs_per_sec = sqlite3_column_int(stmt, 6);
        configs[i].max_queue_depth = sqlite3_column_int(stmt, 7);
        configs[i].quarantined     = sqlite3_column_int(stmt, 8);
        configs[i].error_count     = sqlite3_column_int(stmt, 9);
        
        const char *created = (const char *)sqlite3_column_text(stmt, 10);
        if (created) {
            strncpy(configs[i].created_at, created, sizeof(configs[i].created_at) - 1);
            configs[i].created_at[sizeof(configs[i].created_at) - 1] = '\0';
        }
        
        const char *updated = (const char *)sqlite3_column_text(stmt, 11);
        if (updated) {
            strncpy(configs[i].updated_at, updated, sizeof(configs[i].updated_at) - 1);
            configs[i].updated_at[sizeof(configs[i].updated_at) - 1] = '\0';
        }
        
        i++;
    }
    
    sqlite3_finalize(stmt);
    *count = i;
    return VGPU_OK;
}

/* ============================================================================
 * Phase 3: Scheduler Weight & Isolation Controls
 * ============================================================================ */

int vgpu_set_vm_weight(sqlite3 *db, const char *vm_uuid, int weight) {
    sqlite3_stmt *stmt;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) return VGPU_INVALID_PARAM;
    if (weight < 1 || weight > 100) return VGPU_INVALID_PARAM;
    
    rc = sqlite3_prepare_v2(db,
        "UPDATE vms SET weight = ?, updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;",
        -1, &stmt, NULL);
    if (rc != SQLITE_OK) return VGPU_DB_ERROR;
    
    sqlite3_bind_int(stmt, 1, weight);
    sqlite3_bind_text(stmt, 2, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return VGPU_DB_ERROR;
    return (sqlite3_changes(db) == 0) ? VGPU_NOT_FOUND : VGPU_OK;
}

int vgpu_set_vm_rate_limit(sqlite3 *db, const char *vm_uuid,
                           int max_jobs_per_sec, int max_queue_depth) {
    sqlite3_stmt *stmt;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) return VGPU_INVALID_PARAM;
    if (max_jobs_per_sec < 0 || max_queue_depth < 0) return VGPU_INVALID_PARAM;
    
    rc = sqlite3_prepare_v2(db,
        "UPDATE vms SET max_jobs_per_sec = ?, max_queue_depth = ?,"
        " updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;",
        -1, &stmt, NULL);
    if (rc != SQLITE_OK) return VGPU_DB_ERROR;
    
    sqlite3_bind_int(stmt, 1, max_jobs_per_sec);
    sqlite3_bind_int(stmt, 2, max_queue_depth);
    sqlite3_bind_text(stmt, 3, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return VGPU_DB_ERROR;
    return (sqlite3_changes(db) == 0) ? VGPU_NOT_FOUND : VGPU_OK;
}

int vgpu_quarantine_vm(sqlite3 *db, const char *vm_uuid) {
    sqlite3_stmt *stmt;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) return VGPU_INVALID_PARAM;
    
    rc = sqlite3_prepare_v2(db,
        "UPDATE vms SET quarantined = 1, updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;",
        -1, &stmt, NULL);
    if (rc != SQLITE_OK) return VGPU_DB_ERROR;
    
    sqlite3_bind_text(stmt, 1, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return VGPU_DB_ERROR;
    return (sqlite3_changes(db) == 0) ? VGPU_NOT_FOUND : VGPU_OK;
}

int vgpu_unquarantine_vm(sqlite3 *db, const char *vm_uuid) {
    sqlite3_stmt *stmt;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) return VGPU_INVALID_PARAM;
    
    rc = sqlite3_prepare_v2(db,
        "UPDATE vms SET quarantined = 0, error_count = 0,"
        " updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;",
        -1, &stmt, NULL);
    if (rc != SQLITE_OK) return VGPU_DB_ERROR;
    
    sqlite3_bind_text(stmt, 1, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return VGPU_DB_ERROR;
    return (sqlite3_changes(db) == 0) ? VGPU_NOT_FOUND : VGPU_OK;
}

int vgpu_increment_error_count(sqlite3 *db, const char *vm_uuid) {
    sqlite3_stmt *stmt;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) return VGPU_INVALID_PARAM;
    
    rc = sqlite3_prepare_v2(db,
        "UPDATE vms SET error_count = error_count + 1,"
        " updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;",
        -1, &stmt, NULL);
    if (rc != SQLITE_OK) return VGPU_DB_ERROR;
    
    sqlite3_bind_text(stmt, 1, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return VGPU_DB_ERROR;
    return (sqlite3_changes(db) == 0) ? VGPU_NOT_FOUND : VGPU_OK;
}

int vgpu_reset_error_count(sqlite3 *db, const char *vm_uuid) {
    sqlite3_stmt *stmt;
    int rc;
    
    if (db == NULL || vm_uuid == NULL) return VGPU_INVALID_PARAM;
    
    rc = sqlite3_prepare_v2(db,
        "UPDATE vms SET error_count = 0, updated_at = CURRENT_TIMESTAMP WHERE vm_uuid = ?;",
        -1, &stmt, NULL);
    if (rc != SQLITE_OK) return VGPU_DB_ERROR;
    
    sqlite3_bind_text(stmt, 1, vm_uuid, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) return VGPU_DB_ERROR;
    return (sqlite3_changes(db) == 0) ? VGPU_NOT_FOUND : VGPU_OK;
}

int vgpu_get_vm_config_by_id(sqlite3 *db, int vm_id, vgpu_vm_config_t *config) {
    sqlite3_stmt *stmt;
    int rc;
    
    if (db == NULL || config == NULL) return VGPU_INVALID_PARAM;
    
    rc = sqlite3_prepare_v2(db,
        "SELECT vm_id, vm_uuid, vm_name, pool_id, priority,"
        "       weight, max_jobs_per_sec, max_queue_depth, quarantined, error_count,"
        "       created_at, updated_at "
        "FROM vms WHERE vm_id = ?;",
        -1, &stmt, NULL);
    if (rc != SQLITE_OK) return VGPU_DB_ERROR;
    
    sqlite3_bind_int(stmt, 1, vm_id);
    
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        config->vm_id = sqlite3_column_int(stmt, 0);
        const char *uuid = (const char *)sqlite3_column_text(stmt, 1);
        if (uuid) { strncpy(config->vm_uuid, uuid, 63); config->vm_uuid[63] = '\0'; }
        const char *name = (const char *)sqlite3_column_text(stmt, 2);
        if (name) { strncpy(config->vm_name, name, 255); config->vm_name[255] = '\0'; }
        else { config->vm_name[0] = '\0'; }
        const char *pool = (const char *)sqlite3_column_text(stmt, 3);
        if (pool) config->pool_id = pool[0];
        config->priority         = sqlite3_column_int(stmt, 4);
        config->weight           = sqlite3_column_int(stmt, 5);
        config->max_jobs_per_sec = sqlite3_column_int(stmt, 6);
        config->max_queue_depth  = sqlite3_column_int(stmt, 7);
        config->quarantined      = sqlite3_column_int(stmt, 8);
        config->error_count      = sqlite3_column_int(stmt, 9);
        const char *created = (const char *)sqlite3_column_text(stmt, 10);
        if (created) { strncpy(config->created_at, created, 31); config->created_at[31] = '\0'; }
        const char *updated = (const char *)sqlite3_column_text(stmt, 11);
        if (updated) { strncpy(config->updated_at, updated, 31); config->updated_at[31] = '\0'; }
        sqlite3_finalize(stmt);
        return VGPU_OK;
    }
    
    sqlite3_finalize(stmt);
    return VGPU_NOT_FOUND;
}

-- ============================================================================
-- vGPU Configuration Database Schema
-- Configuration & Management Interface (Step 2-4)
-- ============================================================================
-- This script initializes the SQLite database for vGPU configuration.
-- Run with: sqlite3 /etc/vgpu/vgpu_config.db < init_db.sql
-- ============================================================================

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- ============================================================================
-- Table: pools
-- Stores pool definitions (Pool A and Pool B)
-- ============================================================================
CREATE TABLE IF NOT EXISTS pools (
    pool_id CHAR(1) PRIMARY KEY CHECK(pool_id IN ('A', 'B')),
    pool_name TEXT NOT NULL,
    description TEXT,
    enabled INTEGER DEFAULT 1 CHECK(enabled IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Table: vms
-- Stores VM configuration (pool assignment, priority, vm_id)
-- ============================================================================
CREATE TABLE IF NOT EXISTS vms (
    vm_id INTEGER NOT NULL UNIQUE,
    vm_uuid TEXT UNIQUE NOT NULL,
    vm_name TEXT,
    pool_id CHAR(1) NOT NULL DEFAULT 'A' CHECK(pool_id IN ('A', 'B')),
    priority INTEGER NOT NULL DEFAULT 1 CHECK(priority IN (0, 1, 2)),
    /* Phase 3: Scheduler weight and isolation controls */
    weight REAL NOT NULL DEFAULT 1.0,
    max_jobs_per_sec INTEGER NOT NULL DEFAULT 0,
    max_queue_depth INTEGER NOT NULL DEFAULT 0,
    quarantined INTEGER NOT NULL DEFAULT 0 CHECK(quarantined IN (0, 1)),
    error_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pool_id) REFERENCES pools(pool_id)
);

-- ============================================================================
-- Indexes for efficient queries
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_vm_uuid ON vms(vm_uuid);
CREATE INDEX IF NOT EXISTS idx_vm_pool ON vms(pool_id);
CREATE INDEX IF NOT EXISTS idx_vm_priority ON vms(priority);

-- ============================================================================
-- Initialize Pool A and Pool B
-- ============================================================================
INSERT OR IGNORE INTO pools (pool_id, pool_name, enabled) VALUES 
    ('A', 'Pool A', 1),
    ('B', 'Pool B', 1);

-- ============================================================================
-- Verification queries (for testing)
-- ============================================================================
-- SELECT * FROM pools;
-- SELECT * FROM vms;
-- SELECT COUNT(*) FROM pools;
-- SELECT COUNT(*) FROM vms;

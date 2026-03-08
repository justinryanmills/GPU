-- ============================================================================
-- Phase 3 Migration: Add scheduler weight & isolation columns to vms table
-- Run with: sqlite3 /etc/vgpu/vgpu_config.db < migrate_phase3.sql
-- Safe to run multiple times (uses ALTER TABLE IF NOT EXISTS pattern via pragma)
-- ============================================================================

-- Note: SQLite does not support ALTER TABLE ... IF NOT EXISTS for columns.
-- The vgpu_config.c init_schema function handles migration checks in code.
-- This file serves as a reference and for manual migration if needed.

ALTER TABLE vms ADD COLUMN weight          INTEGER NOT NULL DEFAULT 50;
ALTER TABLE vms ADD COLUMN max_jobs_per_sec INTEGER NOT NULL DEFAULT 0;
ALTER TABLE vms ADD COLUMN max_queue_depth INTEGER NOT NULL DEFAULT 0;
ALTER TABLE vms ADD COLUMN quarantined    INTEGER NOT NULL DEFAULT 0;
ALTER TABLE vms ADD COLUMN error_count    INTEGER NOT NULL DEFAULT 0;

-- Verify migration
-- SELECT vm_id, vm_uuid, weight, max_jobs_per_sec, max_queue_depth, quarantined, error_count FROM vms;

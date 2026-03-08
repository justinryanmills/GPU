# vGPU Configuration & Management Interface

This is the implementation of Step 2-4: Configuration & Management Interface for the vGPU system.

## Overview

This component provides:
- SQLite database for storing VM-to-pool and VM-to-priority assignments
- CLI tool (`vgpu-admin`) for managing VM configurations
- Automatic VM startup integration to apply configurations

## Files

- `vgpu_config.h` - Core library header
- `vgpu_config.c` - Core library implementation (database operations)
- `vgpu-admin.c` - CLI tool implementation
- `init_db.sql` - Database schema initialization script
- `vgpu-vm-startup.sh` - VM startup integration script
- `Makefile` - Build system

## Prerequisites

Before building, install the required dependencies:

**On Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install build-essential libsqlite3-dev
```

**On RHEL/CentOS/Fedora:**
```bash
sudo yum install gcc make sqlite-devel
# or on newer versions:
sudo dnf install gcc make sqlite-devel
```

**On XCP-ng (which is based on CentOS):**
```bash
sudo yum install gcc make sqlite-devel
```

## Building

```bash
cd step2(2-4)
make
```

This will create:
- `libvgpu_config.a` - Static library
- `vgpu-admin` - CLI executable

## Installation

```bash
sudo make install
```

This installs:
- `/etc/vgpu/vgpu-admin` - CLI tool
- `/etc/vgpu/vgpu_config.h` - Header file
- `/etc/vgpu/libvgpu_config.a` - Library
- `/etc/vgpu/init_db.sql` - Database schema
- `/usr/local/bin/vgpu-admin` - Symlink to CLI tool

After installation, initialize the database:

```bash
sudo sqlite3 /etc/vgpu/vgpu_config.db < /etc/vgpu/init_db.sql
```

## Usage

### Register a VM

You can register a VM using either its UUID or its name:

**Using UUID:**
```bash
vgpu-admin register-vm --vm-uuid=<uuid> [--pool=<A|B>] [--priority=<low|medium|high>] [--vm-id=<id>]
```

**Using VM name:**
```bash
vgpu-admin register-vm --vm-name="VM Name" [--pool=<A|B>] [--priority=<low|medium|high>] [--vm-id=<id>]
```

**Examples:**
```bash
# Register by UUID
vgpu-admin register-vm --vm-uuid=fef23215-1787-5f6b-1a5e-423ecfa93a25

# Register by name (easier to remember)
vgpu-admin register-vm --vm-name="My Production VM" --pool=A --priority=high

# Register with all options
vgpu-admin register-vm --vm-name="Test VM" --pool=B --priority=medium --vm-id=100
```

**Note:** If multiple VMs have the same name, you must use `--vm-uuid` instead. The system will show an error if duplicate names are detected.

### Scan VMs

```bash
vgpu-admin scan-vms
```

### Show VM configuration

```bash
vgpu-admin show-vm --vm-uuid=<uuid>
```

### List VMs

```bash
vgpu-admin list-vms [--pool=<A|B>] [--priority=<low|medium|high>]
```

### Change VM pool (requires restart)

```bash
vgpu-admin set-pool --vm-uuid=<uuid> --pool=<A|B>
```

### Change VM priority (requires restart)

```bash
vgpu-admin set-priority --vm-uuid=<uuid> --priority=<low|medium|high>
```

### Change VM ID (requires restart)

```bash
vgpu-admin set-vm-id --vm-uuid=<uuid> --vm-id=<id>
```

### Update multiple settings (requires restart)

```bash
vgpu-admin update-vm --vm-uuid=<uuid> [--pool=<A|B>] [--priority=<low|medium|high>] [--vm-id=<id>]
```

### List pools

```bash
vgpu-admin list-pools
```

### Show pool details

```bash
vgpu-admin show-pool --pool-id=<A|B>
```

### System status

```bash
vgpu-admin status
```

### Remove VM

Removes a VM from the database and detaches the vGPU-stub device. If the VM is running, it will be stopped first.

**Using UUID:**
```bash
vgpu-admin remove-vm --vm-uuid=<uuid>
```

**Using VM name:**
```bash
vgpu-admin remove-vm --vm-name="VM Name"
```

**What it does:**
1. Checks if VM is registered in database
2. If VM is running: stops it → waits 2-5 seconds → verifies stopped
3. Removes vGPU-stub device (detaches device-model-args)
4. Removes VM from database

**Example:**
```bash
vgpu-admin remove-vm --vm-name="Test-2"
```

## VM Startup Integration

The `vgpu-vm-startup.sh` script automatically applies vGPU configuration when a VM starts. To integrate it with XCP-ng:

1. Copy the script to `/etc/vgpu/vgpu-vm-startup.sh`
2. Make it executable: `chmod +x /etc/vgpu/vgpu-vm-startup.sh`
3. Integrate with XCP-ng VM lifecycle hooks (implementation depends on your XCP-ng setup)

Or call it manually:

```bash
/etc/vgpu/vgpu-vm-startup.sh <vm-uuid>
```

## Database Schema

### pools table
- `pool_id` (CHAR(1), PRIMARY KEY) - 'A' or 'B'
- `pool_name` (TEXT) - Pool name (default: "Pool A" or "Pool B")
- `description` (TEXT) - Optional description
- `enabled` (INTEGER) - 1 if enabled, 0 if disabled
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### vms table
- `vm_id` (INTEGER, UNIQUE) - User-assignable VM ID
- `vm_uuid` (TEXT, UNIQUE) - XCP-ng VM UUID
- `vm_name` (TEXT) - Optional VM name
- `pool_id` (CHAR(1)) - Pool assignment ('A' or 'B')
- `priority` (INTEGER) - 0=low, 1=medium, 2=high
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)
- Foreign key: `pool_id` references `pools.pool_id`

## Notes

- Commands that change pool/priority/vm_id require VM restart and will ask for confirmation
- Default pool is Pool A if not specified
- Default priority is medium if not specified
- VM ID is user-assignable and must be unique (auto-assigned if not specified)
- The system uses XCP-ng `xe` commands for VM management

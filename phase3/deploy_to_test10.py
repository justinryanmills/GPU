#!/usr/bin/env python3
"""
Deploy 100% Safe Method to test-10 VM
"""
import pexpect
import sys
import time

VM = "test-10@10.25.33.110"
PASSWORD = "Calvin@123"
LOCAL_DIR = "/home/david/Downloads/gpu/phase3"
REMOTE_DIR = "~/phase3"

def run_ssh_cmd(cmd, timeout=30):
    """Run SSH command and return output"""
    child = pexpect.spawn(f'ssh -o StrictHostKeyChecking=no {VM} "{cmd}"', 
                         encoding='utf-8', timeout=timeout)
    index = child.expect(['password:', 'yes/no'], timeout=5)
    if index == 1:
        child.sendline('yes')
        child.expect('password:', timeout=5)
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF, timeout=timeout)
    return child.before

def scp_file(local_file, remote_path):
    """Copy file via SCP"""
    print(f"Copying {local_file} to {VM}:{remote_path}...")
    child = pexpect.spawn(f'scp -o StrictHostKeyChecking=no {local_file} {VM}:{remote_path}',
                         encoding='utf-8', timeout=60)
    index = child.expect(['password:', 'yes/no'], timeout=10)
    if index == 1:
        child.sendline('yes')
        child.expect('password:', timeout=10)
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF, timeout=60)
    output = child.before
    # Check if transfer completed (look for 100% or successful transfer)
    if "100%" in output or child.exitstatus == 0 or "ETA" in output:
        print(f"Copied {local_file}")
        return True
    else:
        print(f"✗ Failed to copy {local_file}: {output}")
        return False

def main():
    print("=" * 70)
    print("Deploying 100% Safe Method to test-10")
    print("=" * 70)
    
    # Step 1: Create directories on VM
    print("\n[1/5] Creating directories on VM...")
    run_ssh_cmd(f"mkdir -p {REMOTE_DIR}/guest-shim {REMOTE_DIR}/include")
    print("Directories created")
    
    # Step 2: Copy files
    print("\n[2/5] Copying files to VM...")
    files_to_copy = [
        (f"{LOCAL_DIR}/guest-shim/force_load_shim.c", f"{REMOTE_DIR}/guest-shim/"),
        (f"{LOCAL_DIR}/guest-shim/ld_audit_interceptor.c", f"{REMOTE_DIR}/guest-shim/"),
        (f"{LOCAL_DIR}/DEPLOY_100_PERCENT_SAFE_METHOD.sh", f"{REMOTE_DIR}/"),
    ]
    
    for local_file, remote_dir in files_to_copy:
        if not scp_file(local_file, remote_dir):
            print(f"ERROR: Failed to copy {local_file}")
            return 1
    
    # Step 3: Verify files
    print("\n[3/5] Verifying files on VM...")
    result = run_ssh_cmd(f"ls -la {REMOTE_DIR}/guest-shim/ {REMOTE_DIR}/DEPLOY_100_PERCENT_SAFE_METHOD.sh")
    print(result)
    
    # Step 4: Check if shim libraries exist (they should be built first)
    print("\n[4/5] Checking for existing shim libraries...")
    result = run_ssh_cmd("ls -la /usr/lib64/libvgpu-*.so 2>&1")
    if "No such file" in result:
        print("⚠ Shim libraries not found. They need to be built first.")
        print("  The deployment script will check for them.")
    else:
        print("Found existing shim libraries")
        print(result)
    
    # Step 5: Run deployment script
    print("\n[5/5] Running deployment script on VM...")
    print("=" * 70)
    
    child = pexpect.spawn(f'ssh -o StrictHostKeyChecking=no {VM} "cd {REMOTE_DIR} && sudo bash DEPLOY_100_PERCENT_SAFE_METHOD.sh"',
                         encoding='utf-8', timeout=300)
    index = child.expect(['password:', 'yes/no'], timeout=10)
    if index == 1:
        child.sendline('yes')
        child.expect('password:', timeout=10)
    child.sendline(PASSWORD)
    
    # Handle sudo password prompt
    while True:
        try:
            index = child.expect(['password for', '\\$', '# ', pexpect.EOF, pexpect.TIMEOUT], timeout=5)
            if index == 0:
                child.sendline(PASSWORD)
            elif index == 1 or index == 2:
                # Command prompt - script is running
                break
            elif index == 3:
                # EOF - script finished
                break
            elif index == 4:
                # Timeout - continue
                pass
        except pexpect.EOF:
            break
    
    # Let script run and capture output
    child.expect(pexpect.EOF, timeout=300)
    output = child.before
    print(output)
    
    if child.exitstatus == 0:
        print("\n" + "=" * 70)
        print("Deployment completed successfully!")
        print("=" * 70)
        
        # Final verification
        print("\nFinal verification...")
        result = run_ssh_cmd("systemctl is-active ollama && journalctl -u ollama -n 50 --no-pager | grep -i 'library=' | tail -5")
        print(result)
        
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ Deployment failed (exit code: {})".format(child.exitstatus))
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())

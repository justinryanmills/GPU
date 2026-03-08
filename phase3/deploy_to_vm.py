#!/usr/bin/env python3
"""
Deploy code to VM using scp with pexpect for password authentication.
Uses vm_config.py (test-3@10.25.33.11). For full deploy + install use deploy_to_test3.py.
"""
import sys
import pexpect
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD

def deploy_directory(local_path, remote_path):
    """Deploy a directory to VM using scp"""
    scp_cmd = f"scp -r -o StrictHostKeyChecking=no {local_path} {VM_USER}@{VM_HOST}:{remote_path}"
    
    try:
        print(f"Deploying {local_path} to {VM_USER}@{VM_HOST}:{remote_path}...")
        child = pexpect.spawn(scp_cmd, timeout=300, encoding='utf-8')
        
        # Wait for password prompt
        index = child.expect(['password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        
        if index == 0 or index == 1:
            print("Sending password...")
            child.sendline(VM_PASSWORD)
            child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=300)
            output = child.before
            print("Deployment output:")
            print(output)
            child.close()
            return child.exitstatus == 0
        else:
            print(f"Unexpected response: {child.before}")
            child.close()
            return False
            
    except pexpect.ExceptionPexpect as e:
        print(f"Pexpect error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Deploy the entire phase3 directory to VM home (result: ~/phase3)
    local_phase3 = SCRIPT_DIR
    remote_home = f"/home/{VM_USER}"

    if not os.path.exists(local_phase3):
        print(f"Error: {local_phase3} does not exist")
        sys.exit(1)

    success = deploy_directory(local_phase3, remote_home)
    sys.exit(0 if success else 1)

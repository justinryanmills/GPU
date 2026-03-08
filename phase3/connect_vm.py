#!/usr/bin/env python3
"""
Connect to VM using pexpect for password authentication.
Target VM is read from vm_config.py (test-3@10.25.33.11 by default).
"""
import sys
import os
import pexpect

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_HOST, VM_USER, VM_PASSWORD

def connect_and_run_command(command):
    """Connect to VM and run a command"""
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no {VM_USER}@{VM_HOST}"
    
    try:
        print(f"Connecting to {VM_USER}@{VM_HOST}...")
        child = pexpect.spawn(ssh_cmd, timeout=30, encoding='utf-8')
        
        # Wait for password prompt
        index = child.expect(['password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
        
        if index == 0 or index == 1:
            print("Sending password...")
            child.sendline(VM_PASSWORD)
            child.expect([r'\$', '#', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
            
            if child.isalive():
                print("Connected successfully!")
                print(f"Running command: {command}")
                child.sendline(command)
                # Handle sudo password prompt if needed - loop until we get a prompt (long timeout for builds)
                while True:
                    try:
                        index = child.expect(['password:', 'Password:', r'\[sudo\] password', r'\$', '#', pexpect.EOF, pexpect.TIMEOUT], timeout=360)
                        if index in [0, 1, 2]:  # password prompt
                            child.sendline(VM_PASSWORD)
                        elif index in [3, 4]:  # command prompt
                            break
                        else:  # EOF or TIMEOUT
                            break
                    except pexpect.TIMEOUT:
                        break
                output = child.before
                print("Output:")
                print(output)
                child.sendline('exit')
                child.close()
                return output
            else:
                print("Connection failed after password")
                print(child.before)
                return None
        else:
            print(f"Unexpected response: {child.before}")
            return None
            
    except pexpect.ExceptionPexpect as e:
        print(f"Pexpect error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
    else:
        command = "echo 'Connection test'; pwd; whoami"
    
    result = connect_and_run_command(command)
    sys.exit(0 if result else 1)

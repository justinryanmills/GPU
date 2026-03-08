#!/usr/bin/env python3
"""
Run install.sh on VM with sudo password handling
"""
import sys
import os
import pexpect

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD

def run_command_with_sudo(command):
    """Run a command on VM that requires sudo"""
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no {VM_USER}@{VM_HOST}"
    
    try:
        print(f"Connecting to {VM_USER}@{VM_HOST}...")
        child = pexpect.spawn(ssh_cmd, timeout=300, encoding='utf-8')
        
        # Wait for password prompt
        index = child.expect(['password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
        
        if index == 0 or index == 1:
            print("Sending SSH password...")
            child.sendline(VM_PASSWORD)
            child.expect([r'\$', '#', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
            
            if child.isalive():
                print(f"Running command: {command}")
                child.sendline(command)
                
                # Handle sudo password prompt
                while True:
                    index = child.expect(['password:', 'Password:', r'\$', '#', pexpect.EOF, pexpect.TIMEOUT], timeout=300)
                    if index == 0 or index == 1:
                        print("Sending sudo password...")
                        child.sendline(VM_PASSWORD)
                    elif index == 2 or index == 3:
                        # Command completed
                        break
                    elif index == 4:
                        # EOF
                        break
                    elif index == 5:
                        print("Timeout waiting for command")
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
        command = "cd ~/phase3/guest-shim && sudo bash install.sh"
    
    result = run_command_with_sudo(command)
    sys.exit(0 if result else 1)

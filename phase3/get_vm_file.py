#!/usr/bin/env python3
"""
Get file content from VM
"""
import pexpect
import sys

VM_HOST = "10.25.33.110"
VM_USER = "test-10"
VM_PASSWORD = "Calvin@123"

def get_file_content(filepath):
    """Get file content from VM"""
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no {VM_USER}@{VM_HOST}"
    
    try:
        child = pexpect.spawn(ssh_cmd, timeout=30, encoding='utf-8')
        index = child.expect(['password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
        
        if index == 0 or index == 1:
            child.sendline(VM_PASSWORD)
            child.expect([r'\$', '#', pexpect.EOF, pexpect.TIMEOUT], timeout=10)
            
            if child.isalive():
                # Use cat to read file
                child.sendline(f"cat {filepath}")
                child.expect([r'\$', '#', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
                content = child.before
                child.sendline('exit')
                child.close()
                # Extract content between the command and prompt
                lines = content.split('\n')
                # Find where cat output starts (after the command echo)
                start_idx = 0
                for i, line in enumerate(lines):
                    if 'cat' in line and filepath in line:
                        start_idx = i + 1
                        break
                # Find where prompt starts (line with $ or #)
                end_idx = len(lines)
                for i in range(start_idx, len(lines)):
                    if lines[i].strip().endswith('$') or lines[i].strip().endswith('#'):
                        end_idx = i
                        break
                return '\n'.join(lines[start_idx:end_idx])
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 get_vm_file.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    content = get_file_content(filepath)
    if content:
        print(content)
    else:
        print("Failed to get file content", file=sys.stderr)
        sys.exit(1)

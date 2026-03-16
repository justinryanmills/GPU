#!/usr/bin/env python3
"""Connect to VM via pexpect; target from vm_config.py."""
import sys
import os
import pexpect
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_HOST, VM_USER, VM_PASSWORD

CONNECT_TIMEOUT_SEC = int(os.environ.get("CONNECT_VM_CONNECT_TIMEOUT_SEC", "30"))
PROMPT_TIMEOUT_SEC = int(os.environ.get("CONNECT_VM_PROMPT_TIMEOUT_SEC", "25"))
COMMAND_TIMEOUT_SEC = int(os.environ.get("CONNECT_VM_COMMAND_TIMEOUT_SEC", "2700"))
CONNECT_RETRIES = int(os.environ.get("CONNECT_VM_RETRIES", "3"))
CONNECT_RETRY_DELAY_SEC = float(os.environ.get("CONNECT_VM_RETRY_DELAY_SEC", "2"))

def connect_and_run_command(command):
    """Connect to VM and run a command."""
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 {VM_USER}@{VM_HOST}"
    run_cmd = f"/bin/sh -c '{ssh_cmd} 2>&1'"
    
    for attempt in range(1, CONNECT_RETRIES + 1):
        child = None
        try:
            print(f"Connecting to {VM_USER}@{VM_HOST}... (attempt {attempt}/{CONNECT_RETRIES})")
            child = pexpect.spawn(
                run_cmd,
                timeout=CONNECT_TIMEOUT_SEC,
                encoding='utf-8',
            )

            def run_remote_command():
                print("Connected successfully!")
                print(f"Running command: {command}")
                done_marker = "__CONNECT_VM_DONE__"
                wrapped_command = (
                    f"( {command} ); "
                    f"__rc=$?; "
                    f"printf '\\n{done_marker}:%s\\n' \"$__rc\""
                )
                child.sendline(wrapped_command)
                deadline = None if COMMAND_TIMEOUT_SEC <= 0 else (time.time() + COMMAND_TIMEOUT_SEC)
                exit_code = None
                while True:
                    if deadline is not None and time.time() >= deadline:
                        print(f"Command timeout reached ({COMMAND_TIMEOUT_SEC}s).")
                        break
                    try:
                        wait_timeout = 30
                        if deadline is not None:
                            wait_timeout = min(30, max(1, int(deadline - time.time())))
                        index = child.expect([
                            'password:',
                            'Password:',
                            r'\[sudo\] password',
                            rf'{done_marker}:(\d+)',
                            pexpect.EOF,
                            pexpect.TIMEOUT
                        ], timeout=wait_timeout)
                        if index in [0, 1, 2]:  # password prompt
                            child.sendline(VM_PASSWORD)
                        elif index == 3:  # done marker with exit code
                            try:
                                exit_code = int(child.match.group(1))
                            except Exception:
                                exit_code = None
                            break
                        elif index == 4:  # EOF
                            break
                        else:  # TIMEOUT
                            continue
                    except pexpect.TIMEOUT:
                        continue
                output = child.before
                print("Output:")
                print(output)
                if exit_code is not None:
                    print(f"Remote command exit code: {exit_code}")
                child.sendline('exit')
                child.close()
                return output
        
            index = child.expect(
                ['password:', 'Password:', r'\$', '#', pexpect.EOF, pexpect.TIMEOUT],
                timeout=PROMPT_TIMEOUT_SEC
            )
            
            if index == 0 or index == 1:
                print("Sending password...")
                child.sendline(VM_PASSWORD)
                child.expect([r'\$', '#', pexpect.EOF, pexpect.TIMEOUT], timeout=PROMPT_TIMEOUT_SEC)
                
                if child.isalive():
                    return run_remote_command()
                else:
                    print("Connection failed after password")
                    print(child.before)
            elif index == 2 or index == 3:
                return run_remote_command()
            else:
                buf = (child.before or "").strip()
                if buf:
                    print(f"SSH output/error: {buf}")
                else:
                    print("Unexpected response (no output from SSH).")
                    if index == 4:
                        print("Connection closed immediately (EOF). Check: VM reachable? (e.g. ping", VM_HOST, ")")
                    else:
                        print("No prompt within timeout. Check: VM up, SSH on port 22, firewall.")
        except pexpect.ExceptionPexpect as e:
            print(f"Pexpect error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            try:
                if child is not None and child.isalive():
                    child.close(force=True)
            except Exception:
                pass

        if attempt < CONNECT_RETRIES:
            time.sleep(CONNECT_RETRY_DELAY_SEC)

    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
    else:
        command = "echo 'Connection test'; pwd; whoami"
    
    result = connect_and_run_command(command)
    sys.exit(0 if result else 1)

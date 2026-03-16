#!/usr/bin/env python3
"""Connect to mediator host via pexpect; config in vm_config.py."""
import sys
import os
import pexpect
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import MEDIATOR_HOST, MEDIATOR_USER, MEDIATOR_PASSWORD

CONNECT_TIMEOUT_SEC = 15
PROMPT_TIMEOUT_SEC = 10
COMMAND_TIMEOUT_SEC = 60


def connect_and_run(command):
    """Run command on mediator host."""
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no {MEDIATOR_USER}@{MEDIATOR_HOST}"
    done_marker = "__CONNECT_HOST_DONE__"

    child = pexpect.spawn(ssh_cmd, timeout=CONNECT_TIMEOUT_SEC, encoding='utf-8')
    wrapped = f"( {command} ); __rc=$?; printf '\\n{done_marker}:%s\\n' \"$__rc\""

    try:
        index = child.expect(['password:', 'Password:', r'\$', '#', pexpect.EOF, pexpect.TIMEOUT],
                            timeout=PROMPT_TIMEOUT_SEC)
        if index in [0, 1]:
            child.sendline(MEDIATOR_PASSWORD)
            child.expect([r'\$', '#', pexpect.EOF, pexpect.TIMEOUT], timeout=PROMPT_TIMEOUT_SEC)

        if child.isalive():
            child.sendline(wrapped)
            deadline = time.time() + COMMAND_TIMEOUT_SEC
            while time.time() < deadline:
                try:
                    idx = child.expect([rf'{done_marker}:(\d+)', 'password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT],
                                       timeout=10)
                    if idx == 0:
                        exit_code = int(child.match.group(1))
                        output = child.before
                        child.sendline('exit')
                        child.close()
                        return output, exit_code
                    elif idx in [1, 2]:
                        child.sendline(MEDIATOR_PASSWORD)
                except pexpect.TIMEOUT:
                    continue
        return child.before or "", -1
    finally:
        try:
            if child.isalive():
                child.close(force=True)
        except Exception:
            pass
    return "", -1


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo Connected; hostname"
    out, rc = connect_and_run(cmd)
    print(out)
    sys.exit(0 if rc == 0 else 1)

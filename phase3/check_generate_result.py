#!/usr/bin/env python3
"""Check the result of the long-timeout generate started on the VM.
Run after waiting 15–20 min for the model load to complete.
Usage: python3 check_generate_result.py
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_HOST, VM_USER, VM_PASSWORD

def run_vm(cmd, timeout=30):
    import pexpect
    ssh = f"ssh -o StrictHostKeyChecking=no {VM_USER}@{VM_HOST}"
    c = pexpect.spawn(ssh, timeout=timeout, encoding="utf-8")
    c.expect(["password:", "Password:"], timeout=10)
    c.sendline(VM_PASSWORD)
    c.expect([r"\$", "#"], timeout=10)
    c.sendline(cmd)
    c.expect([r"\$", "#"], timeout=15)
    return c.before

def main():
    print("Checking /tmp/generate_result.json and /tmp/generate_curl.log on VM...")
    out = run_vm(
        "echo '=== curl log ==='; cat /tmp/generate_curl.log 2>/dev/null; "
        "echo; echo '=== result file ==='; ls -la /tmp/generate_result.json 2>/dev/null; "
        "echo '=== first 400 chars ==='; head -c 400 /tmp/generate_result.json 2>/dev/null; echo"
    )
    print(out)
    if "response" in out or '"model"' in out:
        print("\nInference completed successfully (JSON response present).")
    elif "error" in out.lower():
        print("\nResponse contains an error.")
    else:
        print("\nStill loading or no result yet; wait longer and run again.")

if __name__ == "__main__":
    main()

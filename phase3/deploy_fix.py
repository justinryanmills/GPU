#!/usr/bin/env python3
"""Deploy cuInit fix to VM and verify"""

import pexpect
import time
import sys

def main():
    print("Connecting to VM...")
    child = pexpect.spawn('ssh -o StrictHostKeyChecking=no test-3@10.25.33.11', 
                         encoding='utf-8', timeout=300)
    
    try:
        # Login
        child.expect(['password:', pexpect.EOF], timeout=10)
        child.sendline('Calvin@123')
        time.sleep(2)
        child.expect(['$', '#', 'test-3@'], timeout=10)
        
        print("Connected")
        
        # Step 1: Rebuild shim
        print("\n[1/5] Rebuilding CUDA shim...")
        child.sendline('cd ~/phase3/guest-shim && sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1')
        child.expect(['$', '#', 'test-3@'], timeout=90)
        build_output = child.before
        if 'error' in build_output.lower():
            print("  Build errors detected:")
            for line in build_output.split('\n'):
                if 'error' in line.lower():
                    print(f"    {line.strip()}")
        else:
            print("  Build completed")
        
        # Step 2: Restart Ollama
        print("\n[2/5] Restarting Ollama...")
        child.sendline('sudo systemctl restart ollama')
        child.expect(['$', '#', 'test-3@'], timeout=15)
        child.sendline('sleep 8')
        child.expect(['$', '#', 'test-3@'], timeout=10)
        print("  Ollama restarted")
        
        # Step 3: Check logs - get raw output first
        print("\n[3/5] Checking logs for library mode...")
        child.sendline('sudo journalctl -u ollama -n 100 --no-pager 2>&1')
        child.expect(['$', '#', 'test-3@'], timeout=30)
        raw_logs = child.before
        print("  Searching for library mode entries...")
        library_lines = [line for line in raw_logs.split('\n') if 'library=' in line.lower()]
        if library_lines:
            print("  Found library mode entries:")
            for line in library_lines[-10:]:
                if 'password' not in line.lower():
                    print(f"    {line.strip()}")
        else:
            print("  No library mode entries found in recent logs")
            # Show last few lines for debugging
            print("  Last 5 log lines:")
            for line in raw_logs.split('\n')[-5:]:
                if line.strip():
                    print(f"    {line.strip()}")
        
        # Step 4: Run test
        print("\n[4/5] Running test inference...")
        child.sendline('timeout 20 ollama run llama3.2:1b "test" 2>&1')
        child.expect(['$', '#', 'test-3@'], timeout=25)
        test_output = child.before
        print("  Test completed")
        if 'error' in test_output.lower():
            print("  Errors detected in test output")
        
        # Step 5: Final check - get more comprehensive output
        print("\n[5/5] Final library mode check...")
        child.sendline('sudo journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1')
        child.expect(['$', '#', 'test-3@'], timeout=30)
        final_output = child.before
        
        print("\n" + "="*70)
        print("FINAL RESULTS:")
        print("="*70)
        
        # Search for library mode in final output
        found_cuda = False
        found_cpu = False
        
        for line in final_output.split('\n'):
            if 'library=' in line.lower() and 'password' not in line.lower():
                print(f"  {line.strip()}")
                if 'library=cuda' in line.lower() or 'library=cuda_v' in line.lower():
                    found_cuda = True
                elif 'library=cpu' in line.lower():
                    found_cpu = True
        
        if not found_cuda and not found_cpu:
            # Show last 20 lines for debugging
            print("\n  No library mode found. Last 20 log lines:")
            for line in final_output.split('\n')[-20:]:
                if line.strip() and 'password' not in line.lower():
                    print(f"    {line.strip()}")
        
        if found_cuda:
            print("\n" + "="*70)
            print("SUCCESS! Ollama is using GPU mode (library=cuda)")
            print("="*70)
            sys.exit(0)
        elif found_cpu:
            print("\n" + "="*70)
            print("✗ Still using CPU mode - need further investigation")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("? Could not find clear library mode indication")
            print("  This might mean:")
            print("  1. Ollama hasn't logged library mode yet")
            print("  2. Logs are in a different format")
            print("  3. Need to check Ollama status differently")
            print("="*70)
        
        child.sendline('exit')
        child.close()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        try:
            child.close()
        except:
            pass
        sys.exit(1)

if __name__ == '__main__':
    main()

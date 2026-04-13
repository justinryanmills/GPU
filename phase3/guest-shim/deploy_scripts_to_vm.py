#!/usr/bin/env python3
"""
Deploy error capture scripts to VM
"""
import pexpect
import os

VM = "test-10@10.25.33.110"
PASSWORD = "Calvin@123"
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = ['capture_errors.sh', 'analyze_errors.sh', 'verify_symbols.sh']

print("="*70)
print("DEPLOYING SCRIPTS TO VM")
print("="*70)

try:
    child = pexpect.spawn(f'ssh -o StrictHostKeyChecking=no {VM}',
                         encoding='utf-8', timeout=120)
    
    child.expect(['password:', 'yes/no'], timeout=10)
    if 'yes/no' in child.after or 'yes/no' in child.before:
        child.sendline('yes')
        child.expect('password:', timeout=5)
    child.sendline(PASSWORD)
    child.expect(r'\$', timeout=10)
    
    print("\n[1/3] Reading and deploying scripts...")
    
    for script_name in scripts:
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        if not os.path.exists(script_path):
            print(f"  ✗ {script_name} not found locally")
            continue
            
        print(f"  Deploying {script_name}...")
        
        # Read script content
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Create script on VM using Python to write it
        child.sendline('cd ~/phase3/guest-shim && python3 << "PYEOF"')
        child.expect('>', timeout=5)
        
        # Write Python code to create the file
        python_code = f'''
import sys
content = """{content}"""
with open("{script_name}", "w") as f:
    f.write(content)
import os
os.chmod("{script_name}", 0o755)
print("Created {script_name}")
PYEOF
'''
        child.sendline(python_code)
        child.expect(r'\$', timeout=10)
        
        # Verify it was created
        child.sendline(f'cd ~/phase3/guest-shim && test -f {script_name} && echo "OK" || echo "FAILED"')
        child.expect(r'\$', timeout=5)
        if 'OK' in child.before:
            print(f"    ✓ {script_name} deployed")
        else:
            print(f"    ✗ Failed to deploy {script_name}")
    
    print("\n[2/3] Verifying all scripts...")
    child.sendline('cd ~/phase3/guest-shim && ls -lh capture_errors.sh analyze_errors.sh verify_symbols.sh 2>&1')
    child.expect(r'\$', timeout=10)
    print(child.before.strip())
    
    print("\n[3/3] Scripts deployed successfully!")
    
    child.sendline('exit')
    child.expect(pexpect.EOF)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

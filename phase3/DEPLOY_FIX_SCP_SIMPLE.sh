#!/bin/bash

VM="test-5@10.25.33.15"
PASSWORD="Calvin@123"
LOCAL_FILE="/home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c"
REMOTE_PATH="~/phase3/guest-shim/libvgpu_cuda.c"

echo "======================================================================"
echo "DEPLOYING FIX VIA SCP (Simple Method)"
echo "======================================================================"

echo ""
echo "[1] Copying file to VM via SCP..."
expect << EOF
set timeout 60
spawn scp -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$LOCAL_FILE" ${VM}:${REMOTE_PATH}
expect {
    "password:" {
        send "$PASSWORD\r"
        exp_continue
    }
    "yes/no" {
        send "yes\r"
        exp_continue
    }
    eof
}
set exit_code [wait]
if {[lindex \$exit_code 3] != 0} {
    exit 1
}
EOF

if [ $? -eq 0 ]; then
    echo "  File copied successfully"
else
    echo "  ✗ File copy failed"
    exit 1
fi

echo ""
echo "[2] Verifying file and rebuilding on VM..."
expect << 'ENDEXPECT'
set timeout 300
spawn ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 test-5@10.25.33.15
expect {
    "password:" {
        send "Calvin@123\r"
    }
    "yes/no" {
        send "yes\r"
        exp_continue
    }
}

expect {
    "$ " {}
    "# " {}
    "test-5@" {}
    timeout { exit 1 }
}

send "test -f ~/phase3/guest-shim/libvgpu_cuda.c && ls -lh ~/phase3/guest-shim/libvgpu_cuda.c && echo 'FILE_OK'\r"
expect {
    "FILE_OK" {
        puts "File verified"
    }
    timeout { exit 1 }
}

send "cd ~/phase3/guest-shim && echo 'Calvin@123' | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1\r"
expect {
    "$ " {}
    "# " {}
    timeout { exit 1 }
}

send "test -f /usr/lib64/libvgpu-cuda.so && ls -lh /usr/lib64/libvgpu-cuda.so && echo 'BUILD_OK'\r"
expect {
    "BUILD_OK" {
        puts "Build successful"
    }
    timeout { exit 1 }
}

send "ls 2>&1 | head -2 > /dev/null; sleep 2; echo 'Calvin@123' | sudo -S journalctl --since '1 minute ago' --no-pager 2>&1 | grep -E 'Found VGPU|fgets.*returning.*device=0x2331' | tail -3\r"
expect {
    "$ " {}
    "# " {}
    timeout {}
}

send "echo 'Calvin@123' | sudo -S systemctl restart ollama && sleep 25 && systemctl is-active ollama && echo 'OLLAMA_OK'\r"
expect {
    "OLLAMA_OK" {
        puts "Ollama restarted"
    }
    timeout {}
}

send "timeout 50 ollama run llama3.2:1b 'test' 2>&1 | head -3; sleep 2; echo 'Calvin@123' | sudo -S journalctl -u ollama --since '2 minutes ago' --no-pager 2>&1 | grep 'library=' | tail -3\r"
expect {
    "$ " {}
    "# " {}
    timeout {}
}

send "exit\r"
expect eof
ENDEXPECT

echo ""
echo "======================================================================"
echo "DEPLOYMENT COMPLETE"
echo "======================================================================"

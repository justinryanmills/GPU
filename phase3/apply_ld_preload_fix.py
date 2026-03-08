#!/usr/bin/env python3
"""Insert LD_PRELOAD removal block into llm/server.go. Run on VM in ollama repo root."""
import sys

path = "llm/server.go"
with open(path) as f:
    lines = f.readlines()

# Find "	for _, e := range cmd.Env {" followed by LD_PRELOAD log (insert our block before it)
inserted = False
for i, line in enumerate(lines):
    if "for _, e := range cmd.Env" in line and i + 1 < len(lines):
        chunk = "".join(lines[i : i + 5])
        if "LD_PRELOAD" in chunk and "runner env LD_PRELOAD" in chunk:
            block = """	// Remove LD_PRELOAD from runner so it is not inherited from os.Environ() (vGPU guest: need real dlopen)
	n := 0
	for _, e := range cmd.Env {
		if !strings.HasPrefix(e, "LD_PRELOAD=") {
			cmd.Env[n] = e
			n++
		}
	}
	cmd.Env = cmd.Env[:n]

"""
            lines = lines[:i] + [block] + lines[i:]
            inserted = True
            break

if not inserted:
    print("Could not find insertion point", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.writelines(lines)
print("Inserted LD_PRELOAD removal block")

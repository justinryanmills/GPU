#!/usr/bin/env python3
"""Apply LD_PRELOAD/LD_LIBRARY_PATH patch to ollama llm/server.go. Run from repo root: python3 apply_ollama_patch.py"""
import sys

path = "llm/server.go"
with open(path) as f:
    lines = f.readlines()

# Find the block: for k, done := range extraEnvsDone ... } then blank then slog.Info("starting runner"
# Insert our block between the closing } and slog.Info.
insertion = '''
\t// Ensure runner inherits LD_PRELOAD and LD_LIBRARY_PATH from current process
\t// (e.g. for vGPU shims in guest VM; some builds filter env and drop these)
\tfor _, key := range []string{"LD_PRELOAD", "LD_LIBRARY_PATH"} {
\t\tif v, ok := os.LookupEnv(key); ok && v != "" {
\t\t\tfound := false
\t\t\tfor i := range cmd.Env {
\t\t\t\tif strings.HasPrefix(cmd.Env[i], key+"=") {
\t\t\t\t\tcmd.Env[i] = key + "=" + v
\t\t\t\t\tfound = true
\t\t\t\t\tbreak
\t\t\t\t}
\t\t\t}
\t\t\tif !found {
\t\t\t\tcmd.Env = append(cmd.Env, key+"="+v)
\t\t\t}
\t\t}
\t}

'''
target = '\tslog.Info("starting runner", "cmd", cmd)\n'
found = False
for i, line in enumerate(lines):
    if line == target and i > 5:
        # Check previous non-empty line is "}"
        j = i - 1
        while j >= 0 and lines[j].strip() == "":
            j -= 1
        if j >= 0 and lines[j].strip() == "}":
            lines.insert(i, insertion)
            found = True
            break
if not found:
    # Fallback: find last "for k, done := range extraEnvsDone" block and insert before slog.Info
    for i in range(len(lines) - 1, 0, -1):
        if 'slog.Info("starting runner"' in lines[i] and "cmd" in lines[i]:
            lines.insert(i, insertion)
            found = True
            break
if not found:
    print("Could not find insertion point")
    sys.exit(1)
with open(path, "w") as f:
    f.writelines(lines)
print("Patch applied OK")

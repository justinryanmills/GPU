#!/usr/bin/env python3
path = "llm/server.go"
with open(path) as f:
    lines = f.readlines()

# Insert right before the line containing: slog.Info("starting runner"
marker = 'slog.Info("starting runner"'
debug_block = """\tfor _, e := range cmd.Env {
\t\tif strings.HasPrefix(e, "LD_PRELOAD=") {
\t\t\tslog.Info("runner env LD_PRELOAD", "value", e)
\t\t\tbreak
\t\t}
\t}

\t"""

new_lines = []
inserted = False
for i, line in enumerate(lines):
    if not inserted and marker in line and "runner" in line:
        new_lines.append(debug_block)
        inserted = True
    new_lines.append(line)

if not inserted:
    print("Marker not found")
    exit(1)
with open(path, "w") as f:
    f.writelines(new_lines)
print("Debug log inserted")

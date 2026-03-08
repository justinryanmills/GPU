#!/usr/bin/env python3
"""Fix server.go: 1) Fix broken logging loops 2) Prepend /opt/vgpu/lib to LD_LIBRARY_PATH for runner."""
import sys

path = "llm/server.go"
with open(path) as f:
    content = f.read()

# 1) In the ensure block, when setting LD_LIBRARY_PATH, prepend /opt/vgpu/lib if not present
old_ensure = """	for _, key := range []string{"LD_LIBRARY_PATH", "OLLAMA_LIBRARY_PATH", "OLLAMA_LLM_LIBRARY"} {
		if v, ok := os.LookupEnv(key); ok && v != "" {
			found := false
			for i := range cmd.Env {
				if strings.HasPrefix(cmd.Env[i], key+"=") {
					cmd.Env[i] = key + "=" + v
					found = true
					break
				}
			}
			if !found {
				cmd.Env = append(cmd.Env, key+"="+v)
			}
		}
	}"""

new_ensure = """	for _, key := range []string{"LD_LIBRARY_PATH", "OLLAMA_LIBRARY_PATH", "OLLAMA_LLM_LIBRARY"} {
		if v, ok := os.LookupEnv(key); ok && v != "" {
			if key == "LD_LIBRARY_PATH" && !strings.Contains(v, "/opt/vgpu/lib") {
				v = "/opt/vgpu/lib:" + v
			}
			found := false
			for i := range cmd.Env {
				if strings.HasPrefix(cmd.Env[i], key+"=") {
					cmd.Env[i] = key + "=" + v
					found = true
					break
				}
			}
			if !found {
				cmd.Env = append(cmd.Env, key+"="+v)
			}
		}
	}"""

if old_ensure in content and new_ensure not in content:
    content = content.replace(old_ensure, new_ensure)
    print("Prepended /opt/vgpu/lib to LD_LIBRARY_PATH in ensure block")
else:
    print("Ensure block unchanged or already patched")

# 2) Fix broken logging: replace the messed-up loop section with a clean one
broken = '''	for _, e := range cmd.Env {
		if strings.HasPrefix(e, "LD_PRELOAD=") {
			slog.Info("runner env LD_PRELOAD", "value", e)
t	}
	}
	for _, e := range cmd.Env {
		if strings.HasPrefix(e, "LD_LIBRARY_PATH=") {
			slog.Info("runner env LD_LIBRARY_PATH", "value", e)
			break
		}
	}
			break
		}
	for _, e := range cmd.Env {
		if strings.HasPrefix(e, "OLLAMA_LIBRARY_PATH=") {
			slog.Info("runner env OLLAMA_LIBRARY_PATH", "value", e)
			break
		}
	}

	}

		slog.Info("starting runner", "cmd", cmd)'''

clean = '''	for _, e := range cmd.Env {
		if strings.HasPrefix(e, "LD_LIBRARY_PATH=") {
			slog.Info("runner env LD_LIBRARY_PATH", "value", e)
			break
		}
	}
	for _, e := range cmd.Env {
		if strings.HasPrefix(e, "OLLAMA_LIBRARY_PATH=") {
			slog.Info("runner env OLLAMA_LIBRARY_PATH", "value", e)
			break
		}
	}

	slog.Info("starting runner", "cmd", cmd)'''

if broken in content:
    content = content.replace(broken, clean)
    print("Fixed broken logging loops")
else:
    print("Logging section not found or already fixed")

with open(path, "w") as f:
    f.write(content)
print("Done.")

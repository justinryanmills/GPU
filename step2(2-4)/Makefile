# Makefile for vGPU Configuration & Management Interface
# Configuration & Management Interface (Step 2-4)

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lsqlite3
PREFIX = /usr/local
ETC_PREFIX = /etc/vgpu

# Check for SQLite development headers
SQLITE_CHECK := $(shell pkg-config --exists sqlite3 2>/dev/null && echo "yes" || echo "no")
ifeq ($(SQLITE_CHECK),no)
    # Try to find sqlite3.h in common locations
    SQLITE_INC := $(shell test -f /usr/include/sqlite3.h && echo "-I/usr/include" || \
                     test -f /usr/local/include/sqlite3.h && echo "-I/usr/local/include" || \
                     echo "")
    ifeq ($(SQLITE_INC),)
        $(warning WARNING: sqlite3.h not found. Please install libsqlite3-dev (Debian/Ubuntu) or sqlite-devel (RHEL/CentOS))
        $(error Install SQLite development package: sudo apt-get install libsqlite3-dev OR sudo yum install sqlite-devel)
    endif
    CFLAGS += $(SQLITE_INC)
endif

# Source files
LIB_SRC = vgpu_config.c
LIB_OBJ = vgpu_config.o
CLI_SRC = vgpu-admin.c
CLI_OBJ = vgpu-admin.o

# Targets
all: libvgpu_config.a vgpu-admin

libvgpu_config.a: $(LIB_OBJ)
	ar rcs $@ $^

vgpu-admin: $(CLI_OBJ) libvgpu_config.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(LIB_OBJ): $(LIB_SRC) vgpu_config.h
	$(CC) $(CFLAGS) -c $(LIB_SRC) -o $@

$(CLI_OBJ): $(CLI_SRC) vgpu_config.h
	$(CC) $(CFLAGS) -c $(CLI_SRC) -o $@

# Installation
install: all
	mkdir -p $(DESTDIR)$(ETC_PREFIX)
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	install -m 755 vgpu-admin $(DESTDIR)$(ETC_PREFIX)/vgpu-admin
	install -m 644 vgpu_config.h $(DESTDIR)$(ETC_PREFIX)/
	install -m 644 libvgpu_config.a $(DESTDIR)$(ETC_PREFIX)/
	install -m 644 init_db.sql $(DESTDIR)$(ETC_PREFIX)/
	ln -sf $(ETC_PREFIX)/vgpu-admin $(DESTDIR)$(PREFIX)/bin/vgpu-admin
	@echo "Installation complete. Run 'sqlite3 $(ETC_PREFIX)/vgpu_config.db < $(ETC_PREFIX)/init_db.sql' to initialize database."

# Clean
clean:
	rm -f $(LIB_OBJ) $(CLI_OBJ) libvgpu_config.a vgpu-admin

.PHONY: all install clean

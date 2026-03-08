#!/bin/bash

if [ ! -f Makefile ]; then
    echo "Error: Makefile not found"
    exit 1
fi

cp Makefile Makefile.backup

sed -i 's/^    /\t/' Makefile 2>/dev/null || sed -i 's/^    /\t/' Makefile

echo "Makefile fixed (backup saved as Makefile.backup)"
echo "Try: make dom0"

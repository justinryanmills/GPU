#!/bin/bash

SCRIPT="/home/david/Downloads/gpu/vm_create/create_vm.sh"

echo "Checking script: $SCRIPT"
echo ""

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script file not found"
    exit 1
fi

echo "1. Checking syntax..."
if bash -n "$SCRIPT" 2>&1; then
    echo "   Syntax OK"
else
    echo "   ✗ Syntax error found"
    exit 1
fi

LINE_COUNT=$(wc -l < "$SCRIPT")
echo "2. Line count: $LINE_COUNT"

echo "3. Checking for unclosed quotes..."
UNCLOSED=$(awk 'BEGIN{open=0; line=0} {line++; for(i=1;i<=length($0);i++){c=substr($0,i,1); if(c=="\"") open=!open}} END{if(open) print "Unclosed quote at line " line}' "$SCRIPT")
if [ -n "$UNCLOSED" ]; then
    echo "   ✗ $UNCLOSED"
    exit 1
else
    echo "   No unclosed quotes"
fi

echo "4. Checking file ending..."
LAST_CHAR=$(tail -c 1 "$SCRIPT" | od -c | awk 'NR==1 {print $2}')
if [ "$LAST_CHAR" = "\\n" ] || [ "$LAST_CHAR" = "\n" ]; then
    echo "   File ends with newline"
else
    echo "   ⚠ File may not end with newline"
fi

echo ""
echo "Script verification complete"

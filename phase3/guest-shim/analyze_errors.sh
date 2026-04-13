#!/bin/bash
#
# Error analysis script for Ollama GPU discovery errors
#
# This script analyzes captured error logs to:
# - Extract full error messages (not truncated)
# - Identify unique error messages
# - Categorize errors by type
# - Generate a report
#
# Usage: ./analyze_errors.sh [capture_directory]
#

set -e

CAPTURE_DIR=${1:-/tmp/ollama_error_capture_$(ls -td /tmp/ollama_error_capture_* 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "")}

if [ ! -d "$CAPTURE_DIR" ]; then
    echo "ERROR: Capture directory not found: $CAPTURE_DIR"
    echo "Usage: $0 [capture_directory]"
    exit 1
fi

echo "=========================================="
echo "Ollama Error Analysis"
echo "=========================================="
echo "Analyzing: $CAPTURE_DIR"
echo ""

ANALYSIS_DIR="$CAPTURE_DIR/analysis"
mkdir -p "$ANALYSIS_DIR"

# Extract unique error messages from filtered log
echo "[1/5] Extracting unique error messages..."
if [ -f "$CAPTURE_DIR/errors_filtered.log" ]; then
    # Extract just the message content (after timestamp and PID)
    grep -oE "SIZE=[0-9]+: .*" "$CAPTURE_DIR/errors_filtered.log" | \
        sed 's/^SIZE=[0-9]*: //' | \
        sort -u > "$ANALYSIS_DIR/unique_errors.txt"
    echo "  Found $(wc -l < "$ANALYSIS_DIR/unique_errors.txt" | tr -d ' ') unique error messages"
fi

# Extract full error messages (not truncated)
echo "[2/5] Extracting full error messages..."
if [ -f "$CAPTURE_DIR/errors_full.log" ]; then
    # Look for error patterns and extract full messages
    grep -iE "(error|failed|timeout|discover|init|cuda|ggml)" "$CAPTURE_DIR/errors_full.log" | \
        sed 's/^\[[0-9.]*\] PID=[0-9]* SIZE=[0-9]*: //' | \
        sort -u > "$ANALYSIS_DIR/full_error_messages.txt"
    echo "  Found $(wc -l < "$ANALYSIS_DIR/full_error_messages.txt" | tr -d ' ') full error messages"
fi

# Categorize errors
echo "[3/5] Categorizing errors..."
cat > "$ANALYSIS_DIR/error_categories.txt" <<EOF
Error Categories
================

EOF

# Count errors by category
if [ -f "$ANALYSIS_DIR/full_error_messages.txt" ]; then
    echo "Initialization Errors:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "(init|initialize|initialization)" "$ANALYSIS_DIR/full_error_messages.txt" | wc -l | xargs echo "  Count:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "(init|initialize|initialization)" "$ANALYSIS_DIR/full_error_messages.txt" | head -5 >> "$ANALYSIS_DIR/error_categories.txt" || true
    echo "" >> "$ANALYSIS_DIR/error_categories.txt"
    
    echo "CUDA Errors:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "cuda" "$ANALYSIS_DIR/full_error_messages.txt" | wc -l | xargs echo "  Count:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "cuda" "$ANALYSIS_DIR/full_error_messages.txt" | head -5 >> "$ANALYSIS_DIR/error_categories.txt" || true
    echo "" >> "$ANALYSIS_DIR/error_categories.txt"
    
    echo "Timeout Errors:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "timeout" "$ANALYSIS_DIR/full_error_messages.txt" | wc -l | xargs echo "  Count:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "timeout" "$ANALYSIS_DIR/full_error_messages.txt" | head -5 >> "$ANALYSIS_DIR/error_categories.txt" || true
    echo "" >> "$ANALYSIS_DIR/error_categories.txt"
    
    echo "Discovery Errors:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "discover" "$ANALYSIS_DIR/full_error_messages.txt" | wc -l | xargs echo "  Count:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "discover" "$ANALYSIS_DIR/full_error_messages.txt" | head -5 >> "$ANALYSIS_DIR/error_categories.txt" || true
    echo "" >> "$ANALYSIS_DIR/error_categories.txt"
    
    echo "GGML Errors:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "ggml" "$ANALYSIS_DIR/full_error_messages.txt" | wc -l | xargs echo "  Count:" >> "$ANALYSIS_DIR/error_categories.txt"
    grep -iE "ggml" "$ANALYSIS_DIR/full_error_messages.txt" | head -5 >> "$ANALYSIS_DIR/error_categories.txt" || true
fi

# Extract from strace
echo "[4/5] Analyzing strace output..."
if [ -f "$CAPTURE_DIR/strace.log" ]; then
    # Extract write() syscalls that contain error messages
    grep -E "write\(|writev\(" "$CAPTURE_DIR/strace.log" | \
        grep -iE "(error|failed|timeout|discover|init|cuda|ggml)" | \
        sed 's/.*= //' | \
        head -20 > "$ANALYSIS_DIR/strace_errors.txt" || true
    echo "  Extracted $(wc -l < "$ANALYSIS_DIR/strace_errors.txt" | tr -d ' ') error-related syscalls"
fi

# Generate report
echo "[5/5] Generating analysis report..."
cat > "$ANALYSIS_DIR/REPORT.txt" <<EOF
Ollama Error Analysis Report
=============================
Analysis Time: $(date)
Capture Directory: $CAPTURE_DIR

SUMMARY
-------
EOF

if [ -f "$ANALYSIS_DIR/unique_errors.txt" ]; then
    echo "Unique Error Messages: $(wc -l < "$ANALYSIS_DIR/unique_errors.txt" | tr -d ' ')" >> "$ANALYSIS_DIR/REPORT.txt"
fi

if [ -f "$ANALYSIS_DIR/full_error_messages.txt" ]; then
    echo "Full Error Messages: $(wc -l < "$ANALYSIS_DIR/full_error_messages.txt" | tr -d ' ')" >> "$ANALYSIS_DIR/REPORT.txt"
fi

cat >> "$ANALYSIS_DIR/REPORT.txt" <<EOF

KEY FINDINGS
------------
EOF

# Find the most common error
if [ -f "$ANALYSIS_DIR/full_error_messages.txt" ]; then
    echo "Most Common Error Messages:" >> "$ANALYSIS_DIR/REPORT.txt"
    sort "$ANALYSIS_DIR/full_error_messages.txt" | uniq -c | sort -rn | head -10 >> "$ANALYSIS_DIR/REPORT.txt" || true
    echo "" >> "$ANALYSIS_DIR/REPORT.txt"
fi

# Look for the truncated error message
if [ -f "$ANALYSIS_DIR/full_error_messages.txt" ]; then
    echo "ggml_cuda_init Error Messages:" >> "$ANALYSIS_DIR/REPORT.txt"
    grep -i "ggml.*init" "$ANALYSIS_DIR/full_error_messages.txt" | head -5 >> "$ANALYSIS_DIR/REPORT.txt" || echo "  (none found)" >> "$ANALYSIS_DIR/REPORT.txt"
    echo "" >> "$ANALYSIS_DIR/REPORT.txt"
fi

cat >> "$ANALYSIS_DIR/REPORT.txt" <<EOF

ERROR CATEGORIES
----------------
EOF
cat "$ANALYSIS_DIR/error_categories.txt" >> "$ANALYSIS_DIR/REPORT.txt"

cat >> "$ANALYSIS_DIR/REPORT.txt" <<EOF

NEXT STEPS
----------
1. Review unique_errors.txt for all unique error messages
2. Review full_error_messages.txt for complete (non-truncated) errors
3. Check error_categories.txt for categorized errors
4. Review strace_errors.txt for syscall-level errors
5. Use these error messages to search for solutions online
EOF

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo "Analysis saved to: $ANALYSIS_DIR"
echo ""
echo "Key Files:"
echo "  - REPORT.txt: Full analysis report"
echo "  - unique_errors.txt: All unique error messages"
echo "  - full_error_messages.txt: Complete error messages (not truncated)"
echo "  - error_categories.txt: Categorized errors"
echo ""
cat "$ANALYSIS_DIR/REPORT.txt"

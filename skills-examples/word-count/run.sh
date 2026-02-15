#!/bin/bash
# word-count skill: count words, lines, and characters.
#
# Environment variables (set by the skill runner):
#   SKILL_PARAM_PATH      — file or directory path (required)
#   SKILL_PARAM_RECURSIVE — "true" to count recursively (optional)
#
# Also receives JSON params on stdin.

set -euo pipefail

TARGET="${SKILL_PARAM_PATH:-}"

if [ -z "$TARGET" ]; then
    echo "Error: no path specified" >&2
    exit 1
fi

if [ ! -e "$TARGET" ]; then
    echo "Error: path does not exist: $TARGET" >&2
    exit 1
fi

if [ -f "$TARGET" ]; then
    # Single file
    wc "$TARGET"
elif [ -d "$TARGET" ]; then
    RECURSIVE="${SKILL_PARAM_RECURSIVE:-false}"
    if [ "$RECURSIVE" = "true" ]; then
        find "$TARGET" -type f -print0 | xargs -0 wc 2>/dev/null || echo "No files found."
    else
        # Only immediate files in directory
        find "$TARGET" -maxdepth 1 -type f -print0 | xargs -0 wc 2>/dev/null || echo "No files found."
    fi
else
    echo "Error: unsupported path type: $TARGET" >&2
    exit 1
fi

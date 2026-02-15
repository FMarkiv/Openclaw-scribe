#!/bin/bash
# git-commit skill: stage files and commit with a message.
#
# Environment variables (set by the skill runner):
#   SKILL_PARAM_FILES   — space-separated file paths to stage (required)
#   SKILL_PARAM_MESSAGE — commit message (optional)
#
# Also receives JSON params on stdin.

set -euo pipefail

if [ -z "${SKILL_PARAM_FILES:-}" ]; then
    echo "Error: no files specified" >&2
    exit 1
fi

# Stage files
# shellcheck disable=SC2086
git add $SKILL_PARAM_FILES

# Generate or use provided message
if [ -n "${SKILL_PARAM_MESSAGE:-}" ]; then
    MESSAGE="$SKILL_PARAM_MESSAGE"
else
    # Auto-generate from staged diff
    DIFF=$(git diff --cached --stat 2>/dev/null || true)
    if [ -z "$DIFF" ]; then
        echo "Nothing staged to commit."
        exit 0
    fi
    MESSAGE="Auto-commit: $(echo "$DIFF" | tail -1 | sed 's/^ *//')"
fi

git commit -m "$MESSAGE"

echo "Committed with message: $MESSAGE"

# git-commit

Use this skill to stage files and create a git commit.

## When to use
- When the user asks you to commit changes
- After completing a code modification that should be saved

## How to use
- Always provide the `files` parameter with the files to stage
- If the user provides a commit message, pass it as `message`
- If no message is given, the script generates one from the diff

## Notes
- The script runs `git add` on the specified files, then `git commit`
- If no message is provided, it uses the diff summary as the commit message
- Operates in the agent's workspace directory

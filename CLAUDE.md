# Global Preferences
- Always respond in Korean
- File encoding: always UTF-8

## Code
- Implement only what is requested (no speculative features or premature abstraction)
- Do not delete existing comments or commented-out code without reason
- Read existing code before editing; preserve its style and structure

## Git & Safety
- Never use `git add .` or push directly to `main` (add modified files individually)
- Never commit `.env` or secret files
- Always ask the user for confirmation before running destructive shell commands (rm -rf, reset, etc.)

## Project Init (/init) Rules
- CLAUDE.md: max 200 lines (~2,000 tokens); only actual project info
- Do not include these meta rules in the project's CLAUDE.md
- Split detailed docs into `_claude_docs/`:
  - ARCHITECTURE.md, DATA_FORMATS.md, WORKFLOWS.md, TROUBLESHOOTING.md
- CLAUDE.md must specify `_claude_docs/` reference triggers using this format:
  - "When doing X → read `_claude_docs/Y.md` first"

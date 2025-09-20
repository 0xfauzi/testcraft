# Pre-commit Setup for TestCraft

This document describes the pre-commit hooks that match our GitHub CI checks exactly.

## 🎯 What Gets Checked

Our pre-commit hooks run the **same checks as GitHub CI**:

### ✅ Always Enabled
1. **🔍 Ruff Check** - Code linting (matches `uv run ruff check`)
2. **✨ Ruff Format** - Code formatting (matches `uv run ruff format --check`)
3. **🔒 Safety Scan** - Security vulnerability check (matches `safety check`)
4. **📚 Documentation Check** - Doc validation (matches `python scripts/doc_check.py`)
5. **📋 Standard Checks** - Whitespace, YAML/TOML/JSON syntax, large files, etc.

### ✅ Enabled and Active
6. **🏷️ MyPy Type Check** - Type checking (matches `mypy testcraft/`)
   - **NOW ACTIVE** with staged rollout focusing on critical files
   - Covers domain models, config, and port interfaces
   - Gradually expanding scope as more files are fixed

## 🚀 Setup Instructions

### Option 1: Pre-commit Framework (Recommended)

```bash
# Install and setup (one-time)
uv pip install pre-commit
uv run pre-commit install

# Test the setup
uv run pre-commit run --all-files
```

### Option 2: Manual Git Hook (Backup)

```bash
# Copy backup hook script
cp scripts/git_hooks/pre-commit-backup .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Option 3: Automated Setup

```bash
# Run our setup script
python scripts/setup_pre_commit.py
```

## 🔧 Usage

### Automatic (Recommended)
Pre-commit hooks run **automatically** on every `git commit`. If checks fail:
1. Fix the reported issues
2. Stage your fixes: `git add .`
3. Commit again: `git commit -m "your message"`

### Manual Testing
```bash
# Run all hooks manually
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff
uv run pre-commit run ruff-format

# Run on specific files
uv run pre-commit run --files testcraft/adapters/llm/openai.py
```

## 🎯 What Happens on Commit

```bash
$ git commit -m "fix: update adapter"

🔍 Running pre-commit checks (matching GitHub CI)...
✅ ruff check (lint)........................Passed
✅ ruff format...............................Passed
🔒 safety security scan.....................Passed
📚 documentation check......................Passed
📋 trailing whitespace......................Passed
📋 end of file fixer........................Passed
📋 check yaml syntax........................Passed
📋 check toml syntax........................Passed
📋 check json syntax........................Passed
📋 check for merge conflicts................Passed
📋 check for large files....................Passed
📋 debug statements.........................Passed

[fix/safe-ruff-formatting abc123d] fix: update adapter
 1 file changed, 5 insertions(+), 3 deletions(-)
```

## 🐛 Troubleshooting

### If Pre-commit Fails
```bash
# See what failed
git commit -v

# Fix issues manually
uv run ruff check . --fix
uv run ruff format .

# Try commit again
git commit -m "your message"
```

### Skip Hooks (Emergency Only)
```bash
# Skip all hooks (NOT RECOMMENDED)
git commit --no-verify -m "emergency fix"

# Skip specific hook
SKIP=mypy git commit -m "skip type check"
```

### Update Hooks
```bash
# Update to latest versions
uv run pre-commit autoupdate

# Reinstall hooks after config changes
uv run pre-commit install --overwrite
```

## 📊 Benefits

✅ **Catch Issues Early** - Find problems before GitHub CI
✅ **Faster Feedback** - Fix issues locally vs waiting for CI
✅ **Consistent Quality** - Same standards locally and in CI
✅ **Team Alignment** - Everyone uses the same quality checks
✅ **No Broken Commits** - Maintain clean git history

## 🔧 Customization

Edit `.pre-commit-config.yaml` to:
- Enable/disable specific hooks
- Add new checks
- Modify hook arguments
- Update tool versions

The configuration is designed to exactly match our GitHub CI pipeline for consistency.

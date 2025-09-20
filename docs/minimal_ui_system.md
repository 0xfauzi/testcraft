# TestCraft Minimal UI System

## Overview

TestCraft now supports two UI styles to provide optimal user experience across different environments:

- **Classic UI**: Rich, colorful interface with detailed panels and comprehensive visual feedback
- **Minimal UI**: Clean, compact interface optimized for CI/CD environments and non-interactive terminals

## UI Style Selection

### Automatic Detection

TestCraft automatically selects the appropriate UI style based on your environment:

1. **CI Environment** (`CI=true`): Uses minimal UI
2. **Non-TTY Output**: Uses minimal UI (e.g., when output is redirected)
3. **Interactive Terminal**: Uses classic UI (default)

### Manual Override

You can explicitly choose a UI style using:

```bash
# Force minimal UI
testcraft generate --ui minimal

# Force classic UI
testcraft generate --ui classic

# Environment variable
export TESTCRAFT_UI=minimal
testcraft generate
```

## Minimal UI Features

### Design Principles

The minimal UI follows a "less is more" philosophy:

- ≤4 core colors (green, red, yellow, cyan)
- No emojis or decorative elements
- Compact single-line summaries
- Thin or no borders on tables
- Lowercase headers for consistency
- Minimal padding and whitespace

### Output Format

#### Single-File Generation
```
done 1/1 • tests 8 • time 12.3s
```

#### Multi-File Generation
```
done 5/7 • tests 23 • Δcov +15.0% • time 45.3s

file                 status  progress  tests  time
example.py          done    ●●●●      5      8.2s
helper.py           active  ●●○○      3      4.1s
utils.py            waiting ○○○○      —      —
```

### Live Tracking (Minimal Mode)

For multi-file operations, minimal UI provides:

- **Single column layout** (no statistics sidebar)
- **2Hz refresh rate** (reduced from 3Hz)
- **Top 10 files** displayed (vs 12 in classic)
- **10-dot progress indicator** in footer
- **Compact table** with lowercase headers

## Performance Benefits

### Reduced Output Volume

Minimal UI produces ~70% less output than classic UI:

- **Classic**: ~25-30 lines per operation
- **Minimal**: ~5-8 lines per operation

### Faster Rendering

- Reduced refresh rate (2Hz vs 3Hz)
- No complex panel layouts
- Simplified table rendering
- No emoji or rich markup processing

### Better CI/CD Integration

- Clean, parseable output
- Reduced log noise in CI systems
- Faster terminal rendering
- Works well with log aggregation tools

## Implementation Details

### UI Style Detection Logic

```python
def detect_ui_style(ui_flag: str | None) -> UIStyle:
    # Priority 1: Explicit --ui flag
    if ui_flag:
        return UIStyle.MINIMAL if ui_flag.lower() == "minimal" else UIStyle.CLASSIC

    # Priority 2: Environment variable
    env_ui = os.getenv("TESTCRAFT_UI")
    if env_ui:
        return UIStyle.MINIMAL if env_ui.lower() == "minimal" else UIStyle.CLASSIC

    # Priority 3: Auto-detect
    if os.getenv("CI") == "true" or not sys.stdout.isatty():
        return UIStyle.MINIMAL

    # Default: Classic for interactive terminals
    return UIStyle.CLASSIC
```

### Minimal Theme Colors

The minimal theme uses a restricted color palette:

- `success`: green (completed operations)
- `error`: red (failed operations)
- `status_working`: yellow (active operations)
- `accent`: cyan (highlights and progress)
- `muted`: dim white (secondary text)
- `primary`: white (main text)
- `border`: dim (table borders when needed)

All other theme colors are aliased to these core colors.

### Centralized Logging

The minimal UI system includes improved logging architecture:

- **Single root handler**: Eliminates duplicate log messages
- **Minimal templates**: Clean, structured log format
- **Reduced verbosity**: Essential information only
- **No emoji in logs**: Plain text for better parsing

## Migration Guide

### For Existing Users

No changes required! The system automatically:

- Detects your environment
- Maintains classic UI for interactive use
- Switches to minimal for CI/non-TTY contexts

### For CI/CD Pipelines

To ensure minimal UI in CI systems:

```yaml
# GitHub Actions example
- name: Run TestCraft
  run: testcraft generate --ui minimal
  env:
    TESTCRAFT_UI: minimal
```

### For Library Integration

When using TestCraft as a library:

```python
from testcraft.adapters.io.enhanced_ui import EnhancedUIAdapter
from testcraft.adapters.io.ui_rich import UIStyle
from testcraft.adapters.io.rich_cli import get_theme
from rich.console import Console

# Create minimal UI
console = Console(theme=get_theme(UIStyle.MINIMAL))
ui = EnhancedUIAdapter(console, ui_style=UIStyle.MINIMAL)
```

## Testing

The minimal UI system includes comprehensive tests:

- UI style detection logic
- Theme color restrictions
- Minimal renderer output format
- Live tracking layout verification
- Integration with existing workflows

Run tests with:
```bash
pytest tests/test_minimal_ui.py -v
```

## Comparison: Before vs After

### Before (Classic Only)
```
🎯 TestCraft Test Generation

┌─────────────────────────────────────────────────────────┐
│                     📈 Project Summary                    │
├─────────────────────────────────────────────────────────┤
│ 📁 Files Analyzed:      5                               │
│ 🧪 Files with Tests:    3 📊 (60%)                       │
│ 📊 Overall Coverage:    🟢 85.0%                         │
│ 🧪 Tests Generated:     23                              │
│ 🎯 Success Rate:        ✅ 100%                          │
└─────────────────────────────────────────────────────────┘

🧪 Test Generation Results
┌─────────────────────────────┬─────────────────┬──────────┬────────┬─────────┐
│ Source File                 │ Test File       │ Status   │ Tests  │ Time    │
├─────────────────────────────┼─────────────────┼──────────┼────────┼─────────┤
│ example.py                  │ test_example.py │ 🎉 Success │ 8      │ 12.3s   │
│ helper.py                   │ test_helper.py  │ 🎉 Success │ 5      │ 8.1s    │
│ utils.py                    │ test_utils.py   │ 🎉 Success │ 10     │ 15.2s   │
└─────────────────────────────┴─────────────────┴──────────┴────────┴─────────┘

✅ Coverage improved by 15.0%
```

### After (Minimal)
```
done 5/5 • tests 23 • Δcov +15.0% • time 35.6s

file        status  progress  tests  time
example.py  done    ●●●●      8      12.3s
helper.py   done    ●●●●      5      8.1s
utils.py    done    ●●●●      10     15.2s
```

**Result**: 85% reduction in output lines while maintaining essential information.

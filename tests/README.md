# End-to-End Test Guide

This suite exercises the TestCraft CLI against both live providers and locally patched adapters. Before running the E2E flow, export API credentials in a repository-root `.env` file. Populate `TESTCRAFT_E2E_PROVIDER`/`TESTCRAFT_E2E_MODEL` if you need to pin a specific vendor or deployment.

## Markers
- `integration`: long-running cross-layer tests.
- `live_service`: requires network access and valid provider credentials.
- `fake_service`: uses sitecustomize shims to simulate outages/failures (no network needed).
- `utility`: focuses on administrative CLI commands (env, cost, sync/reset, models, etc.).

## Running subsets
```bash
# Baseline smoke of live journey
pytest tests/e2e/test_cli_end_to_end.py -m integration

# Full live-service journeys
pytest tests/e2e -m "integration and live_service" --maxfail=1

# Offline/failure simulations
pytest tests/e2e -m fake_service

# Utility-focused coverage (reuses baseline project output)
pytest tests/e2e -m utility
```

The scenarios write per-test logs beneath `.testcraft/cli_logs`. Each test cleans up its temporary project; if diagnostics are needed, copy the logs before rerunning.

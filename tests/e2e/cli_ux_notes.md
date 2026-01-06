# CLI UX Notes

- Running `testcraft generate …` before `testcraft init-config` exits with status `1` and emits no user-facing guidance; the CLI should prompt to initialize configuration instead of failing silently.
- Importing the Bedrock adapter pulls in `numpy`/`langchain_aws`, which triggered a macOS-specific segmentation fault through `numpy.polyfit` even when Bedrock is not the active provider. We mitigated this by lazily importing Bedrock, but the CLI should guard against optional provider imports crashing the process.
- Setting `TESTCRAFT_UI=classic` via environment variables causes configuration validation to fail (`extra_forbidden`) rather than falling back to the closest supported UI style.
- `testcraft generate …` surfaces no logs while waiting on real LLM calls; when network is blocked the command silently hangs for ~7 minutes before timing out, leaving the user without feedback.

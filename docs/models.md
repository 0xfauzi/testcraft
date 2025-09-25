# Model Catalog

TestCraft centralizes model limits, pricing, and feature flags in a single **TOML** file at `testcraft/config/model_catalog.toml`.

## Format

Each model is defined using a `[[models]]` table:

```toml
[[models]]
provider = "openai"            # "openai" | "anthropic"
model_id = "gpt-4.1"           # Canonical model id

[models.limits]
max_context = 1000000           # Total token window
default_max_output = 32768      # Conservative output cap used by TokenCalculator
max_thinking = 32000            # Optional, for models that support configurable thinking

[models.flags]
beta = false
supports_thinking = false
reasoning = false
deprecated = false

[models.pricing.per_million]
input = 5000                    # USD per 1M input tokens
output = 15000                  # USD per 1M output tokens

[models.source]
url = "https://provider/docs/models"
last_verified = "2025-07-01T00:00:00Z"
```

## Normalization

Identifiers from Azure OpenAI and AWS Bedrock are normalized to canonical `(provider, model_id)`
prior to lookups. Examples:

- `azure-openai::gpt-4o` → `openai/gpt-4.1`
- `bedrock::anthropic.claude-3-7-sonnet-v1:0` → `anthropic/claude-3-7-sonnet`

Unknown models are treated as errors and must be added to the catalog.

## Pricing Math

Pricing is stored per-million tokens; costs are computed as:

```
(prompt_tokens * input_per_million + completion_tokens * output_per_million) / 1_000_000
```

## CLI

Use the CLI to inspect and verify the catalog:

```bash
# Show models (table or JSON)
testcraft models show --provider openai --format table

# Verify catalog (duplicates, issues)
testcraft models verify

# Diff current catalog against a previous file
testcraft models diff --file build/reports/model_inventory_previous.toml
```

## Governance

- Keep `last_verified` up to date and link to the official provider docs
- Run `testcraft models verify` in monthly reviews
- Use `diff` in PRs to review changes to limits, pricing, and flags

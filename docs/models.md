# TestCraft Model Management Guide

This guide covers TestCraft's model catalog system, CLI commands for model management, and links to official vendor documentation.

## Table of Contents

1. [Model Catalog Overview](#model-catalog-overview)
2. [CLI Commands](#cli-commands)
3. [Verification and Compliance](#verification-and-compliance)
4. [Official Documentation Links](#official-documentation-links)
5. [Monthly Review Process](#monthly-review-process)
6. [Troubleshooting](#troubleshooting)

## Model Catalog Overview

TestCraft maintains a centralized model catalog (`testcraft/config/model_catalog.toml`) that serves as the single source of truth for all LLM model metadata. This ensures consistent, up-to-date information across your entire application.

### What's Included

The model catalog contains:

- **Model Limits**: Context windows, default output limits, thinking tokens
- **Pricing Data**: Per-million token costs for accurate cost tracking
- **Feature Flags**: Vision, tool use, structured outputs, reasoning capabilities
- **Beta Features**: Extended limits and experimental features (opt-in only)
- **Provenance**: Links to official documentation and last verification dates

### Benefits

✅ **Accurate Budgets**: Token limits never exceed vendor specifications  
✅ **Cost Control**: Real-time cost tracking based on official pricing  
✅ **Feature Safety**: Beta features require explicit configuration  
✅ **Compliance**: Regular verification against vendor documentation  

## CLI Commands

TestCraft provides comprehensive CLI commands for managing and verifying your model catalog.

### `testcraft models show`

Display model information in a formatted table:

```bash
# Show all models
testcraft models show

# Filter by provider
testcraft models show --provider=openai
testcraft models show --provider=anthropic

# Include model aliases
testcraft models show --include-aliases

# Different output formats
testcraft models show --format=json
testcraft models show --format=yaml
testcraft models show --format=table  # default
```

**Example Output:**
```
Model Catalog (6 models)

┌─────────────┬─────────────────┬─────────────────┬─────────────────┬──────────────┬──────────────┐
│ Provider    │ Model ID        │ Max Context     │ Max Output      │ Input $/1M   │ Output $/1M  │
├─────────────┼─────────────────┼─────────────────┼─────────────────┼──────────────┼──────────────┤
│ openai      │ gpt-4o          │ 128,000         │ 4,096           │ $5.00        │ $15.00       │
│ openai      │ gpt-4o-mini     │ 128,000         │ 16,384          │ $0.15        │ $0.60        │
│ anthropic   │ claude-3-7-son… │ 200,000         │ 4,096           │ $3.00        │ $15.00       │
└─────────────┴─────────────────┴─────────────────┴─────────────────┴──────────────┴──────────────┘
```

### `testcraft models verify`

Verify your model catalog against current codebase usage:

```bash
# Verify all models
testcraft models verify

# Verify specific provider
testcraft models verify --provider=openai

# Verbose verification with details
testcraft models verify --verbose
```

**Checks performed:**
- ✅ All referenced models exist in catalog
- ✅ No hardcoded limits exceed catalog values
- ✅ Pricing calculations use catalog data
- ✅ Beta features properly gated behind configuration

### `testcraft models diff`

View changes to the model catalog over time:

```bash
# Show changes since a specific date
testcraft models diff --since=2025-01-01

# Show changes since last week
testcraft models diff --since=7d

# Show changes since last verification
testcraft models diff --since=last-verified

# Different output formats
testcraft models diff --since=2025-01-01 --format=json
```

**Example Output:**
```
Model Catalog Changes Since 2025-01-01

📊 Summary: 3 models updated, 1 model added

┌─────────────┬─────────────────┬─────────────┬─────────────────┬──────────────────┐
│ Provider    │ Model ID        │ Change Type │ Field Modified  │ Previous → New   │
├─────────────┼─────────────────┼─────────────┼─────────────────┼──────────────────┤
│ openai      │ gpt-4o          │ Updated     │ pricing.input   │ $10.00 → $5.00   │
│ anthropic   │ claude-sonnet-4 │ Added       │ -               │ New model        │
└─────────────┴─────────────────┴─────────────┴─────────────────┴──────────────────┘
```

## Verification and Compliance

### Automated Checks

TestCraft automatically ensures compliance with your model catalog:

1. **Token Budget Enforcement**: Requests never exceed catalog limits
2. **Pricing Accuracy**: Cost calculations use catalog data
3. **Beta Feature Gating**: Extended features require explicit opt-in
4. **Parameter Validation**: Request parameters match provider expectations

### Manual Verification

Run verification checks before important deployments:

```bash
# Full verification suite
testcraft models verify

# Check specific components
testcraft models verify --check=limits
testcraft models verify --check=pricing  
testcraft models verify --check=beta-features
```

## Official Documentation Links

Always refer to official vendor documentation for the most current information:

### OpenAI
- **Models Overview**: https://platform.openai.com/docs/models
- **Pricing**: https://openai.com/api/pricing/
- **Rate Limits**: https://platform.openai.com/docs/guides/rate-limits
- **API Reference**: https://platform.openai.com/docs/api-reference

### Anthropic Claude
- **Models**: https://docs.anthropic.com/en/docs/about-claude/models
- **Pricing**: https://docs.anthropic.com/en/docs/about-claude/pricing
- **API Reference**: https://docs.anthropic.com/en/api/getting-started
- **Rate Limits**: https://docs.anthropic.com/en/api/rate-limits

### Azure OpenAI
- **Service Overview**: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/
- **Pricing**: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
- **Quotas & Limits**: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/quotas-limits

### AWS Bedrock
- **Supported Models**: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
- **Pricing**: https://aws.amazon.com/bedrock/pricing/
- **Service Quotas**: https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html

## Monthly Review Process

To ensure your model catalog stays current and accurate, follow this monthly review checklist:

### 🗓️ Monthly Review Checklist

**Week 1 of each month:**

- [ ] **Check Official Documentation**
  - Visit all vendor documentation links above
  - Note any model additions, removals, or limit changes
  - Check for pricing updates

- [ ] **Review Catalog Accuracy**
  ```bash
  testcraft models verify --verbose
  ```

- [ ] **Update Last Verified Dates**
  - Update `last_verified` field in `model_catalog.toml` for reviewed models
  - Document any changes made

- [ ] **Test Beta Features**
  - Review beta feature availability and pricing
  - Update beta feature configurations if needed

- [ ] **Generate Change Report**
  ```bash
  testcraft models diff --since=1m --format=json > monthly-changes.json
  ```

### 🔄 Maintenance Tasks

**Quarterly (every 3 months):**

- [ ] Review all model aliases for accuracy
- [ ] Audit pricing data against vendor billing
- [ ] Check for deprecated models or features
- [ ] Update documentation links if moved

**Annually:**

- [ ] Full catalog audit against vendor documentation
- [ ] Review and update conservative defaults
- [ ] Archive deprecated model entries
- [ ] Update schema if needed

## Troubleshooting

### Common Issues

#### "Model not found in catalog"
```bash
# Check what models are available
testcraft models show --provider=your-provider

# Verify the model exists
testcraft models verify --verbose
```

#### "Token limit exceeded"
```bash
# Check current limits
testcraft models show --provider=your-provider

# Enable beta features if needed (with caution)
# Edit .testcraft.toml:
[llm.beta]
enable_extended_output = true
```

#### "Pricing calculation mismatch"
```bash
# Verify catalog pricing is current
testcraft models show --format=table

# Check for recent pricing changes
testcraft models diff --since=30d
```

#### "Beta features not working"
Make sure beta features are explicitly enabled in your configuration:

```toml
[llm.beta.anthropic]
enable_extended_output = true
enable_extended_context = true
```

### Getting Help

If you encounter issues:

1. **Run verification**: `testcraft models verify --verbose`
2. **Check recent changes**: `testcraft models diff --since=7d`
3. **Consult vendor documentation** (links above)
4. **Review configuration**: Check `.testcraft.toml` beta settings
5. **Open an issue** with verification output and error details

---

**Last Updated**: 2025-09-15  
**Next Review Due**: 2025-10-15

For configuration details, see the [Configuration Reference](configuration.md).

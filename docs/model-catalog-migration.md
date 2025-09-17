# Model Catalog Migration Guide

This guide helps you migrate to TestCraft's new model catalog system introduced in version 0.2.0.

## Overview

TestCraft now uses a centralized model catalog (`testcraft/config/model_catalog.toml`) as the single source of truth for all LLM model metadata. This ensures accurate limits, pricing, and feature flags while maintaining compliance with vendor specifications.

## What Changed

### Before (Hardcoded Limits)
```python
# Old approach - hardcoded in adapters
PROVIDER_LIMITS = {
    "gpt-4": {"max_context": 8000, "max_output": 4000},
    "claude-3-sonnet": {"max_context": 200000, "max_output": 4000}
}
```

### After (Catalog-Driven)
```toml
# New approach - centralized catalog
[[models]]
provider = "openai"
model_id = "gpt-4o"

[models.limits]
max_context = 128000
default_max_output = 4096

[models.source]
url = "https://platform.openai.com/docs/models/gpt-4o"
last_verified = "2025-09-15"
```

## Migration Checklist

### ✅ Immediate Actions (Required)

1. **Verify Your Models**
   ```bash
   # Check that your models are in the catalog
   testcraft models verify
   ```

2. **Review Beta Features**
   If you were using extended limits, you now need explicit configuration:
   ```toml
   # Add to .testcraft.toml
   [llm.beta.anthropic]
   enable_extended_output = true
   enable_extended_context = true
   ```

3. **Test Your Configuration**
   ```bash
   # Ensure everything still works
   testcraft generate your-test-file.py --dry-run
   ```

### 🔍 Optional Actions (Recommended)

1. **Explore New CLI Commands**
   ```bash
   # View your model catalog
   testcraft models show
   
   # Check for recent changes
   testcraft models diff --since=30d
   ```

2. **Review Cost Tracking**
   The new system provides more accurate cost calculations based on official vendor pricing.

3. **Set Up Monthly Reviews**
   Consider implementing the [monthly review process](monthly-model-review-checklist.md) to keep your catalog current.

## Specific Migration Scenarios

### Scenario 1: Standard OpenAI/Anthropic Usage

**Before:**
```toml
[llm]
default_provider = "openai"
openai_model = "gpt-4"
openai_max_tokens = 8000
```

**After:**
```toml
[llm]
default_provider = "openai"
openai_model = "gpt-4o"  # Updated to catalog model
# max_tokens now automatically from catalog
```

**Action:** Models are automatically validated against the catalog. No changes needed.

### Scenario 2: Azure OpenAI Custom Deployments

**Before:**
```toml
[llm]
default_provider = "azure-openai"
azure_openai_deployment = "my-gpt4-deployment"
azure_openai_max_tokens = 12000
```

**After:**
```toml
[llm]
default_provider = "azure-openai"
azure_openai_deployment = "my-gpt4-deployment"
# Deployment mapped to catalog model automatically
# Token limits enforced per catalog specifications
```

**Action:** Verify your deployment maps to a supported model with `testcraft models verify`.

### Scenario 3: Custom/Extended Limits

**Before:**
```toml
[llm]
anthropic_max_tokens = 200000  # Using extended context
```

**After:**
```toml
[llm]
# Standard limits from catalog used by default

[llm.beta.anthropic]
enable_extended_context = true  # Explicit opt-in required
```

**Action:** Add explicit beta feature configuration if you need extended limits.

### Scenario 4: Cost Budget Configuration

**Before:**
```toml
[cost_management]
per_request_limit = 2.0  # Hope the pricing is current
```

**After:**
```toml
[cost_management]
per_request_limit = 2.0  # Now uses accurate catalog pricing
```

**Action:** Review your cost thresholds as pricing accuracy has improved.

## Troubleshooting Common Issues

### Issue: "Model not found in catalog"

**Error:**
```
Error: Model 'gpt-3.5-turbo-custom' not found in catalog
```

**Solution:**
1. Check available models: `testcraft models show --provider=openai`
2. Use a supported model ID or add custom entry to catalog
3. Update your configuration with a supported model

### Issue: "Token limit exceeded"

**Error:**
```
Error: Requested 50000 tokens exceeds catalog limit of 32768
```

**Solutions:**
1. **Enable beta features** (if supported):
   ```toml
   [llm.beta.openai]
   enable_extended_output = true
   ```

2. **Reduce your request size**:
   ```toml
   [generation.prompt_budgets]
   total_chars = 8000  # Reduce context size
   ```

3. **Check catalog limits**: `testcraft models show --provider=your-provider`

### Issue: "Pricing calculation mismatch"

**Problem:** Cost tracking seems different from before

**Solution:**
1. **Verify current pricing**: `testcraft models show --format=table`
2. **Check for recent changes**: `testcraft models diff --since=30d`
3. **Review vendor documentation** for pricing updates

### Issue: "Beta features not working"

**Problem:** Extended context/output limits not applying

**Solution:**
Ensure explicit configuration:
```toml
[llm.beta.anthropic]
enable_extended_output = true
enable_extended_context = true

[llm.beta.openai]
enable_extended_output = true
```

## Configuration Examples

### Minimal Configuration (Recommended)
```toml
[llm]
default_provider = "anthropic"
anthropic_model = "claude-3-5-sonnet"
# All limits and pricing from catalog automatically
```

### Production Configuration
```toml
[llm]
default_provider = "openai" 
openai_model = "gpt-4o"

[llm.beta]
# Conservative: only enable if explicitly needed
enable_extended_output = false
enable_extended_context = false

[cost_management.cost_thresholds]
# Review these values with new accurate pricing
daily_limit = 50.0
per_request_limit = 2.0
warning_threshold = 1.0
```

### Advanced Configuration (Beta Features)
```toml
[llm]
default_provider = "anthropic"
anthropic_model = "claude-sonnet-4"

[llm.beta.anthropic]
# Explicit opt-in for extended capabilities
enable_extended_output = true
enable_extended_context = true

# Note: This may incur additional costs
```

## Validation Commands

After migration, run these commands to ensure everything is working:

```bash
# Full verification suite
testcraft models verify --verbose

# Test cost calculations
testcraft models show --format=table

# Check configuration
testcraft generate --dry-run sample-file.py

# Review any recent catalog changes
testcraft models diff --since=7d
```

## Benefits of Migration

### ✅ What You Gain

- **Accurate Budgets**: Token limits based on official vendor specifications
- **Cost Control**: Real-time pricing from official documentation  
- **Compliance**: Automatic enforcement of vendor limits
- **Transparency**: Clear view of what models and limits you're using
- **Future-Proof**: Automatic updates when vendors change specifications

### ⚠️ What to Watch For

- **Beta Features**: Now require explicit opt-in (more secure)
- **Conservative Defaults**: May be lower than what you were using before
- **Pricing Accuracy**: May show different costs (but more accurate)
- **Model Validation**: Some custom model names may not be recognized

## Getting Help

If you run into issues:

1. **Run diagnostics**: `testcraft models verify --verbose`
2. **Check documentation**: Review [models.md](models.md) for CLI usage
3. **Review changes**: `testcraft models diff --since=30d`
4. **Consult vendor docs**: Links provided in [models.md](models.md)

## Timeline

- **Immediate**: Migration is backward-compatible, no breaking changes
- **Week 1**: Review and test your current setup
- **Month 1**: Implement monthly review process
- **Ongoing**: Regular catalog updates and verification

---

**Migration Version**: 0.2.0  
**Last Updated**: 2025-09-15  
**Questions?** See [Model Management Guide](models.md) or open an issue.

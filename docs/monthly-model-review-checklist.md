# Monthly Model Catalog Review Checklist

This checklist ensures TestCraft's model catalog remains accurate and compliant with vendor specifications.

## 📅 Schedule

**When**: First week of each month  
**Duration**: ~30-60 minutes  
**Owner**: DevOps/Platform team  

## ✅ Review Steps

### 1. Vendor Documentation Review

**OpenAI** (Check: https://platform.openai.com/docs/models)
- [ ] Review model availability and specifications
- [ ] Check for new models or deprecated models
- [ ] Verify context limits and output limits
- [ ] Review pricing changes: https://openai.com/api/pricing/
- [ ] Note any API changes or new beta features

**Anthropic Claude** (Check: https://docs.anthropic.com/en/docs/about-claude/models)
- [ ] Review Claude model family updates
- [ ] Check context window and output limits
- [ ] Verify pricing: https://docs.anthropic.com/en/docs/about-claude/pricing
- [ ] Review new reasoning or thinking capabilities
- [ ] Check for new beta features or headers

**Azure OpenAI** (Check: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [ ] Review deployment model availability
- [ ] Check quota and limit changes
- [ ] Verify pricing: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
- [ ] Review regional availability changes

**AWS Bedrock** (Check: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)
- [ ] Review Anthropic model versions on Bedrock
- [ ] Check service limits and quotas
- [ ] Verify pricing: https://aws.amazon.com/bedrock/pricing/
- [ ] Review model-specific parameter requirements

### 2. Catalog Verification

```bash
# Run full verification
testcraft models verify --verbose

# Check for any issues
testcraft models show --format=table
```

- [ ] All models in catalog are still available
- [ ] No hardcoded limits exceed catalog values
- [ ] Pricing data is current (within 5% of official rates)
- [ ] Beta features properly configured
- [ ] No deprecated models being used

### 3. Update Catalog

For each model requiring updates:

- [ ] Update `testcraft/config/model_catalog.toml`
- [ ] Modify limits if vendor specifications changed
- [ ] Update pricing (convert to per-million token format)
- [ ] Update `last_verified` date to current date
- [ ] Add notes about any significant changes

**Example update:**
```toml
[models.source]
url = "https://platform.openai.com/docs/models/gpt-4o"
last_verified = "2025-09-15"  # ← Update this date
notes = "Pricing reduced from $10/1M to $5/1M input tokens - Sept 2025"
```

### 4. Generate Monthly Report

```bash
# Generate changes since last month
testcraft models diff --since=30d --format=json > reports/model-changes-$(date +%Y-%m).json

# Create summary table
testcraft models diff --since=30d --format=table > reports/monthly-model-summary-$(date +%Y-%m).txt
```

- [ ] Document any model additions or removals
- [ ] Note pricing changes and impact on cost budgets
- [ ] Record new features or capabilities
- [ ] Update team on significant changes

### 5. Configuration Review

- [ ] Review beta feature usage in production
- [ ] Check if any extended limits are needed
- [ ] Verify cost thresholds are appropriate
- [ ] Update team configuration templates if needed

### 6. Testing

- [ ] Run integration tests with updated catalog
- [ ] Verify cost calculations are accurate
- [ ] Test beta feature gating works correctly
- [ ] Confirm no regressions in model routing

## 📊 Monthly Report Template

```markdown
# Model Catalog Review - [Month Year]

## Summary
- ✅ Models reviewed: X
- 📝 Models updated: X  
- 💰 Pricing changes: X
- 🆕 New models: X
- ❌ Deprecated models: X

## Key Changes
1. [Change description with impact]
2. [Change description with impact]

## Actions Taken
- [ ] Updated catalog entries
- [ ] Notified team of pricing changes
- [ ] Updated configuration templates
- [ ] Generated diff report

## Next Review: [First week of next month]
```

## 🚨 Escalation Criteria

**Immediate escalation required if:**
- Model pricing increases >20%
- Critical models are deprecated with <90 days notice
- New usage limits affect production workloads
- Security vulnerabilities in model APIs

**Contact**: Platform team lead + Finance for pricing impacts

## 📚 Resources

- [Model Management Guide](models.md)
- [Configuration Reference](configuration.md)
- **Vendor Documentation** (always check official sources):
  - [OpenAI Models](https://platform.openai.com/docs/models)
  - [Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models)
  - [Azure OpenAI](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
  - [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)

---

**Template Version**: 1.0  
**Last Updated**: 2025-09-15  
**Next Template Review**: 2026-01-15

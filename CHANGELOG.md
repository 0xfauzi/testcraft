# Changelog

All notable changes to TestCraft will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Model Catalog System**: Centralized model metadata management with single source of truth
  - New `testcraft/config/model_catalog.toml` with comprehensive model specifications
  - Automatic token budget enforcement based on official vendor documentation
  - Per-million token pricing calculations for accurate cost tracking
  - Feature flags for vision, tool use, structured outputs, and reasoning capabilities
  - Beta feature gating with explicit opt-in configuration
  - Provenance tracking with vendor documentation links and verification dates

- **Model Management CLI Commands**:
  - `testcraft models show` - Display model catalog with limits and pricing
  - `testcraft models verify` - Verify catalog compliance against codebase usage  
  - `testcraft models diff --since=DATE` - View catalog changes over time
  - Support for multiple output formats (table, JSON, YAML)
  - Provider filtering and detailed verification reporting

- **Documentation & Governance**:
  - New [Model Management Guide](docs/models.md) with CLI usage and vendor links
  - Updated [Configuration Reference](docs/configuration.md) with model catalog integration
  - Monthly review checklist for maintaining catalog accuracy
  - Links to official OpenAI, Anthropic, Azure, and AWS Bedrock documentation

- **Beta Feature Configuration**:
  - Provider-specific beta feature controls (`[llm.beta.anthropic]`, `[llm.beta.openai]`)
  - Extended context/output limits with explicit opt-in only
  - Automatic beta header injection for Anthropic extended features
  - Conservative defaults to ensure vendor compliance

### Changed
- **LLM Adapters**: Now read limits and pricing from model catalog instead of hardcoded values
- **Token Calculator**: Enforces catalog-based limits with safety margins
- **Cost Management**: Uses catalog pricing for consistent cost calculations across providers
- **Parameter Validation**: Ensures request parameters match provider expectations and never exceed catalog caps

### Migration Guide

#### For Existing Configurations

1. **No immediate action required** - existing configurations continue to work
2. **Beta features now require explicit opt-in**:
   ```toml
   # Add to your .testcraft.toml if you need extended limits
   [llm.beta.anthropic]
   enable_extended_output = true
   enable_extended_context = true
   ```

3. **Review your model usage**:
   ```bash
   # Check what models you're using
   testcraft models verify
   
   # View current catalog
   testcraft models show
   ```

#### For Custom Model Configurations

- **Azure OpenAI deployments**: Now mapped to canonical model entries automatically
- **Custom model names**: May need catalog entries or will fallback to safe defaults
- **Hardcoded limits**: Will be replaced by catalog values (safer defaults)

#### Verification Commands

Run these commands to ensure smooth transition:

```bash
# Verify all models exist and limits are respected
testcraft models verify --verbose

# Check for any configuration issues
testcraft models show --provider=your-provider

# Review recent changes
testcraft models diff --since=30d
```

### For Developers

#### Breaking Changes
- **None** - this is a backward-compatible addition

#### New APIs
- `testcraft.config.model_catalog_loader.load_catalog()` - Load model catalog
- `testcraft.adapters.llm.pricing.get_pricing(model_id)` - Get model pricing
- Model capabilities exposed via `llm_router.get_capabilities()`

#### Testing
- All existing tests continue to pass
- New test suite for model catalog validation and CLI commands
- Integration tests verify catalog-driven behavior

### Security Notes
- **Default behavior is more secure**: Conservative limits by default
- **Beta features explicit**: Extended features require opt-in configuration
- **Vendor compliance**: Automatic enforcement of documented limits
- **Pricing accuracy**: Prevents cost overruns from outdated pricing data

---

## [0.1.0] - 2025-09-15

### Added
- Initial release of TestCraft
- AI-powered test generation with OpenAI, Anthropic, Azure OpenAI, and AWS Bedrock
- Coverage analysis and intelligent test refinement
- Rich CLI interface with progress tracking
- Comprehensive TOML configuration system
- Evaluation harness with LLM-as-judge capabilities
- Clean Architecture with modular design
- Telemetry and cost management

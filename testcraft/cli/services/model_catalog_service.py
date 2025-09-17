"""Model catalog business logic service."""

from datetime import datetime, timedelta
from typing import Any

from ...config.model_catalog_loader import load_catalog


class ModelCatalogService:
    """Service for model catalog operations and business logic."""

    def get_catalog_data(self, provider_filter: str | None = None, include_aliases: bool = False) -> dict[str, Any]:
        """Get filtered catalog data."""
        catalog = load_catalog()
        
        filtered_models = []
        for entry in catalog.models:
            if provider_filter is None or entry.provider.lower() == provider_filter.lower():
                filtered_models.append(entry)
        
        # Sort by provider, then by model_id
        filtered_models.sort(key=lambda x: (x.provider, x.model_id))
        
        return {
            "catalog": catalog,
            "models": filtered_models,
            "include_aliases": include_aliases,
            "total_models": len(filtered_models),
            "providers": set(entry.provider for entry in filtered_models)
        }

    def verify_catalog_integrity(self, provider_filter: str | None = None, check_usage: bool = False) -> dict[str, Any]:
        """Verify catalog integrity and optionally check usage compliance."""
        catalog = load_catalog()
        
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check catalog structure and data integrity
        model_ids_seen = set()
        provider_stats = {}
        
        for entry in catalog.models:
            provider_key = entry.provider.lower()
            
            # Skip if filtering by provider
            if provider_filter and provider_key != provider_filter.lower():
                continue
            
            # Track provider stats
            if provider_key not in provider_stats:
                provider_stats[provider_key] = {"models": 0, "issues": 0}
            provider_stats[provider_key]["models"] += 1
            
            # Check for duplicate model IDs within provider
            model_key = f"{provider_key}:{entry.model_id.lower()}"
            if model_key in model_ids_seen:
                results["errors"].append(f"Duplicate model ID: {entry.provider}:{entry.model_id}")
                results["passed"] = False
                provider_stats[provider_key]["issues"] += 1
            model_ids_seen.add(model_key)
            
            # Check required fields
            if not entry.model_id:
                results["errors"].append(f"Missing model ID for {entry.provider} entry")
                results["passed"] = False
                provider_stats[provider_key]["issues"] += 1
            
            if not entry.limits.max_context or entry.limits.max_context <= 0:
                results["errors"].append(f"Invalid max_context for {entry.provider}:{entry.model_id}")
                results["passed"] = False
                provider_stats[provider_key]["issues"] += 1
            
            if not entry.limits.default_max_output or entry.limits.default_max_output <= 0:
                results["errors"].append(f"Invalid default_max_output for {entry.provider}:{entry.model_id}")
                results["passed"] = False
                provider_stats[provider_key]["issues"] += 1
            
            # Check logical constraints
            if entry.limits.default_max_output > entry.limits.max_context:
                results["warnings"].append(
                    f"default_max_output ({entry.limits.default_max_output}) > max_context ({entry.limits.max_context}) "
                    f"for {entry.provider}:{entry.model_id}"
                )
                provider_stats[provider_key]["issues"] += 1
            
            # Check source information
            if not entry.source or not entry.source.url:
                results["warnings"].append(f"Missing source URL for {entry.provider}:{entry.model_id}")
            
            if not entry.source or not entry.source.last_verified:
                results["warnings"].append(f"Missing last_verified date for {entry.provider}:{entry.model_id}")
            elif entry.source.last_verified:
                try:
                    verified_date = datetime.strptime(entry.source.last_verified, "%Y-%m-%d")
                    days_old = (datetime.now() - verified_date).days
                    if days_old > 90:  # Warn if older than 90 days
                        results["warnings"].append(
                            f"Verification date is {days_old} days old for {entry.provider}:{entry.model_id}"
                        )
                except ValueError:
                    results["warnings"].append(f"Invalid date format for {entry.provider}:{entry.model_id}")
            
            # Check alias conflicts
            for alias in entry.aliases:
                alias_key = f"{provider_key}:{alias.lower()}"
                if alias_key in model_ids_seen:
                    results["errors"].append(
                        f"Alias '{alias}' for {entry.provider}:{entry.model_id} conflicts with existing model ID"
                    )
                    results["passed"] = False
                    provider_stats[provider_key]["issues"] += 1
            
            # Add aliases to seen set
            for alias in entry.aliases:
                model_ids_seen.add(f"{provider_key}:{alias.lower()}")
        
        # Check usage compliance if requested
        if check_usage:
            usage_results = self._check_usage_compliance(catalog, provider_filter)
            results["usage_compliance"] = usage_results
            if not usage_results["compliant"]:
                results["passed"] = False
                results["errors"].extend(usage_results["violations"])
        
        results["stats"] = {
            "total_models": sum(stats["models"] for stats in provider_stats.values()),
            "total_providers": len(provider_stats),
            "provider_breakdown": provider_stats,
            "total_errors": len(results["errors"]),
            "total_warnings": len(results["warnings"])
        }
        
        return results

    def generate_catalog_diff(self, since_date: datetime, provider_filter: str | None = None) -> dict[str, Any]:
        """Generate catalog diff since specified date."""
        # This is a simplified implementation. In a real system, this would:
        # - Compare against a historical version of the catalog
        # - Track changes in a version control system
        # - Maintain change logs
        
        catalog = load_catalog()
        
        results = {
            "since_date": since_date.isoformat(),
            "changes": [],
            "summary": {
                "added": 0,
                "modified": 0,
                "removed": 0
            }
        }
        
        # For now, we'll check last_verified dates as a proxy for recent changes
        recent_changes = []
        
        for entry in catalog.models:
            if provider_filter and entry.provider.lower() != provider_filter.lower():
                continue
            
            if entry.source and entry.source.last_verified:
                try:
                    verified_date = datetime.strptime(entry.source.last_verified, "%Y-%m-%d")
                    if verified_date >= since_date:
                        recent_changes.append({
                            "type": "verified",
                            "provider": entry.provider,
                            "model_id": entry.model_id,
                            "date": verified_date.isoformat(),
                            "details": f"Verification updated"
                        })
                except ValueError:
                    # Invalid date format - skip
                    continue
        
        results["changes"] = recent_changes
        results["summary"]["modified"] = len(recent_changes)
        
        # Add a note about the simplified implementation
        results["note"] = (
            "This is a simplified diff showing recent verification updates. "
            "A full implementation would track detailed changes in model limits, "
            "pricing, and capabilities."
        )
        
        return results

    def _check_usage_compliance(self, catalog, provider_filter: str | None = None) -> dict[str, Any]:
        """Check if code usage complies with catalog caps."""
        # This is a simplified implementation - in reality, this would scan
        # the codebase for model usage patterns and validate against catalog limits
        
        results = {
            "compliant": True,
            "violations": [],
            "checks_performed": []
        }
        
        try:
            # Load current config to check default models
            from ...config.loader import ConfigLoader
            loader = ConfigLoader()
            config = loader.load_config()
            
            # Check if default models exist in catalog
            llm_config = getattr(config, 'llm', None)
            if llm_config:
                for provider_config in [llm_config.openai, llm_config.anthropic, llm_config.azure_openai, llm_config.bedrock]:
                    if provider_config and hasattr(provider_config, 'model'):
                        model_id = provider_config.model
                        provider_name = provider_config.__class__.__name__.lower().replace('config', '')
                        
                        # Skip if filtering by provider
                        if provider_filter and provider_name != provider_filter.lower():
                            continue
                        
                        # Check if model exists in catalog
                        entry = catalog.resolve(provider_name, model_id)
                        if not entry:
                            results["violations"].append(
                                f"Default model {provider_name}:{model_id} not found in catalog"
                            )
                            results["compliant"] = False
                        
                        results["checks_performed"].append(f"Verified {provider_name}:{model_id} exists in catalog")
            
            # Additional compliance checks could be added here:
            # - Scan for hardcoded model limits in code
            # - Check token calculator usage patterns
            # - Validate adapter configurations
            
        except Exception as e:
            results["violations"].append(f"Failed to check usage compliance: {e}")
            results["compliant"] = False
        
        return results

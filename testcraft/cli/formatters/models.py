"""Model catalog formatters for different output formats."""

import json
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table


class ModelCatalogFormatter:
    """Formatter for model catalog display."""

    def __init__(self, console: Console):
        """Initialize with console instance."""
        self.console = console

    def format_table(self, catalog_data: dict[str, Any]) -> None:
        """Display model catalog as a formatted table."""
        table = Table(
            title="Model Catalog",
            show_header=True,
            header_style="bold blue"
        )
        
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Model ID", style="green", no_wrap=True)
        
        if catalog_data["include_aliases"]:
            table.add_column("Aliases", style="yellow", max_width=20)
            
        table.add_column("Max Context", justify="right", style="white")
        table.add_column("Max Output", justify="right", style="white")
        table.add_column("Pricing ($/M)", style="bright_cyan", max_width=15)
        table.add_column("Flags", style="magenta", max_width=15)
        table.add_column("Last Verified", style="dim", max_width=12)
        
        for entry in catalog_data["models"]:
            # Format pricing
            pricing_str = "N/A"
            if entry.pricing and entry.pricing.per_million:
                pricing = entry.pricing.per_million
                if pricing.input is not None and pricing.output is not None:
                    pricing_str = f"${pricing.input:.2f}/${pricing.output:.2f}"
                elif pricing.input is not None:
                    pricing_str = f"${pricing.input:.2f}/?"
                elif pricing.output is not None:
                    pricing_str = f"?/${pricing.output:.2f}"
            
            # Format flags
            flags = []
            if entry.flags:
                if entry.flags.vision:
                    flags.append("vision")
                if entry.flags.tool_use:
                    flags.append("tools")
                if entry.flags.structured_outputs:
                    flags.append("json")
                if entry.flags.reasoning_capable:
                    flags.append("reasoning")
            
            flags_str = ", ".join(flags) if flags else "basic"
            
            # Format last verified date
            verified_str = "unknown"
            if entry.source and entry.source.last_verified:
                try:
                    verified_date = datetime.strptime(entry.source.last_verified, "%Y-%m-%d")
                    verified_str = verified_date.strftime("%Y-%m-%d")
                except ValueError:
                    verified_str = entry.source.last_verified
            
            # Format thinking tokens
            max_thinking = ""
            if entry.limits.max_thinking and entry.limits.max_thinking > 0:
                max_thinking = f" (+{entry.limits.max_thinking//1000}K thinking)"
            
            row_data = [
                entry.provider,
                entry.model_id,
            ]
            
            if catalog_data["include_aliases"]:
                aliases_str = ", ".join(entry.aliases[:3])  # Show first 3 aliases
                if len(entry.aliases) > 3:
                    aliases_str += f" (+{len(entry.aliases) - 3})"
                row_data.append(aliases_str if entry.aliases else "none")
            
            row_data.extend([
                f"{entry.limits.max_context//1000}K",
                f"{entry.limits.default_max_output//1000}K{max_thinking}",
                pricing_str,
                flags_str,
                verified_str
            ])
            
            table.add_row(*row_data)
        
        self.console.print(table)
        
        # Display summary
        summary_text = f"Total models: {catalog_data['total_models']}"
        if len(catalog_data['providers']) == 1:
            summary_text += f" (filtered by {list(catalog_data['providers'])[0]})"
        else:
            summary_text += f" across {len(catalog_data['providers'])} providers: {', '.join(sorted(catalog_data['providers']))}"
        
        self.console.print(f"\n[dim]{summary_text}[/]")

    def format_json(self, catalog_data: dict[str, Any]) -> str:
        """Format model catalog as JSON."""
        filtered_models = []
        for entry in catalog_data["models"]:
            filtered_models.append(entry.model_dump())
        
        output = {
            "version": catalog_data["catalog"].version,
            "models": filtered_models,
            "metadata": {
                "total_models": catalog_data["total_models"],
                "generated_at": datetime.now().isoformat()
            }
        }
        
        return json.dumps(output, indent=2)

    def format_yaml(self, catalog_data: dict[str, Any]) -> str:
        """Format model catalog as YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML package required for YAML output. Install with: pip install PyYAML")
        
        filtered_models = []
        for entry in catalog_data["models"]:
            filtered_models.append(entry.model_dump())
        
        output = {
            "version": catalog_data["catalog"].version,
            "models": filtered_models,
            "metadata": {
                "total_models": catalog_data["total_models"],
                "generated_at": datetime.now().isoformat()
            }
        }
        
        return yaml.dump(output, default_flow_style=False, sort_keys=False)

    def format_verification_results(self, results: dict[str, Any]) -> None:
        """Display verification results."""
        from ...adapters.io.ui_rich import RichUIAdapter
        
        ui = RichUIAdapter(self.console)
        stats = results.get("stats", {})
        
        if results["passed"]:
            ui.display_success(
                f"Catalog verification passed! Verified {stats.get('total_models', 0)} models "
                f"across {stats.get('total_providers', 0)} providers",
                "Verification Success"
            )
        else:
            ui.display_error(
                f"Catalog verification failed with {len(results['errors'])} errors",
                "Verification Failed"
            )
        
        # Display errors
        if results["errors"]:
            self.console.print("\n[bold red]Errors:[/]")
            for error in results["errors"]:
                self.console.print(f"  ❌ {error}")
        
        # Display warnings
        if results["warnings"]:
            self.console.print("\n[bold yellow]Warnings:[/]")
            for warning in results["warnings"]:
                self.console.print(f"  ⚠️  {warning}")
        
        # Display provider breakdown
        provider_stats = stats.get("provider_breakdown", {})
        if provider_stats:
            table = Table(
                title="Provider Breakdown",
                show_header=True,
                header_style="bold blue"
            )
            table.add_column("Provider", style="cyan")
            table.add_column("Models", justify="right")
            table.add_column("Issues", justify="right", style="red")
            
            for provider, provider_stat in sorted(provider_stats.items()):
                table.add_row(
                    provider,
                    str(provider_stat["models"]),
                    str(provider_stat["issues"])
                )
            
            self.console.print(table)
        
        # Display usage compliance if checked
        if "usage_compliance" in results:
            compliance = results["usage_compliance"]
            if compliance["compliant"]:
                ui.display_success("Code usage complies with catalog limits", "Usage Compliance")
            else:
                ui.display_error("Code usage violations found", "Usage Compliance Failed")
                for violation in compliance["violations"]:
                    self.console.print(f"  ❌ {violation}")

    def format_catalog_diff(self, diff_results: dict[str, Any]) -> None:
        """Display catalog diff results."""
        from ...adapters.io.ui_rich import RichUIAdapter
        
        ui = RichUIAdapter(self.console)
        since_date = datetime.fromisoformat(diff_results["since_date"])
        since_str = since_date.strftime("%Y-%m-%d")
        changes = diff_results.get("changes", [])
        summary = diff_results.get("summary", {})
        
        if not changes:
            ui.display_info(
                f"No changes found in model catalog since {since_str}",
                "No Changes"
            )
            return
        
        # Display summary
        total_changes = sum(summary.values())
        ui.display_info(
            f"Found {total_changes} changes since {since_str}",
            "Catalog Changes"
        )
        
        # Display changes in a table
        if changes:
            table = Table(
                title=f"Model Catalog Changes Since {since_str}",
                show_header=True,
                header_style="bold blue"
            )
            table.add_column("Date", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Provider", style="cyan")
            table.add_column("Model", style="white")
            table.add_column("Details", style="dim")
            
            # Sort changes by date (newest first)
            changes.sort(key=lambda x: x["date"], reverse=True)
            
            for change in changes:
                change_date = datetime.fromisoformat(change["date"]).strftime("%m-%d")
                table.add_row(
                    change_date,
                    change["type"],
                    change["provider"],
                    change["model_id"],
                    change["details"]
                )
            
            self.console.print(table)
        
        # Display note about implementation
        if "note" in diff_results:
            self.console.print(f"\n[dim]{diff_results['note']}[/]")
        
        # Display summary stats
        self.console.print(f"\n[dim]Summary: {summary.get('modified', 0)} verified, "
                         f"{summary.get('added', 0)} added, {summary.get('removed', 0)} removed[/]")

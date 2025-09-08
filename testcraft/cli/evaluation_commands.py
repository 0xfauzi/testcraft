"""
CLI commands for advanced evaluation operations including A/B testing, 
statistical analysis, and bias detection.

This module provides CLI interfaces for the advanced evaluation capabilities
implemented in subtask 20.2, following 2025 best practices for prompt
evaluation and A/B testing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.json import JSON

from ..adapters.io.rich_cli import RichCliComponents
from ..evaluation.harness import create_evaluation_harness


logger = logging.getLogger(__name__)
console = Console()
rich_cli = RichCliComponents(console)


def add_evaluation_commands(app: click.Group) -> None:
    """Add evaluation commands to the main CLI app."""
    
    @app.group(name='evaluation')
    @click.pass_context
    def evaluation_commands(ctx):
        """
        Advanced evaluation commands for A/B testing, statistical analysis, and bias detection.
        
        These commands implement 2025 best practices for LLM-as-judge evaluation,
        statistical significance testing, and bias mitigation in test generation.
        """
        pass

    @evaluation_commands.command(name='ab-test')
    @click.argument('config_file', type=click.Path(exists=True))
    @click.option('--project-root', type=click.Path(exists=True), default='.',
                  help='Project root directory')
    @click.option('--statistical-testing/--no-statistical-testing', default=True,
                  help='Enable statistical significance testing')
    @click.option('--bias-mitigation/--no-bias-mitigation', default=True,
                  help='Enable bias detection and mitigation analysis')
    @click.option('--output', '-o', type=click.Path(), 
                  help='Output file for results (JSON format)')
    @click.option('--quiet', '-q', is_flag=True, help='Suppress verbose output')
    def run_ab_test(
        config_file: str,
        project_root: str,
        statistical_testing: bool,
        bias_mitigation: bool,
        output: Optional[str],
        quiet: bool
    ):
        """
        Run comprehensive A/B testing pipeline on prompt variants.
        
        CONFIG_FILE should be a JSON file containing:
        - prompt_variants: List of prompt variants to test
        - test_dataset: Test cases to use for evaluation
        - evaluation_config: Optional evaluation configuration
        
        Example config file structure:
        {
            "prompt_variants": [
                {"id": "variant_a", "prompt": "Generate comprehensive tests..."},
                {"id": "variant_b", "prompt": "Create detailed test cases..."}
            ],
            "test_dataset": [
                {"source_file": "src/example.py", "source_content": "def add(a, b): return a + b"}
            ]
        }
        """
        try:
            if not quiet:
                console.print("🧪 [bold blue]Starting A/B Testing Pipeline[/bold blue]")
            
            # Load configuration
            config = _load_ab_test_config(config_file)
            
            # Create evaluation harness
            harness = create_evaluation_harness(project_root=Path(project_root))
            
            # Display configuration summary
            if not quiet:
                _display_ab_test_summary(config)
            
            # Run A/B testing pipeline
            with Progress(console=console) as progress:
                task = progress.add_task("Running A/B testing pipeline...", total=100)
                
                try:
                    results = harness.run_advanced_ab_testing_pipeline(
                        prompt_variants=config['prompt_variants'],
                        test_dataset=config['test_dataset'],
                        statistical_testing=statistical_testing,
                        bias_mitigation=bias_mitigation,
                        **config.get('additional_params', {})
                    )
                    progress.update(task, completed=100)
                    
                except Exception as e:
                    progress.stop()
                    console.print(f"❌ [red]A/B testing failed: {e}[/red]")
                    raise click.ClickException(f"A/B testing pipeline failed: {e}")
            
            # Display results
            if not quiet:
                _display_ab_test_results(results)
            
            # Save results if output specified
            if output:
                _save_results(results, output)
                if not quiet:
                    console.print(f"📄 Results saved to: {output}")
            
            if not quiet:
                console.print("✅ [bold green]A/B testing pipeline completed successfully![/bold green]")
        
        except Exception as e:
            console.print(f"❌ [red]Error: {e}[/red]")
            raise click.ClickException(str(e))

    @evaluation_commands.command(name='statistical-analysis')
    @click.argument('evaluation_data_file', type=click.Path(exists=True))
    @click.option('--analysis-type', default='pairwise_comparison',
                  type=click.Choice(['pairwise_comparison', 'ab_testing', 'batch_evaluation']),
                  help='Type of statistical analysis to perform')
    @click.option('--confidence-level', default=0.95, type=float,
                  help='Statistical confidence level (0.8-0.99)')
    @click.option('--project-root', type=click.Path(exists=True), default='.',
                  help='Project root directory')
    @click.option('--output', '-o', type=click.Path(), 
                  help='Output file for analysis results')
    def run_statistical_analysis(
        evaluation_data_file: str,
        analysis_type: str,
        confidence_level: float,
        project_root: str,
        output: Optional[str]
    ):
        """
        Run statistical significance analysis on evaluation data.
        
        EVALUATION_DATA_FILE should be a JSON file containing evaluation results
        from previous A/B testing runs or individual evaluations.
        """
        try:
            console.print("📊 [bold blue]Running Statistical Significance Analysis[/bold blue]")
            
            # Load evaluation data
            evaluation_data = _load_evaluation_data(evaluation_data_file)
            
            # Create evaluation harness
            harness = create_evaluation_harness(project_root=Path(project_root))
            
            # Run statistical analysis
            with Progress(console=console) as progress:
                task = progress.add_task("Analyzing statistical significance...", total=100)
                
                try:
                    results = harness.run_statistical_significance_analysis(
                        evaluation_data=evaluation_data,
                        analysis_type=analysis_type,
                        confidence_level=confidence_level
                    )
                    progress.update(task, completed=100)
                    
                except Exception as e:
                    progress.stop()
                    console.print(f"❌ [red]Statistical analysis failed: {e}[/red]")
                    raise click.ClickException(f"Statistical analysis failed: {e}")
            
            # Display results
            _display_statistical_analysis_results(results)
            
            # Save results if output specified
            if output:
                _save_results(results, output)
                console.print(f"📄 Analysis results saved to: {output}")
            
            console.print("✅ [bold green]Statistical analysis completed![/bold green]")
        
        except Exception as e:
            console.print(f"❌ [red]Error: {e}[/red]")
            raise click.ClickException(str(e))

    @evaluation_commands.command(name='bias-detection')
    @click.option('--project-root', type=click.Path(exists=True), default='.',
                  help='Project root directory')
    @click.option('--time-window', default=30, type=int,
                  help='Time window in days for analysis')
    @click.option('--evaluation-history', type=click.Path(exists=True),
                  help='JSON file with evaluation history (optional)')
    @click.option('--bias-types', multiple=True,
                  help='Specific bias types to check (can be used multiple times)')
    @click.option('--output', '-o', type=click.Path(), 
                  help='Output file for bias analysis results')
    def run_bias_detection(
        project_root: str,
        time_window: int,
        evaluation_history: Optional[str],
        bias_types: Tuple[str, ...],
        output: Optional[str]
    ):
        """
        Detect and analyze evaluation bias patterns.
        
        This analyzes evaluation consistency, detects systematic biases,
        and provides mitigation recommendations following 2025 best practices.
        """
        try:
            console.print("🔍 [bold blue]Running Bias Detection Analysis[/bold blue]")
            
            # Create evaluation harness
            harness = create_evaluation_harness(project_root=Path(project_root))
            
            # Load evaluation history if provided
            history_data = None
            if evaluation_history:
                history_data = _load_evaluation_history(evaluation_history)
            
            # Convert bias types to list
            bias_types_list = list(bias_types) if bias_types else None
            
            # Run bias detection
            with Progress(console=console) as progress:
                task = progress.add_task("Detecting evaluation biases...", total=100)
                
                try:
                    results = harness.detect_evaluation_bias(
                        evaluation_history=history_data,
                        bias_types=bias_types_list,
                        time_window_days=time_window
                    )
                    progress.update(task, completed=100)
                    
                except Exception as e:
                    progress.stop()
                    console.print(f"❌ [red]Bias detection failed: {e}[/red]")
                    raise click.ClickException(f"Bias detection failed: {e}")
            
            # Display results
            _display_bias_analysis_results(results)
            
            # Save results if output specified
            if output:
                _save_results(results, output)
                console.print(f"📄 Bias analysis results saved to: {output}")
            
            console.print("✅ [bold green]Bias detection analysis completed![/bold green]")
        
        except Exception as e:
            console.print(f"❌ [red]Error: {e}[/red]")
            raise click.ClickException(str(e))

    @evaluation_commands.command(name='campaign')
    @click.argument('campaign_config_file', type=click.Path(exists=True))
    @click.option('--project-root', type=click.Path(exists=True), default='.',
                  help='Project root directory')
    @click.option('--output', '-o', type=click.Path(), 
                  help='Output file for campaign results')
    @click.option('--verbose', '-v', is_flag=True, 
                  help='Enable verbose output')
    def run_evaluation_campaign(
        campaign_config_file: str,
        project_root: str,
        output: Optional[str],
        verbose: bool
    ):
        """
        Run comprehensive evaluation campaign with multiple scenarios.
        
        CAMPAIGN_CONFIG_FILE should contain multiple evaluation scenarios
        with cross-scenario analysis and final recommendations.
        """
        try:
            console.print("🚀 [bold blue]Starting Comprehensive Evaluation Campaign[/bold blue]")
            
            # Load campaign configuration
            campaign_config = _load_campaign_config(campaign_config_file)
            
            # Create evaluation harness
            harness = create_evaluation_harness(project_root=Path(project_root))
            
            # Display campaign summary
            if verbose:
                _display_campaign_summary(campaign_config)
            
            # Run evaluation campaign
            with Progress(console=console) as progress:
                scenarios = campaign_config.get('evaluation_scenarios', [])
                main_task = progress.add_task("Running evaluation campaign...", total=len(scenarios))
                
                try:
                    results = harness.run_comprehensive_evaluation_campaign(
                        campaign_config=campaign_config
                    )
                    progress.update(main_task, completed=len(scenarios))
                    
                except Exception as e:
                    progress.stop()
                    console.print(f"❌ [red]Evaluation campaign failed: {e}[/red]")
                    raise click.ClickException(f"Campaign failed: {e}")
            
            # Display results
            _display_campaign_results(results, verbose=verbose)
            
            # Save results if output specified
            if output:
                _save_results(results, output)
                console.print(f"📄 Campaign results saved to: {output}")
            
            console.print("✅ [bold green]Evaluation campaign completed successfully![/bold green]")
        
        except Exception as e:
            console.print(f"❌ [red]Error: {e}[/red]")
            raise click.ClickException(str(e))


# =============================
# Helper Functions for CLI Operations
# =============================

def _load_ab_test_config(config_file: str) -> Dict[str, Any]:
    """Load A/B test configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['prompt_variants', 'test_dataset']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        return config
    except Exception as e:
        raise click.ClickException(f"Failed to load config file: {e}")


def _load_evaluation_data(data_file: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSON file."""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Evaluation data must be a list of evaluation results")
        
        return data
    except Exception as e:
        raise click.ClickException(f"Failed to load evaluation data: {e}")


def _load_evaluation_history(history_file: str) -> List[Dict[str, Any]]:
    """Load evaluation history from JSON file."""
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if not isinstance(history, list):
            raise ValueError("Evaluation history must be a list of evaluation results")
        
        return history
    except Exception as e:
        raise click.ClickException(f"Failed to load evaluation history: {e}")


def _load_campaign_config(config_file: str) -> Dict[str, Any]:
    """Load campaign configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        if 'evaluation_scenarios' not in config:
            raise ValueError("Missing required field: evaluation_scenarios")
        
        return config
    except Exception as e:
        raise click.ClickException(f"Failed to load campaign config: {e}")


def _save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        raise click.ClickException(f"Failed to save results: {e}")


def _display_ab_test_summary(config: Dict[str, Any]) -> None:
    """Display A/B test configuration summary."""
    variants = config['prompt_variants']
    test_cases = config['test_dataset']
    
    table = Table(title="A/B Test Configuration")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Prompt Variants", str(len(variants)))
    table.add_row("Test Cases", str(len(test_cases)))
    table.add_row("Total Evaluations", str(len(variants) * len(test_cases)))
    
    console.print(table)
    
    # Display variant details
    variant_table = Table(title="Prompt Variants")
    variant_table.add_column("ID", style="yellow")
    variant_table.add_column("Prompt Preview", style="white")
    
    for variant in variants:
        prompt_preview = variant.get('prompt', '')[:80] + ('...' if len(variant.get('prompt', '')) > 80 else '')
        variant_table.add_row(variant.get('id', 'unknown'), prompt_preview)
    
    console.print(variant_table)


def _display_ab_test_results(results: Dict[str, Any]) -> None:
    """Display A/B test results in a formatted way."""
    metadata = results.get('metadata', {})
    
    # Summary panel
    summary_text = f"""
📊 **Evaluation Summary:**
• Variants: {metadata.get('variant_count', 0)}
• Test Cases: {metadata.get('test_case_count', 0)}
• Statistical Testing: {'✅' if metadata.get('statistical_testing_enabled') else '❌'}
• Bias Mitigation: {'✅' if metadata.get('bias_mitigation_enabled') else '❌'}
    """
    
    console.print(Panel(summary_text, title="A/B Testing Results", border_style="blue"))
    
    # Variant performance table
    variant_evals = results.get('variant_evaluations', [])
    if variant_evals:
        perf_table = Table(title="Variant Performance")
        perf_table.add_column("Variant ID", style="yellow")
        perf_table.add_column("Acceptance Rate", style="green")
        perf_table.add_column("Mean LLM Score", style="blue")
        perf_table.add_column("Total Tests", style="white")
        
        for variant_eval in variant_evals:
            stats = variant_eval.get('summary_stats', {})
            perf_table.add_row(
                variant_eval.get('variant_id', 'unknown'),
                f"{stats.get('acceptance_rate', 0):.2%}",
                f"{stats.get('mean_llm_score', 0):.2f}",
                str(stats.get('total_tests', 0))
            )
        
        console.print(perf_table)
    
    # Statistical analysis results
    stat_analysis = results.get('statistical_analysis')
    if stat_analysis and not stat_analysis.get('error'):
        stat_text = f"""
📈 **Statistical Analysis:**
• Test: {stat_analysis.get('statistical_test', 'unknown')}
• P-value: {stat_analysis.get('p_value', 0):.4f}
• Significance: {stat_analysis.get('significance_assessment', 'unknown')}
• Effect Size: {stat_analysis.get('effect_size', {}).get('interpretation', 'unknown')}
        """
        console.print(Panel(stat_text, title="Statistical Analysis", border_style="green"))
    
    # Bias analysis results
    bias_analysis = results.get('bias_analysis')
    if bias_analysis and not bias_analysis.get('error'):
        fairness_score = bias_analysis.get('fairness_score', 0)
        bias_text = f"""
🔍 **Bias Analysis:**
• Fairness Score: {fairness_score:.2f}
• Detected Biases: {len(bias_analysis.get('bias_analysis', {}).get('detected_biases', []))}
• Consistency Score: {bias_analysis.get('evaluation_consistency', {}).get('consistency_score', 0):.2f}
        """
        console.print(Panel(bias_text, title="Bias Analysis", border_style="yellow"))
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        rec_text = "\n".join(f"• {rec}" for rec in recommendations)
        console.print(Panel(rec_text, title="Recommendations", border_style="cyan"))


def _display_statistical_analysis_results(results: Dict[str, Any]) -> None:
    """Display statistical analysis results."""
    if results.get('error'):
        console.print(f"❌ [red]Statistical analysis error: {results['error']}[/red]")
        return
    
    # Main results
    main_text = f"""
📊 **Statistical Test Results:**
• Test Type: {results.get('statistical_test', 'unknown')}
• P-value: {results.get('p_value', 0):.6f}
• Significance: {results.get('significance_assessment', 'unknown')}

📏 **Effect Size:**
• Cohen's d: {results.get('effect_size', {}).get('cohens_d', 0):.3f}
• Interpretation: {results.get('effect_size', {}).get('interpretation', 'unknown')}

📈 **Confidence Interval:**
• Lower: {results.get('confidence_interval', {}).get('lower', 0):.3f}
• Upper: {results.get('confidence_interval', {}).get('upper', 0):.3f}
• Level: {results.get('confidence_interval', {}).get('confidence_level', 0):.0%}
    """
    
    console.print(Panel(main_text, title="Statistical Analysis Results", border_style="blue"))
    
    # Sample adequacy
    sample_info = results.get('sample_adequacy', {})
    sample_text = f"""
📏 **Sample Adequacy:**
• Current Sample: {sample_info.get('current_sample_size', 0)}
• Recommended: {sample_info.get('recommended_minimum', 0)}
• Power Achieved: {sample_info.get('power_achieved', 0):.2%}
    """
    
    console.print(Panel(sample_text, title="Sample Adequacy", border_style="green"))
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        rec_text = "\n".join(f"• {rec}" for rec in recommendations)
        console.print(Panel(rec_text, title="Recommendations", border_style="cyan"))


def _display_bias_analysis_results(results: Dict[str, Any]) -> None:
    """Display bias analysis results."""
    if results.get('error'):
        console.print(f"❌ [red]Bias analysis error: {results['error']}[/red]")
        return
    
    # Main bias analysis
    bias_data = results.get('bias_analysis', {})
    detected_biases = bias_data.get('detected_biases', [])
    
    bias_text = f"""
🔍 **Bias Detection:**
• Fairness Score: {results.get('fairness_score', 0):.2f}/1.0
• Detected Biases: {len(detected_biases)}
• Analysis Confidence: {bias_data.get('confidence', 0):.2f}
    """
    
    if detected_biases:
        bias_text += "\n\n📋 **Detected Bias Types:**\n"
        severity_map = bias_data.get('bias_severity', {})
        for bias in detected_biases:
            severity = severity_map.get(bias, 'unknown')
            bias_text += f"• {bias}: {severity} severity\n"
    
    console.print(Panel(bias_text, title="Bias Analysis Results", border_style="yellow"))
    
    # Consistency analysis
    consistency = results.get('evaluation_consistency', {})
    consistency_text = f"""
📊 **Evaluation Consistency:**
• Consistency Score: {consistency.get('consistency_score', 0):.2f}
• Drift Detected: {'Yes' if consistency.get('drift_detected') else 'No'}
• Variance Analysis: {consistency.get('variance_analysis', 'N/A')}
    """
    
    console.print(Panel(consistency_text, title="Consistency Analysis", border_style="green"))
    
    # Mitigation recommendations
    mitigation = results.get('mitigation_recommendations', {})
    immediate_actions = mitigation.get('immediate_actions', [])
    process_improvements = mitigation.get('process_improvements', [])
    
    if immediate_actions or process_improvements:
        rec_text = ""
        if immediate_actions:
            rec_text += "🚨 **Immediate Actions:**\n"
            rec_text += "\n".join(f"• {action}" for action in immediate_actions)
        
        if process_improvements:
            if rec_text:
                rec_text += "\n\n"
            rec_text += "🔧 **Process Improvements:**\n" 
            rec_text += "\n".join(f"• {improvement}" for improvement in process_improvements)
        
        console.print(Panel(rec_text, title="Mitigation Recommendations", border_style="cyan"))


def _display_campaign_summary(config: Dict[str, Any]) -> None:
    """Display campaign configuration summary."""
    scenarios = config.get('evaluation_scenarios', [])
    
    table = Table(title="Campaign Configuration")
    table.add_column("Scenario", style="cyan")
    table.add_column("Variants", style="yellow")
    table.add_column("Test Cases", style="green")
    table.add_column("Description", style="white")
    
    for i, scenario in enumerate(scenarios):
        table.add_row(
            scenario.get('name', f'Scenario {i+1}'),
            str(len(scenario.get('prompt_variants', []))),
            str(len(scenario.get('test_dataset', []))),
            scenario.get('description', 'No description')[:50] + ('...' if len(scenario.get('description', '')) > 50 else '')
        )
    
    console.print(table)


def _display_campaign_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """Display campaign results."""
    metadata = results.get('campaign_metadata', {})
    scenario_results = results.get('scenario_results', [])
    
    # Campaign summary
    summary_text = f"""
🚀 **Campaign Summary:**
• Total Scenarios: {len(scenario_results)}
• Campaign Timestamp: {metadata.get('timestamp', 'unknown')}
• Harness Version: {metadata.get('harness_version', 'unknown')}
    """
    
    console.print(Panel(summary_text, title="Campaign Results", border_style="blue"))
    
    # Scenario results table
    if scenario_results:
        scenario_table = Table(title="Scenario Results")
        scenario_table.add_column("Scenario", style="cyan")
        scenario_table.add_column("Status", style="green") 
        scenario_table.add_column("Variants", style="yellow")
        scenario_table.add_column("Statistical Sig.", style="blue")
        scenario_table.add_column("Fairness Score", style="magenta")
        
        for scenario in scenario_results:
            scenario_meta = scenario.get('scenario_metadata', {})
            
            # Determine status
            status = "✅ Success" if not scenario.get('error') else "❌ Failed"
            
            # Get statistical significance
            stat_analysis = scenario.get('statistical_analysis', {})
            significance = stat_analysis.get('significance_assessment', 'N/A') if not stat_analysis.get('error') else 'Error'
            
            # Get fairness score
            bias_analysis = scenario.get('bias_analysis', {})
            fairness = f"{bias_analysis.get('fairness_score', 0):.2f}" if not bias_analysis.get('error') else 'Error'
            
            scenario_table.add_row(
                scenario_meta.get('name', 'Unknown'),
                status,
                str(len(scenario.get('variant_evaluations', []))),
                significance,
                fairness
            )
        
        console.print(scenario_table)
    
    # Cross-scenario analysis
    cross_analysis = results.get('cross_scenario_analysis')
    if cross_analysis:
        insights = cross_analysis.get('insights', [])
        if insights:
            insight_text = "\n".join(f"• {insight}" for insight in insights)
            console.print(Panel(insight_text, title="Cross-Scenario Insights", border_style="green"))
    
    # Final recommendations
    recommendations = results.get('final_recommendations', [])
    if recommendations:
        rec_text = "\n".join(f"• {rec}" for rec in recommendations)
        console.print(Panel(rec_text, title="Final Recommendations", border_style="cyan"))
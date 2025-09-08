# TestCraft Advanced Usage Guide

This guide covers advanced TestCraft features, including the evaluation harness, A/B testing, statistical analysis, and best practices for optimizing test generation.

## Table of Contents

1. [Evaluation Harness](#evaluation-harness)
2. [A/B Testing and Prompt Optimization](#ab-testing-and-prompt-optimization)
3. [Statistical Analysis](#statistical-analysis)
4. [Bias Detection and Mitigation](#bias-detection-and-mitigation)
5. [Golden Repository Testing](#golden-repository-testing)
6. [Comprehensive Evaluation Campaigns](#comprehensive-evaluation-campaigns)
7. [Configuration Management](#configuration-management)
8. [Integration Patterns](#integration-patterns)

## Evaluation Harness

The TestCraft evaluation harness provides comprehensive test quality assessment using multiple evaluation methods.

### Core Evaluation Types

#### 1. Automated Acceptance Checks

Automated checks verify that generated tests meet basic quality criteria:

```python
from evaluation.harness import create_evaluation_harness

harness = create_evaluation_harness()

result = harness.evaluate_single_test(
    test_content=generated_test_code,
    source_file="src/example.py"
)

print(f"Syntactically valid: {result.acceptance.syntactically_valid}")
print(f"Imports successfully: {result.acceptance.imports_successfully}")
print(f"Pytest passes: {result.acceptance.pytest_passes}")
print(f"Coverage improvement: {result.acceptance.coverage_improvement}%")
```

#### 2. LLM-as-Judge Evaluation

Rubric-driven evaluation using LLMs to assess test quality across multiple dimensions:

```python
from testcraft.ports.evaluation_port import EvaluationConfig

config = EvaluationConfig(
    llm_judge_enabled=True,
    rubric_dimensions=["correctness", "coverage", "clarity", "safety", "maintainability"]
)

result = harness.evaluate_single_test(
    test_content=test_code,
    source_file="src/example.py",
    config=config
)

# Access LLM scores and rationales
llm_result = result.llm_judge
for dimension, score in llm_result.scores.items():
    rationale = llm_result.rationales[dimension]
    print(f"{dimension}: {score}/5.0 - {rationale}")
```

### Batch Evaluation

Evaluate multiple test variants efficiently:

```python
test_variants = [
    {"test_id": "variant_1", "content": test_code_1},
    {"test_id": "variant_2", "content": test_code_2},
    {"test_id": "variant_3", "content": test_code_3}
]

source_files = ["src/example.py"] * len(test_variants)

results = harness.evaluate_test_batch(
    test_variants=test_variants,
    source_files=source_files,
    config=config
)

for result in results:
    print(f"{result.test_id}: Overall score {result.llm_judge.overall_score}")
```

## A/B Testing and Prompt Optimization

### Setting Up A/B Testing

Create a configuration file for A/B testing different prompt variants:

```json
{
  "prompt_variants": [
    {
      "id": "comprehensive_v1",
      "prompt": "Generate comprehensive pytest tests with fixtures and edge cases for the following Python code:\n\n{source_code}\n\nInclude tests for:\n- Normal operation\n- Edge cases\n- Error conditions\n- Type validation"
    },
    {
      "id": "focused_v1", 
      "prompt": "Create focused pytest tests for the following code, emphasizing correctness and clarity:\n\n{source_code}\n\nPrioritize:\n- Core functionality testing\n- Clear test names\n- Minimal setup"
    }
  ],
  "test_dataset": [
    {
      "source_file": "examples/calculator.py",
      "source_content": "def add(a: int, b: int) -> int:\n    return a + b\n\ndef divide(a: int, b: int) -> float:\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b"
    }
  ]
}
```

### Running A/B Tests via CLI

```bash
# Basic A/B testing
testcraft evaluation ab-test ab_config.json

# With statistical analysis and bias mitigation
testcraft evaluation ab-test ab_config.json \
  --statistical-testing \
  --bias-mitigation \
  --output results.json

# Quiet mode for automated pipelines
testcraft evaluation ab-test ab_config.json --quiet --output results.json
```

### Programmatic A/B Testing

```python
from evaluation.harness import create_evaluation_harness
import json

# Load configuration
with open('ab_config.json') as f:
    config = json.load(f)

harness = create_evaluation_harness()

results = harness.run_advanced_ab_testing_pipeline(
    prompt_variants=config['prompt_variants'],
    test_dataset=config['test_dataset'],
    statistical_testing=True,
    bias_mitigation=True
)

# Access results
print(f"Best performing variant: {results['best_variant']}")
print(f"Statistical significance: {results['statistical_analysis']['significance_assessment']}")
print(f"Fairness score: {results['bias_analysis']['fairness_score']}")
```

### Understanding A/B Test Results

Results include comprehensive analysis:

```python
# Variant performance comparison
for variant_eval in results['variant_evaluations']:
    variant_id = variant_eval['variant_id']
    stats = variant_eval['summary_stats']
    
    print(f"\nVariant: {variant_id}")
    print(f"  Acceptance Rate: {stats['acceptance_rate']:.2%}")
    print(f"  Mean LLM Score: {stats['mean_llm_score']:.2f}")
    print(f"  Coverage Improvement: {stats['mean_coverage_improvement']:.1f}%")

# Statistical analysis
stat_analysis = results['statistical_analysis']
print(f"\nStatistical Test: {stat_analysis['statistical_test']}")
print(f"P-value: {stat_analysis['p_value']:.6f}")
print(f"Effect Size: {stat_analysis['effect_size']['interpretation']}")
print(f"Confidence: {stat_analysis['confidence_interval']}")
```

## Statistical Analysis

### Standalone Statistical Analysis

Run statistical analysis on existing evaluation data:

```bash
# Pairwise comparison analysis
testcraft evaluation statistical-analysis evaluation_data.json \
  --analysis-type pairwise_comparison \
  --confidence-level 0.95

# Batch evaluation analysis  
testcraft evaluation statistical-analysis batch_results.json \
  --analysis-type batch_evaluation \
  --confidence-level 0.99
```

### Statistical Analysis Types

1. **Pairwise Comparison**: Compare two variants directly
2. **A/B Testing**: Compare multiple variants with control
3. **Batch Evaluation**: Analyze batch evaluation results

### Interpreting Statistical Results

```python
results = harness.run_statistical_significance_analysis(
    evaluation_data=evaluation_data,
    analysis_type="pairwise_comparison", 
    confidence_level=0.95
)

# Statistical significance
significance = results['significance_assessment']  # 'significant', 'not_significant', 'highly_significant'

# Effect size interpretation
effect_size = results['effect_size']['interpretation']  # 'small', 'medium', 'large'

# Sample adequacy
sample_info = results['sample_adequacy']
is_adequate = sample_info['current_sample_size'] >= sample_info['recommended_minimum']
```

## Bias Detection and Mitigation

### Running Bias Detection

```bash
# Analyze recent evaluation history
testcraft evaluation bias-detection --time-window 30

# Analyze specific bias types
testcraft evaluation bias-detection \
  --bias-types "prompt_length_bias" \
  --bias-types "complexity_bias" \
  --time-window 14

# Use external evaluation history
testcraft evaluation bias-detection \
  --evaluation-history evaluation_history.json \
  --output bias_report.json
```

### Programmatic Bias Detection

```python
# Detect bias patterns
bias_results = harness.detect_evaluation_bias(
    time_window_days=30,
    bias_types=["prompt_length_bias", "complexity_bias", "domain_bias"]
)

# Analyze results
fairness_score = bias_results['fairness_score']  # 0.0 to 1.0
detected_biases = bias_results['bias_analysis']['detected_biases']
mitigation_actions = bias_results['mitigation_recommendations']['immediate_actions']

print(f"Fairness Score: {fairness_score:.2f}")
for bias_type in detected_biases:
    severity = bias_results['bias_analysis']['bias_severity'][bias_type]
    print(f"Detected: {bias_type} (severity: {severity})")
```

### Bias Mitigation Strategies

Implement recommended mitigation strategies:

```python
# Example mitigation actions based on bias analysis
mitigation_recs = bias_results['mitigation_recommendations']

# Immediate actions
for action in mitigation_recs['immediate_actions']:
    print(f"🚨 Action required: {action}")

# Process improvements  
for improvement in mitigation_recs['process_improvements']:
    print(f"🔧 Process improvement: {improvement}")
```

## Golden Repository Testing

Golden repository testing enables regression detection against known-good test suites.

### Setting Up Golden Repositories

```python
from pathlib import Path

golden_repo_path = Path("golden_repos/example_project")

# Define test generator function
def test_generator(source_file: str) -> str:
    # Your test generation logic here
    return generated_test_code

# Run golden repository evaluation
golden_results = harness.run_golden_repository_evaluation(
    golden_repo_path=golden_repo_path,
    test_generator=test_generator,
    config=config
)
```

### Golden Repository Configuration

```toml
[evaluation]
golden_repos_path = "golden_repos"

# Golden repository specific settings
[evaluation.golden_repo]
acceptance_threshold = 0.8
regression_detection = true
comparison_mode = "strict"
```

### Interpreting Golden Repository Results

```python
# Regression analysis
regression_info = golden_results['regression_analysis']
regression_detected = regression_info['regression_detected']
affected_modules = regression_info['affected_modules']

# Quality comparison
quality_delta = golden_results['quality_comparison']['quality_delta']
if quality_delta < -0.1:  # Quality decreased
    print("⚠️  Quality regression detected")
elif quality_delta > 0.1:  # Quality improved
    print("✅ Quality improvement detected")
```

## Comprehensive Evaluation Campaigns

Evaluation campaigns orchestrate multiple testing scenarios for comprehensive analysis.

### Campaign Configuration

```json
{
  "evaluation_scenarios": [
    {
      "name": "simple_functions",
      "description": "Testing on simple utility functions",
      "prompt_variants": [...],
      "test_dataset": [...],
      "statistical_testing": true,
      "bias_mitigation": true
    },
    {
      "name": "complex_classes", 
      "description": "Testing on complex class hierarchies",
      "prompt_variants": [...],
      "test_dataset": [...],
      "statistical_testing": true,
      "bias_mitigation": true
    }
  ],
  "analysis_options": {
    "cross_scenario_analysis": true,
    "final_recommendations": true
  }
}
```

### Running Campaigns

```bash
# Run comprehensive campaign
testcraft evaluation campaign campaign_config.json --verbose

# Save detailed results
testcraft evaluation campaign campaign_config.json \
  --output campaign_results.json \
  --verbose
```

### Campaign Results Analysis

```python
# Cross-scenario insights
cross_analysis = campaign_results['cross_scenario_analysis']
consistency_analysis = cross_analysis['consistency_analysis']

# Find most consistent variant across scenarios
for variant_id, data in consistency_analysis.items():
    consistency_rating = data['consistency_rating']  # 'high', 'medium', 'low'
    mean_performance = data['mean_performance']
    
    if consistency_rating == 'high':
        print(f"✅ {variant_id}: Consistent high performance ({mean_performance:.2f})")

# Final recommendations
for rec in campaign_results['final_recommendations']:
    print(f"💡 {rec}")
```

## Configuration Management

### Environment-Specific Configuration

```toml
# Development configuration
[llm]
default_provider = "openai"
openai_model = "gpt-3.5-turbo"  # Cheaper for development

[evaluation]
enabled = false  # Disable for faster development

# Production configuration  
[llm]
default_provider = "anthropic"
anthropic_model = "claude-3-sonnet-20240229"

[evaluation] 
enabled = true
llm_judge_enabled = true
statistical_testing = true
```

### Dynamic Configuration

```python
from testcraft.config import TestCraftConfig
from testcraft.config.loader import load_config

# Load base config
config = load_config("base_config.toml")

# Override for specific use case
config.evaluation.llm_judge_enabled = True
config.evaluation.rubric_dimensions = ["correctness", "coverage", "maintainability"]

# Create harness with custom config
harness = TestEvaluationHarness(config=config)
```

### Configuration Validation

```python
from testcraft.config.loader import ConfigLoader, ConfigurationError

try:
    config = ConfigLoader.load_config("invalid_config.toml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
```

## Integration Patterns

### CI/CD Integration

```yaml
# .github/workflows/evaluation.yml
name: Test Generation Evaluation

on:
  pull_request:
    paths: ['prompts/**', 'testcraft/**']

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install TestCraft
        run: |
          pip install -e .
          
      - name: Run A/B Testing
        run: |
          testcraft evaluation ab-test ci_ab_config.json \
            --quiet --output evaluation_results.json
            
      - name: Check for Regressions
        run: |
          python scripts/check_evaluation_regressions.py evaluation_results.json
```

### Custom Evaluation Metrics

```python
from testcraft.ports.evaluation_port import EvaluationPort
from testcraft.adapters.evaluation.main_adapter import TestcraftEvaluationAdapter

class CustomEvaluationAdapter(TestcraftEvaluationAdapter):
    """Custom evaluation adapter with domain-specific metrics."""
    
    def evaluate_single(self, test_content: str, source_file: str, config) -> EvaluationResult:
        # Run base evaluation
        result = super().evaluate_single(test_content, source_file, config)
        
        # Add custom metrics
        custom_metrics = self._calculate_custom_metrics(test_content, source_file)
        result.metadata['custom_metrics'] = custom_metrics
        
        return result
    
    def _calculate_custom_metrics(self, test_content: str, source_file: str) -> dict:
        """Calculate domain-specific quality metrics."""
        return {
            'business_logic_coverage': self._assess_business_logic_coverage(test_content),
            'integration_test_ratio': self._calculate_integration_ratio(test_content),
            'documentation_quality': self._assess_documentation(test_content)
        }
```

### Monitoring and Alerting

```python
import logging
from testcraft.adapters.telemetry.router import TelemetryRouter

# Setup telemetry
telemetry = TelemetryRouter.from_config(config.telemetry)

# Monitor evaluation quality trends
def monitor_evaluation_trends():
    recent_evaluations = harness.analyze_evaluation_history(time_window_days=7)
    
    quality_trend = recent_evaluations['quality_trend']
    if quality_trend['direction'] == 'decreasing':
        alert_message = f"Quality trend decreasing: {quality_trend['change_rate']:.2%} per day"
        
        # Send alert via telemetry
        telemetry.record_metric('evaluation_quality_alert', 1, tags={
            'trend': 'decreasing',
            'change_rate': quality_trend['change_rate']
        })
        
        logging.warning(alert_message)
```

## Best Practices

### 1. Evaluation Strategy

- **Start Simple**: Begin with acceptance checks before adding LLM-as-judge
- **Progressive Enhancement**: Add statistical testing and bias detection gradually
- **Regular Monitoring**: Run bias detection weekly, campaigns monthly
- **Baseline Establishment**: Establish quality baselines with golden repositories

### 2. A/B Testing

- **Sufficient Sample Size**: Ensure adequate statistical power
- **Control for Variables**: Keep test datasets consistent across variants
- **Multiple Dimensions**: Evaluate across correctness, coverage, clarity, and maintainability
- **Document Changes**: Track prompt modifications and their impact

### 3. Configuration Management

- **Environment Separation**: Use different configs for dev/staging/production
- **Version Control**: Track configuration changes with code
- **Validation**: Always validate configuration before deployment
- **Documentation**: Document configuration decisions and trade-offs

### 4. Performance Optimization

- **Batch Processing**: Use batch evaluation for efficiency
- **Caching**: Cache LLM responses where appropriate
- **Parallel Processing**: Run evaluations in parallel when possible
- **Resource Monitoring**: Monitor LLM API usage and costs

---

For more specific implementation details, see the [Configuration Reference](configuration.md) and [Architecture Guide](architecture.md).

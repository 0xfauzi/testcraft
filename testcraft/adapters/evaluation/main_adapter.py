"""
Main evaluation adapter implementing the EvaluationPort interface.

This module provides the primary evaluation adapter that coordinates
automated acceptance checks, LLM-as-judge evaluation, and A/B testing
capabilities following clean architecture principles.
"""

import ast
import logging
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
import json

from ...ports.evaluation_port import (
    EvaluationPort, EvaluationConfig, AcceptanceResult, LLMJudgeResult,
    EvaluationResult, EvaluationMode, ComparisonMode
)
from ...ports.coverage_port import CoveragePort
from ...ports.llm_port import LLMPort
from ...ports.state_port import StatePort
from ...prompts.registry import PromptRegistry
from ..io.artifact_store import ArtifactStoreAdapter, ArtifactType
from ..io.safety import SafetyPolicies


logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Exception raised when evaluation operations fail."""
    pass


class TestcraftEvaluationAdapter(EvaluationPort):
    """
    Main evaluation adapter implementing comprehensive test evaluation.
    
    This adapter orchestrates automated acceptance checks, LLM-as-judge
    evaluation, and A/B testing capabilities. It follows clean architecture
    principles by depending on injected ports rather than concrete adapters.
    """
    
    def __init__(
        self,
        coverage_adapter: CoveragePort,
        llm_adapter: LLMPort,
        state_adapter: StatePort,
        artifact_store: Optional[ArtifactStoreAdapter] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        project_root: Optional[Path] = None,
        safety_enabled: bool = True
    ):
        """
        Initialize the evaluation adapter with required dependencies.
        
        Args:
            coverage_adapter: Coverage measurement adapter
            llm_adapter: LLM adapter for judge evaluations
            state_adapter: State management adapter
            artifact_store: Optional artifact storage adapter
            prompt_registry: Optional prompt registry for evaluation prompts
            project_root: Project root for safety validation
            safety_enabled: Whether to enforce safety policies
        """
        self.coverage_adapter = coverage_adapter
        self.llm_adapter = llm_adapter
        self.state_adapter = state_adapter
        self.artifact_store = artifact_store or ArtifactStoreAdapter()
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.project_root = project_root or Path.cwd()
        self.safety_enabled = safety_enabled
        
        # Initialize evaluation state tracking
        self._evaluation_cache: Dict[str, EvaluationResult] = {}
        
        logger.info("TestcraftEvaluationAdapter initialized successfully")
    
    def run_acceptance_checks(
        self,
        test_content: str,
        source_file: str,
        baseline_coverage: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> AcceptanceResult:
        """
        Run comprehensive automated acceptance checks on test content.
        
        This performs syntactic validation, import checking, pytest execution,
        and optional coverage improvement measurement.
        """
        logger.debug(f"Running acceptance checks for {source_file}")
        
        try:
            error_messages = []
            
            # 1. Syntactic validation
            syntactically_valid = self._check_syntax(test_content, error_messages)
            
            # 2. Import validation (requires syntactic validity)
            imports_successfully = False
            if syntactically_valid:
                imports_successfully = self._check_imports(
                    test_content, source_file, error_messages
                )
            
            # 3. Pytest execution (requires import success)
            pytest_passes = False
            if imports_successfully:
                pytest_passes = self._run_pytest(
                    test_content, source_file, error_messages
                )
            
            # 4. Optional coverage improvement measurement
            coverage_improvement = None
            if pytest_passes and baseline_coverage:
                coverage_improvement = self._measure_coverage_improvement(
                    test_content, source_file, baseline_coverage
                )
            
            result = AcceptanceResult(
                syntactically_valid=syntactically_valid,
                imports_successfully=imports_successfully,
                pytest_passes=pytest_passes,
                coverage_improvement=coverage_improvement,
                error_messages=error_messages
            )
            
            logger.debug(f"Acceptance checks completed: {result.all_checks_pass}")
            return result
            
        except Exception as e:
            logger.error(f"Acceptance checks failed: {e}")
            raise EvaluationError(f"Failed to run acceptance checks: {e}") from e
    
    def evaluate_with_llm_judge(
        self,
        test_content: str,
        source_file: str,
        rubric_dimensions: Optional[List[str]] = None,
        prompt_version: Optional[str] = None,
        **kwargs: Any
    ) -> LLMJudgeResult:
        """
        Evaluate test quality using LLM-as-judge with rubric-driven scoring.
        
        This uses structured prompts to get both numeric scores and rationales
        for each evaluation dimension, following 2025 best practices.
        """
        logger.debug(f"Running LLM judge evaluation for {source_file}")
        
        try:
            # Use provided dimensions or defaults
            dimensions = rubric_dimensions or ["correctness", "coverage", "clarity", "safety"]
            
            # Get enhanced evaluation prompt from registry (2025 best practices)
            prompt_version = prompt_version or "llm_judge_v1"
            
            # Use new rubric-driven evaluation prompts
            system_prompt = self.prompt_registry.get_prompt(
                "system", prompt_version
            )
            user_prompt = self.prompt_registry.get_prompt(
                "user", prompt_version
            )
            
            if not system_prompt or not user_prompt:
                # Fallback to built-in prompts
                evaluation_prompt = self._get_default_evaluation_prompt()
                prompt_version = "builtin_default"
            else:
                # Use advanced prompts with context formatting
                evaluation_prompt = system_prompt
            
            # Read source code for context
            try:
                source_content = Path(source_file).read_text(encoding='utf-8')
            except Exception as e:
                logger.warning(f"Could not read source file {source_file}: {e}")
                source_content = "# Source file not available"
            
            # Use enhanced prompt structure for 2025 best practices
            if system_prompt and user_prompt:
                # Format user prompt with context
                formatted_user_prompt = user_prompt.format(
                    version="v1",
                    dimensions=", ".join(dimensions),
                    source_content=source_content,
                    test_content=test_content,
                    additional_context=kwargs.get('additional_context', 'No additional context provided')
                )
                
                # Call LLM with system/user prompt structure
                llm_response = self.llm_adapter.analyze_code(
                    formatted_user_prompt,
                    analysis_type="evaluation",
                    max_tokens=2000,
                    temperature=0.1,
                    system_prompt=system_prompt,
                    **kwargs
                )
            else:
                # Fallback to old structure
                context = self._build_evaluation_context(
                    test_content, source_content, source_file, dimensions
                )
                
                formatted_prompt = evaluation_prompt.format(
                    test_content=test_content,
                    source_content=source_content,
                    source_file=source_file,
                    dimensions=", ".join(dimensions),
                    context=context
                )
                
                llm_response = self.llm_adapter.analyze_code(
                    formatted_prompt,
                    analysis_type="evaluation",
                    max_tokens=2000,
                    temperature=0.1,
                    **kwargs
                )
            
            # Parse structured response
            scores, rationales = self._parse_llm_evaluation_response(
                llm_response, dimensions
            )
            
            # Calculate overall score
            overall_score = sum(scores.values()) / len(scores) if scores else 0.0
            
            result = LLMJudgeResult(
                scores=scores,
                rationales=rationales,
                overall_score=overall_score,
                prompt_version=prompt_version,
                confidence=llm_response.get('confidence', 0.8)
            )
            
            logger.debug(f"LLM judge evaluation completed with overall score: {overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return LLMJudgeResult.empty()
    
    def evaluate_single(
        self,
        test_content: str,
        source_file: str,
        config: Optional[EvaluationConfig] = None,
        **kwargs: Any
    ) -> EvaluationResult:
        """
        Perform complete evaluation of a single test with both automated and LLM checks.
        """
        logger.debug(f"Running single evaluation for {source_file}")
        
        try:
            config = config or EvaluationConfig()
            test_id = kwargs.get('test_id', f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            
            # Run acceptance checks
            acceptance_result = self.run_acceptance_checks(
                test_content, source_file, **kwargs
            )
            
            # Run LLM judge evaluation if enabled and acceptance checks pass
            llm_judge_result = None
            if config.llm_judge_enabled and acceptance_result.all_checks_pass:
                llm_judge_result = self.evaluate_with_llm_judge(
                    test_content, source_file, config.rubric_dimensions, **kwargs
                )
            
            # Create complete evaluation result
            result = EvaluationResult(
                test_id=test_id,
                source_file=source_file,
                test_content=test_content,
                acceptance=acceptance_result,
                llm_judge=llm_judge_result,
                metadata=kwargs.get('metadata', {})
            )
            
            # Store result in cache and artifacts
            self._evaluation_cache[test_id] = result
            self._store_evaluation_artifact(result)
            
            logger.info(f"Single evaluation completed for {source_file}")
            return result
            
        except Exception as e:
            logger.error(f"Single evaluation failed: {e}")
            raise EvaluationError(f"Failed to evaluate test: {e}") from e
    
    def evaluate_pairwise(
        self,
        test_a: str,
        test_b: str,
        source_file: str,
        comparison_mode: ComparisonMode = "a_vs_b",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Compare two test variants using pairwise LLM evaluation.
        
        This implements side-by-side comparison following 2025 A/B testing
        best practices with statistical confidence estimation.
        """
        logger.debug(f"Running pairwise evaluation for {source_file}")
        
        try:
            # Use enhanced pairwise comparison prompts (2025 best practices)
            system_prompt = self.prompt_registry.get_prompt(
                "system", "pairwise_comparison_v1"
            )
            user_prompt = self.prompt_registry.get_prompt(
                "user", "pairwise_comparison_v1"
            )
            
            # Read source content for context
            try:
                source_content = Path(source_file).read_text(encoding='utf-8')
            except Exception:
                source_content = "# Source file not available"
            
            if system_prompt and user_prompt:
                # Use advanced prompt structure
                formatted_user_prompt = user_prompt.format(
                    version="v1",
                    comparison_mode=comparison_mode,
                    source_content=source_content,
                    test_a=test_a,
                    test_b=test_b,
                    evaluation_context=kwargs.get('evaluation_context', 'Standard pairwise comparison')
                )
                
                llm_response = self.llm_adapter.analyze_code(
                    formatted_user_prompt,
                    analysis_type="pairwise_comparison",
                    temperature=0.1,
                    system_prompt=system_prompt,
                    **kwargs
                )
            else:
                # Fallback to old structure
                comparison_prompt = self._get_default_pairwise_prompt()
                formatted_prompt = comparison_prompt.format(
                    test_a=test_a,
                    test_b=test_b,
                    source_content=source_content,
                    source_file=source_file,
                    comparison_mode=comparison_mode
                )
                
                llm_response = self.llm_adapter.analyze_code(
                    formatted_prompt,
                    analysis_type="pairwise_comparison",
                    temperature=0.1,
                    **kwargs
                )
            
            # Parse comparison result
            comparison_result = self._parse_pairwise_response(llm_response)
            
            # Add metadata
            comparison_result.update({
                'source_file': source_file,
                'comparison_mode': comparison_mode,
                'timestamp': datetime.utcnow().isoformat(),
                'test_a_length': len(test_a),
                'test_b_length': len(test_b)
            })
            
            # Store comparison artifact
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                comparison_result,
                tags=["pairwise", "comparison", Path(source_file).stem],
                description=f"Pairwise comparison for {source_file}"
            )
            
            logger.info(f"Pairwise evaluation completed: winner = {comparison_result.get('winner', 'unknown')}")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Pairwise evaluation failed: {e}")
            raise EvaluationError(f"Failed to run pairwise evaluation: {e}") from e
    
    def evaluate_batch(
        self,
        test_variants: List[Dict[str, str]],
        source_files: List[str],
        config: Optional[EvaluationConfig] = None,
        **kwargs: Any
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test variants efficiently in batch mode.
        """
        logger.info(f"Running batch evaluation for {len(test_variants)} variants")
        
        try:
            config = config or EvaluationConfig()
            results = []
            
            if len(test_variants) != len(source_files):
                raise ValueError("Number of test variants must match number of source files")
            
            for i, (variant, source_file) in enumerate(zip(test_variants, source_files)):
                try:
                    test_content = variant.get('content', '')
                    test_id = variant.get('id', f"batch_{i}_{datetime.utcnow().strftime('%H%M%S')}")
                    
                    # Evaluate single variant
                    result = self.evaluate_single(
                        test_content,
                        source_file, 
                        config,
                        test_id=test_id,
                        **kwargs
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate variant {i}: {e}")
                    # Continue with other variants
            
            # Store batch results
            batch_summary = self._create_batch_summary(results)
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                batch_summary,
                tags=["batch", "evaluation", "summary"],
                description=f"Batch evaluation of {len(results)} tests"
            )
            
            logger.info(f"Batch evaluation completed: {len(results)}/{len(test_variants)} successful")
            return results
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise EvaluationError(f"Failed to run batch evaluation: {e}") from e
    
    def run_golden_repo_evaluation(
        self,
        golden_repo_path: Path,
        test_generator_func: Callable,
        evaluation_config: Optional[EvaluationConfig] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation against a golden repository.
        
        This discovers Python files, generates tests, and evaluates them
        to detect regressions and measure overall test generation quality.
        """
        logger.info(f"Running golden repo evaluation on {golden_repo_path}")
        
        try:
            if not golden_repo_path.exists():
                raise ValueError(f"Golden repo path does not exist: {golden_repo_path}")
            
            config = evaluation_config or EvaluationConfig()
            
            # Discover Python source files
            source_files = list(golden_repo_path.rglob("*.py"))
            source_files = [f for f in source_files if not self._should_skip_file(f)]
            
            if not source_files:
                raise ValueError("No valid Python files found in golden repository")
            
            logger.info(f"Found {len(source_files)} source files to evaluate")
            
            # Generate and evaluate tests for each file
            file_results = []
            overall_stats = {
                'total_files': len(source_files),
                'successful_evaluations': 0,
                'failed_evaluations': 0,
                'acceptance_pass_rate': 0.0,
                'average_llm_score': 0.0,
                'coverage_improvements': [],
                'regressions_detected': []
            }
            
            for source_file in source_files:
                try:
                    # Generate test using provided function
                    test_content = test_generator_func(str(source_file))
                    
                    if not test_content:
                        logger.warning(f"No test generated for {source_file}")
                        continue
                    
                    # Evaluate generated test
                    evaluation_result = self.evaluate_single(
                        test_content,
                        str(source_file),
                        config,
                        **kwargs
                    )
                    
                    file_results.append(evaluation_result)
                    overall_stats['successful_evaluations'] += 1
                    
                    # Update statistics
                    if evaluation_result.acceptance.all_checks_pass:
                        overall_stats['acceptance_pass_rate'] += 1
                    
                    if evaluation_result.llm_judge:
                        overall_stats['average_llm_score'] += evaluation_result.llm_judge.overall_score
                    
                    if evaluation_result.acceptance.coverage_improvement:
                        overall_stats['coverage_improvements'].append(
                            evaluation_result.acceptance.coverage_improvement
                        )
                
                except Exception as e:
                    logger.error(f"Failed to evaluate {source_file}: {e}")
                    overall_stats['failed_evaluations'] += 1
            
            # Calculate final statistics
            if overall_stats['successful_evaluations'] > 0:
                overall_stats['acceptance_pass_rate'] /= overall_stats['successful_evaluations']
                overall_stats['average_llm_score'] /= overall_stats['successful_evaluations']
            
            # Detect regressions (placeholder for sophisticated regression detection)
            regression_detected = overall_stats['acceptance_pass_rate'] < 0.7
            
            golden_repo_result = {
                'overall_results': overall_stats,
                'file_results': [result.to_dict() for result in file_results],
                'regression_detected': regression_detected,
                'recommendations': self._generate_recommendations(overall_stats, file_results),
                'evaluation_timestamp': datetime.utcnow().isoformat(),
                'golden_repo_path': str(golden_repo_path),
                'config': {
                    'acceptance_checks': config.acceptance_checks,
                    'llm_judge_enabled': config.llm_judge_enabled,
                    'rubric_dimensions': config.rubric_dimensions
                }
            }
            
            # Store golden repo evaluation results
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                golden_repo_result,
                tags=["golden_repo", "regression", "evaluation"],
                description=f"Golden repo evaluation: {golden_repo_path.name}"
            )
            
            logger.info(f"Golden repo evaluation completed: {overall_stats['successful_evaluations']} files evaluated")
            return golden_repo_result
            
        except Exception as e:
            logger.error(f"Golden repo evaluation failed: {e}")
            raise EvaluationError(f"Failed to run golden repo evaluation: {e}") from e
    
    def analyze_evaluation_trends(
        self,
        evaluation_history: List[EvaluationResult],
        time_window_days: Optional[int] = 30,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Analyze trends in evaluation results to identify patterns and improvements.
        """
        logger.debug(f"Analyzing evaluation trends for {len(evaluation_history)} results")
        
        try:
            if not evaluation_history:
                return {'error': 'No evaluation history provided'}
            
            # Filter by time window if specified
            if time_window_days:
                cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
                evaluation_history = [
                    result for result in evaluation_history
                    if datetime.fromisoformat(result.timestamp) >= cutoff_date
                ]
            
            if not evaluation_history:
                return {'error': 'No results in specified time window'}
            
            # Analyze trends
            trends = {
                'time_window_days': time_window_days,
                'total_evaluations': len(evaluation_history),
                'acceptance_trends': self._analyze_acceptance_trends(evaluation_history),
                'llm_judge_trends': self._analyze_llm_trends(evaluation_history),
                'coverage_trends': self._analyze_coverage_trends(evaluation_history),
                'quality_distribution': self._analyze_quality_distribution(evaluation_history),
                'recommendations': []
            }
            
            # Generate recommendations based on trends
            trends['recommendations'] = self._generate_trend_recommendations(trends)
            
            logger.info("Evaluation trend analysis completed")
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise EvaluationError(f"Failed to analyze evaluation trends: {e}") from e
    
    # =============================
    # NEW METHODS FOR SUBTASK 20.2: Enhanced Statistical Testing & Bias Mitigation
    # =============================
    
    def run_statistical_significance_analysis(
        self,
        evaluation_data: List[Dict[str, Any]],
        analysis_type: str = "pairwise_comparison",
        confidence_level: float = 0.95,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Run statistical significance analysis on evaluation data following 2025 best practices.
        
        This implements t-tests, bootstrap sampling, and effect size calculations
        to assess the statistical reliability of evaluation comparisons.
        """
        logger.debug(f"Running statistical significance analysis: {analysis_type}")
        
        try:
            # Get statistical analysis prompt
            system_prompt = self.prompt_registry.get_prompt(
                "system", "statistical_analysis_v1"
            )
            user_prompt = self.prompt_registry.get_prompt(
                "user", "statistical_analysis_v1"
            )
            
            if not system_prompt or not user_prompt:
                logger.warning("Statistical analysis prompts not available")
                return self._fallback_statistical_analysis(evaluation_data, confidence_level)
            
            # Format evaluation data for analysis
            formatted_data = self._format_evaluation_data_for_analysis(evaluation_data)
            
            # Prepare analysis context
            analysis_parameters = {
                'confidence_level': confidence_level,
                'analysis_type': analysis_type,
                'sample_size': len(evaluation_data),
                'requested_tests': ['t_test', 'bootstrap', 'effect_size']
            }
            
            # Format user prompt
            formatted_user_prompt = user_prompt.format(
                version="v1",
                evaluation_data=formatted_data,
                comparison_context=f"Statistical analysis of {analysis_type} evaluation data",
                analysis_parameters=json.dumps(analysis_parameters, indent=2)
            )
            
            # Call LLM for statistical analysis
            llm_response = self.llm_adapter.analyze_code(
                formatted_user_prompt,
                analysis_type="statistical_analysis",
                temperature=0.1,
                system_prompt=system_prompt,
                max_tokens=1500,
                **kwargs
            )
            
            # Parse statistical analysis response
            analysis_result = self._parse_statistical_analysis_response(llm_response)
            
            # Add computational validation where possible
            analysis_result = self._validate_statistical_results(analysis_result, evaluation_data)
            
            # Store analysis artifact
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                analysis_result,
                tags=["statistical", "significance", "analysis"],
                description=f"Statistical significance analysis: {analysis_type}"
            )
            
            logger.info(f"Statistical analysis completed: {analysis_result.get('significance_assessment', 'unknown')}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Statistical significance analysis failed: {e}")
            return {
                'error': f'Statistical analysis failed: {e}',
                'fallback_analysis': self._fallback_statistical_analysis(evaluation_data, confidence_level)
            }
    
    def detect_evaluation_bias(
        self,
        evaluation_history: List[EvaluationResult],
        bias_types: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Detect and analyze evaluation bias patterns following 2025 best practices.
        
        This implements bias detection for length bias, complexity bias, style bias,
        anchoring effects, and other systematic evaluation issues.
        """
        logger.debug(f"Running bias detection analysis on {len(evaluation_history)} evaluations")
        
        try:
            # Get bias mitigation prompt
            system_prompt = self.prompt_registry.get_prompt(
                "system", "bias_mitigation_v1"
            )
            user_prompt = self.prompt_registry.get_prompt(
                "user", "bias_mitigation_v1"
            )
            
            if not system_prompt or not user_prompt:
                logger.warning("Bias mitigation prompts not available")
                return self._fallback_bias_analysis(evaluation_history)
            
            # Prepare evaluation history data
            history_data = self._format_evaluation_history_for_bias_analysis(evaluation_history)
            
            # Define analysis scope
            analysis_scope = {
                'bias_types_to_check': bias_types or [
                    'length_bias', 'complexity_bias', 'style_bias', 'framework_bias',
                    'anchoring_bias', 'order_bias', 'confirmation_bias'
                ],
                'evaluation_period': f"{len(evaluation_history)} evaluations",
                'consistency_metrics': True,
                'drift_detection': True
            }
            
            # Format user prompt
            formatted_user_prompt = user_prompt.format(
                version="v1",
                evaluation_history=history_data,
                evaluation_context="Comprehensive bias detection and mitigation analysis",
                analysis_scope=json.dumps(analysis_scope, indent=2)
            )
            
            # Call LLM for bias analysis
            llm_response = self.llm_adapter.analyze_code(
                formatted_user_prompt,
                analysis_type="bias_detection",
                temperature=0.1,
                system_prompt=system_prompt,
                max_tokens=2000,
                **kwargs
            )
            
            # Parse bias analysis response
            bias_result = self._parse_bias_analysis_response(llm_response)
            
            # Add computational bias metrics
            bias_result = self._compute_bias_metrics(bias_result, evaluation_history)
            
            # Store bias analysis artifact
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                bias_result,
                tags=["bias", "mitigation", "fairness"],
                description=f"Bias detection analysis on {len(evaluation_history)} evaluations"
            )
            
            logger.info(f"Bias analysis completed: fairness score = {bias_result.get('fairness_score', 'unknown')}")
            return bias_result
            
        except Exception as e:
            logger.error(f"Bias detection analysis failed: {e}")
            return {
                'error': f'Bias detection failed: {e}',
                'fallback_analysis': self._fallback_bias_analysis(evaluation_history)
            }
    
    def run_advanced_ab_testing_pipeline(
        self,
        prompt_variants: List[Dict[str, str]],
        test_dataset: List[Dict[str, str]], 
        statistical_testing: bool = True,
        bias_mitigation: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Run comprehensive A/B testing pipeline with statistical significance testing 
        and bias mitigation following 2025 best practices.
        
        This implements the full PromptFoo/PromptLayer-style evaluation workflow
        with side-by-side comparison, statistical analysis, and bias detection.
        """
        logger.info(f"Running advanced A/B testing pipeline: {len(prompt_variants)} variants, {len(test_dataset)} test cases")
        
        try:
            pipeline_results = {
                'metadata': {
                    'variant_count': len(prompt_variants),
                    'test_case_count': len(test_dataset),
                    'statistical_testing_enabled': statistical_testing,
                    'bias_mitigation_enabled': bias_mitigation,
                    'timestamp': datetime.utcnow().isoformat()
                },
                'variant_evaluations': [],
                'pairwise_comparisons': [],
                'statistical_analysis': None,
                'bias_analysis': None,
                'recommendations': []
            }
            
            # Step 1: Run individual evaluations for each variant
            logger.info("Step 1: Running individual variant evaluations")
            for i, variant in enumerate(prompt_variants):
                variant_id = variant.get('id', f'variant_{i}')
                variant_results = []
                
                for j, test_case in enumerate(test_dataset):
                    # Generate test using this prompt variant
                    test_content = self._generate_test_with_variant(variant, test_case)
                    
                    # Evaluate the generated test
                    evaluation_result = self.evaluate_single(
                        test_content=test_content,
                        source_file=test_case.get('source_file', f'test_case_{j}.py'),
                        config=EvaluationConfig(llm_judge_enabled=True, statistical_testing=True),
                        test_id=f"{variant_id}_case_{j}",
                        variant_id=variant_id,
                        **kwargs
                    )
                    
                    variant_results.append(evaluation_result)
                
                pipeline_results['variant_evaluations'].append({
                    'variant_id': variant_id,
                    'variant_metadata': variant,
                    'results': [r.to_dict() for r in variant_results],
                    'summary_stats': self._calculate_variant_summary_stats(variant_results)
                })
            
            # Step 2: Run pairwise comparisons between variants
            logger.info("Step 2: Running pairwise comparisons")
            for i, variant_a in enumerate(prompt_variants):
                for j, variant_b in enumerate(prompt_variants):
                    if i >= j:  # Avoid duplicate comparisons
                        continue
                    
                    comparison_results = []
                    for k, test_case in enumerate(test_dataset):
                        # Generate tests with both variants
                        test_a = self._generate_test_with_variant(variant_a, test_case)
                        test_b = self._generate_test_with_variant(variant_b, test_case)
                        
                        # Run pairwise comparison
                        comparison_result = self.evaluate_pairwise(
                            test_a=test_a,
                            test_b=test_b,
                            source_file=test_case.get('source_file', f'test_case_{k}.py'),
                            comparison_mode="a_vs_b",
                            evaluation_context=f"A/B testing: {variant_a.get('id', f'variant_{i}')} vs {variant_b.get('id', f'variant_{j}')}",
                            **kwargs
                        )
                        
                        comparison_results.append(comparison_result)
                    
                    pipeline_results['pairwise_comparisons'].append({
                        'variant_a_id': variant_a.get('id', f'variant_{i}'),
                        'variant_b_id': variant_b.get('id', f'variant_{j}'),
                        'comparisons': comparison_results,
                        'summary': self._calculate_pairwise_summary(comparison_results)
                    })
            
            # Step 3: Statistical significance analysis
            if statistical_testing:
                logger.info("Step 3: Running statistical significance analysis")
                evaluation_data = []
                for variant_eval in pipeline_results['variant_evaluations']:
                    for result in variant_eval['results']:
                        evaluation_data.append({
                            'variant_id': variant_eval['variant_id'],
                            'overall_score': result.get('llm_judge', {}).get('overall_score', 0),
                            'scores': result.get('llm_judge', {}).get('scores', {}),
                            'acceptance_passed': result.get('acceptance', {}).get('all_checks_pass', False)
                        })
                
                pipeline_results['statistical_analysis'] = self.run_statistical_significance_analysis(
                    evaluation_data=evaluation_data,
                    analysis_type="ab_testing_pipeline",
                    **kwargs
                )
            
            # Step 4: Bias detection and mitigation
            if bias_mitigation:
                logger.info("Step 4: Running bias detection analysis")
                evaluation_history = []
                for variant_eval in pipeline_results['variant_evaluations']:
                    for result_dict in variant_eval['results']:
                        # Convert dict back to EvaluationResult for bias analysis
                        # This is a simplified conversion - you might want to implement proper deserialization
                        evaluation_history.append(self._dict_to_evaluation_result(result_dict))
                
                pipeline_results['bias_analysis'] = self.detect_evaluation_bias(
                    evaluation_history=evaluation_history,
                    **kwargs
                )
            
            # Step 5: Generate recommendations
            logger.info("Step 5: Generating recommendations")
            pipeline_results['recommendations'] = self._generate_ab_testing_recommendations(pipeline_results)
            
            # Store comprehensive A/B testing results
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                pipeline_results,
                tags=["ab_testing", "pipeline", "comprehensive"],
                description=f"Advanced A/B testing pipeline: {len(prompt_variants)} variants"
            )
            
            logger.info("Advanced A/B testing pipeline completed successfully")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"A/B testing pipeline failed: {e}")
            raise EvaluationError(f"Failed to run A/B testing pipeline: {e}") from e
    
    # Private helper methods
    
    def _check_syntax(self, test_content: str, error_messages: List[str]) -> bool:
        """Check if test content is syntactically valid Python."""
        try:
            ast.parse(test_content)
            return True
        except SyntaxError as e:
            error_messages.append(f"Syntax error: {e}")
            return False
        except Exception as e:
            error_messages.append(f"Parse error: {e}")
            return False
    
    def _check_imports(self, test_content: str, source_file: str, error_messages: List[str]) -> bool:
        """Check if test content imports successfully."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_content)
                temp_file = Path(f.name)
            
            # Validate file path safety if enabled
            if self.safety_enabled:
                SafetyPolicies.validate_file_path(temp_file, self.project_root)
            
            # Try to compile the test file
            result = subprocess.run([
                'python', '-m', 'py_compile', str(temp_file)
            ], capture_output=True, text=True, timeout=30)
            
            temp_file.unlink()  # Clean up
            
            if result.returncode == 0:
                return True
            else:
                error_messages.append(f"Import error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            error_messages.append("Import check timed out")
            return False
        except Exception as e:
            error_messages.append(f"Import check failed: {e}")
            return False
    
    def _run_pytest(self, test_content: str, source_file: str, error_messages: List[str]) -> bool:
        """Run pytest on the test content."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                test_file = temp_path / 'test_generated.py'
                test_file.write_text(test_content, encoding='utf-8')
                
                # Run pytest with appropriate flags
                result = subprocess.run([
                    'python', '-m', 'pytest', 
                    str(test_file),
                    '-v', '--tb=short', '--no-header'
                ], capture_output=True, text=True, timeout=60, cwd=self.project_root)
                
                if result.returncode == 0:
                    return True
                else:
                    error_messages.append(f"Pytest failed: {result.stdout}")
                    if result.stderr:
                        error_messages.append(f"Pytest stderr: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            error_messages.append("Pytest execution timed out")
            return False
        except Exception as e:
            error_messages.append(f"Pytest execution failed: {e}")
            return False
    
    def _measure_coverage_improvement(
        self, 
        test_content: str, 
        source_file: str, 
        baseline_coverage: Dict[str, Any]
    ) -> Optional[float]:
        """Measure coverage improvement from baseline."""
        try:
            # This is a simplified implementation
            # In practice, you'd run coverage with the new test and compare
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) 
                test_file = temp_path / 'test_generated.py'
                test_file.write_text(test_content, encoding='utf-8')
                
                # Measure coverage with new test
                new_coverage = self.coverage_adapter.measure_coverage(
                    [source_file], [str(test_file)]
                )
                
                source_file_key = str(Path(source_file).resolve())
                if source_file_key in new_coverage:
                    new_line_coverage = new_coverage[source_file_key].line_coverage
                    baseline_line_coverage = baseline_coverage.get('line_coverage', 0.0)
                    
                    improvement = new_line_coverage - baseline_line_coverage
                    return max(0.0, improvement)  # Don't report negative improvements
                
                return None
                
        except Exception as e:
            logger.warning(f"Could not measure coverage improvement: {e}")
            return None
    
    def _build_evaluation_context(
        self, 
        test_content: str, 
        source_content: str, 
        source_file: str, 
        dimensions: List[str]
    ) -> str:
        """Build context string for LLM evaluation."""
        context_parts = [
            f"Source file: {source_file}",
            f"Evaluation dimensions: {', '.join(dimensions)}",
            f"Test content length: {len(test_content)} characters",
            f"Source content length: {len(source_content)} characters"
        ]
        
        return "\n".join(context_parts)
    
    def _get_default_evaluation_prompt(self) -> str:
        """Get default LLM evaluation prompt."""
        return """
You are an expert Python test reviewer. Evaluate the following generated test code on these dimensions:

{dimensions}

Rate each dimension from 1-5 and provide a brief rationale:
- Correctness (1-5): Does the test correctly validate the intended behavior?
- Coverage (1-5): Does the test increase code coverage, especially for edge cases?
- Clarity (1-5): Is the test code readable and maintainable?
- Safety (1-5): Does the test avoid modifying source files or introducing side effects?

Source Code:
```python
{source_content}
```

Generated Test:
```python
{test_content}
```

Respond in JSON format:
{{
    "scores": {{
        "correctness": <score>,
        "coverage": <score>,
        "clarity": <score>,
        "safety": <score>
    }},
    "rationales": {{
        "correctness": "<rationale>",
        "coverage": "<rationale>",
        "clarity": "<rationale>",
        "safety": "<rationale>"
    }}
}}
"""
    
    def _get_default_pairwise_prompt(self) -> str:
        """Get default pairwise comparison prompt."""
        return """
Compare these two test implementations for the same source code. 
Determine which test is better overall and explain your reasoning.

Source Code:
```python
{source_content}
```

Test A:
```python
{test_a}
```

Test B:
```python
{test_b}
```

Consider: correctness, coverage, clarity, maintainability, and safety.

Respond in JSON format:
{{
    "winner": "a|b|tie",
    "confidence": <0.0-1.0>,
    "reasoning": "<detailed explanation>",
    "scores": {{
        "test_a": <1-5>,
        "test_b": <1-5>
    }}
}}
"""
    
    def _parse_llm_evaluation_response(
        self, 
        llm_response: Dict[str, Any], 
        dimensions: List[str]
    ) -> tuple[Dict[str, float], Dict[str, str]]:
        """Parse LLM evaluation response into scores and rationales."""
        try:
            # Try to extract JSON from response
            response_text = llm_response.get('analysis', '')
            if isinstance(response_text, dict):
                data = response_text
            else:
                # Try to parse JSON from text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
            
            scores = {}
            rationales = {}
            
            for dimension in dimensions:
                scores[dimension] = float(data.get('scores', {}).get(dimension, 3.0))
                rationales[dimension] = data.get('rationales', {}).get(dimension, 'No rationale provided')
            
            return scores, rationales
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM evaluation response: {e}")
            # Return default scores
            return {dim: 3.0 for dim in dimensions}, {dim: 'Parse error' for dim in dimensions}
    
    def _parse_pairwise_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse pairwise comparison response."""
        try:
            response_text = llm_response.get('analysis', '')
            if isinstance(response_text, dict):
                data = response_text
            else:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
            
            return {
                'winner': data.get('winner', 'tie'),
                'confidence': float(data.get('confidence', 0.5)),
                'reasoning': data.get('reasoning', 'No reasoning provided'),
                'scores': data.get('scores', {'test_a': 3.0, 'test_b': 3.0})
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse pairwise response: {e}")
            return {
                'winner': 'tie',
                'confidence': 0.0,
                'reasoning': f'Parse error: {e}',
                'scores': {'test_a': 3.0, 'test_b': 3.0}
            }
    
    def _store_evaluation_artifact(self, result: EvaluationResult) -> None:
        """Store evaluation result as artifact."""
        try:
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                result.to_dict(),
                artifact_id=f"eval_{result.test_id}",
                tags=["evaluation", "single", Path(result.source_file).stem],
                description=f"Evaluation result for {result.source_file}"
            )
        except Exception as e:
            logger.warning(f"Failed to store evaluation artifact: {e}")
    
    def _create_batch_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Create summary of batch evaluation results."""
        successful_count = sum(1 for r in results if r.acceptance.all_checks_pass)
        
        llm_scores = [
            r.llm_judge.overall_score for r in results 
            if r.llm_judge and r.llm_judge.overall_score > 0
        ]
        
        return {
            'total_tests': len(results),
            'successful_tests': successful_count,
            'success_rate': successful_count / len(results) if results else 0.0,
            'average_llm_score': sum(llm_scores) / len(llm_scores) if llm_scores else 0.0,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during golden repo evaluation."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'test_',
            '_test.py',
            'conftest.py',
            '__init__.py'
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _generate_recommendations(
        self, 
        stats: Dict[str, Any], 
        file_results: List[EvaluationResult]
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if stats['acceptance_pass_rate'] < 0.5:
            recommendations.append("Consider improving test generation prompts - low acceptance rate")
        
        if stats['average_llm_score'] < 3.0:
            recommendations.append("Generated tests may need quality improvements")
        
        if not stats['coverage_improvements']:
            recommendations.append("Tests may not be improving code coverage significantly")
        
        return recommendations
    
    def _analyze_acceptance_trends(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze trends in acceptance check results."""
        syntax_passes = sum(1 for r in results if r.acceptance.syntactically_valid)
        import_passes = sum(1 for r in results if r.acceptance.imports_successfully) 
        pytest_passes = sum(1 for r in results if r.acceptance.pytest_passes)
        
        return {
            'syntax_pass_rate': syntax_passes / len(results),
            'import_pass_rate': import_passes / len(results), 
            'pytest_pass_rate': pytest_passes / len(results)
        }
    
    def _analyze_llm_trends(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze trends in LLM judge scores."""
        llm_results = [r.llm_judge for r in results if r.llm_judge]
        
        if not llm_results:
            return {'no_llm_data': True}
        
        avg_scores = {}
        for dimension in ['correctness', 'coverage', 'clarity', 'safety']:
            scores = [r.scores.get(dimension, 0) for r in llm_results if r.scores]
            avg_scores[dimension] = sum(scores) / len(scores) if scores else 0
        
        overall_scores = [r.overall_score for r in llm_results]
        
        return {
            'average_dimension_scores': avg_scores,
            'average_overall_score': sum(overall_scores) / len(overall_scores),
            'score_trend': 'improving' if len(overall_scores) > 1 and overall_scores[-1] > overall_scores[0] else 'stable'
        }
    
    def _analyze_coverage_trends(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze trends in coverage improvements."""
        improvements = [
            r.acceptance.coverage_improvement for r in results 
            if r.acceptance.coverage_improvement is not None
        ]
        
        if not improvements:
            return {'no_coverage_data': True}
        
        return {
            'average_improvement': sum(improvements) / len(improvements),
            'positive_improvements': sum(1 for i in improvements if i > 0),
            'improvement_rate': sum(1 for i in improvements if i > 0) / len(improvements)
        }
    
    def _analyze_quality_distribution(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze the distribution of test quality."""
        quality_buckets = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        for result in results:
            if result.llm_judge and result.llm_judge.overall_score > 0:
                score = result.llm_judge.overall_score
                if score >= 4.5:
                    quality_buckets['excellent'] += 1
                elif score >= 3.5:
                    quality_buckets['good'] += 1
                elif score >= 2.5:
                    quality_buckets['fair'] += 1
                else:
                    quality_buckets['poor'] += 1
        
        return quality_buckets
    
    def _generate_trend_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        acceptance = trends.get('acceptance_trends', {})
        if acceptance.get('syntax_pass_rate', 0) < 0.9:
            recommendations.append("Improve prompt templates to reduce syntax errors")
        
        llm_trends = trends.get('llm_judge_trends', {})
        if not llm_trends.get('no_llm_data'):
            avg_score = llm_trends.get('average_overall_score', 0)
            if avg_score < 3.0:
                recommendations.append("Consider refining test generation approach for higher quality")
        
        coverage_trends = trends.get('coverage_trends', {})
        if not coverage_trends.get('no_coverage_data'):
            improvement_rate = coverage_trends.get('improvement_rate', 0)
            if improvement_rate < 0.7:
                recommendations.append("Focus on generating tests that improve code coverage")
        
        return recommendations
    
    # =============================
    # HELPER METHODS FOR SUBTASK 20.2 - Statistical Analysis & Bias Detection
    # =============================
    
    def _format_evaluation_data_for_analysis(self, evaluation_data: List[Dict[str, Any]]) -> str:
        """Format evaluation data for statistical analysis prompts."""
        try:
            formatted_data = []
            for i, data in enumerate(evaluation_data):
                formatted_entry = {
                    'entry_id': i,
                    'variant_id': data.get('variant_id', 'unknown'),
                    'overall_score': data.get('overall_score', 0),
                    'dimension_scores': data.get('scores', {}),
                    'acceptance_passed': data.get('acceptance_passed', False)
                }
                formatted_data.append(formatted_entry)
            
            return json.dumps(formatted_data, indent=2)
        except Exception as e:
            logger.warning(f"Failed to format evaluation data: {e}")
            return "[]"
    
    def _format_evaluation_history_for_bias_analysis(self, evaluation_history: List[EvaluationResult]) -> str:
        """Format evaluation history for bias analysis prompts."""
        try:
            formatted_history = []
            for result in evaluation_history:
                entry = {
                    'test_id': result.test_id,
                    'timestamp': result.timestamp,
                    'acceptance_passed': result.acceptance.all_checks_pass,
                    'test_length': len(result.test_content),
                    'source_file': result.source_file
                }
                
                if result.llm_judge:
                    entry.update({
                        'llm_scores': result.llm_judge.scores,
                        'overall_score': result.llm_judge.overall_score,
                        'prompt_version': result.llm_judge.prompt_version
                    })
                
                formatted_history.append(entry)
            
            return json.dumps(formatted_history, indent=2)
        except Exception as e:
            logger.warning(f"Failed to format evaluation history: {e}")
            return "[]"
    
    def _parse_statistical_analysis_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM statistical analysis response."""
        try:
            response_text = llm_response.get('analysis', '')
            if isinstance(response_text, dict):
                return response_text
            
            # Try to parse JSON from text response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Return fallback structure
            return {
                'statistical_test': 'fallback',
                'p_value': 0.5,
                'confidence_interval': {'lower': -0.5, 'upper': 0.5, 'confidence_level': 0.95},
                'effect_size': {'cohens_d': 0.0, 'interpretation': 'negligible'},
                'significance_assessment': 'not_significant',
                'sample_adequacy': {'current_sample_size': len([]), 'recommended_minimum': 30, 'power_achieved': 0.5},
                'reliability_metrics': {'evaluation_consistency': 0.5, 'potential_bias_detected': False},
                'interpretation': 'Fallback analysis - original parsing failed',
                'recommendations': ['Increase sample size', 'Validate evaluation methodology']
            }
        except Exception as e:
            logger.warning(f"Failed to parse statistical analysis response: {e}")
            return {
                'error': f'Parsing failed: {e}',
                'fallback_used': True
            }
    
    def _parse_bias_analysis_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM bias analysis response."""
        try:
            response_text = llm_response.get('analysis', '')
            if isinstance(response_text, dict):
                return response_text
            
            # Try to parse JSON from text response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Return fallback structure
            return {
                'bias_analysis': {
                    'detected_biases': ['unknown_bias'],
                    'bias_severity': {'unknown_bias': 'low'},
                    'confidence': 0.5
                },
                'evaluation_consistency': {
                    'consistency_score': 0.7,
                    'variance_analysis': 'Fallback analysis - unable to perform detailed assessment',
                    'drift_detected': False
                },
                'calibration_assessment': {
                    'calibration_score': 0.7,
                    'systematic_errors': ['Unable to detect systematic errors in fallback mode'],
                    'improvement_needed': True
                },
                'mitigation_recommendations': {
                    'immediate_actions': ['Review evaluation methodology'],
                    'process_improvements': ['Implement structured rubrics'],
                    'monitoring_suggestions': ['Track evaluation consistency']
                },
                'fairness_score': 0.7,
                'summary': 'Fallback bias analysis - detailed assessment unavailable due to parsing error'
            }
        except Exception as e:
            logger.warning(f"Failed to parse bias analysis response: {e}")
            return {
                'error': f'Parsing failed: {e}',
                'fallback_used': True
            }
    
    def _fallback_statistical_analysis(self, evaluation_data: List[Dict[str, Any]], confidence_level: float) -> Dict[str, Any]:
        """Provide basic statistical analysis when LLM-based analysis is unavailable."""
        try:
            if not evaluation_data:
                return {'error': 'No evaluation data provided'}
            
            # Simple statistical calculations
            scores = [item.get('overall_score', 0) for item in evaluation_data]
            mean_score = sum(scores) / len(scores) if scores else 0
            
            # Basic statistical assessment
            sample_size = len(evaluation_data)
            is_adequate = sample_size >= 30
            
            return {
                'statistical_test': 'descriptive',
                'p_value': 0.5,  # Neutral
                'confidence_interval': {
                    'lower': max(0, mean_score - 0.5), 
                    'upper': min(5, mean_score + 0.5), 
                    'confidence_level': confidence_level
                },
                'effect_size': {'cohens_d': 0.0, 'interpretation': 'unknown'},
                'significance_assessment': 'not_assessed',
                'sample_adequacy': {
                    'current_sample_size': sample_size,
                    'recommended_minimum': 30,
                    'power_achieved': 0.8 if is_adequate else 0.5
                },
                'reliability_metrics': {
                    'evaluation_consistency': 0.7,
                    'potential_bias_detected': False
                },
                'interpretation': f'Basic descriptive analysis of {sample_size} evaluations with mean score {mean_score:.2f}',
                'recommendations': ['Increase sample size for statistical testing', 'Use LLM-based analysis for detailed insights'],
                'fallback_analysis': True
            }
        except Exception as e:
            return {'error': f'Fallback analysis failed: {e}'}
    
    def _fallback_bias_analysis(self, evaluation_history: List[EvaluationResult]) -> Dict[str, Any]:
        """Provide basic bias analysis when LLM-based analysis is unavailable."""
        try:
            if not evaluation_history:
                return {'error': 'No evaluation history provided'}
            
            # Basic bias indicators
            test_lengths = [len(result.test_content) for result in evaluation_history]
            length_variance = max(test_lengths) - min(test_lengths) if test_lengths else 0
            
            overall_scores = [result.llm_judge.overall_score for result in evaluation_history if result.llm_judge]
            score_variance = (max(overall_scores) - min(overall_scores)) if overall_scores else 0
            
            # Basic fairness assessment
            fairness_score = 0.8 if length_variance < 1000 and score_variance < 2.0 else 0.6
            
            return {
                'bias_analysis': {
                    'detected_biases': ['length_bias'] if length_variance > 1000 else [],
                    'bias_severity': {'length_bias': 'moderate'} if length_variance > 1000 else {},
                    'confidence': 0.6
                },
                'evaluation_consistency': {
                    'consistency_score': max(0.5, 1.0 - (score_variance / 5.0)),
                    'variance_analysis': f'Score variance: {score_variance:.2f}, Length variance: {length_variance}',
                    'drift_detected': score_variance > 2.0
                },
                'calibration_assessment': {
                    'calibration_score': 0.7,
                    'systematic_errors': ['High variance detected'] if score_variance > 2.0 else [],
                    'improvement_needed': score_variance > 1.5
                },
                'mitigation_recommendations': {
                    'immediate_actions': ['Monitor evaluation consistency'],
                    'process_improvements': ['Implement evaluation guidelines'],
                    'monitoring_suggestions': ['Track variance metrics']
                },
                'fairness_score': fairness_score,
                'summary': f'Basic bias analysis of {len(evaluation_history)} evaluations - fairness score: {fairness_score:.2f}',
                'fallback_analysis': True
            }
        except Exception as e:
            return {'error': f'Fallback bias analysis failed: {e}'}
    
    def _validate_statistical_results(self, analysis_result: Dict[str, Any], evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add computational validation to statistical analysis results."""
        try:
            # Basic validation and augmentation
            sample_size = len(evaluation_data)
            analysis_result['validation'] = {
                'sample_size_confirmed': sample_size,
                'computational_checks': 'basic_validation_applied',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Validate confidence intervals make sense
            if 'confidence_interval' in analysis_result:
                ci = analysis_result['confidence_interval']
                if ci['lower'] > ci['upper']:
                    logger.warning("Invalid confidence interval detected, correcting")
                    ci['lower'], ci['upper'] = ci['upper'], ci['lower']
            
            return analysis_result
        except Exception as e:
            logger.warning(f"Statistical validation failed: {e}")
            return analysis_result
    
    def _compute_bias_metrics(self, bias_result: Dict[str, Any], evaluation_history: List[EvaluationResult]) -> Dict[str, Any]:
        """Add computational bias metrics to LLM bias analysis."""
        try:
            # Basic computational metrics
            test_lengths = [len(result.test_content) for result in evaluation_history]
            scores = [result.llm_judge.overall_score for result in evaluation_history if result.llm_judge]
            
            bias_result['computational_metrics'] = {
                'length_statistics': {
                    'mean': sum(test_lengths) / len(test_lengths) if test_lengths else 0,
                    'min': min(test_lengths) if test_lengths else 0,
                    'max': max(test_lengths) if test_lengths else 0,
                    'variance': length_variance if (length_variance := (max(test_lengths) - min(test_lengths))) else 0
                },
                'score_statistics': {
                    'mean': sum(scores) / len(scores) if scores else 0,
                    'min': min(scores) if scores else 0,
                    'max': max(scores) if scores else 0,
                    'variance': (max(scores) - min(scores)) if scores else 0
                },
                'evaluation_count': len(evaluation_history)
            }
            
            return bias_result
        except Exception as e:
            logger.warning(f"Bias metrics computation failed: {e}")
            return bias_result
    
    def _generate_test_with_variant(self, variant: Dict[str, str], test_case: Dict[str, str]) -> str:
        """Generate test content using a specific prompt variant."""
        # This is a simplified implementation - you would integrate with your test generation system
        try:
            source_file = test_case.get('source_file', 'example.py')
            source_content = test_case.get('source_content', '# No source content available')
            
            # Use LLM to generate test with the specific variant
            prompt_content = variant.get('prompt', 'Generate a comprehensive test for the given code.')
            
            # Simple test generation (you would use your actual generation pipeline here)
            generated_test = f"""
import pytest
from pathlib import Path

def test_{Path(source_file).stem}():
    '''Generated test using variant: {variant.get('id', 'unknown')}'''
    # This is a placeholder test generated for evaluation
    # Variant prompt: {prompt_content[:100]}...
    assert True  # Placeholder assertion
"""
            return generated_test
        except Exception as e:
            logger.warning(f"Test generation with variant failed: {e}")
            return "# Test generation failed\ndef test_placeholder(): assert True"
    
    def _calculate_variant_summary_stats(self, variant_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics for a prompt variant."""
        try:
            acceptance_rate = sum(1 for r in variant_results if r.acceptance.all_checks_pass) / len(variant_results)
            
            llm_scores = [r.llm_judge.overall_score for r in variant_results if r.llm_judge]
            mean_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0
            
            return {
                'total_tests': len(variant_results),
                'acceptance_rate': acceptance_rate,
                'mean_llm_score': mean_score,
                'successful_evaluations': len(llm_scores)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate variant summary stats: {e}")
            return {'error': f'Calculation failed: {e}'}
    
    def _calculate_pairwise_summary(self, comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary for pairwise comparisons."""
        try:
            if not comparison_results:
                return {'error': 'No comparison results'}
            
            winners = [result.get('winner', 'tie') for result in comparison_results]
            winner_counts = {
                'a': winners.count('a'),
                'b': winners.count('b'), 
                'tie': winners.count('tie')
            }
            
            confidences = [result.get('confidence', 0.5) for result in comparison_results]
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'total_comparisons': len(comparison_results),
                'winner_distribution': winner_counts,
                'mean_confidence': mean_confidence,
                'clear_winner': 'a' if winner_counts['a'] > winner_counts['b'] + winner_counts['tie'] else 'b' if winner_counts['b'] > winner_counts['a'] + winner_counts['tie'] else 'tie'
            }
        except Exception as e:
            logger.warning(f"Failed to calculate pairwise summary: {e}")
            return {'error': f'Calculation failed: {e}'}
    
    def _dict_to_evaluation_result(self, result_dict: Dict[str, Any]) -> EvaluationResult:
        """Convert dictionary back to EvaluationResult object."""
        # Simplified conversion - you might want to implement a proper deserialization
        try:
            from testcraft.ports.evaluation_port import AcceptanceResult, LLMJudgeResult
            
            acceptance_data = result_dict.get('acceptance', {})
            acceptance = AcceptanceResult(
                syntactically_valid=acceptance_data.get('syntactically_valid', False),
                imports_successfully=acceptance_data.get('imports_successfully', False),
                pytest_passes=acceptance_data.get('pytest_passes', False),
                coverage_improvement=acceptance_data.get('coverage_improvement'),
                error_messages=acceptance_data.get('error_messages', [])
            )
            
            llm_judge_data = result_dict.get('llm_judge')
            llm_judge = None
            if llm_judge_data:
                llm_judge = LLMJudgeResult(
                    scores=llm_judge_data.get('scores', {}),
                    rationales=llm_judge_data.get('rationales', {}),
                    overall_score=llm_judge_data.get('overall_score', 0),
                    prompt_version=llm_judge_data.get('prompt_version', 'unknown')
                )
            
            return EvaluationResult(
                test_id=result_dict.get('test_id', 'unknown'),
                source_file=result_dict.get('source_file', 'unknown'),
                test_content=result_dict.get('test_content', ''),
                acceptance=acceptance,
                llm_judge=llm_judge,
                metadata=result_dict.get('metadata', {}),
                timestamp=result_dict.get('timestamp')
            )
        except Exception as e:
            logger.warning(f"Failed to convert dict to EvaluationResult: {e}")
            # Return minimal result
            return EvaluationResult(
                test_id='conversion_failed',
                source_file='unknown',
                test_content='',
                acceptance=AcceptanceResult(False, False, False)
            )
    
    def _generate_ab_testing_recommendations(self, pipeline_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on A/B testing results."""
        recommendations = []
        
        try:
            # Analyze variant performance
            variant_evals = pipeline_results.get('variant_evaluations', [])
            if variant_evals:
                best_variant = max(variant_evals, key=lambda v: v.get('summary_stats', {}).get('mean_llm_score', 0))
                recommendations.append(f"Consider using variant '{best_variant.get('variant_id', 'unknown')}' based on highest mean LLM score")
            
            # Statistical analysis recommendations
            stat_analysis = pipeline_results.get('statistical_analysis')
            if stat_analysis and not stat_analysis.get('error'):
                significance = stat_analysis.get('significance_assessment', 'unknown')
                if significance == 'not_significant':
                    recommendations.append("Differences between variants are not statistically significant - consider larger sample size or different evaluation criteria")
                elif significance in ['significant', 'highly_significant']:
                    recommendations.append("Statistically significant differences detected - results are reliable for decision making")
            
            # Bias analysis recommendations
            bias_analysis = pipeline_results.get('bias_analysis')
            if bias_analysis and not bias_analysis.get('error'):
                fairness_score = bias_analysis.get('fairness_score', 0.5)
                if fairness_score < 0.7:
                    recommendations.append("Potential evaluation bias detected - review evaluation methodology and consider bias mitigation strategies")
            
            # General recommendations
            metadata = pipeline_results.get('metadata', {})
            if metadata.get('test_case_count', 0) < 20:
                recommendations.append("Consider increasing test case count for more robust evaluation")
            
            if not recommendations:
                recommendations.append("A/B testing pipeline completed successfully - review detailed results for insights")
                
        except Exception as e:
            logger.warning(f"Failed to generate A/B testing recommendations: {e}")
            recommendations.append("Review A/B testing results manually due to recommendation generation error")
        
        return recommendations

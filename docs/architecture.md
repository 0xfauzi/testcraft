# TestCraft Architecture Guide

This document provides a comprehensive overview of TestCraft's architecture, design principles, and implementation patterns.

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [Clean Architecture Principles](#clean-architecture-principles)
3. [Core Components](#core-components)
4. [Domain Layer](#domain-layer)
5. [Application Layer](#application-layer)
6. [Ports (Interfaces)](#ports-interfaces)
7. [Adapters](#adapters)
8. [Evaluation Harness Architecture](#evaluation-harness-architecture)
9. [Configuration System](#configuration-system)
10. [Dependency Injection](#dependency-injection)
11. [Error Handling](#error-handling)
12. [Testing Strategy](#testing-strategy)
13. [Extension Points](#extension-points)

## Architectural Overview

TestCraft follows **Clean Architecture** principles, organizing code in concentric layers with clear dependency rules. The architecture promotes:

- **Independence**: Business logic is independent of frameworks, UI, and external services
- **Testability**: Core logic can be tested without external dependencies
- **Flexibility**: Easy to swap implementations without affecting business logic
- **Maintainability**: Clear separation of concerns and well-defined boundaries

```
┌─────────────────────────────────────────┐
│                  CLI                    │
├─────────────────────────────────────────┤
│               Adapters                  │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐  │
│  │   LLM   │ │    I/O   │ │ Coverage │  │
│  └─────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────┤
│         Application Services            │
│  ┌─────────────────┐ ┌────────────────┐ │
│  │  Generate Use   │ │  Evaluate Use  │ │
│  │   Case (Orch)   │ │     Cases      │ │
│  │  ┌──────────┐   │ └────────────────┘ │
│  │  │Generation│   │                    │
│  │  │ Services │   │                    │
│  │  └──────────┘   │                    │
│  └─────────────────┘                    │
├─────────────────────────────────────────┤
│            Domain Models                │
│  ┌─────────────────┐ ┌────────────────┐ │
│  │ Test Generation │ │   Evaluation   │ │
│  │    Models       │ │    Models      │ │
│  └─────────────────┘ └────────────────┘ │
└─────────────────────────────────────────┘
```

## Clean Architecture Principles

### Dependency Rule

Dependencies point inward. Outer layers depend on inner layers, never the reverse:

- **CLI** depends on **Application Services**
- **Adapters** depend on **Ports** (interfaces)
- **Application Services** depend on **Domain Models**
- **Domain Models** depend on nothing

### Layer Responsibilities

1. **Domain**: Business entities and rules
2. **Application**: Use cases and business workflows
3. **Ports**: Interface contracts
4. **Adapters**: External system integrations
5. **CLI**: User interface and command handling

## Core Components

### Project Structure

```
testcraft/
├── domain/              # 🏛️ Business entities and rules
│   └── models.py        #    Core domain models
├── application/         # 🔄 Use cases and workflows
│   ├── generate_usecase.py
│   ├── coverage_usecase.py
│   ├── analyze_usecase.py
│   └── status_usecase.py
├── ports/              # 🔌 Interface definitions
│   ├── llm_port.py
│   ├── coverage_port.py
│   ├── evaluation_port.py
│   └── ...
├── adapters/           # 🔧 External integrations
│   ├── llm/           # LLM provider adapters
│   ├── io/            # File system, UI, storage
│   ├── coverage/      # Test coverage analysis
│   ├── evaluation/    # Evaluation harness
│   └── ...
├── config/            # ⚙️ Configuration management
├── cli/               # 🖥️ Command-line interface
└── prompts/           # 📝 Prompt templates
```

## Domain Layer

The domain layer contains the core business entities and rules, independent of any external concerns.

### Core Models

```python
# testcraft/domain/models.py
@dataclass
class TestGenerationRequest:
    """Domain model for test generation requests."""
    source_files: List[Path]
    target_framework: TestFramework
    coverage_threshold: float
    include_fixtures: bool

@dataclass 
class GeneratedTest:
    """Domain model for generated tests."""
    content: str
    file_path: Path
    source_file: Path
    estimated_coverage: float
    confidence_score: float

@dataclass
class EvaluationCriteria:
    """Domain model for evaluation criteria."""
    acceptance_checks: bool
    llm_judge_enabled: bool
    rubric_dimensions: List[str]
    statistical_testing: bool
```

### Domain Rules

Business rules are encoded in domain models and services:

```python
class TestGenerationRules:
    """Domain rules for test generation."""
    
    @staticmethod
    def should_generate_fixture(source_code: str) -> bool:
        """Business rule: Generate fixtures for classes with complex setup."""
        return has_complex_initialization(source_code)
    
    @staticmethod
    def calculate_confidence_score(metrics: Dict[str, Any]) -> float:
        """Business rule: Calculate confidence based on multiple factors."""
        # Implementation of confidence calculation logic
        pass
```

## Application Layer

Application services orchestrate business workflows using domain models and ports.

### Use Case Pattern

```python
# testcraft/application/generate_usecase.py
class GenerateUseCase:
    """Use case for test generation workflow."""
    
    def __init__(
        self,
        llm_port: LLMPort,
        coverage_port: CoveragePort, 
        writer_port: WriterPort,
        ui_port: UIPort
    ):
        self.llm_port = llm_port
        self.coverage_port = coverage_port
        self.writer_port = writer_port
        self.ui_port = ui_port
    
    async def execute(self, request: TestGenerationRequest) -> GenerationResult:
        """Execute test generation workflow."""
        try:
            # 1. Validate request
            self._validate_request(request)
            
            # 2. Analyze source code
            analysis = await self.coverage_port.analyze_files(request.source_files)
            
            # 3. Generate tests using LLM
            generated_tests = await self.llm_port.generate_tests(
                source_files=request.source_files,
                analysis=analysis,
                config=request.config
            )
            
            # 4. Write tests to files
            written_files = await self.writer_port.write_tests(generated_tests)
            
            # 5. Update UI with progress
            await self.ui_port.display_results(written_files)
            
            return GenerationResult(
                generated_tests=generated_tests,
                written_files=written_files,
                success=True
            )
            
        except Exception as e:
            return GenerationResult(
                error=str(e),
                success=False
            )
```

### Modular Generation Architecture

TestCraft employs a **modular service architecture** for test generation, breaking down the complex workflow into focused, testable services. This design promotes maintainability, testability, and reusability.

```
testcraft/application/generation/
├── config.py                     # Centralized configuration management
└── services/
    ├── state_discovery.py        # State sync & file discovery
    ├── coverage_evaluator.py     # Coverage measurement & deltas
    ├── plan_builder.py           # File selection & plan creation  
    ├── content_builder.py        # Source extraction & test paths
    ├── context_assembler.py      # Unified context building
    ├── enrichment_detectors.py   # Context enrichment detection
    ├── batch_executor.py         # Concurrent test generation
    ├── pytest_refiner.py         # Test execution & refinement
    └── structure.py              # Directory tree building
```

#### Service Responsibilities

- **StateSyncDiscovery**: State sync and source file discovery
- **CoverageEvaluator**: Coverage measurement with graceful failure handling
- **PlanBuilder**: File selection decisions and TestGenerationPlan creation
- **ContentBuilder**: Source code extraction and test path determination
- **ContextAssembler**: Unified context building for generation and refinement
- **EnrichmentDetectors**: Detection of env/config, clients, fixtures, side-effects
- **BatchExecutor**: Concurrent execution with proper error aggregation
- **PytestRefiner**: Test execution, failure formatting, and refinement loops

#### Orchestrator Pattern

The `GenerateUseCase` class serves as a **thin orchestrator** that delegates to focused services while maintaining the original public API contract. This preserves backward compatibility while enabling modular testing and maintenance.

### Use Case Composition

Use cases are composed to handle complex workflows:

```python
class ComprehensiveTestingUseCase:
    """Composite use case for complete testing workflow."""
    
    def __init__(
        self,
        generate_usecase: GenerateUseCase,
        coverage_usecase: CoverageUseCase,
        evaluate_usecase: EvaluateUseCase
    ):
        self.generate = generate_usecase
        self.coverage = coverage_usecase
        self.evaluate = evaluate_usecase
    
    async def execute(self, request: ComprehensiveTestingRequest) -> ComprehensiveResult:
        """Execute complete testing workflow."""
        # 1. Generate tests
        generation_result = await self.generate.execute(request.generation_request)
        
        # 2. Analyze coverage
        coverage_result = await self.coverage.execute(request.coverage_request)
        
        # 3. Evaluate quality (if enabled)
        evaluation_result = None
        if request.evaluation_enabled:
            evaluation_result = await self.evaluate.execute(request.evaluation_request)
        
        return ComprehensiveResult(
            generation=generation_result,
            coverage=coverage_result,
            evaluation=evaluation_result
        )
```

## Ports (Interfaces)

Ports define contracts between the application core and external systems.

### Interface Design

```python
# testcraft/ports/llm_port.py
class LLMPort(Protocol):
    """Port for LLM provider integrations."""
    
    async def generate_tests(
        self,
        source_files: List[Path],
        analysis: CodeAnalysis,
        config: LLMConfig
    ) -> List[GeneratedTest]:
        """Generate tests for source files."""
        ...
    
    async def refine_tests(
        self,
        test_content: str,
        error_messages: List[str],
        config: LLMConfig
    ) -> str:
        """Refine tests based on error feedback."""
        ...
    
    def calculate_token_usage(self, content: str) -> TokenUsage:
        """Calculate token usage for content."""
        ...
```

### Port Benefits

1. **Testability**: Easy to create mock implementations
2. **Flexibility**: Swap implementations without changing business logic
3. **Clear Contracts**: Explicit expectations for implementations

## Adapters

Adapters implement ports and handle integration with external systems.

### LLM Adapters

```python
# testcraft/adapters/llm/openai.py
class OpenAIAdapter(LLMPort):
    """OpenAI implementation of LLM port."""
    
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    async def generate_tests(
        self,
        source_files: List[Path],
        analysis: CodeAnalysis,
        config: LLMConfig
    ) -> List[GeneratedTest]:
        """Generate tests using OpenAI GPT models."""
        # Implementation using OpenAI API
        pass
```

### Adapter Router Pattern

```python
# testcraft/adapters/llm/router.py
class LLMRouter:
    """Router for multiple LLM providers."""
    
    def __init__(self):
        self.providers: Dict[str, LLMPort] = {}
    
    def register(self, name: str, provider: LLMPort) -> None:
        """Register an LLM provider."""
        self.providers[name] = provider
    
    def get_provider(self, name: str) -> LLMPort:
        """Get provider by name."""
        if name not in self.providers:
            raise ValueError(f"Unknown provider: {name}")
        return self.providers[name]
    
    @classmethod
    def from_config(cls, config: LLMConfig) -> LLMPort:
        """Create router from configuration."""
        router = cls()
        
        # Register available providers
        if config.openai_api_key:
            router.register("openai", OpenAIAdapter(config.openai_api_key, config.openai_model))
        
        if config.anthropic_api_key:
            router.register("anthropic", ClaudeAdapter(config.anthropic_api_key, config.anthropic_model))
        
        # Return default provider
        return router.get_provider(config.default_provider)
```

## Evaluation Harness Architecture

The evaluation harness is a sophisticated system for assessing test quality using multiple evaluation methods.

### Evaluation Components

```python
# Architecture of evaluation system
┌─────────────────────────────────────────────┐
│            Evaluation Harness               │
├─────────────────────────────────────────────┤
│  TestEvaluationHarness                     │
│  ├─ Single Test Evaluation                 │
│  ├─ Batch Processing                       │
│  ├─ A/B Testing Pipeline                   │
│  ├─ Statistical Analysis                   │
│  ├─ Bias Detection                         │
│  └─ Campaign Management                    │
├─────────────────────────────────────────────┤
│         Evaluation Adapter                  │
│  ├─ Acceptance Checks                      │
│  ├─ LLM-as-Judge                          │
│  ├─ Statistical Testing                    │
│  └─ Bias Analysis                          │
├─────────────────────────────────────────────┤
│            Support Systems                  │
│  ├─ Artifact Storage                       │
│  ├─ State Management                       │
│  ├─ Prompt Registry                        │
│  └─ Result Aggregation                     │
└─────────────────────────────────────────────┘
```

### Evaluation Port

```python
# testcraft/ports/evaluation_port.py
class EvaluationPort(Protocol):
    """Port for test evaluation operations."""
    
    def evaluate_single(
        self,
        test_content: str,
        source_file: str,
        config: EvaluationConfig
    ) -> EvaluationResult:
        """Evaluate a single test."""
        ...
    
    def evaluate_pairwise(
        self,
        test_a: str,
        test_b: str,
        source_file: str,
        comparison_mode: str
    ) -> Dict[str, Any]:
        """Compare two test variants."""
        ...
    
    def run_statistical_significance_analysis(
        self,
        evaluation_data: List[Dict[str, Any]],
        analysis_type: str,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Run statistical analysis."""
        ...
```

### Evaluation Workflow

```python
class EvaluationWorkflow:
    """Orchestrates evaluation processes."""
    
    def __init__(
        self,
        evaluator: EvaluationPort,
        artifact_store: ArtifactStorePort,
        state_adapter: StatePort
    ):
        self.evaluator = evaluator
        self.artifact_store = artifact_store
        self.state_adapter = state_adapter
    
    async def run_ab_testing_campaign(
        self,
        prompt_variants: List[Dict[str, str]],
        test_dataset: List[Dict[str, str]],
        config: EvaluationConfig
    ) -> CampaignResult:
        """Run comprehensive A/B testing campaign."""
        campaign_id = self._generate_campaign_id()
        
        try:
            # 1. Execute evaluations for each variant
            variant_results = []
            for variant in prompt_variants:
                variant_result = await self._evaluate_variant(
                    variant, test_dataset, config
                )
                variant_results.append(variant_result)
                
                # Store intermediate results
                await self.artifact_store.store_artifact(
                    artifact_type="variant_result",
                    content=variant_result,
                    campaign_id=campaign_id
                )
            
            # 2. Statistical analysis
            statistical_result = await self.evaluator.run_statistical_significance_analysis(
                evaluation_data=variant_results,
                analysis_type="ab_testing",
                confidence_level=config.confidence_level
            )
            
            # 3. Bias detection
            bias_result = await self.evaluator.detect_evaluation_bias(
                evaluation_history=variant_results
            )
            
            # 4. Generate recommendations
            recommendations = self._generate_recommendations(
                variant_results, statistical_result, bias_result
            )
            
            campaign_result = CampaignResult(
                campaign_id=campaign_id,
                variant_results=variant_results,
                statistical_analysis=statistical_result,
                bias_analysis=bias_result,
                recommendations=recommendations
            )
            
            # Store final results
            await self.artifact_store.store_artifact(
                artifact_type="campaign_result",
                content=campaign_result,
                campaign_id=campaign_id
            )
            
            return campaign_result
            
        except Exception as e:
            await self._handle_campaign_error(campaign_id, e)
            raise
```

## Configuration System

The configuration system provides flexible, hierarchical configuration management.

### Configuration Architecture

```python
# Configuration loading hierarchy
┌─────────────────────────────────┐
│        Environment Variables     │  ← Highest priority
├─────────────────────────────────┤
│       Command Line Args         │
├─────────────────────────────────┤
│      Project .testcraft.toml    │
├─────────────────────────────────┤
│     pyproject.toml [tool.testcraft] │
├─────────────────────────────────┤
│    Global ~/.config/testcraft/  │
├─────────────────────────────────┤
│          Default Values         │  ← Lowest priority
└─────────────────────────────────┘
```

### Configuration Models

```python
# testcraft/config/models.py
@dataclass
class TestcraftConfig:
    """Root configuration model."""
    style: StyleConfig
    coverage: CoverageConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    llm: LLMConfig
    
    @classmethod
    def from_toml(cls, toml_path: Path) -> 'TestcraftConfig':
        """Load configuration from TOML file."""
        pass

@dataclass
class EvaluationConfig:
    """Evaluation-specific configuration."""
    enabled: bool = False
    acceptance_checks: bool = True
    llm_judge_enabled: bool = False
    rubric_dimensions: List[str] = field(default_factory=lambda: ["correctness", "coverage"])
    statistical_testing: bool = True
    confidence_level: float = 0.95
```

### Configuration Loader

```python
# testcraft/config/loader.py
class ConfigLoader:
    """Hierarchical configuration loader."""
    
    @classmethod
    def load_config(cls, project_root: Path) -> TestcraftConfig:
        """Load configuration from multiple sources."""
        # 1. Start with defaults
        config = TestcraftConfig.default()
        
        # 2. Load global config
        global_config = cls._load_global_config()
        if global_config:
            config = cls._merge_configs(config, global_config)
        
        # 3. Load project config
        project_config = cls._load_project_config(project_root)
        if project_config:
            config = cls._merge_configs(config, project_config)
        
        # 4. Apply environment overrides
        config = cls._apply_env_overrides(config)
        
        # 5. Validate final configuration
        cls._validate_config(config)
        
        return config
```

## Dependency Injection

TestCraft uses constructor-based dependency injection for loose coupling and testability.

### Dependency Container

```python
# testcraft/cli/dependency_injection.py
class DependencyContainer:
    """Container for managing dependencies."""
    
    def __init__(self, config: TestcraftConfig):
        self.config = config
        self._instances: Dict[str, Any] = {}
    
    def get_llm_adapter(self) -> LLMPort:
        """Get LLM adapter instance."""
        if 'llm_adapter' not in self._instances:
            self._instances['llm_adapter'] = LLMRouter.from_config(self.config.llm)
        return self._instances['llm_adapter']
    
    def get_coverage_adapter(self) -> CoveragePort:
        """Get coverage adapter instance."""
        if 'coverage_adapter' not in self._instances:
            self._instances['coverage_adapter'] = TestcraftCoverageAdapter()
        return self._instances['coverage_adapter']
    
    def get_generate_usecase(self) -> GenerateUseCase:
        """Get test generation use case."""
        return GenerateUseCase(
            llm_port=self.get_llm_adapter(),
            coverage_port=self.get_coverage_adapter(),
            writer_port=self.get_writer_adapter(),
            ui_port=self.get_ui_adapter()
        )
```

### Service Registration

```python
def create_dependency_container(config: TestcraftConfig) -> DependencyContainer:
    """Factory function for creating dependency container."""
    container = DependencyContainer(config)
    
    # Register adapters based on configuration
    if config.evaluation.enabled:
        container.register('evaluation_harness', TestEvaluationHarness(
            config=config,
            coverage_adapter=container.get_coverage_adapter(),
            llm_adapter=container.get_llm_adapter()
        ))
    
    return container
```

## Error Handling

TestCraft uses a structured approach to error handling with custom exception types and error recovery strategies.

### Exception Hierarchy

```python
# testcraft/domain/exceptions.py
class TestcraftError(Exception):
    """Base exception for all TestCraft errors."""
    pass

class ConfigurationError(TestcraftError):
    """Configuration-related errors."""
    pass

class LLMError(TestcraftError):
    """LLM provider errors."""
    
    def __init__(self, message: str, provider: str, retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable

class EvaluationError(TestcraftError):
    """Evaluation system errors."""
    pass

class CoverageAnalysisError(TestcraftError):
    """Coverage analysis errors."""
    pass
```

### Error Recovery

```python
class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, config: TestcraftConfig):
        self.config = config
        self.retry_config = config.error_handling
    
    async def handle_llm_error(self, error: LLMError, context: Dict[str, Any]) -> Any:
        """Handle LLM provider errors with retry logic."""
        if not error.retryable:
            raise error
        
        retry_count = context.get('retry_count', 0)
        if retry_count >= self.retry_config.max_retries:
            raise error
        
        # Exponential backoff
        delay = self.retry_config.base_delay * (2 ** retry_count)
        await asyncio.sleep(delay)
        
        # Retry with alternative provider if available
        if self.retry_config.fallback_provider:
            context['provider'] = self.retry_config.fallback_provider
        
        context['retry_count'] = retry_count + 1
        return await self._retry_operation(context)
```

## Testing Strategy

TestCraft follows a comprehensive testing strategy with different test types for different layers.

### Test Pyramid

```
        ┌─────────────────┐
        │   E2E Tests     │  ← CLI integration, full workflows
        │    (Slow)       │
        ├─────────────────┤
        │Integration Tests│  ← Adapter integration, use cases
        │   (Medium)      │
        ├─────────────────┤
        │   Unit Tests    │  ← Domain logic, individual components
        │    (Fast)       │
        └─────────────────┘
```

### Unit Tests

```python
# tests/test_domain_models.py
class TestGenerationRules:
    """Test domain business rules."""
    
    def test_should_generate_fixture_for_complex_class(self):
        """Test fixture generation rule."""
        source_code = """
        class ComplexService:
            def __init__(self, db_connection, cache, logger):
                self.db = db_connection
                self.cache = cache
                self.logger = logger
        """
        
        result = TestGenerationRules.should_generate_fixture(source_code)
        
        assert result is True
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation."""
        metrics = {
            'syntax_valid': True,
            'imports_work': True,
            'tests_pass': True,
            'coverage_improvement': 85.0
        }
        
        score = TestGenerationRules.calculate_confidence_score(metrics)
        
        assert 0.8 <= score <= 1.0
```

### Integration Tests

```python
# tests/test_evaluation_integration.py
class TestEvaluationIntegration:
    """Integration tests for evaluation harness."""
    
    @pytest.fixture
    def harness(self, tmp_path):
        """Create evaluation harness for testing."""
        config = TestcraftConfig.default()
        config.evaluation.enabled = True
        
        return TestEvaluationHarness(
            config=config,
            project_root=tmp_path
        )
    
    async def test_single_evaluation_workflow(self, harness):
        """Test complete single evaluation workflow."""
        test_content = """
        def test_add():
            assert add(2, 3) == 5
        """
        
        result = harness.evaluate_single_test(
            test_content=test_content,
            source_file="src/calculator.py"
        )
        
        assert result.acceptance.syntactically_valid
        assert result.acceptance.imports_successfully
        assert result.llm_judge.overall_score > 0
```

### CLI Tests

```python
# tests/test_cli_integration.py
class TestCLIIntegration:
    """End-to-end CLI tests."""
    
    def test_evaluation_ab_test_command(self, tmp_path, runner):
        """Test A/B testing CLI command."""
        # Setup test configuration
        config_file = tmp_path / "ab_config.json"
        config_file.write_text(json.dumps({
            "prompt_variants": [...],
            "test_dataset": [...]
        }))
        
        # Run command
        result = runner.invoke(app, [
            'evaluation', 'ab-test', str(config_file),
            '--statistical-testing'
        ])
        
        assert result.exit_code == 0
        assert "A/B testing pipeline completed" in result.output
```

## Extension Points

TestCraft provides several extension points for customization.

### Custom Adapters

```python
# Create custom LLM adapter
class CustomLLMAdapter(LLMPort):
    """Custom LLM provider implementation."""
    
    async def generate_tests(self, source_files, analysis, config):
        """Custom test generation logic."""
        # Your implementation here
        pass

# Register with router
llm_router.register("custom", CustomLLMAdapter())
```

### Custom Evaluation Metrics

```python
# Create custom evaluation adapter
class CustomEvaluationAdapter(EvaluationPort):
    """Custom evaluation metrics."""
    
    def evaluate_single(self, test_content, source_file, config):
        """Add custom evaluation criteria."""
        base_result = super().evaluate_single(test_content, source_file, config)
        
        # Add custom metrics
        custom_score = self._calculate_business_logic_coverage(test_content)
        base_result.custom_metrics = {'business_logic_coverage': custom_score}
        
        return base_result
```

### Custom CLI Commands

```python
# Add custom CLI commands
@app.command()
def custom_analysis(
    source_path: Path,
    output_format: str = "json"
):
    """Custom analysis command."""
    # Your custom logic here
    pass
```

### Plugin System

```python
# Plugin interface
class TestcraftPlugin(Protocol):
    """Interface for TestCraft plugins."""
    
    def name(self) -> str:
        """Plugin name."""
        ...
    
    def initialize(self, container: DependencyContainer) -> None:
        """Initialize plugin with dependency container."""
        ...
    
    def register_adapters(self, container: DependencyContainer) -> None:
        """Register plugin adapters."""
        ...
```

## Best Practices

### 1. Dependency Management

- Use constructor injection for dependencies
- Depend on interfaces, not implementations
- Keep constructors simple - just store dependencies

### 2. Error Handling

- Use specific exception types
- Include context in error messages
- Implement retry logic for transient failures

### 3. Testing

- Test business logic in isolation
- Use integration tests for adapter interactions
- Mock external dependencies in unit tests

### 4. Extension

- Design for extension through interfaces
- Use composition over inheritance
- Keep extension points well-documented

### 5. Configuration

- Validate configuration early
- Use environment-specific configurations
- Document configuration options clearly

---

This architecture enables TestCraft to be maintainable, testable, and extensible while maintaining clear separation of concerns and following established software engineering principles.

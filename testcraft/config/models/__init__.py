"""Configuration models for TestCraft.

This package contains all configuration models organized by concern.
All models are re-exported here to maintain backward compatibility.
"""

# Main configuration model
from .main import TestCraftConfig

# Discovery and patterns
from .discovery import TestDiscoveryConfig, TestPatternConfig

# Environment configuration
from .environment import (
    EnvironmentConfig,
    EnvironmentOverrideConfig,
    TestEnvironmentConfig,
)

# Evaluation and LLM providers
from .evaluation import (
    CostConfig,
    CostThresholdConfig,
    EvaluationConfig,
    LLMProviderConfig,
)

# Generation configuration
from .generation import (
    MergeConfig,
    PostGenerationTestRunnerConfig,
    TestGenerationConfig,
)

# Prompt budget and context
from .prompt_budget import (
    ContextBudgetConfig,
    ContextCategoriesConfig,
    ContextEnrichmentConfig,
    DirectoryTreeBudgetConfig,
    PromptBudgetConfig,
)

# Refinement configuration
from .refine import RefineConfig

# Telemetry configuration
from .telemetry import TelemetryBackendConfig, TelemetryConfig

# UI and logging configuration
from .ui import LoggingConfig, PlanningConfig, UIConfig

# Deprecated models (for backward compatibility)
from .deprecated import CoverageConfig, QualityConfig, TestStyleConfig

__all__ = [
    # Main configuration
    "TestCraftConfig",
    
    # Discovery and patterns
    "TestDiscoveryConfig",
    "TestPatternConfig",
    
    # Environment
    "EnvironmentConfig",
    "EnvironmentOverrideConfig", 
    "TestEnvironmentConfig",
    
    # Evaluation and LLM
    "CostConfig",
    "CostThresholdConfig",
    "EvaluationConfig",
    "LLMProviderConfig",
    
    # Generation
    "MergeConfig",
    "PostGenerationTestRunnerConfig",
    "TestGenerationConfig",
    
    # Prompt budget and context
    "ContextBudgetConfig",
    "ContextCategoriesConfig",
    "ContextEnrichmentConfig",
    "DirectoryTreeBudgetConfig",
    "PromptBudgetConfig",
    
    # Refinement
    "RefineConfig",
    
    # Telemetry
    "TelemetryBackendConfig",
    "TelemetryConfig",
    
    # UI and logging
    "LoggingConfig",
    "PlanningConfig",
    "UIConfig",
    
    # Deprecated (backward compatibility)
    "CoverageConfig",
    "QualityConfig", 
    "TestStyleConfig",
]

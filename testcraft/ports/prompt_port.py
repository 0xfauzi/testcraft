"""
Prompt Port interface definition.

This module defines the interface for prompt management operations,
including system prompts, user prompts, and schema generation.
"""

from typing import Dict, Any, Optional, Union
from typing_extensions import Protocol


class PromptPort(Protocol):
    """
    Interface for prompt management operations.
    
    This protocol defines the contract for prompt operations, including
    system prompt generation, user prompt creation, and schema management.
    """
    
    def get_system_prompt(
        self,
        prompt_type: str = "test_generation",
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> str:
        """
        Get a system prompt for the specified operation type.
        
        Args:
            prompt_type: Type of prompt to generate (test_generation, analysis, etc.)
            context: Optional context information to include in the prompt
            **kwargs: Additional prompt parameters
            
        Returns:
            System prompt string
            
        Raises:
            PromptError: If prompt generation fails
        """
        ...
    
    def get_user_prompt(
        self,
        prompt_type: str = "test_generation",
        code_content: str = "",
        additional_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> str:
        """
        Get a user prompt for the specified operation type.
        
        Args:
            prompt_type: Type of prompt to generate
            code_content: Code content to include in the prompt
            additional_context: Additional context information
            **kwargs: Additional prompt parameters
            
        Returns:
            User prompt string
            
        Raises:
            PromptError: If prompt generation fails
        """
        ...
    
    def get_schema(
        self,
        schema_type: str = "test_structure",
        language: str = "python",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get a schema definition for the specified type and language.
        
        Args:
            schema_type: Type of schema to generate
            language: Programming language for the schema
            **kwargs: Additional schema parameters
            
        Returns:
            Dictionary containing:
                - 'schema': Schema definition
                - 'examples': Example usage of the schema
                - 'validation_rules': Rules for validating against the schema
                - 'metadata': Additional schema metadata
                
        Raises:
            PromptError: If schema generation fails
        """
        ...
    
    def customize_prompt(
        self,
        base_prompt: str,
        customizations: Dict[str, Any],
        **kwargs: Any
    ) -> str:
        """
        Customize a base prompt with specific customizations.
        
        Args:
            base_prompt: The base prompt to customize
            customizations: Dictionary of customizations to apply
            **kwargs: Additional customization parameters
            
        Returns:
            Customized prompt string
            
        Raises:
            PromptError: If prompt customization fails
        """
        ...
    
    def validate_prompt(
        self,
        prompt: str,
        prompt_type: str = "general",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Validate a prompt for correctness and completeness.
        
        Args:
            prompt: The prompt to validate
            prompt_type: Type of prompt being validated
            **kwargs: Additional validation parameters
            
        Returns:
            Dictionary containing:
                - 'is_valid': Whether the prompt is valid
                - 'issues': List of validation issues found
                - 'suggestions': List of improvement suggestions
                - 'validation_metadata': Additional validation information
                
        Raises:
            PromptError: If prompt validation fails
        """
        ...

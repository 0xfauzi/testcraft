"""
Context Port interface definition.

This module defines the interface for context management operations,
including indexing, retrieval, and summarization of code context.
"""

from typing import Dict, Any, Optional, List, Union
from typing_extensions import Protocol
from pathlib import Path


class ContextPort(Protocol):
    """
    Interface for context management operations.
    
    This protocol defines the contract for context operations, including
    indexing code context, retrieving relevant information, and
    summarizing code relationships.
    """
    
    def index(
        self,
        file_path: Union[str, Path],
        content: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Index a file's context for later retrieval.
        
        Args:
            file_path: Path to the file to index
            content: Optional file content (if not provided, will be read from file)
            **kwargs: Additional indexing parameters
            
        Returns:
            Dictionary containing:
                - 'index_id': Unique identifier for the indexed content
                - 'elements_indexed': Number of elements indexed
                - 'context_summary': Brief summary of the indexed context
                - 'metadata': Additional indexing metadata
                
        Raises:
            ContextError: If indexing fails
        """
        ...
    
    def retrieve(
        self,
        query: str,
        context_type: str = "general",
        limit: int = 10,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context based on a query.
        
        Args:
            query: Search query for context retrieval
            context_type: Type of context to retrieve (general, function, class, etc.)
            limit: Maximum number of results to return
            **kwargs: Additional retrieval parameters
            
        Returns:
            Dictionary containing:
                - 'results': List of relevant context items
                - 'relevance_scores': Relevance scores for each result
                - 'total_found': Total number of results found
                - 'query_metadata': Metadata about the query processing
                
        Raises:
            ContextError: If retrieval fails
        """
        ...
    
    def summarize(
        self,
        file_path: Union[str, Path],
        summary_type: str = "comprehensive",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate a summary of a file's context and purpose.
        
        Args:
            file_path: Path to the file to summarize
            summary_type: Type of summary to generate
            **kwargs: Additional summarization parameters
            
        Returns:
            Dictionary containing:
                - 'summary': Generated summary text
                - 'key_functions': List of key functions identified
                - 'key_classes': List of key classes identified
                - 'dependencies': List of key dependencies
                - 'summary_metadata': Metadata about the summarization
                
        Raises:
            ContextError: If summarization fails
        """
        ...
    
    def get_related_context(
        self,
        file_path: Union[str, Path],
        relationship_type: str = "all",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get context related to a specific file.
        
        Args:
            file_path: Path to the file to find related context for
            relationship_type: Type of relationships to find (imports, usage, etc.)
            **kwargs: Additional parameters for relationship discovery
            
        Returns:
            Dictionary containing:
                - 'related_files': List of related file paths
                - 'relationships': Description of relationships found
                - 'usage_context': Context about how the file is used
                - 'dependency_context': Context about file dependencies
                
        Raises:
            ContextError: If context retrieval fails
        """
        ...
    
    def build_context_graph(
        self,
        project_root: Union[str, Path],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Build a context graph for the entire project.
        
        Args:
            project_root: Root directory of the project
            **kwargs: Additional graph building parameters
            
        Returns:
            Dictionary containing:
                - 'graph': Graph representation of project context
                - 'nodes': List of nodes in the graph
                - 'edges': List of edges in the graph
                - 'graph_metadata': Metadata about the graph structure
                
        Raises:
            ContextError: If graph building fails
        """
        ...

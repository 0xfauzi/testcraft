"""
Writer Port interface definition.

This module defines the interface for file writing operations,
including test file creation and content management.
"""

from typing import Dict, Any, Optional, Union
from typing_extensions import Protocol
from pathlib import Path


class WriterPort(Protocol):
    """
    Interface for file writing operations.
    
    This protocol defines the contract for writing files, including
    test files, reports, and other generated content.
    """
    
    def write_file(
        self,
        file_path: Union[str, Path],
        content: str,
        overwrite: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            file_path: Path where the file should be written
            content: Content to write to the file
            overwrite: Whether to overwrite existing files
            **kwargs: Additional writing parameters
            
        Returns:
            Dictionary containing:
                - 'success': Whether the write operation succeeded
                - 'bytes_written': Number of bytes written
                - 'file_path': Path of the written file
                - 'backup_path': Path of backup if file was overwritten
                
        Raises:
            WriterError: If file writing fails
        """
        ...
    
    def write_test_file(
        self,
        test_path: Union[str, Path],
        test_content: str,
        source_file: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Write a test file with proper test file conventions.
        
        Args:
            test_path: Path where the test file should be written
            test_content: Test content to write
            source_file: Optional path to the source file being tested
            **kwargs: Additional parameters for test file writing
            
        Returns:
            Dictionary containing:
                - 'success': Whether the write operation succeeded
                - 'test_path': Path of the written test file
                - 'imports_added': List of imports that were added
                - 'test_functions': List of test function names found
                
        Raises:
            WriterError: If test file writing fails
        """
        ...
    
    def backup_file(
        self,
        file_path: Union[str, Path],
        backup_suffix: str = ".backup"
    ) -> Dict[str, Any]:
        """
        Create a backup of an existing file.
        
        Args:
            file_path: Path of the file to backup
            backup_suffix: Suffix to add to the backup filename
            
        Returns:
            Dictionary containing:
                - 'success': Whether the backup succeeded
                - 'backup_path': Path of the created backup
                - 'original_path': Path of the original file
                
        Raises:
            WriterError: If backup creation fails
        """
        ...
    
    def ensure_directory(
        self,
        directory_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Ensure that a directory exists, creating it if necessary.
        
        Args:
            directory_path: Path of the directory to ensure exists
            
        Returns:
            Dictionary containing:
                - 'success': Whether the directory exists or was created
                - 'directory_path': Path of the directory
                - 'created': Whether the directory was created
                
        Raises:
            WriterError: If directory creation fails
        """
        ...

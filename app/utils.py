"""
Utility functions for validation, error handling, and common operations.
"""

from typing import Optional,List
import re
from pathlib import Path
from fastapi import HTTPException,UploadFile

class ValidationError(Exception):
        """Custom validation error exception."""
        pass

class FileHandler:
    """Validates uploaded files."""

    # Allowed file extensions and their MIME types
    ALLOWED_EXTENSIONS = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.txt': 'text/plain'
    }

    # Maximum file size (50 MB)
    MAX_FILE_SIZE = 50 * 1024 *1024 # 50 MB in bytes

    @staticmethod
    def validate_file(file: UploadFile) -> None:
        """
        Validate uploaded file for type and size.

        Args:
            file: The uploaded file to validate

        Raises:
            ValidationError: If file is invalid
        """

        if not file or file.filename:
            raise ValidationError("No file or filename is provided!!!")
        
        # Check file extension
        file_ext = '.' + file.filename.rsplit('.',1)[-1].lower() if '.' in file.filename else ''

        if file_ext not in FileHandler.ALLOWED_EXTENSIONS:
            allowed_ext = ','.join(FileHandler.ALLOWED_EXTENSIONS.keys())
            raise ValidationError(
                 f"Invalid File type: {file_ext}. Allowed File types: {allowed_ext}"
            )
        
        # Check file size (if available)
        if hasattr(file, 'size') and file.size:
             if file.size > FileHandler.MAX_FILE_SIZE:
                  max_mb = FileHandler.MAX_FILE_SIZE/(1024*1024)
                  raise ValidationError(
                       f"file Size has exceeded the maximum allowed size of {max_mb:.0f} mb"
                  )
             
        @staticmethod
        def get_file_extention(filename: str) -> str:
             """Get the file extension from filename."""
             return Path(filename).suffix.lower()
        
class QueryValidator:
    """Validates query inputs."""

    # Minimum and maximum question length
    MIN_QUESTION_LENGTH = 3
    MAX_QUESTION_LENGTH = 1000

    # SQL keywords that might indicate dangerous operations
    DANGEROUS_SQL_PATTERN = [
        r'\bDROP\s+TABLE\b',
        r'\bDELETE\s+FROM\b',
        r'\bTRUNCATE\b',
        r'\bALTER\s+TABLE\b',
        r'\bCREATE\s+TABLE\b',
        r'\bINSERT\s+INTO\b',
        r'\bUPDATE\s+\w+\s+SET\b'
    ]

    @staticmethod
    def validate_question(question: str, allow_empty: bool = False) -> str:
        """
        Validate a question string.

        Args:
            question: The question to validate
            allow_empty: Whether to allow empty questions

        Returns:
            Cleaned question string

        Raises:
            ValidationError: If question is invalid
        """

        if not question or not question.strip():
            if allow_empty:
                return ""
            raise ValidationError(
                 f"Question cannot be empty!!!"
            )
        
        question = question.strip()

        # Check length
        if len(question) < QueryValidator.MIN_QUESTION_LENGTH:
             raise ValidationError(
                  f"Question is too short (minimum {QueryValidator.MIN_QUESTION_LENGTH} characters)"
             )
        
        if len(question) > QueryValidator.MAX_QUESTION_LENGTH:
             raise ValidationError(
                  f"Question is too big (maximum {QueryValidator.MAX_QUESTION_LENGTH} characters)"
             )
        
        return question
    
    @staticmethod
    def validate_top_k(top_k: int):
        """
        Validate top_k parameter for retrieval.

        Args:
            top_k: Number of chunks to retrieve

        Returns:
            Validated top_k value

        Raises:
            ValidationError: If top_k is invalid
        """
        if not isinstance(top_k, int):
             raise ValidationError("top_k must be an integer")
        
        if top_k < 1:
             raise ValidationError("top_k must atleast be 1")
        
        if top_k > 5:
             raise ValidationError("top_k cannot exceed 5")

        return top_k
    
    @staticmethod
    def check_dangerous_sql(sql: str) -> bool:
        """
        Check if SQL contains potentially dangerous operations.

        Args:
            sql: SQL query to check

        Returns:
            True if dangerous patterns found, False otherwise
        """
        sql_upper = sql.upper()

        for pattern in QueryValidator.DANGEROUS_SQL_PATTERN:
             if re.search(pattern, sql_upper, re.IGNORECASE):
                  return True
             
             return False
        
    @staticmethod
    def sanitize_sql_for_display(sql: str) -> str:
        """
        Sanitize SQL for safe display (remove comments, normalize whitespace).

        Args:
            sql: SQL query to sanitize

        Returns:
            Sanitized SQL string
        """
        # Remove SQL comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

        # Normalize whitespace
        sql = ' '.join(sql.split())

        return sql.strip()
    
class ErrorResponse:
    """Structured error response generator."""

    @staticmethod
    def validation_error(message: str, field: str | None = None) -> dict:
        """Generate validation error response."""
        response = {
             "error": "Validation Error",
             "message": message,
             "type": "validation error"
        }

        if field:
            response['field'] = field

        return response
    
    @staticmethod
    def service_unavailable(service_name: str, reason: Optional[str] = None) -> dict:
        """Generate service unavailable error response."""
        message = f"{service_name} is not available"
        if reason:
             message += f": {reason}"
        
        return {
             "error": "Service Unavailable",
             "message": message,
             "service name": service_name,
             "type": "service unavailable"
        }
    
    @staticmethod
    def internal_error(operation: str, error: Exception) -> dict:
         """Generate internal error response."""
         return {
              "error": "Internal Error",
              "operation": operation,
              "details": str(error),
              "type": "internal error"
         }
    
def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted string (e.g., "2.5 MB")
    """
    
    for unit in ["B", "KB", "MB", "GB"]:
         if size_bytes < 1024.0:
              return f"{size_bytes:.1f} {unit}"
         
         size_bytes /= 1024

    return f"{size_bytes:.1f} TB"

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) < max_length:
         return text
    
    return text[:max_length - len(suffix)] + suffix


     

"""
Abstract storage interface - defines what ALL storage backends must do.

This module provides the base interface that both LocalStorage and S3Storage
implement.It defines a contract (or set of rules) that every storage system must follow.
This allows switching between local files and S3 by changing
one configuration variable.
"""

from abc import ABC,abstractmethod

from typing import List,Dict
from pathlib import Path
import numpy as np

class StorageBackend(ABC):
    """
    Abstract interface for document cache storage.

    Both LocalStorage and S3Storage implement this interface.
    This allows the storage backend to be switched through configuration
    without changing the rest of the application.

    Storage layout is document-ID based, with the file extension passed
    explicitly because it is required by the storage implementations.
    """

    @abstractmethod
    def exists(self, document_id: str, file_extention: str) ->bool:
        """
        Check if all cache files exist for a document.

        Args:
            document_id: SHA-256 hash of document content
            file_extension: File type (pdf, txt, md, docx, etc.)

        Returns:
            True if all 4 files exist (document, chunks, embeddings, metadata)
            The original document file is optional.

            A document is considered cached only when all required artifacts 
            are present;
            - original document
            - chunks.json
            - embeddings.npy
            - metadata.json

        Note: This checks for ALL files - if any are missing, returns False
        """
        pass

    @abstractmethod
    def save_document(self, document_id: str, file_path: Path, file_extention: str) -> None:
        """
        Save original document file to storage.

        Args:
            document_id: SHA-256 hash of document
            file_path: Path to the uploaded file
            file_extension: File extension (pdf, txt, md, docx, etc.)

        Raises:
            Exception if save fails
        """
        pass

    @abstractmethod
    def save_chunks(self, document_id: str, file_extention: str, chunks: List[Dict]) ->None:
        """
        Save chunks.json to storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (needed for S3 folder organization)
            chunks: List of document chunks with text and metadata

        Raises:
            Exception if save fails
        """
        pass

    @abstractmethod
    def save_embeddings(self, document_id: str, file_extention: str, embeddings: np.ndarray) -> None:
        """
        Save embeddings.npy to storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension
            embeddings: NumPy array of shape (num_chunks, 1536)

        Raises:
            Exception if save fails
        """
        pass

    @abstractmethod
    def save_metadata(self, document_id: str, file_extention: str, metadata: Dict) -> None:
        """
        Save metadata.json to storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension
            metadata: Document metadata (filename, timestamps, token counts)

        Raises:
            Exception if save fails
        """
        pass

    @abstractmethod
    def load_chunks(self, document_id: str, file_extention: str) -> List[Dict]:
        """
        Load chunks.json from storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (needed to locate S3 folder)

        Returns:
            List of document chunks

        Raises:
            Exception if file not found or load fails
        """
        pass

    @abstractmethod
    def load_embeddings(self, document_id: str, file_extention: str) -> np.ndarray:
        """
        Load embeddings.npy from storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension

        Returns:
            NumPy array of shape (num_chunks, 1536)

        Raises:
            Exception if file not found or load fails
        """
        pass

    @abstractmethod
    def load_metadata(self, document_id: str, file_extention: str) -> Dict:
        """
        Load metadata.json from storage.

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension

        Returns:
            Document metadata dictionary

        Raises:
            Exception if file not found or load fails
        """
        pass

    @abstractmethod
    def delete(self, document_id: str, file_extention: str) -> None:
        """
        Delete all files for a document from storage.

        Deletes all 4 files: document.{ext}, chunks.json, embeddings.npy, metadata.json

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension

        Raises:
            Exception if delete fails
        """
        pass

    @abstractmethod
    def list_documents(self) ->List[str]:
        """
        List all cached document IDs.

        Returns:
            List of document IDs (SHA-256 hashes)

        Note: This scans storage to find all cached documents
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with stats like:
            - backend: "local" or "s3"
            - total_documents: count
            - total_size_mb: storage used
            - backend-specific info
        """
        pass
    
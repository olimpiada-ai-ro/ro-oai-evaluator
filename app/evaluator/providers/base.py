from abc import ABC, abstractmethod
from typing import Dict, Any
from app.evaluator.schemas.cache import FileMetadata


class DataSourceError(Exception):
    """Base exception for datasource operations."""
    pass


class AuthenticationError(DataSourceError):
    """Raised when authentication fails."""
    pass


class DataSourceNotFoundError(DataSourceError):
    """Raised when dataset or bucket is not found."""
    pass


class DataSourceConnectionError(DataSourceError):
    """Raised when connection to datasource fails."""
    pass


class DataSourceProvider(ABC):
    """
    Abstract base class for all cloud provider implementations.
    
    This interface defines the contract that all datasource providers
    must implement to support the evaluator service.
    """
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with the cloud provider using provided credentials.
        
        Args:
            credentials: Dictionary containing provider-specific credentials
            
        Returns:
            True if authentication successful
            
        Raises:
            AuthenticationError: If credentials are invalid or expired
        """
        pass
    
    @abstractmethod
    async def fetch_dataset(self, path: str) -> bytes:
        """
        Fetch dataset content from the specified path.
        
        Args:
            path: Provider-specific path to the dataset
            
        Returns:
            Dataset content as bytes
            
        Raises:
            DataSourceNotFoundError: If dataset path doesn't exist
            DataSourceConnectionError: If connection fails
            AuthenticationError: If authentication is required but failed
        """
        pass
    
    @abstractmethod
    async def get_file_metadata(self, path: str) -> FileMetadata:
        """
        Get metadata for the file at the specified path.
        
        Args:
            path: Provider-specific path to the file
            
        Returns:
            FileMetadata object with ETag, timestamps, and size
            
        Raises:
            DataSourceNotFoundError: If file path doesn't exist
            DataSourceConnectionError: If connection fails
            AuthenticationError: If authentication is required but failed
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate that the connection to the datasource is working.
        
        Returns:
            True if connection is valid
            
        Raises:
            DataSourceConnectionError: If connection validation fails
        """
        pass
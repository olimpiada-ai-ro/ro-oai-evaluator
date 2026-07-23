from typing import Dict, Any
import logging

from app.evaluator.providers.base import DataSourceProvider, DataSourceError
from app.evaluator.providers.aws import AWSDataSourceProvider
from app.evaluator.providers.remote_url import RemoteURLProvider
from app.evaluator.schemas.evaluation import DataSourceProvider as ProviderEnum

logger = logging.getLogger(__name__)


class UnsupportedProviderError(DataSourceError):
    """Raised when an unsupported provider type is requested."""
    pass


class DataSourceBuilder:
    """
    Factory class for creating datasource provider instances.
    
    This builder implements the factory pattern to abstract provider creation
    and support extensibility for new cloud providers.
    """
    
    # Registry of supported providers
    _providers = {
        ProviderEnum.AWS: AWSDataSourceProvider,
        ProviderEnum.REMOTE_URL: RemoteURLProvider,
    }
    
    @staticmethod
    def create_provider(provider_type: str, credentials: Dict[str, Any]) -> DataSourceProvider:
        """
        Create a datasource provider instance for the specified type.
        
        Args:
            provider_type: Type of provider ("aws", etc.)
            credentials: Provider-specific credentials
            
        Returns:
            Configured DataSourceProvider instance
            
        Raises:
            UnsupportedProviderError: If provider type is not supported
            DataSourceError: If provider creation fails
        """
        try:
            # Validate provider type - check if it's a valid enum value
            try:
                provider_enum = ProviderEnum(provider_type)
            except ValueError:
                supported_providers = [p.value for p in ProviderEnum]
                raise UnsupportedProviderError(
                    f"Unsupported datasource provider: {provider_type}. "
                    f"Supported providers: {supported_providers}"
                )
            
            # Get provider class from registry
            provider_class = DataSourceBuilder._providers.get(provider_enum)
            if not provider_class:
                raise UnsupportedProviderError(f"Provider implementation not found: {provider_enum}")
            
            # Create provider instance
            provider = provider_class()
            
            logger.info(f"Created {provider_type} datasource provider")
            return provider
            
        except UnsupportedProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {str(e)}")
            raise DataSourceError(f"Provider creation failed: {str(e)}")
    
    @staticmethod
    def get_supported_providers() -> list[str]:
        """
        Get list of supported provider types.
        
        Returns:
            List of supported provider type strings
        """
        return [provider.value for provider in ProviderEnum]
    
    @staticmethod
    def register_provider(provider_type: ProviderEnum, provider_class: type):
        """
        Register a new provider implementation.
        
        This method allows extending the builder with new providers
        without modifying the core builder code.
        
        Args:
            provider_type: Provider enum value
            provider_class: Provider implementation class
            
        Raises:
            ValueError: If provider_class doesn't implement DataSourceProvider
        """
        if not issubclass(provider_class, DataSourceProvider):
            raise ValueError("Provider class must implement DataSourceProvider interface")
        
        DataSourceBuilder._providers[provider_type] = provider_class
        logger.info(f"Registered new provider: {provider_type.value}")
    
    @staticmethod
    def is_provider_supported(provider_type: str) -> bool:
        """
        Check if a provider type is supported.
        
        Args:
            provider_type: Provider type string to check
            
        Returns:
            True if provider is supported
        """
        try:
            return ProviderEnum(provider_type) in DataSourceBuilder._providers
        except ValueError:
            return False
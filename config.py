from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError, Field, SecretStr
from functools import lru_cache
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings with enhanced validation and security"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Keys with enhanced security
    pinecone_api_key: SecretStr
    pinecone_environment: str = Field(..., min_length=2)
    openai_api_key: SecretStr
    openai_model: str = Field(default="gpt-3.5-turbo")
    
    # Database configuration
    db_url: SecretStr
    
    # AWS credentials
    aws_access_key: SecretStr
    aws_secret_key: SecretStr
    
    # Application settings
    log_level: str = Field(default="INFO")
    max_tokens: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=20, ge=0)
    max_retries: int = Field(default=3, ge=0)
    timeout: int = Field(default=30, gt=0)

    # Environment validation
    environment: str = Field(
        default="development",
        pattern="^(development|staging|production)$"
    )

    def get_masked_credentials(self):
        """Returns masked version of credentials for logging"""
        return {
            "pinecone_api_key": self.pinecone_api_key.get_secret_value()[:4] + "****",
            "openai_api_key": self.openai_api_key.get_secret_value()[:4] + "****",
            "aws_access_key": self.aws_access_key.get_secret_value()[:4] + "****",
            "environment": self.environment
        }

@lru_cache()
def get_settings() -> Settings:
    """
    Creates and validates settings with caching
    
    Returns:
        Settings: Application settings
        
    Raises:
        SystemExit: If settings validation fails
    """
    try:
        settings = Settings()
        # Log masked credentials
        logger.info(f"Settings loaded: {settings.get_masked_credentials()}")
        return settings
    except ValidationError as e:
        logger.error(f"Settings validation failed: {str(e)}")
        raise SystemExit(1)

settings = get_settings()
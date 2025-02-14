from openai import OpenAI, OpenAIError
from config import settings
from loguru import logger
from typing import List, Dict, Any
import backoff
import time

class OpenAIClientError(Exception):
    """Custom exception for OpenAI client errors"""
    pass

class OpenAiClient:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self.model = settings.openai_model
        self.max_retries = settings.max_retries
        self.timeout = settings.timeout

    @backoff.on_exception(
        backoff.expo,
        OpenAIError,
        max_tries=3
    )
    async def generate_embeddings(self, text_list: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with retry logic
        
        Args:
            text_list (List[str]): List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            OpenAIClientError: If embedding generation fails
        """
        try:
            logger.info(f"Generating embeddings for {len(text_list)} texts")
            start_time = time.time()

            response = await self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text_list,
                timeout=self.timeout
            )
            
            embeddings = [item.embedding for item in response.data]
            
            duration = time.time() - start_time
            logger.success(
                f"Generated {len(embeddings)} embeddings in {duration:.2f}s"
            )
            
            return embeddings

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise OpenAIClientError(f"Failed to generate embeddings: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise OpenAIClientError(f"Unexpected error: {str(e)}")
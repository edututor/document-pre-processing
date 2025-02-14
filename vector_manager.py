from loguru import logger
from typing import List, Dict, Any
import numpy as np
from openai_client import OpenAiClient
import asyncio

class VectorManagerError(Exception):
    """Custom exception for vector manager errors"""
    pass

class VectorManager:
    def __init__(self, batch_size: int = 100):
        """
        Initialize VectorManager
        
        Args:
            batch_size (int): Size of batches for processing
        """
        self.batch_size = batch_size

    def validate_corpus(self, corpus: Union[str, List[str]]) -> List[str]:
        """Validate and prepare corpus for vectorization"""
        if isinstance(corpus, str):
            corpus = [corpus]
        if not all(isinstance(text, str) for text in corpus):
            raise ValueError("All items in corpus must be strings")
        return corpus

    async def vectorize(self, client: OpenAiClient, corpus: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for texts in batches
        
        Args:
            client (OpenAiClient): OpenAI client instance
            corpus (Union[str, List[str]]): Text(s) to vectorize
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            VectorManagerError: If vectorization fails
        """
        try:
            logger.info("Starting vectorization process")
            corpus = self.validate_corpus(corpus)
            
            # Process in batches
            embeddings = []
            for i in range(0, len(corpus), self.batch_size):
                batch = corpus[i:i + self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1}")
                
                batch_embeddings = await client.generate_embeddings(batch)
                embeddings.extend(batch_embeddings)
                
            logger.success(f"Vectorization completed: {len(embeddings)} vectors generated")
            return embeddings
            
        except Exception as e:
            logger.error(f"Vectorization failed: {str(e)}")
            raise VectorManagerError(f"Failed to generate vectors: {str(e)}")

    def normalize_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        """Normalize vectors to unit length"""
        try:
            vectors_array = np.array(vectors)
            norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
            normalized = vectors_array / norms
            return normalized.tolist()
        except Exception as e:
            raise VectorManagerError(f"Vector normalization failed: {str(e)}")
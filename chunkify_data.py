import tiktoken
from loguru import logger
from typing import List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

@dataclass
class ChunkResult:
    """Data class to hold chunking results"""
    chunks: List[str]
    total_tokens: int
    chunk_count: int

class ChunkifyError(Exception):
    """Custom exception for chunking errors"""
    pass

def validate_text(text: str) -> bool:
    """Validates input text"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    return True

def chunkify(text: str, max_tokens: int = 500, overlap: int = 20) -> ChunkResult:
    """
    Splits a long text into chunks that fit within the token limit for OpenAI models.
    
    Args:
        text (str): The input text to chunk
        max_tokens (int): Maximum number of tokens per chunk
        overlap (int): Number of overlapping tokens between chunks
    
    Returns:
        ChunkResult: Object containing chunks and metadata
    
    Raises:
        ChunkifyError: If chunking process fails
        ValueError: If input parameters are invalid
    """
    try:
        logger.info(f"Starting text chunking process: {len(text)} characters")
        validate_text(text)
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= max_tokens:
            raise ValueError("overlap must be less than max_tokens")

        # Initialize tokenizer
        logger.debug("Initializing tokenizer")
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize text
        tokens = tokenizer.encode(text)
        total_tokens = len(tokens)
        logger.info(f"Text tokenized: {total_tokens} tokens")

        chunks = []
        start = 0
        
        # Process chunks in parallel for large texts
        def process_chunk(start_idx):
            end_idx = min(start_idx + max_tokens, len(tokens))
            chunk = tokens[start_idx:end_idx]
            return tokenizer.decode(chunk)

        with ThreadPoolExecutor() as executor:
            futures = []
            while start < len(tokens):
                futures.append(executor.submit(process_chunk, start))
                start = start + max_tokens - overlap

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                chunks.append(future.result())

        logger.success(f"Chunking completed: {len(chunks)} chunks created")
        return ChunkResult(
            chunks=chunks,
            total_tokens=total_tokens,
            chunk_count=len(chunks)
        )

    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        raise ChunkifyError(f"Failed to process text: {str(e)}")
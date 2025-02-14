import tiktoken
from loguru import logger


# Chunkify the document
def chunkify(text, max_tokens=500, overlap=20):
    logger.info("Chunkifier called")
    """
    Splits a long text into chunks that fit within the token limit for OpenAI models.
    Args:
        text (str): The input text to chunk.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        List[str]: A list of text chunks, each within the token limit.
    """
    logger.info("Initialize tokenizer")
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Initialize tokenizer
    tokens = tokenizer.encode(text)  # Tokenize the input text

    chunks = []
    start = 0
    end = 0

    while end < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))  # Decode tokens back to text
        start = end - overlap  # Step back for overlap
        
    return chunks
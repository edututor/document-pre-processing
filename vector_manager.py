from loguru import logger

class VectorManager:

    def __init__(self, embeddings=None):
        """
        Initialize with the precomputed embeddings dictionary.
        """
        self.embeddings = embeddings
        

    def vectorize(self, client, corpus):

        logger.info("Querying to get embeddings")

        # Generate embeddings
        try:
            embeddings_response = client.generate_embeddings(corpus)

            # Extract only embeddings from the response
            embeddings = [item.embedding for item in embeddings_response.data]
            logger.success("Embeddings received and processed")
            return embeddings
        
        except Exception as e:
            logger.error(f"An error occured while qurying OpenAI's Embedding Generator")

        
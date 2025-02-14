# main.py
import boto3
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pdf_manager import PdfManager, PDFManagerError
from openai_client import OpenAiClient, OpenAIClientError
from vector_manager import VectorManager, VectorManagerError
from schemas import PreprocessRequest
from chunkify_data import chunkify, ChunkifyError
from config import settings
from loguru import logger
from datetime import datetime
import asyncio
from typing import Dict, Any
from pinecone import Pinecone, PineconeException
from prometheus_client import Counter, Histogram
import uvicorn
import time

# Initialize metrics
PROCESS_TIME = Histogram('document_process_duration_seconds', 'Time spent processing document')
ERROR_COUNTER = Counter('document_process_errors_total', 'Total processing errors')
SUCCESS_COUNTER = Counter('document_process_success_total', 'Total successful processes')

# Initialize FastAPI app
app = FastAPI(
    title="Document Pre-processing Service",
    description="Service for processing and vectorizing documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with React frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key != settings.api_key.get_secret_value():
        raise HTTPException(status_code=403, detail="Invalid API key")
    return

# Initialize the S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.aws_access_key,
    aws_secret_access_key=settings.aws_secret_key,
)

# Access the Pinecone indexes
pc = Pinecone(api_key=settings.pinecone_api_key)
text_index = pc.Index("document-text-embeddings")
table_index = pc.Index("table-embeddings")

@app.post("/api/preprocess")
async def preprocess_file(request: PreprocessRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()  # Start tracking the processing time
    logger.info(f"Request received: {request.file_url}, {request.company_name}")
    
    file_url = request.file_url
    company_name = request.company_name
    client = OpenAiClient()
    pdf_manager = PdfManager()
    vector_manager = VectorManager()

    try:
        # Parse S3 or HTTPS URL to extract bucket name and key
        s3_parts = await extract_s3_parts(file_url)

        # Fetch the file content from S3 asynchronously
        file_content = await fetch_s3_file(s3_parts)

        # Map MIME type to file type
        file_type = map_mime_to_file_type(file_content["ContentType"])

        # Extract text and chunkify
        extracted_file = pdf_manager.pdf_reader(file_content["Body"].read(), file_type)
        chunkified_text = chunkify(extracted_file.text)

        # Create embeddings asynchronously
        text_embeddings = await vector_manager.vectorize(client, chunkified_text)

        # Get today's date for metadata
        today_date = datetime.now().strftime("%Y/%m/%d")

        # Prepare vectors for Pinecone index
        text_vectors = prepare_text_vectors(chunkified_text, text_embeddings, today_date, company_name, file_url)

        # Write text embeddings to Pinecone
        await upsert_text_embeddings(text_vectors)

        # Record success in metrics
        SUCCESS_COUNTER.inc()

        # Measure and log the processing time
        process_duration = time.time() - start_time
        PROCESS_TIME.observe(process_duration)

        return JSONResponse(status_code=200, content={"message": "File processed successfully"})
    
    except (PDFManagerError, OpenAIClientError, VectorManagerError, ChunkifyError, PineconeException) as e:
        ERROR_COUNTER.inc()
        logger.error(f"Error occurred: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
    
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Unexpected error occurred: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "An unexpected error occurred."})

async def extract_s3_parts(file_url: str):
    """Helper to extract S3 bucket and key parts."""
    if file_url.startswith("s3://"):
        return file_url[5:].split("/", 1)
    elif file_url.startswith("https://"):
        domain = "s3.amazonaws.com/"
        if domain in file_url:
            s3_parts = file_url.replace(f"https://", "").split("/", 1)
            s3_parts[0] = s3_parts[0].split(".s3.amazonaws.com")[0]
            return s3_parts
    raise ValueError("Invalid URL. Must start with 's3://' or 'https://'.")

async def fetch_s3_file(s3_parts: list):
    """Helper to fetch file from S3."""
    bucket_name, object_key = s3_parts
    response = await asyncio.to_thread(s3_client.get_object, Bucket=bucket_name, Key=object_key)
    if response.get("ContentType") != "application/pdf":
        raise ValueError(f"Unsupported file type: {response.get('ContentType')}")
    return response

def map_mime_to_file_type(mime_type: str) -> str:
    """Helper to map MIME type to file type."""
    mime_to_type = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    }
    return mime_to_type.get(mime_type, None)

async def upsert_text_embeddings(text_vectors: list):
    """Helper to upsert text embeddings to Pinecone asynchronously."""
    await asyncio.to_thread(text_index.upsert, vectors=text_vectors)

def prepare_text_vectors(chunkified_text, text_embeddings, today_date, company_name, file_url):
    """Helper to prepare text vectors for Pinecone."""
    return [
        {
            "id": f"text-{i}-{file_url}",
            "values": embedding,
            "metadata": {
                "document_name": file_url,
                "company_name": company_name,
                "chunk": chunk,
                "upload_date": today_date,
            }
        }
        for i, (chunk, embedding) in enumerate(zip(chunkified_text, text_embeddings))
    ]

@app.get("/api/preprocess/health")
async def health_check():
    try:
        # Test Pinecone connection
        text_index.describe_index_stats()
        # Test S3 connection
        s3_client.list_buckets()
        return JSONResponse(status_code=200, content={"status": "healthy"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


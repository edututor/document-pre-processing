import boto3
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf_manager import PdfManager
from openai_client import OpenAiClient
from vector_manager import VectorManager
from schemas import PreprocessRequest
from chunkify_data import chunkify
from config import settings
from loguru import logger
from datetime import datetime
from pinecone import Pinecone

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production (e.g., ["https://your-frontend.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def preprocess_file(request: PreprocessRequest):
    logger.info(f"Request received: {request.file_url}, {request.company_name}")
    file_url = request.file_url
    company_name = request.company_name
    client = OpenAiClient()
    pdf_manager = PdfManager()
    vector_manager = VectorManager()

    logger.info(f"Creating s3_parts")
    try:
        # Parse S3 or HTTPS URL to extract bucket name and key
        if file_url.startswith("s3://"):
            s3_parts = file_url[5:].split("/", 1)
        elif file_url.startswith("https://"):
            domain = "s3.amazonaws.com/"
            if domain in file_url:
                s3_parts = file_url.replace(f"https://", "").split("/", 1)
                s3_parts[0] = s3_parts[0].split(".s3.amazonaws.com")[0]  # Extract bucket name
        else:
            raise ValueError("Invalid URL. Must start with 's3://' or 'https://'.")
        
        logger.info(f"Creating bucket_name and object_key")
        bucket_name, object_key = s3_parts
        logger.success(f"bucket_name: {bucket_name}, object_key: {object_key}")
        # Fetch the file content from S3
        logger.info("Getting the object")
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        logger.success("Object received!")
        # Validate the content type
        content_type = response.get("ContentType", "")
        logger.info(f"Content Type: {content_type}")
        if content_type != "application/pdf":
            raise ValueError(f"Unsupported content type: {content_type}")

        # Read the file content
        logger.info("Reading file content")
        file_content = response["Body"].read()  # Reads the full content into memory
        logger.success(f"File content fetched successfully, size: {len(file_content)} bytes")

        # Map MIME type to file type
        mime_to_type = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        }

        file_type = mime_to_type.get(content_type)
        logger.info(f"File type: {file_type}")
        if not file_type:
            raise ValueError(f"Unsupported file type: {content_type}")
     
        # Extract text
        extracted_file = pdf_manager.pdf_reader(file_content, file_type)

        # Chunkify the text
        chunkified_text = chunkify(extracted_file.text)

        # Create embeddings
        text_embeddings = vector_manager.vectorize(client, chunkified_text)

        # Get today's date
        today_date = datetime.now().strftime("%Y/%m/%d")

        # Write text embeddings to the text index
        text_vectors = [
            {
                "id": f"text-{i}-{object_key}",
                "values": embedding,
                "metadata": {
                    "document_name": object_key,
                    "company_name": company_name,
                    "chunk": chunk,
                    "upload_date": today_date,
                }
            }
            for i, (chunk, embedding) in enumerate(zip(chunkified_text, text_embeddings))
        ]
        try:
            text_index.upsert(vectors=text_vectors)
            logger.success(f"{len(text_vectors)} text embeddings written to the text index.")
        except Exception as e:
            logger.error(f"Failed to upsert text embeddings: {str(e)}")

        # Return success response with preprocessing result
        return JSONResponse(
            status_code=200,
            content={"message": "File processed successfully"}
        )
    except s3_client.exceptions.NoSuchKey:
        return JSONResponse(
            status_code=404,
            content={"message": "File not found in S3 bucket."}
        )
    except s3_client.exceptions.ClientError as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to fetch file from S3: {str(e)}"}
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"message": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )
    

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
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


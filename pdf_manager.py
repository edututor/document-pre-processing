from loguru import logger
from docx import Document
from io import BytesIO
import pdfplumber

class PdfManager:
    def extract(self, client, corpus, schema, agent):

        # Extract data from pdf
        logger.info("Extracting data")
        try:
            logger.info("Prompting ChatGPT")
            prompt = agent.prompt(corpus)
            extracted_data = client.query_gpt(prompt, schema)
            logger.success("Response received from ChatGPT")

            return extracted_data
                
        except Exception as e:
            logger.error(f"An error occured while querying ChatGPT: {e}")
    
    # Extract text using pdfplumber
    def pdf_reader(self, file_content, file_type):
        """
        Extracts text from PDF or Word files.

        Args:
            file_content (bytes): Binary content of the file.
            file_type (str): Type of the file ('pdf' or 'docx').

        Returns:
            str: Extracted text.
        """
        if file_type == "pdf":
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        elif file_type == "docx":
            document = Document(BytesIO(file_content))
            text = ""
            for paragraph in document.paragraphs:
                text += paragraph.text + "\n"
            return text
        else:
            raise ValueError("Unsupported file type. Supported types: 'pdf', 'docx'")
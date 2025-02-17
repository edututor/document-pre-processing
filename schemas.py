from pydantic import BaseModel

class PreprocessRequest(BaseModel):
    file_url: str
    document_name: str

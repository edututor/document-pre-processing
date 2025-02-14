from pydantic import BaseModel

class PreprocessRequest(BaseModel):
    file_url: str
    company_name: str

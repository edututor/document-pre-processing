from pydantic import BaseModel
from typing import List, Dict, Optional

# Analysis generation
class ChunkSummaryResponseSchema(BaseModel):
    summary: str

class TableSchema(BaseModel):
    table_name: str
    columns: List[str]
    rows: List[List[str]]

class TablesSchema(BaseModel):
    tables: List[TableSchema]

class ExtractedTextSchema(BaseModel):
    text: str

class PreProcessingResponseSchema(BaseModel):
    text: str
    tables: List[TableSchema]

class AnalysisResponseSchema(BaseModel):
    analysis: str

# Report generation
class TableData(BaseModel):
    columns: List[str]
    rows: List[List[str]]

class GraphMetadata(BaseModel):
    x_axis: List[str]
    y_data: Optional[Dict[str, List[float]]] = None
    type: str  # e.g., 'line', 'bar', etc.

class ReportSection(BaseModel):
    heading: str
    content: str
    table_caption: Optional[str] = None
    table_data: Optional[TableData] = None
    figure_caption: Optional[str] = None
    graph_metadata: Optional[GraphMetadata] = None

class ReportResponse(BaseModel):
    report_body: List[ReportSection]

class PreprocessRequest(BaseModel):
    file_url: str
    company_name: str

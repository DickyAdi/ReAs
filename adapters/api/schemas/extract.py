from pydantic import BaseModel


# * unused
class ExtractFromCsvRequest(BaseModel):
    text_column: str

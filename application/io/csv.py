from typing import Any

from domain.io.interfaces.csv import CsvInterface


class CsvApplication:
    def __init__(self, service: CsvInterface):
        self.service = service

    def to_dataframe(self, content: bytes) -> Any:
        decoded = self.service.decode(content=content)
        sanitized = self.service.sanitize(content=decoded)
        df = self.service.to_dataframe(content=sanitized)
        return df

    def get_text(self, content: bytes, text_column: str) -> list[str]:
        df = self.to_dataframe(content=content)
        return self.service.get_text(df=df, text_column=text_column)

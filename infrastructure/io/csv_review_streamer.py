from typing import Optional, BinaryIO

from domain.exceptions import ColumnNotFoundError
from .csv_streamer import CsvStreamer


class ReviewCsvStreamer(CsvStreamer):
    def __init__(
        self,
        file: BinaryIO,
        text_column: str,
        rating_column: Optional[str] = None,
        batch_size: Optional[int] = 256,
    ):
        super().__init__(file=file, batch_size=batch_size)
        self.text_column = text_column
        self.rating_column = rating_column

    def __enter__(self):
        super().__enter__()
        if self.text_column not in self.reader.fieldnames:
            raise ColumnNotFoundError(
                text_column=self.text_column,
                available_column=[col for col in self.reader.fieldnames],
            )
        if self.rating_column and self.rating_column not in self.reader.fieldnames:
            raise ColumnNotFoundError(
                text_column=self.rating_column,
                available_column=[col for col in self.reader.fieldnames],
            )
        return self

    def next_batch(self) -> list[dict]:
        batch = super().next_batch()
        data = [
            {
                "text": row.get(self.text_column),
                **(
                    {"rating": row.get(self.rating_column)}
                    if self.rating_column
                    else {}
                ),
            }
            for row in batch
        ]
        return data

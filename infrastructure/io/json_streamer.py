import os
import ijson
from typing import Optional, Self
from datetime import datetime, timezone
import gc

from domain.io.interfaces import StreamerInterface
from domain.exceptions import ColumnNotFoundError


class JsonStreamer(StreamerInterface):
    def __init__(
        self,
        raw_json,
        text_column: str = "review_text",
        rating_column: str = "review_rating",
        batch_size: Optional[int] = 512,
    ):
        super().__init__(batch_size=batch_size)
        if isinstance(raw_json, os.PathLike):
            self.data = open(raw_json, "rb")
            self._should_close = True
        elif hasattr(raw_json, "read"):
            self.data = raw_json
            self._should_close = False
        else:
            raise TypeError("Invalid type, must be path like or openable json object")
        # self.data = open(raw_json, "rb")
        self.text_column = text_column
        self.rating_column = rating_column

    def __enter__(self, prefix: str) -> Self:
        sniffer = ijson.items(self.data, prefix)
        if not next(sniffer):
            raise ValueError(f"Invalid prefix, {prefix} got no results.")
        del sniffer
        gc.collect()  # free up memory
        self.data = ijson.items(self.data, prefix)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data and self._should_close:
            self.data.close()
            gc.collect()

    def next_batch(self):
        batch = super().next_batch()
        extracted = []
        for b in batch:
            if [self.text_column, self.rating_column] not in list(b.keys()):
                raise ColumnNotFoundError(
                    text_column=f"{self.text_column}, {self.rating_column}",
                    available_column=list(b.keys()),
                )
            extracted.append(
                {
                    "text": b.get(self.text_column),
                    "rating": b.get(self.rating_column),
                    "post_date": b.get(
                        "review_datetime_utc", datetime.now(timezone.utc)
                    ),
                }
            )
        return extracted

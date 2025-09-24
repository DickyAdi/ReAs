from csv import DictReader, Sniffer
from typing import Optional
from io import TextIOWrapper

from domain.io.interfaces import StreamerInterface


class CsvStreamer(StreamerInterface):
    def __init__(self, file, batch_size: Optional[int] = 512):
        self.file = file
        super().__init__(batch_size=batch_size)

    def __enter__(self):
        sample = self.file.read(1024).decode("utf-8", errors="ignore")
        self.file.seek(0)
        csv_dialect = Sniffer().sniff(sample)
        self.decoded_file = TextIOWrapper(self.file, encoding="utf-8")
        self.reader = DictReader(self.decoded_file, dialect=csv_dialect)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        if self.decoded_file:
            self.decoded_file.close()

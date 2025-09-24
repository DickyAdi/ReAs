from typing import Optional, Iterator, TypeVar
from itertools import islice

from abc import ABC, abstractmethod

T = TypeVar("T")


class StreamerInterface(ABC):
    def __init__(self, batch_size: Optional[int] = 512):
        self.batch_size = batch_size
        self.reader = None

    @abstractmethod
    def __enter__(self) -> "StreamerInterface": ...
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    def next_batch(self) -> list[dict[str, T]]:
        batch = list(islice(self.reader, self.batch_size))
        return batch

    def __iter__(self) -> Iterator[list]:
        while True:
            batch = self.next_batch()
            if not batch:
                break
            yield batch

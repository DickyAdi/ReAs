from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from domain.entities.datasets import DatasetEntity


@dataclass
class InsightEntity:
    topic: str
    score: float
    id: Optional[UUID] = None  # derived from database
    created_at: Optional[datetime] = None  # derived from database
    updated_at: Optional[datetime] = None  # derived from database
    dataset_id: Optional[UUID] = None  # derived from database

    dataset: Optional["DatasetEntity"] = None

    @classmethod
    def create(cls, topic: str, score: float) -> "InsightEntity":
        return cls(topic=topic, score=score)

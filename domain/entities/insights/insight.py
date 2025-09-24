from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from domain.entities.datasets import DatasetEntity


@dataclass
class InsightEntity:
    topic: str
    emerging_score: float
    trend_score: float
    dataset_id: UUID
    id: Optional[UUID] = None  # derived from database
    created_at: Optional[datetime] = None  # derived from database
    updated_at: Optional[datetime] = None  # derived from database

    dataset: Optional["DatasetEntity"] = None

    @classmethod
    def create(
        cls, topic: str, emerging_score: float, trend_score: float, dataset_id: UUID
    ) -> "InsightEntity":
        return cls(
            topic=topic,
            emerging_score=emerging_score,
            trend_score=trend_score,
            dataset_id=dataset_id,
        )

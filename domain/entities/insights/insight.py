from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from domain.entities.datasets import DatasetEntity


@dataclass
class InsightEntity:
    topic: str
    relevancy_score: float
    consistency_score: float
    dataset_id: UUID
    id: Optional[UUID] = None  # derived from database
    created_at: Optional[datetime] = None  # derived from database
    updated_at: Optional[datetime] = None  # derived from database

    dataset: Optional["DatasetEntity"] = None

    @classmethod
    def create(
        cls,
        topic: str,
        relevancy_score: float,
        consistency_score: float,
        dataset_id: UUID,
    ) -> "InsightEntity":
        return cls(
            topic=topic,
            relevancy_score=relevancy_score,
            consistency_score=consistency_score,
            dataset_id=dataset_id,
        )

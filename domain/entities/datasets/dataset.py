from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass

if TYPE_CHECKING:
    from domain.entities.user import UserEntities
    from domain.entities.reviews import ReviewEntity

# from domain.enums.datasets import DatasetProvider
from domain.enums.datasets import DatasetStatus


@dataclass
class DatasetEntity:
    name: str
    is_empty: bool = True
    status: Optional[DatasetStatus] = DatasetStatus.pending
    total_reviews: int = 0  # derived from scrape post-processing
    id: Optional[UUID] = None  # derived from database
    created_at: Optional[datetime] = None  # derived from database
    updated_at: Optional[datetime] = None  # derived from database
    issuer_id: Optional[int] = None  # derived from database

    # related entity
    user: Optional["UserEntities"] = None
    reviews: Optional[List["ReviewEntity"]] = None

    @classmethod
    def create(
        cls, name: str, status: Optional[DatasetStatus] = DatasetStatus.pending
    ) -> "DatasetEntity":
        return cls(name=name, status=status)

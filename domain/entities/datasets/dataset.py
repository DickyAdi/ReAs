from typing import Optional, List, TYPE_CHECKING, TypedDict
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass

if TYPE_CHECKING:
    from domain.entities.user import UserEntities
    from domain.entities.reviews import ReviewEntity

# from domain.enums.datasets import DatasetProvider
from domain.enums.datasets import DatasetStatus


class DatasetTypedDict(TypedDict):
    name: str
    issuer_id: int
    is_empty: bool
    status: DatasetStatus
    total_reviews: int
    id: UUID
    created_at: datetime
    updated_at: datetime


@dataclass
class DatasetEntity:
    name: str
    issuer_id: int
    is_empty: bool = (
        True  # always True as creating a dataset always starts with empty dataset
    )
    status: Optional[DatasetStatus] = DatasetStatus.pending
    total_reviews: int = 0  # derived from scrape post-processing
    id: Optional[UUID] = None  # derived from database
    created_at: Optional[datetime] = None  # derived from database
    updated_at: Optional[datetime] = None  # derived from database

    # related entity
    user: Optional["UserEntities"] = None
    reviews: Optional[List["ReviewEntity"]] = None

    @classmethod
    def create(
        cls,
        name: str,
        issuer_id: int,
        status: Optional[DatasetStatus] = DatasetStatus.pending,
    ) -> "DatasetEntity":
        return cls(name=name, status=status, issuer_id=issuer_id)

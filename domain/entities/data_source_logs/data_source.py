from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from uuid import UUID
from datetime import datetime

if TYPE_CHECKING:
    from domain.entities.reviews import ReviewEntity

from domain.enums.datasets import DatasetProvider


@dataclass
class DataSourceEntity:
    provider: DatasetProvider
    additional_metadata: Optional[dict] = None  # literally optional
    created_at: Optional[datetime] = None  # derived from database
    id: Optional[UUID] = None  # derived from database

    # related entity
    reviews: Optional[list["ReviewEntity"]] = None

    @classmethod
    def create(
        cls, provider: DatasetProvider, additional_metadata: Optional[dict] = None
    ) -> "DataSourceEntity":
        return cls(provider=provider, additional_metadata=additional_metadata)

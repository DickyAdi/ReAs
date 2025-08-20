from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass

if TYPE_CHECKING:
    from domain.entities.user import UserEntities
    from domain.entities.reviews import ReviewEntity

from domain.enums.datasets import DatasetStatus, DatasetProvider


@dataclass
class DatasetEntity:
    # status: DatasetStatus
    # provider: DatasetProvider
    # total_scrape: Optional[int] = 0  # derived from scrape post-processing
    # id: Optional[UUID] = None  # derived from database
    # created_at: Optional[datetime] = None  # derived from database
    # started_at: Optional[datetime] = None  # derived from database
    # finished_at: Optional[datetime] = None  # derived from database
    # issuer_id: Optional[int] = None  # derived from database

    status: DatasetStatus
    provider: DatasetProvider
    provider_ref: Optional[str] = None
    total_review: Optional[int] = 0  # derived from scrape post-processing
    id: Optional[UUID] = None  # derived from database
    created_at: Optional[datetime] = None  # derived from database
    updated_at: Optional[datetime] = None  # derived from database
    issuer_id: Optional[int] = None  # derived from database

    # related entity
    issuer: Optional["UserEntities"] = None
    reviews: Optional[List["ReviewEntity"]] = None

    # @classmethod
    # def create(
    #     cls,
    #     status: DatasetStatus,
    #     provider: DatasetProvider,
    #     issuer: Optional["UserEntities"],
    #     scraped_texts: Optional[List["ReviewEntity"]],
    # ) -> "DatasetEntity":
    #     return cls(
    #         status=status, provider=provider, issuer=issuer, scraped_texts=scraped_texts
    #     )
    @classmethod
    def create(
        cls,
        status: DatasetStatus,
        provider: DatasetProvider,
        provider_ref: Optional[str] = None,
        total_review: Optional[int] = None,
        issuer: Optional["UserEntities"] = None,
        reviews: Optional[List["ReviewEntity"]] = None,
    ) -> "DatasetEntity":
        return cls(
            status=status,
            provider=provider,
            issuer=issuer,
            reviews=reviews,
            provider_ref=provider_ref,
            total_review=total_review,
        )

    def insert_total_review(self, total: int):
        self.total_review = total

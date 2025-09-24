from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, TypedDict
from datetime import datetime
from uuid import UUID

if TYPE_CHECKING:
    from domain.entities.datasets import DatasetEntity
    from domain.entities.insights import InsightEntity
    from domain.entities.data_source_logs import DataSourceEntity


from domain.enums.texts import TextLanguage, TextPlatform, TextSentiment
# from domain.enums.datasets import DatasetProvider


class ReviewTypedDict(TypedDict):
    text: str
    rating: int
    post_date: datetime
    language: TextLanguage
    sentiment: TextSentiment
    provider_ref: UUID
    dataset_id: UUID
    id: UUID
    created_at: datetime


@dataclass
class ReviewEntity:
    text: str
    # place: str # * Removed due to unnecessary complexity
    post_date: datetime
    platform: TextPlatform
    # provider: DatasetProvider
    language: TextLanguage
    sentiment: TextSentiment
    provider_ref: UUID
    dataset_id: UUID
    rating: Optional[int]
    id: Optional[UUID] = None  # derived from database
    created_at: Optional[datetime] = None  # derived from database

    # related entity
    dataset: Optional["DatasetEntity"] = None
    insights: Optional["InsightEntity"] = None
    data_source: Optional["DataSourceEntity"] = None

    @classmethod
    def create(
        cls,
        text: str,
        # place: str, # * Removed due to unnecessary complexity
        post_date: datetime,
        platform: TextPlatform,
        language: TextLanguage,
        sentiment: TextSentiment,
        provider_ref: UUID,
        dataset_id: UUID,
        rating: Optional[int],
        # dataset: Optional["DatasetEntity"] = None,
        # insights: Optional["InsightEntity"] = None,
        # data_source: Optional["DataSourceEntity"] = None,
    ) -> "ReviewEntity":
        return cls(
            text=text,
            # place=place, # * Removed due to unnecessary complexity
            rating=rating,
            post_date=post_date,
            platform=platform,
            language=language,
            sentiment=sentiment,
            # dataset=dataset, # * dont really know if this needed or not, but just let it be
            provider_ref=provider_ref,
            dataset_id=dataset_id,
            # insights=insights, # * dont really know if this needed or not, but just let it be
            # data_source=data_source # * dont really know if this needed or not, but just let it be
        )

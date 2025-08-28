from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from datetime import datetime
from uuid import UUID

if TYPE_CHECKING:
    from domain.entities.datasets import DatasetEntity


from domain.enums.texts import TextLanguage, TextPlatform
# from domain.enums.datasets import DatasetProvider


@dataclass
class ReviewEntity:
    text: str
    place: str
    rating: int
    post_date: datetime
    platform: TextPlatform
    # provider: DatasetProvider
    language: TextLanguage
    provider_ref: Optional[UUID] = None  # derived from database
    id: Optional[UUID] = None  # derived from database
    dataset_id: Optional[UUID] = None  # derived from database

    # related entity
    dataset: Optional["DatasetEntity"] = None

    @classmethod
    def create(
        cls,
        text: str,
        place: str,
        rating: int,
        post_date: datetime,
        platform: TextPlatform,
        language: TextLanguage,
        dataset: Optional["DatasetEntity"] = None,
        provider_ref: Optional[UUID] = None,
    ) -> "ReviewEntity":
        return cls(
            text=text,
            place=place,
            rating=rating,
            post_date=post_date,
            platform=platform,
            language=language,
            dataset=dataset,
            provider_ref=provider_ref,
        )

    # @classmethod
    # def insert_sentiment(
    #     cls, texts_entity: list["ReviewEntity"], sentiments: list[TextSentiment]
    # ) -> list["ReviewEntity"]:
    #     if not isinstance(texts_entity, list) and not isinstance(sentiments, list):
    #         raise ValueError(
    #             f"Expected `texts_entity` and `sentiments` to be list. Got {type(texts_entity)} and {type(sentiments)} instead."
    #         )
    #     if len(texts_entity) != len(sentiments):
    #         raise KeyError(
    #             f"Length of list entities and list of sentiments doesnt match. list of entities has {len(texts_entity)} items while list of sentiment has {len(sentiments)}."
    #         )
    #     for entity, sentiment in zip(texts_entity, sentiments):
    #         setattr(entity, "sentiment", sentiment)
    #     return texts_entity

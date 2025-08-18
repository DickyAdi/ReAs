from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from datetime import datetime
from uuid import UUID

if TYPE_CHECKING:
    from domain.entities.usages import UsageEntity


from domain.enums.texts import TextLanguage, TextPlatform, TextSentiment

@dataclass
class ScrapedTextEntity:
    text:str
    place:str
    post_date:datetime
    platform: TextPlatform
    language: TextLanguage
    sentiment: Optional[TextSentiment] = None #set via post processing
    id: Optional[UUID] = None #derived from database
    scrape_id: Optional[UUID] = None #derived from database

    #related entity
    scrape_job: Optional["UsageEntity"] = None

    @classmethod
    def create(cls, text:str, place:str, post_date:datetime, platform:TextPlatform, language:TextLanguage, scrape_job:Optional['UsageEntity']) -> "ScrapedTextEntity":
        return cls(
            text=text,
            place=place,
            post_date=post_date,
            platform=platform,
            language=language,
            scrape_job=scrape_job
        )
    @classmethod
    def insert_sentiment(cls, texts_entity:list["ScrapedTextEntity"], sentiments:list[TextSentiment]) -> list["ScrapedTextEntity"]:
        if not isinstance(texts_entity, list) and not isinstance(sentiments, list):
            raise ValueError(f'Expected `texts_entity` and `sentiments` to be list. Got {type(texts_entity)} and {type(sentiments)} instead.')
        if len(texts_entity) != len(sentiments):
            raise KeyError(f'Length of list entities and list of sentiments doesnt match. list of entities has {len(texts_entity)} items while list of sentiment has {len(sentiments)}.')
        for entity, sentiment in zip(texts_entity, sentiments):
            setattr(entity,'sentiment', sentiment)
        return texts_entity
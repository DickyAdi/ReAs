from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass

if TYPE_CHECKING:
    from domain.entities.user import UserEntities
    from domain.entities.scraped import ScrapedTextEntity

from domain.enums.scrape_usages import ScrapeProvider, ScrapeStatus

@dataclass
class UsageEntity:
    status:ScrapeStatus
    provider:ScrapeProvider
    total_scrape:Optional[int] = 0 #derived from scrape post-processing
    id:Optional[UUID] = None #derived from database
    created_at:Optional[datetime] = None #derived from database
    started_at:Optional[datetime] = None #derived from database
    finished_at:Optional[datetime] = None #derived from database
    issuer_id:Optional[int] = None #derived from database

    #related entity
    issuer:Optional['UserEntities'] = None
    scraped_texts:Optional[List['ScrapedTextEntity']] = None

    @classmethod
    def create(cls, status:ScrapeStatus, provider:ScrapeProvider, issuer:Optional['UserEntities'], scraped_texts:Optional[List['ScrapedTextEntity']]) -> "UsageEntity":
        return cls(
            status=status,
            provider=provider,
            issuer=issuer,
            scraped_texts=scraped_texts
        )
    def insert_total_scrape(self, scraped:int):
        self.total_scrape = scraped
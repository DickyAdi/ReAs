from typing import TYPE_CHECKING
from sqlalchemy import DateTime, ForeignKey, Integer
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone

if TYPE_CHECKING:
    from .user_model import User
    from .scraped_text_model import ScrapedText


from ..db import Base

from domain.enums.scrape_usages import ScrapeProvider, ScrapeStatus

class ScrapeUsages(Base):
    __tablename__="scrape_usages"

    id:Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), default=uuid4, unique=True, nullable=False, primary_key=True, index=True)
    total_scrape:Mapped[int] = mapped_column(Integer, nullable=False)
    status:Mapped[ScrapeStatus] = mapped_column(ENUM(ScrapeStatus, name="scrape_usage_status_enum", create_type=True, check_first=True), nullable=False, default=ScrapeStatus.pending)
    provider:Mapped[ScrapeProvider] = mapped_column(ENUM(ScrapeProvider, name="scrape_usage_provider_enum", create_type=True, check_first=True), nullable=False)
    created_at:Mapped[DateTime] = mapped_column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    started_at:Mapped[DateTime] = mapped_column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    finished_at:Mapped[DateTime] = mapped_column(DateTime, nullable=True)
    issuer_id:Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)

    user:Mapped['User'] = relationship("User", back_populates='scrape_usages')
    scraped_texts:Mapped['ScrapedText'] = relationship("ScrapedText", back_populates="scrape_job")
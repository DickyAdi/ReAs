from sqlalchemy import DateTime, ForeignKey, Text, String
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scrape_usages_model import ScrapeUsages

from ..db import Base

from domain.enums.texts import TextLanguage, TextPlatform, TextSentiment

class ScrapedText(Base):
    __tablename__='scraped_text'

    id:Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, default=uuid4, unique=True, primary_key=True, index=True)
    text:Mapped[Text] = mapped_column(Text, nullable=True)
    place:Mapped[str] = mapped_column(String(100), nullable=False)
    post_date:Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    platform:Mapped[TextPlatform] = mapped_column(ENUM(TextPlatform, name="text_platform_enum", create_type=True, check_first=True), nullable=False)
    sentiment:Mapped[TextSentiment] = mapped_column(ENUM(TextSentiment, name="text_sentiment_enum", create_type=True, check_first=True), nullable=True)
    language:Mapped[TextLanguage] = mapped_column(ENUM(TextLanguage, name="text_language_enum", create_type=True, check_first=True), nullable=False)
    scrape_id:Mapped[UUID] = mapped_column(ForeignKey('scrape_usages.id'), nullable=False, unique=True, index=True)

    scrape_job:Mapped["ScrapeUsages"] = relationship("ScrapeUsages", back_populates="scraped_texts")
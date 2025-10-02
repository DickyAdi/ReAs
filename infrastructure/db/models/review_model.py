from sqlalchemy import DateTime, ForeignKey, Text
from sqlalchemy.inspection import inspect
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from typing import TYPE_CHECKING, Optional
from datetime import datetime, timezone

if TYPE_CHECKING:
    from .dataset_model import Datasets
    from .insight_model import Insights
    from .data_source_log_model import DataSource

from ..db import Base

from domain.entities.reviews import ReviewEntity, ReviewTypedDict
from domain.enums.texts import TextLanguage, TextPlatform, TextSentiment


class Reviews(Base):
    __tablename__ = "reviews"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        default=lambda: uuid4(),
        unique=True,
        primary_key=True,
        index=True,
    )
    text: Mapped[Text] = mapped_column(Text, nullable=True)
    # place: Mapped[str] = mapped_column(String(100), nullable=False) # * Removed due to unnecessary field
    rating: Mapped[int] = mapped_column(nullable=True)
    post_date: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    platform: Mapped[TextPlatform] = mapped_column(
        ENUM(
            TextPlatform, name="text_platform_enum", create_type=True, check_first=True
        ),
        nullable=False,
    )
    language: Mapped[TextLanguage] = mapped_column(
        ENUM(
            TextLanguage, name="text_language_enum", create_type=True, check_first=True
        ),
        nullable=False,
    )

    sentiment: Mapped[TextSentiment] = mapped_column(
        ENUM(
            TextSentiment,
            name="text_predicted_sentiment_enum",
            create_type=True,
            check_first=True,
        ),
        nullable=False,  # * will normalize this at later version of reas
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id"), nullable=False, index=True
    )
    provider_ref: Mapped[UUID] = mapped_column(
        ForeignKey("data_source_log.id", ondelete="CASCADE"), nullable=False
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # * related entity

    dataset: Mapped["Datasets"] = relationship("Datasets", back_populates="reviews")
    insights: Mapped["Insights"] = relationship(
        "Insights",
        secondary="insight_reviews",
        back_populates="reviews",
        passive_deletes=True,
    )
    data_source: Mapped["DataSource"] = relationship(
        "DataSource", back_populates="reviews", passive_deletes=True
    )

    @classmethod
    def from_entity(
        cls, review: "ReviewEntity", generate_defaults: Optional[bool] = False
    ) -> "Reviews":
        if generate_defaults:
            return cls(
                id=uuid4(),
                text=review.text,
                rating=review.rating,
                post_date=review.post_date,
                platform=review.platform,
                language=review.language,
                sentiment=review.sentiment,
                dataset_id=review.dataset_id,
                provider_ref=review.provider_ref,
                created_at=datetime.now(timezone.utc),
            )
        else:
            return cls(
                text=review.text,
                rating=review.rating,
                post_date=review.post_date,
                platform=review.platform,
                language=review.language,
                sentiment=review.sentiment,
                dataset_id=review.dataset_id,
                provider_ref=review.provider_ref,
            )

    def to_entity(self) -> "ReviewEntity":
        return ReviewEntity(
            id=self.id,
            rating=self.rating,
            post_date=self.post_date,
            platform=self.platform,
            language=self.language,
            sentiment=self.sentiment,
            provider_ref=self.provider_ref,
            dataset_id=self.dataset_id,
            created_at=self.created_at,
            # dataset=self.dataset, # * dont really know if this needed or not, but just let it be
            # insights=self.insights, # * dont really know if this needed or not, but just let it be
            # data_source=self.data_source # * dont really know if this needed or not, but just let it be
        )

    def to_dict(self) -> ReviewTypedDict:
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from typing import TYPE_CHECKING
from uuid import UUID

from ..db import Base

if TYPE_CHECKING:
    from .insight_model import Insights  # noqa
    from .review_model import Reviews  # noqa


class InsightReviews(Base):
    __tablename__ = "insight_reviews"

    insight_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("insights.id"),
        nullable=False,
        primary_key=True,
    )
    reviews_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("reviews.id"),
        nullable=False,
        primary_key=True,
    )
    # reviews: Mapped[list['Reviews']] = relationship('Reviews', back_populates=)

from sqlalchemy import DateTime, DOUBLE_PRECISION, ForeignKey
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset_model import Datasets
    from .review_model import Reviews
    from domain.entities.insights import InsightEntity

from ..db import Base


class Insights(Base):
    __tablename__ = "insights"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid4,
        unique=True,
        primary_key=True,
        nullable=False,
        index=True,
    )

    topic: Mapped[str] = mapped_column(nullable=False)
    score: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        nullable=False,
    )
    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("datasets.id"), nullable=False)

    dataset: Mapped["Datasets"] = relationship("Datasets", back_populates="insights")
    reviews: Mapped[list["Reviews"]] = relationship(
        "Reviews", secondary="insight_reviews", back_populates="insights"
    )

    def to_entity(self) -> "InsightEntity":
        return InsightEntity(
            created_at=self.created_at,
            updated_at=self.updated_at,
            dataset=self.dataset,
            dataset_id=self.dataset_id,
            id=self.id,
            score=self.score,
            topic=self.topic,
        )

    @classmethod
    def from_entity(cls, insight: "InsightEntity") -> "Insights":
        return cls(
            topic=insight.topic,
            score=insight.score,
        )

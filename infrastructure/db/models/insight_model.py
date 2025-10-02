from sqlalchemy import DateTime, DOUBLE_PRECISION, ForeignKey, UniqueConstraint
from sqlalchemy.inspection import inspect
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

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
    relevancy_score: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    consistency_score: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)

    created_at: Mapped[DateTime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        nullable=False,
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id"), nullable=False, index=True
    )

    dataset: Mapped["Datasets"] = relationship("Datasets", back_populates="insights")
    reviews: Mapped[list["Reviews"]] = relationship(
        "Reviews",
        secondary="insight_reviews",
        back_populates="insights",
        passive_deletes=True,
    )
    __table_args__ = (
        UniqueConstraint("dataset_id", "topic", name="uq_topic_per_dataset"),
    )

    def to_entity(self) -> "InsightEntity":
        return InsightEntity(
            created_at=self.created_at,
            updated_at=self.updated_at,
            dataset=self.dataset,
            dataset_id=self.dataset_id,
            id=self.id,
            relevancy_score=self.relevancy_score,
            consistency_score=self.consistency_score,
            topic=self.topic,
        )

    @classmethod
    def from_entity(
        cls, insight: "InsightEntity", generate_defaults: Optional[bool] = False
    ) -> "Insights":
        if generate_defaults:
            return cls(
                id=uuid4(),
                topic=insight.topic,
                relevancy_score=insight.relevancy_score,
                consistency_score=insight.consistency_score,
                dataset_id=insight.dataset_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        return cls(
            topic=insight.topic,
            relevancy_score=insight.relevancy_score,
            consistency_score=insight.consistency_score,
            dataset_id=insight.dataset_id,
        )

    def to_dict(self):
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

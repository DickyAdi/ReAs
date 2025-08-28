from typing import TYPE_CHECKING, Optional
from sqlalchemy import DateTime, ForeignKey, Integer, String
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone

if TYPE_CHECKING:
    from .user_model import User
    from .review_model import Reviews
    from domain.entities.datasets import DatasetEntity
    from .insight_model import Insights


from ..db import Base

from domain.enums.datasets import DatasetStatus


# class ScrapeUsages(Base):
class Datasets(Base):
    __tablename__ = "datasets"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid4,
        unique=True,
        nullable=False,
        primary_key=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    total_reviews: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_empty: Mapped[bool] = mapped_column(nullable=False, default=True)
    status: Mapped[DatasetStatus] = mapped_column(
        ENUM(
            DatasetStatus,
            name="dataset_status_enum",
            create_type=True,
            check_first=True,
        ),
        nullable=False,
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime, nullable=False, default=datetime.now(timezone.utc)
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
    )
    issuer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)

    # * related entity

    user: Mapped["User"] = relationship("User", back_populates="datasets")
    reviews: Mapped[Optional[list["Reviews"]]] = relationship(
        "Reviews", back_populates="dataset"
    )
    insights: Mapped[list["Insights"]] = relationship(
        "Insights", back_populates="dataset"
    )

    def to_entity(self) -> "DatasetEntity":
        return DatasetEntity(
            is_empty=self.is_empty,
            status=self.status,
            total_reviews=self.total_reviews,
            user=self.user,
            reviews=self.reviews,
            updated_at=self.updated_at,
            created_at=self.created_at,
            issuer_id=self.issuer_id,
            id=self.id,
        )

    @classmethod
    def from_entity(cls, dataset: "DatasetEntity") -> "Datasets":
        return cls(
            name=dataset.name,
            is_empty=dataset.is_empty,
            total_reviews=dataset.total_reviews,
            status=dataset.status,
        )

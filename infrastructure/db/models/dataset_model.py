from typing import TYPE_CHECKING
from sqlalchemy import DateTime, ForeignKey, Integer
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone

if TYPE_CHECKING:
    from .user_model import User
    from .review_model import Reviews


from ..db import Base

from domain.enums.datasets import DatasetStatus, DatasetProvider


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
    total_reviews: Mapped[int] = mapped_column(Integer, nullable=True)
    status: Mapped[DatasetStatus] = mapped_column(
        ENUM(
            DatasetStatus,
            name="dataset_status_enum",
            create_type=True,
            check_first=True,
        ),
        nullable=False,
        default=DatasetStatus.pending,
    )
    provider: Mapped[DatasetProvider] = mapped_column(
        ENUM(
            DatasetProvider,
            name="dataset_provider_enum",
            create_type=True,
            check_first=True,
        ),
        nullable=False,
    )
    provider_ref: Mapped[str] = mapped_column(nullable=False, unique=True)
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

    user: Mapped["User"] = relationship("User", back_populates="datasets")
    reviews: Mapped["Reviews"] = relationship("Reviews", back_populates="dataset")

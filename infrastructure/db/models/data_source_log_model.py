from typing import Optional, TYPE_CHECKING
from uuid import UUID, uuid4
from sqlalchemy import JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone


if TYPE_CHECKING:
    from domain.entities.data_source_logs import DataSourceEntity
    from .review_model import Reviews

from domain.enums.datasets import DatasetProvider
from ..db import Base


class DataSource(Base):
    __tablename__ = "data_source_log"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid4,
        nullable=False,
        primary_key=True,
        unique=True,
        index=True,
    )
    provider: Mapped[DatasetProvider] = mapped_column(
        ENUM(
            DatasetProvider,
            name="data_provider_enum",
            create_type=True,
            check_first=True,
        ),
        nullable=False,
    )
    additional_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime, nullable=False, default=datetime.now(timezone.utc)
    )

    # * related entity

    reviews: Mapped[list["Reviews"]] = relationship(
        "Reviews", back_populates="data_source"
    )

    def to_entity(self) -> "DataSourceEntity":
        return DataSourceEntity(
            id=self.id,
            provider=self.provider,
            created_at=self.created_at,
            additional_metadata=self.additional_metadata,
        )

    @classmethod
    def from_entity(cls, data_source: "DataSourceEntity") -> "DataSource":
        return cls(
            provider=data_source.provider,
            additional_metadata=data_source.additional_metadata,
        )

from sqlalchemy import DateTime, ForeignKey, Text, String
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset_model import Datasets

from ..db import Base

from domain.enums.texts import TextLanguage, TextPlatform


class Reviews(Base):
    __tablename__ = "reviews"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        default=uuid4,
        unique=True,
        primary_key=True,
        index=True,
    )
    text: Mapped[Text] = mapped_column(Text, nullable=True)
    place: Mapped[str] = mapped_column(String(100), nullable=False)
    rating: Mapped[int] = mapped_column(nullable=False)
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
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id"), nullable=False, unique=True, index=True
    )

    dataset: Mapped["Datasets"] = relationship("Datasets", back_populates="reviews")

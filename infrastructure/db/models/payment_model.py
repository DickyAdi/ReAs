from sqlalchemy import DateTime, Numeric
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..db import Base

if TYPE_CHECKING:
    from .transaction_model import Transactions
    from domain.entities.payments import PaymentEntity


from domain.enums.payments import PaymentMethod, PaymentStatus


class Payments(Base):
    __tablename__ = "payments"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid4,
        unique=True,
        index=True,
        nullable=False,
        primary_key=True,
    )
    provider_ref: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True, index=True
    )
    method: Mapped[PaymentMethod] = mapped_column(
        ENUM(
            PaymentMethod,
            name="payment_method_enum",
            create_type=True,
            check_first=True,
        ),
        nullable=False,
    )
    status: Mapped[PaymentStatus] = mapped_column(
        ENUM(
            PaymentStatus,
            name="payment_status_enum",
            create_type=True,
            check_first=True,
        ),
        nullable=False,
    )
    amount_charged: Mapped[Numeric] = mapped_column(
        Numeric(precision=10, scale=2), nullable=False
    )
    created_at: Mapped[DateTime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        nullable=False,
    )
    transaction: Mapped["Transactions"] = relationship(
        "Transactions", back_populates="payment"
    )

    @classmethod
    def from_entity(cls, payment: "PaymentEntity") -> "Payments":
        return cls(
            provider_ref=payment.provider_ref,
            method=payment.method,
            status=payment.status,
            amount_charged=payment.amount_charged,
        )

    def to_entity(self) -> "PaymentEntity":
        return PaymentEntity(
            id=self.id,
            provider_ref=self.provider_ref,
            method=self.method,
            status=self.status,
            amount_charged=self.amount_charged,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

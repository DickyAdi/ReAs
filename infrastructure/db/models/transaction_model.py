from sqlalchemy import DateTime, Numeric, ForeignKey, JSON
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .payment_model import Payments
    from .subscriptions_model import Subscriptions
    from domain.entities.transactions import TransactionEntity


from ..db import Base

from domain.enums.transactions import TransactionStatus
from domain.entities.transactions import PlanSnapshotDict


class Transactions(Base):
    __tablename__ = "transactions"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        unique=True,
        index=True,
        primary_key=True,
        default=uuid4,
    )
    idempotency_key: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True, unique=True, index=True
    )  #!SET NULLABLE TO FALSE WHEN REAS ARE READY FOR LAUNCH. SETTING IT TO FALSE JUST TO MAKE IT EASIER TO DEBUG.
    status: Mapped[TransactionStatus] = mapped_column(
        ENUM(
            TransactionStatus,
            name="transaction_status_enum",
            create_type=True,
            check_first=True,
        ),
        nullable=False,
    )
    amount: Mapped[Numeric] = mapped_column(
        Numeric(precision=10, scale=2), nullable=False
    )
    plan_snapshot: Mapped[PlanSnapshotDict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime, default=datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        nullable=False,
    )
    payment_id: Mapped[UUID] = mapped_column(
        ForeignKey("payments.id"), nullable=False, unique=True
    )

    payment: Mapped["Payments"] = relationship("Payments", back_populates="transaction")
    subscription: Mapped["Subscriptions"] = relationship(
        "Subscriptions", back_populates="transaction"
    )

    @classmethod
    def from_entity(cls, transaction: "TransactionEntity") -> "Transactions":
        # snapshot = transaction.plan_snapshot
        # snapshot["price"] = snapshot["price"]
        transaction.plan_snapshot["price"] = str(transaction.plan_snapshot.get("price"))
        transaction.plan_snapshot["created_at"] = transaction.plan_snapshot.get(
            "created_at"
        ).isoformat()
        return cls(
            idempotency_key=transaction.idempotency_key,
            status=transaction.status,
            amount=transaction.amount,
            payment_id=transaction.payment_id,
            plan_snapshot=transaction.plan_snapshot,
        )

    def to_entity(self) -> "TransactionEntity":
        return TransactionEntity(
            id=self.id,
            idempotency_key=self.idempotency_key,
            status=self.status,
            amount=self.amount,
            plan_snapshot=self.plan_snapshot,
            created_at=self.created_at,
            updated_at=self.updated_at,
            payment_id=self.payment_id,
        )

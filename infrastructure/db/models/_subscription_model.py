from __future__ import annotations

from typing import TYPE_CHECKING
from sqlalchemy import DateTime, ForeignKey
from uuid import UUID, uuid4
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone

if TYPE_CHECKING:
    from domain.entities.subscriptions import SubscriptionEntity
from .user_model import User
from .transaction_model import Transactions
from .subscription_plans_model import SubscriptionPlan


from ..db import Base

from domain.enums.subscriptions import SubscriptionStatus

class Subscriptions(Base):
    __tablename__="subscriptions"

    id:Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), unique=True, nullable=False, primary_key=True, default=uuid4)
    start_at:Mapped[DateTime] = mapped_column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    end_at:Mapped[DateTime] = mapped_column(DateTime, nullable=True)
    status:Mapped[SubscriptionStatus] = mapped_column(ENUM(SubscriptionStatus, name='subscription_status_enum', create_type=True, check_first=True), nullable=False)
    created_at:Mapped[DateTime] = mapped_column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    user_id:Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)
    transactions_id:Mapped[UUID] = mapped_column(ForeignKey('transactions.id'), nullable=False, unique=True)
    plans_id:Mapped[int] = mapped_column(ForeignKey('subscription_plans.id'), nullable=False)

    user:Mapped['User'] = relationship("User", back_populates='subscriptions')
    transaction:Mapped['Transactions'] = relationship("Transactions", back_populates='subscription')
    plan:Mapped['SubscriptionPlan'] = relationship("SubscriptionPlan", back_populates='subscriptions')

    @classmethod
    def from_entity(cls, subscription:'SubscriptionEntity') -> "Subscriptions":
        return cls(
            start_at=subscription.start_at,
            end_at=subscription.end_at,
            status=subscription.status,
            user_id=subscription.user_id,
            transactions_id=subscription.transactions_id,
            plans_id=subscription.plans_id
        )
    
    def to_entity(self) -> 'SubscriptionEntity':
        return SubscriptionEntity(
            id=self.id,
            start_at=self.start_at,
            end_at=self.end_at,
            status=self.status,
            created_at=self.created_at,
            user_id=self.user_id,
            transactions_id=self.transactions_id,
            plans_id=self.plans_id
        )
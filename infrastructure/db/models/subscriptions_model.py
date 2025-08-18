from sqlalchemy import Integer, String, Numeric, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from uuid import UUID, uuid4
from typing import List
from datetime import datetime, timezone

from ..db import Base
from .user_model import User
from .transaction_model import Transactions

from domain.entities.tiers import TierEntity
from domain.entities.subscriptions import SubscriptionPlansEntity
from domain.entities.subscriptions import SubscriptionEntity

from domain.enums.tiers import Tier
from domain.enums.subscriptions import SubscriptionPlan
from domain.enums.tiers_pricing import Currency
from domain.enums.subscriptions import SubscriptionStatus

class Tiers(Base):
    __tablename__ = "tiers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    name: Mapped[Tier] = mapped_column(ENUM(Tier, name="tier_name_enum", create_type=True, check_first=True), nullable=False, unique=True, index=True)
    plans:Mapped[List["SubscriptionPlans"]] = relationship("SubscriptionPlans", back_populates="tier")

    @classmethod
    def from_entity(cls, tier:'TierEntity') -> "Tiers":
        return cls(
            name=tier.name
        )
    
    def to_entity(self) -> 'TierEntity':
        return TierEntity(
            id=self.id,
            name=self.name
        )

class SubscriptionPlans(Base):
    __tablename__="subscription_plans"

    id:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False, unique=True)
    code:Mapped[str] = mapped_column(String(64), unique=False, nullable=False, index=True)
    price:Mapped[Numeric] = mapped_column(Numeric(precision=10, scale=2), nullable=False)
    billing_cycle:Mapped[SubscriptionPlan] = mapped_column(ENUM(SubscriptionPlan, name='billing_cycle_enum', create_type=True, check_first=True), nullable=False)
    currency:Mapped[Currency] = mapped_column(ENUM(Currency, create_type=True, name="currency_enum", check_first=True), nullable=False)
    duration_days:Mapped[int] = mapped_column(Integer, nullable=True)
    is_active:Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    tier_id:Mapped[int] = mapped_column(ForeignKey('tiers.id'), nullable=True)

    tier:Mapped['Tiers'] = relationship("Tiers", back_populates='plans')
    subscriptions:Mapped[List["Subscriptions"]] = relationship("Subscriptions", back_populates='plan')

    def to_entity(self) -> 'SubscriptionPlansEntity':
        return SubscriptionPlansEntity(
            id=self.id,
            code=self.code,
            price=self.price,
            billing_cycle=self.billing_cycle,
            currency=self.currency,
            duration_days=self.duration_days,
            is_active=self.is_active,
            created_at=self.created_at,
            tier_id=self.tier_id
        )

    @classmethod
    def from_entity(cls, plan:'SubscriptionPlansEntity') -> "SubscriptionPlans":
        return cls(
            code=plan.code,
            price=plan.price,
            billing_cycle=plan.billing_cycle,
            currency=plan.currency,
            duration_days=plan.duration_days,
            is_active=plan.is_active,
            tier_id=plan.tier_id
        )

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
    plan:Mapped['SubscriptionPlans'] = relationship("SubscriptionPlans", back_populates='subscriptions')

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
from __future__ import annotations

from sqlalchemy import Integer, Numeric, ForeignKey, Boolean, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ENUM
from datetime import datetime, timezone
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from domain.entities.subscriptions import SubscriptionPlansEntity
from .tier_model import Tiers
from .subscription_model import Subscriptions


from ..db import Base


from domain.enums.subscriptions import SubscriptionPlan
from domain.enums.tiers_pricing import Currency

class SubscriptionPlans(Base):
    __tablename__="subscription_plans"

    id:Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False, unique=True)
    code:Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    price:Mapped[Numeric] = mapped_column(Numeric(precision=10, scale=2), nullable=False)
    billing_cycle:Mapped[SubscriptionPlan] = mapped_column(ENUM(SubscriptionPlan, name='billing_cycle_enum', create_type=True, check_first=True), nullable=False)
    currency:Mapped[Currency] = mapped_column(ENUM(Currency, create_type=True, name="currency_enum", check_first=True), nullable=False)
    duration_days:Mapped[int] = mapped_column(Integer, nullable=False)
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
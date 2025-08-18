from typing import Optional
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
from decimal import Decimal

if TYPE_CHECKING:
    from domain.entities.tiers import TierEntity
    from domain.entities.subscriptions import SubscriptionEntity


from domain.enums.tiers import Tier
from domain.enums.subscriptions import SubscriptionPlan
from domain.enums.tiers_pricing import Currency

@dataclass
class SubscriptionPlansEntity:
    code:str
    price:Decimal
    billing_cycle: SubscriptionPlan
    currency: Currency
    is_active:bool
    duration_days: Optional[int]
    created_at:Optional[datetime] = None #derived from database
    id:Optional[int] = None #derived from database
    tier_id:Optional[int] = None #derived from database

    #related entities
    tier:Optional["TierEntity"] = None #derived from database conditionally
    subscriptions:Optional[List["SubscriptionEntity"]] = None #derived from database conditionally

    @classmethod
    def create(cls, tier:"Tier", cycle:"SubscriptionPlan", currency:"Currency", price:Decimal, duration_days:Optional[int], tier_id:Optional[int]=None, is_active:bool=True, related_tier:Optional["TierEntity"]=None, related_subcscription:Optional["SubscriptionEntity"]=None) -> "SubscriptionPlansEntity":
        # code = f"{tier.name.upper()}-{cycle.upper()}-{currency.upper()}"
        code = cls.parse_tier_code(tier=tier, cycle=cycle, currency=currency)
        return cls(
            tier_id=tier_id,
            code=code,
            price=price,
            billing_cycle=cycle,
            currency=currency,
            duration_days=duration_days,
            is_active=is_active,
            tier=related_tier,
            subscriptions=related_subcscription
        )
    @staticmethod
    def parse_tier_code(tier:"Tier", cycle:"SubscriptionPlan", currency:"Currency") -> str:
        return f"{tier.name.upper()}-{cycle.name.upper()}-{currency.name.upper()}"
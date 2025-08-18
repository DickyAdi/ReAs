from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass

if TYPE_CHECKING:
    from domain.entities.subscriptions import SubscriptionPlansEntity

from domain.enums.tiers import Tier

@dataclass
class TierEntity:
    name: Tier
    id:Optional[int] = None #derived from database

    plans:Optional[List['SubscriptionPlansEntity']] = None

    @classmethod
    def create(cls, name:Tier, plans:Optional[List['SubscriptionPlansEntity']]=None) -> "TierEntity":
        return cls(
            name=name,
            plans=plans
        )
from abc import ABC, abstractmethod
from typing import Optional

from domain.entities.tiers import TierEntity
from domain.enums.tiers import Tier

class TiersInterface(ABC):
    @abstractmethod
    async def get_tier_by_name(self, db, tier_name:str):
        pass
    @abstractmethod
    async def get_tier_by_id(self, db, id:int):
        pass
    @abstractmethod
    async def get_tiers(self, db, offset:int, limit:int):
        pass
    @abstractmethod
    async def create_tier(self, db, tier:TierEntity):
        pass
    @abstractmethod
    async def delete_tier(self, db, id:int):
        pass
    @abstractmethod
    async def edit_tier(self, db, tier):
        pass
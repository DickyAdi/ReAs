from abc import ABC, abstractmethod
from domain.entities.subscriptions import SubscriptionPlansEntity
from typing import Optional

class SubscriptionPlansInterface(ABC):
    @abstractmethod
    async def create_plans(self, db, plan:SubscriptionPlansEntity, commit:bool=True):
        pass

    @abstractmethod
    async def edit_plans(self, db, plan):
        pass

    @abstractmethod
    async def get_plans(self, db, offset:int, limit:int, with_tier:bool, isActive:bool):
        pass

    @abstractmethod
    async def get_plan_by_code(self, db, code:str, is_latestActive:bool):
        pass
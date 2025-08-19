from abc import ABC, abstractmethod
from domain.entities.subscriptions import SubscriptionEntity


class SubscriptionInterface(ABC):
    @abstractmethod
    async def create_subscription(
        self, db, subscription: SubscriptionEntity, commit: bool = True
    ):
        pass

    @abstractmethod
    async def get_subscriptions(self, db, offset: int, limit: int):
        pass

    @abstractmethod
    async def get_subscriptions_by_user_id(
        self, db, user_id: int, is_latestActive: bool = True
    ):
        pass

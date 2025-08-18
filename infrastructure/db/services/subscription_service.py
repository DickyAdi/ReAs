from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from ..models.subscriptions_model import Subscriptions

from domain.entities.subscriptions.interfaces import SubscriptionInterface
from domain.entities.subscriptions import SubscriptionEntity


class SubscriptionService(SubscriptionInterface):
    """Infrastructure layer subscription service.

    Args:
        SubscriptionInterface (Class): Subscription service contract
    """

    async def create_subscription(
        self,
        db: AsyncSession,
        subscription: SubscriptionEntity,
        commit: Optional[bool] = True,
    ):
        """Create new subscription based on domain entity.

        Args:
            db (AsyncSession): Database session
            subscription (SubscriptionEntity): Subscription domain entity
            commit (Optional[bool], optional): Whether to commit DML to database or not. Defaults to True.

        Returns:
            Subscriptions: Subscriptions ORM instance.
        """
        new_sub = Subscriptions.from_entity(subscription=subscription)
        db.add(new_sub)
        await db.flush()
        if commit:
            await db.commit()
            await db.refresh(new_sub)
        return new_sub

    async def get_subscriptions(self, db, offset, limit):
        pass

    async def get_subscriptions_by_user_id(
        self, db: AsyncSession, user_id: int, is_latestActive: bool = True
    ):
        """Get subscriptions based on the given user id.

        Args:
            db (AsyncSession): Database session.
            user_id (int): User id
            is_latestActive (bool, optional): Whether to only retrieve the latest and active subscription or not. Defaults to True.

        Returns:
            - Subscriptions: If is_latestActive is True.
            - List[Subscriptions]: If is_latestActive is False.
        """
        stmt = select(Subscriptions).where(Subscriptions.user_id == user_id)
        if is_latestActive:
            stmt = stmt.order_by(Subscriptions.created_at.desc()).limit(1)
        res = await db.execute(stmt)
        subs = res.scalars()
        if is_latestActive:
            return subs.first()
        return subs.all()

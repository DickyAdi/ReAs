from datetime import timedelta, datetime, timezone
from uuid import UUID
from typing import Union

from domain.entities.subscriptions.interfaces import SubscriptionInterface
from domain.entities.subscriptions import SubscriptionEntity
from domain.enums.subscriptions import SubscriptionStatus


class SubscriptionsApplication:
    """Subscription application for subscription services."""

    def __init__(self, subscription_service: SubscriptionInterface):
        self.subscription_service = subscription_service

    async def get_user_subscriptions(
        self, db, user_id: int, is_latestActive: bool = True
    ):
        """Get users subscriptions based on users id

        Args:
            db (Any): Infrastructure layer DB async session
            user_id (int): Users id
            is_latestActive (bool, optional): Whether to returns the only active subscriptions or not, if False, returns all of the user subscriptions. Defaults to True.

        Returns:
            Subscription: Infrastructure layer subscriptions ORM instances.
        """
        subs = await self.subscription_service.get_subscriptions_by_user_id(
            db=db, user_id=user_id, is_latestActive=is_latestActive
        )
        return subs

    async def issue_subscription(
        self,
        db,
        user_id: int,
        end_subs: Union[int, None],
        plan_id: int,
        transaction_id: UUID,
        subs_status: SubscriptionStatus,
    ):
        """Issue new subscription for user

        Args:
            db (Any): Infrastructure layer DB async session.
            user_id (int): Users id that refers to the new subscription.
            end_subs (Union[int, None]): End date of the subscription.
            plan_id (int): SubscriptionPlan id of the subscription.
            transaction_id (UUID): Transaction id of the subscription.
            subs_status (SubscriptionStatus): Status of the subscription

        Returns:
            Subscription: Infrastructure layer subscriptions ORM instances.
        """
        start_time = datetime.now(timezone.utc)
        end_time = (start_time + timedelta(days=end_subs)) if end_subs else None
        subs_obj = SubscriptionEntity.create(
            start_date=start_time,
            end_date=end_time,
            status=subs_status,
            plan_id=plan_id,
            transaction_id=transaction_id,
            user_id=user_id,
        )
        new_subs = await self.subscription_service.create_subscription(
            db=db, subscription=subs_obj, commit=False
        )
        return new_subs

from decimal import Decimal
from typing import Optional

from domain.entities.subscriptions import SubscriptionPlansEntity
from domain.entities.subscriptions.interfaces import SubscriptionPlansInterface

from domain.enums.tiers_pricing import Currency
from domain.enums.subscriptions import SubscriptionPlan
from domain.enums.tiers import Tier


class SubscriptionPlanApplication:
    """Subscription application for subscription services."""

    def __init__(self, service: SubscriptionPlansInterface):
        self.service = service

    async def get_plan(
        self,
        db,
        tier: Tier,
        cycle: SubscriptionPlan,
        currency: Currency,
        isLatestActive: bool = True,
    ):
        """Get single plan based on the given parameter, this function automatically parse the plan code based on the given parameter.

        Args:
            db (Any): Infrastructure layer DB async session type
            tier (Tier): Tier enums of the plan.
            cycle (SubscriptionPlan): Billing cycle enums of the plan.
            currency (Currency): Currency enums of the plan
            isLatestActive (bool, optional): Whether to get the latest and active plan or not. Defaults to True.

        Returns:
            Plan: Infrastructure layer Plan object.
        """
        code = SubscriptionPlansEntity.parse_tier_code(
            tier=tier, cycle=cycle, currency=currency
        )
        plan = await self.service.get_plan_by_code(
            db=db, code=code, is_latestActive=isLatestActive
        )
        return plan

    async def get_plans(
        self,
        db,
        offset: int,
        limit: int,
        with_tier: bool = False,
        isActive: bool = False,
    ):
        """Get all plans with pagination.

        Args:
            db (Any): Infrastructure layer DB async session type.
            offset (int): Starts row.
            limit (int): Limit number of selected row
            with_tier (bool, optional): Whether to join subscription plan table with tier table or not. Defaults to False.
            isActive (bool, optional): Returns the active plan or not. Defaults to False.

        Raises:
            TypeError: If offset and limit is not integer.

        Returns:
            Plan: Infrastructure layer Plan object.
        """
        if not isinstance(offset, int) or not isinstance(limit, int):
            raise TypeError(
                f"Offset and or limit must be type Integer. Got {type(offset)} and {type(limit)}"
            )
        plans = await self.service.get_plans(
            db=db, offset=offset, limit=limit, with_tier=with_tier, isActive=isActive
        )
        return plans

    async def deactivate_plan(self, db, plan) -> bool:
        """This function is part of append/create new plan flow, which receives ORM Plan object to deactivate the plan.

        Args:
            db (Any): Infrastructure layer DB async session type.
            plan (Any): Infrastructure layer ORM Plan class instance.

        Raises:
            Exception: If deactivating plan encounter an Exception.

        Returns:
            bool: Returns True if class instance is deactivated and successfuly flushed to the database.
        """
        try:
            plan.is_active = False
            await db.flush()
            return True
        except Exception as e:
            raise Exception(f"Something went wrong: {str(e)}")

    async def create_new_plan(
        self,
        db,
        tier_id: int,
        tier: Tier,
        cycle: SubscriptionPlan,
        currency: Currency,
        price: Decimal,
        duration_days: Optional[int],
    ):
        """This method is part of append/create new plan flow.

        Args:
            db (Any): Infrastructure layer DB async session type.
            tier_id (int): Foreign key to the tier id of the created plan.
            tier (Tier): Tier enums for the plan.
            cycle (SubscriptionPlan): Billing cycle enums for the plan.
            currency (Currency): Currency enums for the plan.
            price (Decimal): Price of the plan
            duration_days (Optional[int]): Duration of the plans. Pass None if the duration of the plan is unlimited.

        Returns:
            Plan: Infrastructure layer Plan object.
        """
        plan_entity = SubscriptionPlansEntity.create(
            tier_id=tier_id,
            tier=tier,
            cycle=cycle,
            currency=currency,
            price=price,
            duration_days=duration_days,
        )
        new_plan = await self.service.create_plans(db=db, plan=plan_entity)
        return new_plan

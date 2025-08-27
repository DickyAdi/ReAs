from decimal import Decimal
from typing import Optional

from application.tiers import TierApplication
from domain.entities.subscriptions.interfaces import (
    SubscriptionInterface,
    SubscriptionPlansInterface,
)
from domain.entities.payments.interfaces import PaymentInterface
from domain.entities.transactions.interfaces import TransactionInterface
from domain.entities.tiers.interfaces import TiersInterface

from domain.enums.tiers_pricing import Currency
from domain.enums.subscriptions import SubscriptionPlan
from domain.enums.tiers import Tier

from domain.exceptions import PlanReferenceError, PlanDeactivateError

from .base_class import BaseSubscriptionFlow


class CreateNewPlanFlow(BaseSubscriptionFlow):
    """Use case flow for creating new plan/subscription plan.

    Args:
        BaseSubscriptionFlow (Class): Base subscription flow class
    """

    def __init__(
        self,
        tier_service: TiersInterface,
        subscription_service: SubscriptionInterface,
        plan_service: SubscriptionPlansInterface,
        transaction_service: TransactionInterface,
        payment_service: PaymentInterface,
    ):
        super().__init__(
            subscription_service=subscription_service,
            plan_service=plan_service,
            transaction_service=transaction_service,
            payment_service=payment_service,
        )
        self.tier_app = TierApplication(service=tier_service)

    async def __call__(
        self,
        db,
        tier: Tier,
        cycle: SubscriptionPlan,
        currency: Currency,
        price: Decimal,
        duration_days: Optional[int],
    ):
        """Append a new plan based on previous related plan.

        Args:
            db (Any): Infrastructure layer database session.
            tier (Tier): Tier for the new subscription plan. This must be the same with the previous/existing plan.
            cycle (SubscriptionPlan): Billing cycle for the new subscription plan. This must be the same with the previous/existing plan.
            currency (Currency): Currency for the new subscription plan. This must be the same with the previous/existing plan.
            price (Decimal): Price for the new subscription plan.
            duration_days (Optional[int]): Subscription duration of the new subscription plan.

        Raises:
            PlanRefError: If previous plan does not exist or attempting to create a new plan. Note: Creating new plan is not allowed by the business rules.
            PlanDeactivateError: If deactivating existing plan was failed.

        Returns:
            SubscriptionPlan: Infrastructure layer SubscriptionPlans ORM instance.
        """
        existing_plan = await self.plan_app.get_plan(
            db=db, tier=tier, cycle=cycle, currency=currency
        )
        if not existing_plan:
            raise PlanReferenceError(
                plan_code=self.plan_app.parse_code(
                    tier=tier, cycle=cycle, currency=currency
                )
            )
        deactivated = await self.plan_app.deactivate_plan(db=db, plan=existing_plan)
        if not deactivated:
            raise PlanDeactivateError(plan_code=existing_plan.code)
        selected_tier = await self.tier_app.get_tier(db=db, tier_name=tier)
        created_plan = await self.plan_app.create_new_plan(
            db=db,
            tier_id=selected_tier.id,
            tier=tier,
            cycle=cycle,
            currency=currency,
            price=price,
            duration_days=duration_days,
        )
        return created_plan
